"""Qwen-free-by-default, read-only local Hybrid V2 retrieval and Stage5 ranking."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from .build_features import cap_fields, rr
from .common import jaccard, l2norm, zscore1d
from .intent_parser import parse_intent
from .local_index import array, caption_prompt_info, load_stage5, read_meta, rows, validate_index
from .local_runtime import atomic_json
from .stage5_model import Stage5HybridReranker


def cosine_scores(matrix, query, batch=512):
    result = np.empty(len(matrix), np.float32)
    for s in range(0, len(matrix), batch):
        result[s:s+batch] = l2norm(matrix[s:s+batch]) @ query
    return result


@torch.inference_mode()
def score_candidates(index, query, intent, qfields, qraw, qproj, caption_k=100, stage3_k=100, rerank_k=50, *, loaded=None):
    """Exact historical features/scoring, exposed once for all diagnostic rankers."""
    if min(caption_k, stage3_k, rerank_k) <= 0:
        raise ValueError("All retrieval limits must be positive")
    meta = loaded["meta"] if loaded is not None else read_meta(index)
    d = meta["dimensions"]["bge"]
    if np.asarray(qfields).shape != (4, d) or np.asarray(qraw).shape != (768,) or np.asarray(qproj).shape != (512,):
        raise ValueError("Query embedding geometry differs from index")
    for value in (qfields, qraw, qproj):
        if not np.isfinite(value).all() or not np.allclose(np.linalg.norm(value, axis=-1), 1, atol=1e-3):
            raise ValueError("Query vectors must be finite and normalized")
    get_array = lambda name: loaded["arrays"][name] if loaded is not None else array(index, name)
    full, action, relation, context = [get_array(f"text/caption_{x}.npy") for x in ("full", "action", "relation", "context")]
    fused = get_array("visual/fused_embeddings.npy")
    frames = get_array("visual/projected_frame_embeddings.npy")
    raw = get_array("visual/raw_frame_features.npy")
    captions = loaded["captions"] if loaded is not None else rows(Path(index) / "captions/captions.jsonl")
    manifest = loaded["manifest"] if loaded is not None else rows(Path(index) / "manifest.jsonl")
    capscore, s3score = cosine_scores(full, qfields[0]), cosine_scores(fused, qproj)
    ci, si = np.argsort(-capscore, kind="stable")[:caption_k], np.argsort(-s3score, kind="stable")[:stage3_k]
    rc, rs = {int(j): r+1 for r, j in enumerate(ci)}, {int(j): r+1 for r, j in enumerate(si)}
    def rrf(j):
        return (1/(60+rc[j]) if j in rc else 0) + (1/(60+rs[j]) if j in rs else 0)
    candidates = sorted(set(rc) | set(rs), key=lambda j: (-rrf(j), j))[:rerank_k]
    feats = []
    for j in candidates:
        c, s = float(qfields[0] @ full[j]), float(qproj @ l2norm(fused[j]))
        ac = float(qfields[1] @ action[j]) if intent["action"] else 0.
        re = float(qfields[2] @ relation[j]) if intent["relation"] else 0.
        co = float(qfields[3] @ context[j]) if intent["context"] else 0.
        ps, gs = l2norm(frames[j]) @ qproj, l2norm(raw[j]) @ qraw
        desc, at, rt, _ = cap_fields(captions[j])
        feats.append([
            c, ac, re, co, s,
            float(ps.max()), float(np.sort(ps)[-min(3,len(ps)):].mean()), float(ps.mean()),
            float(gs.max()), float(np.sort(gs)[-min(3,len(gs)):].mean()), float(gs.mean()),
            rr(rc.get(j, -1)), rr(rs.get(j, -1)), jaccard(query, desc),
            jaccard(intent["action"], at) if intent["action"] else 0.,
            jaccard(intent["relation"], rt) if intent["relation"] else 0., c*s, ac*re,
        ])
    X = np.asarray(feats, np.float32)
    base = np.zeros(len(X), np.float32)
    for weight, pos in ((.30,0), (.20,4), (.15,6), (.15,1), (.10,2), (.10,3)):
        base += weight*zscore1d(X[:,pos])
    ck = loaded["stage5_checkpoint"] if loaded is not None else load_stage5(meta["checkpoints"]["stage5"]["path"])
    XX = (X-np.asarray(ck["feature_mean"], np.float32))/np.asarray(ck["feature_std"], np.float32)
    cfg = ck["config"]
    if loaded is not None:
        model = loaded["stage5_model"]
    else:
        model = Stage5HybridReranker(18, int(cfg["hidden"]), float(cfg["dropout"]), float(cfg["residual_scale"]))
        model.load_state_dict(ck["model_state_dict"], strict=True)
        model.eval()
    residual = model.residual(torch.from_numpy(XX)[None])[0].numpy()
    final = base + float(ck["alpha"])*residual
    if not np.isfinite(final).all():
        raise ValueError("Non-finite Stage5 ranking")
    scores = {"caption": X[:,0], "stage3": X[:,4], "base": base, "stage5": final}
    ranks = {}
    for name, values in scores.items():
        ranks[name] = np.empty(len(values), np.int64)
        ranks[name][np.argsort(-values, kind="stable")] = np.arange(1, len(values)+1)
    result = []
    for i, j in enumerate(candidates):
        w = manifest[j]
        result.append({"row": j, "clip_id": w["clip_id"], "start_sec": w["start_sec"],
                       "end_sec": w["end_sec"], "base_score": float(base[i]),
                       "stage5_residual": float(residual[i]), "stage5_final": float(final[i]),
                       "caption_score": float(X[i,0]), "stage3_score": float(X[i,4]),
                       "action_score": float(X[i,1]), "relation_score": float(X[i,2]),
                       "description": captions[j]["description"], "caption_fallback": bool(captions[j].get("fallback")),
                       "features": dict(zip(ck["feature_names"], map(float, X[i]))),
                       "standardized_features": dict(zip(ck["feature_names"], map(float, XX[i]))),
                       "ranks": {name: int(value[i]) for name, value in ranks.items()}})
    cap_sorted = np.sort(capscore)[::-1]
    return {"format": "local-hybrid-candidates-v1", "query": query, "intent": intent,
            "index": str(Path(index).absolute()), **caption_prompt_info(meta),
            "feature_names": ck["feature_names"], "feature_mean": np.asarray(ck["feature_mean"]).tolist(),
            "feature_std": np.asarray(ck["feature_std"]).tolist(), "checkpoint_alpha": float(ck["alpha"]),
            "stage5_checkpoint": meta["checkpoints"]["stage5"],
            "retrieval": {"index_windows": len(manifest), "candidate_count": len(candidates),
                          "caption_k": caption_k, "stage3_k": stage3_k, "rerank_k": rerank_k},
            "caption_top1": float(cap_sorted[0]),
            "caption_margin": float(cap_sorted[0]-cap_sorted[1]) if len(cap_sorted)>1 else None,
            "candidates": result}


def rank_candidates(table, ranker="stage5", top_k=10, debug_scores=False):
    if ranker not in {"caption", "stage3", "base", "stage5"} or top_k <= 0:
        raise ValueError("Valid ranker and positive top-k required")
    score_key = {"caption": "caption_score", "stage3": "stage3_score", "base": "base_score", "stage5": "stage5_final"}[ranker]
    results = []
    for rank, candidate in enumerate(sorted(table["candidates"], key=lambda r: r["ranks"][ranker])[:top_k], 1):
        result = {k: v for k, v in candidate.items() if k not in {"features", "standardized_features", "ranks"}}
        result.update(rank=rank, ranker=ranker, final_score=candidate[score_key])
        if debug_scores:
            result["features"] = candidate["features"]
            result["standardized_features"] = candidate["standardized_features"]
            result["ranks"] = candidate["ranks"]
        results.append(result)
    return results


def not_found_diagnostics(table, ranker="stage5"):
    ranked = rank_candidates(table, ranker, top_k=2)
    return {"ranker": ranker, "top1_score": ranked[0]["final_score"],
            "top1_minus_top2_margin": ranked[0]["final_score"]-ranked[1]["final_score"] if len(ranked)>1 else None,
            "caption_top1": table["caption_top1"], "caption_margin": table["caption_margin"],
            "scope": "selected-ranker margin over rerank candidates; caption margin over entire index",
            "threshold": None, "not_found_decision": "uncalibrated"}


def rank_index(index, query, intent, qfields, qraw, qproj, caption_k=100, stage3_k=100, rerank_k=50,
               top_k=10, ranker="stage5", debug_scores=False):
    table = score_candidates(index, query, intent, qfields, qraw, qproj, caption_k, stage3_k, rerank_k)
    return rank_candidates(table, ranker, top_k, debug_scores)


def legacy_search(index, query, intent_mode="rule", caption_k=100, stage3_k=100, rerank_k=50, top_k=10,
           extract_top_k=0, result_dir=None, qwen_base_url="http://127.0.0.1:8080/v1", vlm_model=None, siglip_full_model=False,
           ranker="stage5", debug_scores=False, candidate_table=None, not_found_stats=False,
           candidate_k=50, debug_entity=False, temporal_iou_threshold=.5, no_temporal_dedup=False,
           privacy=True, privacy_mode="gaussian", privacy_strength="strong", privacy_block_size=16,
           decision_provider=None, laya_device=None, laya_model_path=None, laya_batch_size=None, laya_candidate_count=None):
    if not query.strip() or extract_top_k < 0:
        raise ValueError("Nonempty query and nonnegative extract-top-k required")
    if ranker == "dense-entity":
        from .entity_search import search_dense_entity
        return search_dense_entity(index,query,caption_k,stage3_k,candidate_k,top_k,extract_top_k,result_dir,
                                   siglip_full_model,debug_scores,debug_entity,candidate_table,not_found_stats,
                                   temporal_iou_threshold,no_temporal_dedup, privacy=privacy, privacy_mode=privacy_mode,
                                   privacy_strength=privacy_strength, privacy_block_size=privacy_block_size)
    validation = validate_index(index)
    meta = read_meta(index)
    if ranker == "dense":
        from .dense_metadata import DENSE_FORMAT
        if meta["format"] == DENSE_FORMAT:
            from .dense_search import search_dense
            return search_dense(index,query,caption_k,stage3_k,rerank_k,top_k,extract_top_k,result_dir,
                                siglip_full_model,debug_scores,candidate_table,not_found_stats,
                                privacy=privacy, privacy_mode=privacy_mode, privacy_strength=privacy_strength,
                                privacy_block_size=privacy_block_size)
        print("Dense metadata absent; falling back to caption retrieval (Qwen-free).")
        ranker, intent_mode = "caption", "rule"
    if intent_mode == "rule":
        intent = parse_intent(query)
    elif intent_mode == "none":
        intent = {"action": "", "relation": "", "context": ""}
    elif intent_mode == "qwen":
        intent = parse_intent(query)
    else:
        raise ValueError("Unknown intent mode")
    # Foundation encoders are sequential CPU lifetimes; no GPU needed for normal search.
    from .text_encoder import BGEEncoder
    bge = BGEEncoder(meta["models"]["bge"]["path"], "cpu", local_files_only=True)
    qfields = bge.encode([query, intent["action"] or "none", intent["relation"] or "none", intent["context"] or "none"], batch_size=4)
    del bge
    gc.collect()
    from .query_encoder import SigLIPQueryEncoder
    sig = SigLIPQueryEncoder(meta["checkpoints"]["stage2b"]["path"], "cpu", local_files_only=True,
                            model_path=meta["models"]["siglip"]["path"], text_only=not siglip_full_model)
    qraw, qproj = sig.encode([query], batch_size=1)
    del sig
    gc.collect()
    table = score_candidates(index, query, intent, qfields, qraw[0], qproj[0], caption_k, stage3_k, rerank_k)
    results = rank_candidates(table, ranker, top_k, debug_scores)
    print("Ranker:", ranker, "| Caption prompt:", validation["caption_prompt_version"])
    if candidate_table is not None:
        candidate_table = Path(candidate_table).resolve()
        protected = Path(index).resolve()
        if candidate_table.exists() or candidate_table == protected or protected in candidate_table.parents:
            raise ValueError("Candidate table must be a new file outside the immutable index")
        atomic_json(candidate_table, table)
        print("Candidate table:", candidate_table)
    if not_found_stats:
        print("Not-found diagnostics:", json.dumps(not_found_diagnostics(table, ranker)))
    if debug_scores:
        for candidate in table["candidates"]:
            print("Candidate debug:", json.dumps(candidate))
    print("Intent:", json.dumps(intent, ensure_ascii=False))
    for result in results:
        print(f"#{result['rank']:02d} {result['start_sec']:.3f}s -> {result['end_sec']:.3f}s "
              f"final_score={result['final_score']:.5f} base_score={result['base_score']:.5f} "
              f"caption_score={result['caption_score']:.5f} stage3_score={result['stage3_score']:.5f}")
        print("  generated caption:", result["description"])
    if extract_top_k:
        from .local_runtime import sha256
        if sha256(meta["video"]["path"]) != meta["video"]["sha256"]:
            raise ValueError("Source video changed; refusing clip extraction")
        from .video_export import extract_clip
        from datetime import datetime
        target = Path(result_dir) if result_dir else Path(index)/"results"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for result in results[:extract_top_k]:
            path = target/f"rank_{result['rank']:02d}_{result['start_sec']:.3f}_{result['end_sec']:.3f}.mp4"
            extract_clip(meta["video"]["path"], path, result["start_sec"], result["end_sec"], meta["video"]["fps"],
                         privacy=privacy, privacy_cache=Path(index)/"privacy", privacy_mode=privacy_mode,
                         privacy_strength=privacy_strength, privacy_block_size=privacy_block_size)
            print("Clip:", path)
    return results


def search(index, query, intent_mode="rule", caption_k=100, stage3_k=100, rerank_k=50, top_k=10,
           extract_top_k=0, result_dir=None, qwen_base_url=None, vlm_model=None, siglip_full_model=False,
           ranker="stage5", debug_scores=False, candidate_table=None, not_found_stats=False,
           candidate_k=50, debug_entity=False, temporal_iou_threshold=.5, no_temporal_dedup=False,
           privacy=True, privacy_mode="gaussian", privacy_strength="strong", privacy_block_size=16,
           decision_provider=None, laya_device=None, laya_model_path=None, laya_batch_size=None, laya_candidate_count=None,
           deployment_mode=None):
    from .search_engine import DEFAULT_CACHE
    if ranker == "dense-entity":
        from .dense_metadata import DENSE_FORMAT
        if read_meta(index)["format"] != DENSE_FORMAT:
            raise ValueError("dense structured metadata is required for ranker dense-entity")
    if top_k <= 0 or extract_top_k < 0:
        raise ValueError("INVALID_QUERY")
    if deployment_mode is not None and ranker != "driveevidence_v6":
        raise ValueError("V6_DEPLOYMENT_MODE_REQUIRES_V6")
    deployment_options = {"deployment_mode": deployment_mode} if ranker == "driveevidence_v6" and deployment_mode is not None else {}
    table = DEFAULT_CACHE.search_table(index, query, ranker=ranker, intent_mode=intent_mode,
        caption_k=caption_k, stage3_k=stage3_k, rerank_k=rerank_k, candidate_k=candidate_k,
        temporal_iou_threshold=temporal_iou_threshold, no_temporal_dedup=no_temporal_dedup,
        decision_provider=decision_provider, laya_device=laya_device, laya_model_path=laya_model_path,
        laya_batch_size=laya_batch_size, laya_candidate_count=laya_candidate_count, **deployment_options)
    results = table["candidates"][:top_k]
    if table.get("decision", {}).get("state") == "NOT_FOUND" and not (debug_scores or debug_entity):
        results = []
    if "decision" in table and not (debug_scores or debug_entity):
        results = [{k: v for k, v in row.items() if k not in ("features", "standardized_features", "ranks", "constraint_result", "laya_decision", "entity_explanation", "dense_cosines", "exact_detail_hits", "contradiction_reasons")} for row in results]
    if candidate_table is not None:
        from .entity_search import outside_indexes
        target = outside_indexes(candidate_table, index)
        if target.exists():
            raise FileExistsError("Candidate table must be a new file outside the immutable index")
        atomic_json(target, table)
    if debug_scores or debug_entity:
        print(json.dumps(table, ensure_ascii=True))
        if debug_entity:
            for row in results:
                if "entity_explanation" in row:
                    print("Best entity:", json.dumps(row["entity_explanation"]["best_entity"]))
    else:
        print("Ranker:", ranker, "| Qwen required: NO")
        if "decision" in table:
            print("Decision:", table["decision"]["state"], "| Provider:", table["decision"]["provider"])
        for row in results:
            print(f"#{row['rank']:02d} {row['start_sec']:.3f}s -> {row['end_sec']:.3f}s score={row['final_score']:.5f}")
    if not_found_stats:
        print("Not-found diagnostics:", json.dumps(table.get("decision", {"not_found_decision": "uncalibrated"})))
    if extract_top_k:
        from .local_runtime import ROOT, sha256
        from .video_export import extract_clip
        from .entity_search import outside_indexes
        from datetime import datetime
        meta = read_meta(index)
        if sha256(meta["video"]["path"]) != meta["video"]["sha256"]:
            raise ValueError("Source video changed; refusing extraction")
        target = outside_indexes(result_dir or ROOT / "outputs/search_exports" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"), index)
        for row in results[:extract_top_k]:
            extract_clip(meta["video"]["path"], target / f"rank_{row['rank']:02d}.mp4", row["start_sec"], row["end_sec"], meta["video"]["fps"],
                privacy=privacy, privacy_cache=Path(index) / "privacy", privacy_mode=privacy_mode, privacy_strength=privacy_strength, privacy_block_size=privacy_block_size)
    return results


if __name__ == "__main__":
    from .laptop_cli import main
    import sys
    raise SystemExit(main(["search", *sys.argv[1:]]))
