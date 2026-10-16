"""Dense recall, deterministic same-entity rerank, conservative temporal dedup."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import gc
import json
import math
from pathlib import Path
import time

from .dense_metadata import DENSE_FORMAT, FIELDS
from .dense_search import query_texts, score_dense
from .entity_constraints import VERSION, best_entity_match
from .intent_parser import parse_query
from .local_index import read_meta, rows, validate_index
from .local_runtime import ROOT, atomic_json, sha256
from .temporal_groups import group_temporal


@dataclass(frozen=True)
class EntityRerankConfig:
    """Generic defaults fixed before real-video evaluation, never fitted."""
    semantic_floor: float = .20
    entity_weight: float = .80
    contradiction_cost: float = .04
    contradiction_cap: int = 3

    def validate(self):
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (self.semantic_floor,self.entity_weight,self.contradiction_cost)) or not math.isclose(self.semantic_floor+self.entity_weight,1.) or self.contradiction_cap < 0:
            raise ValueError("Invalid entity rerank configuration")


DEFAULT_CONFIG = EntityRerankConfig()


def combine_scores(dense_semantic_score,explanation,config=DEFAULT_CONFIG):
    config.validate()
    semantic = min(1.,max(0.,(dense_semantic_score+1.)/2.))
    contradictions = len(explanation["contradictions"])
    # Existing dense lexical boosts do not enter this formula: they aggregate
    # facts across entities. Only BGE field cosine and ONE entity determine it.
    penalty = config.contradiction_cost*min(contradictions,config.contradiction_cap)
    return semantic*(config.semantic_floor+config.entity_weight*explanation["entity_score"])-penalty


def rerank_entity_candidates(dense_table,records,config=DEFAULT_CONFIG,temporal_iou_threshold=.5,temporal_dedup=True,normalized_entities=None):
    config.validate()
    started = time.perf_counter()
    parsed = parse_query(dense_table["query"])
    candidates = []
    for source in dense_table["candidates"]:
        record = records[source["row"]]
        if record["clip_id"] != source["clip_id"]:raise ValueError("Entity metadata/candidate alignment mismatch")
        explanation = best_entity_match(record,parsed, normalized_entities[source["row"]] if normalized_entities is not None else None)
        candidates.append({**source,"dense_rank":source["rank"],"dense_score":source["final_score"],
                           "ranker":"dense-entity","entity_explanation":explanation,"entity_score":explanation["entity_score"],
                           "final_score":combine_scores(source["dense_semantic_score"],explanation,config)})
    candidates.sort(key=lambda row:(-row["final_score"],row["row"]))
    for rank,row in enumerate(candidates,1):row["rank"] = rank
    rerank_seconds = time.perf_counter()-started
    started = time.perf_counter()
    grouped = group_temporal(candidates,temporal_iou_threshold,temporal_dedup)
    return {"format":"local-dense-entity-candidates-v1","query":dense_table["query"],"index":dense_table["index"],
            "parsed_query":parsed,"constraint_version":VERSION,"config":asdict(config),
            "field_weights":dense_table["field_weights"],"retrieval":dense_table["retrieval"],"qwen_required":False,
            "dense_candidates":dense_table["candidates"],"raw_pre_dedup_ranking":candidates,
            "candidates":grouped,"temporal_groups":[r["temporal_group"] for r in grouped],
            "temporal_dedup":{"enabled":temporal_dedup,"iou_threshold":temporal_iou_threshold,
                              "comparison":"strictly greater; all members; same video; compatible stored event facts",
                              "raw_count":len(candidates),"grouped_count":len(grouped)},
            "timing":{"entity_rerank_seconds":rerank_seconds,"temporal_grouping_seconds":time.perf_counter()-started}}


def encode_dense_query(meta,query,siglip_full_model=False):
    """One BGE field batch and one projected SigLIP query, reused by all actors."""
    from .text_encoder import BGEEncoder
    started = time.perf_counter()
    bge = BGEEncoder(meta["models"]["bge"]["path"],"cpu",local_files_only=True)
    loaded = time.perf_counter()
    texts = query_texts(query,parse_query(query))
    qfields = bge.encode([texts[f] for f in FIELDS],batch_size=8)
    encoded = time.perf_counter()
    del bge
    gc.collect()
    from .query_encoder import SigLIPQueryEncoder
    sig_started = time.perf_counter()
    sig = SigLIPQueryEncoder(meta["checkpoints"]["stage2b"]["path"],"cpu",local_files_only=True,
                            model_path=meta["models"]["siglip"]["path"],text_only=not siglip_full_model)
    sig_loaded = time.perf_counter()
    _,qproj = sig.encode([query],batch_size=1)
    sig_encoded = time.perf_counter()
    del sig
    gc.collect()
    return qfields,qproj[0],{"bge_load_seconds":loaded-started,"bge_encode_seconds":encoded-loaded,
                            "siglip_load_seconds":sig_loaded-sig_started,"siglip_encode_seconds":sig_encoded-sig_loaded,
                            "encoding_and_release_seconds":time.perf_counter()-started}


def outside_indexes(target,index):
    target = Path(target).resolve()
    for protected in (Path(index).resolve(),(ROOT/"outputs/local_hybrid_indexes").resolve()):
        if target == protected or protected in target.parents:
            raise ValueError("Entity search outputs must be outside immutable index directories")
    return target


def search_dense_entity(index,query,caption_k=100,stage3_k=100,candidate_k=50,top_k=10,
                        extract_top_k=0,result_dir=None,siglip_full_model=False,debug_scores=False,
                        debug_entity=False,candidate_table=None,not_found_stats=False,
                        temporal_iou_threshold=.5,no_temporal_dedup=False, *, privacy=True,
                        privacy_mode="gaussian", privacy_strength="strong", privacy_block_size=16):
    if not query.strip() or min(top_k,caption_k,stage3_k,candidate_k) <= 0 or extract_top_k < 0:
        raise ValueError("Invalid dense-entity query/retrieval settings")
    if not math.isfinite(temporal_iou_threshold) or not 0 <= temporal_iou_threshold <= 1:
        raise ValueError("temporal-iou-threshold must be finite and between 0 and 1")
    if read_meta(index).get("format") != DENSE_FORMAT:
        raise ValueError("dense structured metadata is required for ranker dense-entity")
    started = time.perf_counter()
    default_dir = ROOT/"outputs/entity_search_results"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target = outside_indexes(candidate_table or default_dir/"candidates.json",index)
    if target.exists():raise FileExistsError(f"Candidate table already exists: {target}")
    clips = outside_indexes(result_dir or default_dir/"clips",index)
    validate_index(index)
    meta = read_meta(index)
    validation_seconds = time.perf_counter()-started
    qfields,qproj,encoding_timing = encode_dense_query(meta,query,siglip_full_model)
    retrieval_started = time.perf_counter()
    dense = score_dense(index,query,qfields,qproj,caption_k,stage3_k,candidate_k)
    retrieval_seconds = time.perf_counter()-retrieval_started
    metadata_started = time.perf_counter()
    records = rows(Path(index)/"metadata/dense_metadata.jsonl")
    metadata_seconds = time.perf_counter()-metadata_started
    table = rerank_entity_candidates(dense,records,temporal_iou_threshold=temporal_iou_threshold,
                                     temporal_dedup=not no_temporal_dedup)
    table["timing"].update(validation_seconds=validation_seconds,**encoding_timing,
                          dense_candidate_retrieval_seconds=retrieval_seconds,metadata_read_seconds=metadata_seconds,
                          total_search_seconds=time.perf_counter()-started)
    table["timing"]["pure_ranking_seconds"] = retrieval_seconds+table["timing"]["entity_rerank_seconds"]+table["timing"]["temporal_grouping_seconds"]
    table["artifact_path"] = str(target)
    atomic_json(target,table)  # always preserve complete raw pre-dedup ranking
    results = table["candidates"][:top_k]
    print("Ranker: dense-entity | Qwen required: NO | Parsed:",json.dumps(table["parsed_query"]))
    print("Candidate artifact:",target)
    if not_found_stats:
        print("Not-found diagnostics:",json.dumps({"top1_score":results[0]["final_score"] if results else None,
              "top1_minus_top2_margin":results[0]["final_score"]-results[1]["final_score"] if len(results)>1 else None,
              "threshold":None,"not_found_decision":"uncalibrated"}))
    for row in results:
        group = row["temporal_group"]
        print(f"#{row['rank']:02d} Event group {group['group_start_sec']:.3f}-{group['group_end_sec']:.3f}s | representative {row['start_sec']:.3f}-{row['end_sec']:.3f}s | final={row['final_score']:.5f}")
        print("Supporting windows:",json.dumps(group["member_windows"]))
        if debug_entity:
            e = row["entity_explanation"]
            print("Best entity:",json.dumps(e["best_entity"]))
            for label,key in (("MATCH","matched"),("CLOSE","close"),("UNKNOWN","unknown"),("MISMATCH","missing"),("CONTRADICTIONS","contradictions")):
                print(label+":",json.dumps(e[key]))
            print("Scores:",json.dumps({k:row[k] for k in ("entity_score","dense_semantic_score","dense_score","final_score")}))
    if debug_scores:
        for row in table["raw_pre_dedup_ranking"]:print("Candidate debug:",json.dumps(row))
    if extract_top_k:
        if sha256(meta["video"]["path"]) != meta["video"]["sha256"]:raise ValueError("Source video changed; refusing extraction")
        from .video_export import extract_clip
        for row in results[:extract_top_k]:
            path = clips/f"rank_{row['rank']:02d}_{row['start_sec']:.3f}_{row['end_sec']:.3f}.mp4"
            extract_clip(meta["video"]["path"],path,row["start_sec"],row["end_sec"],meta["video"]["fps"],
                         privacy=privacy, privacy_cache=Path(index)/"privacy", privacy_mode=privacy_mode,
                         privacy_strength=privacy_strength, privacy_block_size=privacy_block_size)
            print("Clip:",path)
    return results
