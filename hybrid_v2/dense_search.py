"""Deterministic field retrieval. Qwen and Stage5 are unnecessary for scoring."""
from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path

import numpy as np

from .dense_metadata import FIELDS, actor_name, meaningful
from .intent_parser import COLORS, parse_query
from .local_index import array, read_meta, rows, validate_index
from .local_runtime import atomic_json, sha256

PROFILE_VERSION = "broad_subject_profiles_v1"
PROFILES = {
    "person":{"full":.15,"people":.25,"action":.15,"relation":.15,"infrastructure":.15,"salient":.15},
    "vehicle":{"full":.20,"vehicle":.25,"action":.20,"relation":.15,"salient":.20},
    "infrastructure":{"full":.20,"infrastructure":.35,"scene":.15,"relation":.10,"salient":.20},
    "scene":{"scene":.60,"full":.40},
}


def field_weights(parsed):
    subject = parsed["subject_type"]
    profile = "vehicle" if subject in ("vehicle","motorcycle","cyclist") else subject
    profile = profile if profile in PROFILES else "scene"
    values = {field:PROFILES[profile].get(field,0.) for field in FIELDS}
    # Broad presence rules, fixed before evaluation. Inactive semantic slots
    # transfer their weights to full rather than embedding the filler "none".
    for field,active in (("action",bool(parsed["action"])),("relation",bool(parsed["relation"])),
                         ("infrastructure",bool(parsed["infrastructure"]))):
        if not active:
            values["full"] += values[field]
            values[field] = 0.
    if parsed["context"] and profile != "scene":
        values = {k:v*.90 for k,v in values.items()}
        values["scene"] += .10
    if parsed["infrastructure"] and values["infrastructure"] == 0:
        values = {k:v*.90 for k,v in values.items()}
        values["infrastructure"] = .10
    total = sum(values.values())
    return {k:v/total for k,v in values.items()}


def query_texts(query, parsed):
    subject = " ".join([*parsed["subject_attributes"],parsed["subject_specific_type"]]).strip()
    return {"scene":parsed["context"] or query, "people":" ".join(filter(None,(subject,parsed["action"],parsed["relation"]))),
            "vehicle":" ".join(filter(None,(subject,parsed["action"],parsed["relation"]))),
            "action":" ".join(filter(None,(subject,parsed["action"]))),
            "relation":" ".join(filter(None,(subject,parsed["relation"]))),
            "infrastructure":"; ".join(parsed["infrastructure"]) or query,"salient":query,"full":query}


def contains(text, term):
    return bool(re.search(r"\b"+re.escape(term)+r"s?\b",text,re.I))


def positive_contains(text, term):
    # Negative clauses cannot earn an exact-detail boost.
    return any(contains(clause,term) and not re.search(r"\b(?:no|not|without|never|uncertain|unknown)\b",clause,re.I)
               for clause in re.split(r"[;.!?]",text))


def subject_entities(record, parsed):
    subject = parsed["subject_type"]
    group = {"person":"people","vehicle":"vehicles","cyclist":"cyclists","motorcycle":"motorcycles"}.get(subject)
    if not group:
        return [],None
    entities = record[group]
    if group == "vehicles":
        specific = parsed["subject_specific_type"]
        if specific != "vehicle":
            # Cars may be described as SUVs; broad unknown types are retained
            # for semantic retrieval but never used as contradictory evidence.
            entities = [e for e in entities if not meaningful(e.get("type","")) or contains(e["type"],specific)
                        or specific == "car" and any(contains(e["type"],v) for v in ("suv","sedan","hatchback","car"))]
    return entities,group


def detail_adjustments(record, parsed):
    if record.get("fallback") or parsed.get("negated"):
        return 0.,0.,[],[]
    texts = record["searchable_texts"]
    entities,group = subject_entities(record,parsed)
    actor_text = "; ".join(actor_name(entity,group)+" "+" ".join(entity.values()) for entity in entities) if group else texts["full_text"]
    if parsed["subject_type"] == "infrastructure":
        specific = parsed["subject_specific_type"]
        key = specific.replace(" ","_")+("" if specific == "parking" else "s")
        actor_text = "; ".join(record["traffic_infrastructure"].get(key,[]))
    elif parsed["subject_type"] == "scene":
        actor_text = texts["scene_text"]
    details = [(a.split()[0],actor_text) for a in parsed["subject_attributes"]]
    details += [(i,texts["infrastructure_text"]) for i in parsed["infrastructure"]]
    side = next((s for s in ("left","right","ahead","behind") if contains(parsed["relation"],s)),"")
    if side:
        # Relation matches are scoped to the queried actor category.
        details.append((side,actor_text))
    hits = [term for term,text in details if positive_contains(text,term)]
    boost = .04*len(hits)/len(details) if details else 0.
    penalties = []
    color = next((a.split()[0] for a in parsed["subject_attributes"] if a.split()[0] in COLORS),"")
    if color and entities:
        values = [[e.get("upper_clothing_color",""),e.get("lower_clothing_color","")] if group == "people"
                  else [e.get("color","")] for e in entities]
        # All compatible actors must have explicitly known incompatible colors.
        # Unknown color or either garment matching => no contradiction.
        if all(all(meaningful(v) and any(contains(v,c) for c in COLORS) and not contains(v,color) for v in colors) for colors in values):
            penalties.append(("explicit incompatible actor color",.015))
    if side in ("left","right") and entities:
        opposite = "right" if side == "left" else "left"
        positions = ["; ".join(e.get(k,"") for k in ("relative_side","relative_position","position_start","position_end")) for e in entities]
        if all(positive_contains(v,opposite) and not contains(v,side) for v in positions):
            penalties.append(("explicit opposite actor side",.02))
    action = parsed["action"]
    if action and entities:
        # Only explicit actor-scoped negation, never a different positive action.
        action_values = ["; ".join(e.get(k,"") for k in ("action","activity","temporal_change","state")) for e in entities]
        negation = r"\b(?:no|not|never)\s+(?:\w+\s+){0,2}"+re.escape(action)+r"\b"
        if all(re.search(negation,v,re.I) for v in action_values):
            penalties.append(("explicit negated actor action",.04))
    return boost,sum(v for _,v in penalties),hits,[reason for reason,_ in penalties]


def matched_metadata(record, parsed):
    entities,group = subject_entities(record,parsed)
    facts = []
    for entity in entities:
        values = [v for k,v in entity.items() if meaningful(v) and k not in ("type","color","upper_clothing_color","lower_clothing_color")]
        facts.append(actor_name(entity,group)+" "+", ".join(values))
    terms = [a.split()[0] for a in parsed["subject_attributes"]]+[parsed["action"],parsed["relation"]]
    relevance = lambda fact:sum(positive_contains(fact,t) for t in terms if t)
    facts.sort(key=lambda fact:-relevance(fact))
    texts = record["searchable_texts"]
    def selected(field):
        values = texts[field+"_text"].split("; ")
        actors = [actor_name(entity,group) for entity in entities]
        scoped = [value for value in values if any(contains(value,actor) for actor in actors)]
        if group:
            values = scoped
        values.sort(key=lambda fact:-relevance(fact))
        return "; ".join(values[:3])[:280]
    return {"scene_summary":record["scene_summary"],"subject_facts":facts[:3],
            "action":selected("action") if parsed["action"] else "",
            "relation":selected("relation") if parsed["relation"] else "",
            "salient_facts":record["salient_facts"][:3]}


def score_dense(index, query, qfields, qproj, caption_k=100, stage3_k=100, rerank_k=50, *, loaded=None):
    if min(caption_k,stage3_k,rerank_k) <= 0:
        raise ValueError("Dense retrieval limits must be positive")
    index = Path(index)
    meta = loaded["meta"] if loaded is not None else read_meta(index)
    if np.asarray(qfields).shape != (len(FIELDS),meta["dimensions"]["bge"]) or np.asarray(qproj).shape != (512,):
        raise ValueError("Dense query embedding geometry mismatch")
    for values in (qfields,qproj):
        if not np.isfinite(values).all() or not np.allclose(np.linalg.norm(values,axis=-1),1,atol=1e-3):
            raise ValueError("Dense query embeddings must be finite and normalized")
    parsed = parse_query(query)
    weights = field_weights(parsed)
    records = loaded["records"] if loaded is not None else rows(index/"metadata/dense_metadata.jsonl")
    manifest = loaded["manifest"] if loaded is not None else rows(index/"manifest.jsonl")
    scores = {}
    for i,field in enumerate(FIELDS):
        matrix = loaded["arrays"][f"text/dense_{field}.npy"] if loaded is not None else array(index,f"text/dense_{field}.npy")
        value = np.empty(len(matrix),np.float32)
        for start in range(0,len(matrix),512):
            value[start:start+512] = matrix[start:start+512] @ qfields[i]
        # Missing metadata isn't a contradiction or a semantic hit.
        value[[j for j,r in enumerate(records) if r["fallback"] or not r["searchable_texts"][field+"_text"]]] = 0.
        scores[field] = value
    fused = loaded["arrays"]["visual/fused_embeddings.npy"] if loaded is not None else array(index,"visual/fused_embeddings.npy")
    s3 = np.empty(len(fused),np.float32)
    for start in range(0,len(fused),512):
        s3[start:start+512] = fused[start:start+512] @ qproj
    subject = {"person":"people","vehicle":"vehicle","cyclist":"vehicle","motorcycle":"vehicle",
               "infrastructure":"infrastructure","scene":"scene"}.get(parsed["subject_type"],"scene")
    channels = {field:scores[field] for field in dict.fromkeys(("full","salient",subject))}
    channels["stage3"] = s3
    ranks = {field:{int(j):rank+1 for rank,j in enumerate(np.argsort(-values,kind="stable")[:stage3_k if field == "stage3" else caption_k])}
             for field,values in channels.items()}
    union = set().union(*(set(v) for v in ranks.values()))
    rrf = lambda j:sum(1/(60+r[j]) for r in ranks.values() if j in r)
    candidates = sorted(union,key=lambda j:(-rrf(j),j))[:rerank_k]
    results = []
    for j in candidates:
        cosines = {field:float(scores[field][j]) for field in FIELDS}
        semantic = sum(weights[field]*cosines[field] for field in FIELDS)
        boost,penalty,hits,reasons = detail_adjustments(records[j],parsed)
        results.append({**{k:manifest[j][k] for k in ("row","clip_id","start_sec","end_sec")},
                        "ranker":"dense","final_score":semantic+boost-penalty,"dense_semantic_score":semantic,
                        "dense_cosines":cosines,"lexical_boost":boost,"contradiction_penalty":penalty,
                        "exact_detail_hits":hits,"contradiction_reasons":reasons,"stage3_candidate_score":float(s3[j]),
                        "metadata_fallback":records[j]["fallback"],"matched_metadata":matched_metadata(records[j],parsed)})
    results.sort(key=lambda r:(-r["final_score"],r["row"]))
    for rank,result in enumerate(results,1):
        result["rank"] = rank
    return {"format":"local-dense-candidates-v1","query":query,"parsed_query":parsed,"index":str(index.resolve()),
            "profile_version":PROFILE_VERSION,"field_weights":weights,"query_texts":query_texts(query,parsed),
            "retrieval":{"index_windows":len(records),"union_count":len(union),"candidate_count":len(results),
                         "channels":list(channels),"field_k":caption_k,"stage3_k":stage3_k,"rerank_k":rerank_k},
            "qwen_required":False,"candidates":results}


def search_dense(index, query, caption_k=100, stage3_k=100, rerank_k=50, top_k=10,
                 extract_top_k=0, result_dir=None, siglip_full_model=False, debug_scores=False,
                 candidate_table=None, not_found_stats=False, *, privacy=True, privacy_mode="gaussian",
                 privacy_strength="strong", privacy_block_size=16):
    if not query.strip() or min(top_k,caption_k,stage3_k,rerank_k) <= 0 or extract_top_k < 0:
        raise ValueError("Invalid dense query/retrieval settings")
    started = time.perf_counter()
    validate_index(index)
    meta = read_meta(index)
    from .text_encoder import BGEEncoder
    bge = BGEEncoder(meta["models"]["bge"]["path"],"cpu",local_files_only=True)
    texts = query_texts(query,parse_query(query))
    qfields = bge.encode([texts[f] for f in FIELDS],batch_size=8)
    del bge
    gc.collect()
    from .query_encoder import SigLIPQueryEncoder
    sig = SigLIPQueryEncoder(meta["checkpoints"]["stage2b"]["path"],"cpu",local_files_only=True,
                            model_path=meta["models"]["siglip"]["path"],text_only=not siglip_full_model)
    _,qproj = sig.encode([query],batch_size=1)
    del sig
    gc.collect()
    table = score_dense(index,query,qfields,qproj[0],caption_k,stage3_k,rerank_k)
    table["search_seconds"] = time.perf_counter()-started
    if candidate_table is not None:
        target,protected = Path(candidate_table).resolve(),Path(index).resolve()
        if target.exists() or target == protected or protected in target.parents:
            raise ValueError("Candidate table must be a new file outside the immutable index")
        atomic_json(target,table)
    results = table["candidates"][:top_k]
    print("Ranker: dense | Qwen required: NO | Parsed:",json.dumps(table["parsed_query"]))
    if not_found_stats:
        print("Not-found diagnostics:",json.dumps({"top1_score":results[0]["final_score"],
              "top1_minus_top2_margin":results[0]["final_score"]-results[1]["final_score"] if len(results)>1 else None,
              "threshold":None,"not_found_decision":"uncalibrated"}))
    if debug_scores:
        print("Dense weights:",json.dumps(table["field_weights"]))
        for result in table["candidates"]:
            print("Candidate debug:",json.dumps(result))
    for result in results:
        print(f"#{result['rank']:02d} {result['start_sec']:.3f}s -> {result['end_sec']:.3f}s dense={result['final_score']:.5f}")
        print("  matched metadata:",json.dumps(result["matched_metadata"],ensure_ascii=False))
    if extract_top_k:
        if sha256(meta["video"]["path"]) != meta["video"]["sha256"]:
            raise ValueError("Source video changed; refusing extraction")
        from .video_export import extract_clip
        from datetime import datetime
        target = Path(result_dir) if result_dir else Path(index)/"results"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        for result in results[:extract_top_k]:
            path = target/f"rank_{result['rank']:02d}_{result['start_sec']:.3f}_{result['end_sec']:.3f}.mp4"
            extract_clip(meta["video"]["path"],path,result["start_sec"],result["end_sec"],meta["video"]["fps"],
                         privacy=privacy, privacy_cache=Path(index)/"privacy", privacy_mode=privacy_mode,
                         privacy_strength=privacy_strength, privacy_block_size=privacy_block_size)
            print("Clip:",path)
    return results
