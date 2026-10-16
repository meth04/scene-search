
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
from .common import read_jsonl, write_json, l2norm, jaccard, zscore1d

FEATURE_NAMES=[
    "caption_full_cos",
    "caption_action_cos",
    "caption_relation_cos",
    "caption_context_cos",
    "stage3_cos",
    "frame_proj_max",
    "frame_proj_top3",
    "frame_proj_mean",
    "generic_siglip_max",
    "generic_siglip_top3",
    "generic_siglip_mean",
    "caption_rr",
    "stage3_rr",
    "lexical_caption",
    "lexical_action",
    "lexical_relation",
    "caption_x_stage3",
    "action_x_relation",
]

def join_list(x):
    return "; ".join([str(v) for v in (x or []) if str(v).strip()])

def cap_fields(r):
    action="; ".join(x for x in [r.get("ego_action",""),join_list(r.get("other_actions")),join_list(r.get("events"))] if x)
    relation=join_list(r.get("relations"))
    context="; ".join(x for x in [join_list(r.get("road_context")),join_list(r.get("traffic_control")),r.get("weather","")] if x)
    return r.get("description",""),action,relation,context

def rr(rank):
    return 0.0 if int(rank)<1 else 1.0/math.log2(int(rank)+1.0)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--stage2a-cache-dir",default="./outputs/driveretriever_stage2a_cache")
    p.add_argument("--stage4-cache-dir",default="./outputs/driveretriever_stage4_cache")
    p.add_argument("--frame-cache-dir",default="./outputs/driveretriever_stage4b_frame_cache")
    p.add_argument("--output-dir",default="./outputs/driveretriever_hybrid_v2")
    p.add_argument("--split",choices=["train","val","test","all"],default="all")
    p.add_argument("--min-confidence",type=float,default=.0)
    p.add_argument("--rebuild",action="store_true")
    return p.parse_args()

def one(a,sp):
    out=Path(a.output_dir)/"features"; out.mkdir(parents=True,exist_ok=True)
    target=out/f"{sp}.npz"
    if target.exists() and not a.rebuild:
        print(f"[skip] {sp} features exist"); return

    queries=read_jsonl(Path(a.output_dir)/"mining"/sp/"queries.jsonl")
    captions=read_jsonl(Path(a.output_dir)/"captions"/sp/"captions.jsonl")
    judgments=read_jsonl(Path(a.output_dir)/"judgments"/sp/"judgments.jsonl")
    pack=np.load(Path(a.output_dir)/"mining"/sp/"candidates.npz")
    qz=np.load(Path(a.output_dir)/"mining"/sp/"query_embeddings.npz")

    cand_idx=pack["candidate_indices"]
    cr=pack["caption_rank"]; sr=pack["stage3_rank"]
    K=cand_idx.shape[1]; Q=len(queries)

    full=np.load(Path(a.output_dir)/"text_index"/sp/"caption_full.npy",mmap_mode="r")
    act=np.load(Path(a.output_dir)/"text_index"/sp/"caption_action.npy",mmap_mode="r")
    rel=np.load(Path(a.output_dir)/"text_index"/sp/"caption_relation.npy",mmap_mode="r")
    ctx=np.load(Path(a.output_dir)/"text_index"/sp/"caption_context.npy",mmap_mode="r")
    fused=np.load(Path(a.stage4_cache_dir)/sp/"fused_embeddings.npy",mmap_mode="r")
    pframes=np.load(Path(a.frame_cache_dir)/sp/"frame_embeddings.npy",mmap_mode="r")
    rawframes=np.load(Path(a.stage2a_cache_dir)/sp/"frame_features.npy",mmap_mode="r")

    X=np.zeros((Q,K,len(FEATURE_NAMES)),np.float32)
    base=np.full((Q,K),-1e9,np.float32)
    labels=np.zeros((Q,K),np.float32)
    conf=np.zeros((Q,K),np.float32)
    mask=np.zeros((Q,K),np.bool_)

    judge_map={int(r["query_index"]):{int(p["candidate_index"]):p for p in r["pairs"]} for r in judgments}

    for qi in tqdm(range(Q),desc=f"features/{sp}"):
        q=queries[qi]
        qfull=qz["full"][qi]; qact=qz["action"][qi]; qrel=qz["relation"][qi]; qctx=qz["context"][qi]
        qraw=qz["siglip_raw"][qi]; qproj=qz["stage3_proj"][qi]
        valid_positions=[]
        for pos,j in enumerate(cand_idx[qi].tolist()):
            if j<0: continue
            valid_positions.append(pos)
            cf=np.asarray(full[j],np.float32); ca=np.asarray(act[j],np.float32)
            rrn=np.asarray(rel[j],np.float32); cc=np.asarray(ctx[j],np.float32)
            fu=np.asarray(fused[j],np.float32); fu=fu/(np.linalg.norm(fu)+1e-8)

            pf=np.asarray(pframes[j],np.float32)
            pf=pf/np.maximum(np.linalg.norm(pf,axis=-1,keepdims=True),1e-8)
            ps=pf@qproj
            pmax=float(ps.max()); pmean=float(ps.mean()); ptop=float(np.sort(ps)[-min(3,len(ps)):].mean())

            rf=np.asarray(rawframes[j],np.float32)
            rf=rf/np.maximum(np.linalg.norm(rf,axis=-1,keepdims=True),1e-8)
            rsim=rf@qraw
            rmax=float(rsim.max()); rmean=float(rsim.mean()); rtop=float(np.sort(rsim)[-min(3,len(rsim)):].mean())

            cap_desc,cap_action,cap_relation,cap_context=cap_fields(captions[j])
            action_cos=float(qact@ca) if q.get("action","").strip() else 0.0
            relation_cos=float(qrel@rrn) if q.get("relation","").strip() else 0.0
            context_cos=float(qctx@cc) if q.get("context","").strip() else 0.0
            capcos=float(qfull@cf); s3=float(qproj@fu)

            feats=[
                capcos, action_cos, relation_cos, context_cos, s3,
                pmax,ptop,pmean,rmax,rtop,rmean,
                rr(cr[qi,pos]),rr(sr[qi,pos]),
                jaccard(q["text"],cap_desc),
                jaccard(q.get("action",""),cap_action) if q.get("action","") else 0.0,
                jaccard(q.get("relation",""),cap_relation) if q.get("relation","") else 0.0,
                capcos*s3,action_cos*relation_cos,
            ]
            X[qi,pos]=np.asarray(feats,np.float32)
            jp=judge_map.get(qi,{}).get(j)
            if jp is not None:
                labels[qi,pos] = float(jp["target"])

                # Keep every successfully judged pair.
                # Confidence is a LOSS WEIGHT, not a hard filtering criterion.
                # Hard filtering at >=0.70 previously removed almost every
                # negative/partial candidate and collapsed 3000 queries to ~5.
                if jp.get("auto_source", False):
                    conf[qi,pos] = 1.0
                else:
                    conf[qi,pos] = max(
                        0.05,
                        min(1.0, float(jp.get("confidence", 0.5)))
                    )

                mask[qi,pos] = True

        if valid_positions:
            v=np.array(valid_positions,dtype=np.int64)
            # Conservative, interpretable fallback score. Per-query z-scores prevent
            # incompatible raw similarity scales from dominating.
            comps=[
                (.30,X[qi,v,0]), # caption
                (.20,X[qi,v,4]), # Stage3A
                (.15,X[qi,v,6]), # projected frame top3
                (.15,X[qi,v,1]), # action
                (.10,X[qi,v,2]), # relation
                (.10,X[qi,v,3]), # context
            ]
            b=np.zeros(len(v),np.float32)
            for w,x in comps: b+=float(w)*zscore1d(x)
            base[qi,v]=b

    # Keep queries with >=1 relevant and >=1 non-relevant valid pair.
    keep=[]
    for i in range(Q):
        y=labels[i][mask[i]]
        if len(y)>=2 and np.any(y>=.75) and np.any(y<=.5): keep.append(i)
    keep=np.asarray(keep,dtype=np.int64)
    np.savez_compressed(target,
        features=X[keep],base_score=base[keep],labels=labels[keep],confidence=conf[keep],
        mask=mask[keep],candidate_indices=cand_idx[keep],
        original_query_index=keep,source_retrieved=pack["source_retrieved"][keep])
    write_json(out/f"{sp}_meta.json",{
        "split":sp,"input_queries":Q,"kept_queries":len(keep),"candidate_k":K,
        "feature_names":FEATURE_NAMES,"min_confidence":a.min_confidence,
        "source_recall_union_kept":float(pack["source_retrieved"][keep].mean()) if len(keep) else 0.0,
    })
    print(json.dumps(json.loads((out/f"{sp}_meta.json").read_text()),indent=2))

def main():
    a=parse_args()
    for sp in (["train","val","test"] if a.split=="all" else [a.split]): one(a,sp)

if __name__=="__main__": main()
