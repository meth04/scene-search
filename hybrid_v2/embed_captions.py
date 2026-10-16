
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .common import read_jsonl, write_json
from .text_encoder import BGEEncoder

def join_list(x):
    return "; ".join([str(v) for v in (x or []) if str(v).strip()])

def fields(r):
    full=r.get("description","")
    action="; ".join(x for x in [r.get("ego_action",""),join_list(r.get("other_actions")),join_list(r.get("events"))] if x)
    relation=join_list(r.get("relations"))
    context="; ".join(x for x in [join_list(r.get("road_context")),join_list(r.get("traffic_control")),r.get("weather","")] if x)
    return full,action or "none",relation or "none",context or "none"

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",default="./outputs/driveretriever_hybrid_v2")
    p.add_argument("--split",choices=["train","val","test","all"],default="all")
    p.add_argument("--text-model",default="BAAI/bge-base-en-v1.5")
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--device",default="cuda")
    p.add_argument("--rebuild",action="store_true")
    return p.parse_args()

def one(a,sp,enc):
    d=Path(a.output_dir)/"captions"/sp
    rows=read_jsonl(d/"captions.jsonl")
    if not rows: raise RuntimeError(f"No captions: {d}")
    out=Path(a.output_dir)/"text_index"/sp; out.mkdir(parents=True,exist_ok=True)
    fp=out/"caption_full.npy"
    if fp.exists() and not a.rebuild:
        print(f"[skip] {sp} caption embeddings exist"); return
    cols=list(zip(*[fields(r) for r in rows]))
    names=["caption_full","caption_action","caption_relation","caption_context"]
    for name,texts in zip(names,cols):
        z=enc.encode(texts,batch_size=a.batch_size)
        np.save(out/f"{name}.npy",z.astype(np.float32))
        print(sp,name,z.shape)
    write_json(out/"stats.json",{"split":sp,"clips":len(rows),"model":a.text_model})

def main():
    a=parse_args(); enc=BGEEncoder(a.text_model,a.device)
    for sp in (["train","val","test"] if a.split=="all" else [a.split]): one(a,sp,enc)

if __name__=="__main__": main()
