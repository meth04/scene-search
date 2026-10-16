
from __future__ import annotations
import argparse, csv, html, json, random
from pathlib import Path
import cv2
from .common import read_jsonl, cache_meta, contact_sheet

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--stage2a-cache-dir",default="./outputs/driveretriever_stage2a_cache")
    p.add_argument("--output-dir",default="./outputs/driveretriever_hybrid_v2")
    p.add_argument("--split",choices=["train","val","test"],default="test")
    p.add_argument("--samples",type=int,default=120)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--score-csv",default="")
    return p.parse_args()

def score_csv(path):
    rows=list(csv.DictReader(open(path,encoding="utf-8")))
    mapping={"relevant":"relevant","partial":"partial","irrelevant":"irrelevant",
             "1":"relevant",".5":"partial","0.5":"partial","0":"irrelevant"}
    valid=[]
    for r in rows:
        h=mapping.get(r.get("human_label","").strip().lower())
        if h: valid.append((r["vlm_label"],h))
    if not valid:
        print("No human labels found. Fill human_label with relevant/partial/irrelevant."); return
    acc=sum(a==b for a,b in valid)/len(valid)
    print(json.dumps({"reviewed":len(valid),"exact_agreement":acc},indent=2))

def main():
    a=parse_args()
    if a.score_csv:
        score_csv(a.score_csv); return
    root=Path(a.output_dir)
    queries=read_jsonl(root/"mining"/a.split/"queries.jsonl")
    judgments=read_jsonl(root/"judgments"/a.split/"judgments.jsonl")
    manifest=read_jsonl(Path(a.stage2a_cache_dir)/a.split/"clips.jsonl")
    videos=Path(cache_meta(a.stage2a_cache_dir)["videos_dir"])
    pool=[]
    for jr in judgments:
        qi=int(jr["query_index"])
        for p in jr["pairs"]:
            if p.get("auto_source"): continue
            pool.append((qi,p))
    rng=random.Random(a.seed)
    bylab={}
    for item in pool: bylab.setdefault(item[1]["label"],[]).append(item)
    chosen=[]
    each=max(1,a.samples//3)
    for lab in ("relevant","partial","irrelevant"):
        arr=bylab.get(lab,[])
        chosen+=rng.sample(arr,min(each,len(arr)))
    if len(chosen)<a.samples:
        rest=[x for x in pool if x not in chosen]
        chosen+=rng.sample(rest,min(a.samples-len(chosen),len(rest)))

    out=root/"audit"/a.split; imgdir=out/"images"; imgdir.mkdir(parents=True,exist_ok=True)
    csvpath=out/"audit.csv"
    rows=[]
    for n,(qi,p) in enumerate(chosen):
        j=int(p["candidate_index"])
        im=contact_sheet(manifest[j],videos,n=6)
        fn=f"{n:04d}.jpg"; cv2.imwrite(str(imgdir/fn),im)
        rows.append({
            "sample":n,"query":queries[qi]["text"],"candidate_clip_id":manifest[j]["clip_id"],
            "vlm_label":p["label"],"vlm_confidence":p["confidence"],"vlm_reason":p["reason"],
            "image":f"images/{fn}","human_label":"","notes":"",
        })
    with csvpath.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)

    trs=[]
    for r in rows:
        trs.append(
            "<tr><td>{}</td><td>{}</td><td><img src='{}' width='640'></td>"
            "<td>{} ({:.2f})<br>{}</td><td>fill audit.csv</td></tr>".format(
                r["sample"],html.escape(r["query"]),r["image"],r["vlm_label"],
                float(r["vlm_confidence"]),html.escape(r["vlm_reason"])
            )
        )
    page="<html><body><h2>Hybrid V2 VLM Audit</h2><p>Review images and fill human_label in audit.csv.</p><table border=1>"+\
         "<tr><th>#</th><th>Query</th><th>Candidate</th><th>VLM judge</th><th>Human</th></tr>"+\
         "".join(trs)+"</table></body></html>"
    (out/"audit.html").write_text(page,encoding="utf-8")
    print(out/"audit.html"); print(csvpath)

if __name__=="__main__": main()
