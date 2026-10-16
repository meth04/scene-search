
from __future__ import annotations
import argparse, json, traceback, re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm.auto import tqdm
from .common import read_jsonl, write_jsonl, write_json, cache_meta, contact_sheet, jpeg_data_url, clean_text
from .vlm_client import VLLMClient

from .caption_quality import CAPTION_PROMPT

def normalize(obj):
    if not isinstance(obj,dict): raise ValueError("caption response must be JSON object")
    out={}
    out["description"]=clean_text(obj.get("description",""))
    for k in ("objects","other_actions","relations","road_context","traffic_control","events"):
        v=obj.get(k,[])
        if isinstance(v,str): v=[v]
        out[k]=[clean_text(x) for x in v if clean_text(x)]
    out["ego_action"]=clean_text(obj.get("ego_action",""))
    out["weather"]=clean_text(obj.get("weather",""))
    qs=[]
    for q in obj.get("queries",[]) or []:
        if isinstance(q,str):
            q={"text":q}
        if not isinstance(q,dict): continue
        text=clean_text(q.get("text",""))
        if len(text)<4: continue

        low = text.lower()

        # Reject absence/negative pseudo-queries.
        banned_negative = [
            "no ", "without ", "none visible",
            "not visible", "no visible"
        ]
        if any(x in low for x in banned_negative):
            continue

        # Reject artifacts from the contact-sheet representation.
        if re.search(r"\\bt[1-9]/\\d+\\b|\\bframe\\s*\\d+\\b|contact[- ]sheet", low):
            continue

        # OCR belongs to a separate module; do not train semantic retrieval on
        # potentially hallucinated literal sign text.
        if ("indicating" in low or "reads " in low or "says " in low) and (
            "'" in text or '"' in text or any(c.isdigit() for c in text)
        ):
            continue

        qs.append({
            "text":text,
            "action":clean_text(q.get("action","")),
            "relation":clean_text(q.get("relation","")),
            "context":clean_text(q.get("context","")),
        })
    if not out["description"]: raise ValueError("empty description")
    if not qs:
        qs=[{"text":out["description"],"action":out["ego_action"],"relation":"","context":""}]
    out["queries"]=qs[:3]
    return out

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--stage2a-cache-dir",default="./outputs/driveretriever_stage2a_cache")
    p.add_argument("--output-dir",default="./outputs/driveretriever_hybrid_v2")
    p.add_argument("--split",choices=["train","val","test","all"],default="all")
    p.add_argument("--base-url",default="http://127.0.0.1:8000/v1")
    p.add_argument("--vlm-model",default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--frames",type=int,default=6)
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--max-clips",type=int,default=0)
    p.add_argument("--rebuild",action="store_true")
    return p.parse_args()

def build_split(a,split,videos_dir):
    records=read_jsonl(Path(a.stage2a_cache_dir)/split/"clips.jsonl")
    if a.max_clips>0: records=records[:a.max_clips]
    out=Path(a.output_dir)/"captions"/split
    rows_dir=out/"rows"; rows_dir.mkdir(parents=True,exist_ok=True)
    cli=VLLMClient(a.base_url,a.vlm_model)

    def one(i):
        fp=rows_dir/f"{i:06d}.json"
        if fp.exists() and not a.rebuild:
            return i,"skip",""
        r=records[i]
        try:
            sheet=contact_sheet(r,videos_dir,n=a.frames)
            obj=cli.chat_json(CAPTION_PROMPT,[jpeg_data_url(sheet)],max_tokens=900,temperature=.05)
            cap=normalize(obj)
            result={"row":i,"clip_id":r["clip_id"],"video_id":r["video_id"],**cap}
            fp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
            return i,"ok",""
        except Exception as e:
            err=out/"errors"; err.mkdir(parents=True,exist_ok=True)
            (err/f"{i:06d}.txt").write_text(traceback.format_exc(),encoding="utf-8")
            return i,"error",str(e)

    # Only schedule missing clips. Existing successful rows are not
    # submitted to the thread pool, so resume progress reflects real work.
    todo = [
        i for i in range(len(records))
        if a.rebuild or not (rows_dir / f"{i:06d}.json").exists()
    ]

    print(f"[resume] total={len(records)} completed={len(records)-len(todo)} remaining={len(todo)}")
    counts={"ok":0,"skip":0,"error":0}
    if a.workers<=1:
        it=(one(i) for i in todo)
        for _,st,msg in tqdm(it,total=len(todo),desc=f"caption/{split}"):
            counts[st]+=1
    else:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs=[ex.submit(one,i) for i in todo]
            for fut in tqdm(as_completed(futs),total=len(futs),desc=f"caption/{split}"):
                _,st,msg=fut.result(); counts[st]+=1

    # ------------------------------------------------------------------
    # FALLBACK FOR PERSISTENT VLM FAILURES
    # ------------------------------------------------------------------
    # Preserve absolute row alignment. Failed VLM clips receive the
    # original CoVLA caption as description but NO generated queries.
    # Therefore they remain valid retrieval candidates without injecting
    # noisy pseudo-query supervision into Stage5.
    fallback_count = 0

    for i in range(len(records)):
        fp = rows_dir / f"{i:06d}.json"

        if fp.exists():
            continue

        r = records[i]

        tv = r.get("text_variants")
        if isinstance(tv, list) and tv:
            desc = str(tv[0]).strip()
        elif isinstance(tv, str) and tv.strip():
            desc = tv.strip()
        else:
            desc = ""

        if not desc:
            for key in ("caption", "text", "description"):
                value = r.get(key)
                if isinstance(value, str) and value.strip():
                    desc = value.strip()
                    break

        if not desc:
            desc = "Dashcam driving scene."

        fallback = {
            "row": i,
            "clip_id": r["clip_id"],
            "video_id": r["video_id"],
            "description": desc,
            "objects": [],
            "ego_action": "",
            "other_actions": [],
            "relations": [],
            "road_context": [],
            "traffic_control": [],
            "weather": "",
            "events": [],
            "queries": [],
            "fallback": True,
            "fallback_reason": "VLM generation repeatedly failed"
        }

        fp.write_text(
            json.dumps(fallback, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

        fallback_count += 1

    if fallback_count:
        print(f"[fallback] created {fallback_count} alignment-preserving rows")

    final=[]
    missing=[]
    for i in range(len(records)):
        fp=rows_dir/f"{i:06d}.json"
        if fp.exists():
            final.append(json.loads(fp.read_text(encoding="utf-8")))
        else:
            missing.append(i)
    write_jsonl(out/"captions.jsonl",final)
    write_json(out/"stats.json",{"split":split,"expected":len(records),"written":len(final),"missing":len(missing),**counts})
    print(json.dumps({"split":split,"expected":len(records),"written":len(final),"missing":len(missing),**counts},indent=2))
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} captions missing. Re-run caption mode; it resumes automatically.")

def main():
    a=parse_args()
    meta=cache_meta(a.stage2a_cache_dir)
    videos_dir=Path(meta["videos_dir"])
    cli=VLLMClient(a.base_url,a.vlm_model)
    print("[vLLM]",cli.health())
    splits=["train","val","test"] if a.split=="all" else [a.split]
    for sp in splits: build_split(a,sp,videos_dir)

if __name__=="__main__": main()
