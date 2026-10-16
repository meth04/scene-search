
from __future__ import annotations
import argparse, subprocess, sys

def run(module,args):
    cmd=[sys.executable,"-m",f"hybrid_v2.{module}"]+args
    print("\n"+"="*100+"\nRUN: "+" ".join(cmd)+"\n"+"="*100)
    subprocess.check_call(cmd)

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["caption","embed","mine","judge","features","train","eval","audit","full"],required=True)
    p.add_argument("--stage2a-cache-dir",default="./outputs/driveretriever_stage2a_cache")
    p.add_argument("--stage4-cache-dir",default="./outputs/driveretriever_stage4_cache")
    p.add_argument("--frame-cache-dir",default="./outputs/driveretriever_stage4b_frame_cache")
    p.add_argument("--stage2b-checkpoint",default="./outputs/driveretriever_stage2b_vision/best_composite.pt")
    p.add_argument("--output-dir",default="./outputs/driveretriever_hybrid_v2")
    p.add_argument("--base-url",default="http://127.0.0.1:8000/v1")
    p.add_argument("--vlm-model",default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--split",default="all")
    p.add_argument("--rebuild",action="store_true")
    return p.parse_args()

def main():
    a=parse_args()
    common=["--output-dir",a.output_dir]
    if a.mode in ("caption","full"):
        x=common+["--stage2a-cache-dir",a.stage2a_cache_dir,"--split",a.split,"--base-url",a.base_url,
                  "--vlm-model",a.vlm_model,"--workers",str(a.workers)]
        if a.rebuild: x+=["--rebuild"]
        run("caption_dataset",x)
        if a.mode=="caption": return
    if a.mode in ("embed","full"):
        x=common+["--split",a.split]
        if a.rebuild: x+=["--rebuild"]
        run("embed_captions",x)
        if a.mode=="embed": return
    if a.mode in ("mine","full"):
        x=common+["--stage2a-cache-dir",a.stage2a_cache_dir,"--stage4-cache-dir",a.stage4_cache_dir,
                  "--stage2b-checkpoint",a.stage2b_checkpoint,"--split",a.split]
        if a.rebuild: x+=["--rebuild"]
        run("mine_candidates",x)
        if a.mode=="mine": return
    if a.mode in ("judge","full"):
        x=common+["--stage2a-cache-dir",a.stage2a_cache_dir,"--split",a.split,"--base-url",a.base_url,
                  "--vlm-model",a.vlm_model,"--workers",str(a.workers)]
        if a.rebuild: x+=["--rebuild"]
        run("judge_pairs",x)
        if a.mode=="judge": return
    if a.mode in ("features","full"):
        x=common+["--stage2a-cache-dir",a.stage2a_cache_dir,"--stage4-cache-dir",a.stage4_cache_dir,
                  "--frame-cache-dir",a.frame_cache_dir,"--split",a.split]
        if a.rebuild: x+=["--rebuild"]
        run("build_features",x)
        if a.mode=="features": return
    if a.mode in ("train","full"):
        run("train_stage5",common)
        if a.mode=="train": return
    if a.mode in ("eval","full"):
        run("eval_stage5",common+["--split","test"])
        if a.mode=="eval": return
    if a.mode in ("audit","full"):
        run("audit",common+["--stage2a-cache-dir",a.stage2a_cache_dir,"--split","test"])
        if a.mode=="audit": return

if __name__=="__main__": main()
