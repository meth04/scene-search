"""Read-only inspection, preservation snapshot and pinned TRAIN acquisition."""
from __future__ import annotations
import concurrent.futures
import json
import random
import sys
import subprocess
import socket
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hybrid_v2.embeddinggemma2 import digest, identity, cached_model_path
from hybrid_v2.embeddinggemma2_index import write_json

OUT = ROOT / 'outputs/driveevidence'
def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def snapshot():
    target = OUT / 'hashes/preservation_before.json'
    if target.exists():
        return
    files = {}
    unavailable = []
    roots = [ROOT/'outputs/local_hybrid_indexes', ROOT/'outputs/embeddinggemma2',
             ROOT/'outputs/metadata_v4_1_pilot50', ROOT/'outputs/metadata_v4_2',
             ROOT/'outputs/roadscapes_hf_200_e2e/query_benchmark', ROOT/'data/model-cache',
             ROOT/'data/model_cache', ROOT/'outputs/driveretriever_stage2b_vision',
             ROOT/'outputs/driveretriever_stage3a', ROOT/'outputs/driveretriever_stage4b',
             ROOT/'outputs/driveretriever_hybrid_v2/stage5']
    for base in roots:
        for p in base.rglob('*'):
            if p.is_file():
                try:
                    files[str(p.relative_to(ROOT))] = digest(p)
                except OSError as exc:
                    unavailable.append({'path':str(p), 'error':repr(exc)})
    for p in list(ROOT.glob('CODEX_*REPORT.md')) + list((ROOT/'hybrid_v2').glob('embeddinggemma2*.py')):
        files[str(p.relative_to(ROOT))] = digest(p)
    write_json(target, {'files':files, 'unavailable':unavailable})

def inspect():
    from hybrid_v2.local_runtime import load_checkpoint
    source = ROOT/'outputs/local_hybrid_indexes/roadscapes_hf_200_e2e_v2'
    meta = read(source/'index_meta.json')
    stages = {}
    for stage, cp, code, inputs, outputs, decision, constraint in [
      ('Stage1','data/model-cache/driveretriever-epoch4/best.pt','src/indexing/models/driveretriever.py',
       'SigLIP2 frame features and text features','normalized clip/text vectors; auxiliary heads',
       'NOT_USED_INCOMPATIBLE','Checkpoint exists and is inspected. Original retrieval requires projected SigLIP2 text features, unavailable under the EG2-only query contract. Stage2B supplies the separate frozen visual-neighbor ablation; no cross-space query substitution.'),
      ('Stage2B','outputs/driveretriever_stage2b_vision/best_composite.pt','hybrid_v2/local_video_index.py',
       '12 RGB 224x224 frames; frozen SigLIP2 plus adapted vision and Stage2A head','512d normalized clip features',
       'USED','Separately stored features; cross-space EG2/SigLIP cosine prohibited. Query channel uses visual neighborhood support from EG2 anchors, no SigLIP query inference.'),
      ('Stage3A','outputs/driveretriever_stage3a/best_composite.pt','stage3a/stage3a_motion.py',
       'YOLO11n/ByteTrack observations; base clip vector; 24x29 track slots','aggregate motion slots and optional 512d fused embedding',
       'USED','Aligned aggregate tracks can be reused, but contain no observation boxes, confidence, original IDs or exact duration. Frozen detector rerun provides exact ROI observations; no fabricated track boxes.'),
      ('Stage4B','outputs/driveretriever_stage4b/best_reranker.pt','stage4b/stage4b_model.py',
       'q3,q2,fused,projected SigLIP frames,retrieve_score and original rank order','learned late-interaction plus residual score',
       'NOT_USED_INCOMPATIBLE','INCOMPATIBLE with EG2-only text query contract: requires original 512d q2/q3 features and rank distribution. No padding, substitution or retraining.')]:
        p = ROOT/cp
        ck = load_checkpoint(p) if p.is_file() else None
        stages[stage] = {'checkpoint_path':cp,'checkpoint_sha256':digest(p) if p.is_file() else None,
            'code_path':code,'code_sha256':digest(ROOT/code),'original_input':inputs,'original_output':outputs,
            'frozen_feature_availability':bool(ck),'checkpoint_keys':list(ck) if ck else [],
            'checkpoint_format':ck.get('format') if ck else None, 'index_time_usability':decision=='USED',
            'rerank_time_usability':stage in ('Stage2B','Stage3A'), 'compatibility_constraints':constraint,
            'compatibility':'INCOMPATIBLE' if stage in ('Stage1','Stage4B') else 'ORIGINAL_INPUTS_ONLY', 'decision':decision}
    verified = {}
    for phase in ('visual','motion','frame-project','fuse'):
        for name, h in meta['phases'][phase]['artifacts'].items():
            if digest(source/name) != h:
                raise ValueError('Invalid stage reuse hash: '+name)
            verified[name]=h
    stages['reuse_validation']={'source_meta_sha256':digest(source/'index_meta.json'),
         'manifest_sha256':digest(source/'manifest.jsonl'),'artifacts':verified,
         'row_alignment':'Exact source row lookup; source/window SHA256 checked before reuse',
         'historical_precision':'Source Stage2B inference used fp16; reused frozen features are float32. New inference uses float32.'}
    write_json(OUT/'inspection/stage_reuse_matrix.json',stages)

def prepare():
    for d in ('inspection','dev/corpus','dev/queries','dev/indexes','dev/optimization','dev/ablations',
              'dev/performance','pilot50/index','pilot50/retrieval','pilot50/per_query','pilot50/categories',
              'pilot50/candidate_recall','pilot50/performance','full200/index','full200/retrieval',
              'full200/per_query','full200/categories','full200/candidate_recall','full200/performance',
              'full200/not_found','tests','hashes','logs'):
        (OUT/d).mkdir(parents=True,exist_ok=True)
    snapshot()
    inspect()
    inv=read(ROOT/'outputs/roadscapes_hf_200_e2e/dataset_inventory.json')
    groups=defaultdict(list)
    for f in inv['files']:
        if f['path'].startswith('train/') and f['path'].endswith('.mp4'):
            groups[f['path'].split('/')[1]].append(f)
    rng=random.Random(20261009)
    names=sorted(groups)
    rng.shuffle(names)
    for name in names:
        groups[name].sort(key=lambda f:f['path'])
        rng.shuffle(groups[name])
    selected=[]
    while len(selected)<100:
        for name in names:
            if groups[name] and len(selected)<100:
                selected.append(groups[name].pop())
    # Some regional CDN addresses accept connections but stall without sending bytes.
    # Select an alternate DNS-advertised US edge; preserve the original HTTPS host,
    # certificate validation and mandatory file-content hashes. No OS DNS changes.
    addresses=[]
    download_needed=any(not (OUT/'dev/corpus/raw'/f['path']).is_file() or digest(OUT/'dev/corpus/raw'/f['path'])!=f['lfs']['sha256'] for f in selected)
    if download_needed:
        import requests
        answer=requests.get('https://dns.google/resolve',params={'name':'us.aws.cdn.hf.co','type':'A','edns_client_subnet':'8.8.8.8/24'},timeout=15).json()
        addresses=[a['data'] for a in answer.get('Answer',[]) if a['type']==1]
        if not addresses: raise RuntimeError('No alternate dataset CDN addresses available')
        write_json(OUT/'dev/corpus/acquisition_transport.json',{'host':'us.aws.cdn.hf.co','dns_provider':'dns.google',
           'edns_client_subnet':'8.8.8.8/24','addresses':addresses,'TLS_validation':True,'source_hash_required':True})
    selection={'dataset':inv['dataset'],'revision':inv['hf_revision'],'seed':20261009,'files':selected}
    write_json(OUT/'dev/corpus/selection.json',selection)
    def acquire(f):
        p=OUT/'dev/corpus/raw'/f['path']
        p.parent.mkdir(parents=True,exist_ok=True)
        expected=f['lfs']['sha256']
        local_root=os.environ.get('ROADSCAPES_TRAIN_ROOT')
        if local_root:
            import shutil
            local=Path(local_root)/f['path']
            if not local.exists(): local=Path(local_root)/Path(f['path']).relative_to('train')
            if local.is_file() and digest(local)==expected: shutil.copy2(local,p)
        if not p.is_file() or digest(p)!=expected:
            url=f"https://huggingface.co/datasets/{inv['dataset']}/resolve/{inv['hf_revision']}/{f['path']}"
            tmp=p.with_suffix('.download')
            ip=addresses[int(expected[:8],16)%len(addresses)]
            result=subprocess.run(['curl.exe','--fail','--location','--connect-timeout','10','--max-time','240',
                                   '--resolve','us.aws.cdn.hf.co:443:'+ip,
                                   '--silent','--show-error','--output',str(tmp),url],capture_output=True,text=True,timeout=250,
                                   creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode: raise RuntimeError('Pinned TRAIN download failed: '+f['path']+' '+result.stderr.strip())
            if digest(tmp)!=expected: raise ValueError('Source LFS mismatch')
            tmp.replace(p)
        print('TRAIN acquired and hash verified: '+f['path'],flush=True)
        return {'source_clip_id':f['path'],'source_path':str(p),'source_sha256':expected,
                'day_night':'Day' if 'Sequence_Day_' in f['path'] else 'Night'}
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=12)
    futures=[pool.submit(acquire,f) for f in selected]
    try:
        rows=[future.result() for future in futures]
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
    test=read(ROOT/'data/evaluation/roadscapes_hf_200/timeline.json')
    if {r['source_sha256'] for r in rows} & {digest(r['source_path']) for r in test}:
        raise ValueError('TRAIN/test content overlap')
    for j,r in enumerate(rows):
        r.update(row=j,source_row=j,clip_id=r['source_clip_id'],segment_id=r['source_clip_id'],
                 start_sec=j*5.,end_sec=(j+1)*5.,source_start_sec=0.,source_end_sec=5.,split='train')
        r['window_sha256']=identity(r)
    value={'dataset':inv['dataset'],'revision':inv['hf_revision'],'seed':20261009,'split':'train',
           'clips':rows,'corpus_sha256':identity(rows),'test_disjoint':True,'count':len(rows)}
    target=OUT/'dev/corpus/manifest.json'
    if target.exists() and read(target)!=value: raise ValueError('Frozen dev corpus differs')
    write_json(target,value)
    print('TRAIN corpus ready: 100 pinned, hash-verified clips',flush=True)

if __name__=='__main__': prepare()
