"""Independent source selection and contact sheets; no evidence-derived labels."""
from __future__ import annotations
import sys, json, random, concurrent.futures, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hybrid_v2.embeddinggemma2 import digest, identity
from hybrid_v2.embeddinggemma2_index import write_json
OUT=ROOT/'outputs/driveevidence_v2'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def preservation(verify=False):
    p=OUT/'hashes/before.json'
    if verify:
        files=read(p)['files']; changed=[n for n,h in files.items() if not (ROOT/n).is_file() or digest(ROOT/n)!=h]
        write_json(OUT/'hashes/after.json',{'files':len(files),'changed':changed,'status':'FAIL' if changed else 'PASS'})
        if changed: raise ValueError(changed)
        return
    if p.exists(): return
    bases=['outputs/local_hybrid_indexes','outputs/embeddinggemma2','outputs/driveevidence','outputs/metadata_v4_laya',
           'outputs/metadata_v4_1_pilot50','outputs/metadata_v4_2','outputs/roadscapes_hf_200_e2e',
           'data/model-cache','data/model_cache','data/evaluation','outputs/driveretriever_stage3a',
           'outputs/driveretriever_stage2b_vision','outputs/driveretriever_stage4b','outputs/driveretriever_hybrid_v2/stage5','src/privacy']
    paths={p for b in bases for p in (ROOT/b).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    paths.update(ROOT.glob('CODEX_*REPORT.md'))
    paths.update((ROOT/'hybrid_v2').glob('embeddinggemma2*.py'))
    paths.update([ROOT/'hybrid_v2/driveevidence.py',ROOT/'hybrid_v2/driveevidence_worker.py',ROOT/'hybrid_v2/decision_provider.py'])
    write_json(p,{'files':{str(f.relative_to(ROOT)):digest(f) for f in sorted(paths)}})
    print('Preservation baseline frozen',len(paths),flush=True)
def select(split='dev'):
    p=OUT/split/'corpus/selection.json'
    if p.exists(): return read(p)
    inv=read(ROOT/'outputs/roadscapes_hf_200_e2e/dataset_inventory.json')
    old=read(ROOT/('outputs/driveevidence/dev/corpus/selection.json' if split=='dev' else 'outputs/roadscapes_hf_200_e2e/selection.json')) if split=='dev' else None
    excluded={x['path'] for x in old['files']} if old else {x['hf_path'] for x in read(ROOT/'data/evaluation/roadscapes_hf_200/timeline.json')}
    groups={}
    for f in inv['files']:
        if f['path'].startswith('train/' if split=='dev' else 'test/') and f['path'].endswith('.mp4') and f['path'] not in excluded:
            groups.setdefault(f['path'].split('/')[1],[]).append(f)
    rng=random.Random(20261010 if split=='dev' else 20261011); names=sorted(groups); rng.shuffle(names)
    for n in names: groups[n].sort(key=lambda x:x['path']); rng.shuffle(groups[n])
    files=[]
    while len(files)<150:
        for n in names:
            if groups[n] and len(files)<150: files.append(groups[n].pop())
    value={'dataset':inv['dataset'],'revision':inv['hf_revision'],'split':'train' if split=='dev' else 'test','files':files,'sequence_folds':{n:i%3 for i,n in enumerate(names)} if split=='dev' else {},'selection_sha256':identity(files)}
    write_json(p,value); return value
def acquire(split='dev'):
    selection=select(split)
    import requests
    answer=requests.get('https://dns.google/resolve',params={'name':'us.aws.cdn.hf.co','type':'A','edns_client_subnet':'8.8.8.8/24'},timeout=20).json()
    addresses=[a['data'] for a in answer.get('Answer',[]) if a['type']==1]
    if not addresses: raise RuntimeError('No dataset CDN DNS addresses')
    def one(item):
        p=OUT/split/'corpus/raw'/item['path']; p.parent.mkdir(parents=True,exist_ok=True); expected=item['lfs']['sha256']
        if not p.is_file() or digest(p)!=expected:
            url=f"https://huggingface.co/datasets/{selection['dataset']}/resolve/{selection['revision']}/{item['path']}"
            tmp=p.with_suffix('.download')
            for attempt in range(3):
                ip=addresses[(int(expected[:8],16)+attempt)%len(addresses)]
                r=subprocess.run(['curl.exe','--fail','--location','--connect-timeout','10','--max-time','120','--resolve','us.aws.cdn.hf.co:443:'+ip,'--silent','--show-error','--output',str(tmp),url],capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW)
                if r.returncode==0 and digest(tmp)==expected: tmp.replace(p); break
            else: raise RuntimeError('Dataset acquisition failed '+item['path'])
        return {'source_clip_id':item['path'],'source_path':str(p),'source_sha256':expected,'sequence':item['path'].split('/')[1],'day_night':'Day' if 'Sequence_Day_' in item['path'] else 'Night'}
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool: rows=list(pool.map(one,selection['files']))
    for i,r in enumerate(rows):
        r.update(row=i,source_row=i,clip_id=r['source_clip_id'],segment_id=r['source_clip_id'],start_sec=i*5.,end_sec=(i+1)*5.,source_start_sec=0.,source_end_sec=5.,split=selection['split'],fold=selection['sequence_folds'].get(r['sequence']))
        r['window_sha256']=identity(r)
    write_json(OUT/split/'corpus/manifest.json',{'clips':rows,'selection_sha256':selection['selection_sha256'],'corpus_sha256':identity(rows),'label_method':'Source-frame visual inspection only; no detector or retrieval outputs'})
    sheets(split,rows)
def sheets(split,rows):
    import cv2
    from PIL import Image, ImageDraw
    base=OUT/split/'source_review'; base.mkdir(parents=True,exist_ok=True)
    for start in range(0,len(rows),5):
        p=base/f'sheet_{start:03d}.jpg'
        if p.exists(): continue
        canvas=Image.new('RGB',(1280,5*210),'white'); draw=ImageDraw.Draw(canvas)
        for line,r in enumerate(rows[start:start+5]):
            cap=cv2.VideoCapture(r['source_path']); fps=cap.get(cv2.CAP_PROP_FPS)
            for k,t in enumerate((0.,1.2,2.4,3.6,4.8)):
                cap.set(cv2.CAP_PROP_POS_FRAMES,round(t*fps)); ok,f=cap.read()
                if not ok: raise ValueError('Source review decode failed')
                im=Image.fromarray(cv2.cvtColor(f,cv2.COLOR_BGR2RGB)); im.thumbnail((256,178)); canvas.paste(im,(k*256,line*210+30))
            draw.text((4,line*210+5),f"row {r['row']:03d} {r['source_clip_id']} fold={r['fold']}",fill='black'); cap.release()
        canvas.save(p,quality=92)
    write_json(base/'receipt.json',{'sheets':{p.name:digest(p) for p in base.glob('sheet_*.jpg')},'timestamps':[0,1.2,2.4,3.6,4.8],'no_model_outputs':True})
if __name__=='__main__':
    if '--verify' in sys.argv: preservation(True)
    else: preservation(); acquire('heldout' if '--heldout' in sys.argv else 'dev')
