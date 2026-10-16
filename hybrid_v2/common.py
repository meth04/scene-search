
from __future__ import annotations
import base64, json, math, random, re, hashlib
from pathlib import Path
import cv2
import numpy as np
import torch

def read_jsonl(path):
    out=[]
    with Path(path).open("r",encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def append_jsonl(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a",encoding="utf-8") as f:
        f.write(json.dumps(obj,ensure_ascii=False)+"\n")

def write_jsonl(path,rows):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r,ensure_ascii=False)+"\n")

def write_json(path,obj):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding="utf-8")

def torch_load(path):
    try:
        return torch.load(path,map_location="cpu",weights_only=False)
    except TypeError:
        return torch.load(path,map_location="cpu")

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def choose_device(name="auto"):
    if name=="cpu": return torch.device("cpu")
    if name=="cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def l2norm(x,axis=-1,eps=1e-8):
    x=np.asarray(x,dtype=np.float32)
    return x/np.maximum(np.linalg.norm(x,axis=axis,keepdims=True),eps)

def stable_hash(s):
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]

def clean_text(s):
    return re.sub(r"\s+"," ",str(s or "")).strip()

def token_set(s):
    return set(re.findall(r"[a-z0-9]+", clean_text(s).lower()))

def jaccard(a,b):
    A=token_set(a); B=token_set(b)
    if not A and not B: return 0.0
    return len(A&B)/max(1,len(A|B))

def cache_meta(stage2a_cache_dir):
    p=Path(stage2a_cache_dir)/"cache_meta.json"
    if not p.exists(): raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))

def video_path_for(record,videos_dir):
    p=Path(videos_dir)/f"{record['video_id']}.mp4"
    if not p.exists():
        raise FileNotFoundError(f"Video not found for clip {record.get('clip_id')}: {p}")
    return p

def chronological_frame_ids(start_frame,end_frame,n=6):
    if not isinstance(n,int) or isinstance(n,bool) or n<2 or end_frame-start_frame<n:
        raise ValueError("Caption frames must be at least two distinct chronological samples")
    return np.linspace(start_frame,end_frame-1,num=n,dtype=np.int64)

def sheet_from_frames(frames,tile_w=256,tile_h=144):
    if not frames or tile_w<=0 or tile_h<=0:
        raise ValueError("Nonempty frames and positive caption tile dimensions required")
    n=len(frames)
    cols=3 if n==6 else 4 if n==8 else int(math.ceil(n/2))
    tiles=[]
    for i,fr in enumerate(frames):
        tile=cv2.resize(fr,(tile_w,tile_h),interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile,(0,0),(min(82,tile_w-1),min(26,tile_h-1)),(0,0,0),-1)
        cv2.putText(tile,f"t{i+1}/{n}",(6,19),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles)%cols: tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[r:r+cols],axis=1) for r in range(0,len(tiles),cols)],axis=0)

def sample_clip_frames(record,videos_dir,n=6):
    video=video_path_for(record,videos_dir)
    sf=int(record["start_frame"]); ef=int(record["end_frame"])
    ef=max(sf+1,ef)
    ids=chronological_frame_ids(sf,ef,n)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {video}")
    frames=[]
    last=None
    for idx in ids.tolist():
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
        ok,frame=cap.read()
        if ok and frame is not None:
            last=frame
        if last is None:
            cap.release()
            raise RuntimeError(f"Cannot decode frame {idx} from {video}")
        frames.append(last.copy())
    cap.release()
    return frames

def contact_sheet(record,videos_dir,n=6,tile_w=320,tile_h=180):
    frames=sample_clip_frames(record,videos_dir,n=n)
    return sheet_from_frames(frames,tile_w,tile_h)

def jpeg_data_url(img,quality=86):
    ok,buf=cv2.imencode(".jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY),int(quality)])
    if not ok: raise RuntimeError("JPEG encode failed")
    return "data:image/jpeg;base64,"+base64.b64encode(buf.tobytes()).decode("ascii")

def extract_json(text):
    s=str(text).strip()
    # Prefer fenced JSON.
    m=re.search(r"```(?:json)?\s*(.*?)\s*```",s,re.S|re.I)
    if m: s=m.group(1).strip()
    starts=[i for i,c in enumerate(s) if c in "[{"]
    for start in starts:
        for end in range(len(s),start,-1):
            if s[end-1] not in "]}": continue
            try: return json.loads(s[start:end])
            except Exception: pass
    raise ValueError(f"No valid JSON found in response: {text[:500]}")

def zscore1d(x):
    x=np.asarray(x,dtype=np.float32)
    return (x-x.mean())/(x.std(ddof=0)+1e-6)

def dcg(gains,k):
    g=np.asarray(gains,dtype=np.float64)[:k]
    if len(g)==0: return 0.0
    discounts=1.0/np.log2(np.arange(2,len(g)+2))
    return float(np.sum(g*discounts))

def metrics_from_group_scores(scores,labels,mask=None,k_values=(1,5,10)):
    scores=np.asarray(scores); labels=np.asarray(labels)
    if mask is None: mask=np.ones_like(labels,dtype=bool)
    rows=[]
    for i in range(len(scores)):
        valid=np.asarray(mask[i],dtype=bool)
        s=scores[i][valid]; y=labels[i][valid]
        if len(s)==0: continue
        order=np.argsort(-s,kind="stable")
        yy=y[order]
        rel=yy>=.75
        rank=int(np.where(rel)[0][0]+1) if rel.any() else 10**9
        gains=np.where(yy>=.75,3.0,np.where(yy>=.25,1.0,0.0))
        ideal=np.sort(gains)[::-1]
        nd=dcg(gains,10)/max(dcg(ideal,10),1e-12)
        rows.append((rank,nd))
    if not rows: return {}
    ranks=np.array([x[0] for x in rows],dtype=np.int64)
    out={f"R@{k}":float(np.mean(ranks<=k)) for k in k_values}
    out["MRR"]=float(np.mean(np.where(ranks<10**8,1.0/ranks,0.0)))
    out["nDCG@10"]=float(np.mean([x[1] for x in rows]))
    out["queries"]=len(rows)
    return out
