"""Independent dense frozen YOLO/ByteTrack extraction and frozen EG2 ROI index."""
from __future__ import annotations
import sys,os,json,time,gc
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
import numpy as np
from prepare_driveevidence_v3 import OUT,read
from hybrid_v2.embeddinggemma2 import digest,identity,frame_indices
from hybrid_v2.embeddinggemma2_index import write_json
from hybrid_v2.driveevidence_v3 import CLASSES,FORMAT,DEFAULT,select_views,track_feature,load_index
from hybrid_v2.driveevidence_v3_worker import DriveEvidenceV3Worker
PROTOCOL={'fps':10,'imgsz':640,'conf':.25,'iou':.5,'dtype':'float32','tracker':'bytetrack.yaml','roi_cap':16,'views':3,'crop_padding':.1}
def decode(row,cache):
    import cv2
    p=cache/'frames.npy';receipt=cache/'decode.json'
    if receipt.exists():
        d=read(receipt)
        if d['source_sha256']!=row['source_sha256'] or d['protocol_sha256']!=identity(PROTOCOL) or digest(p)!=d['frames_sha256']:raise ValueError('DENSE_DECODE_IDENTITY')
        return d
    start=time.perf_counter();cap=cv2.VideoCapture(row['source_path'])
    try:
        fps=cap.get(cv2.CAP_PROP_FPS);total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps<=0 or abs(total/fps-5)>.15:raise ValueError('FIVE_SECOND_SOURCE_REQUIRED')
        track_ids=sorted(set(np.rint(np.arange(0,total/fps,.1)*fps).astype(int).clip(0,total-1).tolist()));global_ids=frame_indices(total,5)
        ids=sorted(set(track_ids+global_ids));wanted=set(ids);frames=[]
        for i in range(total):
            ok,f=cap.read()
            if not ok:raise ValueError('SOURCE_DECODE_FAILED')
            if i in wanted:frames.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
        a=np.stack(frames);cache.mkdir(parents=True,exist_ok=True);np.save(p,a,allow_pickle=False);position={j:k for k,j in enumerate(ids)}
        d={'source_sha256':row['source_sha256'],'protocol_sha256':identity(PROTOCOL),'frames_sha256':digest(p),'timestamps':[i/fps for i in ids],
          'track_positions':[position[i] for i in track_ids],'global_positions':[position[i] for i in global_ids],'source_fps':fps,'width':a.shape[2],'height':a.shape[1],'decode_seconds':time.perf_counter()-start}
        write_json(receipt,d);return d
    finally:cap.release()
def detector(rows,base):
    import torch
    from ultralytics import YOLO
    from stage3a.stage3a_motion import Obs,aggregate_window_tracks
    if torch.cuda.device_count()!=1 or 'GTX 1650' not in torch.cuda.get_device_name(0):raise ValueError('GTX1650_ONLY')
    torch.set_num_threads(4);checkpoint=ROOT/'data/model-cache/yolo11n.pt';h=digest(checkpoint);model=YOLO(str(checkpoint));model.model.requires_grad_(False)
    for j,row in enumerate(rows):
        c=base/f'{j:06d}';d=decode(row,c)
        if (c/'objects.json').exists():
            o=read(c/'objects.json')
            if o['source_sha256']!=row['source_sha256'] or o['detector_sha256']!=h or o['protocol_sha256']!=identity(PROTOCOL):raise ValueError('DENSE_TRACK_IDENTITY')
            continue
        start=time.perf_counter();frames=np.load(c/'frames.npy',mmap_mode='r');byid=defaultdict(list);obs=[]
        if getattr(model,'predictor',None) and getattr(model.predictor,'trackers',None):
            for tracker in model.predictor.trackers:tracker.reset()
        with torch.inference_mode():
            for sample,pos in enumerate(d['track_positions']):
                result=model.track(np.ascontiguousarray(frames[pos,:,:,::-1]),persist=True,tracker='bytetrack.yaml',imgsz=640,conf=.25,iou=.5,device=0,half=False,verbose=False)[0];boxes=result.boxes
                if boxes is None or boxes.id is None:continue
                for k in range(len(boxes)):
                    cls=result.names[int(boxes.cls[k])]
                    if cls not in CLASSES:continue
                    tid=int(boxes.id[k]);cf=float(boxes.conf[k]);xywh=boxes.xywhn[k].cpu().numpy().tolist();box=boxes.xyxy[k].cpu().numpy().astype(int).tolist()
                    byid[tid].append({'class':cls,'confidence':cf,'frame_position':pos,'sample':sample,'timestamp':d['timestamps'][pos],'bbox':box,'xywh':xywh})
                    if cls in CLASSES[:6]:obs.append(Obs(sample,tid,cls,cf,*xywh))
        features,mask=aggregate_window_tracks(obs,len(d['track_positions']));np.save(c/'stage3a_geometry.npy',features,allow_pickle=False);np.save(c/'stage3a_mask.npy',mask,allow_pickle=False)
        objects=[]
        for tid,observations in sorted(byid.items()):
            cls=max(CLASSES,key=lambda x:sum(o['class']==x for o in observations))
            objects.append({'slot':tid,'class':cls,'visible_ratio':len(observations)/len(d['track_positions']),'mean_area':float(np.mean([o['xywh'][2]*o['xywh'][3] for o in observations])),
              'duration':observations[-1]['timestamp']-observations[0]['timestamp'],'confidence':float(np.mean([o['confidence'] for o in observations])),'observations':observations,'physical_id_certified':False})
        write_json(c/'objects.json',{'objects':objects,'source_sha256':row['source_sha256'],'detector_sha256':h,'protocol_sha256':identity(PROTOCOL),'seconds':time.perf_counter()-start,'peak_VRAM_bytes':torch.cuda.max_memory_allocated()})
        print('V3 dense track',j+1,len(rows),'observations',sum(len(x['observations']) for x in objects),flush=True)
    del model;gc.collect();torch.cuda.empty_cache()
def build(split='dev'):
    if split!='dev':raise ValueError('NONDEV_BUILD_REQUIRES_FROZEN_GATE_WORKFLOW')
    from freeze_driveevidence_v3_dev import main as verify
    verify();started=time.perf_counter();rows=read(OUT/split/'corpus/manifest.json')['clips'];base=OUT/split/'evidence_cache';target=OUT/split/'index'
    if (target/'index_meta.json').exists():load_index(target);print('Verified V3 index');return
    detector(rows,base);config=read(ROOT/'outputs/embeddinggemma2/embeddinggemma2_frozen_config.json')['config'];worker=DriveEvidenceV3Worker(config,timeout=240)
    globals=np.load(ROOT/'outputs/driveevidence_v2/dev/index/global_eg2.npy',allow_pickle=False);vectors=[];metadata=[];offsets=[0];tracks=[];metrics=[]
    try:
        worker.load()
        for j,row in enumerate(rows):
            c=base/f'{j:06d}';d=read(c/'decode.json');objects=read(c/'objects.json')['objects'];tracks.append(objects);receipt=c/'eg2_complete.json'
            if receipt.exists():
                r=read(receipt)
                if r['source_sha256']!=row['source_sha256'] or r['config_sha256']!=identity(config):raise ValueError('ROI_IDENTITY')
                for n,h in r['hashes'].items():
                    if digest(c/n)!=h:raise ValueError('ROI_CACHE_HASH')
            else:
                vv=[];mm=[];cost=[]
                for view in select_views(objects):
                    x1,y1,x2,y2=view['bbox'];pad=max(2,round(.1*max(x2-x1,y2-y1)));box=[max(0,x1-pad),max(0,y1-pad),min(d['width'],x2+pad),min(d['height'],y2+pad)]
                    if box[2]-box[0]<8 or box[3]-box[1]<8:continue
                    v,m=worker.encode_frames(c/'frames.npy',[view['frame_position']],box);vv.append(v);cost.append(m)
                    mm.append({**view,'clip_id':row['clip_id'],'bbox_area':view['xywh'][2]*view['xywh'][3],'frame_idx':round(view['timestamp']*d['source_fps']),'crop_bbox':box,'crop_padding':.1})
                np.save(c/'roi.npy',np.stack(vv) if vv else np.empty((0,768),np.float32),allow_pickle=False);write_json(c/'roi_meta.json',mm)
                write_json(receipt,{'source_sha256':row['source_sha256'],'config_sha256':identity(config),'metrics':cost,'hashes':{n:digest(c/n) for n in ('roi.npy','roi_meta.json')}})
            vectors.extend(np.load(c/'roi.npy'));mm=read(c/'roi_meta.json')
            for m in mm:m['embedding_row']=len(metadata);metadata.append(m)
            offsets.append(len(vectors));metrics.extend(read(receipt)['metrics']);print('V3 EG2 ROI',j+1,len(rows),'vectors',len(mm),flush=True)
    finally:worker.close()
    target.mkdir(parents=True,exist_ok=True)
    for n,a in {'global_eg2':globals,'roi_eg2':np.stack(vectors) if vectors else np.empty((0,768),np.float32),'roi_offsets':np.array(offsets,np.int64)}.items():np.save(target/(n+'.npy'),a,allow_pickle=False)
    write_json(target/'manifest.json',{'clips':rows});write_json(target/'roi_meta.json',metadata);write_json(target/'track_records.json',tracks)
    np.savez(target/'roi_meta.npz',track_id=np.array([m['track_id'] for m in metadata]),timestamp=np.array([m['timestamp'] for m in metadata]),bbox=np.array([m['bbox'] for m in metadata]),embedding_row=np.arange(len(metadata)))
    observations=[];feature_rows=[]
    for j,objects in enumerate(tracks):
        for o in objects:
            observations.extend([[j,o['slot'],v['timestamp'],*v['xywh'],v['confidence']] for v in o['observations']]);f=track_feature(o)
            if f:feature_rows.append([j,o['slot'],f['delta_x'],f['delta_y'],f['delta_area'],f['mean_speed'],f['max_speed'],f['duration'],f['visible_ratio']])
    np.savez(target/'track_observations.npz',observations=np.array(observations,float).reshape(-1,8));np.savez(target/'track_features.npz',features=np.array(feature_rows,float).reshape(-1,9))
    costs={'wall_seconds':time.perf_counter()-started,'clips':len(rows),'ROI_vectors':len(vectors),'tracks':sum(len(x) for x in tracks),'observations':len(observations),'peak_VRAM_bytes':max((m['peak_VRAM_bytes'] for m in metrics),default=0),'OOM_count':0}
    write_json(target/'provenance.json',{'protocol':PROTOCOL,'detector_sha256':digest(ROOT/'data/model-cache/yolo11n.pt'),'Stage3A_checkpoint_sha256':digest(ROOT/'outputs/driveretriever_stage3a/best_composite.pt'),'Stage3A_use':'Original frozen YOLO/ByteTrack plus original geometry aggregate; learned fusion head unused. New 10fps observed boxes, no fabricated old/new track association.','global_reuse':{'path':'outputs/driveevidence_v2/dev/index/global_eg2.npy','sha256':digest(ROOT/'outputs/driveevidence_v2/dev/index/global_eg2.npy'),'source_manifest_sha256':digest(OUT/'dev/corpus/manifest.json')},'Stage1_Stage2B_Stage4B':'Excluded: no validated direct evidence contribution','training_performed':False,'costs':costs})
    write_json(target/'index_meta.json',{'format':FORMAT,'ranker':'driveevidence_v3','num_windows':len(rows),'config':config,'rerank':DEFAULT,'rerank_sha256':identity(DEFAULT),'hashes':{p.name:digest(p) for p in target.iterdir() if p.is_file()},'video':{'path':rows[0]['source_path'],'sha256':rows[0]['source_sha256']},'phases':{'validate':{'finite':len(rows)}}})
    write_json(OUT/split/'performance/build.json',costs);load_index(target)
if __name__=='__main__':build(sys.argv[1] if len(sys.argv)>1 else 'dev')
