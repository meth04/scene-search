"""Frozen multi-evidence retrieval. No neural optimization or query video inference."""
from __future__ import annotations
import json
import re
import threading
import time
from pathlib import Path
import numpy as np
from .embeddinggemma2 import digest, identity, normalized, validate_config
from .intent_parser import parse_query

FORMAT = 'driveevidence-v1'
SEGMENTS = ((0.,2.),(1.5,3.5),(3.,5.))
CLASSES = ('person','bicycle','car','motorcycle','bus','truck')
ROUTING = {
 'scene_context':(1.,.4,.1,0.,.6), 'infrastructure':(1.,.5,.15,0.,.3),
 'vehicle':(1.,.5,1.,.2,.2), 'people':(1.,.7,1.,.2,.1),
 'action_temporal':(1.,1.,.4,1.,.1), 'entity_relation':(1.,.5,.7,.5,.2)}

def route(query):
    slots = parse_query(query)
    text=query.lower()
    action=bool(slots.get('action')) or bool(re.search(r'\b(mov\w*|approach\w*|cross\w*|rides?|walk\w*|overtak\w*|chang\w*|away)\b',text))
    spatial=bool(re.search(r'\b(left|right|ahead|behind|beside)\b',text))
    subject=slots['subject_type']
    category=('action_temporal' if action else 'entity_relation' if spatial and subject not in ('scene','infrastructure')
              else 'people' if subject in ('person','cyclist') else 'vehicle' if subject in ('vehicle','motorcycle')
              else 'infrastructure' if subject=='infrastructure' else 'scene_context')
    classes=[]
    for name in CLASSES:
        if re.search(r'\b'+name+r'(?:s|es)?\b',text): classes.append(name)
    if subject=='person' and 'person' not in classes: classes.append('person')
    if subject=='motorcycle' and 'motorcycle' not in classes: classes.append('motorcycle')
    if subject=='vehicle' and not classes: classes=['car','motorcycle','bus','truck']
    return {'category':category,'classes':classes,'left':bool(re.search(r'\bleft\b',text)),
            'right':bool(re.search(r'\bright\b',text)),'action':action,'slots':slots}

def temporal_indices(timestamps, segments=SEGMENTS):
    t=np.asarray(timestamps)
    result=[np.flatnonzero((t>=a)&(t<b if b<5 else t<=b)) for a,b in segments]
    if any(len(x)<2 for x in result): raise ValueError('Insufficient segment frames')
    return result

def select_rois(objects, cap=8, visible=.12, area=.001, duration=.2, confidence=.35):
    if not 0<=cap<=8: raise ValueError('ROI cap exceeds budget')
    eligible=[o for o in objects if o['class'] in CLASSES and o['visible_ratio']>=visible
              and o['mean_area']>=area and o['duration']>=duration and o['confidence']>=confidence]
    return sorted(eligible,key=lambda o:(-(o['visible_ratio']+o['confidence']+.5*min(o['mean_area']*10,1)),o['slot']))[:cap]

def trajectory_evidence(tracks, mask):
    tracks=np.asarray(tracks,np.float32); mask=np.asarray(mask,bool)
    if tracks.shape!=(len(mask),29) or not np.isfinite(tracks).all(): raise ValueError('Invalid motion geometry')
    f=tracks[mask]
    if not len(f): return {'known':False,'count':0,'person_count':0,'vehicle_count':0,
                           'max_displacement':0.,'mean_displacement':0.,'visible_ratio':0.}
    d=np.linalg.norm(f[:,14:16],axis=1)
    return {'known':True,'count':len(f),'person_count':int((f[:,0]>.5).sum()),
            'vehicle_count':int((f[:,2:6].sum(1)>.5).sum()),'max_displacement':float(d.max()),
            'mean_displacement':float(d.mean()),'visible_ratio':float(f[:,20].mean())}

def motion_match(tracks, mask, routing, text):
    f=np.asarray(tracks)[np.asarray(mask,bool)]
    classes=routing['classes']
    if len(f) and classes: f=f[:, :][f[:,[CLASSES.index(c) for c in classes]].sum(1)>.5]
    if not len(f): return 0.,0.,False
    # Motion is measured in image coordinates, not verified world actions.
    score=np.zeros(len(f),np.float32)
    spatial=np.zeros(len(f),np.float32)
    if routing['left']: spatial += np.clip((.5-f[:,6])*2,0,1)
    if routing['right']: spatial += np.clip((f[:,6]-.5)*2,0,1)
    if re.search(r'cross\w*',text): score+=np.clip(np.abs(f[:,14])*4,0,1)*f[:,27]
    elif re.search(r'approach\w*|oncoming',text): score+=np.clip((f[:,19]-1)/2,0,1)
    elif re.search(r'away|reced\w*',text): score+=np.clip(1-f[:,19],0,1)
    elif re.search(r'overtak\w*|pass\w*',text): score+=np.clip(np.abs(f[:,14])*3+np.abs(f[:,15])*2,0,1)
    elif routing['action']: score+=np.clip(np.linalg.norm(f[:,14:16],axis=1)*3,0,1)
    confidence=np.clip(f[:,20],0,1)
    return float((score*confidence).max()),float((spatial*confidence).max()),True

def offset_validate(offsets,total,n):
    o=np.asarray(offsets)
    if o.shape!=(n+1,) or o.dtype.kind not in 'iu' or o[0]!=0 or o[-1]!=total or (np.diff(o)<0).any():
        raise ValueError('Invalid offsets')

def load_index(path):
    path=Path(path)
    meta=json.loads((path/'index_meta.json').read_text())
    if meta['format']!=FORMAT or identity(meta['fusion'])!=meta['fusion_sha256']: raise ValueError('DriveEvidence schema/config mismatch')
    validate_config(meta['config'])
    from .embeddinggemma2_index import FROZEN_CONFIG
    if FROZEN_CONFIG.exists() and meta['config'] != json.loads(FROZEN_CONFIG.read_text())['config']:
        raise ValueError('DriveEvidence must preserve the frozen EG2 backbone configuration')
    for name,h in meta['hashes'].items():
        if digest(path/name)!=h: raise ValueError('DriveEvidence integrity mismatch: '+name)
    rows=[json.loads(s) for s in (path/'manifest.jsonl').read_text().splitlines()]
    n=len(rows)
    arrays={k:np.load(path/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in
            ('global_eg2','temporal_eg2','temporal_offsets','object_eg2','object_offsets','tracks','track_masks','stage2b_features')}
    for key in ('global_eg2','temporal_eg2','object_eg2'):
        a=arrays[key]
        if a.ndim!=2 or a.shape[1]!=768 or a.dtype!=np.float32 or not np.isfinite(a).all(): raise ValueError('Invalid EG2 array')
        if len(a) and not np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-4): raise ValueError('Unnormalized vectors')
    if arrays['global_eg2'].shape!=(n,768): raise ValueError('Row mismatch')
    offset_validate(arrays['temporal_offsets'],len(arrays['temporal_eg2']),n)
    offset_validate(arrays['object_offsets'],len(arrays['object_eg2']),n)
    if (np.diff(arrays['object_offsets'])>8).any(): raise ValueError('ROI cap')
    if arrays['tracks'].shape!=(n,24,29) or arrays['track_masks'].shape!=(n,24): raise ValueError('Tracks misaligned')
    if arrays['stage2b_features'].shape!=(n,512): raise ValueError('Stage2B mismatch')
    objects=json.loads((path/'object_meta.json').read_text())
    if len(objects)!=len(arrays['object_eg2']): raise ValueError('Object metadata mismatch')
    for key,a in arrays.items():
        if not np.isfinite(a).all(): raise ValueError('Nonfinite evidence: '+key)
    for j,r in enumerate(rows):
        if r['row']!=j or identity({k:v for k,v in r.items() if k!='window_sha256'})!=r['window_sha256']:
            raise ValueError('Window identity mismatch')
        if digest(r['source_path'])!=r['source_sha256']: raise ValueError('Source changed')
    if len({r['clip_id'] for r in rows}) != n or meta['num_windows'] != n:
        raise ValueError('Duplicate clip or manifest count mismatch')
    if not (np.diff(arrays['temporal_offsets']) == 3).all():
        raise ValueError('Expected three temporal segments per clip')
    if arrays['track_masks'].dtype != np.bool_ or not np.allclose(np.linalg.norm(arrays['stage2b_features'],axis=1),1,atol=1e-4):
        raise ValueError('Invalid stage feature normalization/mask')
    for name,h in meta.get('source_hashes',{}).items():
        if digest(name)!=h: raise ValueError('Stage reuse/source changed: '+name)
    if meta['fusion'].get('stage4b',0) or meta['fusion'].get('stage1',0): raise ValueError('Incompatible stage fail-safe')
    return {'meta':meta,'rows':rows,'arrays':arrays,'objects':objects,'ready':True}

def evidence(loaded,vector,query,roi_config=None):
    vector=np.asarray(vector,np.float32)
    if vector.shape!=(768,) or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector),1.,atol=1e-4):
        raise ValueError('Query vector must be finite, normalized native EG2')
    a=loaded['arrays']; n=len(a['global_eg2']); routing=route(query)
    global_scores=a['global_eg2']@vector
    ts=a['temporal_eg2']@vector; os=a['object_eg2']@vector
    temporal=np.zeros((n,3),np.float32); obj=global_scores.copy(); obj_known=np.zeros(n,bool)
    motion=np.zeros(n,np.float32); spatial=motion.copy(); motion_known=np.zeros(n,bool)
    contradiction_known=np.zeros(n,bool)
    for j in range(n):
        b,e=a['temporal_offsets'][j:j+2]
        temporal[j]=ts[b:e]
        b,e=a['object_offsets'][j:j+2]
        policy=roi_config or {}
        eligible=[k for k in range(b,e) if loaded['objects'][k]['visible_ratio']>=policy.get('roi_visible',.12)
                  and loaded['objects'][k]['mean_area']>=policy.get('roi_area',.001)]
        eligible=eligible[:policy.get('roi_cap',8)]
        ix=[k for k in eligible if not routing['classes'] or loaded['objects'][k]['class'] in routing['classes']]
        if ix: obj[j]=os[ix].max(); obj_known[j]=True
        confident=[loaded['objects'][k] for k in ix if loaded['objects'][k]['confidence']>=.7
                   and loaded['objects'][k]['visible_ratio']>=.5 and loaded['objects'][k]['mean_area']>=.01
                   and 'observation' in loaded['objects'][k]]
        if confident and routing['category']=='entity_relation':
            centers=[o['observation']['xywh'][0] for o in confident]
            contradiction_known[j]=(routing['left'] and all(x>.7 for x in centers)) or (routing['right'] and all(x<.3 for x in centers))
        motion[j],spatial[j],motion_known[j]=motion_match(a['tracks'][j],a['track_masks'][j],routing,query.lower())
    # Query-aware Stage2B similarity is between clip features only. EG2 selects three visual anchors.
    anchors=np.argsort(-global_scores,kind='stable')[:3]
    anchor_weights=np.exp((global_scores[anchors]-global_scores[anchors].max())/.02)
    anchor_weights/=anchor_weights.sum()
    support=a['stage2b_features']@(a['stage2b_features'][anchors].T@anchor_weights)
    return {'global':global_scores,'temporal':temporal.max(1),'temporal_mean':temporal.mean(1),'segments':temporal,
            'object':obj,'object_known':obj_known,'motion':motion,'spatial':spatial,'motion_known':motion_known,
            'contradiction_known':contradiction_known,
            'stage2b':support,'routing':routing}

def candidate_union(channels,depth):
    if depth<=0: raise ValueError('Candidate depth must be positive')
    return np.array(sorted({int(j) for scores in channels for j in np.argsort(-scores,kind='stable')[:depth]}),np.int64)

def recall_union_order(channels,depth):
    """Fixed reciprocal-rank aggregation for equal-budget recall diagnostics."""
    ids=candidate_union(channels,depth); score=np.zeros(len(channels[0]),np.float32)
    for channel,values in enumerate(channels):
        order=np.argsort(-values,kind='stable')[:depth]
        score[order]+=(1. if channel==0 else .5)/(60+np.arange(1,len(order)+1))
    return ids[np.lexsort((ids,-score[ids]))]

def rerank(features,config):
    g=features['global']; routed=np.array(ROUTING[features['routing']['category']] if config.get('routing',True) else (1.,1.,1.,1.,1.))
    weights=routed*np.array([1.,config.get('temporal',0.),config.get('object',0.),config.get('motion',0.),config.get('stage2b',0.)])
    t=features['temporal'] if config.get('temporal_policy','max')=='max' else features['temporal_mean']
    scores=g.copy()
    # Add residual evidence in compatible semantic space, conservatively bounded deterministic motion.
    scores+=weights[1]*(t-g)+weights[2]*np.where(features['object_known'],features['object']-g,0)
    scores+=weights[3]*.04*(features['motion']+features['spatial'])
    s=features['stage2b']; s=(s-s.mean())/max(float(s.std()),1e-6)
    scores+=weights[4]*.01*s
    # Unknown evidence is never a contradiction; opposite-side evidence needs positive visibility.
    if config.get('penalty',0) and features['routing']['category']=='entity_relation':
        scores-=config['penalty']*.01*features.get('contradiction_known',np.zeros(len(g),bool))
    channels=[g]
    if config.get('temporal',0): channels.append(t)
    if config.get('object',0): channels.append(np.where(features['object_known'],features['object'],-1.))
    union=candidate_union(channels,config.get('depth',30))
    order=union[np.lexsort((union,-scores[union]))]
    # Preserve global ordering in the recall tail; only the union is reranked.
    tail=np.array([j for j in np.argsort(-g,kind='stable') if j not in set(union)],np.int64)
    return np.concatenate([order,tail]),scores,union

class DriveEvidenceSearchEngine:
    def __init__(self,index_path,worker_factory=None):
        from .driveevidence_worker import DriveEvidenceWorker
        self.index_path=Path(index_path).resolve(); self.worker_factory=worker_factory or DriveEvidenceWorker
        self.loaded=self.worker=None; self.lock=threading.RLock(); self.load_count=self.query_count=0
    def load(self):
        with self.lock:
            if self.loaded is None:
                loaded=load_index(self.index_path)
                worker=self.worker_factory(loaded['meta']['config'])
                try: worker.load()
                except Exception:
                    worker.close(); raise
                self.loaded,self.worker=loaded,worker; self.load_count+=1
            return self
    def search_table(self,query,ranker='driveevidence',decision_provider=None,**unused):
        if ranker!='driveevidence' or decision_provider not in (None,'none'): raise ValueError('DriveEvidence requires its own mode and decision_provider=none')
        if not isinstance(query,str) or not query.strip() or len(query)>1000: raise ValueError('INVALID_QUERY')
        with self.lock:
            self.load(); start=time.perf_counter(); vector,metrics=self.worker.encode_text(query)
            t=time.perf_counter(); f=evidence(self.loaded,vector,query,self.loaded['meta']['fusion']); evidence_seconds=time.perf_counter()-t
            t=time.perf_counter(); order,scores,union=rerank(f,self.loaded['meta']['fusion']); rerank_seconds=time.perf_counter()-t
            t=time.perf_counter(); rows=self.loaded['rows']; candidates=[]
            for rank,j in enumerate(order,1):
                r=rows[j]; candidates.append({**{k:r[k] for k in ('clip_id','source_row','source_clip_id','segment_id','start_sec','end_sec','day_night')},
                   'row':int(j),'rank':rank,'ranker':ranker,'final_score':float(scores[j]),'score':float(scores[j]),
                   'features':{k:float(f[k][j]) for k in ('global','temporal','temporal_mean','object','motion','spatial','stage2b')},
                   'missing_evidence':{'object':not bool(f['object_known'][j]),'motion':not bool(f['motion_known'][j])}})
            self.query_count+=1
            return {'format':FORMAT+'-candidates','query':query,'index':str(self.index_path),'candidates':candidates,
                'decision':{'provider':'none','state':'UNCALIBRATED','threshold':None},
                'retrieval':{'index_windows':len(rows),'candidate_count':len(union),'query_routing':f['routing'],
                   'text_encode_seconds':metrics['seconds'],'evidence_lookup_seconds':evidence_seconds,
                   'reranking_seconds':rerank_seconds,'serialization_seconds':time.perf_counter()-t,
                   'warm_seconds':time.perf_counter()-start,'engine_load_count':self.load_count,
                   'peak_VRAM_bytes':metrics['peak_VRAM_bytes'],'Qwen_calls':0,'external_model_API_calls':0}}
    def close(self):
        with self.lock:
            if self.worker: self.worker.close()
            self.loaded=self.worker=None
