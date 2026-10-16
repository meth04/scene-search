"""Frozen EG2 candidates with fail-safe, subject-bound consensus promotion."""
from __future__ import annotations
import json, threading, time
from pathlib import Path
import numpy as np
from .embeddinggemma2 import digest, identity, normalized
from .driveevidence_v2_planner import QueryPlanner, validate
FORMAT='driveevidence-v2'
CLASSES=('person','bicycle','car','motorcycle','bus','truck','traffic light','stop sign')
DEFAULT={'depth':10,'high_margin':.02,'medium_margin':.01,'swap_margin':.015,'extreme_margin':.003,
 'promotion_window':5,'roi_cap':12,'roi_visible':.24,'roi_area':.002,'roi_semantic':.65,'roi_delta':.025,
 'motion_delta':.08,'motion_strength':.65,'consensus':True,'gate':True,'routing':True,'roi':True,'motion':True,'spatial':True,'cap':True,'roi_text':False}
def confidence(scores,config):
    s=np.asarray(scores,dtype=np.float64)
    if s.ndim!=1 or len(s)<5 or not np.isfinite(s).all() or (np.diff(s)>1e-7).any(): return {'state':'INVALID'}
    out={f's{k}':float(s[k-1]) for k in (1,2,3,5)}
    out.update(margin12=float(s[0]-s[1]),margin13=float(s[0]-s[2]),margin15=float(s[0]-s[4]),std=float(s[:config['depth']].std()))
    out['state']='HIGH' if out['margin12']>=config['high_margin'] else 'MEDIUM' if out['margin12']>=config['medium_margin'] else 'LOW'
    return out
def classes_for(plan):
    classes=set()
    for e in plan['entities']:
        if e['type']=='vehicle': classes.update(('car','bus','truck','motorcycle'))
        elif e['type'] in CLASSES: classes.add(e['type'])
    return classes
def select_views(objects,cap=12):
    result=[]
    eligible=[o for o in objects if o['class'] in CLASSES and o['visible_ratio']>=.12 and o['duration']>=.4 and o['confidence']>=.4 and o['mean_area']>=.001]
    eligible.sort(key=lambda o:(-o['visible_ratio']*o['confidence']*min(o['mean_area']*20,1),o['slot']))
    # Start/best-visible/end remain actual observations of the SAME track.
    for o in eligible:
        obs=o['observations']; best=max(obs,key=lambda x:x['confidence']*x['xywh'][2]*x['xywh'][3])
        seen=set()
        for role,v in [('best-visible',best),('start',obs[0]),('end',obs[-1])]:
            if v['frame_position'] in seen: continue
            seen.add(v['frame_position']); result.append({'track_id':o['slot'],'class':o['class'],'visible_ratio':o['visible_ratio'],'duration':o['duration'],'crop_role':role,**v})
            if len(result)>=cap: return result
    return result
def track_feature(track):
    obs=track['observations']; t=np.array([o['timestamp'] for o in obs]); a=np.array([o['xywh'] for o in obs],float)
    if len(obs)<3 or not np.isfinite(a).all() or not np.isfinite(t).all() or (np.diff(t)<=0).any(): return None
    speed=np.linalg.norm(np.diff(a[:,:2],axis=0),axis=1)/np.diff(t)
    return {'track_id':track['slot'],'class':track['class'],'duration':float(t[-1]-t[0]),'visible_ratio':track['visible_ratio'],
      'start':a[0].tolist(),'mid':a[len(a)//2].tolist(),'end':a[-1].tolist(),'delta_x':float(a[-1,0]-a[0,0]),'delta_y':float(a[-1,1]-a[0,1]),
      'delta_area':float(a[-1,2]*a[-1,3]-a[0,2]*a[0,3]),'area_ratio':float(a[-1,2]*a[-1,3]/max(a[0,2]*a[0,3],1e-6)),
      'mean_speed':float(speed.mean()),'max_speed':float(speed.max()),'center_x':float(np.median(a[:,0])),'confidence':track['confidence']}
def motion_support(f,plan,config):
    if not plan['requires_motion'] or f is None or f['duration']<.6 or f['visible_ratio']<config['roi_visible']: return None
    dx=f['delta_x']; direction=plan['motion_direction']; action=plan['action']
    if direction=='left_to_right': value=dx if f['start'][0]<.5 and f['end'][0]>.5 else 0
    elif direction=='right_to_left': value=-dx if f['start'][0]>.5 and f['end'][0]<.5 else 0
    elif direction=='toward' or action=='approaching': value=max(0,f['area_ratio']-1)*.1
    elif direction=='away' or action=='moving_away': value=max(0,1-f['area_ratio'])*.2
    elif action=='stationary': value=max(0,.08-abs(dx)-abs(f['delta_y']))
    elif action in ('crossing','passing','overtaking','moving') or direction=='lateral': value=abs(dx)
    else: return None
    return min(1.,value/max(config['motion_delta'],1e-5))*f['visible_ratio']*f['confidence']
def clip_evidence(loaded,row,vector,plan,config):
    classes=classes_for(plan) if config['routing'] else set(CLASSES)
    if not classes: return []
    if plan['query_type']=='scene_context': return []
    metadata=loaded['roi_meta']; off=loaded['roi_offsets']; trackmap=loaded['features'][row]; bytrack={}
    for k in range(int(off[row]),int(off[row+1])):
        r=metadata[k]
        if r['class'] not in classes or r['visible_ratio']<config['roi_visible'] or r['bbox_area']<config['roi_area']: continue
        bytrack.setdefault(r['track_id'],[]).append((k,r))
    for tid,f in trackmap.items():
        if f is not None and f['class'] in classes and f['visible_ratio']>=config['roi_visible']: bytrack.setdefault(tid,[])
    result=[]
    for tid,views in bytrack.items():
        f=trackmap.get(tid); ev={'track_id':tid,'families':{},'known':{}}
        use_roi=config['roi'] and (plan['requires_object_detail'] or plan['requires_motion'] or plan['requires_spatial_relation'] or plan['requires_infrastructure_detail'])
        if use_roi and views:
            similarities=[float(loaded['roi_eg2'][k]@vector) for k,r in views[:config['roi_cap']]]
            sims=sorted(similarities,reverse=True); ev['families']['roi']=float(np.mean(sims[:2])); ev['known']['roi']=True
            # Consistency supports ROI reliability; it is not counted as an independent semantic family.
            ev['roi_consistency']=min(similarities) if len(similarities)>=2 else None
            ev['roi_reliable']=len(similarities)>=2 and min(similarities)>=ev['families']['roi']-.10
        if config['motion'] and plan['requires_motion']:
            m=motion_support(f,plan,config)
            if m is not None: ev['families']['motion']=m; ev['known']['motion']=True
        # Image-plane x supports an ego-relative side only. Entity-to-entity relations
        # require a second bound actor, which this bounded planner does not certify.
        if config.get('spatial',True) and plan['requires_spatial_relation'] and plan['spatial']['reference']=='ego' and f is not None:
            side=plan['spatial']['side']; x=f['center_x']
            if side in ('left','right','center'):
                ev['families']['spatial']=min(1,max(0,(.5-x)*4 if side=='left' else (x-.5)*4 if side=='right' else 1-abs(x-.5)*4))*f['visible_ratio']
                ev['known']['spatial']=True
        if ev['families'] and all(np.isfinite(v) for v in ev['families'].values()): result.append(ev)
    return result
def promote(scores,evidence,plan,config):
    scores=np.asarray(scores); original=np.argsort(-scores,kind='stable'); order=original.copy(); stats=confidence(scores[original],config)
    diagnostic={'confidence':stats,'activated':False,'swap':False,'promotions':[],'enabled_evidence':[],'fallback':None}
    if stats['state']=='INVALID': diagnostic['fallback']='invalid_confidence'; return order,diagnostic
    try: validate(plan)
    except (TypeError,ValueError,KeyError): diagnostic['fallback']='invalid_plan'; return order,diagnostic
    if config['gate'] and stats['state']=='HIGH': return order,diagnostic
    if plan['query_type']=='scene_context' or not classes_for(plan): return order,diagnostic
    diagnostic['activated']=True
    diagnostic['enabled_evidence']=['roi'] if config['roi'] and plan['requires_object_detail'] else []
    if config['motion'] and plan['requires_motion']: diagnostic['enabled_evidence'].append('motion')
    if plan['requires_spatial_relation']: diagnostic['enabled_evidence'].append('spatial')
    window=min(config['depth'],config['promotion_window'] if config['cap'] else config['depth'],len(order))
    incumbent=int(order[0]); inc=evidence.get(incumbent,[])
    if not inc: diagnostic['fallback']='missing_incumbent_evidence'; return order,diagnostic
    for pos in range(1,window):
        challenger=int(original[pos]); gap=float(scores[incumbent]-scores[challenger])
        limit=config['swap_margin']*(.5 if config['gate'] and stats['state']=='MEDIUM' else 1)
        if gap>limit: continue
        accepted=None
        for ev in evidence.get(challenger,[]):
            agree=[]; strong=[]
            for family,value in ev['families'].items():
                # Unknown is neither opposition nor a zero-valued measurement.
                incumbent_values=[x['families'][family] for x in inc if family in x['families']]
                if not incumbent_values: continue
                old=max(incumbent_values); threshold=config['roi_delta'] if family=='roi' else .2
                floor=config['roi_semantic'] if family=='roi' else config['motion_strength']
                if value>=floor and value-old>=threshold:
                    agree.append(family)
                    if value>=floor+.05 and value-old>=threshold*2 and (family!='roi' or ev.get('roi_reliable',False)): strong.append(family)
            enough=(len(agree)>=2 or (len(strong)>=1 and gap<=config['extreme_margin'])) if config['consensus'] else bool(agree)
            if enough: accepted={'track_id':ev['track_id'],'agree':agree,'strong':strong}; break
        if accepted:
            # At most one stable promotion. All other relative order and the candidate set stay unchanged.
            order=np.concatenate(([challenger],original[:pos],original[pos+1:]))
            diagnostic.update(swap=True,promotions=[{'from_rank':pos+1,'to_rank':1,'candidate_row':challenger,'incumbent_row':incumbent,'gap':gap,**accepted}]); break
    return order,diagnostic
def roi_query(plan):
    """Object semantic text comes ONLY from validated planner fields."""
    try:validate(plan)
    except (TypeError,KeyError,ValueError):return None
    if not plan['entities']:return None
    e=plan['entities'][0]
    return 'A '+' '.join(v for v in (e['color'],e['subtype'],e['type']) if v is not None)
def rank(loaded,vector,plan,config,roi_vector=None):
    try: validate(plan)
    except (ValueError,TypeError,KeyError): plan=None
    start=time.perf_counter(); scores=loaded['global_eg2']@vector; similarity=time.perf_counter()-start
    base=np.argsort(-scores,kind='stable'); start=time.perf_counter(); c=confidence(scores[base],config); conf=time.perf_counter()-start
    start=time.perf_counter(); ev={}
    if plan is not None and c['state']!='INVALID' and not (config['gate'] and c['state']=='HIGH'):
        try:
            if not config.get('roi_text') or roi_vector is not None:
                for j in base[:config['depth']]: ev[int(j)]=clip_evidence(loaded,int(j),roi_vector if config.get('roi_text') else vector,plan,config)
        except (KeyError,IndexError,ValueError,TypeError): ev={}
    lookup=time.perf_counter()-start; start=time.perf_counter(); order,d=promote(scores,ev,plan,config)
    d['timing']={'similarity':similarity,'confidence':conf,'evidence_lookup':lookup,'rerank':time.perf_counter()-start}
    return order,scores,d
def load_index(path):
    path=Path(path); meta=json.loads((path/'index_meta.json').read_text()); rows=[json.loads(x) for x in (path/'manifest.jsonl').read_text().splitlines()]
    if meta['format']!=FORMAT or identity(meta['rerank'])!=meta['rerank_sha256']: raise ValueError('V2_SCHEMA_MISMATCH')
    from .embeddinggemma2_index import FROZEN_CONFIG
    if meta['config']!=json.loads(FROZEN_CONFIG.read_text())['config']: raise ValueError('EG2_CONFIG_CHANGED')
    damaged=[]
    for n,h in meta['hashes'].items():
        if not (path/n).is_file() or digest(path/n)!=h:
            if n in ('manifest.json','manifest.jsonl','global_eg2.npy','provenance.json'): raise ValueError('V2_INDEX_HASH_MISMATCH '+n)
            damaged.append(n)
    if damaged:
        global_vectors=np.load(path/'global_eg2.npy',mmap_mode='r',allow_pickle=False)
        if global_vectors.shape!=(len(rows),768) or not np.isfinite(global_vectors).all() or not np.allclose(np.linalg.norm(global_vectors,axis=1),1,atol=1e-4): raise ValueError('INVALID_GLOBAL_VECTOR')
        for r in rows:
            if digest(r['source_path'])!=r['source_sha256']: raise ValueError('SOURCE_IDENTITY')
        return {'meta':meta,'rows':rows,'global_eg2':global_vectors,'roi_eg2':np.empty((0,768),np.float32),'roi_offsets':np.zeros(len(rows)+1,np.int64),'roi_meta':[],'features':[{} for _ in rows],'degraded_evidence':damaged}
    a={k:np.load(path/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('global_eg2','roi_eg2','roi_offsets')}
    for k in ('global_eg2','roi_eg2'):
        if a[k].ndim!=2 or a[k].shape[1]!=768 or a[k].dtype!=np.float32 or not np.isfinite(a[k]).all() or not np.allclose(np.linalg.norm(a[k],axis=1),1,atol=1e-4): raise ValueError('INVALID_VECTOR')
    from .driveevidence import offset_validate
    offset_validate(a['roi_offsets'],len(a['roi_eg2']),len(rows))
    roi=json.loads((path/'roi_meta.json').read_text()); objects=json.loads((path/'track_records.json').read_text())
    if len(roi)!=len(a['roi_eg2']) or len(objects)!=len(rows) or len(a['global_eg2'])!=len(rows): raise ValueError('V2_ALIGNMENT')
    features=[]
    for j,r in enumerate(rows):
        if r['row']!=j or digest(r['source_path'])!=r['source_sha256']: raise ValueError('SOURCE_IDENTITY')
        features.append({o['slot']:track_feature(o) for o in objects[j]})
        for k in range(int(a['roi_offsets'][j]),int(a['roi_offsets'][j+1])):
            m=roi[k]
            if m['clip_id']!=r['clip_id'] or m['embedding_row']!=k or m['track_id'] not in features[-1]: raise ValueError('TRACK_ROI_IDENTITY')
            track=next(o for o in objects[j] if o['slot']==m['track_id'])
            if not any(o['frame_position']==m['frame_position'] and o['bbox']==m['bbox'] and o['timestamp']==m['timestamp'] for o in track['observations']): raise ValueError('FABRICATED_OBSERVATION')
    return {'meta':meta,'rows':rows,**a,'roi_meta':roi,'features':features}
class DriveEvidenceV2SearchEngine:
    def __init__(self,index_path,worker_factory=None,planner=None):
        from .driveevidence_v2_worker import DriveEvidenceV2Worker
        self.index_path=Path(index_path); self.worker_factory=worker_factory or DriveEvidenceV2Worker; self.planner=planner or QueryPlanner()
        self.lock=threading.RLock(); self.loaded=self.worker=None; self.load_count=0
    def load(self):
        if self.loaded is None: self.loaded=load_index(self.index_path); self.load_count+=1
        return self
    def search_table(self,query,ranker='driveevidence_v2',decision_provider=None,**unused):
        if ranker!='driveevidence_v2' or decision_provider not in (None,'none','laya'): raise ValueError('V2_MODE_REQUIRED')
        with self.lock:
            self.load(); start=time.perf_counter()
            if self.planner.cached(query) is None and self.worker:
                self.worker.close(); self.worker=None
            plan,pm=self.planner.plan(query); self.planner.close()
            if self.worker is None: self.worker=self.worker_factory(self.loaded['meta']['config']); self.worker.load()
            vector,em=self.worker.encode_text(query); rv=None; roi_seconds=0.
            rq=roi_query(plan)
            if self.loaded['meta']['rerank'].get('roi_text') and rq:
                # Encode ROI subject text only if the global confidence gate permits evidence.
                global_scores=self.loaded['global_eg2']@vector; state=confidence(np.sort(global_scores)[::-1],self.loaded['meta']['rerank'])['state']
                if state not in ('HIGH','INVALID'):
                    rv,rm=self.worker.encode_text(rq); roi_seconds=rm['seconds']
            order,scores,d=rank(self.loaded,vector,plan,self.loaded['meta']['rerank'],rv)
            t=time.perf_counter(); candidates=[]
            for i,j in enumerate(order,1):
                r=self.loaded['rows'][j]; candidates.append({**{k:r[k] for k in ('clip_id','source_row','source_clip_id','segment_id','start_sec','end_sec','day_night')},'row':int(j),'rank':i,'ranker':ranker,'score':float(scores[j]),'final_score':float(scores[j])})
            return {'query':query,'format':FORMAT+'-candidates','index':str(self.index_path),'candidates':candidates,'decision':{'state':'UNCALIBRATED','provider':'laya_query_planner'},'retrieval':{'planner':pm,'query_plan':plan,'diagnostic':d,'text_encode_seconds':em['seconds']+roi_seconds,'global_text_encode_seconds':em['seconds'],'roi_text_encode_seconds':roi_seconds,'serialization_seconds':time.perf_counter()-t,'warm_seconds':time.perf_counter()-start,'Qwen_calls':0,'candidate_level_Laya_calls':0,'engine_load_count':self.load_count}}
    def close(self):
        if self.worker: self.worker.close()
        self.planner.close(); self.loaded=self.worker=None
