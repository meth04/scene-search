"""Frozen track evidence and deterministic signed, bounded pairwise promotion."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from .embeddinggemma2 import digest,identity,normalized
from .driveevidence_v3_planner import validate,object_queries
FORMAT='driveevidence-v3'
CLASSES=('person','bicycle','car','motorcycle','bus','truck','traffic light','stop sign')
DEFAULT={'depth':10,'promotion_window':5,'high_margin':.025,'medium_margin':.012,'swap_margin':.02,'extreme_margin':.005,
 'roi_delta':.025,'roi_strong':.06,'motion_delta':.15,'motion_strong':.3,'spatial_delta':.15,'spatial_strong':.3,
 'roi_visible':.12,'roi_area':.001,'roi_floor':.30,'roi_cap':16,'views_per_track':3,'object_variant':'B','object_aggregation':'max',
 'roi':True,'motion':True,'spatial':True,'binding':True,'signed':True,'gate':True,'cap':True,'factorized':True,'object_subquery':True,'overwhelming':False,'policy':'ABC'}
def classes_for(plan):
    out=set()
    for e in plan['entities']:
        if e['type']=='vehicle':
            subtype=e.get('subtype')
            if subtype in ('hatchback','sedan','SUV','van','pickup'):out.add('car')
            elif subtype in ('tanker','goods truck'):out.add('truck')
            elif subtype=='scooter':out.add('motorcycle')
            else:out.update(('car','truck','bus','motorcycle'))
        elif e['type'] in CLASSES:out.add(e['type'])
    return out
def select_views(objects,cap=16):
    eligible=[o for o in objects if o['class'] in CLASSES and o['confidence']>=.35 and o['mean_area']>=.0007 and o['duration']>=.2]
    eligible.sort(key=lambda o:(-o['visible_ratio']*o['confidence']*min(o['mean_area']*20,1),o['slot']))
    result=[]
    for o in eligible:
        obs=o['observations'];best=max(obs,key=lambda x:x['confidence']*x['xywh'][2]*x['xywh'][3]);seen=set()
        for role,v in [('best-visible',best),('start',obs[0]),('end',obs[-1])]:
            if v['frame_position'] in seen:continue
            seen.add(v['frame_position']);result.append({'track_id':o['slot'],'class':o['class'],'visible_ratio':o['visible_ratio'],'duration':o['duration'],'crop_role':role,**v})
            if len(result)>=cap:return result
    return result
def track_feature(track,clip_duration=5):
    obs=track['observations'];t=np.array([o['timestamp'] for o in obs]);a=np.array([o['xywh'] for o in obs],float)
    if len(obs)<3 or not np.isfinite(a).all() or not np.isfinite(t).all() or (np.diff(t)<=0).any():return None
    velocity=np.diff(a[:,:2],axis=0)/np.diff(t)[:,None];speed=np.linalg.norm(velocity,axis=1);area=a[:,2]*a[:,3]
    dx=float(a[-1,0]-a[0,0]);dy=float(a[-1,1]-a[0,1]);growth=float(np.log(max(area[-1],1e-6)/max(area[0],1e-6)))
    reliability=min(1,(t[-1]-t[0])/1.2)*min(1,track['visible_ratio']/.4)*track['confidence']
    clip=lambda x:float(np.clip(x,0,1))*reliability
    supports={'left_to_right_evidence':clip(dx/.25),'right_to_left_evidence':clip(-dx/.25),'lateral_evidence':clip(abs(dx)/.25),
      'approach_like_evidence':clip(max(0,growth)/1.5),'away_like_evidence':clip(max(0,-growth)/1.5),
      'cross_center_evidence':clip(abs(dx)/.25) if (a[0,0]-.5)*(a[-1,0]-.5)<0 else 0.,
      'stationary_like_evidence':clip(1-float(np.median(speed))/.05),
      'passing_like_evidence':clip((abs(dx)/.25+max(0,growth)/1.5)/2),
      'appear_evidence':clip(t[0]/1.0),'disappear_evidence':clip((clip_duration-.1-t[-1])/1.0)}
    return {'track_id':track['slot'],'class':track['class'],'start':a[0].tolist(),'mid':a[len(a)//2].tolist(),'end':a[-1].tolist(),
      'start_time':float(t[0]),'end_time':float(t[-1]),'duration':float(t[-1]-t[0]),'visible_ratio':track['visible_ratio'],'confidence':track['confidence'],
      'delta_x':dx,'delta_y':dy,'delta_area':float(area[-1]-area[0]),'area_growth':growth,'mean_velocity':velocity.mean(0).tolist(),
      'mean_speed':float(speed.mean()),'max_speed':float(speed.max()),'center_x':float(np.median(a[:,0])),'center_y':float(np.median(a[:,1])),
      'supports':supports,'coordinate_scope':'normalized image plane; ego motion is not compensated, physical action not certified'}
def motion_support(f,plan):
    if f is None or f['duration']<.6 or not plan['requires_motion']:return None
    keys=[];d=plan['motion_direction'];a=plan['action']
    if d in ('left_to_right','right_to_left','lateral'):keys.append(d+'_evidence')
    elif d=='toward':keys.append('approach_like_evidence')
    elif d=='away':keys.append('away_like_evidence')
    key={'approaching':'approach_like_evidence','moving_away':'away_like_evidence','crossing':'cross_center_evidence','passing':'passing_like_evidence','overtaking':'passing_like_evidence','stationary':'stationary_like_evidence','appearing':'appear_evidence','disappearing':'disappear_evidence','moving':'lateral_evidence'}.get(a)
    if key and key not in keys:keys.append(key)
    return min(f['supports'][k] for k in keys) if keys else None
def spatial_support(f,plan):
    if f is None or not plan['requires_spatial_relation']:return None
    # Entity-to-entity geometry requires a separately identified second entity.
    if plan['spatial']['reference']=='entity':return None
    side=plan['spatial']['side'];x=f['center_x']
    if side=='unknown':return None
    value=(.5-x)*4 if side=='left' else (x-.5)*4 if side=='right' else 1-abs(x-.5)*4
    return float(np.clip(value,0,1))*min(1,f['visible_ratio']/.4)*f['confidence']
def confidence(scores,cfg):
    s=np.asarray(scores,float)
    if len(s)<5 or not np.isfinite(s).all():return {'state':'INVALID'}
    out={'margin12':float(s[0]-s[1]),'margin13':float(s[0]-s[2]),'margin15':float(s[0]-s[4]),'dispersion':float(s[:cfg['depth']].std())}
    out['state']='HIGH' if out['margin12']>=cfg['high_margin'] and out['margin13']>=cfg['high_margin'] else 'MEDIUM' if out['margin12']>=cfg['medium_margin'] else 'LOW'
    return out
def clip_evidence(loaded,row,object_vector,plan,cfg):
    classes=classes_for(plan);off=loaded['roi_offsets'];group={}
    for k in range(int(off[row]),min(int(off[row+1]),int(off[row])+cfg['roi_cap'])):
        m=loaded['roi_meta'][k]
        if m['class'] not in classes or m['visible_ratio']<cfg['roi_visible'] or m['bbox_area']<cfg['roi_area']:continue
        views=group.setdefault(m['track_id'],[])
        if len(views)<cfg['views_per_track']:views.append((float(loaded['roi_eg2'][k]@object_vector),m))
    result=[]
    for tid,views in group.items():
        views.sort(key=lambda x:-x[0]);best,m=views[0];mean=float(np.mean([x[0] for x in views[:2]]));f=loaded['features'][row].get(tid)
        ev={'track_id':tid,'class':m['class'],'families':{},'winning_roi':m,'best_roi':best,'top2_roi_mean':mean,'same_track_consistency':min(x[0] for x in views),'same_track_binding':True,'motion_fallback':None}
        if cfg['roi'] and best>=cfg['roi_floor']:ev['families']['object']=best if cfg['object_aggregation']=='max' else mean
        if cfg['motion']:
            motion=motion_support(f,plan)
            if motion is None and plan['requires_motion']:
                fallback=[x for x in loaded['features'][row].values() if x and x['class']==m['class'] and motion_support(x,plan) is not None]
                if fallback:
                    chosen=max(fallback,key=lambda x:x['duration']*x['confidence']);motion=motion_support(chosen,plan);ev['motion_fallback']={'reason':'selected_roi_track_has_insufficient_temporal_observations','track_id':chosen['track_id']};ev['same_track_binding']=False
            if motion is not None:ev['families']['motion']=motion
        spatial=spatial_support(f,plan) if cfg['spatial'] else None
        if spatial is not None:ev['families']['spatial']=spatial
        result.append(ev)
    if not result:return None
    selected=max(result,key=lambda e:e['best_roi'])
    if not cfg['binding']:
        selected=json.loads(json.dumps(selected));selected['same_track_binding']=False
        for fam in ('motion','spatial'):
            values=[e['families'][fam] for e in result if fam in e['families']]
            if values:selected['families'][fam]=max(values)
    return selected
def vote(delta,weak,strong):return 2 if delta>=strong else 1 if delta>=weak else -2 if delta<=-strong else -1 if delta<=-weak else 0
def promote(scores,evidence,plan,cfg):
    s=np.asarray(scores);base=np.argsort(-s,kind='stable');c=confidence(s[base],cfg);diag={'confidence':c,'activated':False,'swap':False,'comparisons':[],'fallback':None}
    try:validate(plan)
    except (ValueError,TypeError,KeyError):diag['fallback']='invalid_plan';return base,diag
    if not cfg['factorized']:diag['fallback']='factorized_planning_removed';return base,diag
    if c['state']=='INVALID':diag['fallback']='invalid_global';return base,diag
    if not classes_for(plan):diag['fallback']='scene_or_unsupported_entity';return base,diag
    high=cfg['gate'] and c['state']=='HIGH'
    if high and not cfg['overwhelming']:return base,diag
    diag['activated']=True;inc=int(base[0]);old=evidence.get(inc)
    if old is None:diag['fallback']='missing_incumbent_evidence';return base,diag
    window=min(cfg['promotion_window'] if cfg['cap'] else cfg['depth'],len(base),cfg['depth'])
    for pos in range(1,window):
        row=int(base[pos]);new=evidence.get(row);gap=float(s[inc]-s[row])
        if new is None:continue
        limit=cfg['swap_margin']*(.5 if c['state']=='MEDIUM' and cfg['gate'] else 1)
        if gap>limit:continue
        deltas={};votes={}
        for fam in ('object','motion','spatial'):
            if fam not in new['families'] or fam not in old['families']:continue
            delta=new['families'][fam]-old['families'][fam];prefix='roi' if fam=='object' else fam
            deltas[fam]=float(delta);votes[fam]=vote(delta,cfg[prefix+'_delta'],cfg[prefix+'_strong'])
        positive=sum(v>0 for v in votes.values());strong=sum(v==2 for v in votes.values());contradiction=any(v==-2 for v in votes.values())
        rules=[]
        if positive>=2 and not contradiction:rules.append('A_two_families')
        if strong>=1 and gap<=cfg['extreme_margin'] and not contradiction:rules.append('B_one_strong_tiny_margin')
        if votes.get('object')==2 and votes.get('motion')==2 and new['same_track_binding'] and old['same_track_binding'] and not contradiction:rules.append('C_bound_object_motion')
        if cfg['policy']=='A':rules=[r for r in rules if r.startswith('A')]
        if cfg['policy']=='AB':rules=[r for r in rules if not r.startswith('C')]
        if not cfg['signed']:rules=['unsigned_any_positive'] if positive else []
        if high and not (pos<3 and strong>=2 and positive>=3 and not contradiction and gap<=cfg['extreme_margin']):rules=[]
        comparison={'candidate_row':row,'incumbent_row':inc,'from_rank':pos+1,'to_rank':1,'EG2_scores':[float(s[inc]),float(s[row])],'global_delta':-gap,'deltas':deltas,'votes':votes,'rule':rules,'incumbent_evidence':old,'challenger_evidence':new,'confidence_state':c['state'],'object_subquery':object_queries(plan).get(cfg['object_variant'])}
        diag['comparisons'].append(comparison)
        if rules:
            diag.update(swap=True,promotion=comparison);return np.concatenate(([row],base[:pos],base[pos+1:])),diag
    return base,diag
def rank(loaded,vector,plan,cfg,object_vector=None):
    started=time.perf_counter();scores=loaded['global_eg2']@vector;base=np.argsort(-scores,kind='stable');similarity=time.perf_counter()-started;evidence={};started=time.perf_counter()
    state=confidence(scores[base],cfg)['state'];failure=None
    confidence_seconds=time.perf_counter()-started;started=time.perf_counter()
    if plan and object_vector is not None and state!='INVALID' and (state!='HIGH' or not cfg['gate'] or cfg['overwhelming']):
        try:
            validate(plan)
            for row in base[:cfg['depth']]:evidence[int(row)]=clip_evidence(loaded,int(row),object_vector,plan,cfg)
        except (TypeError,ValueError,KeyError,IndexError):evidence={};failure='invalid_optional_evidence'
    evidence_seconds=time.perf_counter()-started;started=time.perf_counter();order,diag=promote(scores,evidence,plan,cfg);diag['evidence']=evidence
    diag['timing']={'global_similarity':similarity,'confidence_gate':confidence_seconds,'ROI_and_bound_track_evidence':evidence_seconds,'pairwise_reranking':time.perf_counter()-started}
    if failure:diag['fallback']=failure
    return order,scores,diag
def load_index(path):
    p=Path(path);meta=json.loads((p/'index_meta.json').read_text())
    if meta['format']!=FORMAT:raise ValueError('WRONG_V3_INDEX')
    evidence_invalid=[]
    for name,h in meta['hashes'].items():
        if not (p/name).is_file() or digest(p/name)!=h:
            if name in ('global_eg2.npy','manifest.json','provenance.json'):raise ValueError('V3_INDEX_HASH_MISMATCH '+name)
            evidence_invalid.append(name)
    out={'meta':meta}
    out['global_eg2']=np.load(p/'global_eg2.npy',allow_pickle=False);out['rows']=json.loads((p/'manifest.json').read_text())['clips']
    if evidence_invalid:
        out.update(roi_eg2=np.empty((0,768),np.float32),roi_offsets=np.zeros(len(out['rows'])+1,np.int64),roi_meta=[],features=[{} for r in out['rows']],evidence_fallback=evidence_invalid)
    else:
        for n in ('roi_eg2','roi_offsets'):out[n]=np.load(p/(n+'.npy'),allow_pickle=False)
        out['roi_meta']=json.loads((p/'roi_meta.json').read_text());tracks=json.loads((p/'track_records.json').read_text())
        out['features']=[{o['slot']:track_feature(o) for o in objects} for objects in tracks]
        for row,r in enumerate(out['rows']):
            for k in range(int(out['roi_offsets'][row]),int(out['roi_offsets'][row+1])):
                m=out['roi_meta'][k];track=next((o for o in tracks[row] if o['slot']==m['track_id']),None)
                if m['clip_id']!=r['clip_id'] or m['embedding_row']!=k or track is None or not any(v['bbox']==m['bbox'] and v['timestamp']==m['timestamp'] and v['frame_position']==m['frame_position'] for v in track['observations']):raise ValueError('FABRICATED_V3_OBSERVATION')
    for n in ('global_eg2','roi_eg2'):
        if out[n].shape[1]!=768 or not np.isfinite(out[n]).all() or (len(out[n]) and not np.allclose(np.linalg.norm(out[n],axis=1),1,atol=.003)):raise ValueError('INVALID_V3_VECTOR')
    return out

class DriveEvidenceV3SearchEngine:
    def __init__(self,index_path,worker_factory=None,planner=None):
        import threading
        from .driveevidence_v3_planner import QueryPlanner
        from .driveevidence_v3_worker import DriveEvidenceV3Worker
        self.index_path=Path(index_path);self.worker_factory=worker_factory or DriveEvidenceV3Worker;self.planner=planner or QueryPlanner();self.lock=threading.RLock();self.loaded=self.worker=None;self.load_count=0
    def load(self):
        if self.loaded is None:self.loaded=load_index(self.index_path);self.load_count+=1
        return self
    def search_table(self,query,ranker='driveevidence_v3',decision_provider=None,**unused):
        if ranker!='driveevidence_v3' or decision_provider not in (None,'none','laya'):raise ValueError('V3_MODE_REQUIRED')
        with self.lock:
            self.load();start=time.perf_counter();cfg=self.loaded['meta']['rerank']
            if self.planner.cached(query) is None and self.worker:self.worker.close();self.worker=None
            plan,pm=self.planner.plan(query);self.planner.close()
            if self.worker is None:self.worker=self.worker_factory(self.loaded['meta']['config']);self.worker.load()
            vector,em=self.worker.encode_text(query);rv=None;rm=None
            scores=self.loaded['global_eg2']@vector;c=confidence(np.sort(scores)[::-1],cfg);rq=object_queries(plan).get(cfg['object_variant']) if plan else None
            if rq and c['state']!='INVALID' and (c['state']!='HIGH' or not cfg['gate'] or cfg['overwhelming']):
                rv,rm=self.worker.encode_text(rq) if cfg['object_subquery'] else (vector,{'seconds':0})
            order,scores,d=rank(self.loaded,vector,plan,cfg,rv);serialization_started=time.perf_counter();candidates=[]
            for rank_number,j in enumerate(order,1):
                r=self.loaded['rows'][int(j)];candidates.append({**{k:r[k] for k in ('clip_id','source_row','source_clip_id','segment_id','start_sec','end_sec','day_night')},'row':int(j),'rank':rank_number,'ranker':ranker,'score':float(scores[j]),'final_score':float(scores[j])})
            return {'query':query,'format':FORMAT+'-candidates','index':str(self.index_path),'candidates':candidates,'decision':{'state':'UNCALIBRATED','provider':'laya_query_planner'},'retrieval':{'planner':pm,'query_plan':plan,'object_query':rq,'diagnostic':d,'global_text_encode_seconds':em['seconds'],'object_text_encode_seconds':rm['seconds'] if rm else 0,'serialization_seconds':time.perf_counter()-serialization_started,'warm_seconds':time.perf_counter()-start,'Qwen_calls':0,'candidate_level_Laya_calls':0,'engine_load_count':self.load_count,'evidence_fallback':self.loaded.get('evidence_fallback')}}
    def close(self):
        if self.worker:self.worker.close()
        self.planner.close();self.loaded=self.worker=None
