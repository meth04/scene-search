"""Frozen factor evidence, query-relative ranks and bounded rank fusion."""
from __future__ import annotations
import json,time,math,re,threading
from pathlib import Path
import numpy as np
from .embeddinggemma2 import digest,identity
from .driveevidence_v3 import classes_for,track_feature,motion_support
from .driveevidence_v3_planner import validate,object_queries,QueryPlanner
FORMAT='driveevidence-v4'
FAMILIES=('object','motion','spatial','scene')
DEFAULT={'depth':10,'promotion_window':5,'object_variant':'B','object_aggregation':'max','roi':True,'motion':True,'spatial':True,'scene':False,
 'factorized':True,'object_subquery':True,'binding':True,'relative':True,'fusion':'borda','risk':True,'confidence':True,
 'spatial_formula':'median','reference_grounding':False,'motion_formula':'original','roi_quality':'standard','class_routing':'exact',
 'global_weight':1.,'object_weight':1.,'motion_weight':1.,'spatial_weight':1.,'scene_weight':1.,'rank_advantage':.25,
 'global_deficit':.04,'tiny_deficit':.01,'contradiction':True,'strong_exception':True,'phrase_stability':False,'rerank':True}

def dense_feature(track,clip_duration=5):
    obs=track['observations']
    if not obs:return None
    t=np.array([v['timestamp'] for v in obs],float);a=np.array([v['xywh'] for v in obs],float)
    if not np.isfinite(a).all() or not np.isfinite(t).all() or (len(t)>1 and (np.diff(t)<=0).any()):return None
    x,y=a[:,0],a[:,1];area=a[:,2]*a[:,3];dt=np.diff(t)
    vel=np.diff(a[:,:2],axis=0)/dt[:,None] if len(t)>1 else np.zeros((1,2))
    speed=np.linalg.norm(vel,axis=1)
    f=track_feature(track,clip_duration) or {'track_id':track['slot'],'class':track['class'],'duration':float(t[-1]-t[0]),'visible_ratio':track['visible_ratio'],'confidence':track['confidence'],'supports':{}}
    f.update(mean_x=float(x.mean()),median_x=float(np.median(x)),min_x=float(x.min()),max_x=float(x.max()),start_x=float(x[0]),mid_x=float(x[len(x)//2]),end_x=float(x[-1]),
      left_fraction=float(np.mean(x<.4)),center_fraction=float(np.mean((x>=.4)&(x<=.6))),right_fraction=float(np.mean(x>.6)),distance_to_center=float(np.mean(abs(x-.5))),
      mean_y=float(y.mean()),median_y=float(np.median(y)),mean_area=float(area.mean()),min_area=float(area.min()),max_area=float(area.max()),
      acceleration_proxy=float(np.mean(abs(np.diff(speed)))) if len(speed)>1 else 0.,mean_speed=float(speed.mean()),
      normalized_delta_x=float((x[-1]-x[0])/max((t[-1]-t[0])/clip_duration,.02)),normalized_delta_y=float((y[-1]-y[0])/max((t[-1]-t[0])/clip_duration,.02)),
      appearance_time=float(t[0]/clip_duration),disappearance_time=float(t[-1]/clip_duration),coordinate_scope='image-plane only; no lane/world truth; no ego-motion compensation')
    return f

def ground_plan(plan,query,cfg):
    if not plan:return None
    p=json.loads(json.dumps(validate(plan)))
    # Query-text grounding only: an entity reference needs explicit relational syntax.
    if cfg['reference_grounding'] and p['spatial']['reference']=='entity':
        explicit=re.search(r'\b(?:left|right|front|back|side)\s+of\s+(?:the |a |an )?(?:car|truck|bus|van|person|motorcycle|bicycle|vehicle)\b',query.casefold())
        if not explicit:p['spatial']['reference']='ego';p['fallback_fields']['reference']='query_text_image_side_grounding'
    return p

def spatial_support(f,p,cfg):
    if not f or not p['requires_spatial_relation'] or p['spatial']['reference']=='entity':return None
    side=p['spatial']['side']
    if side not in ('left','right','center'):return None
    x=f['median_x'];occ=f[side+'_fraction']
    median=1-x if side=='left' else x if side=='right' else 1-2*abs(x-.5)
    end=1-f['end_x'] if side=='left' else f['end_x'] if side=='right' else 1-2*abs(f['end_x']-.5)
    formula=cfg['spatial_formula']
    v=median if formula=='median' else (.5*median+.3*occ+.2*end) if formula=='occupancy' else (.4*median+.4*occ+.2*min(1,f['visible_ratio']/.4))
    return float(np.clip(v,0,1))

def motion_value(f,p,cfg):
    if not f or not p['requires_motion'] or f['duration']<.6:return None
    if cfg['motion_formula']=='original':return motion_support(f,p)
    keys=[];d=p['motion_direction'];action=p['action']
    dx=f.get('delta_x',0);growth=f.get('area_growth',0)
    vals={'left_to_right':max(0,dx),'right_to_left':max(0,-dx),'lateral':abs(dx),'toward':max(0,growth),'away':max(0,-growth)}
    if d in vals:keys.append(vals[d])
    av={'approaching':max(0,growth),'moving_away':max(0,-growth),'crossing':abs(dx)*(1 if f['min_x']<=.5<=f['max_x'] else .25),
        'passing':abs(dx)+max(0,growth)*.1,'overtaking':abs(dx)+max(0,growth)*.1,'stationary':1/(1+f['mean_speed']*20),
        'appearing':f['appearance_time'],'disappearing':1-f['disappearance_time'],'moving':f['mean_speed']}
    if action in av:keys.append(av[action])
    if not keys:return None
    v=min(keys)
    return float(v*min(1,f['visible_ratio']/.4)) if cfg['motion_formula']=='visibility' else float(v)

def clip_evidence(loaded,row,ov,p,cfg):
    if p is None or ov is None:return None
    classes=classes_for(p)
    if cfg['class_routing']=='broad' and classes.intersection({'car','truck','bus','motorcycle'}):classes.update(('car','truck','bus','motorcycle'))
    off=loaded['roi_offsets'];groups={}
    for k in range(int(off[row]),int(off[row+1])):
        m=loaded['roi_meta'][k]
        if m['class'] not in classes:continue
        if cfg['roi_quality']=='standard' and (m['visible_ratio']<.12 or m['bbox_area']<.001):continue
        if cfg['roi_quality']=='strict' and (m['visible_ratio']<.25 or m['bbox_area']<.002):continue
        score=float(loaded['roi_eg2'][k]@ov)
        if not math.isfinite(score):continue
        groups.setdefault(m['track_id'],[]).append((score,m))
    evidence=[]
    for tid,views in groups.items():
        views.sort(key=lambda z:(-z[0],z[1]['embedding_row']));best,m=views[0]
        object_score=best if cfg['object_aggregation']=='max' else float(np.mean([s for s,m in views[:2]])) if cfg['object_aggregation']=='top2' else float(np.mean([s for s,m in views]))
        f=loaded['features'][row].get(tid);values={}
        if cfg['roi'] and p['requires_object_detail']:values['object']=object_score
        try:mv=motion_value(f,p,cfg) if cfg['motion'] else None
        except (KeyError,ValueError,TypeError):mv=None
        try:sv=spatial_support(f,p,cfg) if cfg['spatial'] else None
        except (KeyError,ValueError,TypeError):sv=None
        binding=True;fallback=None
        if mv is None and cfg['motion'] and p['requires_motion']:
            others=[]
            for x in loaded['features'][row].values():
                try:
                    if x and x['class']==m['class'] and motion_value(x,p,cfg) is not None:others.append(x)
                except (KeyError,ValueError,TypeError):continue
            if others:
                other=max(others,key=lambda x:x['visible_ratio']*x['duration']);mv=motion_value(other,p,cfg);binding=False
                fallback={'track_id':other['track_id'],'reason':'ROI winner lacks usable temporal observations'}
        if mv is not None:values['motion']=mv
        if sv is not None:values['spatial']=sv
        evidence.append({'families':values,'track_id':tid,'winning_roi':m,'same_track_binding':binding,'motion_fallback':fallback,'object_score':object_score})
    if not evidence:return None
    best=max(evidence,key=lambda z:z['object_score'])
    if not cfg['binding']:
        best=json.loads(json.dumps(best));best['same_track_binding']=False
        for family in ('motion','spatial'):
            vals=[e['families'][family] for e in evidence if family in e['families']]
            if vals:best['families'][family]=max(vals)
    return best

def relative_ranks(values):
    valid={k:float(v) for k,v in values.items() if v is not None and math.isfinite(v)};n=len(valid)
    # Midranks ensure equal supports do not acquire arbitrary evidence votes.
    return {k:{'rank':1+sum(x>v for x in valid.values())+(sum(x==v for x in valid.values())-1)/2,'valid_count':n,
        'norm':(sum(x>v for x in valid.values())+(sum(x==v for x in valid.values())-1)/2)/max(n-1,1)} for k,v in valid.items()}

def confidence(s):
    if len(s)<5 or not np.isfinite(s).all():return {'state':'INVALID'}
    top=np.asarray(s[:10],float);span=float(np.ptp(top));m12=float(s[0]-s[1]);m13=float(s[0]-s[2]);m15=float(s[0]-s[4])
    logits=(top-top.max())/max(float(top.std()),1e-6);probs=np.exp(logits);probs/=probs.sum()
    return {'margin12':m12,'margin13':m13,'margin15':m15,'std':float(top.std()),'range':span,'entropy':float(-sum(probs*np.log(probs+1e-12))),
      'relative_margin12':m12/max(span,1e-6),'state':'HIGH' if m12>=.025 and m13>=.025 else 'MEDIUM' if m12>=.012 else 'LOW'}

def promote(scores,evidence,p,cfg):
    started=time.perf_counter();scores=np.asarray(scores,float);base=np.argsort(-scores,kind='stable');c=confidence(scores[base]);diag={'confidence':c,'activated':False,'swap':False,'candidates':[],'comparisons':[],'fallback':None,'stage_timing':{'relative_rank_calculation':0.,'rank_fusion':0.,'promotion':0.}}
    if not cfg['rerank'] or not cfg['factorized']:return base,diag
    try:validate(p)
    except (ValueError,KeyError,TypeError):diag['fallback']='invalid_plan';return base,diag
    if c['state']=='INVALID':diag['fallback']='invalid_global';return base,diag
    begun=time.perf_counter();ids=[int(i) for i in base[:cfg['depth']]];inc=ids[0];ranks={}
    for f in FAMILIES:
        ranks[f]=relative_ranks({i:(evidence.get(i) or {}).get('families',{}).get(f) for i in ids})
    active=[f for f in FAMILIES if ranks[f]];diag['activated']=bool(active);diag['stage_timing']['relative_rank_calculation']=time.perf_counter()-begun;begun=time.perf_counter()
    def norm(f,i):
        if i not in ranks[f]:return None
        if cfg['relative']:return ranks[f][i]['norm']
        return -(evidence[i]['families'][f])  # raw-score removal control, deliberately heterogeneous
    def fusion(i):
        terms=[('global',(ids.index(i))/max(len(ids)-1,1),ids.index(i)+1)]
        terms += [(f,norm(f,i),ranks[f][i]['rank']) for f in active if i in ranks[f]]
        if cfg['fusion']=='none':return -ids.index(i)
        if cfg['fusion']=='rrf':return sum(cfg[f+'_weight']/(5+r) for f,n,r in terms)/sum(cfg[f+'_weight'] for f,n,r in terms)
        if cfg['fusion']=='condorcet':
            return sum(sum((norm(f,j)>norm(f,i))-(norm(f,j)<norm(f,i)) for f in active if i in ranks[f] and j in ranks[f]) for j in ids if j!=i)
        if cfg['fusion']=='majority':return sum(cfg[f+'_weight']*(1 if n<=.25 else 0) for f,n,r in terms)/sum(cfg[f+'_weight'] for f,n,r in terms)
        return -sum(cfg[f+'_weight']*n for f,n,r in terms)/sum(cfg[f+'_weight'] for f,n,r in terms)
    fused={i:float(fusion(i)) for i in ids}
    for i in ids:diag['candidates'].append({'candidate_row':i,'global_rank':ids.index(i)+1,'global_score':float(scores[i]),'ranks':{f:ranks[f].get(i,'UNKNOWN') for f in FAMILIES},'active_factors':active,'fusion_score':fused[i],'evidence':evidence.get(i),'promotion_decision':False})
    diag['stage_timing']['rank_fusion']=time.perf_counter()-begun;begun=time.perf_counter();chosen=[];window=min(cfg['promotion_window'],cfg['depth'],len(ids))
    for i in ids[1:window]:
        common=[f for f in active if i in ranks[f] and inc in ranks[f]]
        advantages={f:float(norm(f,inc)-norm(f,i)) for f in common}
        wins=[f for f,v in advantages.items() if v>1e-12];losses=[f for f,v in advantages.items() if v<-1e-12]
        strong=[f for f,v in advantages.items() if v>=cfg['rank_advantage']];contra=any(v<=-cfg['rank_advantage'] for v in advantages.values())
        gap=float(scores[inc]-scores[i]);bound=bool((evidence.get(i) or {}).get('same_track_binding') and (evidence.get(inc) or {}).get('same_track_binding'))
        rules=[]
        if len(wins)>=2 and gap<=cfg['global_deficit']:rules.append('A_two_factor_rank_wins')
        if strong and len(losses)==0 and gap<=cfg['tiny_deficit']:rules.append('B_strong_single_tiny_global_deficit')
        if 'object' in wins and 'motion' in wins and bound and gap<=cfg['global_deficit']:rules.append('C_same_track_object_motion')
        if len(wins)>len(common)/2 and gap<=cfg['global_deficit'] and c['state']!='HIGH':rules.append('D_factor_majority_ambiguous')
        if cfg['contradiction'] and contra:rules=[]
        if cfg['confidence'] and c['state']=='HIGH' and not (cfg['strong_exception'] and len(strong)>=2 and not losses and gap<=cfg['global_deficit']):rules=[]
        earns=bool(rules and fused[i]>fused[inc]+1e-12) if cfg['risk'] else bool(fused[i]>fused[inc]+1e-12)
        if cfg['phrase_stability'] and not (evidence.get(i) or {}).get('phrase_stable',False):earns=False
        comp={'candidate_row':i,'incumbent_row':inc,'from_rank':ids.index(i)+1,'global_margin':gap,'rank_advantages':advantages,'pairwise_wins':wins,'pairwise_losses':losses,'rule_triggered':rules,'same_track_status':bound,'confidence_state':c['state'],'earns_promotion':earns}
        diag['comparisons'].append(comp)
        if earns:chosen.append((fused[i]-fused[inc],-ids.index(i),i,comp))
    diag['stage_timing']['promotion']=time.perf_counter()-begun
    if chosen:
        _,_,i,comp=max(chosen);pos=ids.index(i);diag.update(swap=True,promotion=comp)
        for record in diag['candidates']:record['promotion_decision']=record['candidate_row']==i
        return np.concatenate(([i],base[:pos],base[pos+1:])),diag
    return base,diag

def load_index(path):
    p=Path(path);meta=json.loads((p/'index_meta.json').read_text(encoding='utf-8'))
    if meta['format']!=FORMAT or meta['rerank_sha256']!=identity(meta['rerank']):raise ValueError('INVALID_V4_IDENTITY')
    missing=[]
    for n,h in meta['hashes'].items():
        if not (p/n).is_file() or digest(p/n)!=h:
            if n in ('global_eg2.npy','manifest.json','provenance.json'):raise ValueError('V4_CRITICAL_HASH '+n)
            missing.append(n)
    g=np.load(p/'global_eg2.npy',allow_pickle=False);rows=json.loads((p/'manifest.json').read_text())['clips']
    if g.shape!=(len(rows),768) or not np.isfinite(g).all() or not np.allclose(np.linalg.norm(g,axis=1),1,atol=.003):raise ValueError('V4_INVALID_GLOBAL')
    out={'meta':meta,'rows':rows,'global_eg2':g}
    if missing:return {**out,'roi_eg2':np.empty((0,768),np.float32),'roi_meta':[],'roi_offsets':np.zeros(len(rows)+1,np.int64),'features':[{} for r in rows],'evidence_fallback':missing}
    roi=np.load(p/'roi_eg2.npy',allow_pickle=False);off=np.load(p/'roi_offsets.npy',allow_pickle=False);rm=json.loads((p/'roi_meta.json').read_text());tracks=json.loads((p/'track_records.json').read_text())
    if roi.shape!=(len(rm),768) or not np.isfinite(roi).all() or not np.allclose(np.linalg.norm(roi,axis=1),1,atol=.003) or len(off)!=len(rows)+1 or off[0]!=0 or off[-1]!=len(roi) or (np.diff(off)<0).any():raise ValueError('V4_INVALID_ROI')
    for row in range(len(rows)):
        for k in range(int(off[row]),int(off[row+1])):
            m=rm[k];t=next((t for t in tracks[row] if t['slot']==m['track_id']),None)
            if m['clip_id']!=rows[row]['clip_id'] or m['embedding_row']!=k or t is None or not any(o['bbox']==m['bbox'] and o['timestamp']==m['timestamp'] and o['frame_position']==m['frame_position'] for o in t['observations']):raise ValueError('V4_FABRICATED_OBSERVATION')
    return {**out,'roi_eg2':roi,'roi_offsets':off,'roi_meta':rm,'features':[{t['slot']:dense_feature(t) for t in group} for group in tracks]}

class DriveEvidenceV4SearchEngine:
    def __init__(self,index_path,worker_factory=None,planner=None):
        from .driveevidence_v4_worker import DriveEvidenceV4Worker
        self.index_path=Path(index_path);self.worker_factory=worker_factory or DriveEvidenceV4Worker;self.planner=planner or QueryPlanner(cache=Path(__file__).resolve().parents[1]/'outputs/driveevidence_v4/query_planner_cache');self.lock=threading.RLock();self.loaded=self.worker=None;self.load_count=0
    def load(self):
        if self.loaded is None:self.loaded=load_index(self.index_path);self.load_count+=1
        return self
    def search_table(self,query,ranker='driveevidence_v4',decision_provider=None,**unused):
        if ranker!='driveevidence_v4' or decision_provider not in (None,'none','laya'):raise ValueError('V4_MODE_REQUIRED')
        with self.lock:
            self.load();start=time.perf_counter();cfg=self.loaded['meta']['rerank']
            if self.planner.cached(query) is None and self.worker:self.worker.close();self.worker=None
            plan,pm=self.planner.plan(query);self.planner.close();plan=ground_plan(plan,query,cfg)
            if self.worker is None:self.worker=self.worker_factory(self.loaded['meta']['config']);self.worker.load()
            vector,em=self.worker.encode_text(query);ov=None;om=None;rq=object_queries(plan).get(cfg['object_variant']) if plan else None
            if rq:ov,om=self.worker.encode_text(rq) if cfg['object_subquery'] else (vector,{'seconds':0})
            begun=time.perf_counter();scores=self.loaded['global_eg2']@vector;base=np.argsort(-scores,kind='stable');similarity=time.perf_counter()-begun;begun=time.perf_counter();ev={};failure=None
            if plan:
                try:
                    for row in base[:cfg['depth']]:ev[int(row)]=clip_evidence(self.loaded,int(row),ov,plan,cfg)
                except (KeyError,ValueError,TypeError,IndexError):ev={};failure='invalid_factor_evidence'
            lookup=time.perf_counter()-begun;begun=time.perf_counter()
            try:order,d=promote(scores,ev,plan,cfg)
            except (KeyError,ValueError,TypeError,IndexError):order=base;d={'fallback':'invalid_rank_logic','swap':False,'timing':{}}
            d['timing']={'global_similarity':similarity,'evidence_lookup':lookup,'relative_ranks_fusion_promotion':time.perf_counter()-begun,**d.get('stage_timing',{})}
            if failure:d['fallback']=failure
            begun=time.perf_counter();candidates=[]
            for n,j in enumerate(order,1):
                r=self.loaded['rows'][int(j)];candidates.append({**{k:r[k] for k in ('clip_id','source_row','source_clip_id','segment_id','start_sec','end_sec','day_night')},'row':int(j),'rank':n,'ranker':ranker,'score':float(scores[j]),'final_score':float(scores[j])})
            return {'query':query,'format':FORMAT+'-candidates','index':str(self.index_path),'candidates':candidates,'decision':{'state':'UNCALIBRATED','provider':'laya_query_planner'},'retrieval':{'planner':pm,'query_plan':plan,'global_query_text':query,'object_query':rq,'diagnostic':d,'global_text_encode_seconds':em['seconds'],'object_text_encode_seconds':om['seconds'] if om else 0,'serialization_seconds':time.perf_counter()-begun,'warm_seconds':time.perf_counter()-start,'Qwen_calls':0,'candidate_level_Laya_calls':0,'engine_load_count':self.load_count,'evidence_fallback':self.loaded.get('evidence_fallback')}}
    def close(self):
        if self.worker:self.worker.close()
        self.planner.close();self.worker=self.loaded=None
