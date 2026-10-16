"""Frozen typed evidence with independent support, reliability and calibrated abstention.

EG2 stable exact ranking is the safe policy. No learned scorer or online tuning.
"""
from __future__ import annotations
import itertools, json, math, time
from pathlib import Path
import numpy as np
from .embeddinggemma2 import digest, identity, validate_config
FORMAT='driveevidence-v5'
CHANNELS=('object','spatial','relation','motion','scene')
REASONS=('EG2_HIGH_CONFIDENCE','NO_RELIABLE_CHALLENGER','OBJECT_EVIDENCE_WEAK','SPATIAL_BINDING_FAILED','RELATION_ASSIGNMENT_AMBIGUOUS','MOTION_UNRELIABLE','CALIBRATED_RISK_REJECT','SAFE_MULTI_FACTOR_OVERRIDE','INDEX_EVIDENCE_MISSING','PLANNER_INVALID','DRIFT_FALLBACK')
DEFAULT={'depth':10,'promotion_window':5,'object_text':'FULL_QUERY_ROI','view_aggregation':'top2','object':True,'spatial':True,'relation':True,'motion':False,'scene':False,'object_reliability':True,'spatial_reliability':True,'relation_reliability':True,'assignment':True,'confidence':True,'planner_confidence':True,'abstention':True,'calibration':True,'risk_penalty':3,'reliability_floor':.55,'factor_probability_floor':.7,'min_factors':1,'min_rank_advantage':.15,'max_global_deficit':.04,'high_margin':.035,'assignment_margin_scale':.03,'multi_view':True,'duration_scale':2.,'visible_scale':.5,'support_margin':False}
CLASSES=['car','truck','bus','motorcycle','bicycle','person','traffic light','stop sign','unknown']
def classes(entity):
    return {'vehicle':{'car','truck','bus'},'car':{'car'},'truck':{'truck'},'bus':{'bus'},'motorcycle':{'motorcycle'},'bicycle':{'bicycle'},'person':{'person'},'traffic light':{'traffic light'},'stop sign':{'stop sign'},'sign':{'stop sign'}}.get(entity['type'],set())
def unknown(reason,provenance=None):return {'support':None,'reliability':None,'reason':reason,'provenance':provenance or {}}
def channel(support,reliability,provenance,**details):
    if not math.isfinite(float(support)) or not math.isfinite(float(reliability)):return unknown('NONFINITE_EVIDENCE')
    return {'support':float(support),'reliability':float(np.clip(reliability,0,1)),'provenance':provenance,**details}
def geometry(obs):
    if len(obs)==0:return {}
    a=np.asarray(obs);t,x,y,w,h,c=a.T;area=w*h
    return {'duration':float(t[-1]-t[0]),'visibility':min(1.,len(t)/50.),'median_x':float(np.median(x)),'mean_x':float(np.mean(x)),'start_x':float(x[0]),'mid_x':float(x[len(x)//2]),'end_x':float(x[-1]),'mean_area':float(np.mean(area)),'x_std':float(np.std(x)),'left_fraction':float(np.mean(x<.4)),'center_fraction':float(np.mean((x>=.4)&(x<=.6))),'right_fraction':float(np.mean(x>.6)),'distance_to_center':float(np.mean(abs(x-.5))),'detector_confidence':float(np.mean(c))}

def object_channel(view_scores, g, factor_p, cfg, roles=None):
    if len(view_scores)==0:return unknown('NO_COMPATIBLE_ROI')
    s=np.sort(np.asarray(view_scores,float))[::-1]
    if not np.isfinite(s).all():return unknown('NONFINITE_ROI')
    top2=float(s[:2].mean());best=float(s[0]);variance=float(np.var(s));agreement=float(np.mean(np.abs(s-top2)<=.06))
    support=best if cfg['view_aggregation']=='max' else top2 if cfg['view_aggregation']=='top2' else float(s.mean())
    if factor_p is None and cfg['planner_confidence']:return channel(support,0,{'scope':'same-track frozen EG2 ROI','roles':roles},reason='UNKNOWN_FACTOR_CONFIDENCE')
    quality=min(1.,g.get('duration',0)/cfg['duration_scale'],g.get('visibility',0)/cfg['visible_scale'],math.sqrt(max(g.get('mean_area',0),0)/.003))
    consistency=math.exp(-variance/.003)*agreement*min(1,len(s)/2) if cfg['multi_view'] else 1.
    rel=min(quality,consistency,float(factor_p) if cfg['planner_confidence'] else 1.) if cfg['object_reliability'] else 1.
    return channel(support,rel,{'scope':'same-track frozen EG2 ROI','roles':roles,'class_compatible':True},best_similarity=best,top2_mean=top2,view_variance=variance,agreement=agreement,views=len(s),quality=quality,same_track_consistency=consistency)

def spatial_channel(obs, side, factor_p, cfg):
    g=geometry(obs)
    if side not in ('left','right','center') or not g:return unknown('SPATIAL_BINDING_FAILED')
    x=g['median_x'];support=1-x if side=='left' else x if side=='right' else 1-2*abs(x-.5)
    occupancy=g[side+'_fraction'];consistency=math.exp(-g['x_std']/.15)
    quality=min(1.,g['duration']/cfg['duration_scale'],g['visibility']/cfg['visible_scale'],consistency,occupancy)
    rel=min(quality,factor_p if factor_p is not None else 0.) if cfg['planner_confidence'] else quality
    if not cfg['spatial_reliability']:rel=1.
    return channel(support,rel,{'scope':'image-plane only; no world/lane position'},**g,position_consistency=consistency,occupancy=occupancy)

def exact_assignment(matrix, distinct=True):
    """Exact two-entity bipartite assignment; incompatibility is -inf, never zero."""
    m=np.asarray(matrix,float)
    if m.ndim!=2 or m.shape[0]!=2:return None
    choices=[]
    for a in range(m.shape[1]):
        for b in range(m.shape[1]):
            if distinct and a==b:continue
            if np.isfinite(m[0,a]) and np.isfinite(m[1,b]):choices.append((float((m[0,a]+m[1,b])/2),a,b))
    choices.sort(key=lambda v:(-v[0],v[1],v[2]))
    if not choices:return None
    score,a,b=choices[0];margin=score-choices[1][0] if len(choices)>1 else None
    return {'tracks':[a,b],'quality':score,'margin':margin,'distinct':a!=b,'alternatives':len(choices)-1}

def relation_channel(obs_a, obs_b, assignment, relation, factor_p, object_reliabilities, cfg):
    if not assignment or not assignment['distinct']:return unknown('RELATION_ASSIGNMENT_AMBIGUOUS')
    if relation not in ('left_of','right_of','beside'):return unknown('WORLD_DEPTH_UNCERTIFIED')
    # Only directly co-observed timestamps, no invented interpolation or boxes.
    a={round(float(o[0]),3):o for o in obs_a};b={round(float(o[0]),3):o for o in obs_b};t=sorted(a.keys()&b.keys())
    if len(t)<2:return unknown('NO_TEMPORAL_OVERLAP')
    dt=np.diff(t);overlap=float(np.minimum(dt,.15).sum());dx=np.array([b[k][1]-a[k][1] for k in t])
    vals=(dx>.02) if relation=='left_of' else (dx<-.02) if relation=='right_of' else (abs(dx)<.35)
    support=float(vals.mean());stability=float(max(np.mean(dx>=0),np.mean(dx<=0)))
    margin=assignment['margin'];assignment_rel=1. if margin is None else min(1.,max(margin,0)/cfg['assignment_margin_scale'])
    semantic=min(object_reliabilities) if object_reliabilities else 0.
    rel=min(semantic,assignment_rel,min(1,overlap/2.),stability,factor_p if factor_p is not None else 0.)
    if not cfg['relation_reliability']:rel=1.
    return channel(support,rel,{'scope':'distinct-track co-observed image geometry; physical depth unsupported','assignment':assignment},overlap_seconds=overlap,overlap_samples=len(t),geometry_consistency=stability,assignment_quality=assignment['quality'],assignment_margin=margin,distinct_track_validity=True)

def load_index(path):
    p=Path(path);meta=json.loads((p/'index_meta.json').read_text(encoding='utf-8'))
    if meta.get('format')!=FORMAT or meta.get('schema')!=1:raise ValueError('V5_SCHEMA_MISMATCH')
    validate_config(meta['config'])
    if meta['policy_sha256']!=identity(meta['policy']) or meta['architecture_sha256']!=identity(meta['architecture']):raise ValueError('V5_POLICY_HASH_MISMATCH')
    from .driveevidence_v5_planner import REVISION,SCHEMA
    if meta['model_versions']['Laya_revision']!=REVISION or meta['model_versions']['planner_schema']!=SCHEMA:raise ValueError('V5_MODEL_HASH_MISMATCH')
    missing=[]
    for name,h in meta['hashes'].items():
        if not (p/name).is_file() or digest(p/name)!=h:
            if name in ('global_vectors.npy','manifest.json','provenance.json'):raise ValueError('V5_CRITICAL_HASH '+name)
            missing.append(name)
    rows=json.loads((p/'manifest.json').read_text(encoding='utf-8'))['clips'];g=np.load(p/'global_vectors.npy',mmap_mode='r',allow_pickle=False)
    if g.shape!=(len(rows),768) or g.dtype!=np.float32 or not np.isfinite(g).all() or not np.allclose(np.linalg.norm(g,axis=1),1,atol=.003):raise ValueError('V5_INVALID_GLOBAL')
    out={'meta':meta,'rows':rows,'global_vectors':g,'path':p,'evidence_fallback':missing}
    if missing:return out
    try:
        v=np.load(p/'track_semantic_vectors.npy',mmap_mode='r',allow_pickle=False);offs=np.load(p/'track_offsets.npy',mmap_mode='r',allow_pickle=False);tm=dict(np.load(p/'track_meta.npz',allow_pickle=False));geo=dict(np.load(p/'track_geometry.npz',allow_pickle=False))
        if v.ndim!=2 or v.shape[1]!=768 or not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v,axis=1),1,atol=.003):raise ValueError('INVALID_TRACK_VECTOR')
        n=len(tm['track_id']);vo=tm['view_offsets'];oo=geo['observation_offsets'];obs=geo['observations']
        if len(offs)!=len(rows)+1 or offs[0]!=0 or offs[-1]!=n or (np.diff(offs)<0).any() or len(vo)!=n+1 or vo[0]!=0 or vo[-1]!=len(v) or (np.diff(vo)<0).any() or len(oo)!=n+1 or oo[0]!=0 or oo[-1]!=len(obs) or (np.diff(oo)<0).any():raise ValueError('INVALID_TRACK_OFFSETS')
        if obs.ndim!=2 or obs.shape[1]!=6 or not np.isfinite(obs).all() or not np.isfinite(tm['quality']).all() or len(tm['class_id'])!=n or (tm['class_id']<0).any() or (tm['class_id']>=len(CLASSES)).any():raise ValueError('INVALID_TRACK_META')
        for row in range(len(rows)):
            tids=tm['track_id'][offs[row]:offs[row+1]]
            if len(set(tids.tolist()))!=len(tids):raise ValueError('DUPLICATE_TRACK_ID')
        out.update(track_vectors=v,track_offsets=offs,track_meta=tm,geometry=geo)
    except (ValueError,KeyError,IndexError,OSError) as exc:out['evidence_fallback']=['CORRUPT_OPTIONAL_TRACK_METADATA',str(exc)]
    return out

def candidate_evidence(loaded,row,plan,global_vector,factor_vectors,cfg):
    ev={f:unknown('INACTIVE_OR_UNAVAILABLE') for f in CHANNELS}
    if not plan or loaded.get('evidence_fallback'):return ev
    entities=plan.get('entities',[])
    if not entities:return ev
    tm=loaded['track_meta'];geo=loaded['geometry'];lo,hi=loaded['track_offsets'][row:row+2];tracks=list(range(int(lo),int(hi)))
    matrix=np.full((len(entities),len(tracks)),-np.inf);details={};factor=plan['factors']
    for eidx,entity in enumerate(entities):
        adaptive=cfg['object_text']=='GENERIC_ADAPTIVE_POLICY' and any(entity.get(k) for k in ('color','subtype','appearance_terms'))
        use_object=cfg['object_text']=='OBJECT_SUBQUERY_ROI' or adaptive or eidx>0
        vec=factor_vectors.get(entity['object_phrase']) if use_object else global_vector
        if vec is None:continue
        for j,k in enumerate(tracks):
            if CLASSES[int(tm['class_id'][k])] not in classes(entity):continue
            o=geo['observations'][geo['observation_offsets'][k]:geo['observation_offsets'][k+1]];g=geometry(o)
            v=loaded['track_vectors'][tm['view_offsets'][k]:tm['view_offsets'][k+1]]
            roles=tm['crop_roles'][tm['view_offsets'][k]:tm['view_offsets'][k+1]].tolist()
            d=object_channel(v@vec,g,entity.get('confidence'),cfg,roles=roles)
            if d['support'] is None:continue
            d['provenance'].update(track_id=int(tm['track_id'][k]),track_row=k,clip_row=row,object_text_strategy='OBJECT_SUBQUERY_ROI' if use_object else 'FULL_QUERY_ROI')
            matrix[eidx,j]=d['support'];details[eidx,j]=(d,o)
    primary=matrix[0];finite=np.where(np.isfinite(primary))[0]
    winner=None
    if len(finite):
        best=sorted(finite,key=lambda j:(-primary[j],int(tm['track_id'][tracks[j]])))[0];winner=best;obj,obs=details[0,best]
        sorted_scores=sorted(primary[finite],reverse=True);margin=float(sorted_scores[0]-sorted_scores[1]) if len(sorted_scores)>1 else None
        obj['best_vs_second_semantic_margin']=margin
        if cfg['support_margin'] and margin is not None:obj['reliability']*=min(1,max(0,margin)/.025)
        if cfg['object']:ev['object']=obj
        if cfg['spatial'] and factor['spatial']['active']:
            sp=spatial_channel(obs,plan['spatial']['side'],factor['spatial']['probability'] if cfg['planner_confidence'] else 1.,cfg)
            sp['reliability']=min(sp['reliability'] or 0,obj['reliability']);sp['provenance'].update(obj['provenance']);ev['spatial']=sp
    graph=plan.get('relation_graph')
    if cfg['relation'] and cfg['assignment'] and graph and matrix.shape[0]==2:
        assignment=exact_assignment(matrix,distinct=True)
        if assignment:
            a,b=assignment['tracks'];d1,o1=details[0,a];d2,o2=details[1,b]
            rel=relation_channel(o1,o2,assignment,graph['relation'],factor['relation']['probability'] if cfg['planner_confidence'] else 1.,[d1['reliability'],d2['reliability']],cfg)
            rel['provenance'].update(physical_track_ids=[int(tm['track_id'][tracks[a]]),int(tm['track_id'][tracks[b]])],clip_row=row)
            # Appearance and relation are bound to the same explicit primary entity.
            if cfg['object']:ev['object']=d1
            ev['relation']=rel
        else:ev['relation']=unknown('RELATION_ASSIGNMENT_AMBIGUOUS')
    if factor['motion']['active']:ev['motion']=unknown('MOTION_UNRELIABLE',{'camera_compensation':'NOT_ENABLED_WITHOUT_NESTED_PROOF'})
    # Scene/global is the same safe EG2 signal, never counted twice as an independent factor.
    return ev

def stable_base(scores):
    s=np.asarray(scores,float)
    if s.ndim!=1 or not len(s) or not np.isfinite(s).all():raise ValueError('V5_NONFINITE_GLOBAL_SCORES')
    return np.argsort(-s,kind='stable')
def propose(scores,evidence,plan,cfg):
    base=stable_base(scores);diag={'state':'KEEP_EG2','reason_code':'NO_RELIABLE_CHALLENGER','swap':False,'proposal':None,'comparisons':[],'evidence':evidence,'coverage':{},'EG2_margin':float(scores[base[0]]-scores[base[1]]) if len(base)>1 else None}
    if not isinstance(plan,dict) or not isinstance(plan.get('factors'),dict):diag.update(state='INSUFFICIENT_EVIDENCE',reason_code='PLANNER_INVALID');return base,diag
    ids=[int(i) for i in base[:cfg['depth']]];inc=ids[0]
    valid={}
    for f in CHANNELS:
        active=plan['factors'].get(f,{})
        allowed=active.get('active',False) and (not cfg['planner_confidence'] or active.get('probability') is not None and active['probability']>=cfg['factor_probability_floor'])
        vals={i:evidence.get(i,{}).get(f,{}) for i in ids};vs={i:v['support'] for i,v in vals.items() if allowed and v.get('support') is not None and v.get('reliability') is not None and v['reliability']>=cfg['reliability_floor']}
        valid[f]={i:(sum(s<o for o in vs.values())+.5*(sum(s==o for o in vs.values())-1))/max(len(vs)-1,1) for i,s in vs.items()}
        diag['coverage'][f]={'active':bool(active.get('active')),'reliable_candidates':len(vs)}
    if cfg['confidence'] and cfg['abstention'] and diag['EG2_margin'] is not None and diag['EG2_margin']>=cfg['high_margin']:diag['reason_code']='EG2_HIGH_CONFIDENCE';return base,diag
    chosen=[]
    for rank,i in enumerate(ids[1:cfg['promotion_window']],2):
        common=[f for f in CHANNELS if inc in valid[f] and i in valid[f]]
        advantages={f:float(valid[f][inc]-valid[f][i]) for f in common};wins=[f for f,a in advantages.items() if a>=cfg['min_rank_advantage']];losses=[f for f,a in advantages.items() if a<=-cfg['min_rank_advantage']]
        rels=[evidence[i][f]['reliability'] for f in wins];gap=float(scores[inc]-scores[i]);minimum=min(rels) if rels else 0.
        # Interpretable deterministic intervention strength, not a predicted probability.
        strength=minimum*(float(np.mean([advantages[f] for f in wins])) if wins else 0.)*max(0,1-gap/max(cfg['max_global_deficit'],1e-6))
        relation=evidence.get(i,{}).get('relation',{})
        record={'incumbent_row':inc,'challenger_row':i,'rank_from':rank,'rank_to':1,'EG2_margin_against_challenger':gap,'number_reliable_active_factors':len(common),'factors_favoring_challenger':wins,'factors_favoring_incumbent':losses,'minimum_challenger_reliability':minimum,'mean_challenger_reliability':float(np.mean(rels)) if rels else None,'assignment_quality':relation.get('assignment_quality'),'assignment_margin':relation.get('assignment_margin'),'same_track_consistency':evidence.get(i,{}).get('object',{}).get('same_track_consistency'),'contradiction_count':len(losses),'factor_availability':common,'advantages':advantages,'intervention_strength':strength,'promotion_depth':rank,'relation_assignment_required':bool(plan.get('relation_graph'))}
        eligible=bool(len(wins)>=cfg['min_factors'] and not losses and gap<=cfg['max_global_deficit'])
        if not cfg['abstention']:eligible=bool(wins and gap<=cfg['max_global_deficit'])
        if plan.get('relation_graph') and 'relation' not in common:eligible=False
        record['eligible']=eligible;diag['comparisons'].append(record)
        if eligible:chosen.append((strength,-rank,record))
    if chosen:diag['proposal']=max(chosen,key=lambda x:(x[0],x[1]))[2]
    elif any(v['active'] for v in diag['coverage'].values()):
        diag['state']='INSUFFICIENT_EVIDENCE'
        diag['reason_code']='RELATION_ASSIGNMENT_AMBIGUOUS' if plan.get('relation_graph') else 'OBJECT_EVIDENCE_WEAK' if not valid['object'] else 'NO_RELIABLE_CHALLENGER'
    return base,diag

def decide(scores,evidence,plan,cfg,policy,drift=False,missing=False):
    base,d=propose(scores,evidence,plan,cfg)
    if missing:d.update(state='INSUFFICIENT_EVIDENCE',reason_code='INDEX_EVIDENCE_MISSING');return base,d
    if drift:d.update(state='KEEP_EG2',reason_code='DRIFT_FALLBACK');return base,d
    p=d['proposal']
    if p is None:return base,d
    if not policy.get('enabled',False) or p['intervention_strength']<policy.get('threshold',1.):d.update(state='KEEP_EG2',reason_code='CALIBRATED_RISK_REJECT');return base,d
    i=p['challenger_row'];pos=int(np.where(base==i)[0][0]);order=np.concatenate(([i],base[:pos],base[pos+1:]))
    d.update(state='SAFE_RERANK',reason_code='SAFE_MULTI_FACTOR_OVERRIDE',swap=True,promotion=p)
    return order,d

def binomial_upper(harmful,n,alpha=.05):
    if n<=0 or harmful>=n:return 1.
    if not 0<=harmful<=n:raise ValueError('INVALID_BINOMIAL_COUNTS')
    if harmful==0:return float(1-alpha**(1/n))
    # Invert exact Binomial CDF, avoiding optional numerical dependencies.
    lo,hi=0.,1.
    for _ in range(90):
        p=(lo+hi)/2
        terms=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log1p(-p) for k in range(harmful+1)]
        peak=max(terms);cdf=math.exp(peak)*sum(math.exp(t-peak) for t in terms)
        if cdf>alpha:lo=p
        else:hi=p
    return float(hi)
def calibrate(records,thresholds,min_support=29):
    trials=[]
    for threshold in sorted(thresholds):
        chosen=[r for r in records if r['strength']>=threshold];groups={}
        for r in chosen:groups.setdefault(r['source'],[]).append(r)
        # Source dependence: one effective Bernoulli decision per source; unknown is unsafe.
        harms=sum(any(r['outcome'] in ('harmful','UNKNOWN') for r in rs) for rs in groups.values());n=len(groups);upper=binomial_upper(harms,n,alpha=.05/max(len(thresholds),1))
        h=sum(r['outcome']=='harmful' for r in chosen);c=sum(r['outcome']=='correct' for r in chosen);neutral=sum(r['outcome']=='neutral' for r in chosen);unk=sum(r['outcome']=='UNKNOWN' for r in chosen)
        safe=bool(n>=min_support and h/max(len(chosen),1)<=.05 and upper<=.1 and not unk)
        trials.append({'threshold':threshold,'interventions':len(chosen),'independent_source_decisions':n,'correct':c,'harmful':h,'neutral':neutral,'unknown':unk,'harmful_rate':h/len(chosen) if chosen else None,'upper_95_bound':upper,'passes':safe})
    passing=[t for t in trials if t['passes']];selected=passing[0] if passing else None
    return {'enabled':bool(selected),'threshold':selected['threshold'] if selected else 1.1,'selected':selected,'candidate_thresholds':trials,'opportunities':len(records),'minimum_independent_support':min_support,'gate':'PASS' if selected else 'HOLD','bound':'One-sided exact Clopper-Pearson, Bonferroni over candidate thresholds, source-sequence clustered; calibration only, independence/exchangeability assumptions, no universal distribution-shift guarantee'}
