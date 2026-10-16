"""Persistent V5 product lifecycle, bounded query cache, drift fallback and shadow serving."""
from __future__ import annotations
import json, threading, time, uuid
from collections import OrderedDict, deque
from pathlib import Path
import numpy as np
from .embeddinggemma2 import ROOT, identity, digest, cached_model_path, normalized
from .embeddinggemma2_index import write_json
from .driveevidence_v2_planner import normalize
from .driveevidence_v5 import load_index, candidate_evidence, decide, DEFAULT
from .driveevidence_v5_planner import QueryPlanner,SCHEMA,REVISION
from .driveevidence_v5_worker import DriveEvidenceV5Worker

FEATURES=('EG2_margin','factor_confidence','object_reliability','spatial_reliability','relation_reliability','motion_reliability','track_visibility','evidence_coverage','SAFE_RERANK','KEEP_EG2','INSUFFICIENT_EVIDENCE','promotion_depth')
def drift_features(plan,diag):
    ev=diag.get('evidence',{});result={'EG2_margin':diag.get('EG2_margin'),'promotion_depth':(diag.get('proposal') or {}).get('rank_from',0)}
    ps=[f['probability'] for f in (plan or {}).get('factors',{}).values() if f.get('active') and f.get('probability') is not None]
    result['factor_confidence']=float(np.mean(ps)) if ps else None
    for name in ('object','spatial','relation','motion'):
        vals=[c[name]['reliability'] for c in ev.values() if c.get(name,{}).get('reliability') is not None]
        result[name+'_reliability']=float(np.mean(vals)) if vals else None
    vis=[c['spatial'].get('visibility') for c in ev.values() if c.get('spatial',{}).get('visibility') is not None]
    result['track_visibility']=float(np.mean(vis)) if vis else None
    cov=diag.get('coverage',{});result['evidence_coverage']=sum(v['reliable_candidates']>0 for v in cov.values())/max(sum(v['active'] for v in cov.values()),1)
    for state in ('SAFE_RERANK','KEEP_EG2','INSUFFICIENT_EVIDENCE'):result[state]=int(diag['state']==state)
    return result
def drift_profile(records):
    baseline={}
    for key in FEATURES:
        vals=[r[key] for r in records if r.get(key) is not None and np.isfinite(r[key])]
        baseline[key]={'count':len(vals),'mean':float(np.mean(vals)) if vals else None,'low':float(np.quantile(vals,.01)) if vals else None,'high':float(np.quantile(vals,.99)) if vals else None}
    return {'version':1,'baseline':baseline,'window':32,'outside_fraction':.5,'mean_tolerance':.3,'range_padding':.1,'calibration_samples':len(records),'offline_only':True}
class DriftMonitor:
    def __init__(self,profile=None):self.profile=profile or {};self.values=deque(maxlen=self.profile.get('window',32));self.alarm=False;self.reason=None
    def observe(self,record):
        self.values.append(record)
        if len(self.values)<self.values.maxlen:return self.alarm
        for key,b in self.profile.get('baseline',{}).items():
            vals=[r[key] for r in self.values if r.get(key) is not None]
            if b.get('count',0)<4 or len(vals)<self.values.maxlen//2:continue
            padding=self.profile.get('range_padding',.1);outside=sum(v<b['low']-padding or v>b['high']+padding for v in vals)/len(vals)
            if outside>self.profile.get('outside_fraction',.5) and abs(float(np.mean(vals))-b['mean'])>self.profile.get('mean_tolerance',.3):self.alarm=True;self.reason=key
        return self.alarm

class DriveEvidenceV5SearchEngine:
    def __init__(self,index_path,worker_factory=None,planner=None,*,deployment_mode='eg2_only',telemetry_path=None,log_raw_query=False,privacy_guard=None):
        if deployment_mode not in ('eg2_only','shadow'):raise ValueError('V5_SHADOW_BEFORE_ACTIVE_ROLLOUT')
        self.index_path=Path(index_path);self.worker_factory=worker_factory or DriveEvidenceV5Worker;self.planner=planner or QueryPlanner();self.mode=deployment_mode;self.lock=threading.RLock();self.loaded=self.worker=None;self.load_count=0;self.query_cache=OrderedDict();self.telemetry_path=Path(telemetry_path) if telemetry_path else None;self.log_raw_query=log_raw_query;self.privacy_guard=privacy_guard;self.startup_seconds=self.warmup_seconds=None;self.planner_failure=None
        self.health={'models_loaded':False,'index_loaded':False,'schema_valid':False,'hashes_valid':False,'privacy_ready':False,'warmup_complete':False,'ready':False,'service_state':'NOT_READY'}
    def load(self):
        with self.lock:
            if self.loaded is not None:return self
            start=time.perf_counter()
            try:
                self.loaded=load_index(self.index_path);self.load_count+=1
                self.health.update(index_loaded=True,schema_valid=True,hashes_valid=True)
                self.drift=DriftMonitor(self.loaded['meta'].get('drift_profile'))
                # Validation of neural weights at startup, never on normal requests.
                for path,h in self.loaded['meta'].get('model_hashes',{}).items():
                    if not (ROOT/path).is_file() and 'metadata_v4_laya/laya/weights/' in path.replace('\\','/'):
                        self.planner_failure='LAYA_WEIGHTS_UNAVAILABLE'
                        continue
                    if digest(ROOT/path)!=h:raise ValueError('V5_MODEL_HASH_MISMATCH')
                self.worker=self.worker_factory(self.loaded['meta']['config']);self.worker.load()
                if self.planner_failure is None:
                    try:self.planner.load()
                    except Exception as exc:self.planner_failure=str(exc)
                from src.privacy.media import private_full_media
                # Guard availability; missing rendered clips remain fail-closed at existing media gateway.
                privacy_ok=bool(self.privacy_guard()) if self.privacy_guard else callable(private_full_media)
                if not privacy_ok:raise ValueError('V5_PRIVACY_GUARD_UNAVAILABLE')
                self.health.update(models_loaded=self.planner_failure is None,privacy_ready=True)
                self.startup_seconds=time.perf_counter()-start;warm=time.perf_counter()
                self.worker.encode_text('a road scene with vehicles')
                if not self.planner_failure:self.planner.plan('a road scene with vehicles')
                self.warmup_seconds=time.perf_counter()-warm
                degraded=bool(self.loaded['evidence_fallback'] or self.planner_failure)
                self.health.update(warmup_complete=True,ready=True,service_state='DEGRADED_EG2_ONLY' if degraded else 'READY_EG2_ONLY' if self.mode=='eg2_only' else 'READY_SHADOW')
            except Exception:
                self.health.update(ready=False,service_state='NOT_READY');self.close();raise
            return self
    def cache_identity(self,query):
        # Laya normalizes its cache; exact EG2 input is separately bound so case/spacing
        # cache aliases cannot replace standalone EG2's literal query embedding.
        return identity({'query':normalize(query),'EG2_query_text':query,'EG2_config':self.loaded['meta']['config'],'Laya_revision':REVISION,'planner_schema':SCHEMA,'planner_questions':__import__('hybrid_v2.driveevidence_v5_planner',fromlist=['QUESTIONS']).QUESTIONS})
    def _query(self,query):
        key=self.cache_identity(query);saved=self.query_cache.get(key)
        if saved is not None:self.query_cache.move_to_end(key);return saved,{'cache_hit':True,'Laya_calls':0,'Laya':0.,'EG2_global_text':0.,'factor_text_encode':0.}
        global_v,gm=self.worker.encode_text(query);plan=None;pm={'Laya_calls':0,'seconds':0};vectors={};factorseconds=0.
        # Healthy EG2-only normal traffic needs no auxiliary inference. Shadow explicitly computes it.
        if self.mode=='shadow' and not self.planner_failure and not self.loaded['evidence_fallback']:
            plan,pm=self.planner.plan(query)
            for e in (plan or {}).get('entities',[]):
                v,m=self.worker.encode_text(e['object_phrase']);vectors[e['object_phrase']]=v;factorseconds+=m['seconds']
        saved={'global_vector':global_v,'plan':plan,'factor_vectors':vectors}
        self.query_cache[key]=saved
        while len(self.query_cache)>256:self.query_cache.popitem(last=False)
        return saved,{'cache_hit':False,'Laya_calls':pm['Laya_calls'],'Laya':pm['seconds'],'EG2_global_text':gm['seconds'],'factor_text_encode':factorseconds}
    def search_table(self,query,ranker='driveevidence_v5',decision_provider=None,**unused):
        if ranker!='driveevidence_v5' or not isinstance(query,str) or not query.strip() or len(query)>1000 or decision_provider not in (None,'none','laya'):raise ValueError('INVALID_V5_QUERY')
        with self.lock:
            self.load()
            if not self.health['ready']:raise ValueError('V5_NOT_READY')
            start=time.perf_counter();data,timing=self._query(query);cfg=self.loaded['meta']['architecture'];begun=time.perf_counter()
            vector=normalized(data['global_vector']);scores=self.loaded['global_vectors']@vector;base=np.argsort(-scores,kind='stable');timing['candidate_search']=time.perf_counter()-begun;begun=time.perf_counter();ev={};optional_failure=bool(self.loaded['evidence_fallback']);plan=data['plan']
            if self.mode=='shadow' and plan and not optional_failure:
                try:
                    for row in base[:cfg['depth']]:ev[int(row)]=candidate_evidence(self.loaded,int(row),plan,vector,data['factor_vectors'],cfg)
                except (ValueError,KeyError,TypeError,IndexError):ev={};optional_failure=True
            timing['evidence_fetch_assignment_reliability']=time.perf_counter()-begun;begun=time.perf_counter()
            shadow,d=decide(scores,ev,plan,cfg,self.loaded['meta']['policy'],self.drift.alarm,optional_failure)
            if self.mode=='eg2_only':d.update(state='KEEP_EG2',reason_code='CALIBRATED_RISK_REJECT',swap=False)
            features=drift_features(plan,d);alarm=self.drift.observe(features)
            if alarm:d.update(state='KEEP_EG2',reason_code='DRIFT_FALLBACK',swap=False);shadow=base;self.health['service_state']='DEGRADED_EG2_ONLY'
            timing['risk_controller']=time.perf_counter()-begun;begun=time.perf_counter();candidates=[]
            # Shadow never changes user-visible ordering, even for qualified offline policy.
            for n,j in enumerate(base,1):
                r=self.loaded['rows'][int(j)];candidates.append({**{k:r[k] for k in ('clip_id','source_row','source_clip_id','segment_id','start_sec','end_sec','day_night')},'row':int(j),'rank':n,'ranker':ranker,'score':float(scores[j]),'final_score':float(scores[j])})
            timing['serialization']=time.perf_counter()-begun;timing['total']=time.perf_counter()-start
            record={'request_id':str(uuid.uuid4()),'index_version':self.loaded['meta']['policy_sha256'],'model_versions':self.loaded['meta']['model_versions'],'query_hash':identity(normalize(query)),'candidate_rows':[int(i) for i in base[:10]],'EG2_scores':[float(scores[i]) for i in base[:10]],'factor_plan':plan,'evidence_availability':d.get('coverage',{}),'evidence_reliability':{str(row):{f:c.get('reliability') for f,c in channels.items()} for row,channels in ev.items()},'decision_state':d['state'],'reason_code':d['reason_code'],'shadow_promotion':d.get('promotion'),'shadow_proposal':d.get('proposal'),'user_ranking':'EG2','latency_stages':timing,'fallback_flags':{'drift':alarm,'planner':bool(self.planner_failure),'optional_evidence':optional_failure},'distributions':features}
            # Default telemetry omits query-derived free text (including object phrases).
            if not self.log_raw_query:
                record['factor_plan']=None if not plan else {'factors':plan['factors'],'typed_probabilities':{k:{'probability':v['probability'],'confidence':v['confidence']} for k,v in plan['typed_fields'].items()}}
            else:record['query']=query
            if self.telemetry_path:
                self.telemetry_path.parent.mkdir(parents=True,exist_ok=True)
                with self.telemetry_path.open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            return {'query':query,'format':'driveevidence-v5-candidates','index':str(self.index_path),'candidates':candidates,'decision':{'state':d['state'],'reason_code':d['reason_code'],'deployment_mode':self.mode,'user_ranking':'EG2','not_found_enabled':False},'retrieval':{'planner':{'Laya_calls':timing['Laya_calls'],'cache_hit':timing['cache_hit']},'diagnostic':d,'warm_seconds':timing['total'],'timing':timing,'engine_load_count':self.load_count,'Qwen_calls':0,'candidate_level_Laya_calls':0,'service_state':self.health['service_state'],'request_id':record['request_id']}}
    def close(self):
        if self.worker:self.worker.close()
        self.planner.close();self.worker=self.loaded=None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except ImportError:pass
        self.health.update(models_loaded=False,index_loaded=False,ready=False,service_state='NOT_READY')
