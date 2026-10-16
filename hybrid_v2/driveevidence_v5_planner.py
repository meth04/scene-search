"""One frozen, persistent, query-only Laya call; genuine typed confidence retained."""
from __future__ import annotations
import copy, json, math, re, threading, time
from pathlib import Path
from .embeddinggemma2 import ROOT, identity
from .embeddinggemma2_index import write_json
from .driveevidence_v2_planner import normalize, choice
from .driveevidence_v3_planner import QUESTIONS as OLD_QUESTIONS, parse as old_parse, ALIASES
from .decision_provider import LayaLocalDecisionProvider, DecisionConfig, MODEL, REVISION
SCHEMA = 'driveevidence-query-v5.1'
QUESTIONS = {**OLD_QUESTIONS,
 'second_entity': choice('Identify the explicitly requested second distinct reference entity. Unknown if the query only describes image side or ego. Do not repeat the primary subject.', ['unknown','vehicle','car','truck','bus','motorcycle','bicycle','person','traffic light','sign']),
 'second_color': choice('Identify only the explicitly stated color of the second reference entity; unknown if no second entity or no color.', ['unknown','white','black','red','blue','green','yellow','silver','gray','brown','orange']),
 'relation': choice('Identify the explicitly requested primary-to-second entity relation, not image-side location. Left/right mean left/right OF that second entity. Front/behind physical depth cannot be certified by image position.', ['unknown','left_of','right_of','beside','front_of','behind'])}
def typed(raw, key):
    try:
        a=raw['answers'][key];p=a['probabilities'];c=str(a['choice']);head=QUESTIONS[key]
        if set(p)!=set(head['criteria']) or c not in p:raise ValueError('schema')
        vals=[float(v) for v in p.values()]
        if any(not math.isfinite(v) or not 0<=v<=1 for v in vals) or abs(sum(vals)-1)>.03:raise ValueError('probabilities')
        # Selected probability and entropy confidence are different real API fields.
        prob=float(p[c]);conf=a.get('confidence');conf=float(conf) if conf is not None and math.isfinite(float(conf)) and 0<=float(conf)<=1 else None
        return {'value':head['criteria'][c],'probability':prob,'confidence':conf,'probabilities':dict(p),'provenance':'real_typed_Laya_output'}
    except (KeyError,ValueError,TypeError):return {'value':'unknown','probability':None,'confidence':None,'probabilities':None,'provenance':'UNKNOWN'}
def uncertainty(active, p, threshold=.7):
    if not active:return 'INACTIVE'
    if p is None:return 'UNKNOWN'
    return 'ACTIVE_RELIABLE' if p>=threshold else 'ACTIVE_UNCERTAIN'
def parse(raw, query):
    plan=old_parse(raw,query);q=normalize(query);fields={k:typed(raw,k) for k in QUESTIONS}
    # Old parser grounds independently; never credit confidence from discarded fields.
    for k in plan['fallback_fields']:fields[k]={**fields[k],'probability':None,'confidence':None,'grounding_failed':True}
    for e in plan['entities']:
        e['confidence']=fields['entity']['probability'];e['factor_probabilities']={k:fields[k] for k in ('entity','subtype','color','appearance')}
    relation=fields['relation']['value'];second=fields['second_entity']['value']
    relpatterns={'left_of':r'\bleft\s+(?:side\s+)?of\b','right_of':r'\bright\s+(?:side\s+)?of\b','beside':r'\bbeside\b|\bnext to\b','front_of':r'\bin front of\b','behind':r'\bbehind\b'}
    graph=None
    if plan['entities'] and relation in relpatterns and re.search(relpatterns[relation],q) and second!='unknown':
        tail=re.split(relpatterns[relation],q,maxsplit=1)[-1]
        if re.search(r'\b(?:'+ALIASES.get(second,re.escape(second))+r')\b',tail):
            color=fields['second_color']['value']
            if color!='unknown' and not re.search(r'\b'+re.escape(color)+r'\b',tail):color='unknown';fields['second_color']['probability']=None
            e={'type':second,'subtype':None,'color':None if color=='unknown' else color,'appearance_terms':[],'object_phrase':('' if color=='unknown' else color+' ')+second,'confidence':fields['second_entity']['probability'],'factor_probabilities':{'entity':fields['second_entity'],'color':fields['second_color']}}
            plan['entities'].append(e)
            graph={'entities':[0,1],'relation':relation,'probability':fields['relation']['probability'],'reference_probability':fields['second_entity']['probability'],'distinct_required':True,'coordinate_scope':'image-plane; physical front/behind UNKNOWN'}
    # Bare side vs temporal direction must be grounded separately.
    side=plan['spatial']['side']
    direction_only=bool(re.search(r'\b(?:left to right|right to left)\b',q))
    if direction_only and not re.search(r'\bon (?:the )?(?:left|right)\b',q):
        plan['spatial']['side']='unknown';fields['side']['probability']=None
    if graph:plan['spatial']['reference']='entity'
    elif plan['spatial']['reference']=='entity':plan['spatial']['reference']='unknown';fields['reference']['probability']=None
    elif plan['spatial']['side']!='unknown':plan['spatial']['reference']='ego'
    plan['relation_graph']=graph;plan['typed_fields']=fields;plan['schema']=SCHEMA
    probabilities={
      'object':fields['entity']['probability'] if plan['entities'] else None,
      'spatial':fields['side']['probability'] if not graph and plan['spatial']['side']!='unknown' else None,
      'relation':min(graph['probability'],graph['reference_probability']) if graph and graph['probability'] is not None and graph['reference_probability'] is not None else None,
      'motion':fields['motion_direction']['probability'] if plan['motion_direction']!='unknown' else fields['action']['probability'] if plan['action'] else None,
      'scene':fields['scene']['probability'] if plan['scene_terms'] else None}
    active={'object':bool(plan['entities']),'spatial':plan['spatial']['side']!='unknown' and not graph,'relation':bool(graph),'motion':plan['requires_motion'],'scene':plan['requires_scene_context']}
    plan['factors']={k:{'active':bool(active[k]),'probability':v,'state':uncertainty(active[k],v)} for k,v in probabilities.items()}
    return plan

class QueryPlanner:
    def __init__(self,cache=None,provider=None,device='cuda'):
        self.cache=Path(cache or ROOT/'outputs/driveevidence_v5/query_planner_cache');self.provider=provider;self.device=device;self.lock=threading.RLock();self.calls=0
    def cached(self,query):
        try:
            d=json.loads((self.cache/(identity(normalize(query))+'.json')).read_text(encoding='utf-8'))
            if d['model_revision']!=REVISION or d['schema']!=SCHEMA or d['questions_sha256']!=identity(QUESTIONS) or d['normalized_query']!=normalize(query):return None
            return d
        except (OSError,KeyError,ValueError,TypeError):return None
    def load(self):
        if self.provider is None:self.provider=LayaLocalDecisionProvider(DecisionConfig(provider='laya',device=self.device,cache_entries=0))
        self.provider.load()
        if isinstance(self.provider,LayaLocalDecisionProvider) and not self.provider.provenance.get('revision_verified'):raise ValueError('LAYA_MODEL_HASH_MISMATCH')
        import torch
        if torch.cuda.device_count()!=1 or 'GTX 1650' not in torch.cuda.get_device_name(0):raise ValueError('GTX1650_ONLY')
        self.provider.agent.model.requires_grad_(False);self.provider.agent.model.eval()
        return self
    def plan(self,query):
        with self.lock:
            start=time.perf_counter();saved=self.cached(query)
            if saved is not None:return saved['plan'],{'cache_hit':True,'Laya_calls':0,'seconds':time.perf_counter()-start,'failure':saved.get('failure')}
            raw=plan=None;calls=0;failure=None
            try:
                self.load();import torch
                calls=1;self.calls+=1
                with torch.inference_mode():raw=self.provider.agent.predict_batch([json.dumps({'query':normalize(query)},ensure_ascii=False)],QUESTIONS,batch_size=1,max_len=512,head_max_len=192)[0]
                plan=parse(raw,query)
            except Exception as exc:failure=type(exc).__name__+': '+str(exc)
            write_json(self.cache/(identity(normalize(query))+'.json'),{'normalized_query':normalize(query),'model':MODEL,'model_revision':REVISION,'schema':SCHEMA,'questions_sha256':identity(QUESTIONS),'plan':plan,'raw_output':raw,'failure':failure,'Laya_calls':calls,'candidate_level_calls':0,'time_unix':time.time()})
            return plan,{'cache_hit':False,'Laya_calls':calls,'seconds':time.perf_counter()-start,'failure':failure}
    def close(self):
        if self.provider and self.provider.agent:self.provider.agent.__exit__(None,None,None);self.provider.agent=None
