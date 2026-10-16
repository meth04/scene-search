"""Query-only Laya planning, exactly one typed runtime call per cache miss."""
from __future__ import annotations
import json, math, threading, time, unicodedata, re
from pathlib import Path
from .embeddinggemma2 import identity, ROOT
from .embeddinggemma2_index import write_json
from .decision_provider import LayaLocalDecisionProvider, DecisionConfig, MODEL, REVISION
SCHEMA_VERSION='driveevidence-query-v2.2'
def choice(instruction,values):
    return {'type':'choice','instructions':instruction+' Extract from the query only. Use unknown when unspecified. Do not infer facts about videos.','criteria':{str(i):v for i,v in enumerate(values)}}
QUESTIONS={
 'query_type':choice('What is the main evidence requirement?', ['scene_context','vehicle','people','infrastructure','entity_relation','action_temporal','mixed']),
 'entity':choice('What subject does the query request?', ['unknown','vehicle','car','truck','bus','motorcycle','bicycle','person','traffic light','stop sign']),
 'subtype':choice('What vehicle body type is explicitly requested?', ['unknown','hatchback','sedan','SUV','van','pickup']),
 'color':choice('What subject color is explicitly requested?', ['unknown','white','black','red','blue','green','yellow','silver','gray','brown']),
 'side':choice('Which image side is explicitly requested?', ['unknown','left','right','center']),
 'depth':choice('What depth relation is explicitly requested?', ['unknown','ahead','behind','near']),
 'reference':choice('What is the reference for the spatial relation?', ['unknown','ego','entity']),
 'action':choice('What action is explicitly requested?', ['unknown','crossing','approaching','moving_away','overtaking','passing','moving','stationary','appearing','disappearing']),
 'motion_direction':choice('What direction is explicitly requested?', ['unknown','left_to_right','right_to_left','toward','away','lateral']),
 'requires_motion':choice('Does satisfying this query require observing movement over time?', ['false','true']),
 'requires_object_detail':choice('Does satisfying this query require identifying a specific object or its appearance?', ['false','true']),
 'requires_spatial_relation':choice('Does satisfying this query require a spatial relation or side?', ['false','true']),
 'requires_infrastructure_detail':choice('Does satisfying this query require infrastructure such as a sign or traffic light?', ['false','true']),
}
def normalize(query):
    if not isinstance(query,str) or not query.strip(): raise ValueError('INVALID_QUERY')
    return ' '.join(unicodedata.normalize('NFKC',query).casefold().split())
def parse(raw):
    values={}
    for key,q in QUESTIONS.items():
        a=raw['answers'][key]; probabilities=a['probabilities']; selected=str(a['choice'])
        if set(probabilities)!=set(q['criteria']) or selected not in q['criteria']: raise ValueError('INVALID_LAYA_SCHEMA')
        p=[float(x) for x in probabilities.values()]
        if any(not math.isfinite(x) or not 0<=x<=1 for x in p) or abs(sum(p)-1)>.03: raise ValueError('INVALID_LAYA_PROBABILITIES')
        values[key]=q['criteria'][selected]
    ent={'type':values.pop('entity'),'subtype':None if values['subtype']=='unknown' else values['subtype'],'color':None if values['color']=='unknown' else values['color']}
    values.pop('subtype'); values.pop('color')
    values['entities']=[] if ent['type']=='unknown' else [ent]
    values['spatial']={k:values.pop(k) for k in ('side','depth','reference')}
    values['action']=None if values['action']=='unknown' else values['action']
    for k in ('requires_motion','requires_object_detail','requires_spatial_relation','requires_infrastructure_detail'): values[k]=values[k]=='true'
    return validate(values)
def validate(plan):
    expected={'query_type','entities','spatial','action','motion_direction','requires_motion','requires_object_detail','requires_spatial_relation','requires_infrastructure_detail'}
    if not isinstance(plan,dict) or set(plan)!=expected: raise ValueError('INVALID_PLAN')
    for k in ('query_type','motion_direction'):
        if plan[k] not in QUESTIONS[k]['criteria'].values(): raise ValueError('INVALID_PLAN_ENUM')
    if plan['action'] is not None and plan['action'] not in QUESTIONS['action']['criteria'].values(): raise ValueError('INVALID_ACTION')
    if not isinstance(plan['entities'],list) or len(plan['entities'])>1: raise ValueError('INVALID_ENTITY')
    for e in plan['entities']:
        if set(e)!={'type','subtype','color'} or e['type'] not in QUESTIONS['entity']['criteria'].values(): raise ValueError('INVALID_ENTITY')
        for k in ('subtype','color'):
            if e[k] is not None and e[k] not in QUESTIONS[k]['criteria'].values(): raise ValueError('INVALID_ATTRIBUTE')
    if set(plan['spatial'])!={'side','depth','reference'}: raise ValueError('INVALID_SPATIAL')
    for k,v in plan['spatial'].items():
        if v not in QUESTIONS[k]['criteria'].values(): raise ValueError('INVALID_SPATIAL')
    for k in expected:
        if k.startswith('requires_') and type(plan[k]) is not bool: raise ValueError('INVALID_FLAG')
    return plan
def ground(plan,query):
    """Discard ungrounded model fields; never synthesize replacement planner choices."""
    plan=json.loads(json.dumps(validate(plan))); q=normalize(query)
    aliases={'person':r'person|people|pedestrian|man|woman|boy|girl','vehicle':r'vehicle|car|bus|truck|van|hatchback|sedan|suv|rickshaw',
             'car':r'car|hatchback|sedan|suv|van','motorcycle':r'motorcycle|motorcyclist|motorbike','bicycle':r'bicycle|cyclist|bike',
             'truck':r'truck|tanker','bus':r'bus','traffic light':r'traffic light|signal','stop sign':r'stop sign'}
    entities=[]
    for e in plan['entities']:
        if not re.search(r'\b(?:'+aliases.get(e['type'],re.escape(e['type']))+r')\b',q): continue
        for k in ('color','subtype'):
            if e[k] is not None and not re.search(r'\b'+re.escape(e[k].casefold())+r'\b',q): e[k]=None
        entities.append(e)
    plan['entities']=entities
    action_alias={'crossing':r'cross\w*','approaching':r'approach\w*|oncoming','moving_away':r'away|reced\w*','overtaking':r'overtak\w*','passing':r'pass\w*','moving':r'mov\w*|rid\w*|driv\w*|walk\w*','stationary':r'stationary|stopped|parked','appearing':r'appear\w*|enter\w*','disappearing':r'disappear\w*|exit\w*'}
    if plan['action'] and not re.search(r'\b(?:'+action_alias[plan['action']]+r')\b',q): plan['action']=None
    dirs={'left_to_right':r'left to right','right_to_left':r'right to left','toward':r'toward|approach\w*|oncoming','away':r'away|reced\w*','lateral':r'lateral|across|cross\w*'}
    if plan['motion_direction']!='unknown' and not re.search(dirs[plan['motion_direction']],q): plan['motion_direction']='unknown'
    plan['requires_motion']=plan['requires_motion'] and (plan['action'] not in (None,'stationary') or plan['motion_direction']!='unknown')
    for k in ('side','depth'):
        v=plan['spatial'][k]
        if v!='unknown' and not re.search(r'\b'+v+r'\b',q): plan['spatial'][k]='unknown'
    plan['requires_spatial_relation']=plan['requires_spatial_relation'] and any(plan['spatial'][k]!='unknown' for k in ('side','depth'))
    plan['requires_object_detail']=plan['requires_object_detail'] and bool(entities)
    plan['requires_infrastructure_detail']=plan['requires_infrastructure_detail'] and any(e['type'] in ('traffic light','stop sign') for e in entities)
    return validate(plan)
class QueryPlanner:
    def __init__(self,cache=None,provider=None):
        self.cache=Path(cache or ROOT/'outputs/driveevidence_v2/query_planner_cache'); self.provider=provider
        self.lock=threading.RLock(); self.calls=0; self.cache_hits=0
    def cached(self,query):
        q=normalize(query); key=identity(q); p=self.cache/(key+'.json')
        if not p.exists(): return None
        try:
            v=json.loads(p.read_text(encoding='utf-8'))
            if v['normalized_query']!=q or v['query_hash']!=key or v['model_revision']!=REVISION or v['schema_version']!=SCHEMA_VERSION or v['prompt_sha256']!=identity(QUESTIONS): return None
            if v['parsed_schema'] is not None: validate(v['parsed_schema'])
            return v
        except (ValueError,KeyError,TypeError): return None
    def plan(self,query):
        with self.lock:
            start=time.perf_counter(); saved=self.cached(query)
            if saved is not None:
                self.cache_hits+=1
                return saved['parsed_schema'],{'cache_hit':True,'Laya_calls':0,'seconds':time.perf_counter()-start,'failure':saved.get('failure')}
            q=normalize(query); raw=None; plan=None; failure=None; calls_before=self.calls
            try:
                if self.provider is None: self.provider=LayaLocalDecisionProvider(DecisionConfig(provider='laya',device='cuda',cache_entries=0))
                self.provider.load()
                if isinstance(self.provider,LayaLocalDecisionProvider) and not self.provider.provenance.get('revision_verified'):
                    raise ValueError('LAYA_PINNED_REVISION_NOT_VERIFIED')
                import torch
                if torch.cuda.device_count()!=1 or 'GTX 1650' not in torch.cuda.get_device_name(0): raise ValueError('GTX1650_ONLY')
                self.provider.agent.model.requires_grad_(False)
                self.calls+=1
                with torch.inference_mode():
                    raw=self.provider.agent.predict_batch([json.dumps({'query':q},ensure_ascii=False)],QUESTIONS,batch_size=1,max_len=512,head_max_len=192)[0]
                if raw.get('usage',{}).get('truncated'): raise ValueError('TRUNCATED_QUERY_PLAN')
                plan=ground(parse(raw),q)
            except Exception as exc: failure=type(exc).__name__+': '+str(exc)
            seconds=time.perf_counter()-start
            calls=self.calls-calls_before
            write_json(self.cache/(identity(q)+'.json'),{'normalized_query':q,'query_hash':identity(q),'raw_output':raw,'parsed_schema':plan,'model':MODEL,'model_revision':REVISION,'schema_version':SCHEMA_VERSION,'prompt_sha256':identity(QUESTIONS),'failure':failure,'seconds':seconds,'candidate_level_calls':0,'Laya_calls':calls})
            return plan,{'cache_hit':False,'Laya_calls':calls,'seconds':seconds,'failure':failure}
    def close(self):
        if self.provider and self.provider.agent:
            self.provider.agent.__exit__(None,None,None); self.provider.agent=None
