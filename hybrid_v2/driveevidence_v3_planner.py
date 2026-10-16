"""Frozen query-only Laya factors; independent validation and cache identity."""
from __future__ import annotations
import json,math,re,threading,time
from pathlib import Path
from .embeddinggemma2 import ROOT,identity
from .embeddinggemma2_index import write_json
from .driveevidence_v2_planner import normalize,choice
from .decision_provider import LayaLocalDecisionProvider,DecisionConfig,MODEL,REVISION
SCHEMA_VERSION='driveevidence-query-v3.1'
QUESTIONS={
 'entity':choice('Identify the primary requested subject, independently of its action or location. A cyclist is bicycle; a scooter rider is motorcycle.', ['unknown','vehicle','car','truck','bus','motorcycle','bicycle','person','traffic light','stop sign','sign','bridge','stall']),
 'subtype':choice('Identify the explicitly requested subject body type.', ['unknown','hatchback','sedan','SUV','van','pickup','tanker','rickshaw','goods truck','scooter','cyclist']),
 'color':choice('Identify the primary subject color or its clothing color, not a background color.', ['unknown','white','black','red','blue','green','yellow','silver','gray','brown','orange','bright','pale']),
 'appearance':choice('Identify the primary subject distinctive appearance.', ['unknown','rear spare wheel','covered','loaded','open bed','decorated','small','tall']),
 'action':choice('Identify the requested primary subject action independently of its type. Camera passing still requires temporal evidence.', ['unknown','crossing','approaching','moving_away','overtaking','passing','moving','stationary','appearing','disappearing']),
 'motion_direction':choice('Identify explicitly requested primary subject image motion, independently of action.', ['unknown','left_to_right','right_to_left','toward','away','lateral']),
 'side':choice('Identify primary subject image side, independently of action.', ['unknown','left','right','center']),
 'depth':choice('Identify primary subject depth relation.', ['unknown','ahead','behind','near']),
 'reference':choice('Is the requested side relative to image/ego or explicitly to another entity?', ['unknown','ego','entity']),
 'scene':choice('Identify independently requested scene context.', ['unknown','day','night','dusk','highway','rural road','junction','street']),
}
ALIASES={'vehicle':r'vehicle|car|truck|bus|van|hatchback|suv|rickshaw','car':r'car|hatchback|sedan|suv|van','motorcycle':r'motorcycl\w*|motorbike|scooter|rider','bicycle':r'bicycle|cyclist','person':r'person|people|pedestrian|man|woman|rider|clothing','truck':r'truck|tanker','sign':r'sign|chevron','bridge':r'bridge|viaduct','stall':r'stall|shop','traffic light':r'traffic light|signal'}
ACTION={'crossing':r'cross\w*','approaching':r'approach\w*|oncoming','moving_away':r'away|reced\w*','overtaking':r'overtak\w*','passing':r'pass\w*','moving':r'mov\w*|rid\w*|driv\w*|walk\w*|falls','stationary':r'stationary|stopped|parked|standing','appearing':r'appear\w*|enter\w*','disappearing':r'disappear\w*|exit\w*|leaves view'}
DIRS={'left_to_right':r'left to right','right_to_left':r'right to left','toward':r'toward|approach\w*|oncoming|foreground','away':r'away|reced\w*','lateral':r'lateral|across|cross\w*|edge'}
def parse(raw,query):
    q=normalize(query);values={};fallback={}
    for key,head in QUESTIONS.items():
        try:
            a=raw['answers'][key];p=a['probabilities'];selected=str(a['choice'])
            if set(p)!=set(head['criteria']) or selected not in head['criteria']:raise ValueError('schema')
            probabilities=[float(x) for x in p.values()]
            if any(not math.isfinite(x) or not 0<=x<=1 for x in probabilities) or abs(sum(probabilities)-1)>.03:raise ValueError('probabilities')
            v=head['criteria'][selected]
            aliases=ALIASES if key=='entity' else ACTION if key=='action' else DIRS if key=='motion_direction' else {}
            pattern=aliases.get(v,re.escape(v.casefold()))
            if key!='reference' and v!='unknown' and not re.search(r'\b(?:'+pattern+r')\b',q):
                fallback[key]='ungrounded';v='unknown'
            values[key]=v
        except (KeyError,ValueError,TypeError):values[key]='unknown';fallback[key]='invalid_field'
    entity=values['entity'];ent=[]
    if entity!='unknown':
        subtype=values['subtype'];color=values['color'];terms=[] if values['appearance']=='unknown' else [values['appearance']]
        compatible={'car':{'hatchback','sedan','SUV','van','pickup'},'truck':{'pickup','tanker','goods truck'},'motorcycle':{'scooter'},'bicycle':{'cyclist'},'vehicle':set(QUESTIONS['subtype']['criteria'].values())}
        if subtype!='unknown' and subtype not in compatible.get(entity,set()):
            subtype='unknown';fallback['subtype']='incompatible_entity_attribute'
        noun=subtype if subtype!='unknown' else entity
        phrase=' '.join(([color] if color!='unknown' else [])+terms+[noun])
        ent=[{'type':entity,'subtype':None if subtype=='unknown' else subtype,'color':None if color=='unknown' else color,'appearance_terms':terms,'object_phrase':phrase}]
    plan={'entities':ent,'action':None if values['action']=='unknown' else values['action'],'motion_direction':values['motion_direction'],
      'spatial':{k:values[k] for k in ('side','depth','reference')},'scene_terms':[] if values['scene']=='unknown' else [values['scene']],
      'fallback_fields':fallback}
    plan['requires_object_detail']=bool(ent)
    plan['requires_motion']=bool(plan['action'] or plan['motion_direction']!='unknown')
    plan['requires_spatial_relation']=values['side']!='unknown' or values['depth']!='unknown'
    plan['requires_scene_context']=bool(plan['scene_terms'])
    return validate(plan)
def validate(plan):
    if not isinstance(plan,dict) or 'query_type' in plan:raise ValueError('INVALID_FACTORIZED_PLAN')
    for flag in ('requires_object_detail','requires_motion','requires_spatial_relation','requires_scene_context'):
        if type(plan[flag]) is not bool:raise ValueError('INVALID_FLAG')
    for e in plan['entities']:
        if e['type'] not in QUESTIONS['entity']['criteria'].values() or not isinstance(e['object_phrase'],str):raise ValueError('INVALID_ENTITY')
    if plan['motion_direction'] not in QUESTIONS['motion_direction']['criteria'].values():raise ValueError('INVALID_DIRECTION')
    if plan['action'] is not None and plan['action'] not in QUESTIONS['action']['criteria'].values():raise ValueError('INVALID_ACTION')
    for k in ('side','depth','reference'):
        if plan['spatial'][k] not in QUESTIONS[k]['criteria'].values():raise ValueError('INVALID_SPATIAL')
    return plan
def object_queries(plan):
    validate(plan)
    if not plan['entities']:return {}
    e=plan['entities'][0];a=e['subtype'] or e['type'];b=e['object_phrase'];side=plan['spatial']['side']
    return {'A':a,'B':b,'C':b+(' on the '+side if side in ('left','right','center') else '')}
class QueryPlanner:
    def __init__(self,cache=None,provider=None):
        self.cache=Path(cache or ROOT/'outputs/driveevidence_v3/query_planner_cache');self.provider=provider;self.lock=threading.RLock();self.calls=0
    def cached(self,query):
        q=normalize(query);p=self.cache/(identity(q)+'.json')
        try:
            v=json.loads(p.read_text(encoding='utf-8'))
            if v['normalized_query']!=q or v['model_revision']!=REVISION or v['schema_version']!=SCHEMA_VERSION or v['prompt_sha256']!=identity(QUESTIONS):return None
            if v['parsed_schema'] is not None:validate(v['parsed_schema'])
            return v
        except (OSError,ValueError,KeyError,TypeError):return None
    def plan(self,query):
        with self.lock:
            start=time.perf_counter();saved=self.cached(query)
            if saved is not None:return saved['parsed_schema'],{'cache_hit':True,'Laya_calls':0,'seconds':time.perf_counter()-start,'failure':saved.get('failure')}
            q=normalize(query);raw=plan=None;failure=None;calls=0
            try:
                if self.provider is None:self.provider=LayaLocalDecisionProvider(DecisionConfig(provider='laya',device='cuda',cache_entries=0))
                self.provider.load()
                if isinstance(self.provider,LayaLocalDecisionProvider) and not self.provider.provenance.get('revision_verified'):raise ValueError('PINNED_REVISION_REQUIRED')
                import torch
                if torch.cuda.device_count()!=1 or 'GTX 1650' not in torch.cuda.get_device_name(0):raise ValueError('GTX1650_ONLY')
                self.provider.agent.model.requires_grad_(False);calls=1;self.calls+=1
                with torch.inference_mode():raw=self.provider.agent.predict_batch([json.dumps({'query':q},ensure_ascii=False)],QUESTIONS,batch_size=1,max_len=512,head_max_len=192)[0]
                plan=parse(raw,q)
            except Exception as exc:failure=type(exc).__name__+': '+str(exc)
            elapsed=time.perf_counter()-start
            write_json(self.cache/(identity(q)+'.json'),{'normalized_query':q,'query_hash':identity(q),'raw_output':raw,'parsed_schema':plan,'model':MODEL,'model_revision':REVISION,'schema_version':SCHEMA_VERSION,'prompt_sha256':identity(QUESTIONS),'failure':failure,'seconds':elapsed,'Laya_calls':calls,'candidate_level_calls':0})
            return plan,{'cache_hit':False,'Laya_calls':calls,'seconds':elapsed,'failure':failure}
    def close(self):
        if self.provider and self.provider.agent:self.provider.agent.__exit__(None,None,None);self.provider.agent=None
