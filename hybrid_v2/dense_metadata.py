"""Versioned grounded metadata, normalization and field-specific searchable text."""
from __future__ import annotations

import copy
import hashlib
import json
import re

from .common import clean_text

DENSE_FORMAT = "driveretriever-local-hybrid-dense-v1"
METADATA_VERSION = "dense_metadata_v1"
NORMALIZATION_VERSION = "dense_normalization_v1"
PROMPT_VERSION = "qwen2b_dense_metadata_v1"
FIELDS = ("scene", "people", "vehicle", "action", "relation", "infrastructure", "salient", "full")
BGE_RECIPE = {"model":"BAAI/bge-base-en-v1.5", "tokenizer":"AutoTokenizer", "encoder":"AutoModel",
              "pooling":"last_hidden_state[:,0]", "dtype":"float32", "normalization":"L2", "max_length":128}
ENV_KEYS = ("scene_type","road_type","visible_lane_count","weather","lighting","road_condition","traffic_density","visibility")
EGO_KEYS = ("action","lane","direction","state")
VEHICLE_KEYS = ("type","color","relative_side","relative_longitudinal_position","lane_relation","state","action","position_start","position_end","temporal_change")
PEOPLE_KEYS = ("appearance","upper_clothing_color","lower_clothing_color","relative_position","location_type","activity","state","near")
CYCLIST_KEYS = ("type","color","relative_position","action","direction","lane","state")
MOTORCYCLE_KEYS = ("color","relative_position","action","direction","lane","state")
EVENT_KEYS = ("type","actor","target","side","direction","evidence","confidence")
INFRA_KEYS = ("traffic_lights","crosswalks","bus_stops","bike_lanes","road_signs","lane_markings","parking",
              "intersections","roundabouts","tunnels","bridges","construction_barriers","bus_lanes")
STRING_LISTS = ("objects","spatial_relations","salient_facts","other_actions","relations","road_context","traffic_control","events")
ENTITY_GROUPS = {"vehicles":VEHICLE_KEYS,"people":PEOPLE_KEYS,"cyclists":CYCLIST_KEYS,"motorcycles":MOTORCYCLE_KEYS,"temporal_events":EVENT_KEYS}
SCHEMA = {"description":"", "scene_summary":"", "environment":dict.fromkeys(ENV_KEYS,""), "ego_vehicle":dict.fromkeys(EGO_KEYS,"")}
SCHEMA.update({group:[dict.fromkeys(keys,"")] for group,keys in ENTITY_GROUPS.items()})
SCHEMA["traffic_infrastructure"] = {key:[] for key in INFRA_KEYS}
SCHEMA.update({key:[] for key in STRING_LISTS})
SCHEMA.update(weather="",queries=[])

# No populated scene answers. Category names describe what to inspect, not what
# is known to occur in any particular video.
# The 2B model cannot reliably emit the full schema in one request. Three
# neutral domain passes see the same chronological sheet; their disjoint
# fields are merged without generating new facts or using search queries.
PASS_KEYS = (("description","scene_summary","environment","ego_vehicle","road_context","weather"),
             ("vehicles","cyclists","motorcycles","temporal_events","other_actions","events"),
             ("people","traffic_infrastructure","objects","spatial_relations","relations","traffic_control","salient_facts","queries"))
BASE_INSTRUCTION = """Read ONE chronological dashcam clip, left to right across the top row then the bottom row.
Ego is the camera vehicle; its hood is not another road user. Track each actor
across time, never one entry per image. Extract distinct visually supported facts.
Do not invent details, identities, occupations, brands, geography, sign text or
intent. Omit uncertainty and negative filler. Colors must be conservative.
Compare relative positions to distinguish overtaking ego from being overtaken
by ego. A temporal event requires multiple views. Standing does not prove waiting.
Return ONE complete compact JSON object. Short phrases only, under 20 words per
string. Do not repeat sentences or entities. At most five salient entities per
category. Omit unknown/empty keys. Empty template values indicate types, not
scene answers. Do not mention numbered frames or contact sheet artifacts.
Extract ONLY the requested fields in this empty template:
"""
DENSE_PROMPTS = tuple(BASE_INSTRUCTION+json.dumps({k:SCHEMA[k] for k in keys},separators=(",",":")) for keys in PASS_KEYS)
DENSE_PROMPT = "\n\n".join(DENSE_PROMPTS)
PROMPT_SHA256 = hashlib.sha256(DENSE_PROMPT.encode()).hexdigest()
ARTIFACT = re.compile(r"\bt\d+(?:/\d+)?\b|\b(?:frame|tile|image)\s*\d+\b|contact[- ]sheet|\b(?:schema|prompt example|placeholder)\b",re.I)
UNCERTAIN = {"uncertain","unknown","unclear","not sure","n/a","none"}


def distinct(values):
    result,seen = [],set()
    for value in values:
        value=clean_text(value)
        key=re.sub(r"[^\w]+"," ",value.casefold()).strip()
        if key and key not in seen:
            seen.add(key);result.append(value)
    return result


def normalize_dense(obj):
    if not isinstance(obj,dict):raise ValueError("Dense response must be a JSON object")
    warnings=[]
    def text(value, path):
        if not isinstance(value,str):
            if value not in (None,[]):warnings.append(path+": dropped non-string value")
            return ""
        value=clean_text(value)
        if ARTIFACT.search(value):
            warnings.append(path+": dropped prompt/frame artifact");return ""
        return value
    def strings(value,path):
        if isinstance(value,str):value=[value]
        if not isinstance(value,list):warnings.append(path+": invalid string list");return []
        def structured(value,depth=0):
            if depth > 3:return ""
            if isinstance(value,str):return text(value,path)
            if isinstance(value,dict):
                return ", ".join(f"{key.replace('_',' ')}: {part}" for key,item in value.items()
                                 if isinstance(key,str) for part in [structured(item,depth+1)] if part)
            if isinstance(value,list):return "; ".join(distinct(structured(item,depth+1) for item in value))
            if value not in (None,[]):warnings.append(path+": dropped unsupported scalar")
            return ""
        values=[]
        for item in value:
            if isinstance(item,(dict,list)):
                warnings.append(path+": normalized structured string-list entry")
            values.append(structured(item))
        result=distinct(values)
        if len([v for v in values if v])!=len(result):warnings.append(path+": deduplicated")
        return result
    def mapping(value,keys,path):
        if not isinstance(value,dict):warnings.append(path+": invalid object");value={}
        return {key:text(value.get(key,""),path+"."+key) for key in keys}
    result={"description":text(obj.get("description",""),"description"),"scene_summary":text(obj.get("scene_summary",""),"scene_summary")}
    result["description"]=result["description"] or result["scene_summary"]
    if not result["description"]:raise ValueError("Dense response has no literal description/scene summary")
    result["environment"]=mapping(obj.get("environment",{}),ENV_KEYS,"environment")
    result["ego_vehicle"]=mapping(obj.get("ego_vehicle",{}),EGO_KEYS,"ego_vehicle")
    for group,keys in ENTITY_GROUPS.items():
        values=obj.get(group,[])
        if not isinstance(values,list):warnings.append(group+": invalid entity list");values=[]
        cleaned=[];seen=set()
        for value in values:
            if not isinstance(value,dict):warnings.append(group+": dropped non-object entity");continue
            item=mapping(value,keys,group)
            if not any(item.values()):continue
            key=json.dumps(item,sort_keys=True).casefold()
            if key in seen:warnings.append(group+": deduplicated entity");continue
            seen.add(key);cleaned.append(item)
        result[group]=cleaned
    infrastructure=obj.get("traffic_infrastructure",{})
    if not isinstance(infrastructure,dict):warnings.append("traffic_infrastructure: invalid object");infrastructure={}
    result["traffic_infrastructure"]={key:strings(infrastructure.get(key,[]),"traffic_infrastructure."+key) for key in INFRA_KEYS}
    for key in STRING_LISTS:result[key]=strings(obj.get(key,[]),key)
    result["weather"]=text(obj.get("weather",""),"weather")
    # Queries are optional; never fabricate a search or use generated labels.
    result["queries"]=[v for v in strings(obj.get("queries",[]),"queries") if not re.search(r"\b(?:no|not|without)\b",v,re.I)]
    for path in ("vehicles","people","temporal_events"):
        if path not in obj:warnings.append(path+": field omitted by model")
    result["metadata_warnings"]=distinct(warnings)
    return result


def empty_metadata(reason):
    result=copy.deepcopy(SCHEMA)
    for group in ENTITY_GROUPS:result[group]=[]
    result.update(description="Dashcam driving scene.",fallback=True,fallback_reason=str(reason),metadata_warnings=["alignment-preserving dense fallback"])
    return result


def meaningful(value):
    return bool(value and value.casefold() not in UNCERTAIN)


def actor_name(entity,group):
    if group=="people":
        name="person"
        if meaningful(entity.get("upper_clothing_color","")):name+=" wearing "+entity["upper_clothing_color"]+" upper clothing"
        if meaningful(entity.get("lower_clothing_color","")):name+=" and "+entity["lower_clothing_color"]+" lower clothing"
        return name
    base=entity.get("type") or {"vehicles":"vehicle","cyclists":"cyclist","motorcycles":"motorcycle"}.get(group,group)
    return " ".join(v for v in (entity.get("color",""),base) if meaningful(v))


def searchable_fields(record):
    def join(values):return "; ".join(distinct(v for v in values if meaningful(v)))
    scene=join([record["scene_summary"],*[f"{k.replace('_',' ')}: {v}" for k,v in record["environment"].items() if meaningful(v)],*record["road_context"],record["weather"]])
    people=[];vehicles=[];actions=list(record["other_actions"])+list(record["events"])
    relations=list(record["spatial_relations"])+list(record["relations"])
    if meaningful(record["ego_vehicle"]["action"]):actions.append("ego vehicle "+record["ego_vehicle"]["action"])
    for group in ("people","vehicles","cyclists","motorcycles"):
        for entity in record[group]:
            actor=actor_name(entity,group)
            details=[f"{k.replace('_',' ')}: {v}" for k,v in entity.items() if meaningful(v) and k not in {"type","color","upper_clothing_color","lower_clothing_color"}]
            fact=actor+(" "+", ".join(details) if details else "")
            (people if group=="people" else vehicles).append(fact)
            for k in ("activity","action","state","temporal_change"):
                if meaningful(entity.get(k,"")):actions.append(actor+" "+entity[k])
            for k in ("relative_position","relative_side","relative_longitudinal_position","lane_relation","location_type","near","position_start","position_end"):
                if meaningful(entity.get(k,"")):relations.append(actor+" "+k.replace('_',' ')+": "+entity[k])
    for event in record["temporal_events"]:
        if event["confidence"].casefold() in {"low","uncertain","unknown"}:continue
        fact=" ".join(v for k,v in event.items() if k!="confidence" and meaningful(v))
        if fact:actions.append(fact)
    infra=join([f"{k.replace('_',' ')}: {v}" for k,values in record["traffic_infrastructure"].items() for v in values]+record["traffic_control"])
    fields={"scene_text":scene,"people_text":join(people),"vehicle_text":join(vehicles),"action_text":join(actions),
            "relation_text":join(relations),"infrastructure_text":infra,"salient_text":join(record["salient_facts"])}
    fields["full_text"]=join([fields[k+"_text"] for k in ("salient","people","vehicle","action","relation","infrastructure","scene")]+[record["description"],*record["objects"]])
    return fields


def legacy_caption(record):
    """Compatible caption slots, composed only from supplied structured facts."""
    fields=searchable_fields(record)
    return {"description":record["description"],"objects":record["objects"],"ego_action":record["ego_vehicle"]["action"],
            "other_actions":distinct([*record["other_actions"],*fields["action_text"].split("; ")]),
            "relations":distinct([*record["relations"],*fields["relation_text"].split("; ")]),
            "road_context":distinct([*record["road_context"],*fields["scene_text"].split("; ")]),
            "traffic_control":distinct([*record["traffic_control"],*fields["infrastructure_text"].split("; ")]),
            "weather":record["weather"] or record["environment"]["weather"],"events":record["events"],"queries":[],
            "caption_warnings":record["metadata_warnings"],"fallback":record.get("fallback",False),
            **({"fallback_reason":record["fallback_reason"]} if record.get("fallback") else {})}


def validate_record(record):
    if not isinstance(record,dict):raise ValueError("Dense row must be an object")
    for key in ("description","scene_summary","weather"):
        if not isinstance(record.get(key),str):raise ValueError("Dense field type: "+key)
    if not record["description"].strip():raise ValueError("Dense empty description")
    for group,keys in (("environment",ENV_KEYS),("ego_vehicle",EGO_KEYS)):
        if not isinstance(record.get(group),dict) or set(record[group])!=set(keys) or any(not isinstance(v,str) for v in record[group].values()):raise ValueError("Dense object schema: "+group)
    for group,keys in ENTITY_GROUPS.items():
        if not isinstance(record.get(group),list):raise ValueError("Dense entity list: "+group)
        for entity in record[group]:
            if not isinstance(entity,dict) or set(entity)!=set(keys) or any(not isinstance(v,str) for v in entity.values()):raise ValueError("Dense entity schema: "+group)
    infra=record.get("traffic_infrastructure")
    if not isinstance(infra,dict) or set(infra)!=set(INFRA_KEYS):raise ValueError("Dense infrastructure schema")
    for key,values in [(k,record.get(k)) for k in (*STRING_LISTS,"queries","metadata_warnings")]+list(infra.items()):
        if not isinstance(values,list) or any(not isinstance(v,str) for v in values):raise ValueError("Dense string list: "+key)
    if record.get("fallback") and not record.get("fallback_reason"):raise ValueError("Dense fallback missing reason")
    if record.get("searchable_texts")!=searchable_fields(record):raise ValueError("Dense searchable texts inconsistent with metadata")
