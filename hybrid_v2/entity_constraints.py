"""Stored-fact, same-actor matching. No encoders, network, or visual inference."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re

VERSION = "entity_constraints_v1"
STATE_FACTORS = {"MATCH":1., "CLOSE":.86, "UNKNOWN":.60, "MISMATCH":.40, "CONTRADICTION":.12}
CATEGORY_WEIGHTS = {"type":1., "attribute":1., "action":1.5, "relation":1., "near":1.25,
                    "scene":1., "negative":1.}
COLORS = ("black","white","red","blue","gray","silver","green","yellow","brown","orange","purple")
ACTION_PATTERNS = {
    "being overtaken":r"\b(?:being overtaken|overtaken by|being passed|passed by)\b",
    "overtaking":r"\bovertak(?:e|es|en|ing)\b|\bpass(?:es|ed|ing)?(?: the)?(?: ego)?\b|\bmov(?:e|es|ed|ing) ahead of (?:the )?ego\b",
    "merging":r"\bmerg(?:e|es|ed|ing)\b|\b(?:enter\w*|join\w*) (?:the )?(?:ego )?lane\b",
    "lane changing":r"\blane chang\w*\b|\bchang\w* lanes?\b",
    "braking":r"\bbrak\w*\b|\bdecelerat\w*\b|\bslowing down\b",
    "stopping":r"\bstop(?:s|ped|ping)?\b|\bcoming to a stop\b",
    "crossing":r"\bcross(?:es|ed|ing)?\b|\bwalk\w* across (?:the )?road\b",
    "waiting":r"\bwait(?:s|ed|ing)?\b",
    "walking":r"\bwalk(?:s|ed|ing)?\b",
    "running":r"\brun(?:s|ning)?\b",
    "standing":r"\bstand(?:s|ing)?\b|\bstood\b",
    "sitting":r"\bsit(?:s|ting)?\b|\bseated\b",
    "parked":r"\bpark(?:s|ed|ing)?\b",
    "accelerating":r"\baccelerat\w*\b|\bspeeding up\b",
    "moving forward":r"\b(?:mov\w*|driv\w*|travel\w*) (?:forward|straight)\b|\btraveling\b",
    "turning left":r"\bturn\w* (?:to the )?left\b",
    "turning right":r"\bturn\w* (?:to the )?right\b",
}


def clean(value):
    return re.sub(r"\s+"," ",str(value or "").casefold().replace("grey","gray").replace("-"," ")).strip()


def has(text, term):
    return bool(re.search(r"\b"+re.escape(clean(term))+r"s?\b",clean(text)))


def uncertain(value):
    return not clean(value) or bool(re.search(r"\b(?:unknown|uncertain|unclear|maybe|possibly|perhaps|not visible|cannot tell|unsure|n/a)\b",clean(value)))


def clauses(value):
    return [v.strip() for v in re.split(r"[;.!?]|\bbut\b",clean(value)) if v.strip()]


def is_negative(value):
    return bool(re.search(r"\b(?:no|not|never|without)\b",clean(value)))


def actions_in(value):
    value = clean(value)
    result = [name for name,pattern in ACTION_PATTERNS.items() if re.search(pattern,value)]
    if "being overtaken" in result:
        result = [a for a in result if a != "overtaking"]
    if "crossing" in result:
        result = [a for a in result if a != "walking"]
    return result


def add_query_constraints(parsed, query, actor_scope=None, relation_scope=None):
    """Enrich slots without changing the historical dense/Stage5 slot values."""
    q = clean(query)
    attrs = []
    # Colors of the target actor only: stop before a related infrastructure noun
    # or a second actor. Multi-actor queries remain outside this finite parser.
    actor_prefix = clean(actor_scope) if actor_scope is not None else q
    for term in parsed["infrastructure"]:
        actor_prefix = re.split(r"\b(?:at|near|by|beside|with)\s+(?:the\s+|a\s+)?(?:(?:"+"|".join(COLORS)+r")\s+)?"+re.escape(term)+r"\b",actor_prefix)[0]
    for match in re.finditer(r"\b(?:(dark|light)\s+)?("+"|".join(COLORS)+r")\b",actor_prefix):
        color = " ".join(v for v in match.groups() if v)
        tail = actor_prefix[match.end():match.end()+35]
        if parsed["subject_type"] == "person":
            garment = "upper clothing" if re.match(r"\s*(?:shirt|jacket|coat|upper)\b",tail) else "lower clothing" if re.match(r"\s*(?:pants|trousers|lower)\b",tail) else "clothing"
            color += " "+garment
        if color not in attrs:attrs.append(color)
    action_text = clean(actor_scope) if actor_scope is not None else q
    for term in parsed["infrastructure"]:
        action_text = re.sub(r"\b"+re.escape(term)+r"s?\b","",action_text)
    actions = actions_in(action_text)
    if not actions and parsed["action"]:actions = [parsed["action"]]
    relations = []
    for term in ("left","right","ahead","behind","beside"):
        # A turn direction is an action, not necessarily the actor's location.
        location_text = re.sub(r"\bturn\w* (?:to the )?(?:left|right)\b|\bfrom (?:left to right|right to left)\b","", clean(relation_scope) if relation_scope is not None else action_text)
        if has(location_text,term):relations.append(term)
    if re.search(r"\bin front of\b",q) and "ahead" not in relations:relations.append("ahead")
    if "into ego lane" in parsed["relation"]:relations.append("into ego lane")
    near,infra_rel = [],[]
    for term in parsed["infrastructure"]:
        m = re.search(r"\b(at|near|by|beside)\s+(?:the\s+|a\s+)?"+re.escape(term)+r"s?\b",q)
        if m:
            near.append(term)
            infra_rel.append(("at" if m[1] == "at" else "near")+" "+term)
    negatives = []
    for clause in clauses(action_text):
        for m in re.finditer(r"\b(?:no|not|never|without)\s+(.+?)(?=\b(?:and|but|with|on|at|near)\b|$)",clause):
            text = m[1]
            for action in actions_in(text):
                negatives.append({"category":"action","value":action})
                actions = [a for a in actions if a != action]
            for attr in list(attrs):
                if has(text,attr.removesuffix(" clothing").removesuffix(" upper clothing").removesuffix(" lower clothing")):
                    negatives.append({"category":"attribute","value":attr});attrs.remove(attr)
            for relation in list(relations):
                if has(text,relation):
                    negatives.append({"category":"relation","value":relation});relations.remove(relation)
    if re.search(r"\bno\s+(?:cars?|vehicles?|people|persons?|pedestrians?)\b",q):
        negatives.append({"category":"presence","value":parsed["subject_type"]})
    return {**parsed,"entity_constraints":{"type":parsed["subject_type"],"subtype":parsed["subject_specific_type"],
            "attributes":attrs,"actions":actions,"relations":relations,"near":near,"infrastructure_relations":infra_rel},
            "scene_constraints":[v for v in parsed["context"].split("; ") if v],"negative_constraints":negatives}


@dataclass
class Entity:
    broad_type: str
    type: str
    index: int
    source_group: str
    color: str = ""
    upper_clothing_color: str = ""
    lower_clothing_color: str = ""
    side: str = ""
    longitudinal: str = ""
    lane_relation: str = ""
    actions: list[str] = field(default_factory=list)
    temporal_change: str = ""
    near: list[str] = field(default_factory=list)
    appearance: str = ""
    evidence: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


def normalize_entities(record):
    result = []
    for group,broad in (("vehicles","vehicle"),("people","person"),("cyclists","cyclist"),("motorcycles","motorcycle")):
        for i,raw in enumerate(record.get(group,[])):
            e = Entity(broad,clean(raw.get("type")) or ("person" if broad == "person" else broad if broad != "vehicle" else ""),i,group,
                       color=raw.get("color",""),upper_clothing_color=raw.get("upper_clothing_color",""),
                       lower_clothing_color=raw.get("lower_clothing_color",""),
                       side=raw.get("relative_side",raw.get("relative_position","")),
                       longitudinal=raw.get("relative_longitudinal_position",""),lane_relation=raw.get("lane_relation",raw.get("lane","")),
                       actions=list(dict.fromkeys(v for k in ("action","activity","state","temporal_change") if (v:=raw.get(k,"")))),
                       temporal_change=raw.get("temporal_change",""),
                       near=[v for k in ("near","location_type","relative_position") if (v:=raw.get(k,""))],
                       appearance=raw.get("appearance",""),raw=dict(raw))
            result.append(e)
    # Only attach a global sentence when its explicit actor phrase resolves to
    # exactly one stored actor. Ambiguous generic people/cars never distribute
    # facts to several entities. No synthetic entities from prose.
    for text in record.get("spatial_relations",[])+record.get("salient_facts",[]):
        for clause in clauses(text):
            matches = [e for e in result if refers_to(clause,e)]
            if len(matches) == 1:
                e = matches[0]
                e.evidence.append(clause)
                e.near.append(clause)
    for category,values in record.get("traffic_infrastructure",{}).items():
        name = category.replace("_"," ").removesuffix("s") if category != "parking" else category
        for i,value in enumerate(values):
            result.append(Entity("infrastructure",name,i,category,color=value,side=value,actions=[value],near=[value],raw={"description":value}))
    return result


def refers_to(text,e):
    actor_text = re.sub(r"\bbus stops?\b|\bbike lanes?\b","",text)
    if len(re.findall(r"\b(?:person|pedestrian|man|woman|car|sedan|suv|truck|bus|van|cyclist|motorcycle)s?\b",actor_text)) != 1:
        return False
    # Leading actor only; do not use the target of "car passes white person".
    text = re.sub(r"^(?:actor(?:\s*1)?:\s*|a |an |the )","",text)
    text = re.sub(r"^relation(?: type)?:[^,]+,\s*actor\s*1:\s*","",text)
    prefix = re.split(r"\b(?:near|at|beside|behind|ahead|passes?|overtak\w*|walk\w*|wait\w*|stand\w*)\b",text)[0]
    if e.broad_type == "person":
        person = re.search(r"\b(person|pedestrian|man|woman)\b",prefix)
        if not person:return False
        if person[1] in ("man","woman") and not has(e.appearance,person[1]):return False
        colors = [c for c in COLORS if has(prefix,c)]
        return all(has(e.upper_clothing_color,c) or has(e.lower_clothing_color,c) for c in colors)
    if e.broad_type in ("vehicle","cyclist","motorcycle"):
        return bool(e.type and has(prefix,e.type)) and all(has(e.color,c) for c in COLORS if has(prefix,c)) and not has(prefix,"ego")
    return False


def color_state(wanted, observed):
    wanted = clean(wanted)
    if uncertain(observed):return "UNKNOWN"
    observed = clean(observed)
    if is_negative(observed) and has(observed,wanted):return "CONTRADICTION"
    colors = re.findall(r"\b(?:(?:dark|light)\s+)?(?:"+"|".join(COLORS)+r")\b",observed)
    if not colors:return "UNKNOWN"
    if wanted in colors:return "MATCH"
    close = {frozenset(("black","dark gray")),frozenset(("gray","silver")),frozenset(("light gray","silver"))}
    if any(frozenset((wanted,c)) in close for c in colors):return "CLOSE"
    base = wanted.split()[-1]
    if any(base == c.split()[-1] for c in colors):return "CLOSE"
    # Dark/light descriptions do not identify a chromatic color.
    return "CONTRADICTION"


def type_state(wanted,e):
    wanted,actual = clean(wanted),clean(e.type)
    if not actual or uncertain(actual):return "UNKNOWN"
    if wanted in ("car","sedan","suv","truck","bus","van") and any(has(actual,t) for t in ("motorcycle","scooter","bicycle","cyclist")):
        return "CONTRADICTION"
    if wanted in ("vehicle","person","cyclist","motorcycle","infrastructure"):
        if wanted == e.broad_type:return "MATCH"
    if wanted in ("person","pedestrian") and e.broad_type == "person":return "MATCH"
    if has(actual,wanted):return "MATCH"
    if wanted == "car" and any(has(actual,t) for t in ("sedan","hatchback","coupe")):return "MATCH"
    if wanted in ("sedan","hatchback","coupe") and actual == "car":return "CLOSE"
    if wanted == "car" and has(actual,"suv") or wanted == "suv" and actual == "car":return "CLOSE"
    if actual == "vehicle" and e.broad_type == "vehicle":return "UNKNOWN"
    if wanted in ("man","woman") and e.broad_type == "person":
        if has(e.appearance,wanted):return "MATCH"
        if has(e.appearance,"woman" if wanted == "man" else "man"):return "MISMATCH"
        return "UNKNOWN"
    if wanted in ("car","sedan","suv","truck","bus","van") and e.broad_type in ("motorcycle","cyclist"):return "CONTRADICTION"
    return "MISMATCH"


def attribute_state(wanted,e):
    color = re.sub(r"\s+(?:upper |lower )?clothing$","",clean(wanted))
    if e.broad_type != "person":return color_state(color,e.color)
    if "upper clothing" in wanted:return color_state(color,e.upper_clothing_color)
    if "lower clothing" in wanted:return color_state(color,e.lower_clothing_color)
    states = [color_state(color,v) for v in (e.upper_clothing_color,e.lower_clothing_color)]
    # Either garment can satisfy generic clothing; an absent garment blocks
    # declaring explicit incompatible clothing from the other garment alone.
    return next(s for s in ("MATCH","CLOSE","UNKNOWN","MISMATCH","CONTRADICTION") if s in states)


def action_state(wanted,e):
    positives,negatives = set(),set()
    for text in e.actions+e.evidence:
        for clause in clauses(text):
            if uncertain(clause):continue
            (negatives if is_negative(clause) else positives).update(actions_in(clause))
    if wanted in negatives:return "CONTRADICTION"
    if wanted in positives:return "MATCH"
    if wanted == "waiting" and "standing" in positives and not positives.intersection({"walking","running","crossing"}) and any(has(v,"bus stop") for v in e.near):return "CLOSE"
    if wanted == "standing" and "waiting" in positives:return "CLOSE"
    if wanted == "overtaking" and "being overtaken" in positives or wanted == "being overtaken" and "overtaking" in positives:return "CONTRADICTION"
    return "MISMATCH" if positives else "UNKNOWN"


def relation_state(wanted,e):
    # Direct side wins over lane wording (which may refer to ego's lane).
    values = [e.side] if wanted in ("left","right") and not uncertain(e.side) else [e.side,e.longitudinal,e.lane_relation]
    values = [v for v in values if not uncertain(v)]
    if not values:return "UNKNOWN"
    text = "; ".join(values)
    if wanted == "ahead":text = re.sub(r"\bin front of\b","ahead",text)
    if is_negative(text) and has(text,wanted):return "CONTRADICTION"
    opposite = {"left":"right","right":"left","ahead":"behind","behind":"ahead"}.get(wanted)
    if opposite and has(text,wanted) and has(text,opposite):return "UNKNOWN"
    if has(text,wanted):return "MATCH"
    if opposite and has(text,opposite):return "CONTRADICTION"
    return "UNKNOWN"  # an orthogonal axis is absent information


def near_state(wanted,e):
    requested = re.sub(r"^(?:near|at) ","",wanted)
    known = [v for v in e.near if not uncertain(v)]
    for value in known:
        if has(value,requested):
            if is_negative(value) or re.search(r"\b(?:far from|away from)\b",clean(value)):return "CONTRADICTION"
            if value in e.evidence and not re.search(r"\b(?:near|at|by|beside|next to)\s+(?:the |a )?"+re.escape(requested)+r"s?\b",clean(value)):
                continue
            # A direct location_type/near field or resolved same-actor sentence.
            return "MATCH"
    # Near ego/street does not establish absence of a bus stop.
    return "UNKNOWN"


@dataclass
class EntityMatchResult:
    type_score: float
    attribute_score: float
    action_score: float
    relation_score: float
    near_score: float
    coverage_score: float
    contradiction_penalty: float
    final_entity_score: float
    matched_constraints: list
    close_constraints: list
    unknown_constraints: list
    missing_constraints: list
    contradictions: list
    constraints: dict
    comparisons: list
    exact_match: bool


def score_entity(query_constraints, entity, scene=None, negative_constraints=()):
    comparisons = []
    def add(category,value,state):
        comparisons.append({"category":category,"requested":value,"state":state,"factor":STATE_FACTORS[state]})
    c = query_constraints
    add("type",c["subtype"] or c["type"],type_state(c["subtype"] or c["type"],entity))
    for value in c["attributes"]:add("attribute",value,attribute_state(value,entity))
    for value in c["actions"]:add("action",value,action_state(value,entity))
    for value in c["relations"]:add("relation",value,relation_state(value,entity))
    # near and infrastructure_relations express the same condition; count once.
    for value in dict.fromkeys(c["near"]+[re.sub(r"^(?:at|near) ","",v) for v in c["infrastructure_relations"]]):add("near",value,near_state(value,entity))
    for value,state in (scene or []):add("scene",value,state)
    for condition in negative_constraints:
        category,value = condition["category"],condition["value"]
        state = {"action":action_state,"attribute":attribute_state,"relation":relation_state}.get(category)
        observed = state(value,entity) if state else "UNKNOWN" if entity.source_group == "absent" else "MATCH"
        # Explicit absence satisfies negation; merely missing evidence does not.
        inverse = {"MATCH":"CONTRADICTION","CLOSE":"MISMATCH","UNKNOWN":"UNKNOWN","MISMATCH":"UNKNOWN","CONTRADICTION":"MATCH"}[observed]
        add("negative","not "+value,inverse)
    weight = sum(CATEGORY_WEIGHTS[r["category"]] for r in comparisons)
    gm = math.exp(sum(CATEGORY_WEIGHTS[r["category"]]*math.log(r["factor"]) for r in comparisons)/weight)
    n = sum(r["state"] == "CONTRADICTION" for r in comparisons)
    factor = .5**n
    score = gm * math.sqrt(min(r["factor"] for r in comparisons)) * factor
    def listed(state):return [r["category"]+"="+r["requested"] for r in comparisons if r["state"] == state]
    def mean(category):
        values = [r["factor"] for r in comparisons if r["category"] == category]
        return math.prod(values)**(1/len(values)) if values else 1.
    return EntityMatchResult(mean("type"),mean("attribute"),mean("action"),mean("relation"),mean("near"),
                             sum(r["state"] in ("MATCH","CLOSE") for r in comparisons)/len(comparisons),1-factor,score,
                             listed("MATCH"),listed("CLOSE"),listed("UNKNOWN"),listed("MISMATCH"),listed("CONTRADICTION"),
                             {r["category"]+"="+r["requested"]:r["state"] for r in comparisons},comparisons,
                             all(r["state"] == "MATCH" for r in comparisons))


def scene_states(record, constraints):
    env = record.get("environment",{})
    states = []
    for value in constraints:
        if value in ("night","day","dusk"):
            text = clean(env.get("lighting",""))
            text = re.sub(r"\bday(?:time|light)\b","day",text)
        elif value in ("rain","snow"):
            text = clean(env.get("weather",""))
            text = re.sub(r"\brain(?:y|ing)\b","rain",text)
            text = re.sub(r"\bsnow(?:y|ing)\b","snow",text)
        else:text = "; ".join(str(v) for v in env.values())+"; "+record.get("scene_summary","")
        opposite = {"night":"day","day":"night"}.get(value)
        state = "UNKNOWN" if uncertain(text) else "CONTRADICTION" if (is_negative(text) and has(text,value)) or opposite and has(text,opposite) and not has(text,value) else "MATCH" if has(text,value) else "UNKNOWN"
        states.append((value,state))
    return states


def best_entity_match(record,parsed,normalized_entities=None):
    c = parsed["entity_constraints"]
    scene = scene_states(record,parsed["scene_constraints"])
    if c["type"] == "scene":
        e = Entity("scene","scene",-1,"scene")
        result = score_entity(c,e,scene,parsed["negative_constraints"])
        return explanation(e,result)
    entities = [e for e in (normalized_entities if normalized_entities is not None else normalize_entities(record)) if e.broad_type == c["type"]
                and not (c["type"] == "vehicle" and type_state(c["subtype"],e) == "CONTRADICTION")]
    if not entities:
        e = Entity(c["type"],"",-1,"absent")
        result = score_entity(c,e,scene,parsed["negative_constraints"])
        result.final_entity_score *= .35
        result.exact_match = False
        out = explanation(e,result)
        out["entity_present"] = False
        out["absence_support_factor"] = .35
        return out
    best = max(((e,score_entity(c,e,scene,parsed["negative_constraints"])) for e in entities),key=lambda pair:pair[1].final_entity_score)
    return explanation(*best)


def explanation(entity,result):
    return {"best_entity_type":entity.type,"best_entity_index":entity.index,"source_group":entity.source_group,
            "best_entity":asdict(entity),**asdict(result),"entity_score":result.final_entity_score,"entity_present":True,
            "matched":result.matched_constraints,"close":result.close_constraints,"unknown":result.unknown_constraints,
            "missing":result.missing_constraints}
