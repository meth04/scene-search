"""Deterministic driving-query intent. Unknown concepts are left empty."""
import re

ACTION_RULES = [
    (r"\bmerg(?:e|es|ed|ing)\b", "vehicle merging"),
    (r"\bchang(?:e|es|ed|ing) lanes?\b|\blane chang(?:e|es|ed|ing)\b", "vehicle changing lanes"),
    (r"\bovertak(?:e|es|ing|en)\b|\bpass(?:es|ed|ing)?\b", "vehicle overtaking"),
    (r"\bcut(?:s|ting)?[ -]in\b", "vehicle cutting in"),
    (r"\bbrak(?:e|es|ed|ing)\b|\bdecelerat\w*\b", "vehicle braking"),
    (r"\bstop(?:s|ped|ping)?\b", "vehicle stopping"),
    (r"\bturn(?:s|ed|ing)? (?:to the )?left\b", "vehicle turning left"),
    (r"\bturn(?:s|ed|ing)? (?:to the )?right\b", "vehicle turning right"),
    (r"\bpedestrians?\b.*\bcross\w*\b|\bcross\w*\b.*\bpedestrians?\b", "pedestrian crossing"),
]
RELATION_RULES = [
    (r"\binto (?:the )?ego(?: vehicle['’]s)? lane\b", "into ego lane"),
    (r"\bin front of (?:the )?ego(?: vehicle)?\b", "in front of ego vehicle"),
    (r"\bdirectly ahead\b", "directly ahead"),
    (r"\b(?:from|on) (?:the )?right\b|\bright lane\b", "on the right"),
    (r"\b(?:from|on) (?:the )?left\b|\bleft lane\b", "on the left"),
    (r"\badjacent lane\b", "adjacent lane"),
    (r"\bcross(?:es|ed|ing)? (?:the )?road\b", "crossing the road"),
]
CONTEXT_RULES = [
    (r"\bhighway\b", "highway"), (r"\bintersection\b", "intersection"),
    (r"\btunnel\b", "tunnel"), (r"\brain(?:y|ing)?\b", "rain"),
    (r"\bnight(?:time)?\b", "night"), (r"\btraffic lights?\b", "traffic light"),
    (r"\bmulti[ -]lane(?: road)?\b", "multi-lane road"),
]


def _without_road_geometry(text):
    """A road's direction does not describe the target actor's position/action."""
    text = re.split(
        r"\b(?:on|along)\s+(?:(?:a|an|the)\s+)?"
        r"(?:(?:narrow|wide|open|town|city|urban|residential|divided|shaded|winding|tree-lined|multi-lane)\s+)*"
        r"(?:road|street|highway)\b", text, maxsplit=1
    )[0]
    return re.split(r"\b(?:road|street|highway)\s+(?:bend\w*|curv\w*|turn\w*)\b", text, maxsplit=1)[0]


def parse_intent(query):
    q = re.sub(r"\s+", " ", query.lower()).strip()
    if re.search(r"\b(?:no|not|without|never)\b", q):
        return {"action": "", "relation": "", "context": ""}
    # Infrastructure nouns are not vehicle actions; retain the established
    # vehicle feature wording for existing Stage5 consumers.
    structured = parse_query(query)
    if structured["subject_type"] in {"person", "cyclist", "motorcycle"}:
        action = structured["action"]
        if structured["subject_type"] == "person":
            action = ("pedestrian " if structured["subject_specific_type"] == "pedestrian" else "person ") + action if action else ""
            return {"action":action, "relation":structured["relation"], "context":structured["context"]}
    context_query = q
    q = _without_road_geometry(re.sub(r"\bbus stops?\b", "", q))
    def first(rules):
        return next((value for pattern, value in rules if re.search(pattern, q)), "")
    return {
        "action": first(ACTION_RULES), "relation": first(RELATION_RULES),
        "context": "; ".join(value for pattern, value in CONTEXT_RULES if re.search(pattern, context_query)),
    }


INFRASTRUCTURE = ("bus stop", "traffic light", "crosswalk", "bike lane", "bus lane",
                  "parking", "intersection", "roundabout", "tunnel", "bridge", "road sign")
COLORS = ("black", "white", "red", "blue", "gray", "silver", "green", "yellow", "brown", "orange", "purple", "pink", "teal")


def parse_query(query):
    """Local semantic slots; no test-query literals, network or generated labels."""
    q = re.sub(r"\s+", " ", query.casefold().replace("grey", "gray")).strip()
    infrastructure = [term for term in INFRASTRUCTURE if re.search(r"\b"+re.escape(term)+r"s?\b", q)]
    subject_text = q
    for term in infrastructure:
        subject_text = re.sub(r"\b"+re.escape(term)+r"s?\b", "", subject_text)
    classes = (("person", "person"), ("people", "person"), ("pedestrian", "person"), ("man", "person"), ("woman", "person"),
               ("cyclist", "cyclist"), ("bicycle", "cyclist"), ("bike", "cyclist"),
               ("motorcycle", "motorcycle"), ("scooter", "motorcycle"),
               ("motorcyclist", "motorcycle"), ("rider", "motorcycle"),
               ("car", "vehicle"), ("truck", "vehicle"), ("bus", "vehicle"), ("van", "vehicle"),
               ("suv", "vehicle"), ("sedan", "vehicle"), ("hatchback", "vehicle"), ("coupe", "vehicle"),
               ("minibus", "vehicle"), ("tanker", "vehicle"), ("auto-rickshaw", "vehicle"), ("pickup", "vehicle"), ("taxi", "vehicle"), ("convertible", "vehicle"),
               ("vehicle", "vehicle"), ("road", "scene"), ("scene", "scene"))
    matches = sorted((m.start(), m.end(), specific, broad) for specific,broad in classes
                     for m in re.finditer(r"\b"+specific+r"(?:s|es)?\b", subject_text))
    actors = [m for m in matches if m[3] != "scene" and not re.search(r"\bego\s*$", subject_text[:m[0]])]
    subject_role = "actor"
    ego_target = bool(re.search(r"\bego(?: vehicle)?\s+(?:overtakes?|passes?|closes? the gap to)\b", q))
    if matches:
        selected = actors[0] if actors else matches[0]
        begin, finish, specific, broad = selected
        if ego_target and actors:
            subject_role = "target_of_ego"
        if broad == "scene" and infrastructure:
            specific,broad = infrastructure[0],"infrastructure"
    elif infrastructure:
        specific,broad = infrastructure[0],"infrastructure"
    else:
        specific,broad = "scene","scene"
    if infrastructure and re.match(r"^(?:a |an |the )?(?:traffic lights?|bus stops?|crosswalks?|intersections?|bridges?|tunnels?)\b", q):
        specific,broad = infrastructure[0],"infrastructure"
    actor_scope = subject_text
    relation_scope = subject_text
    if matches and broad not in ("scene", "infrastructure"):
        next_actor = next((m for m in actors if m[0] > finish), None)
        actor_scope = subject_text[:next_actor[0]] if next_actor else subject_text
        relation_scope = _without_road_geometry(actor_scope)
        # End before the related actor's adjective phrase, or an infrastructure/context clause.
        tail = actor_scope[finish:]
        tail = re.split(r"\b(?:beside|beneath|under|among|with|at|near|behind|ahead of|to the (?:left|right) of)\b", tail)[0]
        prefix = subject_text[:begin]
        if subject_role == "target_of_ego":
            prefix = re.split(r"\b(?:overtakes?|passes?|closes? the gap to)\b", prefix)[-1]
        actor_scope = prefix + subject_text[begin:finish] + tail
        actor_scope = _without_road_geometry(actor_scope)
    if broad == "infrastructure":
        first_infra = re.search(r"\b"+re.escape(specific)+r"s?\b", q)
        actor_scope = q[:first_infra.end()] if first_infra else ""
    if broad == "scene":
        actor_scope = ""
    attributes = [color for color in COLORS if re.search(r"\b"+color+r"\b", actor_scope)]
    if broad == "person":
        attributes = [color+" clothing" if re.search(r"\b(?:wear\w*|clothes|clothing|shirt|jacket|coat|pants|trousers)\b", q) else color for color in attributes]
    relation = ""
    for term in infrastructure:
        match = re.search(r"\b(at|near|by|beside) (?:the |a )?"+re.escape(term)+r"s?\b",q)
        if match:
            relation = ("near" if match.group(1) in ("by","beside") else match.group(1))+" "+term
            break
    if not relation:
        relation = next((v for p,v in RELATION_RULES if re.search(p,relation_scope)), "")
    if not relation:
        relation = next((v for p,v in ((r"\bleft\b","on the left"),(r"\bright\b","on the right"),
                                       (r"\bahead\b","ahead"),(r"\bbehind\b","behind")) if re.search(p,relation_scope)), "")
    action_text = re.sub(r"\bbus stops?\b", "", actor_scope or q)
    if broad in ("scene", "infrastructure"):
        action_text = re.sub(r"\b(?:crosswalk|crossing|parking|stop|turn)\b", "", action_text)
    human_actions = ((r"\bwait\w*\b","waiting"),(r"\bwalk\w*\b","walking"),(r"\brun(?:s|ning)?\b","running"),
                     (r"\bstand\w*\b","standing"),(r"\bsit(?:s|ting)?\b","sitting"),(r"\bcross\w*\b","crossing"))
    action = next((v for p,v in human_actions if re.search(p,action_text)), "") if broad in ("person","cyclist") else ""
    if not action:
        action = next((v.removeprefix("vehicle ").removeprefix("pedestrian ") for p,v in ACTION_RULES if re.search(p,action_text)), "")
        action = {"changing lanes":"lane changing", "cutting in":"cutting in"}.get(action,action)
    if not action:
        action = next((v for p,v in ((r"\bpark(?:s|ed|ing)?\b","parked"),(r"\bturn(?:s|ed|ing)?\b","turning"),
                                    (r"\bwait\w*\b","waiting"),(r"\bcross\w*\b","crossing")) if re.search(p,action_text)),"")
    context_terms = list(CONTEXT_RULES)+[(r"\brural\b","rural"),(r"\bresidential\b","residential"),
                    (r"\b(?:urban|city)\b","urban"),(r"\bsnow\w*\b","snow"),(r"\bdusk\b","dusk"),(r"\bday(?:time)?\b","day")]
    context = "; ".join(v for p,v in context_terms if re.search(p,q) and v not in infrastructure)
    negated = bool(re.search(r"\b(?:no|not|without|never)\b",q))
    if re.search(r"\bbeing overtaken\b|\bovertaken by\b",q):
        action = "being overtaken"
    if subject_role == "target_of_ego" and re.search(r"\b(?:overtakes?|passes?)\b", q):
        action = "being overtaken"
        actor_scope = re.sub(r"\b(?:overtakes?|passes?)\b", "", actor_scope) + " being overtaken"
    if re.search(r"\bturn\w* (?:to the )?(?:left|right)\b", q) and relation in ("on the left", "on the right"):
        relation = ""
    if re.search(r"\bfrom left to right\b|\bfrom right to left\b", q):
        relation = ""
    if broad == "scene":
        attributes, action, relation = [], "", ""
    if negated: action,relation = "",""
    parsed = {"subject_type":broad,"subject_specific_type":specific,"subject_attributes":attributes,
            "action":action,"relation":relation,"context":context,"infrastructure":infrastructure,
            "object_terms":[specific] if specific != "scene" else [], "negated":negated,
            "subject_role":subject_role,
            "temporal_phrases":re.findall(r"\b(?:before|after|then|from left to right|from right to left)\b",q)}
    from .entity_constraints import add_query_constraints
    return add_query_constraints(parsed,query,actor_scope=actor_scope,relation_scope=relation_scope)
