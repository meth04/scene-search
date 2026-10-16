"""Canonical inventories, frozen-ID enrichment and deterministic V4.2 merge."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import defaultdict

from . import metadata_v4 as v4, metadata_v4_1 as v41
from .qwen_metadata import obj, items, string

VERSION = "scene-search-metadata-v4.2"
MAX_RESPONSE_BYTES = 20000
MAX_SCHEMA_BYTES = 14000
BUDGETS = {"pass1": 1100, "pass2a": 1600, "pass2b": 1400, "pass3": 1400}
TIMEOUTS = {"pass1": 18., "pass2a": 28., "pass2b": 28., "pass3": 12.}
MAX_TOKENS = max(BUDGETS.values())
SCHEMA = copy.deepcopy(v4.SCHEMA)
SCHEMA["properties"]["schema_version"] = string([VERSION])
FINAL_SCHEMA = SCHEMA
PASS1_SCHEMA = obj(
    scene=copy.deepcopy(v41.PASS1_SCHEMA["properties"]["scene"]),
    entities=items(obj(id=string(limit=24), category=string(["vehicle", "person"]),
                       coarse_type=string(v41.COARSE_TYPES),
                       coarse_position=string(limit=48),
                       temporal_presence=string(["early", "late", "most_frames", "brief", "unknown"])), 12),
    infrastructure=items(obj(id=string(limit=24), type=string(v41.INFRA_TYPES)), 12),
)
# Per-ID maps are dynamically built from the frozen inventory. No free-form actor list.
APPEARANCE = obj(subtype=string(v4.SUBTYPES), primary_color=string(v4.COLORS),
                 secondary_color=string(v4.COLORS), upper_clothing_color=string(v4.COLORS),
                 lower_clothing_color=string(v4.COLORS), helmet=string(["yes", "no", "unknown"]),
                 distinctive_features=items(string(limit=48), 4))
SPATIAL = obj(lateral=string(["left", "center", "right", "unknown"]),
              longitudinal=string(["ahead", "behind", "alongside", "unknown"]),
              lane_relation=string(["same_lane", "adjacent_lane", "opposing_lane", "roadside", "unknown"]),
              associated_entity=string(limit=24), evidence_frames=copy.deepcopy(v4.EVIDENCE))
PASS3_SCHEMA = copy.deepcopy(v41.PASS3_SCHEMA)
del PASS3_SCHEMA["properties"]["schema_version"]
PASS3_SCHEMA["required"].remove("schema_version")
PASS3_SCHEMA["properties"]["events"]["maxItems"] = 4
PASS3_SCHEMA["properties"]["scene_relations"]["maxItems"] = 4

PROMPTS = {
    "pass1": """Canonical inventory for ONE dashcam clip; 8 chronological tiles 0..7, row-major.
Return each physical actor exactly once. Do not create a new actor because appearance/position
changes across frames. Do not merge distinct physical actors. Prefer coarse correct category
over unsupported subtype. Unknown is allowed. Ego/hood is not an actor. IDs: vehicle_1,
vehicle_2,... and person_1,...; infrastructure_1,... for infrastructure. Include visible riders
as people distinct from their vehicles. At most 12 actors. coarse_position is a SHORT visible
location distinguishing simultaneous actors (e.g. near left, far center); no temporal action,
rich color or subtype. scene.summary is short scene context, not an actor list.
Stage3A hints are aggregates, not ground truth counts or original track IDs; tracks may fragment.
Return compact JSON only.""",
    "pass2a": """Appearance/subtype ONLY for the frozen actors below in 8 chronological tiles 0..7.
Return a JSON map keyed ONLY by the supplied actor IDs. Preserve identity across all frames.
No new actors, no merge/split, no relations/actions/narrative. Use unknown for weak details.
Vehicle clothing fields unknown; person vehicle colors and subtype unknown. Features must be
visible (e.g. roof_rack, rear_spare_wheel, yellow_side_panel).""",
    "pass2b": """Static position/relation ONLY for the frozen actors below, 8 chronological tiles 0..7.
Return a JSON map keyed ONLY by supplied IDs. No new actors or appearance/subtype/action.
Use left/right image position relative to ego, not a road bend. A single front-facing tile
cannot prove behind ego. Association requires visible riding, not mere proximity.
associated_entity must be an existing actor ID, ego, or unknown. Evidence indices sorted/unique.
Use unknown when uncertain.""",
    "pass3": """Temporal/events ONLY for the frozen actors below in 8 chronological tiles 0..7.
References can ONLY be existing actor IDs or ego; unknown allowed only for absent targets.
No new actors or infrastructure references. Report change only when >=2 distinct evidence
frames support start/end relations. Do not infer an event from a static relation. Ego passing
a stationary actor is not that actor overtaking ego. No change: events=[], relations=[].
Keep ego action short; target unknown if no visible target. Return compact JSON.""",
}
PROMPT_SHA256 = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in PROMPTS.items()}
COMBINED_PROMPT_SHA256 = hashlib.sha256(json.dumps({"prompts": PROMPTS, "schemas":
    [PASS1_SCHEMA, APPEARANCE, SPATIAL, PASS3_SCHEMA], "budgets": BUDGETS, "timeouts": TIMEOUTS}, sort_keys=True).encode()).hexdigest()
PASS1_PROMPT = PROMPTS["pass1"]


class ActorIDError(ValueError):
    """Hard identity failure; the orchestrator may not repair it with another actor list."""


def schema_for(name, inventory=()):
    if len(inventory) > 12:
        raise ValueError("V4.2 schema byte cap: inventory exceeds 12 actors")
    if name == "pass1":
        schema = PASS1_SCHEMA
    elif name == "pass3":
        schema = PASS3_SCHEMA
    else:
        fields = {}
        for actor in inventory:
            rule = copy.deepcopy(APPEARANCE if name == "pass2a" else SPATIAL)
            if name == "pass2b":
                rule["properties"]["associated_entity"] = string([*[a["id"] for a in inventory], "ego", "unknown"], limit=24)
            fields[actor["id"]] = rule
        schema = obj(**fields)
    if len(json.dumps(wire_schema(schema, name)).encode()) > MAX_SCHEMA_BYTES:
        raise ValueError("V4.2 schema byte cap")
    return schema


def wire_schema(schema, name):
    """Factor identical map values once; keep the same exact frozen-ID contract."""
    if name not in ("pass2a", "pass2b") or not schema["properties"]:
        return schema
    rule = next(iter(schema["properties"].values()))
    if not all(v == rule for v in schema["properties"].values()):
        raise ValueError("Inconsistent frozen actor schema")
    return {**schema, "properties": {k: {"$ref": "#/$defs/actor"} for k in schema["properties"]},
            "$defs": {"actor": rule}}


def validate_pass1(value):
    v4.validate_shape(value, PASS1_SCHEMA)
    ids = []
    for a in value["entities"]:
        if not re.fullmatch(a["category"] + r"_[1-9][0-9]*", a["id"]):
            raise ValueError("Noncanonical actor ID")
        if (a["category"] == "person") != (a["coarse_type"] == "person"):
            raise ValueError("Coarse category mismatch")
        ids.append(a["id"])
    infra = [a["id"] for a in value["infrastructure"]]
    if len(ids) != len(set(ids)) or len(infra) != len(set(infra)) or set(ids) & set(infra):
        raise ValueError("Duplicate inventory ID")
    if any(not re.fullmatch(r"infrastructure_[1-9][0-9]*", i) for i in infra):
        raise ValueError("Invalid infrastructure ID")
    if not value["scene"]["summary"].strip():
        raise ValueError("Empty scene summary")
    return value


def parse_pass(text, name, inventory=()):
    value = v41._load(text, MAX_RESPONSE_BYTES)
    if name == "pass1":
        return validate_pass1(value)
    ids = {a["id"] for a in inventory}
    if name in ("pass2a", "pass2b"):
        if not isinstance(value, dict) or set(value) - ids:
            raise ActorIDError(name + " unknown actor ID")
        # Missing actors remain unknown; unknown keys are hard failures, even on retries.
        rule = APPEARANCE if name == "pass2a" else SPATIAL
        for i, fields in value.items():
            v4.validate_shape(fields, rule, name + "." + i)
            if name == "pass2b":
                if fields["associated_entity"] not in ids | {"ego", "unknown"} or fields["associated_entity"] == i:
                    raise ActorIDError("Unknown/self associated actor ID")
                if not v41._valid_frames(fields["evidence_frames"]):
                    raise ValueError("Invalid spatial evidence frames")
        return value
    v4.validate_shape(value, PASS3_SCHEMA, name)
    # Reuse strict temporal evidence validation; forbid infrastructure actor references.
    check = {**value, "schema_version": v41.VERSION}
    try:
        v41.validate_pass3(check, ids, [])
    except ValueError as exc:
        if "reference" in str(exc):
            raise ActorIDError(str(exc)) from None
        raise
    return value


def duplicate_suspicions(inventory):
    groups = defaultdict(list)
    for a in inventory:
        pos, presence = a["coarse_position"], a["temporal_presence"]
        if pos not in ("", "unknown") and presence != "unknown":
            groups[(a["category"], a["coarse_type"], pos, presence)].append(a["id"])
    return [v for v in groups.values() if len(v) > 1]


def validate_reconciliation(value, original):
    """Inventory correction before freeze can remove duplicates, never invent/change actors."""
    validate_pass1(value)
    before = {a["id"]: a for a in original["entities"]}
    for a in value["entities"]:
        if a["id"] not in before or a != before[a["id"]]:
            raise ValueError("Reconciliation must preserve original actor records")
    if value["scene"] != original["scene"] or value["infrastructure"] != original["infrastructure"]:
        raise ValueError("Reconciliation changed scene/infrastructure")
    return value


def merge(p1, p2a=None, p2b=None, p3=None):
    validate_pass1(p1)
    actors = p1["entities"]
    for name, value in (("pass2a", p2a), ("pass2b", p2b), ("pass3", p3)):
        if value is not None:
            parse_pass(json.dumps(value), name, actors)
    base = {"schema_version": v41.VERSION, "scene": copy.deepcopy(p1["scene"]),
            "entities": [{k: a[k] for k in ("id", "category", "coarse_type")} for a in actors],
            "infrastructure": copy.deepcopy(p1["infrastructure"]), "salient_facts": []}
    enrich = []
    for a in actors:
        fields = {**(p2a or {}).get(a["id"], {}), **(p2b or {}).get(a["id"], {})}
        fields = {k: fields.get(k, [] if r["type"] == "array" else "unknown")
                  for k, r in v41.PASS2_SCHEMA["properties"]["entities"]["items"]["properties"].items()}
        fields["id"] = a["id"]
        enrich.append(fields)
    temporal = {**p3, "schema_version": v41.VERSION} if p3 else None
    result = v41.merge(base, {"schema_version": v41.VERSION, "entities": enrich}, temporal)
    result["schema_version"] = VERSION
    # Bind temporal evidence to its canonical subject rather than flattening across actors.
    for a in result["entities"]:
        events = [e for e in result["events"] if e["subject"] == a["id"]]
        if events:
            a["motion"]["state"] = events[-1]["type"]
            a["evidence_frames"] = sorted(set(a["evidence_frames"] + [i for e in events for i in e["evidence_frames"]]))[:4]
    return validate_metadata(result)


def validate_metadata(value):
    v4.validate_metadata(value, schema=SCHEMA)
    ids = {a["id"] for a in value["entities"]}
    for a in value["entities"]:
        if not re.fullmatch(a["category"] + r"_[1-9][0-9]*", a["id"]):
            raise ValueError("Noncanonical final ID")
    for e in [*value["events"], *value["scene_relations"]]:
        if e["subject"] not in ids | {"ego"} or e["object"] not in ids | {"ego", "unknown"}:
            raise ValueError("Unknown final actor reference")
    return value


def parse_metadata(text):
    return validate_metadata(v41._load(text, MAX_RESPONSE_BYTES))


def search_text(value):
    validate_metadata(value)
    return v4.search_text(value, schema=SCHEMA)


def to_dense(value):
    validate_metadata(value)
    return v4.to_dense(value, schema=SCHEMA, version=VERSION)


duplicate_warnings = v4.duplicate_warnings
