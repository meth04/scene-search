"""Metadata V4.1: deterministic multi-pass actor-centric extraction.

Qwen never regenerates the whole final JSON. Three bounded passes produce a
compact inventory, then actor enrichment, then temporal/events; a local
deterministic merge assembles the final record. The final record is validated
against a V4-compatible schema so the existing matcher and search text are
reused unchanged, with schema_version = scene-search-metadata-v4.1.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re

from .metadata_v4 import (
    COLORS,
    LEVELS,
    POSITION,
    SCHEMA as V4_SCHEMA,
    STATE,
    SUBTYPES,
    VERSION as V4_VERSION,
    validate_shape,
)
from .qwen_metadata import items, obj, string

VERSION = "scene-search-metadata-v4.1"
MAX_RESPONSE_BYTES = 48000

# Per-pass bounded token limits (payload/output runaway control).
PASS1_MAX_TOKENS = 1100
PASS2_MAX_TOKENS = 2400
PASS3_MAX_TOKENS = 1800
MAX_TOKENS = PASS2_MAX_TOKENS

# Final record schema: V4-compatible with the V4.1 version string.
FINAL_SCHEMA = copy.deepcopy(V4_SCHEMA)
FINAL_SCHEMA["properties"]["schema_version"] = string([VERSION])
SCHEMA = FINAL_SCHEMA  # base contract used for the assembled record (index/validate)

VEHICLE_TYPES = ["car", "truck", "bus", "motorcycle", "bicycle", "auto_rickshaw", "other_vehicle", "unknown"]
COARSE_TYPES = [*VEHICLE_TYPES, "person"]
INFRA_TYPES = (
    "traffic_light crosswalk stop_sign road_sign bus_stop fuel_station shopfront bridge tunnel lane_marking "
    "guardrail traffic_cone hazard_marker streetlight parking intersection roundabout bike_lane bus_lane "
    "construction_barrier other unknown"
).split()

# --- Pass 1: actor/scene inventory (compact) ---------------------------------
PASS1_SCHEMA = obj(
    schema_version=string([VERSION]),
    scene=obj(
        summary=string(limit=200),
        environment=string(limit=40),
        road_type=string(limit=40),
        lighting=string(["day", "night", "dusk", "unknown"]),
        weather=string(limit=40),
        traffic_density=string(["low", "medium", "high", "unknown"]),
        road_condition=string(limit=40),
    ),
    entities=items(
        obj(id=string(limit=24), category=string(["vehicle", "person"]), coarse_type=string(COARSE_TYPES)),
        12,
    ),
    infrastructure=items(obj(id=string(limit=24), type=string(INFRA_TYPES)), 12),
    salient_facts=items(string(limit=100), 4),
)

PASS1_PROMPT = """Pass 1 of 3: INVENTORY ONLY for ONE dashcam clip. 8 chronological tiles, row-major, indices 0..7.
List every clearly visible actor ONCE with a stable local id: e1,e2,... for vehicles/persons,
i1,i2,... for infrastructure. Do NOT describe appearance, color, action, relations or events here.
Rules: list each visible actor exactly once; never duplicate the same actor across tiles; never merge
distinct actors; prefer a correct coarse category over an unsupported subtype; unknown is acceptable.
Bound: at most 16 actors. People use category=person and coarse_type=person. Ego/hood is the camera
vehicle and is NOT an entity. A person or vehicle only mentioned in the scene summary but not listed in
entities is a mistake: put every visible actor in entities. scene fields are short phrases only.
Return compact JSON only conforming to this schema:
""" + json.dumps(PASS1_SCHEMA, separators=(",", ":"))

# --- Pass 2: actor enrichment (frozen ids only) ------------------------------
PASS2_SCHEMA = obj(
    schema_version=string([VERSION]),
    entities=items(
        obj(
            id=string(limit=24),
            subtype=string(SUBTYPES),
            primary_color=string(COLORS),
            secondary_color=string(COLORS),
            upper_clothing_color=string(COLORS),
            lower_clothing_color=string(COLORS),
            helmet=string(["yes", "no", "unknown"]),
            distinctive_features=items(string(limit=48), 6),
            lateral=string(["left", "center", "right", "unknown"]),
            longitudinal=string(["ahead", "behind", "alongside", "unknown"]),
            lane_relation=string(["same_lane", "adjacent_lane", "opposing_lane", "roadside", "unknown"]),
            near=string(limit=48),
            associated_entity=string(limit=24),
            motion_state=string(limit=48),
            relative_to_ego=string(limit=48),
            distance_trend=string(["increasing", "decreasing", "stable", "unknown"]),
            evidence_frames=items({"type": "integer", "minimum": 0, "maximum": 7}, 4),
        ),
        12,
    ),
)

PASS2_PROMPT = """Pass 2 of 3: ENRICH the frozen actors from Pass 1 for ONE dashcam clip. 8 chronological tiles, 0..7.
You MUST return exactly the same entity ids given below. Do NOT invent new ids, do NOT drop ids, and do
NOT merge or split actors. Enrich ONLY these ids:
{frozen_ids}
For each id fill: subtype (vehicle subtype, or unknown for people), primary_color/secondary_color for
vehicles, upper_clothing_color/lower_clothing_color for people, helmet, distinctive_features
(roof rack, spare wheel, yellow side panel, etc), lateral/longitudinal/lane_relation position relative to
ego, near, associated_entity (an existing id, e.g. rider to motorcycle, or unknown), motion_state
(moving/stopped/walking/standing), relative_to_ego, distance_trend, and evidence_frames (unique sorted
indices 0..7 you actually used). Unknown/empty is correct when evidence is weak: never guess.
A person has subtype=unknown and vehicle color fields unknown; a vehicle has clothing fields unknown.
Left/right are image positions relative to ego, not a road bend. Return compact JSON only:
""" + json.dumps(PASS2_SCHEMA, separators=(",", ":"))

# --- Pass 3: temporal/events (frozen ids only) -------------------------------
PASS3_SCHEMA = obj(
    schema_version=string([VERSION]),
    ego=obj(
        action=string(limit=48),
        distance_trend=string(["closing_gap", "increasing_gap", "stable", "unknown"]),
        target=string(limit=24),
        evidence_frames=items({"type": "integer", "minimum": 0, "maximum": 7}, 4),
    ),
    events=items(
        obj(
            id=string(limit=24),
            type=string(limit=48),
            subject=string(limit=24),
            object=string(limit=24),
            start_relation=STATE,
            end_relation=STATE,
            evidence_frames=items({"type": "integer", "minimum": 0, "maximum": 7}, 4),
            confidence=string(LEVELS),
        ),
        8,
    ),
    scene_relations=items(
        obj(
            subject=string(limit=24),
            predicate=string(
                "left_of right_of ahead_of behind near far same_lane adjacent_lane riding crossing passing "
                "overtaking being_overtaken unknown".split()
            ),
            object=string(limit=24),
            confidence=string(LEVELS),
            evidence_frames=items({"type": "integer", "minimum": 0, "maximum": 7}, 4),
        ),
        12,
    ),
)

PASS3_PROMPT = """Pass 3 of 3: TEMPORAL/EVENTS for ONE dashcam clip. 8 chronological tiles, indices 0..7.
Reference ONLY these existing actor ids (or ego):
{frozen_ids}
Report only change visible across tiles. An overtake/passing/crossing/lane-change/gap-change needs >=2
distinct evidence frames with start and end states, the correct subject and object. Ego passing a
stationary object is NOT that object overtaking ego. A static tile cannot prove an actor behind ego.
Do NOT emit an overtaking/temporal event from a single frame of static evidence. Keep facts short.
scene_relations reference only existing ids. Return compact JSON only conforming to this schema:
""" + json.dumps(PASS3_SCHEMA, separators=(",", ":"))

PROMPT_SHA256 = {
    "pass1": hashlib.sha256(PASS1_PROMPT.encode()).hexdigest(),
    "pass2": hashlib.sha256(PASS2_PROMPT.encode()).hexdigest(),
    "pass3": hashlib.sha256(PASS3_PROMPT.encode()).hexdigest(),
}
COMBINED_PROMPT_SHA256 = hashlib.sha256((PASS1_PROMPT + PASS2_PROMPT + PASS3_PROMPT).encode()).hexdigest()


def pass2_prompt(frozen_ids):
    """Substitute frozen ids without str.format (the template embeds JSON braces)."""
    ids = ", ".join(frozen_ids) if not isinstance(frozen_ids, str) else frozen_ids
    return PASS2_PROMPT.replace("{frozen_ids}", ids)


def pass3_prompt(frozen_ids):
    ids = ", ".join(frozen_ids) if not isinstance(frozen_ids, str) else frozen_ids
    return PASS3_PROMPT.replace("{frozen_ids}", ids or "ego")


class PassError(ValueError):
    """A pass produced output that cannot be used; carries a precise reason code."""

    def __init__(self, code, detail=None):
        self.code, self.detail = code, detail
        super().__init__(code + (": " + detail if detail else ""))


def _json_pairs(values):
    result = {}
    for key, value in values:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _load(text, limit):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty pass response")
    if len(text.encode()) > limit:
        raise ValueError("Oversized pass response")
    return json.loads(
        text, object_pairs_hook=_json_pairs, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Non-finite JSON"))
    )


def _check(v, rule, path):
    validate_shape(v, rule, path)


def _valid_frames(frames):
    return frames == sorted(set(frames)) and all(0 <= i <= 7 for i in frames)


def _actor_ids(value):
    ids = [e["id"] for e in value["entities"]]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", i) for i in ids):
        raise ValueError("Actor ids must be unique valid local identifiers")
    if any(i in ("ego", "unknown") for i in ids):
        raise ValueError("Reserved actor id")
    return ids


def validate_pass1(value):
    _check(value, PASS1_SCHEMA, "pass1")
    if value["schema_version"] != VERSION:
        raise ValueError("pass1: wrong version")
    if not value["scene"]["summary"].strip():
        raise ValueError("pass1: empty scene summary")
    ids = _actor_ids(value)
    infra = [i["id"] for i in value["infrastructure"]]
    if len(infra) != len(set(infra)) or set(ids) & set(infra):
        raise ValueError("pass1: infrastructure ids must be unique and distinct from actors")
    return value


def validate_pass2(value, frozen_ids):
    _check(value, PASS2_SCHEMA, "pass2")
    if value["schema_version"] != VERSION:
        raise ValueError("pass2: wrong version")
    known = set(frozen_ids)
    seen = [e["id"] for e in value["entities"]]
    if len(seen) != len(set(seen)):
        raise ValueError("pass2: duplicate id")
    unknown = [i for i in seen if i not in known]
    if unknown:
        raise PassError("PASS2_UNKNOWN_ID", ",".join(unknown))
    for e in value["entities"]:
        if not _valid_frames(e["evidence_frames"]):
            raise ValueError("pass2: invalid evidence frames")
        if e["associated_entity"] not in known | {"unknown", "ego"} or e["associated_entity"] == e["id"]:
            raise ValueError("pass2: invalid associated entity reference")
    return value


def validate_pass3(value, actor_ids, infra_ids):
    _check(value, PASS3_SCHEMA, "pass3")
    if value["schema_version"] != VERSION:
        raise ValueError("pass3: wrong version")
    known = set(actor_ids) | {"ego"}
    if value["ego"]["target"] not in known | {"unknown"}:
        raise ValueError("pass3: invalid ego target reference")
    for e in value["events"]:
        if e["subject"] not in known or e["object"] not in known | {"unknown"} or e["subject"] == e["object"]:
            raise ValueError("pass3: invalid event reference")
        if len(set(e["evidence_frames"])) < 2 or not _valid_frames(e["evidence_frames"]):
            raise ValueError("pass3: temporal event requires two distinct evidence frames")
    relation_ids = known | set(infra_ids)
    for r in value["scene_relations"]:
        if r["subject"] not in relation_ids or r["object"] not in relation_ids or r["subject"] == r["object"]:
            raise ValueError("pass3: invalid relation reference")
        if not _valid_frames(r["evidence_frames"]):
            raise ValueError("pass3: invalid relation frames")
    for e in [value["ego"], *value["events"], *value["scene_relations"]]:
        if not _valid_frames(e["evidence_frames"]):
            raise ValueError("pass3: invalid frames")
    return value


def parse_pass1(text):
    return validate_pass1(_load(text, MAX_RESPONSE_BYTES))


def parse_pass2(text, frozen_ids):
    return validate_pass2(_load(text, MAX_RESPONSE_BYTES), frozen_ids)


def parse_pass3(text, actor_ids, infra_ids):
    return validate_pass3(_load(text, MAX_RESPONSE_BYTES), actor_ids, infra_ids)


# --- Deterministic merge -----------------------------------------------------
def _unknown_appearance():
    return {
        "primary_color": "unknown",
        "secondary_color": "unknown",
        "upper_clothing_color": "unknown",
        "lower_clothing_color": "unknown",
        "helmet": "unknown",
        "distinctive_features": [],
    }


def _enrichment_map(pass2):
    return {e["id"]: e for e in (pass2 or {}).get("entities", [])}


def merge(pass1, pass2=None, pass3=None):
    """Assemble the final V4.1 record deterministically from validated passes."""
    validate_pass1(pass1)
    actors = _actor_ids(pass1)
    infra_ids = [i["id"] for i in pass1["infrastructure"]]
    enrich = _enrichment_map(pass2)
    if pass2 is not None:
        validate_pass2(pass2, actors)
    if pass3 is not None:
        validate_pass3(pass3, actors, infra_ids)

    scene = dict(pass1["scene"])
    entities = []
    for e in pass1["entities"]:
        person = e["category"] == "person"
        info = enrich.get(e["id"], {})
        appearance = _unknown_appearance()
        if person:
            appearance["upper_clothing_color"] = info.get("upper_clothing_color", "unknown")
            appearance["lower_clothing_color"] = info.get("lower_clothing_color", "unknown")
            appearance["helmet"] = info.get("helmet", "unknown")
            subtype = "unknown"
        else:
            appearance["primary_color"] = info.get("primary_color", "unknown")
            appearance["secondary_color"] = info.get("secondary_color", "unknown")
            subtype = info.get("subtype", "unknown")
        appearance["distinctive_features"] = list(info.get("distinctive_features", []))
        entities.append(
            {
                "id": e["id"],
                "category": e["category"],
                "type": "person" if person else e["coarse_type"],
                "subtype": subtype,
                "appearance": appearance,
                "position": {
                    "lateral": info.get("lateral", "unknown"),
                    "longitudinal": info.get("longitudinal", "unknown"),
                    "lane_relation": info.get("lane_relation", "unknown"),
                    "near": info.get("near", "unknown"),
                },
                "motion": {
                    "state": info.get("motion_state", "unknown"),
                    "relative_to_ego": info.get("relative_to_ego", "unknown"),
                    "distance_trend": info.get("distance_trend", "unknown"),
                },
                "associated_entity": info.get("associated_entity", "unknown"),
                "evidence_frames": sorted(set(info.get("evidence_frames", []))),
                "confidence": {k: "low" for k in ("subtype", "color", "action", "position")},
            }
        )

    infrastructure = [
        {
            "id": i["id"],
            "type": i["type"],
            "position": "unknown",
            "attributes": [],
            "confidence": "low",
            "evidence_frames": [],
        }
        for i in pass1["infrastructure"]
    ]

    p3 = pass3 or {}
    ego = p3.get("ego") or {
        "action": "unknown",
        "distance_trend": "unknown",
        "target": "unknown",
        "evidence_frames": [],
    }
    value = {
        "schema_version": VERSION,
        "scene": scene,
        "ego": ego,
        "entities": entities,
        "events": list(p3.get("events", [])),
        "infrastructure": infrastructure,
        "scene_relations": list(p3.get("scene_relations", [])),
        "salient_facts": list(pass1.get("salient_facts", [])),
    }
    # Reject the merge itself if it cannot satisfy the final contract.
    from .metadata_v4 import validate_metadata

    validate_metadata(value, schema=FINAL_SCHEMA)
    return value


def parse_metadata(text):
    """Validate a fully-assembled V4.1 record (used by the store/index validator)."""
    from .metadata_v4 import validate_metadata

    if not isinstance(text, str) or len(text.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("Oversized V4.1 response")
    return validate_metadata(_load(text, MAX_RESPONSE_BYTES), schema=FINAL_SCHEMA)


# --- Final-record view bound to the V4.1 schema (index/embed compatibility) ---
def validate_metadata(value):
    from .metadata_v4 import validate_metadata as _validate

    return _validate(value, schema=FINAL_SCHEMA)


def search_text(value):
    from .metadata_v4 import search_text as _search

    return _search(value, schema=FINAL_SCHEMA)


def to_dense(value):
    from .metadata_v4 import to_dense as _to_dense

    return _to_dense(value, schema=FINAL_SCHEMA, version=VERSION)


def duplicate_warnings(value):
    from .metadata_v4 import duplicate_warnings as _dup

    return _dup(value)


def placeholder_final():
    """Schema-shaped empty record for a declared non-searchable placeholder row.

    Used only so the matcher can consume a row that never reached FINAL_SUCCESS
    without inventing actor facts: every field is empty/unknown, so every
    constraint resolves to UNKNOWN (never a match). The row is never published as
    a successful V4.1 record (§25) and never reaches the index as metadata.
    """
    return {
        "schema_version": VERSION,
        "scene": {"summary": "", "environment": "", "road_type": "", "lighting": "unknown",
                  "weather": "", "traffic_density": "unknown", "road_condition": ""},
        "ego": {"action": "", "distance_trend": "unknown", "target": "unknown", "evidence_frames": []},
        "entities": [],
        "events": [],
        "infrastructure": [],
        "scene_relations": [],
        "salient_facts": [],
    }


PROMPT = PASS1_PROMPT
