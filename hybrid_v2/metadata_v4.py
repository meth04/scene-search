"""Actor-bound V4 contract, strict validation and deterministic search text."""

from __future__ import annotations

import hashlib
import json
import re

from .qwen_metadata import items, obj, string

VERSION = "scene-search-metadata-v4"
MAX_RESPONSE_BYTES = 48000
MAX_TOKENS = 6500
COLORS = "black dark_gray gray silver white red orange blue green yellow brown beige purple pink dark light other unknown".split()
SUBTYPES = "sedan hatchback suv coupe van pickup minivan tanker_truck box_truck flatbed_truck truck city_bus coach_bus bus auto_rickshaw motorcycle scooter bicycle other unknown".split()
LEVELS = ["high", "medium", "low"]
POSITION = obj(
    lateral=string(["left", "center", "right", "unknown"]),
    longitudinal=string(["ahead", "behind", "alongside", "unknown"]),
    lane_relation=string(["same_lane", "adjacent_lane", "opposing_lane", "roadside", "unknown"]),
    near=string(limit=48),
)
STATE = obj(
    state=string(limit=48),
    lateral=string(["left", "center", "right", "unknown"]),
    longitudinal=string(["ahead", "behind", "alongside", "unknown"]),
)
EVIDENCE = items({"type": "integer", "minimum": 0, "maximum": 7}, 4)
SCHEMA = obj(
    schema_version=string([VERSION]),
    scene=obj(
        summary=string(limit=240),
        environment=string(limit=48),
        road_type=string(limit=48),
        lighting=string(["day", "night", "dusk", "unknown"]),
        weather=string(limit=48),
        traffic_density=string(["low", "medium", "high", "unknown"]),
        road_condition=string(limit=48),
    ),
    ego=obj(
        action=string(limit=48),
        distance_trend=string(["closing_gap", "increasing_gap", "stable", "unknown"]),
        target=string(limit=24),
        evidence_frames=EVIDENCE,
    ),
    entities=items(
        obj(
            id=string(limit=24),
            category=string(["vehicle", "person"]),
            type=string(
                ["car", "truck", "bus", "motorcycle", "bicycle", "auto_rickshaw", "person", "other_vehicle", "unknown"]
            ),
            subtype=string(SUBTYPES),
            appearance=obj(
                primary_color=string(COLORS),
                secondary_color=string(COLORS),
                upper_clothing_color=string(COLORS),
                lower_clothing_color=string(COLORS),
                helmet=string(["yes", "no", "unknown"]),
                distinctive_features=items(string(limit=48), 6),
            ),
            position=POSITION,
            motion=obj(
                state=string(limit=48),
                relative_to_ego=string(limit=48),
                distance_trend=string(["increasing", "decreasing", "stable", "unknown"]),
            ),
            associated_entity=string(limit=24),
            evidence_frames=EVIDENCE,
            confidence=obj(
                subtype=string(LEVELS), color=string(LEVELS), action=string(LEVELS), position=string(LEVELS)
            ),
        ),
        12,
    ),
    events=items(
        obj(
            id=string(limit=24),
            type=string(limit=48),
            subject=string(limit=24),
            object=string(limit=24),
            start_relation=STATE,
            end_relation=STATE,
            evidence_frames=EVIDENCE,
            confidence=string(LEVELS),
        ),
        8,
    ),
    infrastructure=items(
        obj(
            id=string(limit=24),
            type=string(
                "traffic_light crosswalk stop_sign road_sign bus_stop fuel_station shopfront bridge tunnel lane_marking guardrail traffic_cone hazard_marker streetlight parking intersection roundabout bike_lane bus_lane construction_barrier other unknown".split()
            ),
            position=string(["left", "center", "right", "ahead", "unknown"]),
            attributes=items(string(limit=48), 4),
            confidence=string(LEVELS),
            evidence_frames=EVIDENCE,
        ),
        12,
    ),
    scene_relations=items(
        obj(
            subject=string(limit=24),
            predicate=string(
                "left_of right_of ahead_of behind near far same_lane adjacent_lane riding crossing passing overtaking being_overtaken unknown".split()
            ),
            object=string(limit=24),
            confidence=string(LEVELS),
            evidence_frames=EVIDENCE,
        ),
        12,
    ),
    salient_facts=items(string(limit=120), 6),
)
PROMPT = """Extract scene-search-metadata-v4 from ONE dashcam clip. First identify the distinct visible actors.
For EVERY clearly visible vehicle or person, populate entities with one complete record; do not
mention actors only in summary and then return entities=[]. Include actors briefly visible in
the first or last tile, foreground motorcycles, pedestrians and riders. Unknown fine details
are acceptable, but do not omit a clearly visible actor merely because its subtype is uncertain.
motion.state is the actor's activity (moving/stopped/walking/standing/etc).
Retain start/end states in events only when a temporal change is visibly supported; no redundant
actor state prose. Put concise optional appearance/attachment facts in distinctive_features.
The 8 contact-sheet tiles are chronological,
row-major, frame indices 0..7. Stable local IDs e1,e2,... for actors, i1,i2,... for infrastructure,
ev1,ev2,... for events; ego is the camera vehicle. One actor has ONE ID across tiles. Do not
duplicate it per frame or merge different actors. No cross-window identity. Hood is ego.
Visible facts only: unknown and empty arrays are correct when evidence is weak. Preserve all
clearly visible people/vehicles, subtype, two-tone colors, roof racks/spare wheels, rider garments,
roadside details. Subtype/features only if actually distinguishable; never guess. A person has
type=person/subtype=unknown and vehicle color fields=unknown. A vehicle has clothing fields=unknown.
Relations/events/associated_entity/ego.target reference existing IDs or ego. Use unknown for
unknown references, not invented IDs. Front-facing video cannot prove an actor behind ego from a
static tile: behind requires a visible temporal transition alongside ego. Left/right are image
positions relative to ego, not a road's bend. Do not label a curving road as an actor turning.
An overtake/passing/crossing/gap change needs >=2 distinct evidence frames, start and end states,
correct subject and object. Ego passing a stationary object is not that object overtaking ego.
Streetlamps/glare are not traffic signals: signals need a visible signal head. A person standing
does not prove waiting. Association of rider and vehicle requires visible association.
Confidence is ordinal high/medium/low, not calibrated probability. Keep facts short normalized
terms, never narrative essays or generated search prose. All fields required, unknown for weak
strings, empty lists for absent evidence. Return compact JSON only conforming to this schema:
""" + json.dumps(SCHEMA, separators=(",", ":"))
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()


def validate_shape(value, schema, path="metadata"):
    """Structural/schema check only (no cross-reference rules); shared by V4.1 passes."""

    def check(v, rule, p):
        kind = rule["type"]
        if kind == "object":
            if type(v) is not dict or set(v) != set(rule["required"]):
                raise ValueError(p + ": missing/extra fields")
            for key, child in rule["properties"].items():
                check(v[key], child, p + "." + key)
        elif kind == "array":
            if type(v) is not list or len(v) > rule["maxItems"]:
                raise ValueError(p + ": invalid array")
            for i, item in enumerate(v):
                check(item, rule["items"], f"{p}[{i}]")
        elif kind == "boolean":
            if type(v) is not bool:
                raise ValueError(p + ": expected boolean")
        elif kind == "integer":
            if type(v) is not int or not rule["minimum"] <= v <= rule["maximum"]:
                raise ValueError(p + ": invalid frame index")
        elif type(v) is not str or len(v) > rule["maxLength"] or ("enum" in rule and v not in rule["enum"]):
            raise ValueError(p + ": invalid string/ontology")

    check(value, schema, path)
    return value


def validate_metadata(value, *, schema=None):
    schema = SCHEMA if schema is None else schema

    def check(v, rule, path):
        kind = rule["type"]
        if kind == "object":
            if type(v) is not dict or set(v) != set(rule["required"]):
                raise ValueError(path + ": missing/extra fields")
            for key, child in rule["properties"].items():
                check(v[key], child, path + "." + key)
        elif kind == "array":
            if type(v) is not list or len(v) > rule["maxItems"]:
                raise ValueError(path + ": invalid array")
            for i, item in enumerate(v):
                check(item, rule["items"], f"{path}[{i}]")
        elif kind == "boolean":
            if type(v) is not bool:
                raise ValueError(path + ": expected boolean")
        elif kind == "integer":
            if type(v) is not int or not rule["minimum"] <= v <= rule["maximum"]:
                raise ValueError(path + ": invalid frame index")
        elif type(v) is not str or len(v) > rule["maxLength"] or ("enum" in rule and v not in rule["enum"]):
            raise ValueError(path + ": invalid string/ontology")

    check(value, schema, "metadata")
    if not value["scene"]["summary"].strip():
        raise ValueError("Empty scene summary")
    objects = [*value["entities"], *value["infrastructure"], *value["events"]]
    ids = [e["id"] for e in objects]
    if len(ids) != len(set(ids)) or any(
        not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,23}", i) or i in ("ego", "unknown") for i in ids
    ):
        raise ValueError("IDs must be unique valid local identifiers")
    actor_ids = {"ego", *(e["id"] for e in value["entities"])}
    relation_ids = actor_ids | {e["id"] for e in value["infrastructure"]}
    for e in value["entities"]:
        if e["category"] == "person" and (e["type"] != "person" or e["subtype"] != "unknown"):
            raise ValueError("Person type/subtype mismatch")
        if e["category"] == "vehicle" and e["type"] == "person":
            raise ValueError("Vehicle category mismatch")
        if e["associated_entity"] not in actor_ids | {"unknown"} or e["associated_entity"] == e["id"]:
            raise ValueError("Invalid associated actor reference")
    if value["ego"]["target"] not in actor_ids | {"unknown"}:
        raise ValueError("Invalid ego target reference")
    for r in value["scene_relations"]:
        if r["subject"] not in relation_ids or r["object"] not in relation_ids or r["subject"] == r["object"]:
            raise ValueError("Invalid relation reference")
    for e in value["events"]:
        if (
            e["subject"] not in actor_ids
            or e["object"] not in relation_ids | {"unknown"}
            or e["subject"] == e["object"]
        ):
            raise ValueError("Invalid event reference")
        if len(set(e["evidence_frames"])) < 2:
            raise ValueError("Temporal event requires two distinct evidence frames")
    for e in [value["ego"], *objects, *value["scene_relations"]]:
        frames = e["evidence_frames"]
        if frames != sorted(set(frames)) or any(i > 7 for i in frames):
            raise ValueError("Evidence frames must be unique chronological supplied indices 0..7")
    if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("Oversized V4 metadata")
    return value


def parse_metadata(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("Oversized V4 response")

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    return validate_metadata(
        json.loads(
            text, object_pairs_hook=pairs, parse_constant=lambda v: (_ for _ in ()).throw(ValueError("Non-finite JSON"))
        )
    )


def known(value):
    return bool(value and value not in ("unknown", "other"))


def words(value):
    if isinstance(value, str):
        return value.replace("_", " ") if known(value) else ""
    return "; ".join(filter(None, (words(v) for v in value)))


def duplicate_warnings(value):
    """Warn only: never merge distinct actors because a few attributes coincide."""
    warnings = []
    if not value["entities"] and re.search(
        r"\b(?:vehicles?|cars?|trucks?|motorcycles?|pedestrians?|people|riders?|buses?)\b",
        value["scene"]["summary"],
        re.I,
    ):
        warnings.append({"type": "summary_actor_inventory_missing", "ids": []})
    for i, a in enumerate(value["entities"]):
        for b in value["entities"][i + 1 :]:
            if (a["category"], a["type"], a["subtype"], a["appearance"], a["position"]) == (
                b["category"],
                b["type"],
                b["subtype"],
                b["appearance"],
                b["position"],
            ) and set(a["evidence_frames"]) & set(b["evidence_frames"]):
                warnings.append({"type": "possible_duplicate_actor", "ids": [a["id"], b["id"]]})
    return warnings


ALIASES = {
    "van": "van, light commercial vehicle",
    "auto_rickshaw": "auto rickshaw, three-wheeler",
    "suv": "suv, sport utility vehicle",
    "fuel_station": "fuel station, petrol station",
    "shopfront": "shopfront, roadside shop",
}


def search_text(value, *, schema=None):
    validate_metadata(value, schema=schema)
    texts = {k: [] for k in ("scene", "vehicle", "people", "infrastructure", "relation", "temporal", "fine_attribute")}
    texts["scene"] = [words(list(value["scene"].values()))]
    texts["temporal"].append("ego: " + words(value["ego"]["action"]))
    if known(value["ego"]["distance_trend"]):
        texts["temporal"].append(f"ego: {words(value['ego']['distance_trend'])} to {value['ego']['target']}")
    for e in value["entities"]:
        ap = e["appearance"]
        name = f"entity {e['id']}: "
        label = words(e["subtype"]) or words(e["type"])
        color = (
            words([ap["primary_color"], ap["secondary_color"]])
            if e["category"] == "vehicle"
            else words([ap["upper_clothing_color"], ap["lower_clothing_color"]]) + " clothing"
        )
        identity = name + words([color, label, *ap["distinctive_features"]])
        if label and e["subtype"] in ALIASES:
            identity += "; " + ALIASES[e["subtype"]]
        position = words(list(e["position"].values()))
        motion = words(list(e["motion"].values()))
        texts["vehicle" if e["category"] == "vehicle" else "people"].append(identity + "; " + position + "; " + motion)
        texts["relation"].append(name + words([label, position]))
        texts["temporal"].append(name + words([label, motion]))
        texts["fine_attribute"].append(identity)
    for i in value["infrastructure"]:
        texts["infrastructure"].append(
            f"infrastructure {i['id']}: " + words([ALIASES.get(i["type"], i["type"]), i["position"], *i["attributes"]])
        )
    for r in value["scene_relations"]:
        texts["relation"].append(f"{r['subject']} {words(r['predicate'])} {r['object']}")
    for e in value["events"]:
        texts["temporal"].append(
            f"event {e['id']}: {e['subject']} {words(e['type'])} {e['object']}; start {words(list(e['start_relation'].values()))}; end {words(list(e['end_relation'].values()))}"
        )
    result = {k + "_text": ". ".join(v for v in values if v) for k, values in texts.items()}
    result["full_text"] = ". ".join(result.values()) + ". " + words(value["salient_facts"])
    return result


def to_dense(value, *, schema=None, version=None):
    """Compatibility view for unchanged BGE/visual retrieval; matching reads native actors."""
    from .dense_metadata import normalize_dense

    record = normalize_dense(
        {
            "description": value["scene"]["summary"],
            "scene_summary": value["scene"]["summary"],
            "environment": value["scene"],
            "salient_facts": value["salient_facts"],
        }
    )
    text = search_text(value, schema=schema)
    record["searchable_texts"] = {
        k: text[k]
        for k in ("scene_text", "people_text", "vehicle_text", "infrastructure_text", "relation_text", "full_text")
    }
    record["searchable_texts"].update(
        action_text=text["temporal_text"],
        salient_text=text["fine_attribute_text"] + ". " + words(value["salient_facts"]),
    )
    record.update(
        fallback=False,
        qwen_schema_version=version or VERSION,
        metadata_warnings=[json.dumps(w, sort_keys=True) for w in duplicate_warnings(value)],
    )
    return record
