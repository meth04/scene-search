"""Compact transport-independent Qwen contract; invalid output is never normalized to success."""

from __future__ import annotations

import hashlib
import json

VERSION = "scene-search-metadata-v3"
VEHICLES = "car truck bus motorcycle bicycle other_vehicle unknown".split()
COLORS = "black dark_gray gray silver white red blue green yellow brown other unknown".split()
RELATIONS = "left right ahead behind same_lane adjacent_lane near far unknown".split()
ACTIONS = "moving stopped accelerating decelerating overtaking passing merging lane_change turning_left turning_right crossing waiting walking unknown".split()
INFRASTRUCTURE = "traffic_light crosswalk bus_stop bike_lane bus_lane parking intersection roundabout tunnel bridge road_sign lane_marking construction_barrier other unknown".split()


def string(values=None, limit=180):
    result = {"type": "string", "maxLength": limit}
    if values:
        result["enum"] = values
    return result


def obj(**properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def items(item, limit=8):
    return {"type": "array", "items": item, "maxItems": limit}


SCHEMA = obj(
    schema_version=string([VERSION]),
    scene=obj(
        summary=string(limit=240),
        road_type=string(),
        lighting=string(["day", "night", "dusk", "unknown"]),
        weather=string(),
    ),
    ego_vehicle=obj(action=string(ACTIONS)),
    vehicles=items(
        obj(
            id=string(limit=24),
            type=string(VEHICLES),
            color=string(COLORS),
            relation=string(RELATIONS),
            longitudinal=string(["ahead", "behind", "unknown"]),
            lane_relation=string(["same_lane", "adjacent_lane", "unknown"]),
            action=string(ACTIONS),
        )
    ),
    people=items(
        obj(
            id=string(limit=24),
            upper_color=string(COLORS),
            lower_color=string(COLORS),
            relation=string(RELATIONS),
            action=string(ACTIONS),
            near=string(),
        )
    ),
    infrastructure=items(obj(type=string(INFRASTRUCTURE), relation=string(RELATIONS))),
    spatial_relations=items(obj(subject=string(limit=24), relation=string(RELATIONS), target=string(limit=40))),
    temporal_events=items(obj(actor=string(limit=24), action=string(ACTIONS), evidence=string())),
    salient_facts=items(string(), 5),
)
SCHEMA["properties"]["scene"]["properties"]["summary"]["minLength"] = 1
PROMPT = """Describe only visible retrieval-critical facts in ONE dashcam clip. Frames are chronological.
Ego is the camera vehicle; its hood is not another vehicle. Only report visible facts.
Use unknown or omit an entity when uncertain. Do not infer identity or license-plate text.
Do not invent color, action, relation, people or infrastructure. A temporal event requires
supported change in multiple supplied frames; standing does not prove waiting.
Each actor appears once, never one entry per frame. Vehicle ids are v1, v2, v3, etc.
Person ids are p1, p2, p3, etc. The ego id is ego_vehicle.
spatial_relations.subject and temporal_events.actor must exactly match an existing actor id
or ego_vehicle. Omit a relation/event if its actor or visible evidence is uncertain.
Relations are relative to ego. Front-facing footage cannot establish a vehicle behind ego.
scene.summary must be a nonempty literal description, even if no road/actors are visible.
Short phrases only, no essays, prose, markdown or explanations. Return exactly one compact JSON
object following this schema. All root keys and each entity's keys are required; arrays may be empty.
""" + json.dumps(SCHEMA, separators=(",", ":"))
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()
MAX_RESPONSE_BYTES = 16000


def validate_metadata(value):
    def check(v, schema, path):
        kind = schema["type"]
        if kind == "object":
            if not isinstance(v, dict) or set(v) != set(schema["required"]):
                raise ValueError(f"{path}: missing/extra object fields")
            for key, rule in schema["properties"].items():
                check(v[key], rule, path + "." + key)
        elif kind == "array":
            if not isinstance(v, list) or len(v) > schema["maxItems"]:
                raise ValueError(f"{path}: invalid list or size")
            for i, item in enumerate(v):
                check(item, schema["items"], f"{path}[{i}]")
        elif not isinstance(v, str) or len(v) > schema["maxLength"] or ("enum" in schema and v not in schema["enum"]):
            raise ValueError(f"{path}: invalid string/ontology value")

    check(value, SCHEMA, "metadata")
    if not value["scene"]["summary"].strip():
        raise ValueError("metadata.scene.summary: empty")
    ids = [e["id"] for g in ("vehicles", "people") for e in value[g]]
    if any(not i.strip() or i in ("ego", "ego_vehicle") for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("metadata: actor ids must be unique and nonempty")
    if any(
        e["actor"] not in ["ego", "ego_vehicle", *ids] or not e["evidence"].strip() for e in value["temporal_events"]
    ):
        raise ValueError("metadata: temporal event needs an existing actor and evidence")
    if any(r["subject"] not in ["ego", "ego_vehicle", *ids] for r in value["spatial_relations"]):
        raise ValueError("metadata: spatial relation needs an existing subject")
    if len(json.dumps(value, ensure_ascii=False).encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("metadata: response too large")
    return value


def parse_metadata(text):
    if not isinstance(text, str) or len(text.encode()) > MAX_RESPONSE_BYTES:
        raise ValueError("metadata: empty or oversized response")

    def pairs(values):
        result = {}
        for k, v in values:
            if k in result:
                raise ValueError("metadata: duplicate JSON key")
            result[k] = v
        return result

    return validate_metadata(json.loads(text, object_pairs_hook=pairs))


def to_dense(value):
    """Adapt validated v3 to the historical retrieval schema without adding facts."""
    from .dense_metadata import normalize_dense, searchable_fields

    validate_metadata(value)

    def clean(text):
        return "" if text == "unknown" else text.replace("_", " ")

    infra = {}
    for entry in value["infrastructure"]:
        key = {"parking": "parking", "construction_barrier": "construction_barriers"}.get(
            entry["type"], entry["type"] + "s"
        )
        infra.setdefault(key, []).append(clean(entry["relation"]) or clean(entry["type"]))
    raw = {
        "description": value["scene"]["summary"],
        "scene_summary": value["scene"]["summary"],
        "environment": {k: clean(v) for k, v in value["scene"].items() if k != "summary"},
        "ego_vehicle": {"action": clean(value["ego_vehicle"]["action"])},
        "vehicles": [
            {
                "type": clean(v["type"]),
                "color": clean(v["color"]),
                "relative_side": clean(v["relation"]) if v["relation"] in ("left", "right") else "",
                "relative_longitudinal_position": clean(v["longitudinal"])
                or (clean(v["relation"]) if v["relation"] in ("ahead", "behind") else ""),
                "lane_relation": clean(v["lane_relation"]) or (clean(v["relation"]) if "lane" in v["relation"] else ""),
                "action": clean(v["action"]),
            }
            for v in value["vehicles"]
        ],
        "people": [
            {
                "upper_clothing_color": clean(p["upper_color"]),
                "lower_clothing_color": clean(p["lower_color"]),
                "relative_position": clean(p["relation"]),
                "activity": clean(p["action"]),
                "near": clean(p["near"]),
            }
            for p in value["people"]
        ],
        "traffic_infrastructure": infra,
        "salient_facts": value["salient_facts"],
        "temporal_events": [
            {"actor": e["actor"], "type": clean(e["action"]), "evidence": e["evidence"], "confidence": "visible"}
            for e in value["temporal_events"]
        ],
    }
    for vehicle in list(raw["vehicles"]):
        if vehicle["type"] == "motorcycle":
            raw.setdefault("motorcycles", []).append(
                {
                    "color": vehicle["color"],
                    "relative_position": "; ".join(
                        filter(None, (vehicle["relative_side"], vehicle["relative_longitudinal_position"]))
                    ),
                    "action": vehicle["action"],
                    "lane": vehicle["lane_relation"],
                }
            )
            raw["vehicles"].remove(vehicle)
        elif vehicle["type"] == "bicycle":
            raw.setdefault("cyclists", []).append(
                {
                    "type": "bicycle",
                    "color": vehicle["color"],
                    "relative_position": "; ".join(
                        filter(None, (vehicle["relative_side"], vehicle["relative_longitudinal_position"]))
                    ),
                    "action": vehicle["action"],
                    "lane": vehicle["lane_relation"],
                }
            )
            raw["vehicles"].remove(vehicle)
    result = normalize_dense(raw)
    result.update(fallback=False, qwen_schema_version=VERSION)
    result["searchable_texts"] = searchable_fields(result)
    return result
