"""Cached native actors, conservative contradictions, fixed untuned ranking weights."""

from __future__ import annotations

import re

from .intent_parser import parse_query
from .metadata_v4 import COLORS, SUBTYPES, known

VERSION = "v4-constraints-1"
WEIGHTS = {"coverage_bonus": 0.12, "contradiction_penalty": 0.08, "laya_term": 0.10}
ACTION_ALIASES = {
    "passing": "overtaking",
    "passes_ego": "overtaking",
    "lane_changing": "lane_change",
    "changing_lanes": "lane_change",
    "slowing_down": "decelerating",
    "braking": "decelerating",
    "parked": "stopped",
    "standing": "standing",
    "being_overtaken": "being_overtaken",
    "stopping": "stopped",
    "moving_forward": "moving",
}
TYPE_FAMILIES = {
    "car": {"car", "sedan", "hatchback", "suv", "coupe", "minivan"},
    "truck": {"truck", "pickup", "tanker_truck", "box_truck", "flatbed_truck"},
    "bus": {"bus", "city_bus", "coach_bus"},
    "motorcycle": {"motorcycle", "scooter"},
}


def canonical(value):
    value = value.casefold().replace("-", "_").replace(" ", "_")
    return ACTION_ALIASES.get(value, value)


def parse_constraints(query):
    parsed = parse_query(query)
    q = query.casefold().replace("-", " ")
    ec = parsed["entity_constraints"]
    subtype = canonical(parsed["subject_specific_type"])
    # Generic fine vocabulary, independent of target windows and labels.
    explicit = next(
        (
            s
            for s in sorted(SUBTYPES, key=len, reverse=True)
            if s not in ("unknown", "other") and re.search(r"\b" + re.escape(s.replace("_", " ")) + r"s?\b", q)
        ),
        None,
    )
    if explicit:
        subtype = explicit
    category = (
        "person"
        if parsed["subject_type"] == "person"
        else "vehicle"
        if parsed["subject_type"] in ("vehicle", "motorcycle", "cyclist") or explicit
        else parsed["subject_type"]
    )
    if re.search(r"\b(?:rider|motorcyclist|cyclist)\b", q):
        category, subtype = "person", "person"
    constraints = []

    def add(kind, value, negative=False):
        item = {"kind": kind, "value": canonical(value), "negative": negative}
        if item not in constraints:
            constraints.append(item)

    if category in ("person", "vehicle"):
        add("type", subtype if category == "vehicle" else "person")
    for attribute in ec.get("attributes", []):
        add("attribute", attribute)
    if category == "person":
        # Garments and rider associations remain tied to this person's ID.
        for color, garment in re.findall(
            r"\b(" + "|".join(COLORS) + r")\s+(?:colored?\s+)?(shirt|t\s?shirt|jacket|top|trousers|pants|skirt)\b", q
        ):
            add(
                "attribute",
                color + ("_upper" if garment in ("shirt", "tshirt", "t shirt", "jacket", "top") else "_lower"),
            )
        if re.search(r"\b(?:helmet|helmeted)\b", q):
            add("helmet", "no" if re.search(r"\b(?:without|no) (?:a )?helmet\b", q) else "yes")
        if re.search(r"\b(?:rider|motorcyclist|cyclist)\b", q):
            add("association", "bicycle" if re.search(r"\bcyclist\b", q) else "motorcycle")
    # Preserve secondary colors in explicit two-tone conjunctions.
    color_pair = re.search(r"\b(" + "|".join(COLORS) + r")\s+(?:and|with)\s+(" + "|".join(COLORS) + r")\b", q)
    if color_pair:
        for c in color_pair.groups():
            add("attribute", c)
    for feature in ("roof rack", "rear spare wheel", "spare wheel", "yellow side panel", "two tone"):
        if feature in q:
            if feature == "spare wheel" and "rear spare wheel" in q:
                continue
            add("feature", feature)
    for action in ec.get("actions", []):
        add("action", action)
    if re.search(r"\bclos(?:e|es|ing) (?:the )?gap\b", q):
        add("gap", "closing_gap")
    for relation in ec.get("relations", []):
        add("relation", relation)
    for near in ec.get("near", []):
        add("near", near)
    for infra in parsed["infrastructure"]:
        if not any(c["kind"] == "near" and c["value"] == canonical(infra) for c in constraints):
            add("infrastructure", infra)
    if category == "infrastructure" and not parsed["infrastructure"]:
        add("infrastructure", parsed["subject_specific_type"])
    for term in parsed.get("scene_constraints", []):
        if isinstance(term, str):
            add("scene", term)
    # Existing parser may not recognize these common facilities.
    for term in ("fuel station", "shopfront", "hazard marker", "streetlight", "guardrail", "traffic cone"):
        if term in q:
            add("infrastructure", term)
    if not constraints:
        add("scene", query)
    for negative in parsed.get("negative_constraints", []):
        if isinstance(negative, dict):
            kind = negative.get("category", negative.get("kind", "attribute"))
            add(kind, negative.get("value", "unknown"), True)
    return {"version": VERSION, "category": category, "constraints": constraints, "legacy_parsed": parsed}


def prepare_features(value):
    """No per-query JSON reads, actor IDs never reconstructed from flattened prose."""
    actors = []
    infra = {i["id"]: i for i in value["infrastructure"]}
    for entity in value["entities"]:
        e = dict(entity)
        e["events"] = [v for v in value["events"] if v["subject"] == e["id"]]
        e["relations"] = [r for r in value["scene_relations"] if r["subject"] == e["id"]]
        actors.append(e)
    return {"actors": actors, "infrastructure": infra, "scene": value["scene"], "ego": value["ego"], "metadata": value}


def compare(expected, observed, confidence="low", *, families=False):
    observed = [canonical(v) for v in observed if known(v)]
    if not observed:
        return "UNKNOWN"
    if expected in observed or families and any(v in TYPE_FAMILIES.get(expected, {expected}) for v in observed):
        return "MATCH" if confidence != "low" else "UNKNOWN"
    if confidence == "high":
        return "CONTRADICTION"
    return "MISSING"


def match_actor(actor, features, spec):
    comparisons = []
    for c in spec["constraints"]:
        kind, expected = c["kind"], c["value"]
        confidence = "low"
        state = "UNKNOWN"
        if kind == "scene":
            tokens = [canonical(v) for v in features["scene"].values() if known(v)]
            state = "MATCH" if any(expected in t for t in tokens) else "UNKNOWN"
        elif kind == "infrastructure":
            aliases = {"shop": "shopfront", "shops": "shopfront", "petrol_station": "fuel_station", "viaduct": "bridge"}
            expected = aliases.get(expected, expected)
            state = (
                "MATCH"
                if any(i["type"] == expected and i["confidence"] != "low" for i in features["infrastructure"].values())
                else "UNKNOWN"
            )
        elif actor is not None:
            ap, pos, conf = actor["appearance"], actor["position"], actor["confidence"]
            if kind == "type":
                confidence = conf["subtype"]
                if (
                    expected in ("person", "vehicle")
                    and actor["category"] == expected
                    or expected == actor["type"]
                    and confidence == "low"
                ):
                    confidence = "medium"
                state = (
                    "MATCH"
                    if expected == "vehicle" and actor["category"] == "vehicle"
                    else compare(expected, [actor["type"], actor["subtype"]], confidence, families=True)
                )
            elif kind == "attribute":
                confidence = conf["color"]
                expected = expected.removesuffix("_clothing")
                if actor["category"] == "person":
                    values = [ap["upper_clothing_color"], ap["lower_clothing_color"]]
                    if expected.endswith("_upper") or expected.endswith("_lower"):
                        expected, garment = expected.rsplit("_", 1)
                        values = [ap[garment + "_clothing_color"]]
                    if expected.startswith("upper_") or expected.startswith("lower_"):
                        garment, expected = expected.split("_", 1)
                        values = [ap[garment + "_clothing_color"]]
                    state = compare(expected, values, confidence)
                    if state == "CONTRADICTION" and any(not known(v) for v in values):
                        state = "UNKNOWN"
                else:
                    state = compare(expected, [ap["primary_color"], ap["secondary_color"]], confidence)
            elif kind == "feature":
                observed_features = [canonical(v) for v in ap["distinctive_features"]]
                state = (
                    "MATCH"
                    if expected in observed_features
                    or expected == "spare_wheel"
                    and "rear_spare_wheel" in observed_features
                    or expected == "two_tone"
                    and known(ap["secondary_color"])
                    and ap["secondary_color"] != ap["primary_color"]
                    else "UNKNOWN"
                )
            elif kind == "helmet":
                state = compare(expected, [ap["helmet"]], conf["color"])
            elif kind == "association":
                associated = next((a for a in features["actors"] if a["id"] == actor["associated_entity"]), None)
                state = (
                    compare(expected, [associated["type"], associated["subtype"]], "medium", families=True)
                    if associated
                    else "UNKNOWN"
                )
            elif kind == "relation":
                confidence = conf["position"]
                state = compare(expected, list(pos.values()), confidence)
                # A different non-opposite relation does not contradict this axis.
                if state == "CONTRADICTION":
                    opposite = {"left": "right", "right": "left", "ahead": "behind", "behind": "ahead"}.get(expected)
                    state = "CONTRADICTION" if opposite in pos.values() else "MISSING"
            elif kind == "near":
                targets = [i for i, v in features["infrastructure"].items() if canonical(v["type"]) == expected]
                state = (
                    "MATCH"
                    if any(
                        r["predicate"] == "near" and r["object"] in targets and r["confidence"] != "low"
                        for r in actor["relations"]
                    )
                    or canonical(pos["near"]) == expected
                    and conf["position"] != "low"
                    else "UNKNOWN"
                )
            elif kind == "action":
                confidence = conf["action"]
                observed = list(actor["motion"].values())
                observed += [e["type"] for e in actor["events"] if e["confidence"] != "low"]
                # Passive event is attached only to this actor as object of ego's event.
                observed += [
                    "being_overtaken"
                    for e in features["metadata"]["events"]
                    if e["object"] == actor["id"]
                    and e["subject"] == "ego"
                    and canonical(e["type"]) == "overtaking"
                    and e["confidence"] != "low"
                ]
                state = compare(expected, observed, confidence)
                if state == "CONTRADICTION":
                    opposites = {
                        "overtaking": {"being_overtaken"},
                        "being_overtaken": {"overtaking"},
                        "moving": {"stopped"},
                        "stopped": {"moving"},
                    }
                    state = (
                        "CONTRADICTION"
                        if any(canonical(v) in opposites.get(expected, set()) for v in observed)
                        else "MISSING"
                    )
                if (
                    expected in ("overtaking", "being_overtaken", "crossing", "lane_change")
                    and len(actor["evidence_frames"]) < 2
                ):
                    state = "UNKNOWN"
            elif kind == "gap":
                ego = features["ego"]
                state = (
                    "MATCH"
                    if ego["target"] == actor["id"]
                    and ego["distance_trend"] == "closing_gap"
                    and len(ego["evidence_frames"]) >= 2
                    else "UNKNOWN"
                )
        if c["negative"]:
            state = "CONTRADICTION" if state == "MATCH" else "UNKNOWN"
        comparisons.append({**c, "state": state, "confidence": confidence})
    counts = {s: sum(c["state"] == s for c in comparisons) for s in ("MATCH", "CONTRADICTION", "UNKNOWN", "MISSING")}
    return {
        "actor_id": actor["id"] if actor else None,
        "query_constraints": len(comparisons),
        "matched": counts["MATCH"],
        "contradicted": counts["CONTRADICTION"],
        "unknown": counts["UNKNOWN"],
        "missing": counts["MISSING"],
        "coverage": counts["MATCH"] / len(comparisons) if comparisons else 0,
        "same_entity_match": actor is not None and bool(comparisons) and counts["MATCH"] == len(comparisons),
        "comparisons": comparisons,
    }


def match_candidate(features, spec, candidate_id):
    actors = [e for e in features["actors"] if e["category"] == spec["category"]]
    results = [match_actor(a, features, spec) for a in actors] or [match_actor(None, features, spec)]
    best = max(
        results, key=lambda r: (r["coverage"] - r["contradicted"] / max(1, r["query_constraints"]), -r["missing"])
    )
    return {"candidate_id": candidate_id, **best, "actors": results}


def temporal_summary(features):
    """Deterministic §22 view of a record's temporal content (no Qwen, no ranking).

    Attached to each candidate so the report can inspect whether V4.1 captured
    subject/object/action/start-end relation/ego relation for action_temporal
    queries. V3 records have no events/ego, so the fields are simply absent.
    """
    meta = features.get("metadata") or {}
    ego = meta.get("ego") or {}
    events = meta.get("events") or []
    top = events[0] if events else {}
    return {
        "subject": top.get("subject"),
        "object": top.get("object"),
        "action": top.get("type"),
        "start_relation": top.get("start_relation"),
        "end_relation": top.get("end_relation"),
        "confidence": top.get("confidence"),
        "evidence_frames": top.get("evidence_frames"),
        "ego_action": ego.get("action"),
        "ego_target": ego.get("target"),
        "ego_distance_trend": ego.get("distance_trend"),
        "events": len(events),
        "scene_relations": len(meta.get("scene_relations") or []),
    }


def rerank(table, features):
    spec = parse_constraints(table["query"])
    base = [dict(r) for r in table["candidates"]]
    for r in base:
        result = match_candidate(features[r["row"]], spec, r["row"])
        r.update(ranker="dense-entity-v4", constraint_result=result, base_retrieval_score=r["dense_semantic_score"])
        r["temporal"] = temporal_summary(features[r["row"]])
        r["final_score"] = (
            r["base_retrieval_score"]
            + WEIGHTS["coverage_bonus"] * result["coverage"]
            - WEIGHTS["contradiction_penalty"] * min(2, result["contradicted"])
        )
    base.sort(key=lambda r: (-r["final_score"], r["row"]))
    for i, r in enumerate(base, 1):
        r["rank"] = i
    table.update(
        v4_constraints=spec,
        base_candidates=table["candidates"],
        candidates=base,
        decision={
            "provider": "none",
            "degraded": False,
            "state": decision_state(base),
            "threshold_status": "provisional_unfitted",
        },
        scoring_weights=WEIGHTS,
    )
    return table


def decision_state(candidates):
    if not candidates:
        return "NOT_FOUND"
    top = candidates[0]["constraint_result"]
    if top["query_constraints"] > 0 and top["matched"] == top["query_constraints"] and not top["contradicted"]:
        return "MATCH"
    # Unknown metadata never proves absence. Require explicit contradictions across the cascade.
    if all(r["constraint_result"]["contradicted"] > 0 for r in candidates[:15]):
        return "NOT_FOUND"
    return "LOW_CONFIDENCE"
