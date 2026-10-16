"""Conservative score-ordered temporal grouping, without transitive chaining."""
from __future__ import annotations

import math
from .entity_constraints import color_state, has


def temporal_iou(a,b):
    start,end = max(a["start_sec"],b["start_sec"]),min(a["end_sec"],b["end_sec"])
    intersection = max(0.,end-start)
    union = (a["end_sec"]-a["start_sec"])+(b["end_sec"]-b["start_sec"])-intersection
    return intersection/union if union > 0 else 0.


def video_id(row):
    return row.get("video_sha256") or row.get("video_path") or row["clip_id"].rsplit(":",2)[0]


def compatible_events(a,b):
    ea = a.get("entity_explanation",{}).get("best_entity",{})
    eb = b.get("entity_explanation",{}).get("best_entity",{})
    if not ea or not eb:return True
    if ea["broad_type"] != eb["broad_type"]:return False
    ca,cb = ea.get("color",""),eb.get("color","")
    if ca and cb and color_state(ca,cb) == "CONTRADICTION":return False
    if ea["broad_type"] == "person":
        for garment in ("upper_clothing_color","lower_clothing_color"):
            ca,cb = ea.get(garment,""),eb.get(garment,"")
            if ca and cb and color_state(ca,cb) == "CONTRADICTION":return False
    for wanted,opposite in (("left","right"),("right","left")):
        if has(ea.get("side",""),wanted) and has(eb.get("side",""),opposite):return False
    # Known distinct activities can describe separate events/phases. Unknowns
    # never certify tracking, but they also do not invent an incompatibility.
    from .entity_constraints import actions_in
    aa = {v for text in ea.get("actions",[]) for v in actions_in(text)}-{"moving forward","standing"}
    ab = {v for text in eb.get("actions",[]) for v in actions_in(text)}-{"moving forward","standing"}
    return not (aa and ab and aa.isdisjoint(ab))


def group_temporal(ranking,threshold=.5,enabled=True):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("temporal-iou-threshold must be finite and between 0 and 1")
    groups = []
    for row in sorted(ranking,key=lambda r:(-r["final_score"],r["row"])):
        if not all(math.isfinite(row[k]) for k in ("start_sec","end_sec")) or row["end_sec"] <= row["start_sec"]:
            raise ValueError("Invalid temporal window")
        group = next((g for g in groups if enabled and video_id(g[0]) == video_id(row)
                      and all(temporal_iou(member,row) > threshold and compatible_events(member,row) for member in g)),None)
        if group is None:groups.append([row])
        else:group.append(row)
    results = []
    for rank,members in enumerate(groups,1):
        representative = members[0]
        group = {"group_start_sec":min(r["start_sec"] for r in members),"group_end_sec":max(r["end_sec"] for r in members),
                 "representative_window":representative["clip_id"],"member_windows":[r["clip_id"] for r in members],
                 "representative_score":representative["final_score"],"video_id":video_id(representative),
                 "members":[{k:r[k] for k in ("row","clip_id","start_sec","end_sec","rank","final_score")} for r in members]}
        results.append({**representative,"rank":rank,"entity_rank":representative["rank"],"temporal_group":group})
    return results
