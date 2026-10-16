"""Deterministic 50-clip pilot selection from the frozen 200-clip evaluation set.

Pure function of the manifest order and fixed targets: same inputs -> same rows.
No RNG. Stratifies by Day/Night and by source sequence so all 14 sequences are
represented, and forces inclusion of mandated rows (e.g. the previous
problematic window 19).
"""

from __future__ import annotations

import re

_SEQUENCE = re.compile(r"(test/Sequence_[A-Za-z]+_\d+)")


def sequence_of(source_clip_id):
    match = _SEQUENCE.match(source_clip_id or "")
    return match.group(1) if match else (source_clip_id or "unknown")


def _even_pick(values, count):
    """Pick `count` evenly spaced items from an ordered list, deterministically."""
    values = list(values)
    if count <= 0:
        return []
    if count >= len(values):
        return values
    if count == 1:
        return [values[0]]
    seen, picked = set(), []
    for j in range(count):
        index = round(j * (len(values) - 1) / (count - 1))
        if index not in seen:
            seen.add(index)
            picked.append(values[index])
    return picked


def _largest_remainder(sizes, total_target):
    """Apportion `total_target` across keys proportional to sizes (Hamilton method)."""
    total = sum(sizes.values())
    if total <= 0 or total_target <= 0:
        return {k: 0 for k in sizes}
    quotas = {k: total_target * size / total for k, size in sizes.items()}
    floors = {k: int(quota) for k, quota in quotas.items()}
    remainder = total_target - sum(floors.values())
    order = sorted(sizes, key=lambda k: (-(quotas[k] - floors[k]), k))
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def select_pilot(manifest, pool, *, day=30, night=20, must_include=(19,)):
    """Return the sorted row indices of the deterministic pilot selection.

    manifest: list of manifest rows (row index == position).
    pool: mapping str(row) -> {"day_night": "Day"|"Night", ...}.
    """
    if len(manifest) != len(pool):
        raise ValueError("pool must describe every manifest row")
    for row in must_include:
        if not 0 <= row < len(manifest):
            raise ValueError(f"must_include row out of range: {row}")

    groups = {}
    for r in manifest:
        info = pool[str(r["row"])]
        day_night = info["day_night"]
        if day_night not in ("Day", "Night"):
            raise ValueError(f"Unknown day/night label: {day_night!r}")
        groups.setdefault((day_night, sequence_of(r["source_clip_id"])), []).append(r["row"])
    for rows in groups.values():
        rows.sort()

    chosen = set(int(r) for r in must_include)
    for day_night, target in (("Day", day), ("Night", night)):
        subset = {k: v for k, v in groups.items() if k[0] == day_night}
        forced = [r for r in chosen if pool[str(r)]["day_night"] == day_night]
        remaining = target - len(forced)
        if remaining < 0:
            raise ValueError(f"must_include exceeds {day_night} target")
        sizes = {k: len(v) for k, v in subset.items()}
        allocation = _largest_remainder(sizes, remaining)
        for key, rows in subset.items():
            available = [r for r in rows if r not in chosen]
            chosen.update(_even_pick(available, min(allocation[key], len(available))))
    return sorted(chosen)


def query_target_rows(manifest, queries, *, per_category=2):
    """Deterministically pick rows that cover distinct frozen positive queries.

    One row per query, then a stable round-robin across categories so the pilot
    covers as many query categories as possible without exceeding the quota.
    """
    covers = {}
    for q in queries:
        if q.get("type") != "positive":
            continue
        for a, b in q.get("acceptable_global_timestamp_ranges", []):
            for r in manifest:
                if min(r["global_end_sec"], b) > max(r["global_start_sec"], a):
                    covers.setdefault(q["id"], []).append(r["row"])
    by_category = {}
    for q in queries:
        if q.get("type") != "positive" or q["id"] not in covers:
            continue
        by_category.setdefault(q["category"], []).append(q["id"])
    picked = set()
    for category in sorted(by_category):
        for qid in by_category[category][:per_category]:
            picked.add(covers[qid][0])
    return sorted(picked)


def mandatory_rows(pool, *, slow=12, empty=10):
    """Deterministic must-include rows for diversity: prior timeout/slow/difficult.

    Keeps the diagnostic matrix focused on the windows most likely to expose
    failure modes, while `select_pilot` still fills the Day/Night quota evenly.
    """
    rows = sorted(int(k) for k in pool)
    picked = {r for r in rows if pool[str(r)].get("prior_timeout")}
    slow_rows = [r for r in rows if pool[str(r)].get("prior_slow") and r not in picked]
    empty_rows = [r for r in rows if pool[str(r)].get("v4_empty") and r not in picked]
    picked.update(_even_pick(slow_rows, slow))
    picked.update(_even_pick([r for r in empty_rows if r not in picked], empty))
    return sorted(picked)


def selection_records(manifest, pool, rows, queries=None):
    """Build the pilot_selection.json payload with reason strings and hashes."""
    import hashlib
    import json

    target_rows = set()
    if queries:
        for q in queries:
            for a, b in q.get("acceptable_global_timestamp_ranges", []):
                for r in manifest:
                    if min(r["global_end_sec"], b) > max(r["global_start_sec"], a):
                        target_rows.add(r["row"])
    records = []
    for index in sorted(rows):
        r = manifest[index]
        info = pool[str(r["row"])]
        reasons = [f"{info['day_night']}/{sequence_of(r['source_clip_id'])}"]
        if r["row"] == 19:
            reasons.append("prior_problematic_window_19")
        if info.get("prior_slow"):
            reasons.append("prior_slow")
        if info.get("prior_timeout"):
            reasons.append("prior_timeout")
        if info.get("v4_empty"):
            reasons.append("v4_empty_inventory")
        if r["row"] in target_rows:
            reasons.append("frozen_query_target")
        records.append(
            {
                "row": r["row"],
                "segment_id": r.get("segment_id"),
                "source_clip_id": r["source_clip_id"],
                "sequence": sequence_of(r["source_clip_id"]),
                "day_night": info["day_night"],
                "global_start_sec": r["global_start_sec"],
                "global_end_sec": r["global_end_sec"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"],
                "source_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "clip_id": r["clip_id"],
                            "start_frame": r["start_frame"],
                            "end_frame": r["end_frame"],
                        },
                        sort_keys=True,
                    ).encode()
                ).hexdigest(),
                "selection_reason": "+".join(reasons),
                "prior_problem": bool(r["row"] == 19 or info.get("prior_slow")),
            }
        )
    return records
