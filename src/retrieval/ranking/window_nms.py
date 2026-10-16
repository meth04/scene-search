from __future__ import annotations


def temporal_iou(a, b):
    overlap = max(0, min(a.end_ms, b.end_ms)-max(a.start_ms, b.start_ms))
    return overlap / ((a.end_ms-a.start_ms)+(b.end_ms-b.start_ms)-overlap)


def rank_window_candidates(candidates, threshold=0.25, limit=300):
    """Attach suppressed windows to the first representative; never chain members."""
    ordered = sorted(candidates, key=lambda p: (-p["score"], p["window"].video_id,
                                               p["window"].start_ms, p["window"].scene_id))
    groups = []
    for candidate in ordered:
        w = candidate["window"]
        group = None
        if threshold is not None:
            for kept in groups:
                rep = kept["representative"]["window"]
                if (w.video_id, w.camera, w.dataset_version, w.index_version) == (rep.video_id, rep.camera, rep.dataset_version, rep.index_version):
                    if temporal_iou(w, rep) >= threshold:
                        group = kept
                        break
        if group:
            group["members"].append(candidate)
        else:
            groups.append({"representative": candidate, "members": [candidate]})
    return groups[:limit]
