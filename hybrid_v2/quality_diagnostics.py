"""Read-only scoring/distribution diagnostics; never calibrates saved models."""
from __future__ import annotations

import numpy as np

from .build_features import FEATURE_NAMES

ALPHAS = (0., .1, .25, .4, .6, .8, 1.)


def diagnostic_z(features, mean, std, epsilon=1e-12):
    """Float64 diagnostic standardization; disclose tiny-std floors explicitly.

    Runtime Stage5 continues to use the exact checkpoint std without this floor.
    """
    x, mean, std = [np.asarray(value, np.float64) for value in (features, mean, std)]
    if (x.ndim != 2 or x.shape[1] != 18 or mean.shape != (18,) or std.shape != (18,)
            or not all(np.isfinite(v).all() for v in (x, mean, std)) or (std < 0).any() or epsilon <= 0):
        raise ValueError("Invalid feature/calibration arrays for distribution diagnostics")
    floored = std < epsilon
    with np.errstate(over="raise", invalid="raise"):
        z = (x-mean)/np.maximum(std, epsilon)
    return z, np.flatnonzero(floored).tolist()


def distribution_summary(tables):
    if not tables:
        return {"candidate_rows": 0, "features": [], "diagnostic_std_floor": 1e-12}
    first = tables[0]
    for table in tables:
        if (table["feature_names"] != FEATURE_NAMES or table["feature_mean"] != first["feature_mean"]
                or table["feature_std"] != first["feature_std"]
                or table["stage5_checkpoint"] != first["stage5_checkpoint"]):
            raise ValueError("Candidate table feature order/calibration/checkpoint mismatch")
    features = [[candidate["features"][name] for name in FEATURE_NAMES]
                for table in tables for candidate in table["candidates"]]
    z, floored = diagnostic_z(features, first["feature_mean"], first["feature_std"])
    summary = []
    for i, name in enumerate(FEATURE_NAMES):
        values, absolute = z[:,i], np.abs(z[:,i])
        summary.append({"feature": name, "mean_current_z": float(values.mean()), "std_current_z": float(values.std()),
                        "median_abs_z": float(np.median(absolute)), "p90_abs_z": float(np.quantile(absolute,.9)),
                        "p99_abs_z": float(np.quantile(absolute,.99)),
                        **{f"fraction_abs_z_gt_{threshold}": float((absolute>threshold).mean()) for threshold in (2,3,5)},
                        "checkpoint_mean": first["feature_mean"][i], "checkpoint_std": first["feature_std"][i],
                        "diagnostic_std_was_floored": i in floored})
    return {"candidate_rows": len(z), "features": summary, "diagnostic_std_floor": 1e-12,
            "floored_features": [FEATURE_NAMES[i] for i in floored],
            "note": "Query/candidate observations are dependent; comparisons with saved moments are not a causal test."}


def alpha_scores(table, alpha):
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("Diagnostic alpha must be nonnegative and finite")
    base = np.asarray([r["base_score"] for r in table["candidates"]], np.float32)
    residual = np.asarray([r["stage5_residual"] for r in table["candidates"]], np.float32)
    return base + float(alpha)*residual


def metrics(tables, ranker="stage5", alpha=None):
    if ranker not in {"caption", "stage3", "base", "stage5"}:
        raise ValueError("Unknown ranker")
    ranks = []
    for table in tables:
        relevant = table.get("relevant_clip_ids", [])
        if not relevant:
            continue  # Absent-scene controls and unlabeled cases are never positives.
        if alpha is None:
            candidates = sorted(table["candidates"], key=lambda r: r["ranks"][ranker])
        else:
            candidates = [table["candidates"][i] for i in np.argsort(-alpha_scores(table, alpha), kind="stable")]
        ranks.append(next((i+1 for i,r in enumerate(candidates) if r["clip_id"] in relevant), None))
    n = len(ranks)
    return {"positive_queries": n, "top1_correct": sum(r == 1 for r in ranks),
            "top3_correct": sum(r is not None and r <= 3 for r in ranks),
            "top1": sum(r == 1 for r in ranks)/n if n else None,
            "top3": sum(r is not None and r <= 3 for r in ranks)/n if n else None,
            "mrr": sum(1/r for r in ranks if r is not None)/n if n else None}


def candidate_diagnostic(candidate):
    largest = sorted(candidate["standardized_features"].items(), key=lambda p: -abs(p[1]))[:5]
    return {"clip_id": candidate["clip_id"], "start_sec": candidate["start_sec"], "end_sec": candidate["end_sec"],
            "caption_rank": candidate["ranks"]["caption"], "base_rank": candidate["ranks"]["base"],
            "stage5_rank": candidate["ranks"]["stage5"], "base_score": candidate["base_score"],
            "residual": candidate["stage5_residual"], "final_score": candidate["stage5_final"],
            "top5_absolute_standardized_features": [{"feature": n, "z": v} for n,v in largest]}


def promotion_audit(tables):
    audits, labels, values = [], [], []
    for table in tables:
        relevant = table.get("relevant_clip_ids", [])
        if not relevant:
            continue
        correct = [r for r in table["candidates"] if r["clip_id"] in relevant]
        if not correct:
            audits.append({"case_id": table["case_id"], "classification": "retrieval_miss",
                           "note": "No manually relevant candidate was retrieved; no source positive was inserted."})
            continue
        best = {name: min(r["ranks"][name] for r in correct) for name in ("caption", "base", "stage5")}
        classify = lambda before: "A_promotes_correct" if best["stage5"]<best[before] else "B_demotes_correct" if best["stage5"]>best[before] else "C_unchanged_rank"
        top = min(table["candidates"], key=lambda r: r["ranks"]["stage5"])
        wrong = top["clip_id"] not in relevant
        newly_promoted = wrong and top["ranks"]["base"]>1
        audits.append({"case_id": table["case_id"], "query": table["query"], "classification": classify("base"),
                       "versus_caption": classify("caption"), "best_relevant_ranks": best,
                       "correct_candidates": [candidate_diagnostic(r) for r in correct],
                       "stage5_top1": candidate_diagnostic(top), "wrong_top1": wrong,
                       "new_wrong_top1_promotion_over_base": newly_promoted})
        for r in table["candidates"]:
            labels.append(int(newly_promoted and r["clip_id"]==top["clip_id"]))
            values.append([r["standardized_features"][n] for n in FEATURE_NAMES])
    associations = []
    if values:
        z, group = np.asarray(values, np.float64), np.asarray(labels, np.bool_)
        for i,name in enumerate(FEATURE_NAMES):
            correlation = None
            if group.any() and (~group).any() and z[:,i].std()>1e-12:
                correlation = float(np.corrcoef(z[:,i], group.astype(float))[0,1])
            associations.append({"feature": name, "correlation_with_new_wrong_top1_promotion": correlation,
                                 "mean_abs_z_promoted": float(np.abs(z[group,i]).mean()) if group.any() else None,
                                 "mean_abs_z_other": float(np.abs(z[~group,i]).mean()) if (~group).any() else None})
        associations.sort(key=lambda r: -abs(r["correlation_with_new_wrong_top1_promotion"] or 0))
    return {"queries": audits, "wrong_promoted_candidates": sum(labels), "association_candidates": len(labels),
            "feature_associations": associations,
            "association_note": "Exploratory correlation with newly wrong Top-1 promotions versus base; dependent small samples, not causality."}


def diagnose_tables(tables):
    positive = [t for t in tables if t.get("relevant_clip_ids")]
    absent = [t for t in tables if t.get("is_absent_control")]
    return {"positive_distribution": distribution_summary(positive), "absent_control_distribution": distribution_summary(absent),
            "metrics": {ranker: metrics(tables, ranker) for ranker in ("caption", "stage3", "base", "stage5")},
            "promotion_audit": promotion_audit(positive),
            "alpha_sweep": [{"alpha": a, **metrics(tables, alpha=a)} for a in ALPHAS],
            "alpha_note": "Diagnostic only; no alpha/default/checkpoint is updated."}
