"""Label-free, training-free bounded V6 scoring and pairwise promotion."""
from __future__ import annotations

import math
import numpy as np
from .driveevidence_v6 import late_interaction, stable_order, temporal_support

DEFAULT = {"global_alpha": .25, "TopK": 5, "min_score_advantage": 0., "min_late_advantage": 0.,
           "max_global_deficit": .08, "aggregation": "weighted_mean", "factorized": False,
           "typed": False, "temporal_strength": 0., "factor_template": "concise",
           "families": ["GLOBAL", "WINDOW", "FRAME", "TUBE"], "pairwise_protection": True}


def validate_config(config):
    if set(config) != set(DEFAULT):
        raise ValueError("V6_RERANK_CONFIG_FIELDS")
    if not isinstance(config["TopK"], int) or isinstance(config["TopK"], bool) or not 1 <= config["TopK"] <= 10:
        raise ValueError("V6_RERANK_TOPK_BOUND")
    bounds = {"global_alpha": (0., 1.), "min_score_advantage": (0., .2),
              "min_late_advantage": (0., .2), "max_global_deficit": (0., .5), "temporal_strength": (0., .1)}
    if any(not math.isfinite(config[key]) or not low <= config[key] <= high for key, (low, high) in bounds.items()):
        raise ValueError("V6_RERANK_NUMERIC_CONFIG")
    if not config["families"] or len(set(config["families"])) != len(config["families"]) or not set(config["families"]) <= {"GLOBAL", "WINDOW", "FRAME", "TUBE"}:
        raise ValueError("V6_RERANK_EVIDENCE_FAMILIES")
    if config["aggregation"] not in ("mean", "weighted_mean", "min_aware", "top_factor", "geometric"):
        raise ValueError("V6_RERANK_AGGREGATION")
    if config["factor_template"] not in ("concise", "bound") or any(type(config[key]) is not bool for key in ("factorized", "typed", "pairwise_protection")):
        raise ValueError("V6_RERANK_FACTOR_CONFIG")
    return config


def local_score(evidence, atoms, vectors, config):
    validate_config(config)
    factors = atoms if config["factorized"] else atoms[:1]
    interaction = late_interaction(factors, vectors, evidence,
        typed=config["typed"] if config["factorized"] else False,
        aggregation=config["aggregation"], families=config["families"])
    if interaction["valid"]:
        interaction["temporal"] = temporal_support(interaction["matches"])
    return interaction


def decide(global_scores, local_scores, config):
    """Only score EG2 Top-K; retain its exact order on unavailable/weak evidence."""
    validate_config(config)
    global_scores = np.asarray(global_scores, np.float32)
    if global_scores.ndim != 1 or not len(global_scores) or not np.isfinite(global_scores).all():
        raise ValueError("V6_RERANK_GLOBAL_VECTOR")
    base = stable_order(global_scores)
    candidates = [int(row) for row in base[:config["TopK"]]]
    incumbent = candidates[0]
    decision = {"state": "KEEP_EG2", "reason_code": "WEAK_OR_MISSING_V6_EVIDENCE", "swap": False,
                "incumbent": incumbent, "challenger": incumbent, "candidate_count": len(candidates), "proposal": None}
    scored = {}
    for row in candidates:
        local = local_scores.get(row)
        if local is None or not local["valid"]:
            continue
        temporal = local.get("temporal", {})
        score = config["global_alpha"] * float(global_scores[row]) + (1 - config["global_alpha"]) * local["score"]
        if temporal.get("available"):
            score += config["temporal_strength"] * temporal["score"]
        if math.isfinite(score):
            scored[row] = {"global": float(global_scores[row]), "late": local["score"], "total": score}
    decision["scored_candidates"] = scored
    if incumbent not in scored:
        return base, decision
    if not config["pairwise_protection"]:
        # Removal ablation: ordinary blended sorting of the complete Top-K prefix.
        if len(scored) != len(candidates):
            return base, decision
        prefix = sorted(candidates, key=lambda row: (-scored[row]["total"], candidates.index(row)))
        order = np.asarray([*prefix, *base[len(candidates):]], np.int64)
        challenger = int(order[0])
    else:
        eligible = []
        for row in candidates[1:]:
            if row not in scored:
                continue
            total_advantage = scored[row]["total"] - scored[incumbent]["total"]
            late_advantage = scored[row]["late"] - scored[incumbent]["late"]
            deficit = scored[incumbent]["global"] - scored[row]["global"]
            if total_advantage > config["min_score_advantage"] + 1e-7 and late_advantage >= config["min_late_advantage"] and deficit <= config["max_global_deficit"]:
                eligible.append(row)
        if not eligible:
            return base, decision
        challenger = max(eligible, key=lambda row: (scored[row]["total"], -candidates.index(row)))
        order = np.asarray([challenger, *[int(row) for row in base if row != challenger]], np.int64)
    if challenger != incumbent:
        decision.update(state="SAFE_RERANK", reason_code="BOUNDED_V6_SEMANTIC_ADVANTAGE", swap=True,
                        challenger=challenger, proposal={"global_deficit": scored[incumbent]["global"] - scored[challenger]["global"],
                        "late_advantage": scored[challenger]["late"] - scored[incumbent]["late"],
                        "score_advantage": scored[challenger]["total"] - scored[incumbent]["total"]})
    return order, decision


def rerank(global_scores, evidence_for_row, atoms, vectors, config=None):
    config = validate_config(config or dict(DEFAULT))
    base = stable_order(global_scores)
    local = {int(row): local_score(evidence_for_row(int(row)), atoms, vectors, config) for row in base[:config["TopK"]]}
    return decide(global_scores, local, config)
