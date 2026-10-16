"""Project the backend's bounded proposal without calculating a new ranking."""

from __future__ import annotations

import copy


def comparison_payload(payload, *, top_k, metadata, pairwise_protection):
    active = copy.deepcopy(payload["results"])
    decision = payload.get("decision", {})
    retrieval = payload.get("retrieval", {})
    shadow_decision = retrieval.get("shadow_decision", {})
    diagnostics = {record["candidate_row"]: record for record in retrieval.get("diagnostic", [])}
    scored = shadow_decision.get("scored_candidates", {})
    for hit in active:
        hit["metadata"] = metadata[hit["row"]]
        hit["eg2_rank"] = hit["rank"]
        hit["evidence"] = diagnostics.get(hit["row"])
        hit["v6_scores"] = scored.get(str(hit["row"]), scored.get(hit["row"]))
    shadow = copy.deepcopy(active)
    available = decision.get("deployment_mode") == "shadow" and bool(diagnostics)
    reason = shadow_decision.get("reason_code", decision.get("reason_code"))
    changed = False
    if shadow_decision.get("swap"):
        # Existing pairwise policy returns a promotion, not a complete sorted list.
        # Its exact contract is challenger first, followed by the original EG2 order.
        challenger = shadow_decision.get("challenger")
        if not available or not pairwise_protection or challenger not in {hit["row"] for hit in shadow}:
            available = False
            reason = "SHADOW_PROPOSAL_UNAVAILABLE"
        else:
            winner = next(hit for hit in shadow if hit["row"] == challenger)
            shadow = [winner, *[hit for hit in shadow if hit["row"] != challenger]]
            changed = True
    for rank, hit in enumerate(shadow, 1):
        hit["shadow_rank"] = rank
        hit["rank_delta"] = hit["eg2_rank"] - rank
    return {
        **payload,
        "results": active[:top_k],
        "demo": {
            "shadow_available": available,
            "shadow_changed": changed,
            "shadow_reason": reason,
            "shadow_results": shadow[:top_k] if available else [],
            "shadow_decision": shadow_decision,
            "diagnostic_depth": len(diagnostics),
            "user_ranking": "EG2",
            "not_found_calibrated": False,
        },
    }
