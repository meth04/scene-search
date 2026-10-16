import copy

import pytest

from demo.comparison import comparison_payload


def payload(*, swap=False, shadow=True, diagnostic=True):
    return {
        "results": [{"row": row, "rank": rank, "score": 1 - rank / 10} for rank, row in enumerate([8, 4, 2], 1)],
        "decision": {"deployment_mode": "shadow" if shadow else "eg2_only", "user_ranking": "EG2"},
        "retrieval": {
            "diagnostic": [{"candidate_row": row, "valid": True, "matches": []} for row in [8, 4, 2]]
            if diagnostic
            else [],
            "shadow_decision": {
                "swap": swap,
                "challenger": 2,
                "incumbent": 8,
                "proposal": {"late_advantage": 0.03} if swap else None,
                "scored_candidates": {"2": {"global": 0.7, "late": 0.8, "total": 0.775}},
            },
        },
    }


def project(value, top_k=3, pairwise=True):
    return comparison_payload(
        value, top_k=top_k, metadata={row: {"split": "train"} for row in [8, 4, 2]}, pairwise_protection=pairwise
    )


def test_genuine_proposal_does_not_mutate_active_results_or_backend_scores():
    value = payload(swap=True)
    before = copy.deepcopy(value)
    result = project(value)
    assert value == before
    assert [r["row"] for r in result["results"]] == [8, 4, 2]
    assert [r["score"] for r in result["results"]] == [r["score"] for r in before["results"]]
    shadow = result["demo"]["shadow_results"]
    assert [r["row"] for r in shadow] == [2, 8, 4]
    assert [r["rank_delta"] for r in shadow] == [2, -1, -1]
    assert shadow[0]["v6_scores"]["total"] == 0.775


def test_top_one_can_show_real_challenger_outside_active_top_one():
    result = project(payload(swap=True), top_k=1)
    assert result["results"][0]["row"] == 8
    assert result["demo"]["shadow_results"][0]["row"] == 2


def test_no_proposal_preserves_shadow_order():
    result = project(payload())
    assert result["demo"]["shadow_available"]
    assert not result["demo"]["shadow_changed"]
    assert [r["row"] for r in result["demo"]["shadow_results"]] == [8, 4, 2]


@pytest.mark.parametrize(
    "value,pairwise", [(payload(shadow=False), True), (payload(diagnostic=False), True), (payload(swap=True), False)]
)
def test_unavailable_explanations_or_unsupported_proposals_are_not_invented(value, pairwise):
    assert not project(value, pairwise=pairwise)["demo"]["shadow_available"]
    assert project(value, pairwise=pairwise)["demo"]["shadow_results"] == []
