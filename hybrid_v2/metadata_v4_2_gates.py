"""Fail-closed measured gates. Missing evidence can never authorize full200."""
import math

SEMANTIC_LIMITS = {"missing_vehicle": 20, "duplicate_actor": 5, "wrong_relation": 20,
                   "missed_person": 20, "missed_infrastructure": 25}


def grounding_gate(review, expected=50):
    return bool(review and review.get("reviewer_type") == "codex_visual" and
                review.get("human_validated") is False and review.get("reviewed") == expected and
                review.get("direct_source_review") is True and
                all(isinstance(review.get("counts", {}).get(k), int) and review["counts"][k] < v
                    for k, v in SEMANTIC_LIMITS.items()) and review.get("materially_improved") is True and
                review.get("gate_critical_unresolved", 1) == 0)


def retrieval_gate(evaluation):
    if not evaluation:
        return False
    versions = evaluation.get("versions", {})
    if not all(k in versions for k in ("v3", "v4.1", "v4.2")):
        return False
    v3, v41, v42 = [versions[k]["metrics"]["overall"] for k in ("v3", "v4.1", "v4.2")]
    return (v42["Top5"] >= v41["Top5"] and v42["MRR"] >= v41["MRR"] and v42["Top1"] >= v3["Top1"] and
            evaluation.get("severe_category_regression") is False and evaluation.get("queries") == 18 and
            all(not any(v.get("query_time_calls", {"unavailable": 1}).values()) for v in versions.values()))


def pilot_gate(generation, grounding, retrieval, tests, hashes):
    p95 = (generation or {}).get("total_latency", {}).get("p95")
    gates = {"metadata_success": (generation or {}).get("success", 0) >= 49 and (generation or {}).get("expected") == 50,
             "latency": isinstance(p95, (int, float)) and math.isfinite(p95) and p95 < 60,
             "grounding": grounding_gate(grounding), "retrieval": retrieval_gate(retrieval),
             "tests": bool(tests and tests.get("passed", 0) > 0 and tests.get("failed") == 0 and tests.get("returncode") == 0),
             "frozen_hashes": bool(hashes and hashes.get("status") == "PASS")}
    return {"decision": "GO_FULL_200" if all(gates.values()) else "HOLD_AND_FIX", "gates": gates,
            "failed_gates": [k for k, v in gates.items() if not v]}
