"""Read-only historical audit, untouched reservation and TRAIN protocol freeze."""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hybrid_v2.embeddinggemma2 import digest, identity
from hybrid_v2.embeddinggemma2_index import write_json
from hybrid_v2.driveevidence_v3 import load_index
from hybrid_v2.driveevidence_v6 import REPRESENTATION, stable_order

OUT = ROOT / "outputs/driveevidence_v6"
V3 = ROOT / "outputs/driveevidence_v3"
V5 = ROOT / "outputs/driveevidence_v5"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reserve_audit():
    path = V5 / "final_v5/benchmark/reservation.json"
    reservation = read(path)
    assert reservation["retrieval_runs"] == reservation["source_inspections"] == 0
    reserved = reservation["final_v5"] + reservation["shadow_reserve"]
    forbidden = [V5 / "final_v5/benchmark/queries.json", V5 / "final_v5/index",
                 V5 / "final_v5/metrics/evaluation.json", V5 / "frozen_v5_manifest.json"]
    assert not any(p.exists() for p in forbidden), "V5_RESERVED_FINAL_CONSUMED"
    sizes = {int(f["lfs"]["size"]) for f in reserved}
    hashes = {f["lfs"]["sha256"] for f in reserved}
    names = {Path(f["path"]).name for f in reserved}
    # Search authoritative on-disk media, including renamed copies with equal size.
    checked, matches = 0, []
    for directory in (ROOT / "data", ROOT / "outputs"):
        for media in directory.rglob("*.mp4"):
            if media.name in names or media.stat().st_size in sizes:
                checked += 1
                if digest(media) in hashes:
                    matches.append(str(media.relative_to(ROOT)))
    assert not matches, "RESERVED_MEDIA_ALREADY_DOWNLOADED " + str(matches)
    receipt = {"passed": True, "reservation_sha256": digest(path), "final_v6_parent": "FINAL_V5",
               "final_v6_count": len(reservation["final_v5"]), "shadow_count": len(reservation["shadow_reserve"]),
               "matching_local_media": matches, "candidate_media_hashed": checked,
               "missing_consumption_artifacts": [str(p.relative_to(ROOT)) for p in forbidden],
               "scope": "V5 receipts plus workspace data/outputs MP4 content scan and absent final artifacts",
               "sequence_overlap_disclosure": reservation["sequence_overlap_disclosure"], "time_unix": time.time()}
    return reservation, receipt


def prepare():
    destination = OUT / "dev/protocol_frozen.json"
    if destination.exists():
        raise ValueError("V6_PROTOCOL_EXISTS_USE_VERIFY")
    reports = ["EMBEDDINGGEMMA2_FULL200", "DRIVEEVIDENCE_DEV_OPTIMIZATION", "DRIVEEVIDENCE_V2_DEV",
               "DRIVEEVIDENCE_V3_DEV", "DRIVEEVIDENCE_V4_DEV", "DRIVEEVIDENCE_V4_NEW_HELDOUT",
               "DRIVEEVIDENCE_V5_DEV", "DRIVEEVIDENCE_V5_CALIBRATION", "DRIVEEVIDENCE_V5_PRODUCT",
               "DRIVEEVIDENCE_V5_FINAL_HELDOUT"]
    reservation, audit = reserve_audit()
    write_json(OUT / "inspection/reserve_at_start.json", audit)
    write_json(OUT / "final_v6/benchmark/reservation.json", {
        "dataset": reservation["dataset"], "revision": reservation["revision"],
        "files": reservation["final_v5"], "parent_sha256": audit["reservation_sha256"],
        "selection_uses_model_results": False, "source_accesses": 0, "retrieval_runs": 0,
        "policy_frozen": False, "time_unix": time.time()})
    write_json(OUT / "shadow_reserve/reservation.json", {
        "files": reservation["shadow_reserve"], "parent_sha256": audit["reservation_sha256"], "source_accesses": 0})
    freeze = read(V3 / "dev/benchmark_frozen.json")
    for name, expected in freeze["hashes"].items():
        assert digest(V3 / "dev" / name) == expected, name
    loaded = load_index(V3 / "dev/index")
    assert not loaded.get("evidence_fallback"), "TRAIN_EXTRACTION_INTEGRITY_FAILED"
    rows = loaded["rows"]
    for row in rows:
        assert row["split"] == "train" and digest(row["source_path"]) == row["source_sha256"], row["clip_id"]
    assert not {r["source_sha256"] for r in rows} & {f["lfs"]["sha256"] for f in reservation["final_v5"] + reservation["shadow_reserve"]}
    queries = [q for q in read(V3 / "dev/queries/queries.json")["queries"] if q["id"] in freeze["eligible_query_ids"]]
    grades = {r["id"]: r for r in read(V3 / "dev/queries/graded_relevance.json")["records"]}
    hard, controls, baseline = [], [], []
    for q in queries:
        vector_path = V3 / "dev/retrieval/text" / (q["id"] + ".json")
        vector = np.array(read(vector_path)["embedding"], np.float32)
        ids = np.array([i for i, r in enumerate(rows) if r["fold"] == q["fold"]])
        order = ids[stable_order(loaded["global_eg2"][ids] @ vector)]
        grade = grades[q["id"]]["grades"]
        assert all(str(int(i)) in grade for i in order[:10]), "UNCLOSED_ORACLE_TOP10"
        incumbent = int(order[0])
        record = {"id": q["id"], "category": q["category"], "fold": q["fold"],
                  "source_sequence": q["source_sequence"], "incumbent": incumbent,
                  "top10": order[:10].tolist(), "text_vector_sha256": digest(vector_path)}
        hit = next((rank for rank, i in enumerate(order, 1) if grade.get(str(int(i))) == "RELEVANT"), None)
        baseline.append({**record, "rank": hit})
        if grade[str(incumbent)] == "RELEVANT":
            controls.append(record)
        elif grade[str(incumbent)] in ("NOT_RELEVANT", "PARTIALLY_RELEVANT", "PARTIAL"):
            challenger = next((int(i) for i in order[1:5] if grade.get(str(int(i))) == "RELEVANT"), None)
            if challenger is not None:
                hard.append({**record, "challenger": challenger,
                             "challenger_rank": int(np.flatnonzero(order == challenger)[0]) + 1})
    for source, target in [(V3 / "dev/queries/queries.json", OUT / "dev/benchmark/queries.json"),
                           (V3 / "dev/queries/graded_relevance.json", OUT / "dev/benchmark/graded_relevance.json")]:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert digest(source) == digest(target)
    write_json(OUT / "dev/oracle/pairs_frozen.json", {"hard_pairs": hard, "preservation": controls,
               "definition": "Source-fold TRAIN EG2 wrong/partial rank1 vs highest reviewed RELEVANT rank2-5; all already-correct queries preserved",
               "UNKNOWN_is_negative": False, "frozen_before_V6_evidence": True, "time_unix": time.time()})
    write_json(OUT / "dev/benchmark/EG2_baseline.json", {"records": baseline})
    write_json(destination, {"TRAIN_only": True, "source_benchmark_sha256": digest(V3 / "dev/benchmark_frozen.json"),
               "source_hashes": freeze["hashes"], "eligible_query_ids": [q["id"] for q in queries],
               "global_vectors_sha256": digest(V3 / "dev/index/global_eg2.npy"),
               "track_records_sha256": digest(V3 / "dev/index/track_records.json"),
               "prior_reports": {n: digest(ROOT / f"CODEX_{n}_REPORT.md") for n in reports},
               "oracle_gate": {"recovery_min": .4, "false_flip_max": .1},
               "source_folds": {str(f): sorted({r["sequence"] for r in rows if r["fold"] == f}) for f in range(3)},
               "search_budget": {"oracle_iterations": 6, "oracle_configs": 120, "rerank_iterations": 8, "rerank_configs": 160},
               "initial_representation": REPRESENTATION, "Top5_loss_tolerance_count": 0,
               "false_flip_max": .1, "final_gate": "oracle GO and >=2/3 nested folds improve with all folds safe",
               "label_limitation": "Inherited agent source judgments; sparse five-frame review, not human certification",
               "time_unix": time.time()})
    print(json.dumps({"TRAIN_clips": len(rows), "queries": len(queries), "hard_pairs": len(hard),
                      "preservation": len(controls), "final_reserved": len(reservation["final_v5"]), "reserve_audit": audit}), flush=True)


def verify():
    protocol = read(OUT / "dev/protocol_frozen.json")
    for name, expected in protocol["source_hashes"].items():
        assert digest(V3 / "dev" / name) == expected, name
    reservation, audit = reserve_audit()
    write_json(OUT / "inspection/reserve_current.json", audit)
    return protocol


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify()
        print("V6 TRAIN freeze and reserved FINAL/SHADOW verification PASS", flush=True)
    else:
        prepare()
