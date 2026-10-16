"""Product loader and corpus filters around the unchanged selected V6 engine."""

from __future__ import annotations

import time

import numpy as np

from demo.product_index import load_index
from hybrid_v2.driveevidence_v6_service import DriftMonitor, DriveEvidenceV6SearchEngine


class ProductSearchEngine(DriveEvidenceV6SearchEngine):
    def __init__(self, folder, library):
        super().__init__(folder, deployment_mode="eg2_only", telemetry_path=library.root / "telemetry/search.jsonl")
        self.library = library

    def load(self):
        if self.loaded is None:
            begun = time.perf_counter()
            self.loaded = load_index(self.index_path, integrity=self.library.integrity)
            self.worker = self.library.encoder()
            self.load_count += 1
            self.drift = DriftMonitor()
            self.startup_seconds = time.perf_counter() - begun
            self.health.update(
                ready=True,
                eg2_model_loaded=True,
                global_index_loaded=True,
                v6_index_loaded=not bool(self.loaded["evidence_fallback"]),
                source_hashes_valid=True,
                privacy_ready=True,
                warmup_complete=self.library.warmed,
            )
        else:
            self.worker = self.library.encoder()
        return self

    def filtered_search(self, loaded, query, mode, filters):
        self.load()
        ids = [
            i
            for i, row in enumerate(loaded["rows"])
            if (not filters.source_ids or row["source_id"] in filters.source_ids)
            and (filters.start_sec is None or row["end_sec"] > filters.start_sec)
            and (filters.end_sec is None or row["start_sec"] < filters.end_sec)
        ]
        if not ids:
            return {
                "candidates": [],
                "decision": {
                    "deployment_mode": mode,
                    "user_ranking": "EG2",
                    "not_found_enabled": False,
                    "reason_code": "EMPTY_FILTERED_CORPUS",
                    "fallback": False,
                },
                "retrieval": {"deployment_mode": mode, "filtered_corpus_size": 0, "fallback": False},
            }
        view = {
            **loaded,
            "rows": [loaded["rows"][i] for i in ids],
            "global_vectors": loaded["global_vectors"][ids],
            "families": {},
        }
        for family, data in loaded["families"].items():
            positions, offsets = [], [0]
            for i in ids:
                positions += list(range(int(data["offsets"][i]), int(data["offsets"][i + 1])))
                offsets.append(len(positions))
            view["families"][family] = {
                "vectors": data["vectors"][positions],
                "metadata": [data["metadata"][p] for p in positions],
                "offsets": np.asarray(offsets, np.int64),
            }
        with self.lock:
            self.loaded = view
            try:
                payload = super().search_table(query, deployment_mode=mode)
                for hit in payload["candidates"]:
                    hit["row"] = ids[hit["row"]]
                payload.pop("index", None)
                payload["retrieval"]["filtered_corpus_size"] = len(ids)
                return payload
            finally:
                self.loaded = loaded

    def close(self):
        # The library owns the one resident worker, shared across collections.
        self.loaded = self.worker = None
        self.query_cache.clear()
