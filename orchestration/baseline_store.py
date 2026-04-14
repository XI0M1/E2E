"""
Persistent baseline store for Phase 1 offline sampling.

Stores baseline TPS measurements per workload in an append-only JSONL file so
that baselines survive restarts, resume runs, and can be reused for training
and evaluation without re-measurement.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class BaselineStore:
    """Append-only persistent store for per-workload baseline TPS records.

    Each record in the JSONL file contains:
        workload_id   – identifier derived from the workload file path
        baseline_tps  – median TPS over *n* repeated runs
        baseline_runs – list of individual TPS values used for the median
        config        – the default knob configuration used
        timestamp     – ISO-8601 UTC timestamp of when the baseline was taken

    Storage is append-only.  ``load()`` returns the *latest* record for a
    given workload; ``load_all()`` returns the latest record per workload.
    Thread-safety is not required.
    """

    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        store_dir = os.path.dirname(store_path)
        if store_dir:
            os.makedirs(store_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_baseline(self, workload_id: str) -> bool:
        """Return True if a baseline record exists for *workload_id*."""
        return self.load(workload_id) is not None

    def save(
        self,
        workload_id: str,
        baseline_tps: float,
        baseline_runs: list[float],
        config: dict[str, Any],
    ) -> None:
        """Append a new baseline record to the JSONL store."""
        record = {
            "workload_id": workload_id,
            "baseline_tps": float(baseline_tps),
            "baseline_runs": [float(v) for v in baseline_runs],
            "config": config,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(self.store_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    def load(self, workload_id: str) -> dict | None:
        """Return the latest baseline record for *workload_id*, or None."""
        latest: dict | None = None
        for record in self._iter_records():
            if record.get("workload_id") == workload_id:
                latest = record
        return latest

    def load_all(self) -> dict[str, dict]:
        """Return a mapping of workload_id → latest baseline record."""
        result: dict[str, dict] = {}
        for record in self._iter_records():
            wid = record.get("workload_id")
            if wid:
                result[wid] = record
        return result

    def get_or_measure(
        self,
        workload_id: str,
        stt: Any,
        default_config: dict[str, Any],
        n_runs: int = 3,
    ) -> dict:
        """Return the stored baseline for *workload_id*, measuring it first if absent.

        If the baseline already exists in the store it is returned immediately
        without touching the database.  Otherwise *n_runs* calls to
        ``stt.test_config(default_config)`` are made, the median TPS is
        computed, the result is persisted, and then returned.

        On any failure the method does **not** raise; it returns
        ``{"baseline_tps": 0.0, "baseline_runs": []}`` and logs a warning.
        """
        existing = self.load(workload_id)
        if existing is not None:
            return {
                "baseline_tps": existing["baseline_tps"],
                "baseline_runs": existing["baseline_runs"],
            }

        try:
            runs: list[float] = [float(stt.test_config(default_config)) for _ in range(n_runs)]
            median_tps = float(statistics.median(runs))
            self.save(workload_id, median_tps, runs, default_config)
            return {"baseline_tps": median_tps, "baseline_runs": runs}
        except Exception:
            logger.warning(
                "BaselineStore: measurement failed for workload '%s'",
                workload_id,
                exc_info=True,
            )
            return {"baseline_tps": 0.0, "baseline_runs": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_records(self):
        """Yield parsed records from the JSONL file, skipping malformed lines."""
        if not os.path.exists(self.store_path):
            return
        with open(self.store_path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "BaselineStore: skipping malformed JSON on line %d of %s",
                        lineno,
                        self.store_path,
                    )
