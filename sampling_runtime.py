"""
Sampling runtime helpers for large-scale Phase 1 generation.

Provides:
- append-only metadata logging
- resume support based on stable sample keys
- failure isolation bookkeeping
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Set


class SamplingRunRecorder:
    def __init__(self, metadata_path: str, resume: bool = False):
        self.metadata_path = metadata_path
        self.resume = resume
        self.completed_keys: Set[str] = set()
        directory = os.path.dirname(metadata_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if resume and os.path.exists(metadata_path):
            self._load_existing()

    def _load_existing(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                if record.get("status") == "success":
                    sample_key = record.get("sample_key")
                    if sample_key:
                        self.completed_keys.add(sample_key)

    def build_sample_key(self, workload_id: str, sample_kind: str, config: Dict[str, Any]) -> str:
        canonical = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()
        return f"{workload_id}:{sample_kind}:{digest}"

    def should_skip(self, sample_key: str) -> bool:
        return self.resume and sample_key in self.completed_keys

    def record(self, payload: Dict[str, Any]):
        record = dict(payload)
        record.setdefault("timestamp", time.time())
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if record.get("status") == "success" and record.get("sample_key"):
            self.completed_keys.add(record["sample_key"])
