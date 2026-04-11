"""
Lightweight rule-based proposal generator.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

from proposal_generators.base import ConfigDict, HistorySamples, KnobConstraints, ProposalGenerator, WorkloadFeatures


class HeuristicProposalGenerator(ProposalGenerator):
    """Generate proposals using simple workload-aware tuning heuristics."""

    def __init__(self, seed: int | None = None, logger: logging.Logger | None = None) -> None:
        super().__init__(logger=logger)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "heuristic"

    def generate(
        self,
        workload_features: WorkloadFeatures,
        history: HistorySamples,
        constraints: KnobConstraints,
        n: int = 10,
    ) -> List[ConfigDict]:
        features = workload_features if isinstance(workload_features, dict) else {}
        proposals = [
            self._build_single_proposal(features, constraints)
            for _ in range(max(0, n))
        ]
        self.logger.info("Generated %s proposals using heuristic strategy", len(proposals))
        return proposals

    def _build_single_proposal(
        self,
        features: Dict[str, Any],
        constraints: KnobConstraints,
    ) -> ConfigDict:
        proposal: ConfigDict = {
            knob_name: knob_detail.get("default")
            for knob_name, knob_detail in constraints.items()
            if "default" in knob_detail
        }

        read_ratio = float(features.get("read_ratio", 0.0) or 0.0)
        join_count = int(features.get("join_count", 0) or 0)
        aggregation_count = int(features.get("aggregation_count", 0) or 0)

        triggered = False
        if read_ratio > 80.0:
            triggered = True
            self._scale_numeric_knob(proposal, constraints, "shared_buffers", 1.25)
            self._scale_numeric_knob(proposal, constraints, "effective_cache_size", 1.25)

        if join_count > 20:
            triggered = True
            self._scale_numeric_knob(proposal, constraints, "work_mem", 1.30)

        if aggregation_count > 15:
            triggered = True
            self._scale_numeric_knob(proposal, constraints, "work_mem", 1.20)
            self._scale_numeric_knob(proposal, constraints, "maintenance_work_mem", 1.30)

        noise_ratio = 0.10 if not triggered else 0.05
        for knob_name, knob_detail in constraints.items():
            if knob_name not in proposal:
                continue
            if knob_detail.get("type") not in {"integer", "float"}:
                continue
            proposal[knob_name] = self._apply_noise(
                float(proposal[knob_name]),
                knob_detail,
                noise_ratio,
            )

        return self.validate_proposal(proposal, constraints)

    def _scale_numeric_knob(
        self,
        proposal: ConfigDict,
        constraints: KnobConstraints,
        knob_name: str,
        factor: float,
    ) -> None:
        knob_detail = constraints.get(knob_name)
        if not knob_detail or knob_detail.get("type") not in {"integer", "float"}:
            return
        current_value = float(proposal.get(knob_name, knob_detail.get("default", knob_detail.get("min", 0))))
        proposal[knob_name] = self._snap_to_step(current_value * factor, knob_detail)

    def _apply_noise(
        self,
        value: float,
        knob_detail: Dict[str, Any],
        noise_ratio: float,
    ) -> int | float:
        noise = self._rng.uniform(-noise_ratio, noise_ratio)
        return self._snap_to_step(value * (1.0 + noise), knob_detail)

