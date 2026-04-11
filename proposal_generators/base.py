"""
Base abstraction for pluggable proposal generators.

Proposal generators only produce candidate configurations. They do not talk to
the database, execute workloads, score configurations, or trigger restarts.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Sequence

from safe.subspace_adaptation import Safe

WorkloadFeatures = Mapping[str, Any] | Sequence[Any]
HistorySamples = List[Dict[str, Any]]
KnobConstraints = Dict[str, Dict[str, Any]]
ConfigDict = Dict[str, Any]


class ProposalGenerator(ABC):
    """Abstract base class for candidate configuration generators."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def name(self) -> str:
        """Short strategy name used for logging."""

    @abstractmethod
    def generate(
        self,
        workload_features: WorkloadFeatures,
        history: HistorySamples,
        constraints: KnobConstraints,
        n: int = 10,
    ) -> List[ConfigDict]:
        """Generate `n` candidate configurations."""

    def validate_proposal(self, config: ConfigDict, constraints: KnobConstraints) -> ConfigDict:
        """
        Clamp a generated configuration into the legal search range.

        This performs lightweight range validation locally through the existing
        `Safe` helper before any proposal leaves the generator layer.
        """
        safe_adapter = self._build_safe_adapter(constraints)
        clamped_config = safe_adapter.clamp_config(config, constraints)
        is_valid, violations = safe_adapter.is_valid_config(clamped_config, constraints)
        if not is_valid:
            raise ValueError(f"Invalid proposal after clamp: {violations}")
        return clamped_config

    def _build_safe_adapter(self, constraints: KnobConstraints) -> Safe:
        defaults: Dict[str, Any] = {
            knob_name: knob_detail.get("default", knob_detail.get("min", 0))
            for knob_name, knob_detail in constraints.items()
        }
        lower_bounds = [float(knob_detail.get("min", 0)) for knob_detail in constraints.values()]
        upper_bounds = [float(knob_detail.get("max", 0)) for knob_detail in constraints.values()]
        steps = [float(knob_detail.get("step", 1) or 1) for knob_detail in constraints.values()]
        return Safe(
            default_performance=0.0,
            default_config=defaults,
            best_performance=0.0,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            steps=steps,
        )

    def _snap_to_step(self, value: float, knob_detail: Dict[str, Any]) -> int | float:
        """
        Snap a numeric value onto the knob step grid relative to the knob min.
        """
        minimum = float(knob_detail.get("min", 0))
        step = float(knob_detail.get("step", 1) or 1)
        if step <= 0:
            step = 1.0

        snapped = minimum + round((float(value) - minimum) / step) * step
        if knob_detail.get("type") == "integer":
            return int(round(snapped))

        precision = self._infer_precision(step)
        return round(float(snapped), precision)

    @staticmethod
    def _infer_precision(step: float) -> int:
        step_text = f"{step:.12f}".rstrip("0").rstrip(".")
        if "." not in step_text:
            return 0
        return len(step_text.split(".")[1])

