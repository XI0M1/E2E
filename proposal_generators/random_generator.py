"""
Random proposal generators for PostgreSQL knob configurations.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List

from scipy.stats import qmc

from proposal_generators.base import ConfigDict, HistorySamples, KnobConstraints, ProposalGenerator, WorkloadFeatures


class RandomProposalGenerator(ProposalGenerator):
    """Generate proposals using uniform random sampling or Latin hypercube sampling."""

    def __init__(
        self,
        strategy: str = "uniform",
        seed: int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(logger=logger)
        if strategy not in {"uniform", "lhs"}:
            raise ValueError(f"Unsupported random proposal strategy: {strategy}")
        self._strategy = strategy
        self._seed = seed
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "random" if self._strategy == "uniform" else "lhs"

    def generate(
        self,
        workload_features: WorkloadFeatures,
        history: HistorySamples,
        constraints: KnobConstraints,
        n: int = 10,
    ) -> List[ConfigDict]:
        if n <= 0:
            return []

        knob_items = list(constraints.items())
        if self._strategy == "uniform":
            proposals = [self._generate_uniform(knob_items, constraints) for _ in range(n)]
        else:
            proposals = self._generate_lhs(knob_items, constraints, n)

        self.logger.info(
            "Generated %s proposals using %s strategy",
            len(proposals),
            self._strategy,
        )
        return proposals

    def _generate_uniform(
        self,
        knob_items: List[tuple[str, Dict[str, Any]]],
        constraints: KnobConstraints,
    ) -> ConfigDict:
        proposal: ConfigDict = {}
        for knob_name, knob_detail in knob_items:
            knob_type = knob_detail.get("type")
            if knob_type == "integer":
                raw_value = self._rng.uniform(
                    float(knob_detail.get("min", 0)),
                    float(knob_detail.get("max", 0)),
                )
                proposal[knob_name] = self._snap_to_step(raw_value, knob_detail)
            elif knob_type == "float":
                raw_value = self._rng.uniform(
                    float(knob_detail.get("min", 0.0)),
                    float(knob_detail.get("max", 0.0)),
                )
                proposal[knob_name] = self._snap_to_step(raw_value, knob_detail)

        return self.validate_proposal(proposal, constraints)

    def _generate_lhs(
        self,
        knob_items: List[tuple[str, Dict[str, Any]]],
        constraints: KnobConstraints,
        n: int,
    ) -> List[ConfigDict]:
        dimensions = len(knob_items)
        if dimensions == 0:
            return []

        sampler = qmc.LatinHypercube(d=dimensions, seed=self._seed)
        design = sampler.random(n=n)

        proposals: List[ConfigDict] = []
        for row in design:
            proposal: ConfigDict = {}
            for index, (knob_name, knob_detail) in enumerate(knob_items):
                knob_type = knob_detail.get("type")
                low = float(knob_detail.get("min", 0))
                high = float(knob_detail.get("max", low))
                raw_value = low + row[index] * (high - low)

                if knob_type in {"integer", "float"}:
                    proposal[knob_name] = self._snap_to_step(raw_value, knob_detail)

            proposals.append(self.validate_proposal(proposal, constraints))

        return proposals
