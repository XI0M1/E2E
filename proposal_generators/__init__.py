"""
Factory and exports for proposal generators.
"""

from __future__ import annotations

from typing import Any

from proposal_generators.base import ProposalGenerator
from proposal_generators.heuristic_generator import HeuristicProposalGenerator
from proposal_generators.random_generator import RandomProposalGenerator


def get_generator(strategy: str, **kwargs: Any) -> ProposalGenerator:
    """
    Build a proposal generator by strategy name.
    """
    normalized = strategy.strip().lower()
    knobs_detail = kwargs.pop("knobs_detail", None)
    if normalized == "random":
        return RandomProposalGenerator(strategy="uniform", **kwargs)
    if normalized == "lhs":
        return RandomProposalGenerator(strategy="lhs", **kwargs)
    if normalized == "heuristic":
        return HeuristicProposalGenerator(**kwargs)
    if normalized == "smac":
        if knobs_detail is None:
            raise ValueError("SMACProposalGenerator requires knobs_detail")
        from proposal_generators.smac_generator import SMACProposalGenerator
        return SMACProposalGenerator(knobs_detail=knobs_detail, **kwargs)
    raise ValueError(f"Unknown proposal generation strategy: {strategy}")


__all__ = [
    "ProposalGenerator",
    "RandomProposalGenerator",
    "HeuristicProposalGenerator",
    "get_generator",
]
