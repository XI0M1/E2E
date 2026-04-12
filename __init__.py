"""
Top-level package exports for the PostgreSQL auto-tuning project.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Database Tuning Team"

from surrogate.train_surrogate import train_surrogate
from Database import Database
from Vectorlib import VectorLibrary
from stress_testing_tool import stress_testing_tool


def tune(*args, **kwargs):
    """
    Lazy wrapper around controller.tune.

    This avoids importing the legacy tuning chain during package import, which
    keeps the package safe even when optional optimization backends evolve.
    """
    from controller import tune as _tune

    return _tune(*args, **kwargs)


__all__ = [
    "tune",
    "train_surrogate",
    "Database",
    "VectorLibrary",
    "stress_testing_tool",
]
