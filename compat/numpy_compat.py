"""NumPy compatibility helpers for legacy third-party packages."""

from __future__ import annotations

import numpy as np


def patch_numpy_legacy_aliases() -> None:
    """
    Restore removed NumPy aliases expected by older dependencies.

    NumPy 2.x removed a few legacy alias names that older SMAC / ConfigSpace
    stacks may still import or reference. Re-introducing those aliases as
    direct bindings to their modern equivalents is a minimal process-local
    compatibility workaround.
    """
    legacy_aliases = {
        "NaN": np.nan,
        "float_": np.float64,
        "int_": np.int64,
        "complex_": np.complex128,
    }

    for alias_name, target in legacy_aliases.items():
        if not hasattr(np, alias_name):
            setattr(np, alias_name, target)


def apply_numpy_compat() -> None:
    """Backward-compatible wrapper for older local imports."""
    patch_numpy_legacy_aliases()
