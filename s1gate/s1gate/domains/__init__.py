"""Domain packs: record parsers and rule checks for one kind of state.

The core of s1gate (schema, backends, metrics, calibration, thresholds, gate) knows nothing
about POS rows; everything that does lives here.
"""
from __future__ import annotations

DOMAINS = ("pos",)


def get(name: str):
    if name == "pos":
        from . import pos
        return pos
    raise ValueError(f"unknown domain {name!r} (known: {', '.join(DOMAINS)})")
