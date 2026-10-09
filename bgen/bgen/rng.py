"""Deterministic RNG streams.

Every consumer gets its own stream derived from a stable hash of (seed, keys...), so editing
one feature node's parameters never reshuffles the random draws of any other node, and
per-panel fracture is reproducible regardless of process scheduling.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def stable_hash(*keys: Any) -> int:
    """128-bit stable hash of a tuple of simple values (independent of PYTHONHASHSEED)."""
    h = hashlib.blake2b(repr(tuple(keys)).encode("utf-8"), digest_size=16)
    return int.from_bytes(h.digest(), "little")


def stream(*keys: Any) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(stable_hash(*keys)))


def spec_rng(seed: int) -> np.random.Generator:
    return stream("spec", int(seed))


def node_rng(seed: int, node_index: int, node_name: str) -> np.random.Generator:
    return stream("node", int(seed), int(node_index), str(node_name))


def reserve_rng(seed: int, node_index: int, node_name: str) -> np.random.Generator:
    return stream("reserve", int(seed), int(node_index), str(node_name))


def panel_rng(seed: int, panel_id: int) -> np.random.Generator:
    return stream("panel", int(seed), int(panel_id))


def is_range(v: Any) -> bool:
    return (isinstance(v, (list, tuple)) and len(v) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v))


def rv(v: Any, rng: np.random.Generator) -> Any:
    """Sample `v` if it is a numeric [lo, hi] range (ints -> inclusive integer), else return it."""
    if not is_range(v):
        return v
    lo, hi = v
    if isinstance(lo, int) and isinstance(hi, int):
        return int(rng.integers(min(lo, hi), max(lo, hi) + 1))
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def choice(v: Any, rng: np.random.Generator) -> Any:
    """Pick one element if `v` is a list of non-numbers (e.g. [rect, L]); else return v."""
    if isinstance(v, (list, tuple)) and v and not is_range(v):
        return v[int(rng.integers(len(v)))]
    return v
