"""Bounded 2D Voronoi via mirrored seeds (DESIGN.md §2.5 step 3)."""
from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi

from .convex2d import clean_ring, ensure_ccw


def bounded_voronoi(seeds: np.ndarray, bounds: tuple[float, float, float, float]) -> list[np.ndarray]:
    """Voronoi cells of `seeds` clipped to the box `bounds` = (minx, miny, maxx, maxy).

    Seeds are mirrored across the four box sides so that every original cell is bounded
    and its outer edges lie exactly on the box. Returns one convex CCW polygon per seed,
    in seed order (seeds outside the box are clamped into it; exact duplicates get an
    empty (0,2) cell).
    """
    minx, miny, maxx, maxy = map(float, bounds)
    box = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])
    S = np.asarray(seeds, dtype=np.float64).reshape(-1, 2)
    N = len(S)
    if N == 0:
        return []
    w, h = maxx - minx, maxy - miny
    eps = 1e-9 * max(w, h, 1e-12)
    S = np.column_stack([np.clip(S[:, 0], minx + eps, maxx - eps), np.clip(S[:, 1], miny + eps, maxy - eps)])
    _, first, inv = np.unique(S, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    uniq_idx = np.sort(first)
    if len(uniq_idx) == 1:
        out = [np.zeros((0, 2)) for _ in range(N)]
        out[uniq_idx[0]] = box.copy()
        return out
    U = S[uniq_idx]
    M = len(U)
    pts = np.vstack([
        U,
        np.column_stack([2 * minx - U[:, 0], U[:, 1]]),
        np.column_stack([2 * maxx - U[:, 0], U[:, 1]]),
        np.column_stack([U[:, 0], 2 * miny - U[:, 1]]),
        np.column_stack([U[:, 0], 2 * maxy - U[:, 1]]),
    ])
    vor = Voronoi(pts)
    V = vor.vertices
    lo, hi = np.array([minx, miny]), np.array([maxx, maxy])
    out = [np.zeros((0, 2)) for _ in range(N)]
    for k in range(M):
        reg = vor.regions[vor.point_region[k]]
        if -1 in reg or len(reg) < 3:  # cannot happen for interior seeds; be safe
            raise RuntimeError("unbounded Voronoi cell for an interior seed")
        cell = np.clip(V[reg], lo, hi)
        cell = clean_ring(cell, tol=1e-12 * max(w, h))
        out[uniq_idx[k]] = ensure_ccw(cell)
    return out
