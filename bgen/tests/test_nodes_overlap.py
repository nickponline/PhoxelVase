"""No two panels may overlap in volume.

Each panel profile is convex-decomposed; every convex part extruded by the panel thickness is
a convex polytope (set of half-spaces). Two polytopes overlap with real volume iff a ball of
radius > tol fits in their intersection (Chebyshev-center LP).
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog

from bgen.pipeline import run_nodes
from bgen.spec import preset_spec

try:
    from bgen.geom.convex2d import convex_decompose
except ImportError:  # pragma: no cover
    convex_decompose = None

TOL = 2e-3   # 2 mm interpenetration ball radius


def _parts(profile):
    if convex_decompose is not None:
        return convex_decompose(profile)
    import mapbox_earcut as earcut
    rings = [np.asarray(profile.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in profile.interiors]
    V = np.vstack(rings)
    tri = earcut.triangulate_float64(V, np.cumsum([len(r) for r in rings]).astype(np.uint32)).reshape(-1, 3)
    return [V[t] for t in tri]


def polytopes(panels):
    """-> list of (panel_id, planes (m,4) unit normals n·x + d <= 0, aabb_lo, aabb_hi)."""
    out = []
    for p in panels:
        R, T = p.frame[:3, :3], p.frame[:3, 3]
        for poly in _parts(p.profile):
            P = np.asarray(poly, float)
            x, y = P[:, 0], P[:, 1]
            if (x * np.roll(y, -1) - np.roll(x, -1) * y).sum() < 0:
                P = P[::-1]
            n = len(P)
            pl = []
            for i in range(n):
                e = P[(i + 1) % n] - P[i]
                nn = np.array([e[1], -e[0], 0.0])
                nn /= np.linalg.norm(nn)
                pl.append((nn, -nn[:2] @ P[i]))
            pl += [(np.array([0, 0, -1.0]), 0.0), (np.array([0, 0, 1.0]), -p.thickness)]
            planes = np.array([[*(R @ a), b - (R @ a) @ T] for a, b in pl])
            loc = np.vstack([np.column_stack([P, np.zeros(n)]), np.column_stack([P, np.full(n, p.thickness)])])
            W = loc @ R.T + T
            out.append((p.id, planes, W.min(0), W.max(0)))
    return out


def cheb_radius(A: np.ndarray) -> float:
    """Max r such that a ball of radius r fits in {x: A[:, :3] x + A[:, 3] <= 0}."""
    c = np.array([0, 0, 0, -1.0])
    Aub = np.column_stack([A[:, :3], np.ones(len(A))])
    bub = -A[:, 3]
    res = linprog(c, A_ub=Aub, b_ub=bub, bounds=[(None, None)] * 3 + [(0, 1.0)], method="highs")
    return float(res.x[3]) if res.status == 0 else 0.0


def find_overlaps(panels, tol=TOL, max_report=20):
    P = polytopes(panels)
    lo = np.array([q[2] for q in P])
    hi = np.array([q[3] for q in P])
    pid = np.array([q[0] for q in P])
    bad = []
    order = np.argsort(lo[:, 0])
    for ii, i in enumerate(order):
        for j in order[ii + 1:]:
            if lo[j, 0] > hi[i, 0] - tol:
                break
            if pid[i] == pid[j]:
                continue
            if np.any(lo[j] > hi[i] - tol) or np.any(lo[i] > hi[j] - tol):
                continue
            r = cheb_radius(np.vstack([P[i][1], P[j][1]]))
            if r > tol:
                bad.append((int(pid[i]), int(pid[j]), r))
                if len(bad) >= max_report:
                    return bad
    return bad


def describe(ctx, bad):
    k = {p.id: (p.kind, p.tags.get("role"), p.tags.get("floor")) for p in ctx.panels}
    return [(a, k[a], b, k[b], round(r, 4)) for a, b, r in bad]


@pytest.mark.parametrize("preset,seed", [("house", 1), ("house", 2), ("kyoto", 1), ("warehouse", 2),
                                         ("apartment", 1), ("office", 3)])
def test_no_panel_overlap(preset, seed):
    ctx = run_nodes(preset_spec(preset), seed)
    bad = find_overlaps(ctx.panels)
    assert not bad, describe(ctx, bad)


def test_checker_detects_overlap():
    from shapely.geometry import box
    from bgen.model import BuildingContext
    from bgen.nodes._util import add_panel, horiz_frame, wall_frame
    ctx = BuildingContext(spec={}, params={})
    add_panel(ctx, "floor", horiz_frame(0.0), box(0, 0, 4, 4), 0.3, "concrete")
    add_panel(ctx, "floor", horiz_frame(0.3), box(0, 0, 4, 4), 0.3, "concrete")          # touching: ok
    assert not find_overlaps(ctx.panels)
    add_panel(ctx, "ext_wall", wall_frame([1, 1], [1, 0], 0.2), box(0, 0, 2, 2), 0.2, "brick")  # pierces both
    bad = find_overlaps(ctx.panels)
    assert {(a, b) for a, b, _ in bad} == {(0, 2), (1, 2)}
