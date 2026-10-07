"""`exterior_walls` node: one wall panel per facade segment per floor, split into bays.

Walls sit between slabs (rule 2) and inside the footprint (outer face flush with the facade).
Corners follow `corner_rule` (rule 3): `x_runs_through` (X-ish wall runs through convex
corners and extends over reflex corners, Y-ish walls butt into it) or `post` (corner posts).
Non-orthogonal corners always get corner posts so every contact stays coplanar.
"""
from __future__ import annotations

import numpy as np

from ..model import BuildingContext
from ..rng import rv
from ._util import chain_walls, emit_chain, parts


def seg_key(P0, P1) -> str:
    return "%.2f,%.2f>%.2f,%.2f" % (P0[0], P0[1], P1[0], P1[1])


def make_bays(L: float, bw, rng) -> list[float]:
    lo, hi = (bw if isinstance(bw, (list, tuple)) else (bw, bw))
    mean = 0.5 * (lo + hi)
    n = max(1, int(round(L / mean)))
    w = np.array([rv([float(lo), float(hi)], rng) if lo != hi else float(lo) for _ in range(n)])
    w *= L / w.sum()
    return [0.0] + [float(round(x, 4)) for x in np.cumsum(w)[:-1]] + [float(L)]


def reserve(ctx: BuildingContext, params: dict, rng) -> None:
    """Sample bay rhythms per facade segment; identical segments on different floors share them."""
    bays = {}
    bw = params.get("bay_width", [3.0, 4.5])
    for k, segs in enumerate(ctx.blockout.facades):
        for P0, P1 in segs:
            key = seg_key(P0, P1)
            if key not in bays:
                bays[key] = make_bays(float(np.linalg.norm(P1 - P0)), bw, rng)
    ctx.meta["bays"] = bays


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    if "bays" not in ctx.meta:
        reserve(ctx, params, rng)
    bo = ctx.blockout
    t = float(ctx.params["ext_wall_thickness"])
    ts = float(ctx.params["slab_thickness"])
    mat = params.get("material", ctx.params.get("ext_wall_material", "concrete"))
    rule = params.get("corner_rule", "x_runs_through")
    for k, fp in enumerate(bo.floor_polys):
        z0, z1 = bo.floor_z[k], bo.floor_z[k + 1] - ts
        for part in parts(fp):
            pts = np.asarray(part.exterior.coords)[:-1]

            def extra(w):
                key = seg_key(w["P0"], w["P1"])
                return {"facade": key, "bays": ctx.meta["bays"].get(key) or make_bays(w["L"], 4.0, rng)}

            emit_chain(ctx, pts, True, t, z0, z1, "ext_wall", mat, {"floor": k, "role": "facade"},
                       corner_rule=rule, extra=extra)
    return ctx
