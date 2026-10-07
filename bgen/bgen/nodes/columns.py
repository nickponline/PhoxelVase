"""`columns` node: structural grid columns inside each floor plus inner-corner columns.

Columns occupy their own footprint (rule 4): they keep clear of exterior walls (except corner
columns, which sit flush in the inner corner touching both walls' inner faces) and of stair
cores. Interior walls are clipped around them by the `rooms` node.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import box

from ..model import BuildingContext
from ..rng import rv
from ._util import WallSpec, parts, snap, snapf


def column_spec(x0, y0, x1, y1, z0, z1, mat, tags) -> WallSpec:
    # d = +X, right-hand normal = -Y: base line at y1, thickness down to y0.
    return WallSpec(A=snap([x0, y1]), d=np.array([1.0, 0.0]), x0=0.0, x1=snapf(x1 - x0),
                    t=snapf(y1 - y0), z0=z0, z1=z1, kind="column", material=mat, tags=tags)


def core_keepouts(ctx, k):
    out = []
    for c in ctx.meta.get("stair_cores", []):
        if k in c["floors"]:
            out.append(box(*c["outer"]))
    return out


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    bo = ctx.blockout
    t = float(ctx.params["ext_wall_thickness"])
    ts = float(ctx.params["slab_thickness"])
    g = float(rv(params.get("grid", 6.0), rng))
    s = float(params.get("size", 0.5))
    clear = float(params.get("clearance", 1.2))
    mat = params.get("material", ctx.params.get("column_material", "concrete"))
    corners = params.get("corners", True)
    base = bo.floor_polys[0]
    bx0, by0, bx1, by1 = base.bounds
    nx, ny = max(1, int((bx1 - bx0) // g)), max(1, int((by1 - by0) // g))
    ox = bx0 + 0.5 * ((bx1 - bx0) - nx * g)
    oy = by0 + 0.5 * ((by1 - by0) - ny * g)
    xs = [round(ox + i * g, 2) for i in range(nx + 1)]
    ys = [round(oy + j * g, 2) for j in range(ny + 1)]
    ctx.meta["column_grid"] = {"xs": xs, "ys": ys, "size": s}
    for k, fp in enumerate(bo.floor_polys):
        z0, z1 = bo.floor_z[k], bo.floor_z[k + 1] - ts
        keep = core_keepouts(ctx, k)
        placed = []
        for part in parts(fp):
            inner = part.buffer(-(t + clear), join_style=2)
            for x in xs:
                for y in ys:
                    sq = box(x - s / 2, y - s / 2, x + s / 2, y + s / 2)
                    if inner.is_empty or not inner.contains(sq):
                        continue
                    if any(sq.buffer(clear).intersects(c) for c in keep):
                        continue
                    placed.append(sq)
            if corners:
                P = np.asarray(part.exterior.coords)[:-1]
                for i in range(len(P)):
                    a, v, b = P[i - 1], P[i], P[(i + 1) % len(P)]
                    d1 = (v - a) / np.linalg.norm(v - a)
                    d2 = (b - v) / np.linalg.norm(b - v)
                    if abs(d1 @ d2) > 1e-6 or d1[0] * d2[1] - d1[1] * d2[0] <= 0:
                        continue
                    M = v - t * d1 + t * d2
                    q = np.array([M, M - s * d1, M - s * d1 + s * d2, M + s * d2])
                    sq = box(*q.min(0), *q.max(0))
                    if not part.contains(sq) or any(sq.buffer(clear).intersects(c) for c in keep):
                        continue
                    if any(sq.intersects(o) for o in placed):
                        continue
                    placed.append(sq)
        for sq in placed:
            x0, y0, x1, y1 = sq.bounds
            column_spec(x0, y0, x1, y1, z0, z1, mat, {"floor": k, "role": "column"}).to_panel(ctx)
    return ctx
