"""`balconies` node: cantilever slab + parapet + balcony_rail socket at bay positions.

reserve() picks bays (same bays on every chosen floor of a facade) before `openings` runs,
so openings can put a balcony door in those bays. The balcony slab sits at the floor slab
level outside the facade (its inner face meets the floor slab's edge face); the parapet is
an open wall chain on the slab whose ends butt into the facade wall's outer face.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..model import BuildingContext, Socket
from ..rng import rv
from ._util import add_panel, emit_chain, horiz_frame, socket_xform, wall_frame
from .exterior_walls import seg_key

DIRS = {"south": (0, -1), "north": (0, 1), "east": (1, 0), "west": (-1, 0)}


def reserve(ctx: BuildingContext, params: dict, rng) -> None:
    bo = ctx.blockout
    ctx.meta["balcony_bays"] = []
    names = params.get("facades", []) or []
    if isinstance(names, str):
        names = [names]
    if not names or len(bo.floor_polys) < 2:
        return
    every = max(1, int(params.get("every_n_floors", 1)))
    prob = float(params.get("probability", 0.6))
    min_w = float(params.get("min_width", 2.4))
    depth = float(rv(params.get("depth", [1.2, 1.5]), rng))
    allfp = unary_union(bo.floor_polys)
    chosen: dict = {}
    reserved: dict[int, list[Polygon]] = {}
    bays_all = ctx.meta.get("bays", {})
    for k in range(1, len(bo.floor_polys)):
        if k % every:
            continue
        for P0, P1 in bo.facades[k]:
            d = (P1 - P0) / np.linalg.norm(P1 - P0)
            n = np.array([d[1], -d[0]])
            if not any(n @ np.asarray(DIRS[f], float) > 0.9 for f in names if f in DIRS):
                continue
            key = seg_key(P0, P1)
            bays = bays_all.get(key)
            if not bays:
                continue
            lk = "%.2f|%.2f,%.2f" % (float(n @ P0), n[0], n[1])
            for bi, (b0, b1) in enumerate(zip(bays[:-1], bays[1:])):
                if b1 - b0 < min_w:
                    continue
                ck = (lk, round(float(d @ P0 + b0), 2))
                if ck not in chosen:
                    chosen[ck] = bool(rng.uniform() < prob)
                if not chosen[ck]:
                    continue
                x0, x1 = b0 + 0.1, b1 - 0.1
                q = [P0 + x0 * d, P0 + x0 * d + depth * n, P0 + x1 * d + depth * n, P0 + x1 * d]
                rect = Polygon(q)
                if rect.intersection(allfp).area > 1e-6:
                    continue
                if any(rect.buffer(0.05).intersects(o) for o in reserved.get(k, [])):
                    continue
                reserved.setdefault(k, []).append(rect)
                ctx.meta["balcony_bays"].append(dict(floor=k, facade=key, x0=float(b0), x1=float(b1),
                                                     bx0=float(x0), bx1=float(x1), depth=depth,
                                                     P0=[float(v) for v in P0], d=[float(v) for v in d]))


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    if "balcony_bays" not in ctx.meta:
        reserve(ctx, params, rng)
    bo = ctx.blockout
    ts = float(ctx.params["slab_thickness"])
    tp = float(params.get("parapet_thickness", 0.15))
    hp = float(params.get("parapet_height", 1.0))
    mat = params.get("material", ctx.params.get("slab_material", "concrete"))
    pmat = params.get("parapet_material", mat)
    for bb in ctx.meta["balcony_bays"]:
        k = bb["floor"]
        P0, d = np.asarray(bb["P0"]), np.asarray(bb["d"])
        n = np.array([d[1], -d[0]])
        x0, x1, dd = bb["bx0"], bb["bx1"], bb["depth"]
        q = [P0 + x0 * d, P0 + x0 * d + dd * n, P0 + x1 * d + dd * n, P0 + x1 * d]
        z = bo.floor_z[k]
        slab = add_panel(ctx, "balcony", horiz_frame(z - ts), Polygon(q), ts, mat,
                         {"floor": k, "role": "balcony_slab", "cantilever": True})
        if slab is None:
            continue
        emit_chain(ctx, q, False, tp, z, z + hp, "parapet", pmat,
                   {"floor": k, "role": "balcony_parapet", "host": slab.id})
        mid = P0 + 0.5 * (x0 + x1) * d + (dd - tp / 2) * n
        ctx.sockets.append(Socket("balcony_rail", socket_xform([mid[0], mid[1], z + hp], [d[0], d[1], 0], [0, 0, 1]),
                                  slab.id))
        bb["slab"] = slab.id
    return ctx
