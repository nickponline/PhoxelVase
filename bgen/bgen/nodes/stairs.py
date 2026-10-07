"""`stairs` node: switchback stair cores (DESIGN §2.4).

reserve(): picks core rectangles that are identical on every floor they serve (stored in
ctx.meta['stair_cores']), so columns/rooms can keep out of them.
node(): core walls (if `rooms` did not already build them), slab cut-outs (floors >= 1, the
flight zone u >= Ld of the core), per floor below the top: two inclined flight panels flush
against the core side walls, a mid landing spanning the far end, and cosmetic step prisms.

Core local coords: u along the core's long axis from the inner min corner, v across.
Landing zone u in [0, Ld] is floor on every level; door zone of the core walls is there.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon, box

from ..model import BuildingContext
from ..rng import rv
from ._util import (WallSpec, add_from_world, add_panel, chain_walls, cut_opening, horiz_frame,
                    line_intervals, parts, snap, snapf)


# ----------------------------------------------------------------------------- geometry

def core_dims(ctx: BuildingContext, params: dict) -> dict:
    bo = ctx.blockout
    Hs = [bo.floor_z[k + 1] - bo.floor_z[k] for k in range(len(bo.floor_polys))]
    riser_max = float(params.get("riser_max", 0.185))
    tread = float(params.get("tread", 0.27))
    wf = float(params.get("flight_width", 1.15))
    gap = float(params.get("gap", 0.2))
    Ld = float(params.get("landing", 1.4))
    nmax = max(math.ceil((H / 2) / riser_max) for H in Hs)
    run = nmax * tread
    W = round(2 * wf + gap, 2)
    L = round(2 * Ld + run, 2)
    return dict(W=W, L=L, Ld=Ld, wf=wf, tread=tread, riser_max=riser_max, run_max=run)


def uv_world(c: dict, u: float, v: float) -> np.ndarray:
    x0, y0 = c["inner"][0], c["inner"][1]
    return np.array([x0 + u, y0 + v]) if c["axis"] == "x" else np.array([x0 + v, y0 + u])


def u_dir(c):
    return np.array([1.0, 0, 0]) if c["axis"] == "x" else np.array([0, 1.0, 0])


def v_dir(c):
    return np.array([0, 1.0, 0]) if c["axis"] == "x" else np.array([1.0, 0, 0])


def uv_rect(c, u0, u1, v0, v1) -> Polygon:
    a, b = uv_world(c, u0, v0), uv_world(c, u1, v1)
    return box(min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


def P3(c, u, v, z):
    w = uv_world(c, u, v)
    return np.array([w[0], w[1], z])


# ----------------------------------------------------------------------------- reserve

def reserve(ctx: BuildingContext, params: dict, rng) -> None:
    bo = ctx.blockout
    n = len(bo.floor_polys)
    ctx.meta["stair_cores"] = []
    if n < 2 or int(params.get("cores", 1)) <= 0:
        return
    te = float(ctx.params["ext_wall_thickness"])
    ti = float(ctx.params["int_wall_thickness"])
    dims = core_dims(ctx, params)
    ncores = int(rv(params.get("cores", 1), rng))
    interiors = [fp.buffer(-te, join_style=2) for fp in bo.floor_polys]
    cores = []
    for ci in range(ncores):
        best = None
        for kmax in range(n - 1, 0, -1):
            region = interiors[0]
            for k in range(1, kmax + 1):
                region = region.intersection(interiors[k])
            if region.is_empty:
                continue
            bx0, by0, bx1, by1 = bo.floor_polys[kmax].bounds
            axis = "x" if (bx1 - bx0) >= (by1 - by0) else "y"
            cand_axes = [axis, "y" if axis == "x" else "x"]
            target = np.array(bo.floor_polys[kmax].centroid.coords[0]) + rng.uniform(-1.0, 1.0, 2)
            for margin in (float(params.get("margin", 1.3)), 0.6, 0.0):
                for ax in cand_axes:
                    ow = dims["L"] + 2 * ti if ax == "x" else dims["W"] + 2 * ti
                    oh = dims["W"] + 2 * ti if ax == "x" else dims["L"] + 2 * ti
                    allowed = region.buffer(-margin, join_style=2)
                    if allowed.is_empty:
                        continue
                    ax0, ay0, ax1, ay1 = allowed.bounds
                    xs = np.arange(ax0, ax1 - ow + 1e-9, 0.25)
                    ys = np.arange(ay0, ay1 - oh + 1e-9, 0.25)
                    if not len(xs) or not len(ys):
                        continue
                    X, Y = np.meshgrid(xs, ys)
                    X, Y = X.ravel(), Y.ravel()
                    cx, cy = X + ow / 2, Y + oh / 2
                    if cores:
                        dist = -np.min([np.hypot(cx - o["center"][0], cy - o["center"][1]) for o in cores], axis=0)
                    else:
                        dist = np.hypot(cx - target[0], cy - target[1])
                    for idx in np.argsort(dist, kind="stable"):
                        r = box(X[idx], Y[idx], X[idx] + ow, Y[idx] + oh)
                        if not allowed.contains(r):
                            continue
                        if any(r.buffer(3.0).intersects(box(*o["outer"])) for o in cores):
                            continue
                        ox0, oy0 = round(X[idx] / 0.05) * 0.05, round(Y[idx] / 0.05) * 0.05
                        r = box(ox0, oy0, ox0 + ow, oy0 + oh)
                        if not allowed.buffer(1e-6).contains(r):
                            continue
                        best = (kmax, ax, r)
                        break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        if not best:
            break
        kmax, ax, r = best
        ox0, oy0, ox1, oy1 = r.bounds
        inner = tuple(round(v, 4) for v in (ox0 + ti, oy0 + ti, ox1 - ti, oy1 - ti))
        c = dict(id=ci, axis=ax, outer=tuple(round(v, 4) for v in r.bounds), inner=inner,
                 floors=list(range(0, kmax + 1)), center=[(ox0 + ox1) / 2, (oy0 + oy1) / 2], **dims)
        cores.append(c)
    ctx.meta["stair_cores"] = cores


# ----------------------------------------------------------------------------- core walls

def core_wall_specs(ctx: BuildingContext, c: dict, k: int, material: str) -> list[WallSpec]:
    bo = ctx.blockout
    ti = float(ctx.params["int_wall_thickness"])
    ts = float(ctx.params["slab_thickness"])
    z0, z1 = bo.floor_z[k], bo.floor_z[k + 1] - ts
    ox0, oy0, ox1, oy1 = c["outer"]
    pts = [(ox0, oy0), (ox1, oy0), (ox1, oy1), (ox0, oy1)]
    walls, _ = chain_walls(pts, True, ti)
    landing = uv_rect(c, 0.0, c["Ld"], 0.0, c["W"])
    out = []
    for w in walls:
        ws = WallSpec(A=snap(w["A"]), d=w["d"], x0=snapf(w["x0"]), x1=snapf(w["x1"]), t=ti, z0=z0, z1=z1,
                      kind="int_wall", material=material,
                      tags={"floor": k, "role": "core_wall", "core": c["id"]})
        iv = line_intervals(landing, ws, 0)
        if iv:
            ws.door_zone = (iv[0][0], iv[-1][1])
        else:
            ws.no_door = True
        out.append(ws)
    return out


# ----------------------------------------------------------------------------- node

def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    cores = ctx.meta.get("stair_cores")
    if cores is None:
        reserve(ctx, params, rng)
        cores = ctx.meta["stair_cores"]
    bo = ctx.blockout
    ts = float(ctx.params["slab_thickness"])
    mat = params.get("material", ctx.params.get("slab_material", "concrete"))
    wall_mat = ctx.params.get("int_wall_material", "brick")
    t_f = float(params.get("flight_thickness", 0.18))
    t_l = float(params.get("landing_thickness", 0.2))
    steps = bool(params.get("steps", True))
    links = ctx.meta.setdefault("stair_links", [])
    core_rooms = ctx.meta.get("core_rooms", {})
    for c in cores:
        floors = c["floors"]
        if not ctx.meta.get("core_walls_built"):
            for k in floors:
                for ws in core_wall_specs(ctx, c, k, wall_mat):
                    ws.to_panel(ctx)
        # slab cut-outs above the ground floor
        hole = uv_rect(c, c["Ld"], c["L"], 0.0, c["W"])
        for k in floors:
            if k == 0:
                continue
            for p in ctx.panels:
                if p.kind == "floor" and p.tags.get("floor") == k and p.profile.contains(hole):
                    cut_opening(ctx, p, hole)
                    p.tags.setdefault("stair_holes", []).append(c["id"])
        U, V = u_dir(c), v_dir(c)
        for k in floors:
            if k + 1 not in floors:
                continue
            z0, z2 = bo.floor_z[k], bo.floor_z[k + 1]
            H = z2 - z0
            nr = math.ceil((H / 2) / c["riser_max"])
            r, g = (H / 2) / nr, c["tread"]
            run = nr * g
            zm = z0 + H / 2
            a = math.atan2(r, g)
            Ld, W, wf, L = c["Ld"], c["W"], c["wf"], c["L"]
            tags = {"floor": k, "core": c["id"]}
            # flight 1: rises +u on v in [0, wf]
            nrm1 = -math.sin(a) * U + math.cos(a) * np.array([0, 0, 1.0])
            us, ue = Ld + t_f / math.sin(a), Ld + run
            ze = zm - t_f / math.cos(a)
            f1 = add_from_world(ctx, "stair", [P3(c, us, 0, z0), P3(c, ue, 0, ze), P3(c, ue, wf, ze),
                                               P3(c, us, wf, z0)], nrm1, t_f, mat, dict(tags, flight=1))
            # mid landing
            add_panel(ctx, "landing", horiz_frame(zm - t_l), uv_rect(c, Ld + run, L, 0, W), t_l, mat,
                      dict(tags, role="mid_landing"))
            # flight 2: rises -u on v in [W - wf, W]
            nrm2 = math.sin(a) * U + math.cos(a) * np.array([0, 0, 1.0])
            us2, ue2 = Ld + run - t_f / math.sin(a), Ld
            ze2 = z2 - t_f / math.cos(a)
            f2 = add_from_world(ctx, "stair", [P3(c, us2, W - wf, zm), P3(c, ue2, W - wf, ze2),
                                               P3(c, ue2, W, ze2), P3(c, us2, W, zm)], nrm2, t_f, mat,
                                dict(tags, flight=2))
            if steps:
                for i in range(nr):
                    u, z = Ld + i * g, z0 + i * r
                    add_from_world(ctx, "step", [P3(c, u, 0, z), P3(c, u + g, 0, z + r), P3(c, u, 0, z + r)],
                                   V, wf, mat, dict(tags, cosmetic=True, host=f1.id if f1 else -1))
                    u, z = Ld + run - i * g, zm + i * r
                    add_from_world(ctx, "step", [P3(c, u, W - wf, z), P3(c, u, W - wf, z + r),
                                                 P3(c, u - g, W - wf, z + r)],
                                   V, wf, mat, dict(tags, cosmetic=True, host=f2.id if f2 else -1))
            a_id, b_id = core_rooms.get(f"{c['id']}:{k}"), core_rooms.get(f"{c['id']}:{k + 1}")
            if a_id is not None and b_id is not None:
                links.append([a_id, b_id])
    return ctx
