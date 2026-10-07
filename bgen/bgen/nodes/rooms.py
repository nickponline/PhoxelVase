"""`rooms` node: BSP room partition + interior walls (DESIGN §2.4).

Per floor the clear interior (inside exterior walls, minus stair cores incl. their walls) is
split recursively by axis-aligned walls of thickness `int_wall_thickness`. Each wall is cut
from its parent cell and spans it fully, so it butts into exterior-wall inner faces, core
walls or earlier walls (coplanar contacts). Walls are clipped around columns.

Outputs: ctx.rooms, interior wall panels, ctx.meta['room_links'] (door-capable intervals:
{floor, wall, a, b, x0, x1}, b = -1 means outside), ctx.meta['core_rooms'].
The partition is retried (same RNG stream) until every room on the floor is reachable from
the floor's root (stair core, or the outside on the ground floor) through door-capable links.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box

from ..model import BuildingContext, Room
from ..rng import rv
from ._util import (MIN_SOLID, WallSpec, blocked_intervals, intersect_iv, line_intervals, merge_iv,
                    parts, plan_footprint, shrink_iv, snap, snapf, snap_geom, subtract_iv, wallspec_of,
                    z_range)
from .stairs import core_wall_specs

BIG = 1e5


# ----------------------------------------------------------------------------- splitting

def _axis_iv(C: Polygon, axis: str, c: float):
    """Intervals (along the other axis) of C on the line axis=c."""
    x0, y0, x1, y1 = C.bounds
    ln = LineString([(c, y0 - 1), (c, y1 + 1)]) if axis == "x" else LineString([(x0 - 1, c), (x1 + 1, c)])
    g = C.intersection(ln)
    out = []
    for q in getattr(g, "geoms", [g]):
        if q.is_empty or q.geom_type != "LineString":
            continue
        cs = np.asarray(q.coords)
        k = 1 if axis == "x" else 0
        out.append((float(cs[:, k].min()), float(cs[:, k].max())))
    return merge_iv(out)


def split_cell(C: Polygon, axis: str, c: float, t: float):
    lo, hi = c - t / 2, c + t / 2
    I = intersect_iv(_axis_iv(C, axis, lo + 1e-5), _axis_iv(C, axis, hi - 1e-5))
    walls = [(axis, c, a, b) for a, b in I if b - a > 0.05]
    if axis == "x":
        A = box(-BIG, -BIG, lo, BIG)
        B = box(hi, -BIG, BIG, BIG)
    else:
        A = box(-BIG, -BIG, BIG, lo)
        B = box(-BIG, hi, BIG, BIG)
    ca = [snap_geom(p) for p in parts(C.intersection(A))]
    cb = [snap_geom(p) for p in parts(C.intersection(B))]
    ca = [p for q in ca for p in parts(q) if p.area > 0.5]
    cb = [p for q in cb for p in parts(q) if p.area > 0.5]
    return walls, ca, cb


def _vertex_ok(C: Polygon, axis: str, c: float, t: float) -> bool:
    k = 0 if axis == "x" else 1
    vs = np.asarray(C.exterior.coords)[:, k]
    for ring in C.interiors:
        vs = np.concatenate([vs, np.asarray(ring.coords)[:, k]])
    lo, hi = c - t / 2, c + t / 2
    for v in vs:
        if abs(v - lo) < 1e-6 or abs(v - hi) < 1e-6:
            continue
        if lo - 0.05 < v < hi + 0.05:
            return False
    return True


def _ok_room(p: Polygon, min_room: float, min_w: float) -> bool:
    return p.area >= min_room and not p.buffer(-min_w / 2 + 1e-3, join_style=2).is_empty


def wall_spec(w, t, z0, z1, mat, k) -> WallSpec:
    axis, c, a, b = w
    if axis == "x":   # line x=c, runs +Y, thickness toward +X
        A, d = np.array([c - t / 2, a]), np.array([0.0, 1.0])
    else:             # line y=c, runs +X, thickness toward -Y
        A, d = np.array([a, c + t / 2]), np.array([1.0, 0.0])
    return WallSpec(A=snap(A), d=d, x0=0.0, x1=snapf(b - a), t=t, z0=z0, z1=z1, kind="int_wall",
                    material=mat, tags={"floor": k, "role": "partition"})


def bsp(free: Polygon, P: dict, t: float, rng, cores: list[Polygon]):
    """Returns (cells: list[(Polygon, tag|None)], walls: list[(axis, c, a, b)])."""
    min_room, min_w = P["min_room"], P["min_w"]
    cells, walls = [], []
    queue = deque()
    # optional corridor along the long axis
    x0, y0, x1, y1 = free.bounds
    long_x = (x1 - x0) >= (y1 - y0)
    short = (y1 - y0) if long_x else (x1 - x0)
    if P["corridor"] and short >= P["corridor_min"]:
        wc = P["corridor_w"]
        ax = "y" if long_x else "x"
        lo_b, hi_b = (y0, y1) if long_x else (x0, x1)
        cands = []
        for co in cores:
            cx0, cy0, cx1, cy1 = co.bounds
            clo, chi = (cy0, cy1) if long_x else (cx0, cx1)
            cands += [(chi, chi + wc), (clo - wc, clo)]
        mid = 0.5 * (lo_b + hi_b) + rng.uniform(-1.0, 1.0)
        cands.append((mid - wc / 2, mid + wc / 2))
        for ya, yb in cands:
            if ya - t - min_w < lo_b or yb + t + min_w > hi_b:
                continue
            c1, c2 = ya - t / 2, yb + t / 2
            if not (_vertex_ok(free, ax, c1, t) and _vertex_ok(free, ax, c2, t)):
                continue
            w1, A1, B1 = split_cell(free, ax, c1, t)
            mids, rest_hi, w2 = [], [], []
            for cB in B1:
                ww, A2, B2 = split_cell(cB, ax, c2, t)
                w2 += ww
                mids += A2
                rest_hi += B2
            if not (w1 and w2 and mids):
                continue
            walls += w1 + w2
            cells += [(m, "corridor") for m in mids]
            queue.extend(A1 + rest_hi)
            break
        else:
            queue.append(free)
    else:
        queue.append(free)
    while queue:
        C = queue.popleft()
        cap = float(rv(P["max_room"], rng))
        if C.area < max(cap, 2 * min_room):
            cells.append((C, None))
            continue
        bx0, by0, bx1, by1 = C.bounds
        prefer = "x" if (bx1 - bx0) >= (by1 - by0) else "y"
        done = False
        for attempt in range(10):
            ax = prefer if (attempt % 3 != 2) else ("y" if prefer == "x" else "x")
            lo, hi = (bx0, bx1) if ax == "x" else (by0, by1)
            a, b = lo + min_w + t / 2, hi - min_w - t / 2
            if b <= a:
                continue
            c = round(float(rng.uniform(a, b)) / 0.05) * 0.05
            if not _vertex_ok(C, ax, c, t):
                continue
            ws, ca, cb = split_cell(C, ax, c, t)
            if not ws or not ca or not cb:
                continue
            if not all(_ok_room(p, min_room, min_w) for p in ca + cb):
                continue
            walls += ws
            queue.extend(ca + cb)
            done = True
            break
        if not done:
            cells.append((C, None))
    return cells, walls


def clip_by_columns(ws: WallSpec, cols: list[Polygon], min_len=0.3) -> list[WallSpec]:
    fp = ws.footprint()
    cut = []
    for c in cols:
        if c.intersects(fp) and c.intersection(fp).area > 1e-8:
            cs = np.asarray(c.exterior.coords)
            xs = (cs - ws.A) @ ws.d
            cut.append((float(xs.min()), float(xs.max())))
    if not cut:
        return [ws]
    out = []
    for a, b in subtract_iv([(ws.x0, ws.x1)], cut):
        if b - a >= min_len:
            out.append(WallSpec(A=ws.A, d=ws.d, x0=snapf(a), x1=snapf(b), t=ws.t, z0=ws.z0, z1=ws.z1,
                                kind=ws.kind, material=ws.material, tags=dict(ws.tags),
                                door_zone=ws.door_zone, no_door=ws.no_door))
    return out


# ----------------------------------------------------------------------------- links

def compute_links(walls: list[WallSpec], ext: list[WallSpec], rooms: list[Polygon], solids: list[Polygon],
                  ground: bool, door_w: float):
    """Door-capable intervals. Returns list of (wall_index_in(walls+ext), a, b, x0, x1)."""
    allw = walls + ext
    fps = [w.footprint() for w in allw]
    tree = shapely.STRtree(rooms) if rooms else None
    stree = shapely.STRtree(fps + solids)
    allsol = fps + solids
    links = []
    for wi, w in enumerate(allw):
        if w.no_door:
            continue
        is_ext = wi >= len(walls)
        if is_ext and not ground:
            continue
        fp = fps[wi]
        others = [allsol[j] for j in stree.query(fp.buffer(0.01)) if j != wi]
        allowed = subtract_iv([(w.x0, w.x1)], blocked_intervals(w, others))
        if w.door_zone is not None:
            allowed = intersect_iv(allowed, [w.door_zone])
        allowed = shrink_iv(allowed, MIN_SOLID)
        if not allowed:
            continue
        near = [int(j) for j in tree.query(fp.buffer(0.01))] if tree is not None else []
        side = {0: [], 1: []}
        for j in near:
            for s in (0, 1):
                iv = line_intervals(rooms[j], w, s)
                if iv:
                    side[s].append((j, iv))
        if is_ext:
            for j, iv in side[0]:
                for a, b in intersect_iv(iv, allowed):
                    if b - a >= door_w:
                        links.append((wi, j, -1, a, b))
        else:
            for j0, iv0 in side[0]:
                for j1, iv1 in side[1]:
                    if j0 == j1:
                        continue
                    for a, b in intersect_iv(intersect_iv(iv0, iv1), allowed):
                        if b - a >= door_w:
                            links.append((wi, j0, j1, a, b))
    return links


def reachable(n_rooms: int, links, roots: list[int]) -> set[int]:
    adj = {i: set() for i in range(-1, n_rooms)}
    for _, a, b, _, _ in links:
        adj[a].add(b)
        adj[b].add(a)
    seen = set(roots)
    q = deque(roots)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def assign_tags(cells, tags: dict, rng) -> list[str]:
    names = list(tags.keys())
    w = np.array([float(tags[k]) for k in names])
    w = w / w.sum()
    out = []
    small = {"toilet", "bathroom"}
    for poly, tag in cells:
        if tag:
            out.append(tag)
            continue
        mrr = poly.minimum_rotated_rectangle
        cs = np.asarray(mrr.exterior.coords)
        e = sorted([np.linalg.norm(cs[1] - cs[0]), np.linalg.norm(cs[2] - cs[1])])
        if e[0] < 2.6 and e[1] / max(e[0], 1e-6) > 3.0:
            out.append("corridor")
            continue
        choice = names[int(rng.choice(len(names), p=w))]
        if choice in small and poly.area > 20:
            others = [n for n in names if n not in small] or names
            choice = others[int(rng.integers(len(others)))]
        out.append(choice)
    return out


# ----------------------------------------------------------------------------- node

def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    bo = ctx.blockout
    te = float(ctx.params["ext_wall_thickness"])
    ti = float(ctx.params["int_wall_thickness"])
    ts = float(ctx.params["slab_thickness"])
    mat = params.get("material", ctx.params.get("int_wall_material", "brick"))
    door_w = float(params.get("door_clear", 1.0))
    P = dict(min_room=float(params.get("min_room", 9.0)), min_w=float(params.get("min_width", 2.2)),
             max_room=params.get("max_room", [25.0, 45.0]), corridor=bool(params.get("corridor", False)),
             corridor_w=float(params.get("corridor_width", 1.8)),
             corridor_min=float(params.get("corridor_min_depth", 11.0)))
    tags_w = params.get("tags", {"room": 1.0})
    if isinstance(tags_w, list):
        tags_w = {t: 1.0 for t in tags_w if t != "corridor"} or {"room": 1.0}
    attempts = int(params.get("attempts", 12))
    cores = ctx.meta.get("stair_cores", [])
    links_out = ctx.meta.setdefault("room_links", [])
    core_rooms = ctx.meta.setdefault("core_rooms", {})
    unreach_floor = ctx.meta.setdefault("rooms_unreachable_at_gen", {})
    ctx.meta["core_walls_built"] = True

    for k, fp in enumerate(bo.floor_polys):
        z0, z1 = bo.floor_z[k], bo.floor_z[k + 1] - ts
        cols = [plan_footprint(p) for p in ctx.panels
                if p.tags.get("floor") == k and p.kind == "column"]
        cols = [c for c in cols if c is not None]
        ext = [wallspec_of(p) for p in ctx.panels if p.kind == "ext_wall" and p.tags.get("floor") == k]
        core_ws, core_polys, core_ids = [], [], []
        for c in cores:
            if k in c["floors"]:
                core_ws += core_wall_specs(ctx, c, k, mat)
                core_polys.append(box(*c["outer"]))
                core_ids.append((c["id"], box(*c["inner"])))
        free_parts = []
        for part in parts(fp):
            inner = snap_geom(part.buffer(-te, join_style=2))
            for ip in parts(inner):
                f = ip
                for cp in core_polys:
                    f = f.difference(cp)
                free_parts += [q for q in parts(snap_geom(f)) if q.area > 0.5]
        best = None
        for att in range(attempts):
            cells, walls = [], []
            for fpart in free_parts:
                cc, ww = bsp(fpart, P, ti, rng, [cp for cp in core_polys if cp.intersects(fpart.buffer(0.5))])
                cells += cc
                walls += ww
            wspecs = []
            for w in walls:
                wspecs += clip_by_columns(wall_spec(w, ti, z0, z1, mat, k), cols)
            room_polys = [cp for _, cp in core_ids] + [c for c, _ in cells]
            room_tags = ["stair"] * len(core_ids) + assign_tags(cells, tags_w, rng)
            int_ws = core_ws + wspecs
            links = compute_links(int_ws, ext, room_polys, cols, k == 0, door_w)
            roots = list(range(len(core_ids)))
            if k == 0:
                roots.append(-1)
            if not roots:
                roots = [int(np.argmax([p.area for p in room_polys]))] if room_polys else []
            seen = reachable(len(room_polys), links, roots)
            miss = [i for i in range(len(room_polys)) if i not in seen]
            ent_ok = (k != 0) or any(b == -1 for _, _, b, _, _ in links)
            score = len(miss) + (0 if ent_ok else 100)
            if best is None or score < best[0]:
                best = (score, room_polys, room_tags, int_ws, links, miss)
            if score == 0:
                break
        _, room_polys, room_tags, int_ws, links, miss = best
        base_id = len(ctx.rooms)
        for i, (poly, tag) in enumerate(zip(room_polys, room_tags)):
            ctx.rooms.append(Room(id=base_id + i, floor=k, polygon=poly, tag=tag))
        for i, (cid, _) in enumerate(core_ids):
            core_rooms[f"{cid}:{k}"] = base_id + i
        for ws in int_ws:
            ws.to_panel(ctx)
        allw = int_ws + ext
        for wi, a, b, x0, x1 in links:
            pid = allw[wi].panel_id
            if pid < 0:
                continue
            links_out.append(dict(floor=k, wall=pid, a=base_id + a, b=(base_id + b) if b >= 0 else -1,
                                  x0=float(x0), x1=float(x1)))
        if miss:
            unreach_floor[str(k)] = [base_id + m for m in miss]
    return ctx
