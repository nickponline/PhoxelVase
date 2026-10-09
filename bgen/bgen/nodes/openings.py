"""`openings` node: doors + windows cut from wall profiles.

Doors: per floor a BFS spanning tree over the door-capable links from `rooms` (root: stair
core rooms, plus the outside on the ground floor), so every room gets >= 1 door and is
reachable; `per_room` > 1 adds extra doors. Balcony bays (reserved by `balconies`) get a
door. Windows: one per bay on exterior walls in a style (punched, ribbon, arched, round,
storefront), kept >= 0.15 m from panel edges, junctions and other openings. Each window
gets a glass panel (thin, centered in the wall thickness, inside the hole) and a
window_frame socket; doors get a door socket.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
import shapely
from shapely.geometry import Polygon, box

from ..model import BuildingContext, Socket
from ..rng import choice, rv
from ._util import (MIN_SOLID, add_panel, blocked_intervals, cut_opening, merge_iv, plan_footprint,
                    shrink_iv, socket_xform, subtract_iv, wallspec_of, z_range)

GLASS_T = 0.02


def shape_poly(style: str, x0: float, x1: float, y0: float, y1: float, seg: int = 12) -> Polygon:
    w = x1 - x0
    if style == "round":
        r = min(w, y1 - y0) / 2
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        a = np.linspace(0, 2 * math.pi, 2 * seg, endpoint=False)
        return Polygon(np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]))
    if style == "arched":
        r = w / 2
        ys = y1 - r
        if ys - y0 < 0.3:
            return box(x0, y0, x1, y1)
        a = np.linspace(0, math.pi, seg + 1)
        arc = np.column_stack([x0 + r + r * np.cos(a), ys + r * np.sin(a)])
        return Polygon([(x0, y0), (x1, y0)] + [tuple(p) for p in arc] + [(x0, ys)])
    return box(x0, y0, x1, y1)


class WallState:
    """Per wall: allowed solid intervals + used opening intervals."""

    def __init__(self, panel, others):
        self.p = panel
        self.ws = wallspec_of(panel)
        allowed = subtract_iv([(self.ws.x0, self.ws.x1)], blocked_intervals(self.ws, others))
        self.allowed = shrink_iv(allowed, MIN_SOLID)
        self.used: list[tuple[float, float]] = []
        b = panel.profile.bounds
        self.h = b[3] - b[1]

    def free(self, a: float, b: float) -> bool:
        if not any(lo - 1e-9 <= a and b <= hi + 1e-9 for lo, hi in self.allowed):
            return False
        return all(b + MIN_SOLID <= u0 + 1e-9 or a >= u1 + MIN_SOLID - 1e-9 for u0, u1 in self.used)

    def fit(self, a: float, b: float, min_w: float):
        """Shrink [a,b] to fit inside one allowed interval and away from used ones."""
        c = 0.5 * (a + b)
        segs = self.allowed
        for u0, u1 in self.used:
            segs = subtract_iv(segs, [(u0 - MIN_SOLID, u1 + MIN_SOLID)])
        for lo, hi in segs:
            if lo - 1e-9 <= c <= hi + 1e-9:
                a2, b2 = max(a, lo), min(b, hi)
                if b2 - a2 >= min_w:
                    return a2, b2
        return None


def _cut(ctx, st: WallState, poly: Polygon, glass: bool, sock_type: str, tags: dict):
    p = st.p
    if not cut_opening(ctx, p, poly):
        return False
    b = poly.bounds
    st.used.append((b[0], b[2]))
    st.used.sort()
    F = p.frame
    c = poly.centroid
    origin = F[:3, 3] + F[:3, 0] * c.x + F[:3, 1] * c.y + F[:3, 2] * (p.thickness / 2)
    if glass:
        G = F.copy()
        G[:3, 3] = F[:3, 3] + F[:3, 2] * round((p.thickness - GLASS_T) / 2, 4)
        G[:3, 3] = np.round(G[:3, 3] / 1e-4) * 1e-4
        add_panel(ctx, "glass", G, poly, GLASS_T, "glass", dict(tags, host=p.id, floor=p.tags.get("floor", -1)))
    ctx.sockets.append(Socket(sock_type, socket_xform(origin, F[:3, 0], F[:3, 1]), p.id))
    return True


def compute_connectivity(ctx: BuildingContext) -> dict:
    n = len(ctx.rooms)
    if n == 0:
        return {"ok": True, "unreachable": [], "n_rooms": 0}
    adj = {i: set() for i in range(-1, n)}
    for d in ctx.meta.get("doors", []):
        a, b = d["rooms"]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for a, b in ctx.meta.get("stair_links", []):
        adj[a].add(b)
        adj[b].add(a)
    seen = {-1}
    q = deque([-1])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                q.append(v)
    un = [r.id for r in ctx.rooms if r.id not in seen]
    return {"ok": not un, "unreachable": un, "n_rooms": n,
            "n_doors": len(ctx.meta.get("doors", [])), "n_stair_links": len(ctx.meta.get("stair_links", []))}


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    bo = ctx.blockout
    wp = dict(params.get("windows", {}) or {})
    gp = dict(params.get("ground", {}) or {})
    dp = dict(params.get("doors", {}) or {})
    ep = dict(params.get("entrance", {}) or {})
    door_w = float(dp.get("width", 0.9))
    door_h = float(dp.get("height", 2.1))
    per_room = int(dp.get("per_room", 1))
    doors = ctx.meta.setdefault("doors", [])
    windows = ctx.meta.setdefault("windows", [])

    # wall states (allowed intervals) per vertical wall panel
    walls = [p for p in ctx.panels if p.kind in ("ext_wall", "int_wall") and wallspec_of(p) is not None]
    solids = [(p, plan_footprint(p)) for p in ctx.panels if p.kind in ("ext_wall", "int_wall", "column")]
    solids = [(p, f, z_range(p)) for p, f in solids if f is not None]
    tree = shapely.STRtree([f for _, f, _ in solids])
    states = {}
    for p in walls:
        fp = plan_footprint(p)
        z0, z1 = z_range(p)
        others = []
        for j in tree.query(fp.buffer(0.01)):
            q, f, (a, b) = solids[int(j)]
            if q.id != p.id and a < z1 - 1e-6 and b > z0 + 1e-6:
                others.append(f)
        states[p.id] = WallState(p, others)

    # ---------------------------------------------------------------- doors (spanning tree)
    links = ctx.meta.get("room_links", [])
    by_floor = {}
    for L in links:
        by_floor.setdefault(L["floor"], []).append(L)
    core_rooms = set(ctx.meta.get("core_rooms", {}).values())

    def place_door(L, width, height, glass=False, tag="door"):
        st = states.get(L["wall"])
        if st is None:
            return False
        lo, hi = L["x0"] + width / 2, L["x1"] - width / 2
        if hi < lo - 1e-9:
            return False
        h = min(height, st.h - MIN_SOLID)
        for _ in range(8):
            c = float(rng.uniform(lo, hi)) if hi > lo else lo
            a, b = c - width / 2, c + width / 2
            if st.free(a, b):
                y0 = st.p.profile.bounds[1]
                if _cut(ctx, st, box(a, y0, b, y0 + h), glass, "door", {"opening": tag}):
                    doors.append(dict(floor=L["floor"], wall=L["wall"], rooms=[L["a"], L["b"]],
                                      x0=a, x1=b, kind=tag))
                    return True
        return False

    for k in sorted(by_floor):
        Ls = by_floor[k]
        rooms_k = sorted({L["a"] for L in Ls} | {L["b"] for L in Ls if L["b"] >= 0}
                         | {r.id for r in ctx.rooms if r.floor == k})
        adj = {}
        for L in Ls:
            if L["b"] == -1:
                continue
            adj.setdefault(L["a"], []).append(L)
            adj.setdefault(L["b"], []).append(L)
        seen = set(r for r in rooms_k if r in core_rooms) if k > 0 else set()
        if k == 0:
            # entrance: prefer corridor / largest room with an outside link
            ent = [L for L in Ls if L["b"] == -1]
            tag_of = {r.id: r.tag for r in ctx.rooms}
            area_of = {r.id: r.polygon.area for r in ctx.rooms}
            ent.sort(key=lambda L: (tag_of.get(L["a"]) != "corridor", -area_of.get(L["a"], 0),
                                    -(L["x1"] - L["x0"]), L["wall"]))
            ew = float(ep.get("width", 1.4))
            eh = float(ep.get("height", 2.4))
            for L in ent:
                if place_door(L, min(ew, L["x1"] - L["x0"]), eh, tag="entrance"):
                    seen.add(L["a"])
                    break
        if not seen:
            areas = {r.id: r.polygon.area for r in ctx.rooms if r.floor == k}
            if areas:
                seen = {max(areas, key=lambda i: (areas[i], -i))}
        q = deque(sorted(seen))
        while q:
            u = q.popleft()
            cand = sorted(adj.get(u, []), key=lambda L: (-(L["x1"] - L["x0"]), L["wall"]))
            nb = {}
            for L in cand:
                v = L["b"] if L["a"] == u else L["a"]
                if v in seen:
                    continue
                nb.setdefault(v, []).append(L)
            for v in sorted(nb, key=lambda v: (nb[v][0]["wall"], v)):
                if v in seen:
                    continue
                for L in nb[v]:
                    if place_door(L, door_w, door_h):
                        seen.add(v)
                        q.append(v)
                        break
        if per_room > 1:
            for L in Ls:
                if L["b"] >= 0 and rng.uniform() < (per_room - 1) * 0.3:
                    place_door(L, door_w, door_h)

    # ---------------------------------------------------------------- balcony doors
    ext_by_key = {(q.tags.get("floor"), q.tags.get("facade")): q.id for q in ctx.panels if q.kind == "ext_wall"}
    for bb in ctx.meta.get("balcony_bays", []):
        bb["wall"] = ext_by_key.get((bb["floor"], bb["facade"]), -1)
        st = states.get(bb["wall"])
        if st is None:
            continue
        w = min(float(bb.get("door_width", 1.0)), bb["x1"] - bb["x0"] - 2 * (MIN_SOLID + 0.2))
        c = 0.5 * (bb["x0"] + bb["x1"])
        if w >= 0.8 and st.free(c - w / 2, c + w / 2):
            y0 = st.p.profile.bounds[1]
            room = bb.get("room", -2)
            if _cut(ctx, st, box(c - w / 2, y0, c + w / 2, y0 + min(2.2, st.h - MIN_SOLID)), True, "door",
                    {"opening": "balcony_door"}):
                doors.append(dict(floor=bb["floor"], wall=st.p.id, rooms=[room, room], x0=c - w / 2,
                                  x1=c + w / 2, kind="balcony_door"))
                bb["door"] = True

    # ---------------------------------------------------------------- windows
    style_cache = {}
    for p in ctx.panels:
        if p.kind != "ext_wall" or p.id not in states:
            continue
        st = states[p.id]
        k = int(p.tags.get("floor", 0))
        ground = bool(k == 0 and gp)
        P = dict(wp)
        if ground:
            P.update(gp)
        if ground not in style_cache:
            style_cache[ground] = choice(P.get("style", "punched"), rng)
        style = style_cache[ground]
        y0 = p.profile.bounds[1]
        h = st.h
        if style == "storefront":
            sill, head = float(P.get("sill", 0.1)) if ground else 0.1, h - 0.5
        else:
            sill, head = float(rv(P.get("sill", 0.9), rng)), float(rv(P.get("head", 2.4), rng))
        head = min(head, h - MIN_SOLID - 0.05)
        if head - sill < 0.4:
            continue
        bays = p.tags.get("bays") or [st.ws.x0, st.ws.x1]
        for b0, b1 in zip(bays[:-1], bays[1:]):
            bw = b1 - b0
            if style in ("ribbon", "storefront"):
                a, b = b0 + 0.15, b1 - 0.15
            else:
                w = float(rv(P.get("width", [0.45 * bw, 0.6 * bw]), rng))
                w = min(w, bw - 0.6)
                c = 0.5 * (b0 + b1)
                a, b = c - w / 2, c + w / 2
            if style == "round":
                d = min(b - a, head - sill, 1.2)
                c = 0.5 * (a + b)
                a, b = c - d / 2, c + d / 2
            fit = st.fit(a, b, 0.5)
            if fit is None:
                continue
            a, b = fit
            if style == "round" and abs((b - a) - (head - sill)) > 1e-6:
                d = min(b - a, head - sill)
                c = 0.5 * (a + b)
                a, b = c - d / 2, c + d / 2
                ys, ye = (sill + head) / 2 - d / 2, (sill + head) / 2 + d / 2
            else:
                ys, ye = sill, head
            poly = shape_poly(style, a, b, y0 + ys, y0 + ye)
            if _cut(ctx, st, poly, True, "window_frame", {"opening": "window", "style": style}):
                windows.append(dict(floor=k, wall=p.id, x0=a, x1=b, style=style))

    ctx.meta["room_connectivity"] = compute_connectivity(ctx)
    return ctx
