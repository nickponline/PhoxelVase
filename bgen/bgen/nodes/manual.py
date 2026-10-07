"""`manual` node: persistent hand edits keyed by world position + kind (not by id).

params: {file: overrides.yaml (relative to the spec dir)} and/or {overrides: [...]}.
Each override:
  {op: delete_panel,   kind: int_wall, at: [x,y,z], tol: 0.3}
  {op: move_panel,     kind: column,   at: [x,y,z], offset: [dx,dy,dz]}
  {op: add_panel,      kind: column, material: concrete, box: [[x0,y0,z0],[x1,y1,z1]], floor: 0}
  {op: add_opening,    kind: ext_wall, at: [x,y,z], size: [w,h], shape: rect|arched|round, glass: true}
  {op: delete_opening, kind: ext_wall, at: [x,y,z]}      # point inside the opening
An edit whose target no longer exists at that location is skipped and reported in
ctx.meta['manual'] (Building Creator semantics).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import Point, Polygon, box

from ..model import BuildingContext
from ._util import (MIN_SOLID, add_panel, cut_opening, horiz_frame, remove_panels, snap, to_local)
from .openings import _cut, compute_connectivity, shape_poly, WallState


def find_panel(ctx: BuildingContext, kind: str | None, at, tol: float = 0.3):
    best, bd = None, tol
    p3 = np.asarray(at, float)
    for p in ctx.panels:
        if kind and p.kind != kind:
            continue
        loc = to_local(p, p3[None])[0]
        dz = max(0.0, -loc[2], loc[2] - p.thickness)
        if dz > bd:
            continue
        dxy = p.profile.exterior.distance(Point(loc[0], loc[1])) if not Polygon(p.profile.exterior).contains(
            Point(loc[0], loc[1])) else 0.0
        dist = max(dz, dxy)
        if dist <= bd:
            best, bd = p, dist
    return best


def load_overrides(ctx, params) -> list[dict]:
    ov = list(params.get("overrides", []) or [])
    f = params.get("file")
    if f:
        path = Path(f)
        if not path.is_absolute():
            path = Path(ctx.meta.get("spec_dir", ".")) / path
        if path.exists():
            data = yaml.safe_load(path.read_text()) or []
            ov += data.get("overrides", []) if isinstance(data, dict) else data
        else:
            ctx.meta.setdefault("manual", {}).setdefault("warnings", []).append(f"overrides file not found: {path}")
    return ov


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    ov = load_overrides(ctx, params)
    rep = ctx.meta.setdefault("manual", {})
    rep.setdefault("applied", 0)
    skipped = rep.setdefault("skipped", [])
    for o in ov:
        op = o.get("op")
        kind = o.get("kind")
        tol = float(o.get("tol", 0.3))
        ok = False
        if op == "add_panel":
            (x0, y0, z0), (x1, y1, z1) = o["box"]
            p = add_panel(ctx, kind or "other", horiz_frame(min(z0, z1)), box(x0, y0, x1, y1), abs(z1 - z0),
                          o.get("material", "concrete"), {"floor": int(o.get("floor", -1)), "manual": True})
            ok = p is not None
        else:
            p = find_panel(ctx, kind, o.get("at", [0, 0, 0]), tol)
            if p is not None and op == "delete_panel":
                glass = [q.id for q in ctx.panels if q.tags.get("host") == p.id and q.kind == "glass"]
                remove_panels(ctx, [p.id] + glass)
                ok = True
            elif p is not None and op == "move_panel":
                p.frame[:3, 3] = snap(p.frame[:3, 3] + np.asarray(o.get("offset", [0, 0, 0]), float))
                ok = True
            elif p is not None and op == "add_opening":
                loc = to_local(p, np.asarray(o["at"], float)[None])[0]
                w, h = o.get("size", [1.0, 1.2])
                poly = shape_poly(o.get("shape", "rect"), loc[0] - w / 2, loc[0] + w / 2, loc[1] - h / 2, loc[1] + h / 2)
                inner = p.profile.buffer(-MIN_SOLID, join_style=2)
                if inner.contains(poly):
                    st = WallState.__new__(WallState)
                    st.p, st.used, st.allowed = p, [], []
                    ok = _cut(ctx, st, poly, bool(o.get("glass", True)),
                              o.get("socket", "window_frame"), {"opening": "manual"})
            elif p is not None and op == "delete_opening":
                loc = to_local(p, np.asarray(o["at"], float)[None])[0]
                pt = Point(loc[0], loc[1])
                holes = [r for r in p.profile.interiors if Polygon(r).contains(pt)]
                if holes:
                    keep = [r for r in p.profile.interiors if not Polygon(r).contains(pt)]
                    p.profile = Polygon(p.profile.exterior, keep)
                    glass = [q.id for q in ctx.panels if q.kind == "glass" and q.tags.get("host") == p.id
                             and Polygon(holes[0]).buffer(1e-3).contains(q.profile)]
                    wp = np.asarray(o["at"], float)
                    ctx.sockets = [s for s in ctx.sockets if not (s.owner_panel == p.id and
                                                                  np.linalg.norm(s.xform[:3, 3] - wp) < 1.5)]
                    for key in ("doors", "windows"):
                        ctx.meta[key] = [d for d in ctx.meta.get(key, []) if not (
                            d["wall"] == p.id and d["x0"] - 1e-6 <= loc[0] <= d["x1"] + 1e-6)]
                    remove_panels(ctx, glass)
                    ok = True
        if ok:
            rep["applied"] += 1
        else:
            skipped.append(o)
    if ov:
        ctx.meta["room_connectivity"] = compute_connectivity(ctx)
    return ctx
