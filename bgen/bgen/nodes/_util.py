"""Shared helpers for feature nodes: frames, snapping, vertical wall specs, wall chains,
documented panel operations (cut_opening, remove_panels, split_panel).

Conventions (DESIGN §2.3):
  * floor_z[k] = top of slab k (floor level of floor k); slab k occupies [floor_z[k]-ts, floor_z[k]].
    floor_z has n_floors+1 entries; the last is the roof level.
  * Walls of floor k occupy z in [floor_z[k], floor_z[k+1]-ts].
  * A vertical panel ("wall spec") has frame x = wall direction d, y = world up,
    z = d x up = right-hand normal of d. Its base line (local z=0 face) starts at A.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient

from ..model import BuildingContext, Panel

Q = 1e-4          # snap grid (0.1 mm)
EPS = 1e-6
MIN_SOLID = 0.15  # minimum solid material between openings / to panel edges


# ----------------------------------------------------------------------------- snapping

def snap(x):
    return np.round(np.asarray(x, dtype=np.float64) / Q) * Q + 0.0


def snapf(x: float) -> float:
    return float(round(float(x) / Q) * Q) + 0.0


def snap_geom(g):
    g = shapely.set_precision(g, Q)
    return g


def clean_rot(R: np.ndarray) -> np.ndarray:
    R = np.array(R, dtype=np.float64)
    R[np.abs(R) < 1e-12] = 0.0
    R[np.abs(R - 1) < 1e-12] = 1.0
    R[np.abs(R + 1) < 1e-12] = -1.0
    return R


def parts(g) -> list[Polygon]:
    """Polygon parts of any geometry (drops lines/points/empties), sorted deterministically."""
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    out = []
    for h in getattr(g, "geoms", []):
        out += parts(h)
    out = [p for p in out if p.area > 1e-9]
    out.sort(key=lambda p: (round(p.bounds[0], 4), round(p.bounds[1], 4), -p.area))
    return out


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# ----------------------------------------------------------------------------- frames

def frame_from_axes(origin, x_axis, y_axis) -> np.ndarray:
    x = unit(x_axis)
    y = np.asarray(y_axis, float)
    y = unit(y - (y @ x) * x)
    z = np.cross(x, y)
    F = np.eye(4)
    F[:3, :3] = clean_rot(np.column_stack([x, y, z]))
    # Snap to the 0.1 mm grid only when the frame is axis-aligned; snapping a tilted frame
    # would move its faces off the exact shared planes of its neighbours.
    F[:3, 3] = snap(origin) if axis_aligned(F) else np.round(np.asarray(origin, float), 9)
    return F


def axis_aligned(F: np.ndarray) -> bool:
    R = F[:3, :3]
    return bool(np.all((np.abs(R) < 1e-12) | (np.abs(np.abs(R) - 1) < 1e-12)))


def wall_frame(A, d, z0) -> np.ndarray:
    d = unit([d[0], d[1], 0.0])
    return frame_from_axes([A[0], A[1], z0], d, [0, 0, 1.0])


def horiz_frame(z0: float) -> np.ndarray:
    F = np.eye(4)
    F[2, 3] = snapf(z0)
    return F


def add_panel(ctx: BuildingContext, kind: str, frame: np.ndarray, profile: Polygon,
              thickness: float, material: str, tags: dict | None = None) -> Panel | None:
    prof = orient(profile, 1.0)
    prof = snap_geom(prof) if axis_aligned(frame) else shapely.set_precision(prof, 1e-9)
    ps = parts(prof)
    if not ps:
        return None
    prof = orient(max(ps, key=lambda p: p.area), 1.0)
    if prof.area < 1e-6:
        return None
    return ctx.add_panel(kind=kind, frame=frame, profile=prof, thickness=float(round(thickness, 6)),
                         material=material, tags=dict(tags or {}))


def add_from_world(ctx: BuildingContext, kind: str, pts3d, normal, thickness: float, material: str,
                   tags: dict | None = None, x_axis=None, origin=None, holes=None) -> Panel | None:
    """Planar polygon given in building space (lying in the panel's base plane) extruded along
    `normal` by `thickness`."""
    P = np.asarray(pts3d, dtype=np.float64)
    n = unit(normal)
    o = P[0] if origin is None else np.asarray(origin, float)
    x = unit(P[1] - P[0]) if x_axis is None else unit(x_axis)
    x = unit(x - (x @ n) * n)
    y = np.cross(n, x)
    F = frame_from_axes(o, x, y)
    o = F[:3, 3]
    off = (P - o) @ n
    assert np.abs(off).max() < 2e-3, f"points not planar ({np.abs(off).max():.2e})"
    loc = np.column_stack([(P - o) @ F[:3, 0], (P - o) @ F[:3, 1]])
    hl = []
    for h in holes or []:
        H = np.asarray(h, float)
        hl.append(np.column_stack([(H - o) @ F[:3, 0], (H - o) @ F[:3, 1]]))
    return add_panel(ctx, kind, F, Polygon(loc, hl), thickness, material, tags)


def add_extruded(ctx: BuildingContext, kind: str, pts3d, direction, length: float, material: str,
                 tags: dict | None = None, seg_len: float = 1.2) -> list[Panel]:
    """Prism of a small cross-section extruded a long way along `direction`, emitted as
    consecutive segments (<= seg_len) so fracture (which only cuts in the profile plane)
    still yields reasonably shaped chunks. Segments meet on coplanar faces."""
    n = unit(direction)
    k = max(1, int(np.ceil(length / seg_len - 1e-9)))
    step = length / k
    P = np.asarray(pts3d, float)
    out = []
    for i in range(k):
        p = add_from_world(ctx, kind, P + i * step * n, n, step, material, dict(tags or {}, segment=i))
        if p is not None:
            out.append(p)
    return out


def to_local(panel_or_frame, pts3d) -> np.ndarray:
    F = panel_or_frame.frame if isinstance(panel_or_frame, Panel) else panel_or_frame
    P = np.asarray(pts3d, float) - F[:3, 3]
    return P @ F[:3, :3]


# ----------------------------------------------------------------------------- wall specs

@dataclass
class WallSpec:
    """A vertical box-ish panel: base line A + x*d for x in [x0, x1], thickness t to the right."""
    A: np.ndarray            # (2,)
    d: np.ndarray            # (2,) unit
    x0: float
    x1: float
    t: float
    z0: float
    z1: float
    kind: str = "int_wall"
    material: str = "brick"
    tags: dict = field(default_factory=dict)
    door_zone: tuple[float, float] | None = None   # allowed local-x range for doors (None = any)
    no_door: bool = False
    panel_id: int = -1

    @property
    def n(self) -> np.ndarray:   # right-hand normal (thickness direction)
        return np.array([self.d[1], -self.d[0]])

    def pt(self, x: float, side: float = 0.0) -> np.ndarray:
        return self.A + x * self.d + side * self.n

    def footprint(self) -> Polygon:
        return Polygon([self.pt(self.x0), self.pt(self.x1), self.pt(self.x1, self.t), self.pt(self.x0, self.t)])

    def face_line(self, side: int) -> LineString:
        s = 0.0 if side == 0 else self.t
        return LineString([self.pt(self.x0, s), self.pt(self.x1, s)])

    def to_panel(self, ctx: BuildingContext) -> Panel | None:
        F = wall_frame(self.A, self.d, self.z0)
        prof = box(self.x0, 0.0, self.x1, self.z1 - self.z0)
        tags = dict(self.tags)
        if self.door_zone is not None:
            tags["door_zone"] = [float(self.door_zone[0]), float(self.door_zone[1])]
        if self.no_door:
            tags["no_door"] = True
        p = add_panel(ctx, self.kind, F, prof, self.t, self.material, tags)
        self.panel_id = p.id if p is not None else -1
        return p


def is_vertical(p: Panel) -> bool:
    return abs(p.frame[2, 1] - 1.0) < 1e-9


def wallspec_of(p: Panel) -> WallSpec | None:
    """Recover a WallSpec (outer extent) from a vertical panel."""
    if not is_vertical(p):
        return None
    x = p.frame[:3, 0]
    d = np.array([x[0], x[1]])
    A = p.frame[:2, 3]
    bx0, by0, bx1, by1 = p.profile.bounds
    z0 = p.frame[2, 3]
    return WallSpec(A=A, d=d, x0=bx0, x1=bx1, t=p.thickness, z0=z0 + by0, z1=z0 + by1,
                    kind=p.kind, material=p.material, tags=p.tags, panel_id=p.id)


def plan_footprint(p: Panel) -> Polygon | None:
    """Plan (XY) footprint of a vertical panel or a horizontal (plan-profile) panel."""
    if is_vertical(p):
        return wallspec_of(p).footprint()
    if abs(p.frame[2, 2] - 1.0) < 1e-9:
        return shapely.affinity.translate(p.profile, p.frame[0, 3], p.frame[1, 3])
    return None


def z_range(p: Panel) -> tuple[float, float]:
    b = p.profile.bounds
    if is_vertical(p):
        return p.frame[2, 3] + b[1], p.frame[2, 3] + b[3]
    if abs(p.frame[2, 2] - 1.0) < 1e-9:
        return p.frame[2, 3], p.frame[2, 3] + p.thickness
    return -1e9, 1e9


# ----------------------------------------------------------------------------- intervals

def merge_iv(iv):
    iv = sorted((a, b) for a, b in iv if b - a > 1e-9)
    out = []
    for a, b in iv:
        if out and a <= out[-1][1] + 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def subtract_iv(base, cut):
    out = []
    cut = merge_iv(cut)
    for a, b in base:
        cur = a
        for c0, c1 in cut:
            if c1 <= cur or c0 >= b:
                continue
            if c0 > cur:
                out.append((cur, c0))
            cur = max(cur, c1)
        if cur < b:
            out.append((cur, b))
    return [(a, b) for a, b in out if b - a > 1e-9]


def intersect_iv(A, B):
    out = []
    for a0, a1 in A:
        for b0, b1 in B:
            lo, hi = max(a0, b0), min(a1, b1)
            if hi - lo > 1e-9:
                out.append((lo, hi))
    return merge_iv(out)


def shrink_iv(iv, m):
    return [(a + m, b - m) for a, b in iv if b - a > 2 * m + 1e-9]


def line_intervals(geom, ws: WallSpec, side: int) -> list[tuple[float, float]]:
    """Local-x intervals of wall face `side` that touch polygon `geom` (within 1 mm)."""
    fl = ws.face_line(side)
    g = geom.buffer(1e-3, join_style=2).intersection(fl)
    out = []
    for ln in _lines(g):
        cs = np.asarray(ln.coords)
        xs = (cs - ws.A) @ ws.d
        out.append((float(xs.min()), float(xs.max())))
    return merge_iv(out)


def _lines(g):
    if g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    if hasattr(g, "geoms"):
        r = []
        for h in g.geoms:
            r += _lines(h)
        return r
    return []


def blocked_intervals(ws: WallSpec, others: list[Polygon]) -> list[tuple[float, float]]:
    """Intervals of either face of `ws` touched by other vertical solids' footprints."""
    out = []
    for side in (0, 1):
        fl = ws.face_line(side).buffer(2e-3, cap_style=2)
        for f in others:
            if not f.intersects(fl):
                continue
            g = f.intersection(fl)
            if g.area < 1e-8:
                continue
            cs = np.asarray(g.exterior.coords) if isinstance(g, Polygon) else np.vstack(
                [np.asarray(q.exterior.coords) for q in parts(g)])
            xs = (cs - ws.A) @ ws.d
            out.append((float(xs.min()), float(xs.max())))
    return merge_iv(out)


# ----------------------------------------------------------------------------- wall chains

def simplify_ring(coords) -> np.ndarray:
    """Drop duplicate / collinear vertices from a closed ring (no closing point returned)."""
    P = [np.asarray(c, float) for c in coords]
    if np.allclose(P[0], P[-1]):
        P = P[:-1]
    changed = True
    while changed and len(P) > 3:
        changed = False
        for i in range(len(P)):
            a, b, c = P[i - 1], P[i], P[(i + 1) % len(P)]
            u, v = b - a, c - b
            if np.linalg.norm(u) < 1e-6 or abs(u[0] * v[1] - u[1] * v[0]) < 1e-7 * np.linalg.norm(u) * np.linalg.norm(v) and u @ v > 0:
                P.pop(i)
                changed = True
                break
    return np.array(P)


def chain_walls(pts, closed: bool, t: float, corner_rule: str = "x_runs_through"):
    """Build wall extents for a CCW chain (interior on the left of each segment).

    Returns (walls, posts): walls = list of dicts {i, P0, P1, d, L, A, x0, x1};
    posts = list of plan polygons filling non-orthogonal corners (or all corners for
    corner_rule='post'). Walls occupy the strip inward of the chain by `t`.
    """
    P = [np.asarray(p, float) for p in pts]
    nseg = len(P) if closed else len(P) - 1
    segs = []
    for i in range(nseg):
        a, b = P[i], P[(i + 1) % len(P)]
        L = float(np.linalg.norm(b - a))
        d = (b - a) / L
        n_in = np.array([-d[1], d[0]])
        segs.append(dict(i=i, P0=a, P1=b, d=d, L=L, A=a + t * n_in, n_in=n_in, es=0.0, ee=0.0))
    posts = []
    corners = range(nseg) if closed else range(1, nseg)
    for j in corners:
        s1, s2 = segs[(j - 1) % nseg], segs[j]
        d1, d2 = s1["d"], s2["d"]
        Pv = s2["P0"]
        cr = d1[0] * d2[1] - d1[1] * d2[0]
        dot = float(d1 @ d2)
        if abs(cr) < 1e-9 and dot > 0:
            continue                                  # collinear continuation
        ortho = abs(dot) < 1e-6 and corner_rule != "post"
        convex = cr > 0
        if ortho:
            x_is_1 = abs(d1[0]) >= abs(d2[0])
            if convex:
                if x_is_1:
                    s2["es"] = t
                else:
                    s1["ee"] = t
            else:
                if x_is_1:
                    s1["ee"] = -t
                else:
                    s2["es"] = -t
            continue
        phi = np.arctan2(cr, dot)                     # signed turn angle
        if convex:
            e = t * np.tan(phi / 2.0)
            s1["ee"] = e
            s2["es"] = e
            M = Pv - e * d1 + t * s1["n_in"]
            poly = Polygon([Pv - e * d1, Pv, Pv + e * d2, M])
        else:
            a1 = Pv + t * s1["n_in"]
            a2 = Pv + t * s2["n_in"]
            # intersection of inner lines a1 + l d1 and a2 + m d2
            Mx = np.array([d1, -d2]).T
            lm = np.linalg.solve(Mx, a2 - a1)
            M = a1 + lm[0] * d1
            poly = Polygon([Pv, a1, M, a2])
        poly = orient(poly.buffer(0), 1.0)
        if poly.area > 1e-6:
            posts.append(poly)
    walls = []
    for s in segs:
        x0, x1 = s["es"], s["L"] - s["ee"]
        if x1 - x0 > 0.05:
            walls.append(dict(i=s["i"], P0=s["P0"], P1=s["P1"], d=s["d"], L=s["L"], A=s["A"],
                              x0=float(x0), x1=float(x1), n_in=s["n_in"]))
    return walls, posts


def emit_chain(ctx, pts, closed, t, z0, z1, kind, material, tags, corner_rule="x_runs_through",
               extra=None) -> list[Panel]:
    walls, posts = chain_walls(pts, closed, t, corner_rule)
    out = []
    for w in walls:
        tg = dict(tags)
        tg["seg"] = [list(map(float, w["P0"])), list(map(float, w["P1"]))]
        tg["normal"] = [float(-w["n_in"][0]), float(-w["n_in"][1])]
        if extra:
            tg.update(extra(w))
        ws = WallSpec(A=snap(w["A"]), d=w["d"], x0=snapf(w["x0"]), x1=snapf(w["x1"]), t=t, z0=z0, z1=z1,
                      kind=kind, material=material, tags=tg)
        p = ws.to_panel(ctx)
        if p is not None:
            out.append(p)
    for poly in posts:
        tg = dict(tags)
        tg["role"] = "corner_post"
        p = add_panel(ctx, "column", horiz_frame(z0), poly, z1 - z0, material, tg)
        if p is not None:
            out.append(p)
    return out


# ----------------------------------------------------------------------------- panel ops

def cut_opening(ctx: BuildingContext, panel: Panel, poly_local: Polygon) -> bool:
    """Subtract a local-XY polygon from a panel profile (documented op). Returns success."""
    poly_local = snap_geom(poly_local)
    new = snap_geom(panel.profile.difference(poly_local))
    ps = parts(new)
    if len(ps) != 1:
        return False
    panel.profile = orient(ps[0], 1.0)
    return True


def remove_panels(ctx: BuildingContext, ids) -> dict[int, int]:
    """Remove panels and re-index densely. Returns old->new id map (removed ids absent)."""
    ids = set(int(i) for i in ids)
    remap, keep = {}, []
    for p in ctx.panels:
        if p.id in ids:
            continue
        remap[p.id] = len(keep)
        p.id = len(keep)
        keep.append(p)
    ctx.panels = keep
    ctx.sockets = [s for s in ctx.sockets if s.owner_panel < 0 or s.owner_panel in remap]
    for s in ctx.sockets:
        if s.owner_panel >= 0:
            s.owner_panel = remap[s.owner_panel]
    for p in ctx.panels:
        if "host" in p.tags:
            p.tags["host"] = remap.get(p.tags["host"], -1)
    for key in ("doors", "room_links", "windows"):
        lst = ctx.meta.get(key)
        if lst:
            ctx.meta[key] = [dict(r, wall=remap[r["wall"]]) for r in lst if r.get("wall") in remap]
    return remap


def split_panel(ctx: BuildingContext, panel: Panel, cutter_local: Polygon) -> list[Panel]:
    """Split a panel by removing `cutter_local` from its profile; each remaining part becomes
    its own panel (the first part keeps the original id). Returns the resulting panels."""
    ps = parts(snap_geom(panel.profile.difference(snap_geom(cutter_local))))
    if not ps:
        remove_panels(ctx, [panel.id])
        return []
    panel.profile = orient(ps[0], 1.0)
    out = [panel]
    for q in ps[1:]:
        p = add_panel(ctx, panel.kind, panel.frame.copy(), q, panel.thickness, panel.material, dict(panel.tags))
        if p is not None:
            out.append(p)
    return out


def floor_params(ctx: BuildingContext):
    bo = ctx.blockout
    return bo.floor_polys, bo.floor_z, float(ctx.params["slab_thickness"])


def socket_xform(origin, x_axis, y_axis) -> np.ndarray:
    return frame_from_axes(origin, x_axis, y_axis)
