"""`sculpt` node: free-form structures (bridges, towers, statues) from 3D CSG primitives.

The solid is sliced into horizontal layers of height `layer`; each layer's cross-section is a
plan polygon per (material, kind) class, emitted as one horizontal panel per connected part.
Consecutive layers meet on coplanar faces (z = k*layer), and classes inside one layer share
exact boolean boundaries, so the graph gets both vertical and in-layer contacts. Sloped and
curved surfaces come out terraced at `layer` resolution (like a contour model).

params:
  layer: 0.5                 # slice height (m); primitive z extents snap to it
  template: eiffel           # optional parametric generator (see sculpt_templates.py),
  <template params...>       #   its params may be [lo, hi] ranges sampled from the node rng
  primitives: [...]          # extra primitives, appended after the template's
  ground: {pad: 3.0, thickness: 0.3, material: concrete}   # anchor slab under the footprint
  simplify: 0.02             # plan tolerance for curved primitives (m)
  min_part: 0.01             # drop slice parts smaller than this (m^2)

Primitive (dict with exactly one shape key, plus material / kind / op / mirror):
  {box: {min: [x,y,z], max: [x,y,z], yaw: deg}}
  {lathe: {center: [x,y], profile: [[z, r], ...] | [[z, r, cx, cy], ...], sides: 24, rot: deg}}
        r is the inradius (half-width for sides: 4); optional cx, cy per row lean the axis
  {ellipsoid: {center: [x,y,z], radii: [a,b,c], yaw: deg, sides: 24}}
  {tube: {path: [[x,y,z], ...], radius: r | [r per point], section: round|square}}
        swept sphere (round) or axis-aligned cube (square) along a polyline
  {extrude: {plane: xz|yz, span: [lo, hi], poly: [[u,z], ...], holes: [[[u,z], ...], ...]}}
        polygon drawn in elevation, extruded across `span` (y for xz, x for yz)
  material: concrete|brick|wood|metal (default concrete); kind: element kind (default other)
  op: add|sub (sub removes from every class); later primitives win over earlier ones
  mirror: [x] | [y] | [x, y]  with about: [cx, cy]  -> also add the mirrored copies
"""
from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon, box
from shapely import affinity

from ..model import BuildingContext
from ..rng import rv
from ._util import add_panel, horiz_frame, parts

SHAPES = ("box", "lathe", "ellipsoid", "tube", "extrude")


# ----------------------------------------------------------------------------- primitives

def _shape_of(p: dict) -> str:
    ks = [k for k in SHAPES if k in p]
    if len(ks) != 1:
        raise ValueError(f"sculpt primitive needs exactly one of {SHAPES}: {p}")
    return ks[0]


def expand_mirrors(prims: list[dict]) -> list[dict]:
    out = []
    for p in prims:
        out.append(p)
        axes = p.get("mirror") or []
        if not axes:
            continue
        cx, cy = p.get("about", [0.0, 0.0])
        flips = [(sx, sy) for sx in ((1, -1) if "x" in axes else (1,)) for sy in ((1, -1) if "y" in axes else (1,))]
        for sx, sy in flips[1:]:
            out.append(_mirror(p, sx, sy, cx, cy))
    return out


def _mirror(p: dict, sx: int, sy: int, cx: float, cy: float) -> dict:
    fx = lambda x: cx + sx * (x - cx)
    fy = lambda y: cy + sy * (y - cy)
    q = {k: v for k, v in p.items() if k not in ("mirror", "about")}
    s = _shape_of(p)
    g = dict(p[s])
    if s == "box":
        (x0, y0, z0), (x1, y1, z1) = g["min"], g["max"]
        xs, ys = sorted([fx(x0), fx(x1)]), sorted([fy(y0), fy(y1)])
        g["min"], g["max"] = [xs[0], ys[0], z0], [xs[1], ys[1], z1]
        if g.get("yaw"):
            g["yaw"] = g["yaw"] * sx * sy
    elif s == "lathe":
        c = g.get("center", [0.0, 0.0])
        g["center"] = [fx(c[0]), fy(c[1])]
        g["profile"] = [[r[0], r[1], sx * r[2], sy * r[3]] if len(r) > 2 else list(r) for r in g["profile"]]
    elif s == "ellipsoid":
        c = g["center"]
        g["center"] = [fx(c[0]), fy(c[1]), c[2]]
        if g.get("yaw"):
            g["yaw"] = g["yaw"] * sx * sy
    elif s == "tube":
        g["path"] = [[fx(x), fy(y), z] for x, y, z in g["path"]]
    elif s == "extrude":
        along_x = g.get("plane", "xz") == "xz"
        fu, fs = (fx, fy) if along_x else (fy, fx)
        flip_u = (sx if along_x else sy) < 0
        if "geom" in g:
            c = cx if along_x else cy
            g["geom"] = affinity.scale(g["geom"], -1.0 if flip_u else 1.0, 1.0, origin=(c, 0.0))
        else:
            g["poly"] = [[fu(u), z] for u, z in g["poly"]]
            g["holes"] = [[[fu(u), z] for u, z in h] for h in g.get("holes", [])]
        lo, hi = g["span"]
        g["span"] = sorted([fs(lo), fs(hi)])
        if flip_u and "poly" in g:
            g["poly"] = g["poly"][::-1]
    q[s] = g
    return q


class Prim:
    """A primitive prepared for slicing: z extent + section(zm) -> shapely geometry or None."""

    def __init__(self, p: dict, tol: float, h: float):
        self.shape = _shape_of(p)
        self.g = p[self.shape]
        self.material = p.get("material", "concrete")
        self.kind = p.get("kind", "other")
        self.op = p.get("op", "add")
        self.tol = tol
        getattr(self, f"_prep_{self.shape}")(h)

    # -- box
    def _prep_box(self, h):
        (x0, y0, z0), (x1, y1, z1) = self.g["min"], self.g["max"]
        self.z0, self.z1 = min(z0, z1), max(z0, z1)
        r = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        if self.g.get("yaw"):
            r = affinity.rotate(r, float(self.g["yaw"]), origin="centroid")
        self._poly = r

    def _sec_box(self, zm):
        return self._poly

    # -- lathe
    def _prep_lathe(self, h):
        prof = np.asarray(sorted(self.g["profile"], key=lambda r: r[0]), float)
        self._prof = prof
        self.z0, self.z1 = float(prof[0, 0]), float(prof[-1, 0])
        n = int(self.g.get("sides", 24))
        rot = math.radians(float(self.g.get("rot", 45.0 if n == 4 else 0.0)))
        a = rot + 2 * math.pi * np.arange(n) / n
        self._unit = np.column_stack([np.cos(a), np.sin(a)]) / math.cos(math.pi / n)
        self._c = np.asarray(self.g.get("center", [0.0, 0.0]), float)

    def _sec_lathe(self, zm):
        P = self._prof
        r = float(np.interp(zm, P[:, 0], P[:, 1]))
        if r <= 1e-3:
            return None
        c = self._c.copy()
        if P.shape[1] >= 4:
            c += [np.interp(zm, P[:, 0], P[:, 2]), np.interp(zm, P[:, 0], P[:, 3])]
        return Polygon(c + r * self._unit)

    # -- ellipsoid
    def _prep_ellipsoid(self, h):
        self._c = np.asarray(self.g["center"], float)
        self._r = np.asarray(self.g["radii"], float)
        self.z0, self.z1 = self._c[2] - self._r[2], self._c[2] + self._r[2]
        n = int(self.g.get("sides", 24))
        a = 2 * math.pi * np.arange(n) / n
        self._unit = np.column_stack([np.cos(a), np.sin(a)])
        yaw = math.radians(float(self.g.get("yaw", 0.0)))
        self._R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

    def _sec_ellipsoid(self, zm):
        t = (zm - self._c[2]) / self._r[2]
        if abs(t) >= 1.0:
            return None
        s = math.sqrt(1 - t * t)
        pts = (self._unit * self._r[:2] * s) @ self._R.T + self._c[:2]
        return Polygon(pts)

    # -- tube: swept sphere / cube, sampled densely along the path
    def _prep_tube(self, h):
        P = np.asarray(self.g["path"], float)
        R = self.g.get("radius", 0.5)
        R = np.full(len(P), float(R)) if np.isscalar(R) else np.asarray(R, float)
        self._square = self.g.get("section", "round") == "square"
        xs, rs, seg = [], [], []
        for i in range(len(P) - 1):
            L = float(np.linalg.norm(P[i + 1] - P[i]))
            step = max(0.04, min(R[i], R[i + 1], h) * 0.35)
            k = max(1, int(math.ceil(L / step)))
            t = (np.arange(0, k + 1) / k)[:, None]
            xs.append(P[i] + t * (P[i + 1] - P[i]))
            rs.append(R[i] + t[:, 0] * (R[i + 1] - R[i]))
            seg.append(np.full(k + 1, i))
        if len(P) == 1:
            xs, rs, seg = [P], [R], [np.zeros(1, int)]
        self._S = np.vstack(xs)
        self._R = np.concatenate(rs)
        self._seg = np.concatenate(seg)
        self.z0 = float((self._S[:, 2] - self._R).min())
        self.z1 = float((self._S[:, 2] + self._R).max())

    _CIRC = np.column_stack([np.cos(np.arange(16) * math.pi / 8), np.sin(np.arange(16) * math.pi / 8)])
    _SQ = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    def _sec_tube(self, zm):
        """Each straight segment sweeps a convex solid, so its slice is the convex hull of the
        sampled sphere / cube slices; the tube's slice is the union over segments."""
        dz = np.abs(self._S[:, 2] - zm)
        m = dz < self._R
        if self._square:
            rr, unit = self._R, self._SQ
        else:
            rr, unit = np.sqrt(np.maximum(self._R ** 2 - dz ** 2, 0.0)), self._CIRC
            m &= rr > 0.02
        if not m.any():
            return None
        hulls = []
        for sid in np.unique(self._seg[m]):
            k = m & (self._seg == sid)
            pts = (self._S[k, None, :2] + rr[k, None, None] * unit[None]).reshape(-1, 2)
            hulls.append(shapely.MultiPoint(pts).convex_hull)
        return shapely.union_all(hulls)

    # -- extrude: elevation polygon swept across `span`
    def _prep_extrude(self, h):
        geom = self.g.get("geom")    # templates may pass a shapely (Multi)Polygon directly
        self._poly = geom if geom is not None else Polygon(self.g["poly"], self.g.get("holes", []) or [])
        if not self._poly.is_valid:
            self._poly = self._poly.buffer(0)
        _, self.z0, _, self.z1 = self._poly.bounds
        self._xz = self.g.get("plane", "xz") == "xz"
        self._span = sorted(map(float, self.g["span"]))

    def _sec_extrude(self, zm):
        u0, _, u1, _ = self._poly.bounds
        seg = self._poly.intersection(LineString([(u0 - 1, zm), (u1 + 1, zm)]))
        lo, hi = self._span
        rects = []
        for s in getattr(seg, "geoms", [seg]):
            if s.is_empty or s.length < 1e-4:
                continue
            us = [c[0] for c in s.coords]
            a, b = min(us), max(us)
            rects.append(box(a, lo, b, hi) if self._xz else box(lo, a, hi, b))
        return shapely.union_all(rects) if rects else None

    def section(self, zm):
        if not (self.z0 <= zm <= self.z1):
            return None
        g = getattr(self, f"_sec_{self.shape}")(zm)
        if g is None or g.is_empty:
            return None
        if self.shape in ("tube", "ellipsoid", "lathe") and self.tol > 0:
            g = g.simplify(self.tol)
        return g if g.is_valid else g.buffer(0)


# ----------------------------------------------------------------------------- slicing

def slice_layers(prims: list[Prim], h: float, z_base: float = 0.0):
    """-> list of (k, z0, {(material, kind): geometry}) for every non-empty layer."""
    adds = [p for p in prims if p.op == "add"]
    if not adds:
        return []
    ztop = max(p.z1 for p in adds)
    nk = int(math.ceil((ztop - z_base) / h - 1e-6))
    out = []
    for k in range(nk):
        z0 = z_base + k * h
        zm = z0 + 0.5 * h
        classes: dict[tuple[str, str], object] = {}
        for p in prims:
            g = p.section(zm)
            if g is None:
                continue
            if p.op == "sub":
                for key in list(classes):
                    classes[key] = classes[key].difference(g)
                continue
            key = (p.material, p.kind)
            for other in list(classes):
                if other != key:
                    classes[other] = classes[other].difference(g)
            classes[key] = classes[key].union(g) if key in classes else g
        classes = {k2: g for k2, g in classes.items() if not g.is_empty}
        if classes:
            out.append((k, round(z0, 6), classes))
    return out


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    from .sculpt_templates import TEMPLATES
    h = float(params.get("layer", 0.5))
    tol = float(params.get("simplify", 0.02))
    min_part = float(params.get("min_part", 0.01))
    prims_raw: list[dict] = []
    meta = ctx.meta.setdefault("sculpt", {})
    tname = params.get("template")
    if tname:
        if tname not in TEMPLATES:
            raise KeyError(f"unknown sculpt template {tname!r}; have {sorted(TEMPLATES)}")
        reserved = {"layer", "simplify", "min_part", "template", "primitives", "ground"}
        tp = {k: rv(v, rng) for k, v in params.items() if k not in reserved}
        prims_raw += TEMPLATES[tname](dict(tp, layer=h), rng)
        meta["template"] = tname
        meta["template_params"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in tp.items()}
    prims_raw += list(params.get("primitives", []) or [])
    prims = [Prim(p, tol, h) for p in expand_mirrors(prims_raw)]
    layers = slice_layers(prims, h)
    n_before = len(ctx.panels)
    lo = np.array([np.inf, np.inf])
    hi = -lo
    for k, z0, classes in layers:
        for (mat, kind), g in sorted(classes.items()):
            for pi, part in enumerate(parts(g)):
                if part.area < min_part:
                    continue
                if add_panel(ctx, kind, horiz_frame(z0), part, h, mat,
                             {"floor": 0, "role": "sculpt", "layer": k, "part": pi}) is None:
                    continue
                b = part.bounds
                lo, hi = np.minimum(lo, b[:2]), np.maximum(hi, b[2:])
    gp = params.get("ground", {"pad": 3.0})
    if gp is not None and np.isfinite(lo).all():
        pad = float(gp.get("pad", 3.0))
        t = float(gp.get("thickness", ctx.params.get("slab_thickness", 0.3)))
        x0, y0 = np.floor(lo - pad)
        x1, y1 = np.ceil(hi + pad)
        add_panel(ctx, "floor", horiz_frame(-t), box(x0, y0, x1, y1), t, gp.get("material", "concrete"),
                  {"floor": 0, "role": "ground_slab", "anchor": True})
    meta.update(layers=len(layers), layer=h, primitives=len(prims), panels=len(ctx.panels) - n_before,
                height=round(len(layers) and (layers[-1][1] + h), 3),
                footprint=[float(v) for v in (*lo, *hi)] if np.isfinite(lo).all() else None)
    return ctx
