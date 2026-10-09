"""Blockout: masses -> per-floor footprints, floor levels, facade segments."""
from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from .model import Blockout
from .nodes._util import parts, simplify_ring, snap, snap_geom
from .rng import choice, rv


def _r(x, q=0.1):
    return float(round(x / q) * q)


def generate_masses(b: dict, rng: np.random.Generator) -> list[dict]:
    """Procedural blockout from preset ranges. All shapes axis-aligned, coords on a 0.1 m grid."""
    shape = choice(b.get("shape", "rect"), rng)
    W = _r(rv(b.get("width", [20.0, 30.0]), rng))
    D = _r(rv(b.get("depth", [14.0, 18.0]), rng))
    if D > W:
        W, D = D, W
    F = int(rv(b.get("floors", [3, 5]), rng))
    roof = choice(b.get("roof", "flat"), rng)
    rect = lambda x0, y0, x1, y1: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    masses = [dict(footprint=rect(0, 0, W, D), floors=F, roof=roof)]
    if shape == "L":
        w2 = _r(W * rv([0.35, 0.5], rng))
        d2 = _r(D * rv([0.6, 1.0], rng))
        masses.append(dict(footprint=rect(0, D, w2, D + d2), floors=F, roof=roof))
    elif shape == "T":
        w2 = _r(W * rv([0.3, 0.45], rng))
        d2 = _r(D * rv([0.5, 0.8], rng))
        x0 = _r((W - w2) / 2)
        masses.append(dict(footprint=rect(x0, D, x0 + w2, D + d2), floors=F, roof=roof))
    elif shape == "annex":
        wa = _r(W * rv([0.25, 0.4], rng))
        da = _r(D * rv([0.5, 0.8], rng))
        fa = max(1, int(round(F * rv([0.3, 0.6], rng))))
        masses.append(dict(footprint=rect(W, 0, W + wa, da), floors=fa, roof=b.get("annex_roof", "flat")))
    elif shape == "setback":
        fs = max(1, int(round(F * rv([0.25, 0.4], rng))))
        inset = _r(rv([3.0, 5.0], rng))
        masses[0]["roof"] = "flat"
        masses[0]["floors"] = F - fs
        masses.append(dict(footprint=rect(0, 0, W - inset, D), floors=F, roof=roof))
    elif shape == "podium":
        tw = _r(rv(b.get("tower_width", [18.0, 24.0]), rng))
        td = _r(rv(b.get("tower_depth", [16.0, 22.0]), rng))
        tw, td = min(tw, W - 4), min(td, D)
        pf = int(rv(b.get("podium_floors", [1, 2]), rng))
        x0 = _r((W - tw) / 2)
        masses[0]["floors"] = pf
        masses[0]["roof"] = "flat"
        masses.append(dict(footprint=rect(x0, 0, x0 + tw, td), floors=F, roof=roof))
    return masses


def build_blockout(masses: list[dict], params: dict) -> Blockout:
    H = float(params["floor_height"])
    gh = float(params.get("ground_floor_height", H))
    nfl = max(int(m["floors"]) for m in masses)
    polys = [Polygon(m["footprint"]) for m in masses]
    floor_polys, roof_types = [], []
    for k in range(nfl):
        u = unary_union([p for p, m in zip(polys, masses) if int(m["floors"]) > k])
        ps = []
        for p in parts(snap_geom(u.buffer(0))):
            ring = simplify_ring(snap(np.asarray(p.exterior.coords)))
            ps.append(orient(Polygon(ring), 1.0))          # holes filled
        g = ps[0] if len(ps) == 1 else unary_union(ps)
        floor_polys.append(g)
        tops = [m.get("roof", "flat") for m in masses if int(m["floors"]) == k + 1]
        roof_types.append(tops[0] if tops else "flat")
    floor_z = [0.0]
    for k in range(nfl):
        floor_z.append(round(floor_z[-1] + (gh if k == 0 else H), 4))
    facades = []
    for g in floor_polys:
        segs = []
        for p in parts(g):
            c = np.asarray(p.exterior.coords)[:-1]
            for i in range(len(c)):
                segs.append((c[i].copy(), c[(i + 1) % len(c)].copy()))
        facades.append(segs)
    bo = Blockout(floor_polys=floor_polys, floor_z=floor_z, roof_types=roof_types, facades=facades)
    bo.masses = masses  # type: ignore[attr-defined]
    return bo


def roof_type_at(bo: Blockout, level: int, pt) -> str:
    """Roof type of the mass whose top floor is level-1 and contains pt."""
    from shapely.geometry import Point
    for m in getattr(bo, "masses", []):
        if int(m["floors"]) == level and Polygon(m["footprint"]).buffer(1e-3).contains(Point(pt)):
            return m.get("roof", "flat")
    return bo.roof_types[level - 1] if 0 < level <= len(bo.roof_types) else "flat"
