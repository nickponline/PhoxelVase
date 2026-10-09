"""Parametric generators for the `sculpt` node: params (ranges already sampled) -> primitives.

    aqueduct           three-tier stone arch viaduct (Pont du Gard-like)
    colossus           robed figure with a raised torch on a stepped pedestal

All coordinates are metres, z up, ground at z = 0. Tube radii stay >= layer/2.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def _snap(z: float, h: float) -> float:
    return round(z / h) * h


def _tube(path, r, mat="metal", kind="beam", section="square") -> dict:
    return {"tube": {"path": [list(map(float, p)) for p in path], "radius": r, "section": section},
            "material": mat, "kind": kind}


def _box(x0, y0, z0, x1, y1, z1, mat="concrete", kind="other", **kw) -> dict:
    return {"box": {"min": [x0, y0, z0], "max": [x1, y1, z1], **kw}, "material": mat, "kind": kind}


def _arch_hole(x0: float, x1: float, z_spring: float, z_bottom: float, n: int = 20) -> Polygon:
    """Opening of width x1-x0: vertical jambs from z_bottom to z_spring, semicircular head."""
    r = 0.5 * (x1 - x0)
    c = 0.5 * (x0 + x1)
    a = np.linspace(0.0, math.pi, n)
    head = np.column_stack([c + r * np.cos(a), z_spring + r * np.sin(a)])
    return Polygon([(x1, z_bottom), *head.tolist(), (x0, z_bottom)])


# ----------------------------------------------------------------------------- aqueduct

def aqueduct(p: dict, rng) -> list[dict]:
    h = float(p.get("layer", 0.5))
    n = int(p.get("arches", 5))
    s1 = float(p.get("bay", 18.0))
    pier = float(p.get("pier", 4.0))
    h1 = _snap(float(p.get("tier1", 18.0)), h)
    h2 = _snap(float(p.get("tier2", 14.0)), h)
    h3 = _snap(float(p.get("tier3", 6.0)), h)
    d1, d2, d3 = float(p.get("depth1", 6.5)), float(p.get("depth2", 5.0)), float(p.get("depth3", 3.6))
    mat = p.get("material", "brick")
    L = n * s1 + pier
    out: list[dict] = []

    def tier(z0, z1, bay, pw, k, depth, crown=1.5, wide_center=0.0):
        rect = box(0.0, z0, L, z1)
        holes = []
        nb = int(round(L / bay)) if k is None else k
        bay = (L - pw) / nb
        for i in range(nb):
            x0 = pw + i * bay
            x1 = x0 + bay - pw
            r = 0.5 * (x1 - x0)
            zs = z1 - crown - r
            holes.append(_arch_hole(x0, x1, max(zs, z0 + 0.5), z0 - 1.0))
        g = rect.difference(unary_union(holes))
        out.append({"extrude": {"plane": "xz", "span": [-depth / 2, depth / 2], "geom": g},
                    "material": mat, "kind": "ext_wall"})

    tier(0.0, h1, s1, pier, n, d1)
    tier(h1, h1 + h2, s1, pier * 0.8, n, d2)
    tier(h1 + h2, h1 + h2 + h3, s1 / 3, 1.2, 3 * n, d3, crown=0.8)
    zc = h1 + h2 + h3
    # conduit on top: two side walls + cover slabs
    for s in (-1, 1):
        y0, y1 = sorted([s * d3 / 2, s * (d3 / 2 - 0.8)])
        out.append(_box(0.0, y0, zc, L, y1, zc + 1.5, mat=mat, kind="ext_wall"))
    out.append(_box(0.0, -d3 / 2, zc + 1.5, L, d3 / 2, zc + 2.0, mat=mat, kind="roof"))
    # cutwaters on the tier-1 piers, both faces
    bay = (L - pier) / n
    for i in range(n + 1):
        xc = pier / 2 + i * bay
        for s in (-1, 1):
            out.append({"box": {"min": [xc - pier / 2.9, s * d1 / 2 - pier / 2.9, 0.0],
                                "max": [xc + pier / 2.9, s * d1 / 2 + pier / 2.9, _snap(0.35 * h1, h)], "yaw": 45},
                        "material": mat, "kind": "column"})
    # abutments at both ends
    for x0, x1 in ((-6.0, 0.0), (L, L + 6.0)):
        out.append(_box(x0, -d1 / 2 - 1.0, 0.0, x1, d1 / 2 + 1.0, h1 * 0.6, mat="concrete", kind="column"))
    return out


# ----------------------------------------------------------------------------- colossus

def colossus(p: dict, rng) -> list[dict]:
    F = float(p.get("figure", 28.0))
    h = float(p.get("layer", 0.4))
    raise_deg = float(p.get("arm_raise", 10.0))      # lean of the torch arm outward from vertical
    mat = p.get("material", "metal")
    base = p.get("pedestal_material", "concrete")
    out: list[dict] = []
    # stepped pedestal
    P = float(p.get("pedestal", 15.0))
    hw = 0.2 * F + 1.0
    steps = [(0.0, 3.0, hw + 3.5), (3.0, P - 2.0, hw + 0.5), (P - 2.0, P - 1.0, hw + 1.2), (P - 1.0, P, hw)]
    for z0, z1, r in steps:
        out.append(_box(-r, -r, _snap(z0, h), r, r, _snap(z1, h), mat=base, kind="column"))
    z0 = _snap(P, h)
    f = lambda v: v * F
    Z = lambda v: z0 + v * F
    # robe / body (slight forward lean of the upper body)
    out.append({"lathe": {"center": [0, 0], "sides": 28, "profile": [
        [Z(0.0), f(0.17), 0, 0], [Z(0.08), f(0.145), 0, 0], [Z(0.3), f(0.12), 0, -f(0.005)],
        [Z(0.45), f(0.10), 0, -f(0.01)], [Z(0.56), f(0.11), 0, -f(0.01)], [Z(0.63), f(0.09), 0, -f(0.005)],
        [Z(0.66), f(0.05), 0, 0]]}, "material": mat, "kind": "column"})
    out.append({"ellipsoid": {"center": [0, 0, Z(0.62)], "radii": [f(0.16), f(0.085), f(0.05)]},
                "material": mat, "kind": "column"})
    out.append({"lathe": {"center": [0, 0], "profile": [[Z(0.65), f(0.042)], [Z(0.71), f(0.04)]]},
                "material": mat, "kind": "column"})
    out.append({"ellipsoid": {"center": [0, -f(0.005), Z(0.755)], "radii": [f(0.055), f(0.062), f(0.075)]},
                "material": mat, "kind": "other"})
    # crown band + spikes
    out.append({"lathe": {"center": [0, -f(0.005)], "profile": [[Z(0.775), f(0.066)], [Z(0.80), f(0.064)]]},
                "material": mat, "kind": "other"})
    rmin = 0.5 * h + 0.06
    for ang in np.linspace(-160, -20, 7):
        a = math.radians(ang)
        d = np.array([math.cos(a), math.sin(a), 0.9])
        d /= np.linalg.norm(d)
        p0 = np.array([0.0, -f(0.005), Z(0.79)]) + d * f(0.05)
        p1 = p0 + d * f(0.1)
        out.append(_tube([p0, p1], [max(f(0.014), rmin), rmin], mat=mat, kind="other", section="round"))
    # torch arm: shoulder -> elbow -> hand, then torch
    ra = math.radians(raise_deg)
    sh = np.array([f(0.13), 0.0, Z(0.63)])
    el = sh + f(0.17) * np.array([math.sin(ra), -0.05, math.cos(ra)])
    hd = el + f(0.17) * np.array([math.sin(ra * 0.3), -0.1, 1.0])
    out.append(_tube([sh, el, hd], [f(0.05), f(0.042), f(0.036)], mat=mat, kind="beam", section="round"))
    tz = hd[2]
    out.append(_tube([hd, hd + [0, 0, f(0.07)]], f(0.03), mat=mat, kind="beam", section="round"))
    out.append({"lathe": {"center": [float(hd[0]), float(hd[1])], "profile": [
        [tz + f(0.06), f(0.03)], [tz + f(0.1), f(0.062)], [tz + f(0.11), f(0.062)]]}, "material": mat, "kind": "other"})
    out.append({"ellipsoid": {"center": [float(hd[0]), float(hd[1]), tz + f(0.15)],
                              "radii": [f(0.035), f(0.035), f(0.055)]}, "material": "metal", "kind": "other"})
    # tablet arm: shoulder -> elbow -> hand, tablet held against the side
    sl = np.array([-f(0.13), 0.0, Z(0.62)])
    el2 = np.array([-f(0.16), -f(0.03), Z(0.47)])
    hd2 = np.array([-f(0.13), -f(0.09), Z(0.52)])
    out.append(_tube([sl, el2, hd2], [f(0.045), f(0.04), f(0.035)], mat=mat, kind="beam", section="round"))
    out.append(_box(-f(0.18), -f(0.13), Z(0.38), -f(0.09), -f(0.075), Z(0.58), mat=mat, kind="other", yaw=-12))
    return out


TEMPLATES = {"aqueduct": aqueduct, "colossus": colossus}
