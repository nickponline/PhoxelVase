"""Parametric generators for the `sculpt` node: params (ranges already sampled) -> primitives.

    eiffel             wrought-iron lattice tower: 4 truss legs merging into a tapering shaft
    suspension_bridge  two-tower suspension bridge: deck, portal towers, parabolic cables, hangers
    aqueduct           three-tier stone arch viaduct (Pont du Gard-like)
    colossus           robed figure with a raised torch on a stepped pedestal

All coordinates are metres, z up, ground at z = 0. Elements that must survive slicing
(horizontal beams) sit at layer mid-heights via `_mid`; tube radii stay >= layer/2.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union


def _mid(z: float, h: float) -> float:
    """Snap z to the nearest layer mid-height (thin horizontal members land in one layer)."""
    return (math.floor(z / h) + 0.5) * h


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


# ----------------------------------------------------------------------------- eiffel

def eiffel(p: dict, rng) -> list[dict]:
    H = float(p.get("height", 96.0))
    h = float(p.get("layer", 0.5))
    chord = float(p.get("chord", 0.4))
    brace = float(p.get("brace", 0.28))
    # outer half-width w(z) and leg half-width a(z), normalised by H (after the real tower)
    zf = np.array([0.0, 0.10, 0.19, 0.30, 0.385])
    wf = np.array([0.208, 0.158, 0.117, 0.085, 0.067]) * float(p.get("spread", 1.0))
    af = np.array([0.0435, 0.041, 0.038, 0.035, 0.0335])
    z1, z2 = _snap(0.19 * H, h), _snap(0.385 * H, h)
    w = lambda z: float(np.interp(z / H, zf, wf)) * H
    a = lambda z: min(float(np.interp(z / H, zf, af)) * H, 0.5 * w(z))
    c = lambda z: w(z) - a(z)
    sz = np.array([0.385, 0.5, 0.7, 0.92, 0.95])
    sw = np.array([wf[-1], 0.048, 0.028, 0.016, 0.014])
    ws = lambda z: float(np.interp(z / H, sz, sw)) * H
    z3 = _snap(0.92 * H, h)
    out: list[dict] = []

    def truss(cx, cy, half, z_lo, z_hi, sx, sy):
        """Square lattice between z_lo..z_hi: centre (cx(z), cy(z)), half-width half(z);
        sx, sy flip the quadrant. Chords + rings + X-bracing on all 4 faces."""
        zs = np.linspace(z_lo, z_hi, max(2, int((z_hi - z_lo) / 2.0) + 1))
        for ex in (-1, 1):
            for ey in (-1, 1):
                out.append(_tube([(sx * (cx(z) + ex * (half(z) - chord)),
                                   sy * (cy(z) + ey * (half(z) - chord)), z) for z in zs], chord, kind="column"))
        levels = [z_lo]
        while True:
            nz = _mid(levels[-1] + max(1.6 * half(levels[-1]), 3.0), h)
            if nz > z_hi - 2.0:
                break
            levels.append(nz)
        levels = levels[1:]
        for zl in levels:
            C = [(sx * (cx(zl) + ex * (half(zl) - chord)), sy * (cy(zl) + ey * (half(zl) - chord)), zl)
                 for ex, ey in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            for i in range(4):
                out.append(_tube([C[i], C[(i + 1) % 4]], brace + 0.05))
        bounds = [z_lo] + levels + [z_hi]
        for zl, zu in zip(bounds[:-1], bounds[1:]):
            def corner(z, ex, ey):
                return (sx * (cx(z) + ex * (half(z) - chord)), sy * (cy(z) + ey * (half(z) - chord)), z)
            for (e0, e1) in (((-1, -1), (1, -1)), ((1, -1), (1, 1)), ((1, 1), (-1, 1)), ((-1, 1), (-1, -1))):
                out.append(_tube([corner(zl, *e0), corner(zu, *e1)], brace))
                out.append(_tube([corner(zl, *e1), corner(zu, *e0)], brace))

    # four legs, base to the second platform
    for sx in (-1, 1):
        for sy in (-1, 1):
            truss(c, c, a, 0.0, z2, sx, sy)
            cx0, a0 = c(0.0), a(0.0)
            out.append(_box(sx * cx0 - a0 - 0.6, sy * cx0 - a0 - 0.6, 0.0,
                            sx * cx0 + a0 + 0.6, sy * cx0 + a0 + 0.6, 2.0, mat="concrete", kind="column"))
    # decorative arches under the first platform, one per face
    zs_ = _snap(0.05 * H, h)
    t = np.linspace(0.0, math.pi, 25)
    for along_x, side in ((True, -1), (True, 1), (False, -1), (False, 1)):
        pts = []
        for tt in t:
            zz = zs_ + (z1 - 0.5 * h - zs_) * math.sin(tt)   # crown in the layer under platform 1
            u = (c(zz) - a(zz) + 0.4) * math.cos(tt)      # across the face, leg to leg
            v = side * (w(zz) - chord)                     # in the (sloping) outer face
            pts.append((u, v, zz) if along_x else (v, u, zz))
        out.append(_tube(pts, 0.35, kind="beam"))
    # platforms (rings with a central void) and the shaft
    for zp, wo, wi, th in ((z1, w(z1) + 1.6, max(c(z1) - a(z1) - 0.5, 2.0), 1.0),
                           (z2, w(z2) + 1.2, 0.45 * w(z2), 1.0)):
        for part in (box(-wo, -wo, wo, -wi), box(-wo, wi, wo, wo), box(-wo, -wi, -wi, wi), box(wi, -wi, wo, wi)):
            x0, y0, x1, y1 = part.bounds
            out.append(_box(x0, y0, zp, x1, y1, zp + th, mat="metal", kind="floor"))
        for x0, y0, x1, y1 in ((-wo, -wo, wo, -wo + 0.3), (-wo, wo - 0.3, wo, wo),
                               (-wo, -wo + 0.3, -wo + 0.3, wo - 0.3), (wo - 0.3, -wo + 0.3, wo, wo - 0.3)):
            out.append(_box(x0, y0, zp + th, x1, y1, zp + th + 1.0, mat="metal", kind="parapet"))
    zero = lambda z: 0.0
    truss(zero, zero, ws, z2 + 1.0, z3, 1, 1)
    # top cabin, lantern and spire
    wc = ws(z3) + 0.8
    out.append(_box(-wc, -wc, z3, wc, wc, z3 + 3.0, mat="metal", kind="floor"))
    out.append({"lathe": {"center": [0, 0], "sides": 4, "profile": [[z3 + 3.0, wc - 0.6], [z3 + 6.0, 1.0]]},
                "material": "metal", "kind": "roof"})
    out.append(_tube([(0, 0, z3 + 5.5), (0, 0, H)], 0.35, kind="column", section="round"))
    return out


# ----------------------------------------------------------------------------- suspension bridge

def suspension_bridge(p: dict, rng) -> list[dict]:
    L = float(p.get("span", 110.0))
    Ls = float(p.get("side_ratio", 0.4)) * L
    h = float(p.get("layer", 0.5))
    zd = _snap(float(p.get("deck_z", 16.0)), h)            # deck underside
    W = float(p.get("width", 14.0))
    zt = _snap(zd + float(p.get("tower_ratio", 0.17)) * L + 6.0, h)  # tower top
    hanger_dx = float(p.get("hanger_spacing", 5.0))
    tower_mat = p.get("tower_material", "metal")
    td = 1.0                                           # deck thickness
    gw = 1.0                                           # edge (stiffening) girder width
    gd = float(p.get("girder_depth", 3.0))             # girder depth below the deck
    yc = W / 2 + 0.5 * gw                              # cable / edge girder line
    yl = W / 2 + gw + 1.5                              # tower leg centre (legs clear the girders)
    lw = 1.4                                           # leg half-width at the base
    out: list[dict] = []
    x_lo, x_hi = -Ls, L + Ls
    # deck + edge girders (girders stand 1.5 m above the deck as rails)
    out.append(_box(x_lo, -W / 2, zd, x_hi, W / 2, zd + td, kind="floor"))
    for s in (-1, 1):
        y0, y1 = sorted([s * W / 2, s * (W / 2 + gw)])
        out.append(_box(x_lo, y0, zd - gd, x_hi, y1, zd + td + 1.5, mat="metal", kind="beam"))
    # towers: tapered legs, pier, portal beams
    for xt in (0.0, L):
        out.append(_box(xt - 4.0, -yl - 4.0, 0.0, xt + 4.0, yl + 4.0, 4.0, kind="column"))
        for s in (-1, 1):
            out.append({"lathe": {"center": [xt, s * yl], "sides": 4,
                                  "profile": [[4.0, lw], [zt, lw * 0.7]]}, "material": tower_mat, "kind": "column"})
        portals = [zd - gd - 1.0, _snap(zd + 0.45 * (zt - zd), h), _snap(zd + 0.8 * (zt - zd), h), zt - 2.5]
        for zp in portals:
            out.append(_box(xt - 1.0, -yl, zp, xt + 1.0, yl, zp + 2.0, mat=tower_mat, kind="beam"))
        out.append(_box(xt - 1.5, -yl - 0.9, zt, xt + 1.5, yl + 0.9, zt + 1.0, mat=tower_mat, kind="beam"))
    # cables: parabolic main span, side spans down to the anchorages
    zc_top = zt + 1.0 + 0.45
    z_sag = zd + td + 1.5 + 2.0
    za = zd + td + 3.5                                 # cables end in anchor blocks just past the deck
    xa = x_lo - 4.0                                    # ends, so they never cross the deck

    def cable_z(x):
        if 0.0 <= x <= L:
            u = x / L
            return zc_top - 4.0 * (zc_top - z_sag) * u * (1 - u)
        x0, x1 = (xa, 0.0) if x < 0 else (L, L - xa)
        za_, zb = (za, zc_top) if x < 0 else (zc_top, za)
        u = (x - x0) / (x1 - x0)
        return za_ + (zb - za_) * u - 0.06 * (x1 - x0) * 4 * u * (1 - u)

    xs = np.concatenate([np.linspace(xa, 0.0, 16), np.linspace(0.0, L, 40)[1:], np.linspace(L, L - xa, 16)[1:]])
    for s in (-1, 1):
        out.append(_tube([(x, s * yc, cable_z(x)) for x in xs], 0.45, kind="beam", section="round"))
    # anchorages (the cables end in them) + deck-end abutment walls
    for side in (-1, 1):
        xe = x_lo if side < 0 else x_hi
        xb = xa if side < 0 else L - xa
        out.append(_box(xb - 3.0, -yc - 1.5, 0.0, xb + 3.0, -yc + 1.5, za + 1.5, kind="column", mirror=["y"]))
        x0, x1 = sorted([xe, xe - side * 3.0])          # under the deck end: girders + deck rest on it
        out.append(_box(x0, -W / 2 - gw, 0.0, x1, W / 2 + gw, zd - gd, kind="column"))
        out.append(_box(x0, -W / 2, zd - gd, x1, W / 2, zd, kind="column"))
    # hangers
    for s in (-1, 1):
        x = x_lo + hanger_dx
        while x < x_hi - hanger_dx * 0.5:
            if min(abs(x), abs(x - L)) > 2.5:
                ztop = cable_z(x)
                if ztop > zd + td + 2.0:
                    out.append(_tube([(x, s * yc, zd + td + 1.5), (x, s * yc, ztop)], 0.25, kind="beam"))
            x += hanger_dx
    return out


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


TEMPLATES = {"eiffel": eiffel, "suspension_bridge": suspension_bridge, "aqueduct": aqueduct, "colossus": colossus}
