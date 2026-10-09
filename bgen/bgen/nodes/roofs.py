"""`roofs` node.

For every level k, the exposed region floor_polys[k-1] - floor_polys[k] gets a roof slab
(same level/thickness as a floor slab, so it meets floor slab k edge to edge). Then:
  flat   parapet walls along the exposed boundary that lies on the facade below.
  gable  rectangle parts not touching higher floors: two inclined panels from the eave line
         (passing through the slab-top facade edge) to the ridge, a ridge strip filling the
         wedge between their end faces, gable end walls whose top faces lie in the roof
         underside planes, trusses every <= 4.5 m, fascia strips on the eave end faces.
  kyoto  as gable but each side is a concave polyline of strips (sori curve); concave bends
         are bridged by triangular filler strips so prisms never overlap.
  hip    four planes + ridge strip + four hip strips (triangular prisms whose faces are
         the adjacent planes' end faces) + trusses in the ridge zone.
Every prism side face is perpendicular to its panel plane, so all contacts are coplanar.
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.polygon import orient

from ..blockout import roof_type_at
from ..model import BuildingContext
from ..rng import rv
from ._util import add_extruded, add_from_world, add_panel, emit_chain, horiz_frame, parts, simplify_ring, unit

UP = np.array([0.0, 0.0, 1.0])


def _rect_frame(part: Polygon):
    cs = simplify_ring(np.asarray(part.exterior.coords))
    if len(cs) != 4 or abs(part.area - part.minimum_rotated_rectangle.area) > 1e-3:
        return None
    e1, e2 = cs[1] - cs[0], cs[2] - cs[1]
    if abs(e1 @ e2) > 1e-6 * np.linalg.norm(e1) * np.linalg.norm(e2):
        return None
    if np.linalg.norm(e1) >= np.linalg.norm(e2) - 1e-9:
        r, L, W = e1 / np.linalg.norm(e1), np.linalg.norm(e1), np.linalg.norm(e2)
    else:
        r, L, W = e2 / np.linalg.norm(e2), np.linalg.norm(e2), np.linalg.norm(e1)
    if r[0] < -1e-9 or (abs(r[0]) < 1e-9 and r[1] < 0):
        r = -r
    C = np.asarray(part.centroid.coords[0])
    return C, r, float(L), float(W)


class Sec:
    """Section coordinates: s along ridge (0..L), u across (-W/2..W/2), z."""

    def __init__(self, C, r, L, W):
        self.C, self.L, self.W = C, L, W
        self.r2 = np.asarray(r, float)
        self.a2 = np.array([-r[1], r[0]])
        self.r = np.array([r[0], r[1], 0.0])
        self.a = np.array([self.a2[0], self.a2[1], 0.0])

    def P(self, s, u, z):
        xy = self.C + (s - self.L / 2) * self.r2 + u * self.a2
        return np.array([xy[0], xy[1], z])

    def vec(self, du, dz):
        return du * self.a + dz * UP


def _up_normal(d):
    n = np.array([-d[1], d[0]])
    return n if n[1] > 0 else -n


def section_roof(ctx, sec: Sec, z_top: float, Q: list, p: dict, k: int):
    """Q: left-side polyline from ridge apex outward [(u, z), ...], u decreasing, containing the
    vertex u = -W/2 at z_top. Right side is mirrored."""
    t = float(p["t"])
    mat = p["mat"]
    rake = float(p["rake"])
    s0, s1 = -rake, sec.L + rake
    Lr = s1 - s0
    tags = {"floor": k - 1, "level": k, "role": "roof"}
    Q = [np.asarray(q, float) for q in Q]
    # strips = merge collinear segments
    S = [Q[0]]
    for i in range(1, len(Q) - 1):
        d0, d1 = unit(Q[i] - S[-1]), unit(Q[i + 1] - Q[i])
        if abs(d0[0] * d1[1] - d0[1] * d1[0]) > 1e-9:
            S.append(Q[i])
    S.append(Q[-1])
    ds = [unit(S[i + 1] - S[i]) for i in range(len(S) - 1)]
    ns = [_up_normal(d) for d in ds]
    for i in range(1, len(ds)):
        assert ns[i - 1] @ ds[i] > -1e-9, "only concave (flattening) roof profiles are supported"
    for side in (-1.0, 1.0):          # -1 = left (as given), +1 = mirrored right
        m = np.array([1.0 if side < 0 else -1.0, 1.0])
        V = lambda q: q * m
        a0 = 0.0
        for i, d in enumerate(ds):
            if i > 0:
                a0 = t * float(ns[i - 1] @ d)
            qa = S[i] + a0 * d
            qb = S[i + 1]
            A, B = V(qa), V(qb)
            n2 = V(ns[i])
            nrm = sec.vec(n2[0], n2[1])
            add_from_world(ctx, "roof", [sec.P(s0, A[0], A[1]), sec.P(s1, A[0], A[1]),
                                         sec.P(s1, B[0], B[1]), sec.P(s0, B[0], B[1])],
                           nrm, t, mat, dict(tags, strip=i, side=int(side)), x_axis=sec.r)
            if i + 1 < len(ds):
                a1 = t * float(ns[i] @ ds[i + 1])
                if a1 > 1e-4:
                    R = S[i + 1]
                    tri = [V(R), V(R + t * ns[i]), V(R + a1 * ds[i + 1])]
                    add_extruded(ctx, "roof", [sec.P(s0, q[0], q[1]) for q in tri], sec.r, Lr, mat,
                                 dict(tags, role="roof_filler"), p["seg"])
        # fascia on the eave end face
        if p.get("fascia", True):
            E, nE, dE = V(S[-1]), V(ns[-1]), V(ds[-1])
            T = E + t * nE
            nseg = max(1, int(math.ceil(Lr / (2 * p["seg"]))))
            for j in range(nseg):
                sa, sb = s0 + Lr * j / nseg, s0 + Lr * (j + 1) / nseg
                add_from_world(ctx, "fascia", [sec.P(sa, E[0], E[1]), sec.P(sb, E[0], E[1]),
                                               sec.P(sb, T[0], T[1]), sec.P(sa, T[0], T[1])],
                               sec.vec(dE[0], dE[1]), 0.1, mat, dict(tags, role="fascia", segment=j),
                               x_axis=sec.r)
    # ridge strip
    R0 = S[0]
    nl, nr_ = ns[0], ns[0] * np.array([-1.0, 1.0])
    tri = [R0, R0 + t * nl, R0 + t * nr_]
    add_extruded(ctx, "ridge", [sec.P(s0, q[0], q[1]) for q in tri], sec.r, Lr, mat, dict(tags, role="ridge"), p["seg"])
    # gable end walls + trusses (top faces = roof underside polyline)
    W2 = sec.W / 2
    top = [q for q in Q if q[0] >= -W2 - 1e-9]
    prof = [(-W2, z_top), (W2, z_top)] + [(-q[0], q[1]) for q in reversed(top)][1:] + [(q[0], q[1]) for q in top][1:]
    prof = [np.asarray(q) for q in prof]
    # drop duplicate apex
    clean = [prof[0]]
    for q in prof[1:]:
        if np.linalg.norm(q - clean[-1]) > 1e-9:
            clean.append(q)
    if np.linalg.norm(clean[-1] - clean[0]) < 1e-9:
        clean.pop()
    te, ti = float(p["t_ext"]), float(p["t_int"])
    add_from_world(ctx, "ext_wall", [sec.P(0.0, u, z) for u, z in clean], sec.r, te, p["wall_mat"],
                   dict(tags, role="gable"))
    rev = [clean[1], clean[0]] + clean[2:][::-1]
    add_from_world(ctx, "ext_wall", [sec.P(sec.L, u, z) for u, z in rev], -sec.r, te, p["wall_mat"],
                   dict(tags, role="gable"))
    span = sec.L - 2 * te
    ntr = int(math.ceil(span / p["truss_spacing"])) - 1
    for j in range(ntr):
        s = te + span * (j + 1) / (ntr + 1) - ti / 2
        add_from_world(ctx, "int_wall", [sec.P(s, u, z) for u, z in clean], sec.r, ti, p["truss_mat"],
                       dict(tags, role="truss"))


def gable_profile(W, z_top, pitch, eave):
    tp = math.tan(pitch)
    return [(0.0, z_top + W / 2 * tp), (-W / 2, z_top), (-W / 2 - eave, z_top - eave * tp)]


def kyoto_profile(W, z_top, pitch, eave, n_in=3, n_eave=2, drop=8.0):
    pitches = [math.radians(max(4.0, math.degrees(pitch) - drop * i)) for i in range(n_in + n_eave)]
    widths = [W / 2 / n_in] * n_in + [eave / n_eave] * n_eave
    z_r = z_top + sum(w * math.tan(a) for w, a in zip(widths[:n_in], pitches[:n_in]))
    Q = [(0.0, z_r)]
    u, z = 0.0, z_r
    for w, a in zip(widths, pitches):
        u, z = u - w, z - w * math.tan(a)
        Q.append((u, z))
    Q[n_in] = (-W / 2, z_top)
    return Q


def hip_roof(ctx, sec: Sec, z_top: float, pitch: float, p: dict, k: int):
    t, mat = float(p["t"]), p["mat"]
    L, W, e = sec.L, sec.W, float(p["eave"])
    tags = {"floor": k - 1, "level": k, "role": "roof"}
    de = W / 2 if L - W >= 1.0 else max(0.5, (L - 1.0) / 2)
    tp = math.tan(pitch)
    pe = math.atan(W / 2 * tp / de)
    tpe = math.tan(pe)
    zr = z_top + W / 2 * tp
    ze = z_top - e * tp
    eC = e * tp / tpe
    c00, c10 = sec.P(-eC, -W / 2 - e, ze), sec.P(L + eC, -W / 2 - e, ze)
    c11, c01 = sec.P(L + eC, W / 2 + e, ze), sec.P(-eC, W / 2 + e, ze)
    R0, R1 = sec.P(de, 0, zr), sec.P(L - de, 0, zr)
    nA = -math.sin(pitch) * sec.a + math.cos(pitch) * UP
    nB = math.sin(pitch) * sec.a + math.cos(pitch) * UP
    nC = -math.sin(pe) * sec.r + math.cos(pe) * UP
    nD = math.sin(pe) * sec.r + math.cos(pe) * UP
    ridge = L - 2 * de > 1e-3
    if ridge:
        add_from_world(ctx, "roof", [c00, c10, R1, R0], nA, t, mat, dict(tags, plane="A"), x_axis=sec.r)
        add_from_world(ctx, "roof", [c11, c01, R0, R1], nB, t, mat, dict(tags, plane="B"), x_axis=-sec.r)
        add_extruded(ctx, "ridge", [R0, R0 + t * nA, R0 + t * nB], sec.r, L - 2 * de, mat, dict(tags, role="ridge"),
                     p["seg"])
    else:
        add_from_world(ctx, "roof", [c00, c10, R0], nA, t, mat, dict(tags, plane="A"), x_axis=sec.r)
        add_from_world(ctx, "roof", [c11, c01, R0], nB, t, mat, dict(tags, plane="B"), x_axis=-sec.r)
    add_from_world(ctx, "roof", [c01, c00, R0], nC, t, mat, dict(tags, plane="C"), x_axis=-sec.a)
    add_from_world(ctx, "roof", [c10, c11, R1], nD, t, mat, dict(tags, plane="D"), x_axis=sec.a)
    for c, n1, n2, R in ((c00, nA, nC, R0), (c01, nB, nC, R0), (c10, nA, nD, R1), (c11, nB, nD, R1)):
        h = R - c
        add_extruded(ctx, "ridge", [c, c + t * n1, c + t * n2], h, float(np.linalg.norm(h)), mat,
                     dict(tags, role="hip"), p["seg"])
    # trusses inside the ridge zone
    ti = float(p["t_int"])
    zone = L - 2 * de - ti
    prof = [sec.P(0, -W / 2, z_top), sec.P(0, W / 2, z_top), sec.P(0, 0, zr)]
    if zone >= 0:
        ntr = max(1, int(math.ceil((L - 2 * de) / p["truss_spacing"])))
        for j in range(ntr):
            s = de + (zone * (j + 0.5) / ntr if ntr > 1 else zone / 2)
            pts = [q + s * sec.r for q in prof]
            add_from_world(ctx, "int_wall", pts, sec.r, ti, p["truss_mat"], dict(tags, role="truss"))


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    bo = ctx.blockout
    n = len(bo.floor_polys)
    ts = float(ctx.params["slab_thickness"])
    te = float(ctx.params["ext_wall_thickness"])
    ti = float(ctx.params["int_wall_thickness"])
    flat = dict(params.get("flat", {}) or {})
    ph = float(rv(flat.get("parapet", 1.0), rng))
    slab_mat = ctx.params.get("slab_material", "concrete")
    wall_mat = ctx.params.get("ext_wall_material", "concrete")
    roof_mat = ctx.params.get("roof_material", "concrete")
    t_roof = float(ctx.params.get("roof_thickness", 0.2))
    ctx.meta.setdefault("roofs", [])
    for k in range(1, n + 1):
        below = bo.floor_polys[k - 1]
        above = bo.floor_polys[k] if k < n else None
        exposed = below if above is None else below.difference(above)
        z = bo.floor_z[k]
        for part in parts(exposed):
            if part.area < 0.5:
                continue
            part = orient(Polygon(simplify_ring(np.asarray(part.exterior.coords)),
                                  [simplify_ring(np.asarray(r.coords)) for r in part.interiors]), 1.0)
            rp = part.representative_point()
            rtype = roof_type_at(bo, k, (rp.x, rp.y))
            add_panel(ctx, "roof", horiz_frame(z - ts), part, ts, slab_mat if k < n or rtype == "flat" else slab_mat,
                      {"floor": k - 1, "level": k, "role": "roof_slab"})
            free = above is None or part.distance(above) > 0.5
            rf = _rect_frame(part) if (rtype in ("gable", "hip", "kyoto") and free) else None
            if rf is None:
                rtype = "flat"
            ctx.meta["roofs"].append({"level": k, "type": rtype, "area": float(part.area)})
            if rtype == "flat":
                _parapets(ctx, part, below, z, ph, te, wall_mat, k)
                continue
            C, r, L, W = rf
            sec = Sec(C, r, L, W)
            rp_ = dict(params.get(rtype, {}) or {})
            pitch = math.radians(float(rv(rp_.get("pitch_deg", 35.0 if rtype != "kyoto" else 40.0), rng)))
            eave = float(rv(rp_.get("eaves", 0.4 if rtype != "kyoto" else 0.9), rng))
            P = dict(t=t_roof, mat=roof_mat, rake=float(rp_.get("rake", eave)), t_ext=te, t_int=ti,
                     wall_mat=wall_mat, truss_mat=rp_.get("truss_material", roof_mat), eave=eave,
                     truss_spacing=float(rp_.get("truss_spacing", 4.5)), fascia=rp_.get("fascia", True),
                     seg=float(rp_.get("strip_segment", 1.2)))
            if rtype == "gable":
                section_roof(ctx, sec, z, gable_profile(W, z, pitch, eave), P, k)
            elif rtype == "kyoto":
                ns = int(rp_.get("strips", 4))
                section_roof(ctx, sec, z, kyoto_profile(W, z, pitch, eave, n_in=max(2, ns - 1), n_eave=2), P, k)
            else:
                hip_roof(ctx, sec, z, pitch, P, k)
    return ctx


def _parapets(ctx, part, below, z, ph, te, mat, k):
    if ph <= 0:
        return
    ring = simplify_ring(np.asarray(part.exterior.coords))
    nv = len(ring)
    bnd = below.boundary
    on = []
    for i in range(nv):
        a, b = ring[i], ring[(i + 1) % nv]
        on.append(all(bnd.distance(Point(*q)) < 1e-4 for q in (a, b, 0.5 * (a + b))))
    tags = {"floor": k - 1, "level": k, "role": "parapet"}
    if all(on):
        emit_chain(ctx, ring, True, te, z, z + ph, "parapet", mat, tags)
        return
    if not any(on):
        return
    start = next(i for i in range(nv) if not on[i])
    i = (start + 1) % nv
    run = []
    for _ in range(nv):
        if on[i]:
            run.append(i)
        elif run:
            pts = [ring[j] for j in run] + [ring[(run[-1] + 1) % nv]]
            emit_chain(ctx, pts, False, te, z, z + ph, "parapet", mat, tags)
            run = []
        i = (i + 1) % nv
    if run:
        pts = [ring[j] for j in run] + [ring[(run[-1] + 1) % nv]]
        emit_chain(ctx, pts, False, te, z, z + ph, "parapet", mat, tags)
