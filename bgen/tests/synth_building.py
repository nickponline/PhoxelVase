"""Synthetic pre-fractured building for render tests (stand-in for the real generator).

Floors of slabs + exterior walls with punched windows (glass panels), one interior wall and
columns per floor, a roof slab with parapets. Every panel is split into a jittered grid of convex
quads (fake Voronoi), so the chunk count is controlled by `cell`. Edges come from AABB contacts.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, box

from bgen.export.bld import assemble
from bgen.geom.prism import make_prism
from bgen.materials import DEFAULT_MATERIALS, FACE_CAP_EDGE, FACE_INNER, F_ANCHOR, F_GLASS, bond_strength
from bgen.model import BuildingContext, Edge, Room


def _frame_wall(p0, p1, z0, t, inward):
    """Frame for a vertical wall along p0->p1 at base z0; solid spans the facade line inward by t."""
    ex = np.array([p1[0] - p0[0], p1[1] - p0[1], 0.0]); ex /= np.linalg.norm(ex)
    ey = np.array([0.0, 0.0, 1.0])
    ez = np.cross(ex, ey)
    o = np.array([p0[0], p0[1], z0], float)
    inw = np.array([inward[0], inward[1], 0.0])
    if ez @ inw < 0:
        o = o + t * inw  # solid spans o + s*ez, s in [0,t] = facade line .. inward
    F = np.eye(4)
    F[:3, 0], F[:3, 1], F[:3, 2], F[:3, 3] = ex, ey, ez, o
    return F


def _frame_slab(z0):
    F = np.eye(4); F[2, 3] = z0
    return F


def _grid_cells(w, h, cell, rng, holes=()):
    """Jittered grid of convex quads covering [0,w]x[0,h]; cells inside `holes` boxes skipped.

    Returns list of (poly (4,2), side_flags (4,)) where border/hole sides are FACE_CAP_EDGE."""
    xs = sorted(set([0.0, w] + [v for hb in holes for v in (hb[0], hb[2])]))
    ys = sorted(set([0.0, h] + [v for hb in holes for v in (hb[1], hb[3])]))

    def subdiv(vals):
        out = [vals[0]]
        for a, b in zip(vals[:-1], vals[1:]):
            k = max(1, int(round((b - a) / cell)))
            out += list(np.linspace(a, b, k + 1)[1:])
        return np.array(out)

    gx, gy = subdiv(xs), subdiv(ys)
    nx, ny = len(gx), len(gy)
    X, Y = np.meshgrid(gx, gy, indexing="ij")
    jx = np.diff(gx).min() * 0.18 if nx > 1 else 0
    jy = np.diff(gy).min() * 0.18 if ny > 1 else 0
    fixed_x = np.isin(gx, xs)[:, None] | np.zeros((1, ny), bool)
    fixed_y = np.isin(gy, ys)[None, :] | np.zeros((nx, 1), bool)
    X = X + np.where(fixed_x, 0, rng.uniform(-jx, jx, X.shape))
    Y = Y + np.where(fixed_y, 0, rng.uniform(-jy, jy, Y.shape))

    def in_hole(i, j):
        cx, cy = (gx[i] + gx[i + 1]) / 2, (gy[j] + gy[j + 1]) / 2
        return any(hb[0] < cx < hb[2] and hb[1] < cy < hb[3] for hb in holes)

    out = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if in_hole(i, j):
                continue
            poly = np.array([[X[i, j], Y[i, j]], [X[i + 1, j], Y[i + 1, j]],
                             [X[i + 1, j + 1], Y[i + 1, j + 1]], [X[i, j + 1], Y[i, j + 1]]])
            # sides: bottom(j), right(i+1), top(j+1), left(i)
            nb = [(i, j - 1), (i + 1, j), (i, j + 1), (i - 1, j)]
            sf = np.array([FACE_CAP_EDGE if not (0 <= a < nx - 1 and 0 <= b < ny - 1) or in_hole(a, b)
                           else FACE_INNER for a, b in nb], np.uint16)
            out.append((poly, sf))
    return out


def synth_building(floors: int = 3, w: float = 12.0, d: float = 8.0, fh: float = 3.0, cell: float = 1.0,
                   seed: int = 0, windows: bool = True):
    rng = np.random.default_rng(seed)
    ctx = BuildingContext(spec={"name": f"synth_{floors}f", "seed": seed},
                          params={"floor_height": fh, "slab_thickness": 0.25, "wall_thickness": 0.25})
    st, wt = 0.25, 0.25
    chunks = []

    def add(kind, frame, pw, ph, t, mat, floor, holes=(), anchor=False, cell_k=1.0):
        p = ctx.add_panel(kind=kind, frame=frame, profile=box(0, 0, pw, ph), thickness=t, material=mat,
                          tags={"floor": floor, "anchor": anchor})
        dens = DEFAULT_MATERIALS[mat].density
        for poly, sf in _grid_cells(pw, ph, cell * cell_k, rng, holes):
            c = make_prism(poly, t, frame, dens, p.id, mat, sf, flags=F_ANCHOR if anchor else 0, hp=100.0)
            chunks.append(c)
        return p

    glass = []
    for f in range(floors + 1):
        z = f * fh
        last = f == floors
        add("roof" if last else "floor", _frame_slab(z), w, d, st, "concrete", f if not last else floors - 1,
            anchor=(f == 0), cell_k=1.6)
        if last:
            break
        zb, hh = z + st, fh - st
        walls = [((0, 0), (w, 0), (0, 1), w), ((w, d), (0, d), (0, -1), w),
                 ((0, d - wt), (0, wt), (1, 0), d - 2 * wt), ((w, wt), (w, d - wt), (-1, 0), d - 2 * wt)]
        for wi, (p0, p1, inw, L) in enumerate(walls):
            holes = []
            if windows:
                n_win = max(1, int(L // 3))
                for k in range(n_win):
                    cx = (k + 0.5) * L / n_win
                    if f == 0 and wi == 0 and k == 0:
                        holes.append((cx - 0.5, 0.0, cx + 0.5, 2.1))  # door
                    else:
                        holes.append((cx - 0.7, 0.9, cx + 0.7, 2.2))
            F = _frame_wall(p0, p1, zb, wt, inw)
            add("ext_wall", F, L, hh, wt, "brick" if f % 2 else "concrete", f, holes)
            for hb in holes:
                if hb[1] > 0:
                    glass.append((F, hb, f))
        # interior wall along y at x = w*0.55 with a door
        xi = w * 0.55
        F = _frame_wall((xi, wt), (xi, d - wt), zb, 0.15, (1, 0))
        add("int_wall", F, d - 2 * wt, hh, 0.15, "wood", f, [(1.0, 0.0, 2.0, 2.1)])
        # two columns
        for cx in (w * 0.25, w * 0.8):
            F = _frame_wall((cx - 0.15, d / 2 - 0.15), (cx + 0.15, d / 2 - 0.15), zb, 0.3, (0, 1))
            add("column", F, 0.3, hh, 0.3, "concrete", f, cell_k=1.0)
        ctx.rooms.append(Room(len(ctx.rooms), f, Polygon([(wt, wt), (xi, wt), (xi, d - wt), (wt, d - wt)]),
                              "office" if f else "lobby"))
        ctx.rooms.append(Room(len(ctx.rooms), f, Polygon([(xi + .15, wt), (w - wt, wt), (w - wt, d - wt), (xi + .15, d - wt)]),
                              "meeting" if f % 2 else "kitchen"))
    # parapets
    zr = floors * fh + st
    for p0, p1, inw, L in [((0, 0), (w, 0), (0, 1), w), ((w, d), (0, d), (0, -1), w)]:
        add("parapet", _frame_wall(p0, p1, zr, 0.2, inw), L, 0.9, 0.2, "concrete", floors - 1)
    # glass panes (single chunk each), mid-thickness of wall
    for F, hb, f in glass:
        G = F.copy(); G[:3, 3] = F[:3, 3] + F[:3, 2] * (wt / 2 - 0.01)
        p = ctx.add_panel(kind="glass", frame=G, profile=box(*hb), thickness=0.02, material="glass",
                          tags={"floor": f})
        poly = np.array([[hb[0], hb[1]], [hb[2], hb[1]], [hb[2], hb[3]], [hb[0], hb[3]]])
        chunks.append(make_prism(poly, 0.02, G, 2500.0, p.id, "glass",
                                 np.full(4, FACE_CAP_EDGE, np.uint16), flags=F_GLASS, hp=1.0))

    edges = _aabb_edges(chunks)
    return assemble(ctx, chunks, edges)


def _aabb_edges(chunks, tol=1e-3):
    lo = np.array([c.hull_verts.min(0) for c in chunks])
    hi = np.array([c.hull_verts.max(0) for c in chunks])
    ctr, rad = (lo + hi) / 2, np.linalg.norm(hi - lo, axis=1) / 2
    tree = cKDTree(ctr)
    pairs = tree.query_pairs(2 * rad.max() + tol, output_type="ndarray")
    a, b = pairs[:, 0], pairs[:, 1]
    ov = np.minimum(hi[a], hi[b]) - np.maximum(lo[a], lo[b])
    ok = (ov > -tol).all(1)
    a, b, ov = a[ok], b[ok], np.maximum(ov[ok], 0)
    s = np.sort(ov, axis=1)
    area = s[:, 1] * s[:, 2]
    touch = (s[:, 0] < 0.05) & (area > 0.01)
    edges = []
    for i, j, ar in zip(a[touch], b[touch], area[touch]):
        ci, cj = chunks[i], chunks[j]
        if "glass" in (ci.material, cj.material):
            continue
        cen = (ci.com + cj.com) / 2
        nrm = cj.com - ci.com; nrm /= max(np.linalg.norm(nrm), 1e-9)
        edges.append(Edge(int(i), int(j), float(ar), float(ar * bond_strength(ci.material, cj.material)), cen, nrm))
    return edges
