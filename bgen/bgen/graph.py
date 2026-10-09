"""Chunk connection graph + anchors.

Edges:
  * intra-panel: shared 2D boundaries between chunk polygons of the same panel
    (area = shared length × thickness), found by `convex2d.polygon_adjacency`.
  * inter-panel: anti-parallel (dot < -0.999), coplanar (|Δplane| < 1e-4 m) hull faces of
    chunks in different panels whose AABBs (expanded by 1e-3) overlap; contact area is the
    shapely intersection of both faces projected into the plane.

strength = area × bond_strength(mat_a, mat_b). Pairs whose bond strength is zero (glass is
non-structural) or that involve a F_COSMETIC_ATTACHED / F_GLASS chunk get an *attachment*
edge instead: strength = area × ATTACH_STRENGTH (1e4 N/m²), so glass and cosmetic chunks
detach together with their parents but never carry structural load. Attachment edges are
kept for any positive contact area (min_contact applies to structural edges only), so a
thin glass pane is never left floating. Likewise a (tiny) chunk all of whose contacts are
below min_contact keeps its largest contact, so it never floats.
"""
from __future__ import annotations

import numpy as np
import shapely

from .geom.convex2d import bbox_pairs, polygon_adjacency, snap
from .materials import F_ANCHOR, F_COSMETIC_ATTACHED, F_GLASS, bond_strength
from .model import ChunkGeom, Edge, Panel

ATTACH_STRENGTH = 1.0e4      # N/m² for glass / cosmetic attachment edges
PLANE_TOL = 1e-4             # m, coplanarity tolerance for inter-panel contacts
ANTI_DOT = -0.999
AABB_PAD = 1e-3
_ATTACH_MIN_AREA = 1e-6


def chunk_aabbs(chunks: list[ChunkGeom]) -> tuple[np.ndarray, np.ndarray]:
    lo = np.array([c.hull_verts.min(0) for c in chunks]).reshape(-1, 3)
    hi = np.array([c.hull_verts.max(0) for c in chunks]).reshape(-1, 3)
    return lo, hi


def _edge_strength(ma: str, mb: str, fa: int, fb: int, area: float) -> tuple[float, bool]:
    """(strength, is_attachment)."""
    s = bond_strength(ma, mb)
    if s <= 0.0 or ((fa | fb) & (F_COSMETIC_ATTACHED | F_GLASS)):
        return area * ATTACH_STRENGTH, True
    return area * s, False


def _intra_panel_edges(panels_by_id: dict[int, Panel], chunks: list[ChunkGeom]):
    pid = np.array([c.panel_id for c in chunks])
    I, J, L, mid, nrm = polygon_adjacency([c.poly2d for c in chunks], groups=pid, tol=1e-6)
    out = []
    for i, j, l, m2, n2 in zip(I.tolist(), J.tolist(), L.tolist(), mid, nrm):
        p = panels_by_id[chunks[i].panel_id]
        R, T = p.frame[:3, :3], p.frame[:3, 3]
        area = l * p.thickness
        cen = R @ np.array([m2[0], m2[1], 0.5 * p.thickness]) + T
        nw = R @ np.array([n2[0], n2[1], 0.0])
        out.append((i, j, area, cen, nw))
    return out


def _inter_panel_edges(chunks: list[ChunkGeom], batch: int = 4096):
    N = len(chunks)
    lo, hi = chunk_aabbs(chunks)
    A, B = bbox_pairs(lo, hi, pad=AABB_PAD)
    pid = np.array([c.panel_id for c in chunks])
    k = pid[A] != pid[B]
    A, B = A[k], B[k]
    if len(A) == 0:
        return []
    F = max(len(c.planes) for c in chunks)
    PL = np.full((N, F, 4), np.nan)
    for i, c in enumerate(chunks):
        PL[i, : len(c.planes)] = c.planes
    matches = []
    for s in range(0, len(A), batch):
        a, b = A[s:s + batch], B[s:s + batch]
        na, nb = PL[a, :, None, :3], PL[b, None, :, :3]
        dot = (na * nb).sum(-1)                          # (K,F,F)
        dd = np.abs(PL[a, :, None, 3] + PL[b, None, :, 3])  # anti-parallel: d_a ≈ -d_b
        hit = (dot < ANTI_DOT) & (dd < PLANE_TOL)
        kk, fa, fb = np.nonzero(hit)
        matches.append(np.column_stack([a[kk], b[kk], fa, fb]))
    M = np.vstack(matches)
    if len(M) == 0:
        return []
    ia, ib, fa, fb = M[:, 0], M[:, 1], M[:, 2], M[:, 3]
    # flat hull-vertex table
    nverts = np.array([len(c.hull_verts) for c in chunks])
    HV = np.vstack([c.hull_verts for c in chunks])
    hoff = np.concatenate([[0], np.cumsum(nverts)[:-1]])
    npoly = nverts // 2

    def face_vertex_index(ch, f):
        """Ragged face loops (global HV indices) for faces f of chunks ch -> (idx, owner)."""
        n = npoly[ch]
        L = np.where(f < n, 4, n)
        owner = np.repeat(np.arange(len(ch)), L)
        k = np.arange(L.sum()) - np.repeat(np.cumsum(L) - L, L)
        nn, ff, base = n[owner], f[owner], hoff[ch][owner]
        side = ff < nn
        f1 = (ff + 1) % nn
        side_idx = np.select([k == 0, k == 1, k == 2], [ff, f1, nn + f1], nn + ff)
        bot_idx = nn - 1 - k
        top_idx = nn + k
        loc = np.where(side, side_idx, np.where(ff == nn, bot_idx, top_idx))
        return base + loc, owner, L

    va_i, va_o, La = face_vertex_index(ia, fa)
    vb_i, vb_o, Lb = face_vertex_index(ib, fb)
    nrm = PL[ia, fa, :3]
    dpl = PL[ia, fa, 3]
    # exact coplanarity check using the vertices of b's face
    dist = np.abs(np.einsum("ij,ij->i", HV[vb_i], nrm[vb_o]) + dpl[vb_o])
    worst = np.zeros(len(M))
    np.maximum.at(worst, vb_o, dist)
    # in-plane basis per match
    ref = np.where(np.abs(nrm[:, :1]) < 0.9, np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    u = np.cross(nrm, ref)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(nrm, u)
    org = HV[va_i[np.concatenate([[0], np.cumsum(La)[:-1]])]]

    def project(vi, vo):
        d = HV[vi] - org[vo]
        return snap(np.column_stack([np.einsum("ij,ij->i", d, u[vo]), np.einsum("ij,ij->i", d, v[vo])]))

    ga = shapely.polygons(shapely.linearrings(project(va_i, va_o), indices=va_o))
    gb = shapely.polygons(shapely.linearrings(project(vb_i, vb_o), indices=vb_o))
    ok = worst <= PLANE_TOL
    inter = shapely.intersection(ga[ok], gb[ok])
    areas = np.zeros(len(M))
    areas[ok] = shapely.area(inter)
    cx = np.zeros(len(M)); cy = np.zeros(len(M))
    good = areas > 0.0
    cen = shapely.centroid(inter[good[ok]])
    cx[good], cy[good] = shapely.get_x(cen), shapely.get_y(cen)
    p3 = org + cx[:, None] * u + cy[:, None] * v
    sel = np.flatnonzero(good)
    if len(sel) == 0:
        return []
    # aggregate per chunk pair (convex chunks touch in at most one plane, but be safe)
    key = ia[sel] * N + ib[sel]
    uk, inv = np.unique(key, return_inverse=True)
    ar = areas[sel]
    tot = np.bincount(inv, ar)
    cen3 = np.column_stack([np.bincount(inv, ar * p3[sel, k]) for k in range(3)]) / tot[:, None]
    # normal of the largest contributing face
    order = np.lexsort((-ar, inv))
    first = order[np.concatenate([[True], inv[order][1:] != inv[order][:-1]])]
    nr = nrm[sel][first]
    return [(int(k // N), int(k % N), float(t), c, n) for k, t, c, n in zip(uk, tot, cen3, nr)]


def build_graph(panels: list[Panel], chunks: list[ChunkGeom], min_contact: float = 0.01) -> list[Edge]:
    """Connection graph between chunks (indices into `chunks`); edges have a < b,
    normal pointing from a to b, sorted by (a, b). See module doc for strengths."""
    if len(chunks) < 2:
        return []
    pbi = {p.id: p for p in panels}
    raw = _intra_panel_edges(pbi, chunks) + _inter_panel_edges(chunks)
    best: dict[tuple[int, int], Edge] = {}
    small: dict[int, tuple] = {}     # chunk -> its largest sub-min_contact contact
    has_edge = np.zeros(len(chunks), bool)
    for i, j, area, cen, nrm in raw:
        ca, cb = chunks[i], chunks[j]
        strength, attach = _edge_strength(ca.material, cb.material, ca.flags, cb.flags, area)
        if area < (_ATTACH_MIN_AREA if attach else min_contact):
            for k in (i, j):
                if area > _ATTACH_MIN_AREA and (k not in small or area > small[k][2]):
                    small[k] = (i, j, area, cen, nrm)
            continue
        has_edge[i] = has_edge[j] = True
        nrm = np.asarray(nrm, float)
        nrm = nrm / np.linalg.norm(nrm)
        if i > j:
            i, j, nrm = j, i, -nrm
        key = (i, j)
        if key in best:  # same pair through two routes: accumulate
            e = best[key]
            tot = e.area + area
            e.centroid = (e.centroid * e.area + cen * area) / tot
            e.strength += strength
            e.area = tot
        else:
            best[key] = Edge(i, j, float(area), float(strength), np.asarray(cen, float), nrm)
    # a tiny chunk whose every contact is below min_contact would float: keep its largest one
    for k, (i, j, area, cen, nrm) in sorted(small.items()):
        if has_edge[k]:
            continue
        ca, cb = chunks[i], chunks[j]
        strength, _ = _edge_strength(ca.material, cb.material, ca.flags, cb.flags, area)
        nrm = np.asarray(nrm, float) / np.linalg.norm(nrm)
        if i > j:
            i, j, nrm = j, i, -nrm
        if (i, j) not in best:
            best[(i, j)] = Edge(i, j, float(area), float(strength), np.asarray(cen, float), nrm)
        has_edge[i] = has_edge[j] = True
    return [best[k] for k in sorted(best)]


def mark_anchors(panels: list[Panel], chunks: list[ChunkGeom], ground_z: float = 0.0, eps: float = 1e-3) -> None:
    """Set F_ANCHOR on chunks touching the ground (any hull vertex z <= ground_z + eps)
    or belonging to a panel tagged anchor=True."""
    anchored = {p.id for p in panels if (p.tags or {}).get("anchor")}
    for c in chunks:
        if c.panel_id in anchored or float(c.hull_verts[:, 2].min()) <= ground_z + eps:
            c.flags |= F_ANCHOR
