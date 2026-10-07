"""2D convex helpers: convex decomposition (earcut + Hertel–Mehlhorn), polygon cleanup,
convex hulls and vectorised edge-adjacency between polygon soups (DESIGN.md §2.5/§2.6).
"""
from __future__ import annotations

import mapbox_earcut
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

# relative tolerance for "straight" / "convex" turn tests (cross / (|e1||e2|))
_REL_TURN = 1e-10


SNAP = 1e-9   # m; coordinate grid for generated vertices (see snap())


def snap(p: np.ndarray, q: float = SNAP) -> np.ndarray:
    """Round coordinates to a q-grid. Vertices computed independently by different boolean
    ops (e.g. two neighbouring cells clipped by the same profile edge) can differ by a few
    ulps; GEOS 3.13 overlay then occasionally returns grossly wrong results (a whole
    polygon as the intersection of two edge-adjacent ones). Snapping makes them identical."""
    return np.round(np.asarray(p, dtype=np.float64) / q) * q


def _nxt(p: np.ndarray) -> np.ndarray:
    return np.concatenate([p[1:], p[:1]])


def _prv(p: np.ndarray) -> np.ndarray:
    return np.concatenate([p[-1:], p[:-1]])


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]) + x[-1] * y[0] - x[0] * y[-1])


def clean_ring(pts: np.ndarray, tol: float = 1e-9, collinear: bool = True) -> np.ndarray:
    """Drop the closing duplicate, consecutive near-duplicates and (optionally) straight vertices."""
    p = np.asarray(pts, dtype=np.float64)
    if len(p) > 1 and abs(p[0, 0] - p[-1, 0]) <= tol and abs(p[0, 1] - p[-1, 1]) <= tol:
        p = p[:-1]
    while len(p) >= 3:
        d = np.linalg.norm(_nxt(p) - p, axis=1)
        keep = d > tol
        if not keep.all():
            p = p[keep]
            continue
        if not collinear:
            break
        e_in = p - _prv(p)
        e_out = _nxt(p) - p
        cr = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
        ln = np.linalg.norm(e_in, axis=1) * np.linalg.norm(e_out, axis=1)
        # straight vertices and zero-width spikes (reversals) carry no area
        straight = np.abs(cr) <= _REL_TURN * ln
        if not straight.any():
            break
        # remove one at a time from each run (every other) to stay stable
        idx = np.flatnonzero(straight)
        drop = np.zeros(len(p), bool)
        drop[idx[0]] = True
        for i in idx[1:]:
            if not drop[i - 1]:
                drop[i] = True
        p = p[~drop]
    return p


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if signed_area(poly) > 0 else poly[::-1].copy()


def is_convex(poly: np.ndarray, rel_tol: float = 1e-9) -> bool:
    """Strictly simple convex CCW polygon: no right turns (beyond rel_tol) and total
    turning of exactly 2π (rejects self-overlapping rings that only turn left)."""
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return False
    e_in = p - _prv(p)
    e_out = _nxt(p) - p
    cr = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
    dt = (e_in * e_out).sum(1)
    ln = np.linalg.norm(e_in, axis=1) * np.linalg.norm(e_out, axis=1)
    if not (cr >= -rel_tol * ln).all() or signed_area(p) <= 0:
        return False
    ang = np.arctan2(cr, dt)
    if (ang > np.pi - 1e-6).any() or (ang < -1e-6).any():   # spikes / reversals
        return False
    return abs(ang.sum() - 2 * np.pi) < 1e-6


def poly_aspect(poly: np.ndarray) -> float:
    """Convex polygon aspect ratio: diameter / minimum width (rotating-calipers widths)."""
    p = np.asarray(poly, dtype=np.float64)
    d = p[:, None, :] - p[None, :, :]
    diam = np.sqrt((d ** 2).sum(-1)).max()
    e = _nxt(p) - p
    n = np.column_stack([e[:, 1], -e[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1), 1e-300)[:, None]
    proj = p @ n.T
    width = (proj.max(0) - proj.min(0)).min()
    return float(diam / max(width, 1e-300))


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew monotone chain; CCW, no collinear points."""
    p = np.unique(np.asarray(pts, dtype=np.float64), axis=0)  # sorted lexicographically
    if len(p) < 3:
        return p

    def half(seq):
        h: list = []
        for q in seq:
            while len(h) >= 2:
                a, b = h[-2], h[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    h.pop()
                else:
                    break
            h.append(q)
        return h

    lo = half(p)
    hi = half(p[::-1])
    return np.array(lo[:-1] + hi[:-1])


def _tri_ccw(tris: np.ndarray, V: np.ndarray) -> np.ndarray:
    a, b, c = V[tris[:, 0]], V[tris[:, 1]], V[tris[:, 2]]
    cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    t = tris.copy()
    neg = cr < 0
    t[neg] = t[neg][:, [0, 2, 1]]
    return t


def _hertel_mehlhorn(V: np.ndarray, tris: np.ndarray) -> list[list[int]]:
    polys: dict[int, list[int]] = {i: [int(a), int(b), int(c)] for i, (a, b, c) in enumerate(tris)}
    owner: dict[tuple[int, int], int] = {}
    for i, ring in polys.items():
        for k in range(3):
            owner[(ring[k], ring[(k + 1) % 3])] = i
    diags = [(u, v) for (u, v) in owner if u < v and (v, u) in owner]
    if not diags:
        return list(polys.values())
    dv = np.array(diags)
    ln = np.linalg.norm(V[dv[:, 0]] - V[dv[:, 1]], axis=1)
    order = np.lexsort((dv[:, 1], dv[:, 0], -ln))  # longest diagonal first, deterministic

    def turn_ok(a, b, c) -> bool:
        e1 = V[b] - V[a]
        e2 = V[c] - V[b]
        cr = e1[0] * e2[1] - e1[1] * e2[0]
        return cr >= -_REL_TURN * np.linalg.norm(e1) * np.linalg.norm(e2)

    for k in order:
        u, v = diags[k]
        P, Q = owner.get((u, v)), owner.get((v, u))
        if P is None or Q is None or P == Q:
            continue
        rp, rq = polys[P], polys[Q]
        iu = next(i for i in range(len(rp)) if rp[i] == u and rp[(i + 1) % len(rp)] == v)
        ivq = next(i for i in range(len(rq)) if rq[i] == v and rq[(i + 1) % len(rq)] == u)
        prot = rp[iu + 1:] + rp[:iu + 1]            # v, b, ..., a, u
        qrot = rq[ivq + 1:] + rq[:ivq + 1]          # u, d, ..., c, v
        a, b = prot[-2], prot[1]
        d, c = qrot[1], qrot[-2]
        if not (turn_ok(a, u, d) and turn_ok(c, v, b)):
            continue
        merged = prot + qrot[1:-1]
        del owner[(u, v)], owner[(v, u)]
        for i in range(len(rq)):
            e = (rq[i], rq[(i + 1) % len(rq)])
            if e in owner:
                owner[e] = P
        polys[P] = merged
        del polys[Q]
    return [polys[k] for k in sorted(polys)]


def _decompose_one(poly: Polygon, tol: float) -> list[np.ndarray]:
    poly = orient(poly, 1.0)
    rings = [clean_ring(np.asarray(poly.exterior.coords), tol)]
    rings += [clean_ring(np.asarray(r.coords), tol) for r in poly.interiors]
    rings = [r for r in rings if len(r) >= 3]
    if not rings:
        return []
    if len(rings) == 1 and is_convex(rings[0]):
        return [ensure_ccw(rings[0])]
    V = np.vstack(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    tris = np.asarray(mapbox_earcut.triangulate_float64(V, ends), dtype=np.int64).reshape(-1, 3)
    if len(tris) == 0:
        return []
    tris = _tri_ccw(tris, V)
    out = []
    for ring in _hertel_mehlhorn(V, tris):
        p = clean_ring(V[ring], tol)
        if len(p) >= 3 and signed_area(p) > tol * tol:
            out.append(ensure_ccw(p))
    return out


def convex_decompose(poly: Polygon, tol: float = 1e-9) -> list[np.ndarray]:
    """Polygon (holes allowed; MultiPolygon accepted) -> convex CCW polygons tiling it exactly.

    Earcut triangulation followed by Hertel–Mehlhorn diagonal removal. Straight
    (180°) vertices are removed from the output, so neighbouring parts may meet at
    T-junctions; use `polygon_adjacency` (geometric, tolerance based) for adjacency.
    """
    if isinstance(poly, MultiPolygon):
        return [p for g in poly.geoms for p in _decompose_one(g, tol)]
    if poly.is_empty:
        return []
    return _decompose_one(poly, tol)


# --------------------------------------------------------------------------- adjacency

def _edge_arrays(polys: list[np.ndarray]):
    n = np.array([len(p) for p in polys], dtype=np.int64)
    P0 = np.vstack(polys)
    starts = np.concatenate([[0], np.cumsum(n)[:-1]])
    nxt = np.arange(len(P0)) + 1
    nxt[starts + n - 1] = starts
    P1 = P0[nxt]
    owner = np.repeat(np.arange(len(polys)), n)
    return P0, P1, owner, starts, n


def bbox_pairs(lo: np.ndarray, hi: np.ndarray, pad: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """All index pairs i<j whose (D-dim) AABBs overlap after padding by `pad` (sweep and prune)."""
    lo = np.asarray(lo, np.float64) - pad
    hi = np.asarray(hi, np.float64) + pad
    N, D = lo.shape
    if N < 2:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    # sweep along the axis where boxes are most spread out relative to their size
    ax = int(np.argmax(np.ptp(0.5 * (lo + hi), axis=0) / np.maximum((hi - lo).mean(0), 1e-12)))
    order = np.argsort(lo[:, ax], kind="stable")
    slo, shi = lo[order], hi[order]
    # for each i (sorted), candidates are j>i with slo[j] <= shi[i]
    end = np.searchsorted(slo[:, ax], shi[:, ax], side="right")
    cnt = np.maximum(end - np.arange(N) - 1, 0)
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    I = np.repeat(np.arange(N), cnt)
    off = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    J = I + 1 + off
    ok = np.ones(tot, bool)
    for d in range(D):
        if d == ax:
            continue
        ok &= (slo[J, d] <= shi[I, d]) & (slo[I, d] <= shi[J, d])
    return _sorted_pairs(order[I[ok]], order[J[ok]])


def _sorted_pairs(a: np.ndarray, b: np.ndarray):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    o = np.lexsort((hi, lo))
    return lo[o], hi[o]


def polygon_adjacency(polys: list[np.ndarray], groups: np.ndarray | None = None,
                      tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shared-boundary adjacency between CCW polygons (T-junctions allowed).

    Two polygons are adjacent when an edge of one is collinear (within `tol`) and
    anti-parallel to an edge of the other with positive overlap. If `groups` is
    given, only polygons of the same group are paired (coordinates of different
    groups are treated as living in different planes).

    Returns (i, j, length, midpoint (K,2), normal (K,2)) with i<j; `midpoint` is the
    length-weighted midpoint of the shared boundary and `normal` the unit outward
    normal of polygon i across it.
    """
    empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2)))
    if len(polys) < 2:
        return empty
    P0, P1, owner, starts, n = _edge_arrays(polys)
    lo = np.array([p.min(0) for p in polys])
    hi = np.array([p.max(0) for p in polys])
    if groups is not None:
        g = np.asarray(groups)
        _, gi = np.unique(g, return_inverse=True)
        # shift groups apart along x for the broadphase only
        span = float((hi[:, 0] - lo[:, 0]).max() + (np.abs(lo).max() + np.abs(hi).max()) * 2 + 10.0)
        shift = gi * span
        lo = lo.copy(); hi = hi.copy()
        lo[:, 0] += shift; hi[:, 0] += shift
    A, B = bbox_pairs(lo, hi, pad=tol)
    if groups is not None and len(A):
        k = g[A] == g[B]
        A, B = A[k], B[k]
    if len(A) == 0:
        return empty
    # expand to edge pairs
    na, nb = n[A], n[B]
    cnt = na * nb
    pair = np.repeat(np.arange(len(A)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    ea = starts[A][pair] + off // nb[pair]
    eb = starts[B][pair] + off % nb[pair]
    p, q = P0[ea], P1[ea]
    r, s = P0[eb], P1[eb]
    d = q - p
    L = np.linalg.norm(d, axis=1)
    u = d / L[:, None]
    db = s - r
    Lb = np.linalg.norm(db, axis=1)
    ub = db / Lb[:, None]
    anti = (u * ub).sum(1) < -(1 - 1e-6)
    nrm = np.column_stack([u[:, 1], -u[:, 0]])
    dr = ((r - p) * nrm).sum(1)
    ds = ((s - p) * nrm).sum(1)
    col = (np.abs(dr) < tol) & (np.abs(ds) < tol)
    tr = ((r - p) * u).sum(1)
    ts = ((s - p) * u).sum(1)
    lo_t = np.maximum(0.0, np.minimum(tr, ts))
    hi_t = np.minimum(L, np.maximum(tr, ts))
    ov = hi_t - lo_t
    hit = anti & col & (ov > tol)
    if not hit.any():
        return empty
    pair, ov = pair[hit], ov[hit]
    mid = p[hit] + u[hit] * (0.5 * (lo_t[hit] + hi_t[hit]))[:, None]
    nr = nrm[hit]
    K = len(A)
    length = np.bincount(pair, ov, minlength=K)
    mx = np.bincount(pair, ov * mid[:, 0], minlength=K)
    my = np.bincount(pair, ov * mid[:, 1], minlength=K)
    nx = np.bincount(pair, ov * nr[:, 0], minlength=K)
    ny = np.bincount(pair, ov * nr[:, 1], minlength=K)
    keep = length > 0
    length = length[keep]
    midp = np.column_stack([mx[keep], my[keep]]) / length[:, None]
    nn = np.column_stack([nx[keep], ny[keep]])
    nn /= np.maximum(np.linalg.norm(nn, axis=1), 1e-300)[:, None]
    return A[keep], B[keep], length, midp, nn


def to_shapely(polys: list[np.ndarray]) -> np.ndarray:
    """Ragged list of (n_i, 2|3) rings -> object array of shapely Polygons (vectorised)."""
    if len(polys) == 0:
        return np.empty(0, dtype=object)
    n = np.array([len(p) for p in polys])
    rings = shapely.linearrings(np.vstack(polys), indices=np.repeat(np.arange(len(polys)), n))
    return shapely.polygons(rings)
