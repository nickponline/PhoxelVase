"""Per-panel pre-fracture: decompose -> seed -> Voronoi clip -> merge -> extrude.

cfg keys (all optional; defaults derived from the materials table):
  cell_size       target Voronoi cell size in m (default: material.cell_size)
  edge_bias       extra seed density near opening (hole) edges, as a multiple of the base
                  density in a band of width cell_size/2 around the holes (default 0)
  border_bias     same for the outer panel border (default 0.25 * edge_bias)
  anisotropy      [ax, ay]: cells are stretched ax/ay times along local x (wood grain)
  min_area        sliver threshold (default 0.02 * cell_size²)
  min_chunk_area  pieces below this are merged into a neighbour when the union is
                  convex (default 0.25 * cell_size²)
  merge_cells     merge pieces of the same Voronoi cell split by convex-part
                  diagonals when their union is convex (default True)
"""
from __future__ import annotations

import heapq
import math

import numpy as np
import shapely
import shapely.affinity
import shapely.geometry
from shapely.geometry.polygon import orient

from ..materials import (DEFAULT_MATERIALS, FACE_CAP_EDGE, FACE_INNER, F_COSMETIC_ATTACHED,
                         F_GLASS, F_INDESTRUCTIBLE, Material)
from ..model import ChunkGeom, Panel
from .convex2d import (clean_ring, convex_decompose, convex_hull, ensure_ccw, is_convex,
                       polygon_adjacency, poly_aspect, signed_area, snap, to_shapely, _nxt, _prv)
from .prism import make_prism
from .voronoi2d import bounded_voronoi


def panel_flags(panel: Panel) -> int:
    t = panel.tags or {}
    f = 0
    if panel.material == "glass":
        f |= F_GLASS
    if t.get("cosmetic") or t.get("cosmetic_attached"):
        f |= F_COSMETIC_ATTACHED
    if t.get("indestructible"):
        f |= F_INDESTRUCTIBLE
    return f


# ----------------------------------------------------------------------------- seeds

def _sample_in(geom, n: int, rng: np.random.Generator) -> np.ndarray:
    """n uniform points inside `geom` by rejection sampling on its bounds."""
    if n <= 0 or geom.is_empty or geom.area <= 0:
        return np.zeros((0, 2))
    minx, miny, maxx, maxy = geom.bounds
    shapely.prepare(geom)
    out = []
    got = 0
    frac = max(geom.area / max((maxx - minx) * (maxy - miny), 1e-12), 0.01)
    for _ in range(50):
        m = int((n - got) / frac * 1.3) + 8
        x = rng.uniform(minx, maxx, m)
        y = rng.uniform(miny, maxy, m)
        k = shapely.contains_xy(geom, x, y)
        pts = np.column_stack([x[k], y[k]])[: n - got]
        out.append(pts)
        got += len(pts)
        if got >= n:
            break
    return np.vstack(out) if out else np.zeros((0, 2))


def _corner_seeds(geom, cell: float, offset: float) -> np.ndarray:
    """One seed per reflex profile vertex (opening corners, notches), placed `offset`·cell
    into the material along the corner's bisector. A cell then owns each corner instead
    of a neighbouring cell wrapping around it by a hair, which is what produces slivers."""
    out = []
    rings = [np.asarray(geom.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in geom.interiors]
    shapely.prepare(geom)
    for ring in rings:
        if len(ring) < 3:
            continue
        prv, nxt = np.roll(ring, 1, 0), np.roll(ring, -1, 0)
        e1, e2 = ring - prv, nxt - ring
        cr = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        # shapely orient(): exterior CCW, holes CW -> material on the left of travel for both;
        # a right turn (cr < 0) is a reflex corner of the material
        refl = cr < -1e-9 * np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1)
        if not refl.any():
            continue
        u1 = e1 / np.linalg.norm(e1, axis=1)[:, None]
        u2 = e2 / np.linalg.norm(e2, axis=1)[:, None]
        bis = u1 - u2                      # points into the material, away from the corner
        bis /= np.maximum(np.linalg.norm(bis, axis=1), 1e-12)[:, None]
        pts = ring[refl] + bis[refl] * offset * cell * np.sqrt(2.0)
        out.append(pts)
    if not out:
        return np.zeros((0, 2))
    P = np.vstack(out)
    P = P[shapely.contains_xy(geom, P[:, 0], P[:, 1])]
    # merge corner seeds closer than half a cell (narrow mullions between openings)
    keep = []
    for p in P:
        if all(np.hypot(*(p - q)) > 0.5 * cell for q in keep):
            keep.append(p)
    return np.array(keep).reshape(-1, 2)


def _cover_seeds(geom, S: np.ndarray, cell: float, rng: np.random.Generator) -> np.ndarray:
    """Extra seeds where the grid leaves material far from every seed (narrow strips such as
    lintels between ribbon windows and the slab can get no grid seed, and cells reaching in
    from far away turn them into long slats). Probe points along each convex part's
    diameter every ~cell/2; a probe farther than 0.6·cell from all seeds becomes a seed."""
    from scipy.spatial import cKDTree
    probes = []
    for part in convex_decompose(geom):
        d = part[:, None, :] - part[None, :, :]
        dd = (d ** 2).sum(-1)
        i, j = np.unravel_index(np.argmax(dd), dd.shape)
        m = int(2 * np.sqrt(dd[i, j]) / cell)
        if m < 1:
            continue
        c = part.mean(0)
        t = (np.arange(m) + 0.5) / m
        pts = part[i] + t[:, None] * (part[j] - part[i])
        probes.append(pts + 0.4 * (c - pts))
    if not probes:
        return np.zeros((0, 2))
    P = np.vstack(probes)
    out = []
    tree = cKDTree(S) if len(S) else None
    for p in P:   # sequential so new seeds also cover their neighbours
        far = tree is None or tree.query(p)[0] > 0.6 * cell
        if far and all(np.hypot(*(p - q)) > 0.9 * cell for q in out):
            out.append(p + rng.uniform(-0.05, 0.05, 2) * cell)
    out = np.array(out).reshape(-1, 2)
    return out[shapely.contains_xy(geom, out[:, 0], out[:, 1])] if len(out) else out


def _jittered_grid(geom, cell: float, rng: np.random.Generator) -> np.ndarray:
    """One uniform point per cell×cell stratum of geom's bounds, kept if inside geom."""
    minx, miny, maxx, maxy = geom.bounds
    nx = max(1, int(round((maxx - minx) / cell)))
    ny = max(1, int(round((maxy - miny) / cell)))
    sx, sy = (maxx - minx) / nx, (maxy - miny) / ny
    gx, gy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    gx, gy = gx.ravel(), gy.ravel()
    x = minx + (gx + rng.uniform(0.05, 0.95, gx.size)) * sx
    y = miny + (gy + rng.uniform(0.05, 0.95, gy.size)) * sy
    shapely.prepare(geom)
    k = shapely.contains_xy(geom, x, y)
    return np.column_stack([x[k], y[k]])


# ----------------------------------------------------------------------------- pieces

_ASPECT_FIX = 8.0       # pieces thinner than this (diameter / width; × anisotropy) get fixed up
_MIN_EDGE = 1e-6        # m; shortest edge kept in output pieces
_RESCUE_MAX_VERTS = 14  # skip rescue re-cuts of unions with more vertices (curved openings)


def _clip_cells(profile, cells: list[np.ndarray]):
    """Intersect every Voronoi cell with the profile (vectorised shapely); cells that come
    out non-convex (they contain an opening corner) are convex-decomposed locally."""
    cell_ids = np.array([i for i, c in enumerate(cells) if len(c) >= 3])
    cg = to_shapely([cells[i] for i in cell_ids])
    shapely.prepare(profile)
    inside = shapely.contains(profile, cg)
    inter = cg.copy()
    hit = ~inside
    inter[hit] = shapely.intersection(cg[hit], profile)
    pieces, cell_of = [], []
    for k, g in enumerate(inter):
        for sub in shapely.get_parts(g):
            if shapely.get_type_id(sub) != 3 or sub.area <= 1e-14:
                continue
            ring = clean_ring(snap(np.asarray(sub.exterior.coords)), tol=1e-10)
            if len(ring) < 3:
                continue
            ring = ensure_ccw(ring)
            if len(sub.interiors) == 0 and is_convex(ring, 1e-7):
                subs = [ring]
            else:
                # the cell wraps an opening corner
                subs = [ensure_ccw(clean_ring(snap(q), tol=1e-10)) for q in convex_decompose(sub)]
                subs = [q for q in subs if len(q) >= 3]
            for r in subs:
                pieces.append(r)
                cell_of.append(int(cell_ids[k]))
    return pieces, np.array(cell_of)


def _split_by_chord(R: np.ndarray, d: np.ndarray, eps: float) -> list[np.ndarray] | None:
    """Split simple CCW ring R (R[0] = cut vertex) by the line through R[0] along d, if the
    line meets the boundary in exactly one other place (so the cut is a single chord).
    Vertices R[1] / R[-1] lying on the line (edge extensions) are allowed."""
    n = len(R)
    sd = d[0] * (R[:, 1] - R[0, 1]) - d[1] * (R[:, 0] - R[0, 0])
    sd[np.abs(sd) < eps] = 0.0
    sd[0] = 0.0
    contacts = []   # (kind, index): ("v", j) vertex j on line, ("e", i) crossing on edge i->i+1
    for j in range(2, n - 1):
        if sd[j] == 0.0:
            contacts.append(("v", j))
    for i in range(1, n - 1):
        if sd[i] * sd[i + 1] < 0:
            contacts.append(("e", i))
    if len(contacts) != 1:
        return None
    kind, j = contacts[0]
    if kind == "v":
        A = R[: j + 1]
        B = np.vstack([R[j:], R[:1]])
        sa, sb = sd[1:j], sd[j + 1:]
    else:
        t = sd[j] / (sd[j] - sd[j + 1])
        q = R[j] + t * (R[j + 1] - R[j])
        A = np.vstack([R[: j + 1], q])
        B = np.vstack([q, R[j + 1:], R[:1]])
        sa, sb = sd[1:j + 1], sd[j + 1:]
    if not (((sa >= 0).all() and (sb <= 0).all()) or ((sa <= 0).all() and (sb >= 0).all())):
        return None
    out = []
    for P in (A, B):
        P = clean_ring(snap(P), tol=1e-10)
        if len(P) < 3 or signed_area(P) <= 0:
            return None
        out.append(P)
    return out


def _line_cuts(ring: np.ndarray, k: int) -> list[list[np.ndarray]]:
    """All 2-piece splits of CCW `ring` by a single straight chord from vertex k. Candidate
    directions: extensions of k's two edges, directions to the other vertices, the axes."""
    n = len(ring)
    r = ring[k]
    e_in = r - ring[(k - 1) % n]
    e_out = ring[(k + 1) % n] - r
    others = [j for j in range(n) if j not in (k, (k - 1) % n, (k + 1) % n)]
    D = np.vstack([e_in, -e_out, ring[others] - r, [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]])
    dn = np.linalg.norm(D, axis=1)
    D = D[dn > 1e-12] / dn[dn > 1e-12, None]
    R = np.concatenate([ring[k:], ring[:k]])
    eps = 1e-12 * float((ring.max(0) - ring.min(0)).sum() + 1.0)
    out = []
    for d in D:
        res = _split_by_chord(R, d, eps)
        if res is not None:
            out.append(res)
    return out


def _reflex_vertices(ring: np.ndarray) -> np.ndarray:
    e_in = ring - _prv(ring)
    e_out = _nxt(ring) - ring
    cr = e_in[:, 0] * e_out[:, 1] - e_in[:, 1] * e_out[:, 0]
    ln = np.linalg.norm(e_in, axis=1) * np.linalg.norm(e_out, axis=1)
    return np.flatnonzero(cr < -1e-7 * ln)


def _convex_split(ring: np.ndarray, min_area: float, max_pieces: int = 3) -> list[np.ndarray] | None:
    """Split a simple polygon into <= max_pieces convex pieces, all >= min_area, by straight
    cuts through reflex vertices (exhaustive over candidate cuts, recursive). Returns the
    split with the smallest worst aspect ratio, or None."""
    refl = _reflex_vertices(ring)
    if len(refl) == 0:
        return [ring] if signed_area(ring) >= min_area and is_convex(ring, 1e-7) else None
    if len(refl) > max_pieces - 1:
        return None
    best, best_score = None, np.inf
    deferred = []   # cuts that leave a non-convex half: recurse only into the most promising
    for k in refl:
        for a, b in _line_cuts(ring, int(k)):
            if signed_area(a) < min_area or signed_area(b) < min_area:
                continue
            ra, rb = len(_reflex_vertices(a)), len(_reflex_vertices(b))
            if 2 + ra + rb > max_pieces:
                continue
            if ra == 0 and rb == 0:
                if is_convex(a, 1e-7) and is_convex(b, 1e-7):
                    score = max(poly_aspect(a), poly_aspect(b))
                    if score < best_score - 1e-12:
                        best, best_score = [a, b], score
                continue
            conv = a if ra == 0 else b
            deferred.append((poly_aspect(conv), len(deferred), a, b))
    deferred.sort(key=lambda t: (t[0], t[1]))
    for pre, _, a, b in deferred[:3]:
        if pre >= best_score:
            break
        sa = _convex_split(a, min_area, max_pieces - 1)
        if sa is None:
            continue
        sb = _convex_split(b, min_area, max_pieces - len(sa))
        if sb is None:
            continue
        res = sa + sb
        score = max(poly_aspect(q) for q in res)
        if score < best_score - 1e-12:
            best, best_score = res, score
    return best


def _merge_pieces(pieces: list[np.ndarray], cell_of: np.ndarray, min_chunk_area: float,
                  merge_cells: bool, scale: float, min_area: float = 0.0, rescue: bool = True,
                  aspect_fix: float = 8.0) -> tuple[list[np.ndarray], dict]:
    """Greedy convex merging driven by the shared-boundary adjacency graph."""
    n = len(pieces)
    polys = {i: p for i, p in enumerate(pieces)}
    area = {i: signed_area(p) for i, p in polys.items()}
    I, J, L, _, _ = polygon_adjacency(pieces, tol=1e-7 * scale)
    nbr: dict[int, dict[int, float]] = {i: {} for i in range(n)}
    for i, j, l in zip(I.tolist(), J.tolist(), L.tolist()):
        nbr[i][j] = nbr[i].get(j, 0.0) + l
        nbr[j][i] = nbr[j].get(i, 0.0) + l
    stats = {"merges": 0, "resplits": 0, "unmerged_small": 0}
    budget = [2 * n + 10]   # cap on rescue attempts per panel (each costs a few GEOS overlays)
    next_id = [n]

    def try_merge(i: int, j: int) -> bool:
        """Merge i into j if their union is convex. Returns success."""
        h = convex_hull(np.vstack([polys[i], polys[j]]))
        if len(h) < 3:
            return False
        a_sum = area[i] + area[j]
        if signed_area(h) > a_sum * (1 + 1e-9) + 1e-14:
            return False
        polys[j] = clean_ring(h, tol=1e-10)
        area[j] = a_sum
        for k, l in nbr.pop(i).items():
            nbr[k].pop(i, None)
            if k == j:
                continue
            nbr[j][k] = nbr[j].get(k, 0.0) + l
            nbr[k][j] = nbr[k].get(j, 0.0) + l
        del polys[i], area[i]
        stats["merges"] += 1
        return True

    def try_resplit(group: list[int]) -> bool:
        """Sliver rescue: re-cut the union of `group` (sliver first + 1 or 2 neighbours)
        into <= len(group)+1 convex pieces with a better worst aspect ratio and a larger
        smallest piece; they replace the group."""
        if budget[0] <= 0:
            return False
        budget[0] -= 1
        try:
            u = shapely.union_all([shapely.Polygon(polys[g]) for g in group])
        except shapely.errors.GEOSException:
            return False
        a_sum = sum(area[g] for g in group)
        # GEOS overlay can mis-handle near-coincident vertices: verify the union's area
        if shapely.get_type_id(u) != 3 or len(u.interiors) or abs(u.area - a_sum) > 1e-9 * a_sum:
            return False
        ring = ensure_ccw(clean_ring(np.asarray(u.exterior.coords), tol=1e-10))
        if len(ring) < 3 or len(ring) > _RESCUE_MAX_VERTS or not shapely.Polygon(ring).is_valid:
            return False
        res = _convex_split(ring, max(min_area, 1.5 * min(area[g] for g in group)), len(group) + 1)
        if res is None or abs(sum(signed_area(q) for q in res) - a_sum) > 1e-9 * a_sum:
            return False
        if max(poly_aspect(q) for q in res) >= max(poly_aspect(polys[g]) for g in group):
            return False
        cand = sorted(set().union(*(nbr[g] for g in group)) - set(group))
        for g in group:
            for o in nbr.pop(g):
                if o in nbr:
                    nbr[o].pop(g, None)
        new_ids = list(group[:len(res)])
        while len(new_ids) < len(res):
            new_ids.append(next_id[0])
            next_id[0] += 1
        for g in group[len(res):]:
            del polys[g], area[g]
        for g, q in zip(new_ids, res):
            polys[g], area[g], nbr[g] = q, signed_area(q), {}
        ids = new_ids + cand
        A, B, LL, _, _ = polygon_adjacency([polys[k] for k in ids], tol=1e-7 * scale)
        for a, b, l in zip(A.tolist(), B.tolist(), LL.tolist()):
            ia, ib = ids[a], ids[b]
            if ia in new_ids or ib in new_ids:
                nbr[ia][ib] = l
                nbr[ib][ia] = l
        stats["resplits"] += 1
        return True

    def rescue_piece(i: int) -> bool:
        cands = [j for j, _ in sorted(nbr[i].items(), key=lambda kv: (-kv[1], kv[0]))]
        for j in cands:
            if try_resplit([i, j]):
                return True
        for x in range(len(cands)):
            for y in range(x + 1, len(cands)):
                if try_resplit([i, cands[x], cands[y]]):
                    return True
        return False

    if merge_cells and len(I):
        cell = {i: int(c) for i, c in enumerate(cell_of)}
        order = np.lexsort((J, I, -L))
        for k in order:
            i, j = int(I[k]), int(J[k])
            if i in polys and j in polys and cell[i] == cell[j]:
                try_merge(i, j)

    heap = [(a, i) for i, a in area.items() if a < min_chunk_area]
    heapq.heapify(heap)
    while heap:
        a, i = heapq.heappop(heap)
        if i not in polys or area[i] != a:
            continue
        cands = sorted(nbr[i].items(), key=lambda kv: (-kv[1], kv[0]))
        for j, _ in cands:
            # don't grow thin strips (e.g. a lintel above a ribbon window) into long slats
            if a >= min_area:
                h = convex_hull(np.vstack([polys[i], polys[j]]))
                lim = max(aspect_fix, poly_aspect(polys[i]), poly_aspect(polys[j]))
                if len(h) < 3 or poly_aspect(h) > lim:
                    continue
            if try_merge(i, j):
                if area[j] < min_chunk_area:
                    heapq.heappush(heap, (area[j], j))
                break

    # phase 3: slivers and thin pieces left over -> convex merge with any neighbour that
    # improves the aspect ratio, else local re-cut (rescue)
    def bad(i):
        return area[i] < min_area or poly_aspect(polys[i]) > aspect_fix

    for _ in range(3):
        todo = sorted((i for i in polys if bad(i)), key=lambda i: (-poly_aspect(polys[i]), i))
        changed = False
        for i in todo:
            if i not in polys or not bad(i):
                continue
            asp = poly_aspect(polys[i])
            done = False
            for j, _l in sorted(nbr[i].items(), key=lambda kv: (-kv[1], kv[0])):
                h = convex_hull(np.vstack([polys[i], polys[j]]))
                if len(h) >= 3 and (area[i] < min_area or poly_aspect(h) < asp) and try_merge(i, j):
                    done = True
                    break
            if not done and rescue:
                done = rescue_piece(i)
            changed |= done
        if not changed:
            break
    stats["unmerged_small"] = int(sum(a < min_chunk_area for a in area.values()))
    return [polys[k] for k in sorted(polys)], stats


def _side_flags(pieces: list[np.ndarray], profile, tol: float) -> list[np.ndarray]:
    """FACE_CAP_EDGE for edges lying on the profile border (outer ring or holes), else INNER."""
    if not pieces:
        return []
    P0 = np.vstack(pieces)
    P1 = np.vstack([_nxt(p) for p in pieces])
    bnd = profile.boundary
    shapely.prepare(bnd)
    on = np.ones(len(P0), bool)
    for pts in (P0, P1, 0.5 * (P0 + P1), 0.75 * P0 + 0.25 * P1):
        on &= shapely.dwithin(bnd, shapely.points(pts), tol)
    fl = np.where(on, FACE_CAP_EDGE, FACE_INNER).astype(np.uint16)
    return np.split(fl, np.cumsum([len(p) for p in pieces])[:-1])


def _dedupe(pieces: list[np.ndarray]) -> list[np.ndarray]:
    return [q2 for q2 in (clean_ring(q, tol=_MIN_EDGE) for q in pieces) if len(q2) >= 3]


def _tiling_ok(pieces: list[np.ndarray], area: float) -> bool:
    return all(is_convex(q, 1e-7) for q in pieces) and \
        abs(sum(signed_area(q) for q in pieces) - area) <= 1e-9 * max(area, 1e-12)


def fracture_pieces(panel: Panel, cfg: dict | None, rng: np.random.Generator,
                    mat: Material | None = None) -> tuple[list[np.ndarray], dict]:
    """2D part of fracture_panel: convex CCW pieces (panel-local XY) tiling the profile."""
    cfg = cfg or {}
    mat = mat or DEFAULT_MATERIALS[panel.material]
    cell = float(cfg.get("cell_size", mat.cell_size))
    cell_area = cell * cell
    min_area = float(cfg.get("min_area", 0.02 * cell_area))
    min_chunk_area = float(cfg.get("min_chunk_area", 0.25 * cell_area))
    min_chunk_area = max(min_chunk_area, min_area)
    edge_bias = float(cfg.get("edge_bias", 0.0))
    border_bias = float(cfg.get("border_bias", 0.25 * edge_bias))
    ax, ay = (float(v) for v in cfg.get("anisotropy", (1.0, 1.0)))
    profile = orient(panel.profile, 1.0)
    parts = convex_decompose(profile)
    info = {"parts": len(parts), "seeds": 0, "merges": 0, "unmerged_small": 0}
    flags = panel_flags(panel)
    if (flags & (F_GLASS | F_INDESTRUCTIBLE)) or profile.area < cell_area or not parts:
        return _dedupe(parts), info

    # work in an area-preserving scaled space where cells are isotropic
    sx = math.sqrt(ax / ay)
    sy = 1.0 / sx
    sp = shapely.affinity.scale(profile, 1 / sx, 1 / sy, origin=(0, 0))
    grid = _jittered_grid(sp, cell, rng)
    corner = _corner_seeds(sp, cell, float(cfg.get("corner_offset", 0.3))) \
        if cfg.get("corner_seeds", True) else np.zeros((0, 2))
    if len(corner) and len(grid):
        # drop grid seeds crowding a corner seed
        d = np.linalg.norm(grid[:, None, :] - corner[None, :, :], axis=-1).min(1)
        grid = grid[d > 0.5 * cell]
    seeds = [corner, grid]
    seeds.append(_cover_seeds(sp, np.vstack([corner, grid]), cell, rng))
    if edge_bias > 0 or border_bias > 0:
        band_w = 0.5 * cell
        for rings, bias in ((list(sp.interiors), edge_bias), ([sp.exterior], border_bias)):
            if bias <= 0 or not rings:
                continue
            lines = shapely.geometry.MultiLineString([np.asarray(r.coords) for r in rings])
            band = shapely.intersection(sp, shapely.buffer(lines, band_w))
            seeds.append(_sample_in(band, int(round(bias * band.area / cell_area)), rng))
    S = np.vstack(seeds)
    if len(S) < 2:
        return _dedupe(parts), info
    info["seeds"] = len(S)
    minx, miny, maxx, maxy = sp.bounds
    m = 1e-3 * max(maxx - minx, maxy - miny) + 1e-3
    cells = bounded_voronoi(S, (minx - m, miny - m, maxx + m, maxy + m))
    cells = [snap(c * np.array([sx, sy])) if len(c) else c for c in cells]
    pieces, cell_of = _clip_cells(profile, cells)
    scale = max(profile.bounds[2] - profile.bounds[0], profile.bounds[3] - profile.bounds[1], 1.0)
    merge_cells = bool(cfg.get("merge_cells", True))
    raw = pieces
    afix = _ASPECT_FIX * max(ax / ay, ay / ax)
    pieces, st = _merge_pieces(raw, cell_of, min_chunk_area, merge_cells, scale, min_area, aspect_fix=afix)
    if not _tiling_ok(pieces, profile.area):
        # safety net: the sliver rescue relies on GEOS overlays; never trade correctness
        pieces, st = _merge_pieces(raw, cell_of, min_chunk_area, merge_cells, scale, min_area, rescue=False,
                                   aspect_fix=afix)
        st["fallback"] = 1
        if not _tiling_ok(pieces, profile.area):
            pieces, st = raw, {"fallback": 2}
    info.update(st)
    # drop near-duplicate vertices (independent overlays can leave edges ~1e-8 m long, which
    # collapse when meshes are welded). Removing vertices of a convex polygon keeps it convex.
    pieces = _dedupe(pieces)
    info["slivers"] = int(sum(signed_area(p) < min_area for p in pieces))
    return pieces, info


def fracture_panel(panel: Panel, cfg: dict | None, rng: np.random.Generator) -> list[ChunkGeom]:
    """Pre-fracture one panel into convex prism chunks (building space). See module doc."""
    mat = DEFAULT_MATERIALS[panel.material]
    pieces, _ = fracture_pieces(panel, cfg, rng, mat)
    profile = orient(panel.profile, 1.0)
    scale = max(profile.bounds[2] - profile.bounds[0], profile.bounds[3] - profile.bounds[1], 1.0)
    sides = _side_flags(pieces, profile, 1e-7 * scale)
    flags = panel_flags(panel)
    out = []
    for poly, sf in zip(pieces, sides):
        vol = signed_area(poly) * panel.thickness
        hp = max(mat.min_hp, mat.hp_per_m3 * vol)
        out.append(make_prism(poly, panel.thickness, panel.frame, mat.density, panel.id,
                              panel.material, sf, flags=flags, hp=hp))
    return out
