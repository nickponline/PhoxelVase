"""Build validation (DESIGN.md §2.7), geometry + graph part.

Implemented here: hull convexity, mesh watertightness / winding / positive volume, per-panel
volume conservation, chunk interpenetration (SAT), anchor connectivity, aspect-ratio
warnings. Static stability (rubble-py) and room connectivity are checked elsewhere and can
be appended to the report by their owners.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geom.convex2d import bbox_pairs, poly_aspect
from .graph import chunk_aabbs
from .materials import F_ANCHOR, F_COSMETIC_ATTACHED
from .model import ChunkGeom, Edge, Panel

CONVEX_TOL = 1e-5
VOLUME_RTOL = 1e-3
TOUCH_TOL = 1e-4
ASPECT_WARN = 20.0
_MAX_LIST = 10


@dataclass
class ValidationReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def merge(self, other: "ValidationReport") -> "ValidationReport":
        self.ok = self.ok and other.ok
        self.errors += other.errors
        self.warnings += other.warnings
        self.stats.update(other.stats)
        return self


def _fmt_ids(ids) -> str:
    ids = list(ids)
    s = ", ".join(str(i) for i in ids[:_MAX_LIST])
    return s + (f", ... (+{len(ids) - _MAX_LIST})" if len(ids) > _MAX_LIST else "")


# ----------------------------------------------------------------------------- checks

def check_convex(chunks: list[ChunkGeom], rep: ValidationReport) -> None:
    bad = []
    for i, c in enumerate(chunks):
        P = c.planes
        if not np.allclose(np.linalg.norm(P[:, :3], axis=1), 1.0, atol=1e-6):
            bad.append(i)
            continue
        if (c.hull_verts @ P[:, :3].T + P[:, 3]).max() > CONVEX_TOL:
            bad.append(i)
    if bad:
        rep.error(f"non-convex hull (vertex outside a face plane) in {len(bad)} chunks: {_fmt_ids(bad)}")


def mesh_check_fast(chunks: list[ChunkGeom]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised over all chunks: welds vertices (1e-7 m grid, per chunk), then requires
    every directed edge to occur exactly once and its reverse exactly once (closed, 2-manifold
    edges, consistent winding). Returns (watertight_ok (C,), winding_ok (C,), signed volume (C,))."""
    C = len(chunks)
    nv = np.array([len(c.mesh_verts) for c in chunks])
    nt = np.array([len(c.mesh_tris) for c in chunks])
    V = np.vstack([c.mesh_verts for c in chunks])
    owner_v = np.repeat(np.arange(C), nv)
    voff = np.concatenate([[0], np.cumsum(nv)[:-1]])
    T = np.vstack([c.mesh_tris.astype(np.int64) for c in chunks]) + np.repeat(voff, nt)[:, None]
    owner_t = np.repeat(np.arange(C), nt)
    # weld per chunk
    q = np.round(V / 1e-7).astype(np.int64)
    key = np.column_stack([owner_v, q])
    _, wid = np.unique(key, axis=0, return_inverse=True)
    wid = wid.reshape(-1)
    W = wid[T]
    nW = int(wid.max()) + 1
    u = W.ravel()
    v = W[:, [1, 2, 0]].ravel()
    eo = np.repeat(owner_t, 3)
    degenerate = u == v
    code = u * nW + v
    rcode = v * nW + u
    uc, cnt = np.unique(code, return_counts=True)
    dup = np.isin(code, uc[cnt > 1])
    pos = np.searchsorted(uc, rcode)
    pos = np.minimum(pos, len(uc) - 1)
    has_rev = uc[pos] == rcode
    wt_bad = np.bincount(eo[degenerate | ~has_rev], minlength=C) > 0
    wind_bad = np.bincount(eo[dup], minlength=C) > 0
    # signed volume (relative to the chunk's first vertex, for precision)
    base = V[voff][owner_t]
    a, b, c = V[T[:, 0]] - base, V[T[:, 1]] - base, V[T[:, 2]] - base
    tv = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    vol = np.bincount(owner_t, tv, minlength=C)
    return ~wt_bad, ~wind_bad, vol


def mesh_check_trimesh(chunks: list[ChunkGeom]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import trimesh
    C = len(chunks)
    wt, wd, vol = np.zeros(C, bool), np.zeros(C, bool), np.zeros(C)
    for i, c in enumerate(chunks):
        m = trimesh.Trimesh(c.mesh_verts, c.mesh_tris, process=False)
        m.merge_vertices()
        wt[i], wd[i], vol[i] = m.is_watertight, m.is_winding_consistent, m.volume
    return wt, wd, vol


def check_meshes(chunks: list[ChunkGeom], rep: ValidationReport, method: str = "fast") -> None:
    wt, wd, vol = (mesh_check_trimesh if method == "trimesh" else mesh_check_fast)(chunks)
    cv = np.array([c.volume for c in chunks])
    for name, bad in (("not watertight", ~wt), ("inconsistent winding", ~wd),
                      ("non-positive volume", (vol <= 0) | (cv <= 0)),
                      ("mesh volume != chunk volume",
                       np.abs(vol - cv) > 1e-4 * np.maximum(np.abs(cv), 1e-9))):
        ids = np.flatnonzero(bad)
        if len(ids):
            rep.error(f"chunk mesh {name}: {len(ids)} chunks: {_fmt_ids(ids)}")


def check_volume(panels: list[Panel], chunks: list[ChunkGeom], rep: ValidationReport) -> None:
    vol: dict[int, float] = {}
    for c in chunks:
        vol[c.panel_id] = vol.get(c.panel_id, 0.0) + c.volume
    for p in panels:
        want = p.profile.area * p.thickness
        got = vol.get(p.id, 0.0)
        if abs(got - want) > VOLUME_RTOL * max(want, 1e-12):
            rep.error(f"panel {p.id} ({p.kind}): chunk volume {got:.6g} != profile volume {want:.6g}")
    known = {p.id for p in panels}
    orphan = sorted({c.panel_id for c in chunks} - known)
    if orphan:
        rep.error(f"chunks reference unknown panels: {_fmt_ids(orphan)}")


def _padded(chunks: list[ChunkGeom]):
    """Padded hull verts, face normals and edge directions (padding repeats entry 0)."""
    C = len(chunks)
    Vm = max(len(c.hull_verts) for c in chunks)
    Fm = max(len(c.planes) for c in chunks)
    Em = max(len(c.poly2d) + 1 for c in chunks)
    HV = np.empty((C, Vm, 3))
    FN = np.empty((C, Fm, 3))
    ED = np.empty((C, Em, 3))
    for i, c in enumerate(chunks):
        hv = c.hull_verts
        n = len(c.poly2d)
        HV[i, : len(hv)] = hv
        HV[i, len(hv):] = hv[0]
        FN[i, : len(c.planes)] = c.planes[:, :3]
        FN[i, len(c.planes):] = c.planes[0, :3]
        top = hv[n:]
        e = np.vstack([np.roll(top, -1, axis=0) - top, hv[n:n + 1] - hv[:1]])
        e /= np.maximum(np.linalg.norm(e, axis=1), 1e-300)[:, None]
        ED[i, : len(e)] = e
        ED[i, len(e):] = e[0]
    return HV, FN, ED


def _sat_overlap(HV, a, b, axes, valid=None):
    """Per pair: True if some valid axis separates A and B (gap > -TOUCH_TOL)."""
    pa = np.einsum("kvd,kad->kva", HV[a], axes)
    pb = np.einsum("kvd,kad->kva", HV[b], axes)
    ov = np.minimum(pa.max(1) - pb.min(1), pb.max(1) - pa.min(1))   # (K, A) overlap length
    sep = ov <= TOUCH_TOL
    if valid is not None:
        sep &= valid
    return sep.any(1), ov


def check_interpenetration(chunks: list[ChunkGeom], rep: ValidationReport, batch: int = 2048) -> None:
    lo, hi = chunk_aabbs(chunks)
    A, B = bbox_pairs(lo, hi, pad=-TOUCH_TOL)
    rep.stats["sat_pairs"] = int(len(A))
    if len(A) == 0:
        return
    HV, FN, ED = _padded(chunks)
    bad = []
    for s in range(0, len(A), batch):
        a, b = A[s:s + batch], B[s:s + batch]
        sep, _ = _sat_overlap(HV, a, b, np.concatenate([FN[a], FN[b]], axis=1))
        rest = ~sep
        if not rest.any():
            continue
        a2, b2 = a[rest], b[rest]
        cr = np.cross(ED[a2][:, :, None, :], ED[b2][:, None, :, :]).reshape(len(a2), -1, 3)
        nn = np.linalg.norm(cr, axis=-1)
        valid = nn > 1e-6
        cr = cr / np.maximum(nn, 1e-300)[..., None]
        sep2, ov = _sat_overlap(HV, a2, b2, cr, valid)
        for k in np.flatnonzero(~sep2):
            bad.append((int(a2[k]), int(b2[k])))
    if bad:
        rep.error(f"{len(bad)} interpenetrating chunk pairs: {_fmt_ids(bad)}")


def check_connectivity(chunks: list[ChunkGeom], edges: list[Edge], rep: ValidationReport) -> None:
    C = len(chunks)
    anchor = np.array([bool(c.flags & F_ANCHOR) for c in chunks])
    cosmetic = np.array([bool(c.flags & F_COSMETIC_ATTACHED) for c in chunks])
    rep.stats["n_anchors"] = int(anchor.sum())
    if C == 0:
        return
    ea = np.array([e.a for e in edges], dtype=np.int64)
    eb = np.array([e.b for e in edges], dtype=np.int64)
    g = coo_matrix((np.ones(len(ea)), (ea, eb)), shape=(C, C))
    ncomp, lab = connected_components(g, directed=False)
    rep.stats["n_components"] = int(ncomp)
    grounded = np.zeros(ncomp, bool)
    grounded[lab[anchor]] = True
    floating = np.flatnonzero(~grounded[lab] & ~cosmetic)
    if not anchor.any():
        rep.error("no anchor chunks")
    if len(floating):
        rep.error(f"{len(floating)} non-cosmetic chunks not connected to an anchor: {_fmt_ids(floating)}")
    fc = np.flatnonzero(~grounded[lab] & cosmetic)
    if len(fc):
        rep.warn(f"{len(fc)} cosmetic chunks not connected to an anchor: {_fmt_ids(fc)}")


def chunk_aspect(c: ChunkGeom) -> float:
    """2D aspect ratio of the chunk polygon: diameter / minimum width."""
    return poly_aspect(c.poly2d)


def check_aspect(chunks: list[ChunkGeom], rep: ValidationReport) -> None:
    asp = np.array([chunk_aspect(c) for c in chunks]) if chunks else np.zeros(0)
    rep.stats["max_aspect"] = float(asp.max()) if len(asp) else 0.0
    ids = np.flatnonzero(asp > ASPECT_WARN)
    if len(ids):
        rep.warn(f"{len(ids)} chunks with aspect ratio > {ASPECT_WARN:g}: {_fmt_ids(ids)}")


def validate(panels: list[Panel], chunks: list[ChunkGeom], edges: list[Edge],
             mesh_check: str = "fast") -> ValidationReport:
    """Run the §2.7 geometry/graph checks. `mesh_check` = "fast" (vectorised weld + directed
    edge pairing, same criteria as trimesh) or "trimesh" (per-chunk trimesh, slower)."""
    rep = ValidationReport()
    rep.stats.update(n_panels=len(panels), n_chunks=len(chunks), n_edges=len(edges),
                     total_volume=float(sum(c.volume for c in chunks)),
                     total_mass=float(sum(c.mass for c in chunks)))
    if not chunks:
        rep.error("no chunks")
        return rep
    timings = {}
    for name, fn in (("convex", lambda: check_convex(chunks, rep)),
                     ("mesh", lambda: check_meshes(chunks, rep, mesh_check)),
                     ("volume", lambda: check_volume(panels, chunks, rep)),
                     ("interpenetration", lambda: check_interpenetration(chunks, rep)),
                     ("connectivity", lambda: check_connectivity(chunks, edges, rep)),
                     ("aspect", lambda: check_aspect(chunks, rep))):
        t = time.perf_counter()
        fn()
        timings[name] = round(time.perf_counter() - t, 4)
    rep.stats["timings"] = timings
    return rep
