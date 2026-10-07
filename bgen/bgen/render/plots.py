"""Matplotlib plots: floor plans and connection-graph / stress overlays."""
from __future__ import annotations

import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shapely  # noqa: E402
import shapely.ops  # noqa: E402
from matplotlib.collections import LineCollection, PatchCollection  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon  # noqa: E402

from .scene import Camera, RenderMesh, glass_mask  # noqa: E402

TAG_COLORS = {
    "corridor": "#e8e3d3", "hall": "#e8e3d3", "lobby": "#f1d7a8", "office": "#b9d3ee",
    "living_room": "#f4c7a1", "bedroom": "#c9b8e6", "kitchen": "#f6e39b", "bathroom": "#a8dfe0",
    "meeting": "#c4e3b2", "storage": "#d9d9d9", "stair": "#cfcfcf", "retail": "#f2b8c6",
    "dining": "#f8d0a0", "warehouse": "#d8cbb8", "utility": "#d0d0c0",
}
_FALLBACK = plt.get_cmap("Pastel1").colors + plt.get_cmap("Pastel2").colors


def tag_color(tag: str):
    if tag in TAG_COLORS:
        return TAG_COLORS[tag]
    return _FALLBACK[zlib.crc32(tag.encode()) % len(_FALLBACK)]


def _poly_patch(geom, **kw) -> list[PathPatch]:
    """shapely (Multi)Polygon -> PathPatches (holes supported)."""
    out = []
    geoms = geom.geoms if isinstance(geom, MultiPolygon) else [geom]
    for g in geoms:
        if g.is_empty or not isinstance(g, Polygon):
            continue
        verts, codes = [], []
        for ring in [g.exterior, *g.interiors]:
            c = np.asarray(ring.coords)[:, :2]
            verts += c.tolist()
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(c) - 2) + [MPath.CLOSEPOLY]
        out.append(PathPatch(MPath(verts, codes), **kw))
    return out


def section_polygons(mesh: RenderMesh, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal section of every chunk at height z: (chunk ids, shapely polygons).

    Chunks are convex, so each section is the convex hull of the triangle/plane crossing points."""
    sel = (mesh.aabb_min[:, 2] < z) & (mesh.aabb_max[:, 2] > z)
    if not sel.any():
        return np.zeros(0, int), np.zeros(0, object)
    tmask = sel[mesh.tchunk]
    T = mesh.tris[tmask]
    tc = mesh.tchunk[tmask]
    P = mesh.pos[T].astype(np.float64)              # (t,3,3)
    d = P[:, :, 2] - z
    pts, ids = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        da, db = d[:, a], d[:, b]
        m = (da * db) < 0
        t = da[m] / (da[m] - db[m])
        pts.append(P[m, a, :2] + t[:, None] * (P[m, b, :2] - P[m, a, :2]))
        ids.append(tc[m])
    pts, ids = np.concatenate(pts), np.concatenate(ids)
    if not len(pts):
        return np.zeros(0, int), np.zeros(0, object)
    order = np.argsort(ids, kind="stable")
    pts, ids = pts[order], ids[order]
    uniq, inv = np.unique(ids, return_inverse=True)
    mp = shapely.multipoints(pts, indices=inv)
    hulls = shapely.convex_hull(mp)
    ok = shapely.area(hulls) > 1e-5
    return uniq[ok], hulls[ok]


def plan_floor(mesh: RenderMesh, meta: dict, floor: int, z0: float, z1: float, path: Path,
               size=(1280, 960), dpi: int = 100) -> Path:
    """Floor plan: slab footprint, rooms colored by tag with labels, chunk sections at mid-height
    (walls dark, columns black, glass light blue), stairs hatched."""
    zc = z0 + 0.5 * (z1 - z0)
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    ax = fig.add_axes([0.05, 0.06, 0.72, 0.86])
    ax.set_aspect("equal")
    ax.set_facecolor("#fbfbfa")

    # slab footprint (floor slab chunks of this floor)
    slab = (mesh.elem_kind == "floor") & (mesh.elem_floor == floor)
    if slab.any():
        polys = [shapely.box(*mesh.aabb_min[i, :2], *mesh.aabb_max[i, :2]) for i in np.nonzero(slab)[0]]
        fp = shapely.union_all(polys).buffer(0)
        for p in _poly_patch(fp, facecolor="#ecebe7", edgecolor="#9a9a96", lw=0.8, zorder=0):
            ax.add_patch(p)

    ids, hulls = section_polygons(mesh, zc)
    obstacles = shapely.union_all(hulls).buffer(0.6) if len(hulls) else None

    # rooms
    handles = {}
    for room in meta.get("rooms", []) or []:
        if int(room.get("floor", -1)) != floor:
            continue
        poly = Polygon(room["polygon"])
        if poly.is_empty:
            continue
        col = tag_color(room["tag"])
        for p in _poly_patch(poly, facecolor=col, edgecolor="none", alpha=0.85, zorder=1):
            ax.add_patch(p)
        free = poly.difference(obstacles) if obstacles is not None else poly
        if free.is_empty or free.area < 0.2 * poly.area:
            free = poly
        if isinstance(free, MultiPolygon):
            free = max(free.geoms, key=lambda g: g.area)
        rp = shapely.ops.polylabel(free, tolerance=0.05) if isinstance(free, Polygon) else poly.representative_point()
        ax.text(rp.x, rp.y, f"{room['tag'].replace('_', ' ')}\n{poly.area:.0f} m²", ha="center", va="center",
                fontsize=8, color="#333333", zorder=6)
        handles.setdefault(room["tag"], col)

    # sections
    if len(ids):
        glass = glass_mask(mesh)[ids]
        kind = mesh.elem_kind[ids]
        col = np.where(glass, "#9fd3f5", np.where(kind == "column", "#111111",
                       np.where(np.isin(kind, ["stair", "step", "landing"]), "#8c8c8c", "#3b3d40")))
        for h, c, g in zip(hulls, col, glass):
            for p in _poly_patch(h, facecolor=c, edgecolor=c if not g else "#5aa9da", lw=0.3,
                                 zorder=4 if g else 3):
                ax.add_patch(p)
    # stairs (footprints, hatched)
    st = np.isin(mesh.elem_kind, ["stair", "step", "landing"]) & (mesh.elem_floor == floor)
    for i in np.nonzero(st)[0]:
        b = shapely.box(*mesh.aabb_min[i, :2], *mesh.aabb_max[i, :2])
        for p in _poly_patch(b, facecolor="none", edgecolor="#777777", hatch="///", lw=0.4, zorder=2):
            ax.add_patch(p)

    lo, hi = mesh.bounds
    pad = 0.06 * max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.tick_params(labelsize=7, colors="#777777")
    for s in ax.spines.values():
        s.set_color("#cccccc")
    ax.set_xlabel("x (m)", fontsize=8, color="#777777")
    ax.set_ylabel("y (m)", fontsize=8, color="#777777")
    name = meta.get("name", "building")
    ax.set_title(f"{name} · floor {floor} · cut at z = {zc:.2f} m", fontsize=11, loc="left", color="#222222")

    # legend
    from matplotlib.patches import Patch
    items = [Patch(facecolor=c, label=t.replace("_", " ")) for t, c in handles.items()]
    items += [Patch(facecolor="#3b3d40", label="wall (section)"), Patch(facecolor="#111111", label="column"),
              Patch(facecolor="#9fd3f5", edgecolor="#5aa9da", label="glass")]
    if st.any():
        items.append(Patch(facecolor="none", edgecolor="#777777", hatch="///", label="stairs"))
    fig.legend(handles=items, loc="center left", bbox_to_anchor=(0.79, 0.5), frameon=False, fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)
    return path


def graph_overlay(bg: np.ndarray, cam: Camera, mesh: RenderMesh, edges, anchors: np.ndarray, path: Path,
                  values: np.ndarray | None = None, mode: str = "strength", dpi: int = 100) -> Path:
    """Connection graph projected over a faint render: edges com(a)->com(b) colored by strength
    (log scale) or by utilization (0..1, red >= 1); anchors highlighted."""
    H, W = bg.shape[:2]
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow((bg.astype(np.float32) * 0.55 + 255 * 0.45).astype(np.uint8), interpolation="nearest")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    px, depth = cam.project(mesh.com, (W, H)) if mesh.n_chunks else (np.zeros((0, 2)), np.zeros(0))
    n_e = 0 if edges is None else len(edges)
    if n_e:
        a, b = edges["a"].astype(np.int64), edges["b"].astype(np.int64)
        seg = np.stack([px[a], px[b]], 1)
        dz = (depth[a] + depth[b]) / 2
        order = np.argsort(-dz)  # far first
        lw = float(np.clip(2.6 - 0.45 * np.log10(n_e), 0.35, 2.2))
        if mode == "utilization":
            v = np.asarray(values, float)
            norm, cmap, label = Normalize(0.0, 1.0), plt.get_cmap("RdYlGn_r"), "utilization"
            # stressed edges drawn last and thicker
            order = np.lexsort((-dz, v))
            widths = lw * (0.6 + 1.6 * np.clip(v, 0, 1))[order]
        else:
            v = np.asarray(edges["strength"], float)
            pos = v[v > 0]
            vmin = pos.min() if len(pos) else 1.0
            vmax = max(pos.max() if len(pos) else 10.0, vmin * 1.0001)
            v = np.maximum(v, vmin)
            norm, cmap, label = LogNorm(vmin, vmax), plt.get_cmap("viridis"), "bond strength (N, log)"
            widths = np.full(n_e, lw)
        lc = LineCollection(seg[order], colors=cmap(norm(v[order])), linewidths=widths, alpha=0.9,
                            capstyle="round")
        ax.add_collection(lc)
        cax = fig.add_axes([0.80, 0.06, 0.16, 0.022])
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    if anchors is not None and anchors.any():
        s = float(np.clip(300.0 / np.sqrt(max(anchors.sum(), 1)), 3, 40))
        ax.scatter(px[anchors, 0], px[anchors, 1], s=s, marker="s", c="#e4572e", edgecolors="#7a1f0a",
                   linewidths=0.5, zorder=5, label="anchor")
    title = "static utilization" if mode == "utilization" else "connection graph"
    extra = f" · max {np.max(values):.2f}" if mode == "utilization" and values is not None and len(values) else ""
    ax.text(16, 24, f"{title} · {mesh.n_chunks} chunks · {n_e} edges · {int(anchors.sum()) if anchors is not None else 0} anchors{extra}",
            fontsize=11, color="#222222", va="top")
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path
