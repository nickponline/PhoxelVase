"""High-level render entry points (DESIGN.md §2.8, §3.6 render-sim).

    render_all(bd, out_dir)                      -> every preview image of a building
    render_frames(bd, transforms, out_dir, ...)  -> destruction sequence frames
    make_gif(paths, out_path, fps)               -> animated gif

All functions take a `BuildingData` (as returned by `bgen.export.bld.read_bld` / `assemble`), so
rendering is decoupled from the generator. `backend` is 'auto' (GL, numpy fallback), 'gl' or 'numpy'.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from ..export.bld import bitset_to_mask
from . import plots
from .scene import (Camera, RenderJob, RenderMesh, Style, build_mesh, fit_camera, floor_levels,
                    named_camera, random_chunk_colors)

log = logging.getLogger(__name__)

ISO_VIEWS = ("iso_ne", "iso_nw", "iso_se", "iso_sw")


# --------------------------------------------------------------------------- backends

def make_renderer(mesh: RenderMesh, size=(1280, 960), backend: str = "auto", **kw):
    """Create a renderer bound to `mesh`. 'auto' tries GL then falls back to numpy."""
    if backend in ("auto", "gl"):
        try:
            from .gl import GLRenderer
            return GLRenderer(mesh, size, **kw)
        except Exception as e:  # no GL context available
            if backend == "gl":
                raise
            log.warning("GL renderer unavailable (%s); falling back to numpy rasterizer", e)
    if backend not in ("auto", "gl", "numpy"):
        raise ValueError(f"unknown backend {backend!r}")
    from .raster_np import NumpyRenderer
    return NumpyRenderer(mesh, size)


def _as_mesh(bd) -> RenderMesh:
    return bd if isinstance(bd, RenderMesh) else build_mesh(bd)


def _save(img: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, optimize=False, compress_level=3)
    return path


def _xf_bounds(mesh: RenderMesh, xforms: np.ndarray | None, visible: np.ndarray | None):
    """World AABB of (optionally transformed, optionally masked) chunks."""
    sel = np.ones(mesh.n_chunks, bool) if visible is None else visible.astype(bool)
    if not sel.any():
        return mesh.bounds
    lo, hi = mesh.aabb_min[sel], mesh.aabb_max[sel]
    if xforms is None:
        return lo.min(0), hi.max(0)
    X = xforms[sel]
    c, e = (lo + hi) / 2, (hi - lo) / 2
    wc = np.einsum("nij,nj->ni", X[:, :3, :3], c) + X[:, :3, 3]
    we = np.einsum("nij,nj->ni", np.abs(X[:, :3, :3]), e)
    return (wc - we).min(0), (wc + we).max(0)


def _translations(off: np.ndarray) -> np.ndarray:
    X = np.repeat(np.eye(4)[None], len(off), 0)
    X[:, :3, 3] = off
    return X


def exploded_offsets(mesh: RenderMesh, chunk_factor: float = 0.15, panel_factor: float = 0.30) -> np.ndarray:
    """Per-chunk offset: away from its panel centroid (×chunk_factor) and the panel away from the
    building center (×panel_factor), so both the fracture pattern and the interior are visible."""
    n = mesh.n_chunks
    if n == 0:
        return np.zeros((0, 3))
    w = np.maximum(mesh.mass, 1e-6)
    ne = int(mesh.elem.max()) + 1
    ws = np.bincount(mesh.elem, w, ne)
    pc = np.stack([np.bincount(mesh.elem, w * mesh.com[:, k], ne) for k in range(3)], 1) / np.maximum(ws, 1e-9)[:, None]
    lo, hi = mesh.bounds
    bc = (lo + hi) / 2
    pcc = pc[mesh.elem]
    return chunk_factor * (mesh.com - pcc) + panel_factor * (pcc - bc)


# --------------------------------------------------------------------------- single views

def render_view(bd, name: str = "iso_ne", size=(1280, 960), backend: str = "auto", renderer=None,
                colors=None, visible=None, xforms=None, style: Style | None = None,
                camera: Camera | None = None):
    """Render one named view; returns (rgb uint8 HxWx3, building mask HxW bool, camera)."""
    mesh = renderer.mesh if renderer is not None else _as_mesh(bd)
    own = renderer is None
    r = renderer or make_renderer(mesh, size, backend)
    try:
        lo, hi = _xf_bounds(mesh, xforms, visible)
        cam = camera or named_camera(name, lo, hi, r.size)
        job = RenderJob(cam, mesh.base_colors() if colors is None else colors, visible, xforms,
                        style or Style(), ground_z=min(0.0, float(mesh.bounds[0][2])),
                        ground_center=(mesh.bounds[0] + mesh.bounds[1])[:2] / 2,
                        ground_radius=_radius(mesh))
        img, mask = r.render(job, world_bounds=(lo, hi))
    finally:
        if own:
            r.release()
    return img, mask, cam


def _radius(mesh: RenderMesh) -> float:
    lo, hi = mesh.bounds
    return float(max(0.5 * np.linalg.norm((hi - lo)[:2]), 1.0))


# --------------------------------------------------------------------------- render_all

def render_all(bd, out_dir, size=(1280, 960), backend: str = "auto",
               edge_utilization: np.ndarray | None = None, timings: dict | None = None) -> list[Path]:
    """Render every image of DESIGN §2.8 into `out_dir`; returns the written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = _as_mesh(bd)
    meta = getattr(bd, "meta", {}) or {}
    paths: list[Path] = []
    tm = timings if timings is not None else {}
    r = make_renderer(mesh, size, backend)
    tm["backend"] = r.backend

    def view(fname, name, **kw):
        t = time.perf_counter()
        img, _, cam = render_view(None, name, renderer=r, **kw)
        paths.append(_save(img, out / fname))
        tm[fname] = time.perf_counter() - t
        return img, cam

    try:
        for v in ISO_VIEWS:
            view(f"{v}.png", v)
        view("front.png", "front")
        view("side.png", "side")
        view("top.png", "top")
        fr_style = Style(chunk_lines=0.35, inner_factor=0.8)
        view("fractured_iso.png", "iso_ne", colors=random_chunk_colors(mesh.n_chunks), style=fr_style)
        view("exploded.png", "iso_ne", xforms=_translations(exploded_offsets(mesh)),
             style=Style(chunk_lines=0.25))

        levels = floor_levels(mesh, meta)
        for i, (z0, z1) in enumerate(levels):
            vis = ~((mesh.elem_floor > i) | (mesh.com[:, 2] > z1 + 1e-3))
            if not vis.any():
                continue
            view(f"cutaway_floor_{i}.png", "oblique", visible=vis, style=Style(chunk_lines=0.0))
        t = time.perf_counter()
        for i, (z0, z1) in enumerate(levels):
            paths.append(plots.plan_floor(mesh, meta, i, z0, z1, out / f"plan_floor_{i}.png", size))
        tm["plans"] = time.perf_counter() - t

        # graph overlays share a faint iso render
        t = time.perf_counter()
        faint = np.full((mesh.n_chunks, 3), 0.92, np.float32)
        bg, _, cam = render_view(None, "iso_ne", renderer=r, colors=faint,
                                 style=Style(shadows=False, line_alpha=0.35))
        anchors = bitset_to_mask(bd.anchors, mesh.n_chunks) if hasattr(bd, "anchors") else np.zeros(mesh.n_chunks, bool)
        edges = getattr(bd, "edges", None)
        paths.append(plots.graph_overlay(bg, cam, mesh, edges, anchors, out / "graph.png"))
        if edge_utilization is not None:
            paths.append(plots.graph_overlay(bg, cam, mesh, edges, anchors, out / "stress.png",
                                             values=np.asarray(edge_utilization, float), mode="utilization"))
        tm["graph"] = time.perf_counter() - t
    finally:
        r.release()
    return paths


# --------------------------------------------------------------------------- sequences

def render_frames(bd, chunk_transforms, out_dir, camera: str | Camera = "iso_ne",
                  alive_mask: np.ndarray | None = None, size=(960, 720), backend: str = "auto",
                  colors: np.ndarray | None = None, prefix: str = "frame") -> list[Path]:
    """Render a destruction sequence.

    chunk_transforms: (F,N,4,4) array or an iterable of (N,4,4) building->world chunk transforms.
    alive_mask: optional (F,N) bool; dead chunks are hidden. The camera is fixed for the whole
    sequence: fitted to the building plus the debris spread when the transforms are an array, or to
    the building bounds with a margin for iterators.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mesh = _as_mesh(bd)
    r = make_renderer(mesh, size, backend)
    lo, hi = mesh.bounds
    if isinstance(camera, Camera):
        cam = camera
    else:
        flo, fhi = lo.copy(), hi.copy()
        if isinstance(chunk_transforms, np.ndarray) and chunk_transforms.ndim == 4 and len(chunk_transforms):
            T = chunk_transforms
            step = max(1, len(T) // 32)
            pts = np.einsum("fnij,nj->fni", T[::step, :, :3, :3], mesh.com) + T[::step, :, :3, 3]
            if alive_mask is not None:
                pts = pts[np.asarray(alive_mask[::step], bool)]
            pts = pts.reshape(-1, 3)
            if len(pts):
                q0, q1 = np.percentile(pts, 1, axis=0), np.percentile(pts, 99, axis=0)
                flo, fhi = np.minimum(flo, q0), np.maximum(fhi, q1)
        else:
            c, e = (lo + hi) / 2, (hi - lo) / 2
            e[:2] *= 1.25
            flo, fhi = c - e, c + e
        flo[2] = min(flo[2], 0.0)
        cam = named_camera(camera, flo, fhi, r.size)
    base_colors = mesh.base_colors() if colors is None else colors
    gc, gr = (lo + hi)[:2] / 2, _radius(mesh) * 1.4
    paths = []
    try:
        for f, X in enumerate(chunk_transforms):
            X = np.asarray(X, np.float32)
            vis = None if alive_mask is None else np.asarray(alive_mask[f], bool)
            job = RenderJob(cam, base_colors, vis, X, Style(), ground_z=0.0, ground_center=gc, ground_radius=gr)
            img, _ = r.render(job, world_bounds=_xf_bounds(mesh, X, vis))
            paths.append(_save(img, out / f"{prefix}_{f:05d}.png"))
    finally:
        r.release()
    return paths


def make_gif(paths: Iterable[Path], out_path, fps: int = 30, max_width: int | None = 640) -> Path:
    """Assemble frames into a looping gif (frames downscaled to `max_width`)."""
    rgb = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if max_width and im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        rgb.append(im)
    frames = []
    if rgb:
        # one shared palette from a strip of sampled frames: fast and flicker-free
        idx = np.linspace(0, len(rgb) - 1, min(len(rgb), 6)).round().astype(int)
        w, h = rgb[0].size
        strip = Image.new("RGB", (w, h * len(idx)))
        for k, i in enumerate(idx):
            strip.paste(rgb[i], (0, h * k))
        strip = strip.resize((w // 2, h * len(idx) // 2))
        pal = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        frames = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in rgb]
    if not frames:
        raise ValueError("no frames")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=max(1, round(1000 / fps)),
                   loop=0, optimize=False, disposal=1)
    return out


__all__ = ["render_all", "render_frames", "make_gif", "render_view", "make_renderer", "exploded_offsets"]
