"""Foundation: a concrete cuboid under the whole building that blasts can break into.

Spec (top level, next to `indestructible`; `foundation: null` keeps the old thin ground slab):

    foundation: {depth: 1.2, layers: 3, pad: 2.0, material: concrete, cell_size: 1.6}

Under a basement (see `basement.py`) it has the basement's outline (no `pad`) and spans
z = floor - depth up to the basement floor. Otherwise it spans the footprint of everything
standing on z = 0, grown by `pad`, from z = -depth up to 0, and replaces the `ground_slab`
panels. Either way it is cut into `layers` horizontal
slabs so a blast eats into it from the top instead of punching full-depth holes. The upper
layers are ordinary fractured concrete; the bottom one is anchored and indestructible
(bedrock), so ground-floor walls rest on breakable concrete rather than being anchors
themselves. Under a basement the upper layers are anchored too (still breakable): they bear
straight down on the bedrock. The top layer is a `floor` (level 0 for floor-based logic); the rest are `other`.
"""
from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import box

from .model import BuildingContext
from .nodes._util import add_panel, horiz_frame, parts, remove_panels, snapf

DEFAULTS = {"depth": 1.2, "layers": 3, "pad": 2.0, "material": "concrete", "cell_size": 1.6}
EPS = 1e-3


def _panel_verts(p) -> np.ndarray:
    ring = np.asarray(p.profile.exterior.coords)[:, :2]
    lo = p.to_world(np.column_stack([ring, np.zeros(len(ring))]))
    hi = p.to_world(np.column_stack([ring, np.full(len(ring), p.thickness)]))
    return np.vstack([lo, hi])


def _rects(geom) -> list[list[float]]:
    """Axis-aligned rectangles [x0, y0, x1, y1] tiling a (rectilinear) footprint: the cells of the
    grid on its vertex coordinates whose centres are inside, merged into runs along x."""
    rings = [np.asarray(r.coords) for p in parts(geom) for r in (p.exterior, *p.interiors)]
    xs, ys = (sorted({round(float(v), 4) for r in rings for v in r[:, k]}) for k in (0, 1))
    shapely.prepare(geom)
    out = []
    for j in range(len(ys) - 1):
        ym, i = 0.5 * (ys[j] + ys[j + 1]), 0
        while i < len(xs) - 1:
            if not shapely.contains_xy(geom, 0.5 * (xs[i] + xs[i + 1]), ym):
                i += 1
                continue
            s = i
            while i < len(xs) - 1 and shapely.contains_xy(geom, 0.5 * (xs[i] + xs[i + 1]), ym):
                i += 1
            out.append([xs[s], ys[j], xs[i], ys[j + 1]])
    return out


def add_foundation(ctx: BuildingContext, cfg, basement=None) -> float | None:
    """Add the layered foundation, under `basement` ((footprint, floor z) from `add_basement`)
    or replacing the ground slabs. Returns its bottom z (the new anchor level), or None when
    disabled or there is nothing standing on the ground."""
    if cfg is None or cfg is False:
        return None
    cfg = {**DEFAULTS, **(cfg if isinstance(cfg, dict) else {})}
    depth, n = float(cfg["depth"]), max(int(cfg["layers"]), 1)
    if depth <= 0:
        return None
    t = snapf(depth / n)
    frac = {"cell_size": float(cfg["cell_size"])} if cfg.get("cell_size") else {}

    def layer_tags(k):
        tags = {"floor": -1 if basement else 0, "role": "foundation", "layer": k}
        if k == 0:
            tags.update(anchor=True, indestructible=True)
        else:
            if frac:
                tags["fracture"] = frac
            if basement is not None:
                # resting on the bedrock, these bear straight down (no load routed sideways
                # through slivers); still breakable, and what stands on them still falls
                tags["anchor"] = True
        return tags

    if basement is not None:
        foot, top = basement
        for k in reversed(range(n)):
            for part in parts(foot):
                add_panel(ctx, "floor" if k == n - 1 else "other", horiz_frame(top - t * (n - k)),
                          part, t, cfg["material"], layer_tags(k))
        bottom = snapf(top - t * n)
        # ground level stays at 0: the basement and its foundation sit in a pit of this outline
        ctx.meta["foundation"] = {"rects": _rects(foot), "top": 0.0, "bottom": bottom}
        return bottom
    slabs = [p for p in ctx.panels if p.tags.get("role") == "ground_slab"]
    slab_ids = {p.id for p in slabs}
    # footprint of what stands on the ground; the slabs only when nothing else does
    # (a sculpt's ground slab is already padded)
    standing = [v for p in ctx.panels if p.id not in slab_ids for v in [_panel_verts(p)] if v[:, 2].min() <= EPS]
    pts = np.vstack(standing or [_panel_verts(p) for p in slabs] or [np.zeros((0, 3))])
    if not len(pts):
        return None
    pad = float(cfg["pad"])
    x0, y0 = np.floor((pts[:, :2].min(0) - pad) * 10) / 10
    x1, y1 = np.ceil((pts[:, :2].max(0) + pad) * 10) / 10
    # the layers take over the ground slabs' panel ids (top first) so every other panel keeps
    # its id, and with it its fracture pattern (the fracture RNG is seeded per panel id)
    reuse = sorted(slab_ids)
    for k in reversed(range(n)):
        p = add_panel(ctx, "floor" if k == n - 1 else "other", horiz_frame(-t * (n - k)),
                      box(x0, y0, x1, y1), t, cfg["material"], layer_tags(k))
        if p is not None and reuse:
            ctx.panels.pop()
            p.id = reuse.pop(0)
            ctx.panels[p.id] = p
    if reuse:
        remove_panels(ctx, reuse)
    # its extent, so a scene can lay ground level with its top around it (the pit it sits in)
    ctx.meta["foundation"] = {"rects": [[float(x0), float(y0), float(x1), float(y1)]],
                              "top": 0.0, "bottom": snapf(-t * n)}
    return snapf(-t * n)
