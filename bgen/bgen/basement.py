"""Basement: one storey below ground level under the lowest floor, so blasts on the ground floor
break through its slab into a space below the street.

Spec (top level, next to `foundation`; `basement: null` for none):

    basement: {height: null, wall_thickness: null, material: concrete, copy_ground_floor: true}

The basement has the outline of the ground slabs (the lowest floor; a sculpt's square ground
pad), closed by solid exterior walls from z = -height up to the underside of the ground slab,
which stops being an anchor and becomes an ordinary breakable floor slab. `height` defaults to
the ground floor's (3 m without floors). With `copy_ground_floor` the ground floor's interior
walls (with their doorways) and columns are copied straight down, so loads keep their path
to the ground; a sculpt has none, so its basement is empty. No stairs lead down. The
foundation goes under it (see `foundation.py`).
"""
from __future__ import annotations

import copy

import numpy as np
import shapely

from .model import BuildingContext
from .nodes._util import emit_chain, parts, snapf

DEFAULTS = {"height": None, "wall_thickness": None, "material": "concrete", "copy_ground_floor": True}
EPS = 1e-3


def _verts(p) -> np.ndarray:
    ring = np.asarray(p.profile.exterior.coords)[:, :2]
    return np.vstack([p.to_world(np.column_stack([ring, np.full(len(ring), h)])) for h in (0.0, p.thickness)])


def add_basement(ctx: BuildingContext, cfg) -> tuple[shapely.Geometry, float] | None:
    """Build the basement storey. Returns (footprint, floor z) for the foundation under it, or
    None when disabled or the building has no ground slab."""
    if cfg is None or cfg is False:
        return None
    cfg = {**DEFAULTS, **(cfg if isinstance(cfg, dict) else {})}
    slabs = [p for p in ctx.panels if p.tags.get("role") == "ground_slab"]
    fz = ctx.blockout.floor_z if ctx.blockout is not None else []
    floors = len(fz) > 1 and any(p.kind == "ext_wall" for p in ctx.panels)
    h = float(cfg["height"] or (fz[1] - fz[0] if floors else 3.0))
    if not slabs or h <= 0:
        return None
    z0 = snapf(-h)
    z1 = snapf(min(p.frame[2, 3] for p in slabs))  # underside of the ground slab
    t = float(cfg["wall_thickness"] or ctx.params.get("ext_wall_thickness", 0.3))
    mat = cfg["material"]
    # ground slabs are horizontal panels: their profiles are in world xy
    foot = shapely.union_all([p.profile for p in slabs])
    for p in slabs:
        p.tags.pop("anchor", None)
        p.tags.pop("indestructible", None)
        p.tags["role"] = "floor_slab"
    if cfg["copy_ground_floor"] and floors:
        # interior walls and columns standing on the ground slab, within the ground floor and
        # over the basement (not, say, a canopy post outside the walls)
        for p in list(ctx.panels):
            if p.kind not in ("int_wall", "column") or p.tags.get("cosmetic") or p.tags.get("role") == "corner_post":
                continue
            v = _verts(p)
            if abs(v[:, 2].min()) > EPS or v[:, 2].max() > fz[1] + EPS \
                    or not foot.contains(shapely.Point(*v[:, :2].mean(0))):
                continue
            F = p.frame.copy()
            F[2, 3] = snapf(F[2, 3] - h)
            q = ctx.add_panel(kind=p.kind, frame=F, profile=p.profile, thickness=p.thickness,
                              material=p.material, tags={**copy.deepcopy(p.tags), "floor": -1})
            q.tags.pop("host", None)
    for part in parts(foot):
        emit_chain(ctx, np.asarray(part.exterior.coords)[:-1], True, t, z0, z1, "ext_wall", mat,
                   {"floor": -1, "role": "basement_wall"})
    ctx.meta["basement"] = {"height": h, "floor_z": z0}
    return foot, z0
