"""`floors` node: one slab per floor part spanning the full floor footprint (junction rule 1)."""
from __future__ import annotations

from ..model import BuildingContext
from ._util import add_panel, horiz_frame, parts


def node(ctx: BuildingContext, params: dict, rng) -> BuildingContext:
    bo = ctx.blockout
    ts = float(ctx.params["slab_thickness"])
    indestr = set(ctx.spec.get("indestructible", []))
    mat_up = params.get("material", ctx.params.get("slab_material", "concrete"))
    mat_ground = params.get("ground_material", "concrete")
    for k, fp in enumerate(bo.floor_polys):
        for pi, part in enumerate(parts(fp)):
            role = "ground_slab" if k == 0 else "floor_slab"
            tags = {"floor": k, "role": role, "part": pi}
            if k == 0:
                tags["anchor"] = True
            if role in indestr or "floor" in indestr:
                tags["indestructible"] = True
            add_panel(ctx, "floor", horiz_frame(bo.floor_z[k] - ts), part, ts,
                      mat_ground if k == 0 else mat_up, tags)
    return ctx
