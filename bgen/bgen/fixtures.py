"""Hand-built golden fixtures for the .bld contract (DESIGN.md §6 M0)."""
from __future__ import annotations

import numpy as np
from shapely.geometry import box

from .export.bld import assemble
from .geom.prism import make_prism
from .materials import DEFAULT_MATERIALS, FACE_OUTER, F_ANCHOR, bond_strength
from .model import BuildingContext, Edge, Socket


def two_box() -> "BuildingData":
    """Two 2x1x1 concrete slabs stacked; each split into two 1x1x1 chunks.

    chunks: 0=(0..1,0..1,0..1) anchor, 1=(1..2,..) anchor, 2/3 = same at z 1..2
    edges: 0-1, 2-3 (intra panel), 0-2, 1-3 (inter panel), area 1 m² each
    """
    ctx = BuildingContext(spec={"name": "two_box", "seed": 0}, params={"floor_height": 1.0})
    dens = DEFAULT_MATERIALS["concrete"].density
    chunks = []
    for pz in (0.0, 1.0):
        fr = np.eye(4); fr[2, 3] = pz
        p = ctx.add_panel(kind="floor", frame=fr, profile=box(0, 0, 2, 1), thickness=1.0,
                          material="concrete", tags={"floor": int(pz), "anchor": pz == 0.0})
        for x0 in (0.0, 1.0):
            poly = np.array([[x0, 0], [x0 + 1, 0], [x0 + 1, 1], [x0, 1]], float)
            side = np.full(4, FACE_OUTER, np.uint16)
            side[1 if x0 == 0 else 3] = 2  # inner fracture face
            chunks.append(make_prism(poly, 1.0, fr, dens, p.id, "concrete", side,
                                     flags=F_ANCHOR if pz == 0.0 else 0, hp=2000.0))
    s = bond_strength("concrete", "concrete")
    E = lambda a, b, c, n: Edge(a, b, 1.0, s * 1.0, np.array(c, float), np.array(n, float))
    edges = [E(0, 1, (1, .5, .5), (1, 0, 0)), E(2, 3, (1, .5, 1.5), (1, 0, 0)),
             E(0, 2, (.5, .5, 1), (0, 0, 1)), E(1, 3, (1.5, .5, 1), (0, 0, 1))]
    xf = np.eye(4); xf[:3, 3] = (1, 0, 2)
    ctx.sockets.append(Socket("prop", xf, 1))
    return assemble(ctx, chunks, edges)
