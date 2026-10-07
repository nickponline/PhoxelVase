"""`manifest.json`: spec echo, seed, version, stats, materials, rooms, sockets, timings,
validation report (DESIGN §2.8)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..materials import SOCKET_TYPES, materials_json
from .bld import BLD_VERSION


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def stats(res) -> dict:
    ctx, chunks, edges = res.ctx, res.chunks, res.edges
    mats = Counter(c.material for c in chunks)
    mass = Counter()
    for c in chunks:
        mass[c.material] += float(c.mass)
    return {
        "floors": len(ctx.blockout.floor_polys),
        "panels": len(ctx.panels),
        "panels_by_kind": dict(sorted(Counter(p.kind for p in ctx.panels).items())),
        "chunks": len(chunks),
        "chunks_by_material": dict(sorted(mats.items())),
        "mass_by_material": {k: round(v, 1) for k, v in sorted(mass.items())},
        "edges": len(edges),
        "anchors": int(sum(1 for c in chunks if c.flags & 1)),
        "mass": round(float(sum(c.mass for c in chunks)), 1),
        "volume": round(float(sum(c.volume for c in chunks)), 3),
        "rooms": len(ctx.rooms),
        "sockets": dict(Counter(s.type for s in ctx.sockets)),
        "bounds": res.bd.meta.get("bounds"),
    }


def manifest(res) -> dict:
    ctx = res.ctx
    spec = {k: v for k, v in ctx.spec.items() if not k.startswith("_")}
    return {
        "name": spec.get("name"), "seed": spec.get("seed"), "generator": spec.get("generator"),
        "bld_version": BLD_VERSION,
        "spec_dir": ctx.meta.get("spec_dir"),
        "spec": spec,
        "params": ctx.params,
        "stats": stats(res),
        "materials": materials_json(),
        "rooms": [{"id": r.id, "floor": r.floor, "tag": r.tag, "area": round(r.polygon.area, 2),
                   "polygon": [[round(x, 4), round(y, 4)] for x, y in r.polygon.exterior.coords]}
                  for r in ctx.rooms],
        "sockets": [{"type": s.type, "owner_panel": s.owner_panel,
                     "xform": np.round(np.asarray(s.xform), 5).tolist()} for s in ctx.sockets],
        "room_connectivity": ctx.meta.get("room_connectivity"),
        "roofs": ctx.meta.get("roofs", []),
        "manual": ctx.meta.get("manual"),
        "timings": res.timings,
        "validation": res.report,
    }


def write_manifest(res, path) -> Path:
    Path(path).write_text(json.dumps(manifest(res), indent=1, default=_jd))
    return Path(path)
