"""Basement + foundation: a storey below ground level on a layered foundation, anchored only
under the building."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bgen.materials import F_ANCHOR, F_INDESTRUCTIBLE
from bgen.pipeline import generate
from bgen.spec import load_spec

SPECS = Path(__file__).resolve().parents[1] / "specs"


def _zmin(c) -> float:
    return float(c.hull_verts[:, 2].min())


def _layers(res):
    return sorted((p for p in res.ctx.panels if p.tags.get("role") == "foundation"), key=lambda p: p.tags["layer"])


def _area(rects) -> float:
    return float(sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rects))


def test_foundation_layers_and_anchors_without_basement():
    spec = load_spec(SPECS / "house.yaml")
    spec["basement"] = None
    spec["foundation"] = {"depth": 1.2, "layers": 3, "pad": 2.0}
    res = generate(spec, 1, jobs=1)
    assert res.ctx.meta["ground_z"] == -1.2
    assert [p.id for p in res.ctx.panels] == list(range(len(res.ctx.panels)))
    assert not [p for p in res.ctx.panels if p.tags.get("role") == "ground_slab"]
    layers = _layers(res)
    assert [p.frame[2, 3] for p in layers] == [-1.2, -0.8, -0.4]
    assert [p.kind for p in layers] == ["other", "other", "floor"]
    by_panel = {p.id: [c for c in res.chunks if c.panel_id == p.id] for p in layers}
    # bedrock: one indestructible anchor; the layers above are fractured and breakable
    bed = by_panel[layers[0].id]
    assert len(bed) == 1 and bed[0].flags & F_ANCHOR and bed[0].flags & F_INDESTRUCTIBLE
    for p in layers[1:]:
        assert len(by_panel[p.id]) > 1
        assert not any(c.flags & (F_ANCHOR | F_INDESTRUCTIBLE) for c in by_panel[p.id])
    assert [c for c in res.chunks if c.flags & F_ANCHOR] == bed
    # the cuboid is padded around everything standing on it
    on_ground = np.vstack([c.hull_verts for c in res.chunks if abs(_zmin(c)) < 1e-3])
    x0, y0, x1, y1 = layers[-1].profile.bounds
    assert x0 <= on_ground[:, 0].min() - 2.0 + 1e-6 and x1 >= on_ground[:, 0].max() + 2.0 - 1e-6
    assert y0 <= on_ground[:, 1].min() - 2.0 + 1e-6 and y1 >= on_ground[:, 1].max() + 2.0 - 1e-6
    assert res.bd.meta["ground_z"] == -1.2
    assert res.bd.meta["foundation"] == {"rects": [[x0, y0, x1, y1]], "top": 0.0, "bottom": -1.2}


def test_basement_under_lowest_floor():
    spec = load_spec(SPECS / "house.yaml")
    res = generate(spec, 1, jobs=1)
    panels, fz = res.ctx.panels, res.ctx.blockout.floor_z
    h = fz[1] - fz[0]
    assert res.ctx.meta["basement"]["height"] == h
    # the ground slab is now an ordinary breakable floor over the basement
    slab = [p for p in panels if p.kind == "floor" and p.tags.get("floor") == 0 and p.tags.get("role") != "foundation"]
    assert slab and all(p.tags["role"] == "floor_slab" and "anchor" not in p.tags for p in slab)
    assert not any(c.flags & (F_ANCHOR | F_INDESTRUCTIBLE) for c in res.chunks if c.panel_id in {p.id for p in slab})
    # basement: solid walls round the slab's outline, the ground floor's interior walls copied down
    base = [p for p in panels if p.tags.get("floor") == -1 and p.tags.get("role") != "foundation"]
    walls = [p for p in base if p.tags.get("role") == "basement_wall"]
    assert walls and all(not p.profile.interiors for p in walls)
    ground_int = [p for p in panels if p.kind == "int_wall" and p.tags.get("floor") == 0]
    assert len([p for p in base if p.kind == "int_wall"]) == len(ground_int) > 0
    zs = np.concatenate([c.hull_verts[:, 2] for c in res.chunks if c.panel_id in {p.id for p in base}])
    assert zs.min() >= -h - 1e-6 and zs.max() <= slab[0].frame[2, 3] + 1e-6
    # foundation under the basement, its outline only, all anchored (bedrock indestructible)
    layers = _layers(res)
    assert np.isclose(layers[-1].frame[2, 3] + layers[-1].thickness, -h)
    top = [p for p in layers if p.tags["layer"] == layers[-1].tags["layer"]]
    assert np.isclose(sum(p.profile.area for p in top), sum(p.profile.area for p in slab))
    found = [c for c in res.chunks if c.panel_id in {p.id for p in layers}]
    assert all(c.flags & F_ANCHOR for c in found)
    assert {bool(c.flags & F_INDESTRUCTIBLE) for c in found} == {True, False}
    # nothing else is an anchor: the house stands on the basement
    assert all(c.panel_id in {p.id for p in layers} for c in res.chunks if c.flags & F_ANCHOR)
    f = res.bd.meta["foundation"]
    assert f["top"] == 0.0 and np.isclose(f["bottom"], res.ctx.meta["ground_z"])
    assert np.isclose(_area(f["rects"]), sum(p.profile.area for p in slab))


def test_colossus_basement_is_empty():
    res = generate(load_spec(SPECS / "colossus.yaml"), 1, jobs=1)
    base = [p for p in res.ctx.panels if p.tags.get("floor") == -1 and p.tags.get("role") != "foundation"]
    assert base and {p.tags["role"] for p in base} <= {"basement_wall", "corner_post"}
    f = res.bd.meta["foundation"]
    assert len(f["rects"]) == 1  # the square ground pad


def test_foundation_and_basement_off_keep_ground_slab():
    spec = load_spec(SPECS / "house.yaml")
    spec["foundation"] = None
    spec["basement"] = None
    res = generate(spec, 1, jobs=1)
    assert res.ctx.meta["ground_z"] == 0.0
    assert [p for p in res.ctx.panels if p.tags.get("role") == "ground_slab"]
    assert not [p for p in res.ctx.panels if p.tags.get("role") == "foundation"]
    assert res.bd.meta["foundation"] is None
