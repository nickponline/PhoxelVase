"""`sculpt` node: CSG primitives sliced into layers; structure templates end to end."""
from __future__ import annotations

from pathlib import Path

import pytest

from bgen.nodes.sculpt import Prim, expand_mirrors, slice_layers
from bgen.pipeline import generate
from bgen.spec import load_spec

from test_nodes_overlap import describe, find_overlaps
from test_pipeline import panel_level_floating

SPECS = Path(__file__).resolve().parents[1] / "specs"
STRUCTURES = ["aqueduct", "colossus"]


def test_mirror_expansion():
    p = {"box": {"min": [1, 2, 0], "max": [3, 5, 1]}, "mirror": ["x", "y"], "about": [0, 0]}
    out = expand_mirrors([p])
    assert len(out) == 4
    mins = sorted(tuple(q["box"]["min"][:2]) for q in out)
    assert mins == [(-3, -5), (-3, 2), (1, -5), (1, 2)]
    t = expand_mirrors([{"tube": {"path": [[1, 1, 0], [2, 3, 4]], "radius": 0.3}, "mirror": ["y"], "about": [0, 1]}])
    assert t[1]["tube"]["path"] == [[1, 1, 0], [2, -1, 4]]


def test_slice_classes_and_sub():
    prims = [Prim(p, 0.0, 0.5) for p in [
        {"box": {"min": [0, 0, 0], "max": [4, 4, 2]}},
        {"box": {"min": [1, 1, 1], "max": [3, 3, 2]}, "material": "metal"},
        {"box": {"min": [0, 0, 0], "max": [1, 1, 2]}, "op": "sub"}]]
    layers = slice_layers(prims, 0.5)
    assert [k for k, _, _ in layers] == [0, 1, 2, 3]
    _, _, top = layers[-1]
    assert top[("metal", "other")].area == pytest.approx(4.0)
    assert top[("concrete", "other")].area == pytest.approx(16 - 4 - 1)


@pytest.mark.parametrize("name", STRUCTURES)
def test_structure_end_to_end(name):
    res = generate(load_spec(SPECS / f"{name}.yaml"), 1, stability=False)
    ctx = res.ctx
    assert ctx.meta["sculpt"]["template"] == name and ctx.meta["sculpt"]["height"] > 20
    assert res.report["ok"], res.report["errors"]
    floating = panel_level_floating(res)
    assert not floating, [(i, ctx.panels[i].kind, ctx.panels[i].tags) for i in floating[:5]]
    bad = find_overlaps(ctx.panels)
    assert not bad, describe(ctx, bad)
    assert 1000 < len(res.chunks) < 20000
