"""End-to-end: all presets generate; determinism; parallel == serial; CLI outputs; timing."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bgen.cli import main as cli_main, parse_seeds
from bgen.export.bld import read_bld, write_bld
from bgen.materials import F_ANCHOR
from bgen.pipeline import generate
from bgen.spec import load_spec, preset_spec

from test_nodes_overlap import describe, find_overlaps

SPECS = Path(__file__).resolve().parents[1] / "specs"
PRESETS = ["office", "apartment", "warehouse", "tower", "house", "kyoto"]


def panel_level_floating(res) -> list[int]:
    """Non-cosmetic panels none of whose chunks reach an anchor. Robust to single fracture
    slivers: this checks the junction geometry (panel contacts), not fracture quality."""
    n = len(res.chunks)
    a = np.array([e.a for e in res.edges], int)
    b = np.array([e.b for e in res.edges], int)
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)), directed=False)
    good = {lab[i] for i, c in enumerate(res.chunks) if c.flags & F_ANCHOR}
    ok_panels = {c.panel_id for i, c in enumerate(res.chunks) if lab[i] in good}
    return [p.id for p in res.ctx.panels if p.id not in ok_panels and not p.tags.get("cosmetic")
            and p.material != "glass"]


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("seed", [1, 2])
def test_presets_end_to_end(preset, seed, tmp_path):
    res = generate(load_spec(SPECS / f"{preset}.yaml"), seed)
    ctx = res.ctx
    assert len(ctx.panels) > 20 and len(res.chunks) >= len(ctx.panels) and res.edges
    assert res.report["room_connectivity"]["ok"], res.report["room_connectivity"]
    floating = panel_level_floating(res)
    assert not floating, [(i, ctx.panels[i].kind, ctx.panels[i].tags) for i in floating[:5]]
    bad = find_overlaps(ctx.panels)
    assert not bad, describe(ctx, bad)
    ids = [c.panel_id for c in res.chunks]
    assert ids == sorted(ids)
    # realistic dimensions
    fh = ctx.params["floor_height"]
    nf = len(ctx.blockout.floor_polys)
    assert 2.8 <= fh <= 8.0
    if preset == "tower":
        assert 8 <= nf <= 15
    if preset == "house":
        x0, y0, x1, y1 = ctx.blockout.floor_polys[0].bounds
        assert 8 <= max(x1 - x0, y1 - y0) <= 13
    p = tmp_path / "b.bld"
    write_bld(res.bd, p)
    rd = read_bld(p)
    assert len(rd.chunks) == len(res.chunks) and len(rd.elements) == len(ctx.panels)
    # validation errors, if any, may only come from fracture quality (not junction geometry)
    for e in res.report["errors"]:
        assert any(k in e for k in ("not connected to an anchor", "non-convex hull", "interpenetrating",
                                    "volume", "watertight", "winding", "aspect")), e


def test_determinism_same_seed_identical_bytes(tmp_path):
    spec = load_spec(SPECS / "house.yaml")
    a, b = generate(spec, 9), generate(spec, 9)
    write_bld(a.bd, tmp_path / "a.bld")
    write_bld(b.bd, tmp_path / "b.bld")
    assert (tmp_path / "a.bld").read_bytes() == (tmp_path / "b.bld").read_bytes()
    c = generate(spec, 10)
    write_bld(c.bd, tmp_path / "c.bld")
    assert (tmp_path / "c.bld").read_bytes() != (tmp_path / "a.bld").read_bytes()


def test_parallel_equals_serial(tmp_path):
    spec = load_spec(SPECS / "office.yaml")
    a = generate(spec, 3, jobs=1)
    b = generate(spec, 3, jobs=4)
    assert len(a.ctx.panels) > 200                       # exercises the process pool
    write_bld(a.bd, tmp_path / "a.bld")
    write_bld(b.bd, tmp_path / "b.bld")
    assert (tmp_path / "a.bld").read_bytes() == (tmp_path / "b.bld").read_bytes()


def test_node_edit_does_not_reshuffle_others():
    base = preset_spec("office")
    a = generate(dict(base, nodes=None), 4, validate=False)
    spec = {"name": "office", "preset": "office"}
    from bgen.spec import resolve_spec
    rs = resolve_spec(spec, 4)
    nodes = rs["nodes"]
    i = [list(n)[0] for n in nodes].index("roofs")
    nodes[i]["roofs"]["flat"]["parapet"] = 1.4
    b = generate(rs, 4, validate=False)
    pa = [(p.kind, np.round(p.frame, 6).tobytes(), p.profile.wkb) for p in a.ctx.panels if p.kind != "parapet"]
    pb = [(p.kind, np.round(p.frame, 6).tobytes(), p.profile.wkb) for p in b.ctx.panels if p.kind != "parapet"]
    assert pa == pb


def test_office_five_floors_under_30s():
    spec = load_spec(SPECS / "office.yaml")
    spec["blockout"] = {"shape": "rect", "width": 30.0, "depth": 16.0, "floors": 5}
    t = time.perf_counter()
    res = generate(spec, 1)
    dt = time.perf_counter() - t
    assert len(res.ctx.blockout.floor_polys) == 5
    assert dt < 30.0, res.timings


def test_parse_seeds():
    assert parse_seeds("1..3") == [1, 2, 3] and parse_seeds("4,7") == [4, 7] and parse_seeds("2-3") == [2, 3]


def test_cli_build_validate_batch(tmp_path):
    out = tmp_path / "out"
    rc = cli_main(["build", str(SPECS / "house.yaml"), "--seed", "3", "--out", str(out), "--no-render",
                   "--graph-json"])
    d = out / "house_3"
    for f in ("building.bld", "building.glb", "manifest.json", "graph.json"):
        assert (d / f).exists(), f
    m = json.loads((d / "manifest.json").read_text())
    assert m["seed"] == 3 and m["stats"]["chunks"] > 0 and m["rooms"] and m["sockets"]
    assert m["stats"]["chunks_by_material"] and m["timings"]["total"] > 0
    assert rc == (0 if m["validation"]["ok"] else 2)
    import trimesh
    sc = trimesh.load(d / "building.glb")
    assert len(sc.geometry) == m["stats"]["chunks"]
    raw = (d / "building.glb").read_bytes()
    js = json.loads(raw[20:20 + int.from_bytes(raw[12:16], "little")])
    ex = [n["extras"] for n in js["nodes"] if "extras" in n]
    assert len(ex) == m["stats"]["chunks"] and {"chunk_id", "panel_id", "material", "flags"} <= set(ex[0])
    rc = cli_main(["validate", str(d)])
    assert rc in (0, 2)
    rc = cli_main(["batch", str(SPECS / "house.yaml"), "--seeds", "1..2", "--jobs", "2", "--out", str(out),
                   "--no-render", "--no-glb"])
    assert (out / "house_1" / "building.bld").exists() and (out / "house_2" / "building.bld").exists()
    assert cli_main(["district"]) == 3
