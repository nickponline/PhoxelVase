"""Per-node tests: each feature node in isolation (or with only the nodes it depends on)."""
from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

from bgen.blockout import build_blockout
from bgen.model import BuildingContext
from bgen.nodes import get_node, get_reserve
from bgen.nodes._util import MIN_SOLID, chain_walls, plan_footprint, wallspec_of
from bgen.pipeline import run_nodes
from bgen.rng import node_rng, rv, stream
from bgen.spec import resolve_spec

from test_nodes_overlap import find_overlaps, describe

GLOBAL = dict(floor_height=3.5, slab_thickness=0.3, ext_wall_thickness=0.3, int_wall_thickness=0.15,
              slab_material="concrete", ext_wall_material="concrete", int_wall_material="brick",
              column_material="concrete", roof_material="concrete", roof_thickness=0.2)


def make_ctx(masses, **g):
    params = dict(GLOBAL, **g)
    ctx = BuildingContext(spec={"name": "t", "seed": 0, "indestructible": ["ground_slab"]}, params=params)
    ctx.blockout = build_blockout(masses, params)
    return ctx


def run(ctx, chain):
    """chain: list of (name, params). Runs reserves then nodes like the pipeline does."""
    for i, (n, p) in enumerate(chain):
        r = get_reserve(n)
        if r:
            r(ctx, p, node_rng(0, i, n + "r"))
    for i, (n, p) in enumerate(chain):
        ctx = get_node(n)(ctx, p, node_rng(0, i, n))
    return ctx


RECT = [dict(footprint=[[0, 0], [20, 0], [20, 12], [0, 12]], floors=3, roof="flat")]
LSHAPE = [dict(footprint=[[0, 0], [24, 0], [24, 10], [0, 10]], floors=3),
          dict(footprint=[[0, 10], [10, 10], [10, 22], [0, 22]], floors=2)]


def check_frames(ctx):
    for p in ctx.panels:
        R = p.frame[:3, :3]
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9), p.id
        assert np.linalg.det(R) > 0.999999, p.id
        if np.all(np.isin(np.round(np.abs(R), 12), [0.0, 1.0])):    # axis-aligned -> on 0.1 mm grid
            t = p.frame[:3, 3] / 1e-4
            assert np.allclose(t, np.round(t), atol=1e-6), (p.id, p.frame[:3, 3])
        assert p.profile.is_valid and p.profile.area > 0


# ------------------------------------------------------------------ rng / spec / blockout

def test_rng_streams_independent_and_stable():
    a = node_rng(5, 2, "rooms").uniform(size=4)
    assert np.allclose(a, node_rng(5, 2, "rooms").uniform(size=4))
    assert not np.allclose(a, node_rng(5, 3, "rooms").uniform(size=4))
    assert not np.allclose(a, node_rng(6, 2, "rooms").uniform(size=4))
    g = stream("x")
    assert isinstance(rv([3, 6], g), int) and 3 <= rv([3, 6], g) <= 6
    assert 1.0 <= rv([1.0, 2.0], g) <= 2.0 and rv("abc", g) == "abc"


def test_spec_resolution_samples_ranges_deterministically():
    s = {"name": "x", "preset": "office", "global": {"floor_height": [3.0, 4.0]}}
    a, b = resolve_spec(s, 3), resolve_spec(s, 3)
    assert a == b
    assert 3.0 <= a["global"]["floor_height"] <= 4.0
    assert all(isinstance(m["floors"], int) for m in a["blockout"])
    c = resolve_spec(s, 4)
    assert (c["global"]["floor_height"], c["blockout"]) != (a["global"]["floor_height"], a["blockout"])
    # editing one node's params does not change the others' resolved params
    s2 = dict(s, nodes=[{"floors": {}}, {"rooms": {"min_room": 12.0}}])
    r2 = resolve_spec(s2, 3)
    assert r2["nodes"][1]["rooms"]["min_room"] == 12.0 and "corridor" in r2["nodes"][1]["rooms"]


def test_blockout_union_levels_facades():
    ctx = make_ctx(LSHAPE)
    bo = ctx.blockout
    assert len(bo.floor_polys) == 3 and bo.floor_z == [0.0, 3.5, 7.0, 10.5]
    assert abs(bo.floor_polys[0].area - (240 + 120)) < 1e-6
    assert abs(bo.floor_polys[2].area - 240) < 1e-6
    assert len(bo.facades[0]) == 6 and len(bo.facades[2]) == 4          # L has 6 straight segments
    for g in bo.floor_polys:
        assert g.exterior.is_ccw


# ------------------------------------------------------------------ floors / walls / columns

def test_floors_node():
    ctx = run(make_ctx(LSHAPE), [("floors", {})])
    slabs = [p for p in ctx.panels if p.kind == "floor"]
    assert len(slabs) == 3
    g = slabs[0]
    assert g.tags["anchor"] and g.tags["indestructible"] and g.tags["role"] == "ground_slab"
    assert abs(g.frame[2, 3] + 0.3) < 1e-9 and abs(g.profile.area - 360) < 1e-6
    check_frames(ctx)


@pytest.mark.parametrize("rule", ["x_runs_through", "post"])
def test_exterior_walls_corner_rule_tiles_ring(rule):
    ctx = run(make_ctx(LSHAPE), [("floors", {}), ("exterior_walls", {"corner_rule": rule})])
    walls = [p for p in ctx.panels if p.tags.get("floor") == 0 and p.kind in ("ext_wall", "column")]
    fp0 = ctx.blockout.floor_polys[0]
    ring = fp0.difference(fp0.buffer(-0.3, join_style=2))
    foot = [plan_footprint(p) for p in walls]
    assert abs(sum(f.area for f in foot) - ring.area) < 1e-6          # exact tiling, no overlap
    assert unary_union(foot).symmetric_difference(ring).area < 1e-6
    for p in walls:
        if p.kind == "ext_wall":
            z0, z1 = p.frame[2, 3], p.frame[2, 3] + p.profile.bounds[3]
            assert abs(z0 - 0.0) < 1e-9 and abs(z1 - 3.2) < 1e-9                  # between slabs
            assert p.tags["bays"][0] == 0.0 and len(p.tags["bays"]) >= 2
    if rule == "x_runs_through":
        assert not any(p.kind == "column" for p in walls)
    assert not find_overlaps(ctx.panels)
    check_frames(ctx)


def test_chain_walls_non_orthogonal_posts():
    pts = [(0, 0), (10, 0), (14, 5), (6, 9), (0, 6)]
    walls, posts = chain_walls(pts, True, 0.3)
    poly = Polygon(pts)
    ring = poly.difference(poly.buffer(-0.3, join_style=2))
    assert len(posts) == 4                       # only the (0,0) corner is orthogonal
    from bgen.nodes._util import WallSpec
    fps = [WallSpec(A=w["A"], d=w["d"], x0=w["x0"], x1=w["x1"], t=0.3, z0=0, z1=1).footprint() for w in walls]
    tot = sum(f.area for f in fps) + sum(p.area for p in posts)
    assert abs(tot - ring.area) < 1e-6


def test_columns_clear_of_walls_and_stacked():
    ctx = run(make_ctx(RECT), [("floors", {}), ("exterior_walls", {}), ("columns", {"grid": 6.0, "size": 0.5})])
    cols = [p for p in ctx.panels if p.kind == "column"]
    assert cols
    by_floor = {}
    for c in cols:
        by_floor.setdefault(c.tags["floor"], set()).add(tuple(np.round(plan_footprint(c).bounds, 4)))
    assert by_floor[0] == by_floor[1] == by_floor[2]                      # same grid every floor
    assert not find_overlaps(ctx.panels)
    check_frames(ctx)


# ------------------------------------------------------------------ rooms / stairs

@pytest.fixture(scope="module")
def office_ctx():
    s = {"name": "o", "preset": "office", "blockout": LSHAPE}
    return run_nodes(s, 2)


def test_rooms_tile_interior(office_ctx):
    ctx = office_ctx
    t = ctx.params["ext_wall_thickness"]
    for k, fp in enumerate(ctx.blockout.floor_polys):
        rooms = [r.polygon for r in ctx.rooms if r.floor == k]
        assert rooms
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                assert rooms[i].intersection(rooms[j]).area < 1e-6
        walls = [plan_footprint(p) for p in ctx.panels if p.kind == "int_wall" and p.tags.get("floor") == k]
        cols = [plan_footprint(p) for p in ctx.panels if p.kind == "column" and p.tags.get("floor") == k]
        interior = fp.buffer(-t, join_style=2)
        covered = unary_union(rooms + walls + cols)
        assert covered.difference(interior).area < 1e-4
        # notches (where a wall was clipped around a column) are the only uncovered area
        assert interior.difference(covered).area < 0.02 * interior.area
    tags = {r.tag for r in ctx.rooms}
    assert "stair" in tags and "corridor" in tags


def test_stair_core_identical_and_slab_holes(office_ctx):
    ctx = office_ctx
    cores = ctx.meta["stair_cores"]
    assert len(cores) == 1
    c = cores[0]
    assert c["floors"] == list(range(len(ctx.blockout.floor_polys)))
    inner = box(*c["inner"])
    for k in c["floors"]:
        stair_rooms = [r for r in ctx.rooms if r.floor == k and r.tag == "stair"]
        assert len(stair_rooms) == 1 and stair_rooms[0].polygon.equals_exact(inner, 1e-6)
        slab = [p for p in ctx.panels if p.kind == "floor" and p.tags["floor"] == k][0]
        assert (len(slab.profile.interiors) == 1) == (k > 0)
    n = len(c["floors"]) - 1
    kinds = [p.kind for p in ctx.panels]
    assert kinds.count("stair") == 2 * n and kinds.count("landing") == n
    steps = [p for p in ctx.panels if p.kind == "step"]
    assert steps and all(p.tags.get("cosmetic") for p in steps)
    assert len(ctx.meta["stair_links"]) == n


def test_rooms_all_reachable_and_doors(office_ctx):
    ctx = office_ctx
    conn = ctx.meta["room_connectivity"]
    assert conn["ok"], conn
    with_door = set()
    for d in ctx.meta["doors"]:
        with_door.update(d["rooms"])
    assert all(r.id in with_door for r in ctx.rooms)                       # every room has a door
    assert any(d["kind"] == "entrance" for d in ctx.meta["doors"])


# ------------------------------------------------------------------ openings

def test_openings_glass_in_holes_and_min_solid(office_ctx):
    ctx = office_ctx
    P = {p.id: p for p in ctx.panels}
    glass = [p for p in ctx.panels if p.kind == "glass"]
    assert glass
    for g in glass:
        host = P[g.tags["host"]]
        holes = [Polygon(r) for r in host.profile.interiors]
        assert any(h.equals_exact(g.profile, 1e-6) or h.symmetric_difference(g.profile).area < 1e-8 for h in holes)
        # glass sits inside the wall thickness
        off = (g.frame[:3, 3] - host.frame[:3, 3]) @ host.frame[:3, 2]
        assert 0 <= off and off + g.thickness <= host.thickness + 1e-9
        assert np.allclose(g.frame[:3, :3], host.frame[:3, :3])
    for p in ctx.panels:
        if p.kind not in ("ext_wall", "int_wall"):
            continue
        ext = p.profile.exterior
        holes = [Polygon(r) for r in p.profile.interiors]
        for i, h in enumerate(holes):
            assert h.exterior.distance(ext) >= MIN_SOLID - 1e-6
            for h2 in holes[i + 1:]:
                assert h.distance(h2) >= MIN_SOLID - 1e-6
    sockets = {s.type for s in ctx.sockets}
    assert {"window_frame", "door"} <= sockets
    assert not find_overlaps(ctx.panels), describe(ctx, find_overlaps(ctx.panels))


@pytest.mark.parametrize("style", ["punched", "ribbon", "arched", "round", "storefront"])
def test_window_styles(style):
    ctx = make_ctx(RECT)
    ctx = run(ctx, [("floors", {}), ("exterior_walls", {}), ("openings", {"windows": {"style": style}})])
    glass = [p for p in ctx.panels if p.kind == "glass"]
    assert glass
    if style == "round":
        assert all(len(g.profile.exterior.coords) > 10 for g in glass)
    if style == "arched":
        assert any(len(g.profile.exterior.coords) > 6 for g in glass)
    assert not find_overlaps(ctx.panels)


# ------------------------------------------------------------------ roofs / balconies

@pytest.mark.parametrize("roof", ["flat", "gable", "hip", "kyoto"])
def test_roofs(roof):
    masses = [dict(footprint=[[0, 0], [14, 0], [14, 9], [0, 9]], floors=2, roof=roof)]
    ctx = run(make_ctx(masses), [("floors", {}), ("exterior_walls", {}), ("roofs", {})])
    assert ctx.meta["roofs"][-1]["type"] == roof
    kinds = [p.kind for p in ctx.panels]
    if roof == "flat":
        assert kinds.count("parapet") == 4
    else:
        assert kinds.count("roof") >= 3 and "ridge" in kinds
        top = max(p.to_world(np.asarray(p.profile.exterior.coords))[:, 2].max() for p in ctx.panels)
        assert top > ctx.blockout.floor_z[-1] + 1.5
    bad = find_overlaps(ctx.panels)
    assert not bad, describe(ctx, bad)
    check_frames(ctx)


def test_setback_roof_parapets_open_chain():
    masses = [dict(footprint=[[0, 0], [30, 0], [30, 20], [0, 20]], floors=2),
              dict(footprint=[[0, 0], [20, 0], [20, 20], [0, 20]], floors=4)]
    ctx = run(make_ctx(masses), [("floors", {}), ("exterior_walls", {}), ("roofs", {})])
    par = [p for p in ctx.panels if p.kind == "parapet"]
    assert len([p for p in par if p.tags["level"] == 2]) == 3        # open U chain on the setback
    assert len([p for p in par if p.tags["level"] == 4]) == 4
    assert not find_overlaps(ctx.panels)


def test_balconies():
    masses = [dict(footprint=[[0, 0], [24, 0], [24, 12], [0, 12]], floors=4)]
    ctx = run(make_ctx(masses), [("floors", {}), ("exterior_walls", {"bay_width": [3.5, 4.0]}),
                                 ("openings", {}), ("balconies", {"facades": ["south"], "probability": 1.0})])
    bal = [p for p in ctx.panels if p.kind == "balcony"]
    assert len(bal) >= 3 and all(p.tags["floor"] >= 1 for p in bal)
    for b in bal:
        assert plan_footprint(b).bounds[3] <= 1e-9                       # outside the south facade
    assert sum(s.type == "balcony_rail" for s in ctx.sockets) == len(bal)
    assert any(d["kind"] == "balcony_door" for d in ctx.meta["doors"])
    assert not find_overlaps(ctx.panels)


# ------------------------------------------------------------------ manual

def test_manual_overrides(tmp_path):
    base = [("floors", {}), ("exterior_walls", {}), ("columns", {}), ("openings", {})]
    ctx0 = run(make_ctx(RECT), list(base))
    col = next(p for p in ctx0.panels if p.kind == "column")
    cpt = plan_footprint(col).centroid
    win = next(p for p in ctx0.panels if p.kind == "glass")
    wpt = win.to_world(np.asarray([win.profile.centroid.coords[0]]))[0]
    wall = next(p for p in ctx0.panels if p.kind == "ext_wall" and p.tags["floor"] == 1)
    (tmp_path / "o.yaml").write_text(f"""
overrides:
  - {{op: delete_panel, kind: column, at: [{cpt.x}, {cpt.y}, 1.0]}}
  - {{op: delete_opening, kind: ext_wall, at: [{wpt[0]}, {wpt[1]}, {wpt[2]}]}}
  - {{op: add_panel, kind: column, box: [[-2, -2, 0], [-1.5, -1.5, 3]], material: concrete}}
  - {{op: delete_panel, kind: column, at: [500, 500, 500]}}
""")
    ctx = make_ctx(RECT)
    ctx.meta["spec_dir"] = str(tmp_path)
    ctx = run(ctx, base + [("manual", {"file": "o.yaml"})])
    rep = ctx.meta["manual"]
    assert rep["applied"] == 3 and len(rep["skipped"]) == 1
    n_glass0 = sum(p.kind == "glass" for p in ctx0.panels)
    assert sum(p.kind == "glass" for p in ctx.panels) == n_glass0 - 1
    assert [p.id for p in ctx.panels] == list(range(len(ctx.panels)))      # dense ids after removal
    assert all(0 <= s.owner_panel < len(ctx.panels) for s in ctx.sockets)
