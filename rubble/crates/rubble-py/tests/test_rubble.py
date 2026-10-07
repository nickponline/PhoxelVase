"""rubble-py binding tests. Build first: `maturin develop --release` (in rubble/crates/rubble-py).

Fixture fixtures/two_box.bld: chunks 0,1 (bottom, anchored) at z in [0,1], chunks 2,3 on top,
edges 0-1, 0-2, 1-3, 2-3; 2400 kg each.
"""
from pathlib import Path

import numpy as np
import pytest

rubble = pytest.importorskip("rubble")
if not hasattr(rubble, "World"):  # repo-root `rubble/` dir shadowing an uninstalled module
    pytest.skip("rubble extension not installed", allow_module_level=True)

FIX = str(Path(__file__).resolve().parents[4] / "fixtures" / "two_box.bld")


def make_world(**cfg):
    w = rubble.World(cfg or None)
    b = w.load_building(FIX)
    w.add_ground_plane(0.0)
    return w, b


def test_static_report_module_function():
    r = rubble.static_stress_report(FIX)
    assert r["converged"]
    assert r["unsupported_nodes"] == []
    assert 0.0 < r["max_util"] < 0.5
    u = r["utilization"]
    assert isinstance(u, np.ndarray) and u.dtype == np.float32 and u.shape == (4,)
    assert r["max_util"] == pytest.approx(float(u.max()))
    # load = top two chunks' weight flows into the anchors
    assert r["total_load"] == pytest.approx(2 * 2400 * 9.81, rel=1e-3)
    assert r["anchor_flow"] == pytest.approx(r["total_load"], rel=1e-3)
    e, util, a, b = r["worst_edges"][0]
    assert util == pytest.approx(r["max_util"]) and {a, b} in ({0, 2}, {1, 3})
    r2 = rubble.static_stress_report(FIX, bending=False)
    assert r2["converged"]


def test_static_report_bad_path():
    with pytest.raises(OSError):
        rubble.static_stress_report("/nonexistent.bld")


def test_world_basics():
    w, b = make_world()
    assert b == 0 and w.n_buildings == 1 and w.n_chunks(b) == 4
    xf = w.chunk_world_transforms(b)
    assert xf.shape == (4, 4, 4) and xf.dtype == np.float32
    np.testing.assert_allclose(xf, np.broadcast_to(np.eye(4), (4, 4, 4)), atol=1e-6)
    assert w.alive(b).dtype == bool and w.alive(b).all()
    assert w.edge_utilization(b).shape == (4,)
    assert w.chunk_state(b).tolist() == [0, 0, 0, 0]
    s = w.stats()
    assert s["chunks_total"] == 4 and s["static_chunks"] == 4 and "timings" in s
    assert "World(" in repr(w)
    with pytest.raises(IndexError):
        w.alive(5)
    with pytest.raises(IndexError):
        w.damage_chunk(b, 99)


def test_placement_pose():
    w = rubble.World()
    b = w.load_building(FIX, pos=(10.0, 0.0, 0.0), yaw=np.pi / 2)
    X = w.chunk_world_transforms(b)[0]
    p = X @ np.array([1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(p[:3], [10.0, 1.0, 0.0], atol=1e-5)


def test_config_override_and_bad_config():
    w = rubble.World({"collapse_delay_min": 0.05, "stress": {"bending": False}})
    assert w.config["collapse_delay_min"] == pytest.approx(0.05)
    assert w.config["stress"]["bending"] is False
    assert w.config["stress"]["enabled"] is True  # untouched nested defaults survive
    with pytest.raises(ValueError):
        rubble.World({"gravity": "down"})


def test_stress_utilization_after_step():
    w, b = make_world()
    w.step(1 / 60, n=5)
    u = w.edge_utilization(b)
    assert u.max() > 0.0
    rep = w.static_stress_report(b)
    assert rep["converged"] and rep["max_util"] == pytest.approx(float(u.max()), rel=0.05)


def test_destroy_base_top_falls():
    w, b = make_world(collapse_delay_min=0.1, collapse_delay_max=0.2)
    w.damage_chunk(b, [0, 1], 1e9)
    w.step(1 / 60, n=180)
    alive = w.alive(b)
    assert alive.tolist() == [False, False, True, True]
    ev = w.drain_events()
    types = {e["type"] for e in ev}
    assert "ChunkDestroyed" in types and "ClusterDetached" in types
    assert w.drain_events() == []
    xf = w.chunk_world_transforms(b)
    com_top = np.array([1.0, 0.5, 1.5, 1.0])
    z = (xf[2] @ com_top)[2]
    assert z < 0.6, f"top slab should have dropped ~1 m onto the ground, com z = {z}"
    st = w.chunk_state(b)
    assert st[0] == 4 and st[1] == 4 and st[2] in (2, 3)


def test_explosion_and_fire():
    w, b = make_world()
    w.fire((-5.0, 0.5, 0.5), (1.0, 0.0, 0.0), weapon="sniper")
    w.fire((-5.0, 0.5, 1.5), (1.0, 0.0, 0.0))  # default AR
    w.fire((-5.0, 0.5, 1.5), (1.0, 0.0, 0.0), weapon="launcher", ballistic=True)
    custom = {"custom": {"damage": 5.0, "splash_r": 0.0, "splash_frac": 0.0, "penetration": 0.0,
                         "range": 100.0, "speed": 800.0, "radius": 0.02, "impulse": 1.0, "gravity_scale": 1.0}}
    w.fire((-5.0, 0.5, 0.5), (1.0, 0.0, 0.0), weapon=custom)
    with pytest.raises(ValueError):
        w.fire((0, 0, 0), (1, 0, 0), weapon="bazooka")
    w.step(n=30)
    assert w.alive(b).all()  # bullets chip, don't destroy 2.4 t concrete boxes
    w.explode((1.0, 0.5, 1.0), radius=3.0, damage=1e7, impulse=1e4, inner_radius=2.0)
    w.step(n=10)
    assert not w.alive(b).any()
    assert sum(e["type"] == "ChunkDestroyed" for e in w.drain_events()) == 4


def test_determinism():
    def run():
        w, b = make_world()
        w.damage_chunk(b, 0)
        w.step(n=120)
        return w.chunk_world_transforms(b)

    np.testing.assert_array_equal(run(), run())
