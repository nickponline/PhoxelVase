"""fracture_panel: conservation, convexity, flags, determinism, performance."""
import time

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from bgen.geom.convex2d import is_convex, signed_area, to_shapely
from bgen.geom.fracture import fracture_panel, fracture_pieces
from bgen.materials import (DEFAULT_MATERIALS, FACE_CAP_EDGE, FACE_INNER, FACE_OUTER,
                            F_COSMETIC_ATTACHED, F_GLASS, F_INDESTRUCTIBLE)
from bgen.model import Panel
from bgen.validate import ValidationReport, check_convex, check_meshes, check_volume, chunk_aspect

from test_geom_convex2d import random_polygon


def windowed_wall(L=30.0, H=3.0, n=10, door=True):
    w = box(0, 0, L, H)
    step = L / n
    for i in range(n):
        w = w.difference(box(i * step + 1, 0.9, i * step + 2.5, 2.4))
    if door:
        w = w.difference(box(L - 1.3, 0, L - 0.4, 2.1))
    return w


def rot_x(deg):
    a = np.radians(deg)
    f = np.eye(4)
    f[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    return f


def panel(profile, material="concrete", frame=None, t=0.3, tags=None, pid=0, kind="ext_wall"):
    return Panel(pid, kind, np.eye(4) if frame is None else frame, profile, t, material, tags or {})


def check_chunks(p, chunks):
    rep = ValidationReport()
    check_convex(chunks, rep)
    check_meshes(chunks, rep)
    check_volume([p], chunks, rep)
    assert rep.ok, rep.errors
    u = shapely.union_all(to_shapely([c.poly2d for c in chunks]))
    # exact up to the 1e-6 m short-edge cleanup of output pieces
    assert u.symmetric_difference(p.profile).area < 1e-6 * p.profile.area
    assert sum(signed_area(c.poly2d) for c in chunks) == pytest.approx(p.profile.area, rel=1e-6)
    for c in chunks:
        assert is_convex(c.poly2d, 1e-7)
        mat = DEFAULT_MATERIALS[p.material]
        assert c.mass == pytest.approx(mat.density * c.volume)
        assert c.hp == pytest.approx(max(mat.min_hp, mat.hp_per_m3 * c.volume))
        assert c.panel_id == p.id and c.material == p.material


def side_face_flags(c):
    n = len(c.poly2d)
    return c.mesh_face_flags[2 * n::4]


def test_wall_with_windows():
    p = panel(windowed_wall())
    ch = fracture_panel(p, {}, np.random.default_rng(0))
    check_chunks(p, ch)
    cell = DEFAULT_MATERIALS["concrete"].cell_size
    assert 0.5 * p.profile.area / cell ** 2 < len(ch) < 3 * p.profile.area / cell ** 2
    # sliver / aspect quality over several seeds (slivers are rescued where geometry allows)
    n = sl = 0
    for s in range(6):
        pcs, info = fracture_pieces(p, {}, np.random.default_rng(s))
        n += len(pcs)
        sl += info["slivers"]
        assert max(chunk_aspect(c) for c in fracture_panel(p, {}, np.random.default_rng(s))) < 10
    assert sl <= 0.01 * n


def test_face_flags():
    p = panel(windowed_wall(10, 3, 3))
    ch = fracture_panel(p, {}, np.random.default_rng(1))
    cap_len = 0.0
    for c in ch:
        n = len(c.poly2d)
        assert (c.mesh_face_flags[:2 * n] == FACE_OUTER).all()
        sf = side_face_flags(c)
        assert set(np.unique(sf)) <= {FACE_INNER, FACE_CAP_EDGE}
        L = np.linalg.norm(np.roll(c.poly2d, -1, 0) - c.poly2d, axis=1)
        cap_len += (L * (sf == FACE_CAP_EDGE)).sum()
    # cap-edge faces exactly cover the outer border and the opening edges
    assert cap_len == pytest.approx(p.profile.length, rel=1e-9)


def test_deterministic():
    p = panel(windowed_wall(12, 3, 4))
    a = fracture_panel(p, {"edge_bias": 2.0}, np.random.default_rng(42))
    b = fracture_panel(p, {"edge_bias": 2.0}, np.random.default_rng(42))
    c = fracture_panel(p, {"edge_bias": 2.0}, np.random.default_rng(43))
    assert len(a) == len(b)
    assert all(np.array_equal(x.mesh_verts, y.mesh_verts) for x, y in zip(a, b))
    assert len(a) != len(c) or not all(np.array_equal(x.poly2d, y.poly2d) for x, y in zip(a, c))


@pytest.mark.parametrize("seed", range(25))
def test_random_panels(seed):
    rng = np.random.default_rng(seed)
    prof = random_polygon(rng)
    mat = ["concrete", "brick", "wood", "metal"][seed % 4]
    cfg = {"cell_size": rng.uniform(0.4, 1.2), "edge_bias": rng.uniform(0, 3),
           "anisotropy": [rng.uniform(1, 3), 1.0]}
    fr = rot_x(rng.uniform(0, 90))
    fr[:3, 3] = rng.uniform(-10, 10, 3)
    p = panel(prof, mat, fr, t=rng.uniform(0.1, 0.5))
    check_chunks(p, fracture_panel(p, cfg, rng))


def test_tilted_roof_35deg():
    fr = rot_x(35.0)
    fr[:3, 3] = [0, 0, 6.6]
    p = panel(box(0, 0, 12, 5).difference(box(4, 2, 5, 3)), frame=fr, t=0.25, kind="roof")
    ch = fracture_panel(p, {}, np.random.default_rng(3))
    check_chunks(p, ch)
    up = fr[:3, 2]
    for c in ch:  # outer cap normal is the frame's +Z
        assert np.allclose(c.planes[-1, :3], up)


def test_edge_bias_adds_seeds_near_openings():
    p = panel(windowed_wall(12, 3, 4, door=False))
    _, i0 = fracture_pieces(p, {}, np.random.default_rng(0))
    _, i1 = fracture_pieces(p, {"edge_bias": 3.0}, np.random.default_rng(0))
    assert i1["seeds"] > i0["seeds"]
    ch = fracture_panel(p, {"edge_bias": 3.0}, np.random.default_rng(0))
    check_chunks(p, ch)


def test_anisotropy_elongates_cells():
    p = panel(box(0, 0, 20, 10), "wood", t=0.1)

    def mean_ratio(cfg):
        ch = fracture_panel(p, cfg, np.random.default_rng(5))
        r = [np.ptp(c.poly2d[:, 0]) / np.ptp(c.poly2d[:, 1]) for c in ch]
        return np.median(r)

    assert mean_ratio({"anisotropy": [3, 1]}) > 2.0 * mean_ratio({})


def test_glass_single_chunk_per_part():
    p = panel(box(0, 0, 1.5, 1.2), "glass", t=0.02, kind="glass")
    ch = fracture_panel(p, {}, np.random.default_rng(0))
    assert len(ch) == 1 and ch[0].flags & F_GLASS
    arch = box(0, 0, 1, 1).union(Point(0.5, 1).buffer(0.5, 8))  # arched window: convex anyway
    lshape = box(0, 0, 3, 1).union(box(0, 0, 1, 3))
    for prof, nparts in ((arch, 1), (lshape, 2)):
        ch = fracture_panel(panel(prof, "glass", t=0.02), {}, np.random.default_rng(0))
        assert len(ch) == nparts and all(c.flags == F_GLASS for c in ch)
        check_chunks(panel(prof, "glass", t=0.02), ch)


def test_indestructible_and_cosmetic_flags():
    prof = box(0, 0, 10, 10).difference(box(4, 4, 6, 6))
    p = panel(prof, tags={"indestructible": True}, kind="floor")
    ch = fracture_panel(p, {}, np.random.default_rng(0))
    assert len(ch) == 4 and all(c.flags & F_INDESTRUCTIBLE for c in ch)  # convex parts only
    check_chunks(p, ch)
    p = panel(box(0, 0, 1.0, 0.3), tags={"cosmetic_attached": True}, kind="step")
    ch = fracture_panel(p, {}, np.random.default_rng(0))
    assert len(ch) == 1 and ch[0].flags == F_COSMETIC_ATTACHED
    p = panel(box(0, 0, 4, 3), tags={"cosmetic": True})
    assert all(c.flags & F_COSMETIC_ATTACHED for c in fracture_panel(p, {}, np.random.default_rng(0)))


def test_small_panel_is_convex_parts():
    p = panel(box(0, 0, 0.5, 0.5), "concrete")
    ch = fracture_panel(p, {}, np.random.default_rng(0))
    assert len(ch) == 1
    p = panel(box(0, 0, 0.1, 0.1), "concrete")
    assert fracture_panel(p, {}, np.random.default_rng(0))[0].hp == DEFAULT_MATERIALS["concrete"].min_hp


def test_min_chunk_area_merge():
    p = panel(windowed_wall(12, 3, 4))
    big = fracture_panel(p, {"min_chunk_area": 0.6}, np.random.default_rng(0))
    small = fracture_panel(p, {"min_chunk_area": 0.0, "merge_cells": False}, np.random.default_rng(0))
    assert len(big) < len(small)
    check_chunks(p, big)
    check_chunks(p, small)


def test_perf_30x3_wall():
    p = panel(windowed_wall())
    fracture_panel(p, {}, np.random.default_rng(0))  # warm-up
    t = time.perf_counter()
    for s in range(5):
        fracture_panel(p, {}, np.random.default_rng(s))
    dt = (time.perf_counter() - t) / 5
    print(f"30x3 wall fracture: {dt * 1e3:.1f} ms")
    assert dt < 0.5
