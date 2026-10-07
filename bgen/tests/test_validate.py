"""validate() (DESIGN §2.7 geometry/graph checks) on the box room and on broken inputs."""
import copy

import numpy as np
import pytest
from shapely.geometry import box

from bgen.geom.fracture import fracture_panel
from bgen.geom.prism import make_prism
from bgen.graph import build_graph, mark_anchors
from bgen.materials import F_ANCHOR, F_COSMETIC_ATTACHED
from bgen.model import Panel
from bgen.validate import ValidationReport, mesh_check_fast, mesh_check_trimesh, validate

from test_graph import box_room, fracture_all


@pytest.fixture(scope="module")
def room():
    P = box_room()
    C = fracture_all(P)
    mark_anchors(P, C)
    E = build_graph(P, C)
    return P, C, E


def test_box_room_valid(room):
    P, C, E = room
    r = validate(P, C, E)
    assert isinstance(r, ValidationReport)
    assert r.ok, r.errors
    assert r.stats["n_chunks"] == len(C) and r.stats["n_components"] == 1
    assert r.stats["max_aspect"] < 20 and not r.warnings


def test_tilted_roof_valid():
    a = np.radians(35)
    fr = np.eye(4)
    fr[1:3, 1:3] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    P = [Panel(0, "roof", fr, box(0, 0, 8, 5).difference(box(3, 2, 4, 3)), 0.25, "wood", {"anchor": True})]
    C = fracture_all(P, cfg={"anisotropy": [3, 1]})
    mark_anchors(P, C)
    r = validate(P, C, build_graph(P, C))
    assert r.ok, r.errors


def test_fast_mesh_check_matches_trimesh(room):
    _, C, _ = room
    sub = C[:25]
    a, b = mesh_check_fast(sub), mesh_check_trimesh(sub)
    assert a[0].all() and b[0].all() and a[1].all() and b[1].all()
    np.testing.assert_allclose(a[2], b[2], rtol=1e-9)
    np.testing.assert_allclose(a[2], [c.volume for c in sub], rtol=1e-9)


def test_detects_broken_mesh(room):
    P, C, E = room
    C2 = list(C)
    c = copy.copy(C[3])
    c.mesh_tris = c.mesh_tris[:-1]            # hole
    C2[3] = c
    d = copy.copy(C[5])
    d.mesh_tris = d.mesh_tris[:, ::-1].copy()  # inverted -> negative volume
    C2[5] = d
    r = validate(P, C2, E)
    assert not r.ok
    assert any("not watertight" in e and "3" in e for e in r.errors)
    assert any("non-positive volume" in e for e in r.errors)


def test_detects_nonconvex_hull(room):
    P, C, E = room
    C2 = list(C)
    c = copy.copy(C[0])
    c.hull_verts = c.hull_verts.copy()
    c.hull_verts[0] += 10 * c.planes[0, :3]
    C2[0] = c
    assert any("non-convex" in e for e in validate(P, C2, E).errors)


def test_detects_volume_loss(room):
    P, C, E = room
    drop = [i for i, c in enumerate(C) if c.panel_id == 2][0]
    C2 = [c for i, c in enumerate(C) if i != drop]
    r = validate(P, C2, build_graph(P, C2))
    assert any("panel 2" in e for e in r.errors)


def test_detects_interpenetration():
    fr = np.eye(4)
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    P = [Panel(0, "floor", fr, box(0, 0, 1, 1), 1.0, "concrete", {}),
         Panel(1, "floor", fr, box(0, 0, 1, 1), 1.0, "concrete", {})]
    f2 = np.eye(4)
    f2[:3, 3] = [0.5, 0.5, 0.5]
    a = make_prism(sq, 1.0, fr, 2400, 0, "concrete", flags=F_ANCHOR)
    b = make_prism(sq, 1.0, f2, 2400, 1, "concrete")
    r = validate(P, [a, b], [])
    assert any("interpenetrating" in e for e in r.errors)
    # touching (shared face) is fine
    f3 = np.eye(4)
    f3[:3, 3] = [1.0, 0.0, 0.0]
    c = make_prism(sq, 1.0, f3, 2400, 1, "concrete")
    r = validate(P, [a, c], build_graph(P, [a, c]))
    assert not any("interpenetrating" in e for e in r.errors)
    # rotated prism whose AABB overlaps but which does not touch: no false positive
    R = np.eye(4)
    t = np.radians(45)
    R[:3, :3] = [[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]
    R[:3, 3] = [0.0, 1.0 + np.sqrt(0.5) + 0.01, 0.0]
    d = make_prism(sq, 1.0, R, 2400, 1, "concrete")
    r = validate(P, [a, d], [])
    assert not any("interpenetrating" in e for e in r.errors)


def test_detects_floating_chunks(room):
    P, C, E = room
    ceiling = {i for i, c in enumerate(C) if c.panel_id == 1}
    E2 = [e for e in E if not ((e.a in ceiling) ^ (e.b in ceiling))]  # cut ceiling loose
    r = validate(P, C, E2)
    assert any("not connected to an anchor" in e for e in r.errors)
    # cosmetic floating chunks only warn
    C2 = [copy.copy(c) for c in C]
    for i in ceiling:
        C2[i].flags |= F_COSMETIC_ATTACHED
    r = validate(P, C2, E2)
    assert not any("not connected" in e for e in r.errors)
    assert any("cosmetic" in w for w in r.warnings)


def test_no_anchor_is_error(room):
    P, C, E = room
    C2 = [copy.copy(c) for c in C]
    for c in C2:
        c.flags &= ~F_ANCHOR
    assert any("no anchor" in e for e in validate(P, C2, E).errors)


def test_aspect_warning():
    poly = np.array([[0, 0], [5, 0], [5, 0.1], [0, 0.1]], float)
    P = [Panel(0, "beam", np.eye(4), box(0, 0, 5, 0.1), 0.2, "metal", {"anchor": True})]
    c = make_prism(poly, 0.2, np.eye(4), 7800, 0, "metal", flags=F_ANCHOR)
    r = validate(P, [c], [])
    assert r.ok and any("aspect" in w for w in r.warnings)


def _rand_frame(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    f = np.eye(4)
    f[:3, :3] = R
    f[:3, 3] = rng.uniform(-0.8, 0.8, 3)
    return f


def test_sat_matches_lp_on_random_prisms():
    """SAT (faces + edge cross products) agrees with an LP intersection test."""
    from scipy.optimize import linprog
    rng = np.random.default_rng(0)
    tri = np.array([[0, 0], [1, 0], [0.3, 0.8]], float)
    P = [Panel(0, "x", np.eye(4), box(0, 0, 1, 1), 1, "concrete", {}),
         Panel(1, "x", np.eye(4), box(0, 0, 1, 1), 1, "concrete", {})]
    checked = 0
    for _ in range(300):
        a = make_prism(tri, 0.4, _rand_frame(rng), 1, 0, "concrete", flags=F_ANCHOR)
        b = make_prism(tri, 0.4, _rand_frame(rng), 1, 1, "concrete")
        Pl = np.vstack([a.planes, b.planes])
        # max t s.t. n·x + d + t <= 0 for all planes
        res = linprog([0, 0, 0, -1], A_ub=np.column_stack([Pl[:, :3], np.ones(len(Pl))]),
                      b_ub=-Pl[:, 3], bounds=[(None, None)] * 3 + [(None, 1)])
        t = -res.fun
        if abs(t) < 2e-3:
            continue
        r = validate(P, [a, b], [])
        assert any("interpenetrating" in e for e in r.errors) == (t > 0)
        checked += 1
    assert checked > 200
