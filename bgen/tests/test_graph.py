"""Connection graph on a mini box room built with the junction rules."""
import time

import numpy as np
import pytest
from shapely.geometry import box

from bgen.geom.fracture import fracture_panel
from bgen.graph import ATTACH_STRENGTH, build_graph, mark_anchors
from bgen.materials import F_ANCHOR, F_GLASS, bond_strength
from bgen.model import Panel

EX, EY, EZ = np.eye(3)


def frame(ex, ey, origin):
    ex, ey = np.asarray(ex, float), np.asarray(ey, float)
    f = np.eye(4)
    f[:3, 0], f[:3, 1], f[:3, 2] = ex, ey, np.cross(ex, ey)
    f[:3, 3] = origin
    return f


def box_room(W=6.0, D=4.0, H=3.3, slab=0.3, wall=0.2, z0=0.0, pid0=0, glass=True, door=True):
    """Floor + ceiling slabs spanning the full footprint, 4 walls between them; X walls run
    through the corners, Y walls butt into them. South wall has a window with a glass
    pane, east wall a door."""
    ch = H - 2 * slab
    P = []

    def add(**kw):
        P.append(Panel(id=pid0 + len(P), **kw))
        return P[-1]

    add(kind="floor", frame=frame(EX, EY, (0, 0, z0)), profile=box(0, 0, W, D), thickness=slab,
        material="concrete", tags={"anchor": z0 == 0.0})
    add(kind="floor", frame=frame(EX, EY, (0, 0, z0 + H - slab)), profile=box(0, 0, W, D),
        thickness=slab, material="concrete", tags={})
    win = box(2.0, 1.0, 3.5, 2.2)
    add(kind="ext_wall", frame=frame(EX, EZ, (0, wall, z0 + slab)),
        profile=box(0, 0, W, ch).difference(win), thickness=wall, material="concrete", tags={})
    add(kind="ext_wall", frame=frame(EX, EZ, (0, D, z0 + slab)), profile=box(0, 0, W, ch),
        thickness=wall, material="brick", tags={})
    add(kind="ext_wall", frame=frame(EY, EZ, (0, wall, z0 + slab)), profile=box(0, 0, D - 2 * wall, ch),
        thickness=wall, material="concrete", tags={})
    east = box(0, 0, D - 2 * wall, ch)
    if door:
        east = east.difference(box(1.0, 0.0, 2.0, 2.1))
    add(kind="ext_wall", frame=frame(EY, EZ, (W - wall, wall, z0 + slab)), profile=east,
        thickness=wall, material="concrete", tags={})
    if glass:
        g = 0.02
        add(kind="glass", frame=frame(EX, EZ, (0, wall - (wall - g) / 2, z0 + slab)), profile=win,
            thickness=g, material="glass", tags={})
    return P


def fracture_all(panels, seed=0, cfg=None):
    chunks = []
    for p in panels:
        chunks += fracture_panel(p, cfg or {}, np.random.default_rng([seed, p.id]))
    return chunks


@pytest.fixture(scope="module")
def room():
    P = box_room()
    C = fracture_all(P)
    mark_anchors(P, C)
    E = build_graph(P, C)
    return P, C, E


def panel_pairs(C, E):
    return {tuple(sorted((C[e.a].panel_id, C[e.b].panel_id))) for e in E}


def test_inter_panel_contacts(room):
    P, C, E = room
    pp = panel_pairs(C, E)
    floor, ceil, south, north, west, east, glass = range(7)
    for w in (south, north, west, east):
        assert (floor, w) in pp and (ceil, w) in pp, w
    for xw in (south, north):
        for yw in (west, east):
            assert (xw, yw) in pp
    assert (south, glass) in pp
    assert (floor, ceil) not in pp and (south, north) not in pp and (west, east) not in pp
    # contact areas between a wall and the floor sum to the wall's footprint (exact with
    # min_contact=0; the default drops sub-0.01 m² slivers of contact)
    E = build_graph(P, C, min_contact=0.0)
    for w in (north, west):
        a = sum(e.area for e in E if {C[e.a].panel_id, C[e.b].panel_id} == {floor, w})
        p = P[w]
        assert a == pytest.approx(p.profile.bounds[2] * p.thickness, rel=1e-6)
    # west wall end face against the south wall: (D - 2*wall... height) * thickness
    a = sum(e.area for e in E if {C[e.a].panel_id, C[e.b].panel_id} == {south, west})
    assert a == pytest.approx(P[west].profile.bounds[3] * P[west].thickness, rel=1e-6)


def test_edge_invariants(room):
    P, C, E = room
    keys = [(e.a, e.b) for e in E]
    assert keys == sorted(set(keys))
    for e in E:
        assert e.a < e.b and e.area >= 0.0 and e.strength > 0
        assert np.linalg.norm(e.normal) == pytest.approx(1.0)
        # normal points from a to b
        assert (C[e.b].com - C[e.a].com) @ e.normal > 0
        ma, mb = C[e.a].material, C[e.b].material
        if (C[e.a].flags | C[e.b].flags) & F_GLASS:
            assert e.strength == pytest.approx(e.area * ATTACH_STRENGTH)
        else:
            assert e.area >= 0.01
            assert e.strength == pytest.approx(e.area * bond_strength(ma, mb))
        # contact centroid lies on both hulls (within tolerance)
        for c in (C[e.a], C[e.b]):
            assert (c.planes[:, :3] @ e.centroid + c.planes[:, 3]).max() < 1e-6


def test_intra_panel_areas(room):
    P, C, E = room
    # sum over a panel's intra edges of area == (total inner fracture face area)/2
    E = build_graph(P, C, min_contact=0.0)
    for p in P[:6]:
        idx = [i for i, c in enumerate(C) if c.panel_id == p.id]
        s = set(idx)
        intra = sum(e.area for e in E if e.a in s and e.b in s)
        inner = 0.0
        for i in idx:
            c = C[i]
            poly = c.poly2d
            fl = c.mesh_face_flags[::4][2:]  # 2 caps first (n verts each) -> compute directly below
            n = len(poly)
            side_flags = c.mesh_face_flags[2 * n::4]
            L = np.linalg.norm(np.roll(poly, -1, 0) - poly, axis=1)
            inner += (L * (side_flags == 2)).sum() * p.thickness
        assert intra == pytest.approx(inner / 2, rel=1e-6), p.id


def test_anchors(room):
    P, C, E = room
    for c in C:
        assert bool(c.flags & F_ANCHOR) == (c.panel_id == 0)
    # anchor tag on an elevated panel
    P2 = box_room(z0=10.0, glass=False)
    P2[3].tags["anchor"] = True
    C2 = fracture_all(P2)
    mark_anchors(P2, C2)
    assert all(bool(c.flags & F_ANCHOR) == (c.panel_id == 3) for c in C2)


def test_two_box_fixture_graph():
    from bgen.fixtures import two_box  # noqa: F401  (format fixture; rebuild panels here)
    P = [Panel(i, "floor", frame(EX, EY, (0, 0, float(i))), box(0, 0, 2, 1), 1.0, "concrete", {})
         for i in range(2)]
    C = fracture_all(P, cfg={"cell_size": 1.0})
    E = build_graph(P, C)
    mark_anchors(P, C)
    assert sum(e.area for e in E if C[e.a].panel_id != C[e.b].panel_id) == pytest.approx(2.0)
    assert all(c.flags & F_ANCHOR for c in C if c.panel_id == 0)


def test_deterministic(room):
    P, C, E = room
    C2 = fracture_all(P)
    E2 = build_graph(P, C2)
    assert [(e.a, e.b, e.area) for e in E] == [(e.a, e.b, e.area) for e in E2]


def big_building(nx=4, ny=4, floors=4):
    P = []
    for f in range(floors):
        for i in range(nx):
            for j in range(ny):
                # independent rooms stacked (shared slabs not needed for the perf test)
                rp = box_room(z0=f * 3.3, pid0=len(P))
                for p in rp:
                    p.frame = p.frame.copy()
                    p.frame[0, 3] += i * 6.0
                    p.frame[1, 3] += j * 4.0
                P += rp
    return P


def test_graph_perf_10k():
    P = big_building(3, 3, 3)
    C = []  # ~10k chunks with a finer cell size
    for p in P:
        C += fracture_panel(p, {"cell_size": 0.5}, np.random.default_rng(p.id))
    assert len(C) > 9000
    t = time.perf_counter()
    E = build_graph(P, C)
    dt = time.perf_counter() - t
    print(f"build_graph: {len(C)} chunks, {len(E)} edges, {dt:.2f}s")
    assert dt < 6.0
    assert len(E) > len(C)
