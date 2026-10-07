"""convex_decompose / polygon helpers: property tests over random polygons with holes."""
import numpy as np
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from bgen.geom.convex2d import (bbox_pairs, clean_ring, convex_decompose, convex_hull, is_convex,
                                poly_aspect, polygon_adjacency, signed_area, to_shapely)


def random_polygon(rng, n_holes=None):
    """Random star-shaped polygon with up to 3 random (box/circle/triangle) holes."""
    n = rng.integers(5, 16)
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(2.0, 5.0, n)
    poly = Polygon(np.column_stack([r * np.cos(ang), r * np.sin(ang)]))
    if not poly.is_valid:
        poly = shapely.make_valid(poly)
    k = rng.integers(0, 4) if n_holes is None else n_holes
    for _ in range(k):
        c = rng.uniform(-1.2, 1.2, 2)
        t = rng.integers(3)
        if t == 0:
            h = box(c[0] - .3, c[1] - .2, c[0] + .25, c[1] + .35)
        elif t == 1:
            h = Point(c).buffer(rng.uniform(.1, .4), 6)
        else:
            h = Polygon(c + rng.uniform(-.4, .4, (3, 2)))
        if h.is_valid and h.area > 1e-3:
            poly = poly.difference(h)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly


def check_tiling(poly, parts):
    assert parts
    for p in parts:
        assert len(p) >= 3
        assert is_convex(p, 1e-7), p
        assert signed_area(p) > 0
    area = sum(signed_area(p) for p in parts)
    assert area == pytest.approx(poly.area, rel=1e-9)
    u = shapely.union_all(to_shapely(parts))
    assert u.symmetric_difference(poly).area < 1e-9 * poly.area
    # no overlaps: union area equals sum of areas
    assert u.area == pytest.approx(area, rel=1e-9)


@pytest.mark.parametrize("seed", range(60))
def test_random_polygons_with_holes(seed):
    rng = np.random.default_rng(seed)
    poly = random_polygon(rng)
    parts = convex_decompose(poly)
    check_tiling(poly, parts)


def test_simple_shapes():
    assert len(convex_decompose(box(0, 0, 3, 1))) == 1
    L = box(0, 0, 3, 1).union(box(0, 0, 1, 3))
    parts = convex_decompose(L)
    check_tiling(L, parts)
    assert len(parts) == 2
    U = box(0, 0, 3, 3).difference(box(1, 1, 2, 3))
    check_tiling(U, convex_decompose(U))
    # wall with windows and a door touching the border
    w = box(0, 0, 10, 3)
    for x in range(1, 9, 3):
        w = w.difference(box(x, 1, x + 1.2, 2.2))
    w = w.difference(box(8.5, 0, 9.5, 2.1))
    parts = convex_decompose(w)
    check_tiling(w, parts)
    # HM bound: no worse than 4x optimal; for this wall a handful of parts
    assert len(parts) <= 20


def test_collinear_and_cw_input():
    p = Polygon([(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (0, 2)][::-1])  # CW, collinear verts
    parts = convex_decompose(p)
    assert len(parts) == 1 and len(parts[0]) == 4
    check_tiling(p, parts)


def test_multipolygon():
    mp = box(0, 0, 1, 1).union(box(3, 0, 4, 1))
    parts = convex_decompose(mp)
    assert len(parts) == 2


def test_deterministic():
    poly = random_polygon(np.random.default_rng(3), n_holes=3)
    a, b = convex_decompose(poly), convex_decompose(poly)
    assert all(np.array_equal(x, y) for x, y in zip(a, b)) and len(a) == len(b)


def test_clean_ring_and_hull():
    r = clean_ring(np.array([[0, 0], [0.5, 0], [1, 0], [1, 1], [1, 1], [0, 1], [0, 0]], float))
    assert len(r) == 4
    pts = np.random.default_rng(0).uniform(0, 1, (200, 2))
    h = convex_hull(pts)
    assert is_convex(h)
    assert signed_area(h) == pytest.approx(shapely.MultiPoint(pts).convex_hull.area)
    assert poly_aspect(np.array([[0, 0], [10, 0], [10, 1], [0, 1]], float)) == pytest.approx(np.hypot(10, 1))


def test_polygon_adjacency_tjunction():
    a = np.array([[0, 0], [1, 0], [1, 2], [0, 2]], float)
    b = np.array([[1, 0], [2, 0], [2, 1], [1, 1]], float)       # T-junction on a's right edge
    c = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], float)
    d = np.array([[5, 5], [6, 5], [6, 6]], float)
    I, J, L, M, N = polygon_adjacency([a, b, c, d])
    got = {(i, j): (l, m, n) for i, j, l, m, n in zip(I, J, L, M, N)}
    assert set(got) == {(0, 1), (0, 2), (1, 2)}
    l, m, n = got[(0, 1)]
    assert l == pytest.approx(1) and np.allclose(m, [1, .5]) and np.allclose(n, [1, 0])
    l, m, n = got[(1, 2)]
    assert l == pytest.approx(1) and np.allclose(n, [0, 1])
    # groups separate otherwise-touching polygons
    I, J, *_ = polygon_adjacency([a, b, c], groups=np.array([0, 1, 1]))
    assert list(zip(I, J)) == [(1, 2)]


def test_bbox_pairs_bruteforce():
    rng = np.random.default_rng(1)
    lo = rng.uniform(0, 10, (300, 3))
    hi = lo + rng.uniform(0.1, 1.5, (300, 3))
    A, B = bbox_pairs(lo, hi)
    ov = ((lo[:, None] <= hi[None]) & (lo[None] <= hi[:, None])).all(-1)
    iu = np.triu_indices(300, 1)
    want = set(zip(iu[0][ov[iu]], iu[1][ov[iu]]))
    assert set(zip(A.tolist(), B.tolist())) == want
