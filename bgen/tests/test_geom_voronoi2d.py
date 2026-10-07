import numpy as np
import pytest
import shapely

from bgen.geom.convex2d import is_convex, signed_area, to_shapely
from bgen.geom.voronoi2d import bounded_voronoi


@pytest.mark.parametrize("seed", range(10))
def test_cells_tile_box(seed):
    rng = np.random.default_rng(seed)
    b = (-1.0, 2.0, 7.0, 5.0)
    S = np.column_stack([rng.uniform(b[0], b[2], 80), rng.uniform(b[1], b[3], 80)])
    cells = bounded_voronoi(S, b)
    assert len(cells) == len(S)
    tot = 0.0
    for s, c in zip(S, cells):
        assert is_convex(c, 1e-7)
        assert shapely.Polygon(c).buffer(1e-9).contains(shapely.Point(s))
        tot += signed_area(c)
        assert (c >= np.array(b[:2]) - 1e-12).all() and (c <= np.array(b[2:]) + 1e-12).all()
    assert tot == pytest.approx(8.0 * 3.0, rel=1e-9)
    u = shapely.union_all(to_shapely(cells))
    assert u.area == pytest.approx(24.0, rel=1e-9)
    # nearest-seed property at random probes
    q = np.column_stack([rng.uniform(b[0], b[2], 50), rng.uniform(b[1], b[3], 50)])
    near = np.argmin(((q[:, None] - S[None]) ** 2).sum(-1), axis=1)
    for p, k in zip(q, near):
        assert shapely.Polygon(cells[k]).buffer(1e-9).contains(shapely.Point(p))


def test_degenerate_inputs():
    b = (0, 0, 2, 1)
    one = bounded_voronoi(np.array([[0.5, 0.5]]), b)
    assert signed_area(one[0]) == pytest.approx(2.0)
    two = bounded_voronoi(np.array([[0.5, 0.5], [1.5, 0.5]]), b)
    assert [signed_area(c) for c in two] == pytest.approx([1.0, 1.0])
    col = bounded_voronoi(np.array([[0.2, 0.5], [0.8, 0.5], [1.4, 0.5]]), b)  # collinear seeds
    assert sum(signed_area(c) for c in col) == pytest.approx(2.0)
    dup = bounded_voronoi(np.array([[0.5, 0.5], [0.5, 0.5], [1.5, 0.5]]), b)
    assert len(dup) == 3 and len(dup[1]) == 0
    assert sum(signed_area(c) for c in dup if len(c)) == pytest.approx(2.0)
    assert bounded_voronoi(np.zeros((0, 2)), b) == []


def test_deterministic():
    S = np.random.default_rng(0).uniform(0, 1, (50, 2))
    a, c = bounded_voronoi(S, (0, 0, 1, 1)), bounded_voronoi(S, (0, 0, 1, 1))
    assert all(np.array_equal(x, y) for x, y in zip(a, c))
