"""Render tests: both backends on the two-box fixture and a synthetic fractured building."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

from bgen.export.bld import read_bld  # noqa: E402
from bgen.fixtures import two_box  # noqa: E402
from bgen.render.raster_np import rasterize  # noqa: E402
from bgen.render.scene import build_mesh  # noqa: E402
from bgen.render.views import make_gif, render_all, render_frames, render_view  # noqa: E402
from synth_building import synth_building  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _gl_ok() -> bool:
    try:
        import moderngl
        moderngl.create_standalone_context(require=330).release()
        return True
    except Exception:
        return False


BACKENDS = ["numpy", pytest.param("gl", marks=pytest.mark.skipif(not _gl_ok(), reason="no GL context"))]


@pytest.fixture(scope="module")
def synth():
    return synth_building(floors=2, w=10, d=7, cell=1.2, seed=3)


def _check_image(p: Path, min_std=8.0):
    assert p.exists() and p.stat().st_size > 1000, p
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)
    assert a.std() > min_std, f"{p.name} looks blank"
    return a


def _expected(bd, util):
    n_floors = max(int(bd.elements["floor"].max()), max([r["floor"] for r in bd.meta["rooms"]] + [-1])) + 1
    names = {f"iso_{d}.png" for d in ("ne", "nw", "se", "sw")} | {
        "front.png", "side.png", "top.png", "fractured_iso.png", "exploded.png", "graph.png"}
    names |= {f"plan_floor_{i}.png" for i in range(n_floors)}
    if util:
        names.add("stress.png")
    return names


@pytest.mark.parametrize("backend", BACKENDS)
def test_render_all_two_box(tmp_path, backend):
    bd = read_bld(ROOT / "fixtures" / "two_box.bld")
    paths = render_all(bd, tmp_path, size=(320, 240), backend=backend)
    names = {p.name for p in paths}
    assert _expected(bd, False) <= names
    assert "stress.png" not in names
    for p in paths:
        _check_image(p, 4.0)


@pytest.mark.parametrize("backend", BACKENDS)
def test_render_all_synth(tmp_path, synth, backend):
    util = np.linspace(0, 1.2, len(synth.edges))
    tm = {}
    paths = render_all(synth, tmp_path, size=(400, 300), backend=backend, edge_utilization=util, timings=tm)
    assert tm["backend"] == backend
    names = {p.name for p in paths}
    assert _expected(synth, True) <= names
    assert {"cutaway_floor_0.png", "cutaway_floor_1.png"} <= names
    for p in paths:
        _check_image(p)
    # the fractured view must have many more distinct colors than the uniform one
    a = np.asarray(Image.open(tmp_path / "fractured_iso.png"))
    b = np.asarray(Image.open(tmp_path / "iso_ne.png"))
    assert len(np.unique(a.reshape(-1, 3) // 8, axis=0)) > 2 * len(np.unique(b.reshape(-1, 3) // 8, axis=0))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("view", ["iso_ne", "iso_sw", "front", "top", "oblique"])
def test_framing(synth, backend, view):
    img, mask, _ = render_view(synth, view, size=(320, 240), backend=backend)
    assert img.shape == (240, 320, 3)
    frac = mask.mean()
    assert 0.12 < frac < 0.9, f"{view}: building covers {frac:.2f} of frame"
    ys, xs = np.nonzero(mask)
    # centered-ish and not clipped by the border
    assert 0 < xs.min() and xs.max() < 319 and 0 < ys.min() and ys.max() < 239
    assert abs((xs.min() + xs.max()) / 2 - 160) < 40


def test_backend_parity(synth):
    if not _gl_ok():
        pytest.skip("no GL")
    _, m1, _ = render_view(synth, "iso_ne", size=(320, 240), backend="gl")
    _, m2, _ = render_view(synth, "iso_ne", size=(320, 240), backend="numpy")
    iou = (m1 & m2).sum() / (m1 | m2).sum()
    assert iou > 0.95


def test_two_box_fixture_matches_builder():
    bd = two_box()
    m = build_mesh(bd)
    assert m.n_chunks == 4 and len(m.tris) == 48
    assert (np.bincount(m.tchunk) == 12).all()


def test_rasterize_area():
    # right triangle with legs 100 px -> ~5000 px, and a full-screen quad covers everything
    xy = np.array([[10, 10], [110, 10], [10, 110]], float)
    z, tb = rasterize(xy, np.zeros(3), np.array([[0, 1, 2]]), 200, 200)
    assert abs((tb >= 0).sum() - 5000) < 150
    xy = np.array([[0, 0], [64, 0], [64, 48], [0, 48]], float)
    z, tb = rasterize(xy, np.array([0.1, 0.1, 0.5, 0.5]), np.array([[0, 1, 2], [0, 2, 3]]), 64, 48)
    assert (tb >= 0).all()
    assert z[0, 0] < z[-1, 0]  # depth interpolates (row 0 = top = y 0)


def _falling(bd, F=3):
    m = build_mesh(bd)
    N = m.n_chunks
    X = np.repeat(np.repeat(np.eye(4)[None, None], F, 0), N, 1)
    rng = np.random.default_rng(0)
    v = rng.normal(0, 1.5, (N, 3)); v[:, 2] = 0
    for f in range(F):
        t = f * 0.4
        drop = np.minimum(0.5 * 9.81 * t * t, m.aabb_min[:, 2])
        X[f, :, :3, 3] = v * t
        X[f, :, 2, 3] = -drop
        a = t * 0.8
        c, s = np.cos(a), np.sin(a)
        X[f, :, :3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        # rotate about the chunk com: x' = R(x - com) + com + offset
        X[f, :, :3, 3] += m.com - np.einsum("ij,nj->ni", X[f, 0, :3, :3], m.com)
    alive = np.ones((F, N), bool)
    alive[1:, ::7] = False
    return X, alive


@pytest.mark.parametrize("backend", BACKENDS)
def test_render_frames_gif(tmp_path, synth, backend):
    X, alive = _falling(synth)
    paths = render_frames(synth, X, tmp_path / "frames", camera="iso_ne", alive_mask=alive,
                          size=(320, 240), backend=backend)
    assert len(paths) == 3
    imgs = [_check_image(p) for p in paths]
    assert np.abs(imgs[0] - imgs[2]).mean() > 1.0  # frames differ
    gif = make_gif(paths, tmp_path / "seq.gif", fps=10)
    im = Image.open(gif)
    assert im.n_frames == 3


def test_render_frames_iterator(tmp_path, synth):
    X, _ = _falling(synth, F=2)
    paths = render_frames(synth, iter(list(X)), tmp_path, size=(160, 120), backend="numpy")
    assert len(paths) == 2 and all(p.exists() for p in paths)
