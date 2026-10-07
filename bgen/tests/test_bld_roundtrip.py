import numpy as np

from bgen.export.bld import read_bld, write_bld, bitset_to_mask
from bgen.fixtures import two_box


def test_two_box_roundtrip(tmp_path):
    bd = two_box()
    p = tmp_path / "two_box.bld"
    write_bld(bd, p)
    rd = read_bld(p)
    assert rd.meta["name"] == "two_box"
    for f in ["elements", "chunks", "mesh_verts", "edges", "sockets"]:
        assert getattr(rd, f).tobytes() == getattr(bd, f).tobytes(), f
    np.testing.assert_array_equal(rd.hull_verts, bd.hull_verts.astype(np.float32))
    assert list(bitset_to_mask(rd.anchors, 4)) == [True, True, False, False]
    c = rd.chunks
    np.testing.assert_allclose(c["volume"], 1.0, rtol=1e-6)
    np.testing.assert_allclose(c["mass"], 2400.0, rtol=1e-6)
    np.testing.assert_allclose(c["com"][:, 2], [.5, .5, 1.5, 1.5], rtol=1e-6)
    # cube inertia m*(a²+b²)/12 = 400
    np.testing.assert_allclose(c["inertia"][:, :3], 400.0, rtol=1e-5)
    np.testing.assert_allclose(c["inertia"][:, 3:], 0.0, atol=1e-3)
    assert len(rd.edges) == 4 and c["mesh_i_cnt"][0] == 36
    # hull verts satisfy own planes
    for i in range(4):
        v = rd.hull_verts[c["hull_v_off"][i]:c["hull_v_off"][i] + c["hull_v_cnt"][i]]
        P = rd.hull_planes[c["hull_p_off"][i]:c["hull_p_off"][i] + c["hull_p_cnt"][i]]
        assert (v @ P[:, :3].T + P[:, 3] <= 1e-5).all()
