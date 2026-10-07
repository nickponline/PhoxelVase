"""bgen.sim / bgen.stability glue (needs the rubble extension: rubble/crates/rubble-py)."""
from pathlib import Path

import numpy as np
import pytest

from bgen import stability
from bgen.sim import read_transforms_bin, run_scenario

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "fixtures" / "two_box.bld"
needs_rubble = pytest.mark.skipif(not stability.available(), reason="rubble extension not installed")


@needs_rubble
def test_check_stability_two_box():
    r = stability.check_stability(FIX)
    assert not r["skipped"] and r["ok"] and r["max_util"] < 0.5 and r["converged"]
    r = stability.check_stability(FIX, max_util=1e-4)
    assert not r["ok"] and r["overloaded_edges"]


@needs_rubble
def test_run_scenario_two_box(tmp_path):
    sc = {
        "buildings": [{"path": str(FIX), "pos": [5, 0, 0], "yaw": 0.3}],
        "steps": 150, "dt": 1 / 60, "ground": 0.0,
        "actions": [{"t": 0.2, "damage": {"building": 0, "chunks": [0, 1], "amount": 1e9}}],
        "expect": {"collapse": True},
    }
    res = run_scenario(sc, tmp_path, render=True, every=10)
    assert res["passed"]
    b = res["buildings"][0]
    assert b["gone"] == 2 and b["fell"] == 2 and b["max_drop"] > 0.9
    tb = read_transforms_bin(tmp_path / "transforms.bin")
    assert tb["xf"].shape == (16, 4, 4, 4) and tb["every"] == 10
    assert tb["alive"][0].all() and tb["alive"][-1].tolist() == [False, False, True, True]
    # frame 0 = load pose (yaw 0.3 about z, then +5 in x)
    np.testing.assert_allclose(tb["xf"][0, 0, :3, 3], [5, 0, 0], atol=1e-5)
    assert (tmp_path / "sim.gif").exists() and (tmp_path / "contact_sheet.png").exists()
    assert res["render"]["frames"] == 16
