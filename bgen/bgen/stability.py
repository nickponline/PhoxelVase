"""Static stability check through rubble's stress solver.

    check_stability("assets/buildings/x/building.bld", max_util=0.5)
      -> {'skipped': False, 'ok': True, 'max_util': 0.12, 'worst_edges': [...], ...}

Uses the `rubble` extension module (rubble/crates/rubble-py, `maturin develop --release`).
If it is not importable the check is skipped: {'skipped': True, 'reason': ...}.
"""
from __future__ import annotations

from pathlib import Path

try:  # optional native dependency
    import rubble as _rubble
except Exception as _e:  # pragma: no cover - depends on the environment
    _rubble = None
    _IMPORT_ERROR = repr(_e)
else:
    _IMPORT_ERROR = None


def available() -> bool:
    return _rubble is not None


def check_stability(bld_path, max_util: float = 0.5, bending: bool = True, bend_scale: float = 1.0,
                    include_utilization: bool = False) -> dict:
    """Solve the undamaged building statically and judge it.

    ok = solver converged, no unsupported (floating) structural chunks, and every edge's
    utilization < `max_util`. `worst_edges` is a list of (edge, utilization, chunk_a, chunk_b),
    worst first; `overloaded_edges` lists every edge at or above `max_util`. The per-edge
    utilization ndarray is included only with `include_utilization=True`.
    """
    if _rubble is None:
        return {"skipped": True, "reason": f"rubble module not importable: {_IMPORT_ERROR}"}
    path = str(Path(bld_path))
    rep = _rubble.static_stress_report(path, bending=bending, bend_scale=bend_scale)
    util = rep["utilization"]
    over = [int(e) for e in (util >= max_util).nonzero()[0]]
    over.sort(key=lambda e: -float(util[e]))
    reasons = []
    if not rep["converged"]:
        reasons.append("stress solve did not converge")
    if rep["unsupported_nodes"]:
        reasons.append(f"{len(rep['unsupported_nodes'])} structural chunks not connected to an anchor")
    if over:
        reasons.append(f"{len(over)} edges with utilization >= {max_util} (max {rep['max_util']:.3f})")
    out = {
        "skipped": False,
        "ok": not reasons,
        "reasons": reasons,
        "max_util": float(rep["max_util"]),
        "threshold": max_util,
        "worst_edges": [(int(e), float(u), int(a), int(b)) for e, u, a, b in rep["worst_edges"]],
        "overloaded_edges": over,
        "unsupported_nodes": list(rep["unsupported_nodes"]),
        "converged": bool(rep["converged"]),
        "iters": int(rep["iters"]),
        "total_load": float(rep["total_load"]),
        "anchor_flow": float(rep["anchor_flow"]),
        "bending": bending,
    }
    if include_utilization:
        out["utilization"] = util
    return out


__all__ = ["check_stability", "available"]
