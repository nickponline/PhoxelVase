"""End-to-end generation (DESIGN §2.4–2.8).

    generate(spec, seed) -> GenResult(ctx, chunks, edges, bd, report, timings)
    build(spec_path, seed, out_root) -> Path   (writes assets/buildings/<name>_<seed>/...)
"""
from __future__ import annotations

import json
import os
import time
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .blockout import build_blockout
from .export.bld import assemble, write_bld
from .materials import DEFAULT_MATERIALS, F_ANCHOR, F_COSMETIC_ATTACHED, F_GLASS, F_INDESTRUCTIBLE
from .model import BuildingContext, BuildingData, ChunkGeom, Edge, Panel
from .nodes import get_node, get_reserve
from .nodes.openings import compute_connectivity
from .rng import node_rng, panel_rng, reserve_rng
from .spec import GENERATOR_VERSION, load_spec, resolve_spec

PARALLEL_MIN_PANELS = 200

# --------------------------------------------------------------------------- optional deps
try:
    from .geom.fracture import fracture_panel as _fracture_panel
except ImportError:  # pragma: no cover - fallback until geom.fracture lands
    _fracture_panel = None
try:
    from .graph import build_graph as _build_graph, mark_anchors as _mark_anchors
except ImportError:  # pragma: no cover
    _build_graph = _mark_anchors = None
try:
    from .validate import validate as _validate
except ImportError:  # pragma: no cover
    _validate = None


def _fallback_fracture(panel: Panel, cfg: dict, rng) -> list[ChunkGeom]:
    """No fracturing: one convex prism per convex part of the profile."""
    from .geom.prism import make_prism
    try:
        from .geom.convex2d import convex_decompose
        polys = convex_decompose(panel.profile)
    except ImportError:
        import mapbox_earcut as earcut
        ext = np.asarray(panel.profile.exterior.coords)[:-1]
        rings = [ext] + [np.asarray(r.coords)[:-1] for r in panel.profile.interiors]
        V = np.vstack(rings)
        ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
        tri = earcut.triangulate_float64(V, ends).reshape(-1, 3)
        polys = [V[t] for t in tri]
    mat = DEFAULT_MATERIALS[panel.material]
    fl = (F_GLASS if panel.material == "glass" else 0) | (F_ANCHOR if panel.tags.get("anchor") else 0) \
        | (F_INDESTRUCTIBLE if panel.tags.get("indestructible") else 0) \
        | (F_COSMETIC_ATTACHED if panel.tags.get("cosmetic") else 0)
    out = []
    for p in polys:
        c = make_prism(p, panel.thickness, panel.frame, mat.density, panel.id, panel.material, flags=fl)
        c.hp = max(mat.min_hp, mat.hp_per_m3 * c.volume)
        out.append(c)
    return out


def _fracture_batch(args):
    panels, cfgs, seed = args
    fn = _fracture_panel or _fallback_fracture
    out = []
    for p, cfg in zip(panels, cfgs):
        try:
            ch = fn(p, cfg, panel_rng(seed, p.id))
        except Exception as e:  # never lose the building to one bad panel: keep it unfractured
            ch = _fallback_fracture(p, cfg, None)
            for c in ch:
                c.__dict__["_fracture_error"] = f"panel {p.id} ({p.kind}): {type(e).__name__}: {e}"[:300]
        out.append(ch)
    return out


def fracture_all(panels: list[Panel], fracture_cfg: dict, seed: int, jobs: int | None = None) -> list[ChunkGeom]:
    """Fracture every panel with a per-panel RNG; returns chunks grouped by ascending panel id."""
    cfgs = [dict(fracture_cfg.get(p.material, {}) or {}) for p in panels]
    order = sorted(range(len(panels)), key=lambda i: panels[i].id)
    panels = [panels[i] for i in order]
    cfgs = [cfgs[i] for i in order]
    jobs = jobs if jobs is not None else min(8, os.cpu_count() or 1)
    if len(panels) > PARALLEL_MIN_PANELS and jobs > 1:
        # interleaved batches balance big slabs vs. small steps
        nb = jobs * 4
        idx = [list(range(b, len(panels), nb)) for b in range(nb)]
        batches = [([panels[i] for i in ix], [cfgs[i] for i in ix], seed) for ix in idx if ix]
        res: dict[int, list[ChunkGeom]] = {}
        try:
            ctx_mp = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else None
            with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx_mp) as ex:
                for ix, out in zip([ix for ix in idx if ix], ex.map(_fracture_batch, batches)):
                    for i, ch in zip(ix, out):
                        res[i] = ch
            per = [res[i] for i in range(len(panels))]
        except BrokenProcessPool:  # pragma: no cover - e.g. restricted environments
            per = _fracture_batch((panels, cfgs, seed))
    else:
        per = _fracture_batch((panels, cfgs, seed))
    chunks = [c for ch in per for c in ch]
    return chunks


def _fallback_graph(panels, chunks):
    return []


@dataclass
class GenResult:
    ctx: BuildingContext
    chunks: list[ChunkGeom]
    edges: list[Edge]
    bd: BuildingData
    report: dict
    timings: dict = field(default_factory=dict)


def _report_dict(rep) -> dict:
    if rep is None:
        return {"ok": True, "errors": [], "warnings": ["validate module unavailable"], "stats": {}}
    if isinstance(rep, dict):
        return rep
    return {"ok": bool(rep.ok), "errors": list(rep.errors), "warnings": list(rep.warnings),
            "stats": dict(rep.stats)}


def run_nodes(spec: dict, seed: int, timings: dict | None = None) -> BuildingContext:
    """spec (raw or resolved) -> BuildingContext with panels/rooms/sockets (no fracture)."""
    timings = {} if timings is None else timings
    rs = spec if spec.get("generator") == GENERATOR_VERSION and "global" in spec and seed == spec.get("seed") \
        else resolve_spec(spec, seed)
    rs = dict(rs)
    spec_dir = rs.pop("_dir", None)
    params = dict(rs["global"])
    ctx = BuildingContext(spec=rs, params=params)
    if spec_dir:
        ctx.meta["spec_dir"] = spec_dir
    t = time.perf_counter()
    ctx.blockout = build_blockout(rs["blockout"], params)
    timings["blockout"] = time.perf_counter() - t
    nodes = [(i, *next(iter(n.items()))) for i, n in enumerate(rs["nodes"])]
    for i, name, p in nodes:
        res = get_reserve(name)
        if res is not None:
            res(ctx, p or {}, reserve_rng(seed, i, name))
    for i, name, p in nodes:
        t = time.perf_counter()
        ctx = get_node(name)(ctx, p or {}, node_rng(seed, i, name)) or ctx
        timings[f"node.{name}"] = time.perf_counter() - t
    indestr = set(rs.get("indestructible", []))
    for p in ctx.panels:
        if p.kind in indestr or p.tags.get("role") in indestr:
            p.tags["indestructible"] = True
        assert np.linalg.det(p.frame[:3, :3]) > 0.999, f"panel {p.id} frame not a proper rotation"
    ctx.meta["room_connectivity"] = compute_connectivity(ctx)
    return ctx


def generate(spec: dict, seed: int, jobs: int | None = None, validate: bool = True,
             stability: bool = True) -> GenResult:
    timings: dict = {}
    t_all = time.perf_counter()
    ctx = run_nodes(spec, seed, timings)
    t = time.perf_counter()
    chunks = fracture_all(ctx.panels, ctx.spec.get("fracture", {}), seed, jobs)
    timings["fracture"] = time.perf_counter() - t
    t = time.perf_counter()
    if _mark_anchors is not None:
        _mark_anchors(ctx.panels, chunks)
    edges = (_build_graph or _fallback_graph)(ctx.panels, chunks)
    orphans = demote_orphan_slivers(chunks, edges)
    timings["graph"] = time.perf_counter() - t
    t = time.perf_counter()
    rep = _report_dict(_validate(ctx.panels, chunks, edges) if (validate and _validate) else None)
    ferr = sorted({c.__dict__["_fracture_error"] for c in chunks if "_fracture_error" in c.__dict__})
    if ferr:
        rep["warnings"].append(f"fracture failed on {len(ferr)} panel(s), kept unfractured: {ferr[:5]}")
    if orphans:
        rep["warnings"].append(f"{len(orphans)} tiny structural chunk(s) with only non-structural contacts "
                               f"demoted to cosmetic: {orphans[:10]}")
    conn = ctx.meta["room_connectivity"]
    rep["room_connectivity"] = conn
    if not conn["ok"]:
        rep["ok"] = False
        rep["errors"].append(f"rooms unreachable: {conn['unreachable'][:20]}")
    timings["validate"] = time.perf_counter() - t
    t = time.perf_counter()
    bd = assemble(ctx, chunks, edges, extra_meta=build_extra_meta(ctx, chunks, edges))
    timings["assemble"] = time.perf_counter() - t
    if stability:
        t = time.perf_counter()
        rep["stability"] = reinforce_until_stable(bd, **ctx.spec.get("stability", {}))
        st = rep["stability"]
        if not st.get("skipped") and not st["ok"]:
            rep["ok"] = False
            rep["errors"].append("static stability: " + "; ".join(st["reasons"]))
        timings["stability"] = time.perf_counter() - t
    timings["total"] = time.perf_counter() - t_all
    return GenResult(ctx, chunks, edges, bd, rep, {k: round(v, 4) if isinstance(v, float) else v
                                                    for k, v in timings.items()})


def demote_orphan_slivers(chunks: list[ChunkGeom], edges: list[Edge], max_volume: float = 0.01) -> list[int]:
    """Structural chunks whose only contacts are glass/cosmetic attachments can't carry load
    (the stress solver sees them as floating). Small ones become COSMETIC_ATTACHED so they
    detach with their neighbours; returns their ids."""
    nonstruct = F_GLASS | F_COSMETIC_ATTACHED
    has_struct = np.zeros(len(chunks), bool)
    for e in edges:
        if not (chunks[e.a].flags & nonstruct) and not (chunks[e.b].flags & nonstruct) and e.strength > 0:
            has_struct[e.a] = has_struct[e.b] = True
    out = []
    for i, c in enumerate(chunks):
        if not has_struct[i] and not (c.flags & (nonstruct | F_ANCHOR)) and c.volume < max_volume:
            c.flags |= F_COSMETIC_ATTACHED
            out.append(i)
    return out


def reinforce_until_stable(bd: BuildingData, max_util: float = 0.5, max_rounds: int = 8,
                           margin: float = 1.25, accept_util: float | None = None) -> dict:
    """DESIGN §2.7: solve the intact building with rubble's stress solver; while any edge is at or
    above `max_util`, 'add rebar' by scaling those edges' strength so they would sit at
    max_util/margin, then re-check. Mutates bd.edges['strength'] and records the edit in bd.meta."""
    import tempfile
    from .stability import available, check_stability
    if not available():
        return {"skipped": True, "reason": "rubble module not importable"}
    reinforced: set[int] = set()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "b.bld"
        for rnd in range(max_rounds + 1):
            write_bld(bd, path)
            res = check_stability(path, max_util=max_util, include_utilization=True)
            util = res.pop("utilization", None)
            if util is None or rnd == max_rounds or not res["overloaded_edges"]:
                break
            over = np.asarray(res["overloaded_edges"], dtype=np.int64)
            scale = np.maximum(util[over] / max_util * margin, 1.05)
            bd.edges["strength"][over] *= scale.astype(np.float32)
            reinforced.update(int(e) for e in over)
    if accept_util is not None and not res["ok"] and res["converged"] and not res["unsupported_nodes"] \
            and res["max_util"] < accept_util:
        res["ok"] = True
        res["reasons"] = []
        res["note"] = f"accepted: max_util {res['max_util']:.3f} < accept_util {accept_util} (target {max_util})"
    res.pop("overloaded_edges", None)
    res["worst_edges"] = res["worst_edges"][:5]
    res["reinforced_edges"] = len(reinforced)
    res["rounds"] = rnd
    bd.meta["reinforced_edges"] = len(reinforced)
    return res


def build_extra_meta(ctx, chunks, edges) -> dict:
    return {"generator": GENERATOR_VERSION,
            "sockets_count": len(ctx.sockets),
            "room_connectivity": ctx.meta.get("room_connectivity"),
            "stair_cores": [{k: v for k, v in c.items() if k in ("id", "axis", "outer", "inner", "floors")}
                            for c in ctx.meta.get("stair_cores", [])],
            "floor_z": list(ctx.blockout.floor_z)}


def out_dir_for(rs_name: str, seed: int, out_root) -> Path:
    return Path(out_root) / f"{rs_name}_{seed}"


def build(spec_path, seed: int | None = None, out_root="assets/buildings", jobs: int | None = None,
          render: bool = True, graph_json: bool = False, glb: bool = True) -> Path:
    from .export.manifest import write_manifest
    spec = load_spec(spec_path) if not isinstance(spec_path, dict) else dict(spec_path)
    seed = int(spec.get("seed", 0) if seed is None else seed)
    res = generate(spec, seed, jobs=jobs)
    out = out_dir_for(res.ctx.spec["name"], seed, out_root)
    out.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    write_bld(res.bd, out / "building.bld")
    res.timings["write_bld"] = round(time.perf_counter() - t, 4)
    if glb:
        from .export.gltf import write_glb
        t = time.perf_counter()
        write_glb(res.ctx, res.chunks, out / "building.glb")
        res.timings["write_glb"] = round(time.perf_counter() - t, 4)
    if graph_json:
        write_graph_json(res, out / "graph.json")
    if render:
        res.timings["render"] = render_dir(res.bd, out / "renders", out / "building.bld")
    write_manifest(res, out / "manifest.json")
    return out


def render_dir(bd, out: Path, bld_path: Path | None = None):
    util = None
    if bld_path is not None:
        from .stability import available, check_stability
        if available():
            util = check_stability(bld_path, include_utilization=True).get("utilization")
    try:
        from .render.views import render_all
    except ImportError as e:  # pragma: no cover
        warnings.warn(f"renderer unavailable: {e}")
        return None
    t = time.perf_counter()
    try:
        render_all(bd, out, edge_utilization=util)
    except Exception as e:  # renderer failures must not lose the building
        warnings.warn(f"render failed: {e!r}")
        return f"failed: {e!r}"
    return round(time.perf_counter() - t, 3)


def write_graph_json(res: GenResult, path: Path) -> None:
    data = {"chunks": [{"id": i, "panel": c.panel_id, "material": c.material, "flags": int(c.flags),
                        "com": [round(float(v), 4) for v in c.com], "mass": round(float(c.mass), 3)}
                       for i, c in enumerate(res.chunks)],
            "edges": [{"a": e.a, "b": e.b, "area": round(float(e.area), 5), "strength": float(e.strength)}
                      for e in res.edges]}
    Path(path).write_text(json.dumps(data))
