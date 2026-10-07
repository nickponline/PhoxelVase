"""Scripted destruction runs + renders through rubble-py (DESIGN.md §3.6 render-sim, §3.7).

    from bgen.sim import run_scenario
    res = run_scenario("scenarios/two_box_drop.yaml", "runs/two_box_drop")
    python -m bgen.sim scenarios/two_box_drop.yaml [--out DIR] [--no-render] [--every K]

Scenario YAML (same schema as rubble-sim, see rubble/crates/rubble-sim/README.md):

    buildings:
      - {path: fixtures/two_box.bld, pos: [0,0,0], yaw: 0}
    steps: 600
    dt: 0.016666
    ground: 0.0                      # null = no ground plane
    config: {stress: {bending: true}} # partial WorldConfig override
    actions:
      - {t: 0.5, explode: {center: [2,2,0.5], radius: 3, inner_radius: 0.5, damage: 800, impulse: 4000}}
      - {t: 1.0, fire: {origin: [-20,8,1.6], dir: [1,0,0], weapon: ar, count: 30, rate: 10, ballistic: false}}
      - {t: 2.0, damage: {building: 0, chunks: [0, 1], amount: 1.0e9}}
    record: {out: runs/x/, every: 2}  # frame every `every` steps (default: ~120 frames per run)
    render: {camera: iso_ne, size: [960, 720], fps: 30, backend: auto, cutaway_z: null}  # optional
    expect: {collapse: true}         # optional regression check (see `collapse_threshold`)

Paths are tried absolute, then relative to the scenario file, then to the cwd, then to the
repository root. Outputs in `out_dir`: events.jsonl, transforms.bin (rubble-sim layout),
summary.json, frames/frame_*.png, sim.gif, contact_sheet.png.

Multi-building rendering: all buildings are merged into one world-space RenderMesh (each
building's mesh is baked at its load pose), and per-frame chunk transforms are expressed
relative to that pose (X_f · X_0⁻¹), so one camera and one renderer cover the whole scene.
"""
from __future__ import annotations

import argparse
import json
import struct
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- transforms.bin

def read_transforms_bin(path) -> dict:
    """Read rubble-sim's `transforms.bin`.

    Returns dict(dt, every, buildings=[(chunk_offset, n_chunks, name)], ticks (F,) u32,
    times (F,) f32, xf (F,N,4,4) f32 building->world per chunk, alive (F,N) bool).
    """
    raw = Path(path).read_bytes()
    magic, ver, n_frames, n, n_bld, dt, every, _ = struct.unpack_from("<4sIIIIfII", raw, 0)
    if magic != b"RBTF" or ver != 1:
        raise ValueError(f"{path}: not a RBTF v1 file ({magic!r}, {ver})")
    off, blds = 32, []
    for _ in range(n_bld):
        c0, nc, ln = struct.unpack_from("<III", raw, off)
        off += 12
        blds.append((c0, nc, raw[off:off + ln].decode()))
        off += (ln + 3) // 4 * 4
    fsz = 8 + n * 65 + (4 - n % 4) % 4
    n_frames = min(n_frames, (len(raw) - off) // fsz) if fsz else 0  # tolerate truncated runs
    xf = np.empty((n_frames, n, 4, 4), np.float32)
    alive = np.empty((n_frames, n), bool)
    ticks = np.empty(n_frames, np.uint32)
    times = np.empty(n_frames, np.float32)
    for f in range(n_frames):
        ticks[f], times[f] = struct.unpack_from("<If", raw, off)
        xf[f] = np.frombuffer(raw, np.float32, n * 16, off + 8).reshape(n, 4, 4)
        alive[f] = np.frombuffer(raw, np.uint8, n, off + 8 + n * 64) != 0
        off += fsz
    return dict(dt=dt, every=every, buildings=blds, ticks=ticks, times=times, xf=xf, alive=alive)


def write_transforms_bin(path, xf: np.ndarray, alive: np.ndarray, ticks, times, buildings, dt: float,
                         every: int) -> Path:
    """Write the rubble-sim `transforms.bin` layout. `buildings` = [(chunk_offset, n_chunks, name)]."""
    F, n = alive.shape
    out = bytearray(struct.pack("<4sIIIIfII", b"RBTF", 1, F, n, len(buildings), dt, every, 0))
    for c0, nc, name in buildings:
        nb = name.encode()
        out += struct.pack("<III", c0, nc, len(nb)) + nb + b"\0" * ((4 - len(nb) % 4) % 4)
    pad = b"\0" * ((4 - n % 4) % 4)
    xf = np.ascontiguousarray(xf, np.float32)
    for f in range(F):
        out += struct.pack("<If", int(ticks[f]), float(times[f]))
        out += xf[f].tobytes() + np.asarray(alive[f], np.uint8).tobytes() + pad
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(bytes(out))
    return p


# --------------------------------------------------------------------------- scenario

def _resolve(p, base: Path | None) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p
    for root in ([base] if base else []) + [Path.cwd(), REPO_ROOT]:
        q = root / p
        if q.exists():
            return q.resolve()
    raise FileNotFoundError(f"{p} (tried scenario dir, cwd, {REPO_ROOT})")


def load_scenario(path_or_dict) -> tuple[dict, Path | None]:
    if isinstance(path_or_dict, dict):
        return path_or_dict, None
    p = Path(path_or_dict)
    return yaml.safe_load(p.read_text()), p.resolve().parent


def _schedule(actions: list[dict]) -> list[tuple[float, int, str, dict]]:
    """Expand actions to (t, seq, kind, payload); fire count/rate expands into single shots."""
    out = []
    for a in actions or []:
        t = float(a.get("t", 0.0))
        if "explode" in a:
            out.append((t, len(out), "explode", a["explode"]))
        if "fire" in a:
            f = a["fire"]
            count, rate = int(f.get("count", 1)), float(f.get("rate", 10.0))
            for i in range(count):
                out.append((t + i / max(rate, 1e-3), len(out), "fire", f))
        if "damage" in a:
            out.append((t, len(out), "damage", a["damage"]))
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def _apply(w, kind: str, p: dict):
    if kind == "explode":
        w.explode(tuple(p["center"]), float(p.get("radius", 4.0)), damage=float(p.get("damage", 400.0)),
                  impulse=float(p.get("impulse", 3000.0)), inner_radius=float(p.get("inner_radius", 1.0)))
    elif kind == "fire":
        w.fire(tuple(p["origin"]), tuple(p["dir"]), weapon=p.get("weapon", "ar"),
               ballistic=bool(p.get("ballistic", False)))
    elif kind == "damage":
        chunks = p.get("chunks", p.get("chunk"))
        w.damage_chunk(int(p.get("building", 0)), chunks, float(p.get("amount", 1.0e9)))
    else:
        raise ValueError(kind)


def simulate(sc: dict, base: Path | None = None, every: int | None = None) -> dict:
    """Run the scenario in rubble; returns recorded arrays and metadata (no rendering)."""
    import rubble

    w = rubble.World(sc.get("config"))
    ground = sc.get("ground", 0.0)
    if ground is not None:
        w.add_ground_plane(float(ground))
    blds, paths = [], []
    for spec in sc["buildings"]:
        if "path" not in spec:
            raise ValueError(f"only .bld buildings are supported by bgen.sim (got {spec})")
        p = _resolve(spec["path"], base)
        b = w.load_building(str(p), pos=tuple(spec.get("pos", (0, 0, 0))), yaw=float(spec.get("yaw", 0.0)))
        blds.append(b)
        paths.append(p)
    counts = [w.n_chunks(b) for b in blds]
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(int)
    steps = int(sc.get("steps", 600))
    dt = float(sc.get("dt", 1 / 60))
    rec = sc.get("record") or {}
    every = int(every or rec.get("every") or max(1, round(steps / 120)))
    sched = _schedule(sc.get("actions"))

    def grab():
        return (np.concatenate([w.chunk_world_transforms(b) for b in blds]),
                np.concatenate([w.alive(b) for b in blds]),
                np.concatenate([w.chunk_state(b) for b in blds]))

    X, A, S = grab()
    xs, al, st, ticks, times = [X], [A], [S], [0], [0.0]
    events, k, t0 = [], 0, time.perf_counter()
    tick_ms = []
    for step in range(steps):
        while k < len(sched) and sched[k][0] <= w.time + 1e-6:
            _apply(w, sched[k][2], sched[k][3])
            k += 1
        w.step(dt)
        tick_ms.append(w.stats()["timings"]["total_ms"])
        for e in w.drain_events():
            e["tick"], e["time"] = w.tick - 1, w.time - dt
            events.append(e)
        if (step + 1) % every == 0:
            X, A, S = grab()
            xs.append(X); al.append(A); st.append(S); ticks.append(w.tick); times.append(w.time)
    wall = time.perf_counter() - t0
    return dict(
        xf=np.stack(xs), alive=np.stack(al), state=np.stack(st), ticks=np.array(ticks, np.uint32),
        times=np.array(times, np.float32), events=events, paths=paths, offsets=offs, dt=dt, every=every,
        steps=steps, first_action_t=sched[0][0] if sched else None, stats=w.stats(), wall_s=wall,
        tick_ms=np.array(tick_ms, np.float32),
        utilization=[w.edge_utilization(b) for b in blds],
    )


# --------------------------------------------------------------------------- analysis

def _outcome(run: dict, bds: list, drop_m: float = 1.0) -> list[dict]:
    """Per building: how much structural mass fell (> drop_m below its start) or vanished."""
    out = []
    xf, alive = run["xf"], run["alive"]
    for bi, bd in enumerate(bds):
        a, b = run["offsets"][bi], run["offsets"][bi + 1]
        com = bd.chunks["com"].astype(np.float64)
        mass = bd.chunks["mass"].astype(np.float64)
        structural = (bd.chunks["flags"] & (4 | 8)) == 0  # not GLASS / COSMETIC_ATTACHED
        p0 = np.einsum("nij,nj->ni", xf[0, a:b, :3, :3], com) + xf[0, a:b, :3, 3]
        p1 = np.einsum("nij,nj->ni", xf[-1, a:b, :3, :3], com) + xf[-1, a:b, :3, 3]
        dz = p0[:, 2] - p1[:, 2]
        gone = ~alive[-1, a:b]
        fell = (dz > drop_m) & ~gone
        m = mass * structural
        tot = max(m.sum(), 1e-9)
        out.append(dict(
            path=str(run["paths"][bi]), n_chunks=int(b - a),
            gone=int(gone.sum()), fell=int(fell.sum()),
            fell_mass_frac=float(m[fell].sum() / tot), gone_mass_frac=float(m[gone].sum() / tot),
            lost_mass_frac=float(m[fell | gone].sum() / tot),
            max_drop=float(dz[~gone].max()) if (~gone).any() else 0.0,
            final_states={n: int(c) for n, c in zip(("static", "detaching", "in_cluster", "frozen", "gone"),
                                                    np.bincount(run["state"][-1, a:b], minlength=5))},
        ))
    return out


def _key_frames(run: dict, n: int = 6) -> list[int]:
    """Frame 0, the first action, then frames at 1/3, 2/3 and 100% of cumulative motion, plus the last."""
    xf, F = run["xf"], len(run["xf"])
    if F <= n:
        return list(range(F))
    d = np.abs(np.diff(xf[:, :, :3, 3], axis=0)).sum(axis=(1, 2))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    picks = [0]
    if run["first_action_t"] is not None:
        picks.append(int(np.searchsorted(run["times"], run["first_action_t"] + 1e-6)))
    if cum[-1] > 0:
        for q in (0.15, 0.4, 0.75, 0.98):
            picks.append(int(np.searchsorted(cum, q * cum[-1])))
    picks.append(F - 1)
    picks = sorted(set(min(max(p, 0), F - 1) for p in picks))
    while len(picks) < n:  # fill the largest gaps
        gaps = np.diff(picks)
        i = int(np.argmax(gaps))
        if gaps[i] < 2:
            break
        picks.insert(i + 1, picks[i] + gaps[i] // 2)
    while len(picks) > n:  # drop the frame closest to a neighbour (never the first/last)
        gaps = np.diff(picks)
        i = int(np.argmin(gaps[1:])) + 1 if len(gaps) > 1 else 1
        picks.pop(min(i, len(picks) - 2))
    return picks


def contact_sheet(paths, labels, out_path, cols: int = 3, max_width: int = 480):
    from PIL import Image, ImageDraw

    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        ims.append(im)
    w, h = ims[0].size
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    for i, (im, lab) in enumerate(zip(ims, labels)):
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(im, (x, y))
        d = ImageDraw.Draw(sheet)
        d.rectangle([x + 4, y + 4, x + 8 + 7 * len(lab), y + 20], fill=(0, 0, 0))
        d.text((x + 6, y + 6), lab, fill=(255, 255, 255))
    out = Path(out_path)
    sheet.save(out)
    return out


# --------------------------------------------------------------------------- rendering

def combined_mesh(bds: list, xf0: np.ndarray, offsets: np.ndarray):
    """Merge buildings into one world-space RenderMesh baked at the frame-0 chunk transforms."""
    from .render.scene import build_mesh

    parts = [build_mesh(bd) for bd in bds]
    pos, nrm, flags, vch, tris, tch = [], [], [], [], [], []
    com, lo, hi, mat, cfl, elem, ekind, efloor, mass = [], [], [], [], [], [], [], [], []
    voff, eoff = 0, 0
    for bi, m in enumerate(parts):
        c0 = int(offsets[bi])
        X = xf0[c0:c0 + m.n_chunks].astype(np.float64)
        Xv = X[m.vchunk] if m.n_chunks else np.zeros((0, 4, 4))
        pos.append(np.einsum("vij,vj->vi", Xv[:, :3, :3], m.pos) + Xv[:, :3, 3])
        nrm.append(np.einsum("vij,vj->vi", Xv[:, :3, :3], m.nrm))
        flags.append(m.flags); vch.append(m.vchunk + c0); tris.append(m.tris + voff); tch.append(m.tchunk + c0)
        com.append(np.einsum("nij,nj->ni", X[:, :3, :3], m.com) + X[:, :3, 3])
        c, e = (m.aabb_min + m.aabb_max) / 2, (m.aabb_max - m.aabb_min) / 2
        wc = np.einsum("nij,nj->ni", X[:, :3, :3], c) + X[:, :3, 3]
        we = np.einsum("nij,nj->ni", np.abs(X[:, :3, :3]), e)
        lo.append(wc - we); hi.append(wc + we)
        mat.append(m.material); cfl.append(m.chunk_flags); elem.append(m.elem + eoff)
        ekind.append(m.elem_kind); efloor.append(m.elem_floor); mass.append(m.mass)
        voff += len(m.pos)
        eoff += int(m.elem.max()) + 1 if m.n_chunks else 0
    cat = np.concatenate
    return replace(parts[0], pos=cat(pos).astype(np.float32), nrm=cat(nrm).astype(np.float32), flags=cat(flags),
                   vchunk=cat(vch).astype(np.int32), tris=cat(tris).astype(np.uint32),
                   tchunk=cat(tch).astype(np.int32), n_chunks=int(offsets[-1]), com=cat(com), aabb_min=cat(lo),
                   aabb_max=cat(hi), material=cat(mat), chunk_flags=cat(cfl), elem=cat(elem),
                   elem_kind=cat(ekind), elem_floor=cat(efloor), mass=cat(mass))


def render_run(run: dict, bds: list, out_dir: Path, camera: str = "iso_ne", size=(960, 720), fps: int = 30,
               backend: str = "auto", cutaway_z: float | None = None) -> dict:
    """Render all recorded frames + gif + contact sheet. `cutaway_z`: hide chunks whose initial
    world AABB starts at or above this height (e.g. the roof) to look inside."""
    from .render.views import make_gif, render_frames

    xf = run["xf"].astype(np.float64)
    mesh = combined_mesh(bds, xf[0], run["offsets"])
    rel = (xf @ np.linalg.inv(xf[0])[None]).astype(np.float32)
    vis = run["alive"]
    if cutaway_z is not None:
        vis = vis & (mesh.aabb_min[:, 2] < float(cutaway_z))[None]
    frames = render_frames(mesh, rel, out_dir / "frames", camera=camera, alive_mask=vis, size=size,
                           backend=backend)
    gif = make_gif(frames, out_dir / "sim.gif", fps=fps)
    keys = _key_frames(run)
    sheet = contact_sheet([frames[i] for i in keys], [f"t={run['times'][i]:.2f}s" for i in keys],
                          out_dir / "contact_sheet.png")
    return dict(frames=len(frames), gif=str(gif), contact_sheet=str(sheet), key_frames=keys)


# --------------------------------------------------------------------------- entry point

def run_scenario(scenario_yaml, out_dir=None, render: bool = True, every: int | None = None,
                 collapse_threshold: float = 0.25) -> dict:
    """Run a scenario through rubble-py, record per-frame transforms/alive, render, summarize.

    A building counts as *collapsed* when ≥ `collapse_threshold` of its structural mass ends the
    run fallen (> 1 m below its start) or destroyed. If the scenario has `expect: {collapse: bool}`
    the result's `passed` reports whether every building matched.
    """
    from .export.bld import read_bld

    sc, base = load_scenario(scenario_yaml)
    rec = sc.get("record") or {}
    out = Path(out_dir or rec.get("out") or "runs/" + (Path(str(scenario_yaml)).stem if base else "scenario"))
    out.mkdir(parents=True, exist_ok=True)
    run = simulate(sc, base, every)
    bds = [read_bld(p) for p in run["paths"]]

    with open(out / "events.jsonl", "w") as f:
        for e in run["events"]:
            f.write(json.dumps(e) + "\n")
    names = [(int(run["offsets"][i]), int(run["offsets"][i + 1] - run["offsets"][i]), str(p))
             for i, p in enumerate(run["paths"])]
    write_transforms_bin(out / "transforms.bin", run["xf"], run["alive"], run["ticks"], run["times"], names,
                         run["dt"], run["every"])

    ev_counts: dict[str, int] = {}
    for e in run["events"]:
        ev_counts[e["type"]] = ev_counts.get(e["type"], 0) + 1
    outcome = _outcome(run, bds)
    for o in outcome:
        o["collapsed"] = o["lost_mass_frac"] >= collapse_threshold
    tm = run["tick_ms"]
    res = dict(
        scenario=str(scenario_yaml) if base else "<dict>", out_dir=str(out), steps=run["steps"], dt=run["dt"],
        every=run["every"], frames=int(len(run["xf"])), sim_wall_s=round(run["wall_s"], 3),
        tick_ms=dict(avg=float(tm.mean()) if len(tm) else 0.0, max=float(tm.max()) if len(tm) else 0.0),
        events=ev_counts, final_stats=run["stats"], buildings=outcome,
        max_edge_utilization=[float(u.max()) if len(u) else 0.0 for u in run["utilization"]],
    )
    exp = sc.get("expect")
    if exp and "collapse" in exp:
        res["expect"] = exp
        res["passed"] = all(o["collapsed"] == bool(exp["collapse"]) for o in outcome)
    if render:
        r = sc.get("render") or {}
        t0 = time.perf_counter()
        res["render"] = render_run(run, bds, out, camera=r.get("camera", "iso_ne"),
                                   size=tuple(r.get("size", (960, 720))),
                                   fps=int(r.get("fps", max(1, round(1 / (run["dt"] * run["every"]))))),
                                   backend=r.get("backend", "auto"), cutaway_z=r.get("cutaway_z"))
        res["render"]["wall_s"] = round(time.perf_counter() - t0, 2)
    (out / "summary.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bgen.sim", description=__doc__.split("\n")[0])
    ap.add_argument("scenario")
    ap.add_argument("--out")
    ap.add_argument("--every", type=int)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    res = run_scenario(a.scenario, a.out, render=not a.no_render, every=a.every)
    print(json.dumps({k: res[k] for k in ("out_dir", "frames", "events", "buildings") if k in res}
                     | {k: res[k] for k in ("passed", "render") if k in res}, indent=2, default=str))
    return 0 if res.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
