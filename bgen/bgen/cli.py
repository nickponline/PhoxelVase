"""`bgen` command line.

  bgen build specs/office.yaml --seed 1234 --out assets/buildings/
  bgen batch specs/ --seeds 1..20 --jobs 8
  bgen render assets/buildings/office_1234
  bgen validate assets/buildings/office_1234
  bgen district --grid 4x4 --seed 7          (not implemented yet)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_seeds(s: str) -> list[int]:
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        if ".." in part:
            a, b = part.split("..")
            out += list(range(int(a), int(b) + 1))
        elif "-" in part[1:]:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _spec_files(paths: list[str]) -> list[Path]:
    files = []
    for p in paths:
        P = Path(p)
        if P.is_dir():
            files += sorted(q for q in P.glob("*.yaml") if not q.name.endswith(".overrides.yaml"))
        else:
            files.append(P)
    return files


def _summary(out: Path) -> dict:
    m = json.loads((out / "manifest.json").read_text())
    s = m["stats"]
    return dict(out=str(out), panels=s["panels"], chunks=s["chunks"], edges=s["edges"],
                ok=m["validation"]["ok"], total_s=m["timings"].get("total"))


def cmd_build(a) -> int:
    from .pipeline import build
    t = time.perf_counter()
    out = build(a.spec, a.seed, a.out, jobs=a.jobs, render=not a.no_render, graph_json=a.graph_json,
                glb=not a.no_glb)
    s = _summary(out)
    m = json.loads((out / "manifest.json").read_text())
    print(f"{out}: panels={s['panels']} chunks={s['chunks']} edges={s['edges']} "
          f"valid={s['ok']} gen={s['total_s']}s wall={time.perf_counter() - t:.1f}s")
    for e in m["validation"]["errors"]:
        print("  ERROR", e)
    for w in m["validation"]["warnings"]:
        print("  warn ", w)
    return 0 if s["ok"] else 2


def _batch_one(args):
    spec, seed, out, render, glb = args
    from .pipeline import build
    try:
        o = build(spec, seed, out, jobs=1, render=render, glb=glb)
        return _summary(o)
    except Exception as e:  # report and continue with the rest of the batch
        return dict(out=f"{spec}@{seed}", ok=False, error=f"{type(e).__name__}: {e}")


def cmd_batch(a) -> int:
    jobs = [(str(f), s, a.out, not a.no_render, not a.no_glb)
            for f in _spec_files(a.specs) for s in parse_seeds(a.seeds)]
    n_ok = 0
    t = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.jobs or os.cpu_count()) as ex:
        futs = [ex.submit(_batch_one, j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            n_ok += bool(r.get("ok"))
            print(json.dumps(r), flush=True)
    print(f"{n_ok}/{len(jobs)} valid in {time.perf_counter() - t:.1f}s")
    return 0 if n_ok == len(jobs) else 2


def cmd_render(a) -> int:
    from .export.bld import read_bld
    from .pipeline import render_dir
    d = Path(a.dir)
    bd = read_bld(d / "building.bld")
    r = render_dir(bd, d / "renders")
    print(f"rendered {d / 'renders'} ({r})")
    return 0 if not (isinstance(r, str) or r is None) else 1


def cmd_validate(a) -> int:
    """Re-derive the building from manifest (spec + seed), check it reproduces building.bld
    byte-for-byte, and print the validation report."""
    import tempfile

    from .export.bld import read_bld, write_bld
    from .pipeline import generate
    d = Path(a.dir)
    m = json.loads((d / "manifest.json").read_text())
    rd = read_bld(d / "building.bld")          # header/hash check
    spec = dict(m["spec"])
    if m.get("spec_dir"):
        spec["_dir"] = m["spec_dir"]
    res = generate(spec, int(m["seed"]), jobs=a.jobs)
    with tempfile.TemporaryDirectory() as td:
        write_bld(res.bd, Path(td) / "b.bld")
        same = (Path(td) / "b.bld").read_bytes() == (d / "building.bld").read_bytes()
    rep = res.report
    print(json.dumps({"bld_reproducible": same, "ok": rep["ok"], "errors": rep["errors"],
                      "warnings": rep["warnings"], "room_connectivity": rep.get("room_connectivity"),
                      "chunks": int(len(rd.chunks)), "edges": int(len(rd.edges))}, indent=1, default=str))
    return 0 if (rep["ok"] and same) else 2


def cmd_district(a) -> int:
    print("bgen district: not implemented", file=sys.stderr)
    return 3


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bgen", description="procedural pre-fractured building generator")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="generate one building")
    b.add_argument("spec")
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--out", default="assets/buildings")
    b.add_argument("--jobs", type=int, default=None, help="fracture worker processes")
    b.add_argument("--no-render", action="store_true")
    b.add_argument("--no-glb", action="store_true")
    b.add_argument("--graph-json", action="store_true")
    b.set_defaults(fn=cmd_build)

    bt = sub.add_parser("batch", help="many specs x seeds, one building per process")
    bt.add_argument("specs", nargs="+", help="spec files and/or directories")
    bt.add_argument("--seeds", default="1..3")
    bt.add_argument("--jobs", type=int, default=None)
    bt.add_argument("--out", default="assets/buildings")
    bt.add_argument("--no-render", action="store_true")
    bt.add_argument("--no-glb", action="store_true")
    bt.set_defaults(fn=cmd_batch)

    r = sub.add_parser("render", help="re-render an output directory")
    r.add_argument("dir")
    r.set_defaults(fn=cmd_render)

    v = sub.add_parser("validate", help="regenerate + validate an output directory")
    v.add_argument("dir")
    v.add_argument("--jobs", type=int, default=None)
    v.set_defaults(fn=cmd_validate)

    d = sub.add_parser("district", help="(stub) block of buildings -> arena.json")
    d.add_argument("--grid", default="4x4")
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(fn=cmd_district)

    a = ap.parse_args(argv)
    return int(a.fn(a) or 0)


if __name__ == "__main__":
    sys.exit(main())
