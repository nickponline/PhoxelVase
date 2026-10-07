"""Build the images for docs/explainer.html.

    .venv/bin/python docs/explainer/make_figures.py

Needs generated buildings in assets/buildings (office_1, building4_4, house_1 ...) and scenario runs
in runs/. img/viewer.jpg is a rubble-viewer screenshot (--screenshot) and is not rebuilt here.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPoly
from PIL import Image
from shapely.geometry import Polygon, box

from bgen.geom.convex2d import convex_decompose
from bgen.geom.fracture import fracture_panel
from bgen.geom.voronoi2d import bounded_voronoi
from bgen.model import Panel

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

INK, FACE, HOLE = "#2b2f36", "#e9e6df", "#ffffff"


def _ax(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(-0.4, 12.4)
    ax.set_ylim(-0.4, 3.9)
    ax.axis("off")
    ax.set_title(title, fontsize=11, color=INK, loc="left")


def fracture_steps():
    wall = box(0, 0, 12, 3.4)
    holes = [box(1.0, 0.9, 3.2, 2.6), box(4.4, 0.9, 6.6, 2.6), box(9.4, 0.0, 10.4, 2.1)]  # 2 windows + door
    prof = wall
    for h in holes:
        prof = prof.difference(h)
    prof = Polygon(prof.exterior, [i for i in prof.interiors]) if prof.geom_type == "Polygon" else prof
    rng = np.random.default_rng(3)
    cmap = plt.get_cmap("tab20")

    fig, axs = plt.subplots(4, 1, figsize=(9, 10.5))
    # 1 profile
    ax = axs[0]
    _ax(ax, "Step 1 - profile: the wall outline, with holes for 2 windows and 1 door")
    geoms = [prof] if prof.geom_type == "Polygon" else list(prof.geoms)
    for g in geoms:
        ax.add_patch(MplPoly(np.array(g.exterior.coords), fc=FACE, ec=INK, lw=1.5))
        for r in g.interiors:
            ax.add_patch(MplPoly(np.array(r.coords), fc=HOLE, ec=INK, lw=1.5))
    # 2 convex parts
    ax = axs[1]
    _ax(ax, "Step 2 - convex decomposition: cut the profile into convex parts")
    parts = convex_decompose(prof)
    for i, p in enumerate(parts):
        ax.add_patch(MplPoly(p, fc=cmap(i % 20), ec=INK, lw=1.2, alpha=0.85))
    # 3 voronoi
    ax = axs[2]
    _ax(ax, "Step 3 - Voronoi cells: scatter seed points, each cell = area nearest to one seed")
    seeds = np.column_stack([rng.uniform(0, 12, 46), rng.uniform(0, 3.4, 46)])
    for g in geoms:
        ax.add_patch(MplPoly(np.array(g.exterior.coords), fc=FACE, ec=INK, lw=1.5))
        for r in g.interiors:
            ax.add_patch(MplPoly(np.array(r.coords), fc=HOLE, ec=INK, lw=1.5))
    for cell in bounded_voronoi(seeds, (0, 0, 12, 3.4)):
        if len(cell):
            ax.add_patch(MplPoly(cell, fill=False, ec="#c0392b", lw=0.9, ls="--"))
    ax.plot(seeds[:, 0], seeds[:, 1], ".", color="#c0392b", ms=4)
    # 4 chunks
    ax = axs[3]
    _ax(ax, "Step 4 - chunks: each convex part ∩ each cell; every chunk is convex")
    panel = Panel(id=0, kind="ext_wall", frame=np.eye(4), profile=prof, thickness=0.3, material="concrete", tags={})
    chunks = fracture_panel(panel, {"cell_size": 0.9}, np.random.default_rng(7))
    for i, c in enumerate(chunks):
        ax.add_patch(MplPoly(c.poly2d, fc=cmap((i * 7) % 20), ec=INK, lw=0.8))
    fig.text(0.01, 0.005, f"Real output of bgen.fracture_panel: {len(chunks)} chunks from one 12 m x 3.4 m wall.",
             fontsize=9, color="#555")
    fig.tight_layout()
    fig.savefig(OUT / "fracture_steps.png", dpi=130)
    plt.close(fig)
    return len(chunks)


def blockout_fig():
    masses = [((0, 0, 40, 32), 4, "podium"), ((7, 3, 33, 29), 40, "shaft"),
              ((10, 6, 30, 26), 48, "setback"), ((14, 10, 26, 22), 52, "crown")]
    cols = ["#d9cfc1", "#b8c7d9", "#9fb6cf", "#7f9cbd"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    for (x0, y0, x1, y1), fl, name in masses:
        pass
    for ((x0, y0, x1, y1), fl, name), c in zip(masses, cols):
        a.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc=c, ec=INK, lw=1.2))
        a.text(x0 + 0.6, y0 + 0.6, f"{name}\n{fl} floors", fontsize=8, color=INK, va="bottom")
        b.add_patch(plt.Rectangle((x0, 0), x1 - x0, fl * 3.6, fc=c, ec=INK, lw=1.2))
    a.set_xlim(-2, 42); a.set_ylim(-2, 34); a.set_aspect("equal"); a.set_title("Plan (from above): 4 masses", loc="left", fontsize=11)
    a.set_xlabel("x (m)"); a.set_ylabel("y (m)")
    b.set_xlim(-2, 42); b.set_ylim(0, 195); b.set_title("Elevation (from the side)", loc="left", fontsize=11)
    b.set_xlabel("x (m)"); b.set_ylabel("height (m)")
    for k in range(0, 53, 4):
        b.axhline(k * 3.6, color="#ffffff", lw=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "blockout_building4.png", dpi=120)
    plt.close(fig)


def copy_img(src: Path, name: str, width: int = 1200, trim: bool = False):
    if not src.exists():
        print("missing", src)
        return
    im = Image.open(src).convert("RGB")
    if trim:  # cut away the near-white margin around matplotlib plots
        a = np.asarray(im).min(axis=2) < 235
        ys, xs = np.nonzero(a)
        im = im.crop((max(xs.min() - 12, 0), max(ys.min() - 12, 0), xs.max() + 12, ys.max() + 12))
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    im.save(OUT / name, quality=85, optimize=True)


def copies():
    r = ROOT / "assets/buildings/office_1/renders"
    for n in ["iso_ne", "fractured_iso", "exploded", "cutaway_floor_1", "plan_floor_1", "graph", "stress"]:
        copy_img(r / f"{n}.png", f"office_{n}.jpg", trim=n.startswith("plan"))
    copy_img(ROOT / "assets/buildings/building4_4/renders/iso_ne.png", "building4_iso.jpg")
    for s in ["office_columns_out", "office_top_blast", "building4_shaft_cut", "two_box_drop",
              "tower_columns_out", "apartment_ground_out"]:
        copy_img(ROOT / f"runs/{s}/contact_sheet.png", f"sheet_{s}.jpg", 1400)
    for n in ["house_1", "kyoto_1", "warehouse_1", "apartment_1", "tower_1"]:
        copy_img(ROOT / f"assets/buildings/{n}/renders/iso_ne.png", f"preset_{n}.jpg", 600)


GIFS = {  # run name -> (output name, keep every n-th frame)
    "office_columns_out": ("gif_office_collapse.gif", 2),
    "tower_columns_out": ("gif_tower_collapse.gif", 2),
    "apartment_ground_out": ("gif_apartment_collapse.gif", 2),
    "office_top_blast": ("gif_office_top_blast.gif", 2),
    "building4_shaft_cut": ("gif_building4_limit.gif", 3),
}


def make_gifs(width: int = 480, frame_ms: int = 133, hold_ms: int = 1500):
    """Small looping GIFs from the frames that bgen.sim saved (runs/<name>/frames/*.png).
    The runs record every 4th tick (15 fps of sim time); keeping every 2nd frame at 133 ms
    plays close to real time."""
    for run, (out, every) in GIFS.items():
        frames = sorted((ROOT / "runs" / run / "frames").glob("*.png"))[::every]
        if not frames:
            print("missing frames for", run)
            continue
        ims = []
        for f in frames:
            im = Image.open(f).convert("RGB")
            im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
            ims.append(im.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE))
        dur = [frame_ms * every // 2] * len(ims)
        dur[-1] = hold_ms
        ims[0].save(OUT / out, save_all=True, append_images=ims[1:], duration=dur, loop=0, optimize=True)
        print(out, len(ims), "frames", (OUT / out).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    n = fracture_steps()
    blockout_fig()
    copies()
    make_gifs()
    print("chunks in demo wall:", n, "->", OUT)
