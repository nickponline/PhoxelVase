"""Building spec loading, preset defaults and seeded sampling.

`(spec, seed)` fully determines the output: anything not given comes from the preset, and
numeric [lo, hi] ranges in `global` and in a procedural `blockout` are sampled from the
spec RNG stream. Node params keep their ranges; nodes sample them from their own stream.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .blockout import generate_masses
from .rng import is_range, rv, spec_rng

GENERATOR_VERSION = "bgen-0.1.0"

_BASE_GLOBAL = dict(floor_height=[3.4, 3.8], slab_thickness=0.3, ext_wall_thickness=0.3,
                    int_wall_thickness=0.15, slab_material="concrete", ext_wall_material="concrete",
                    int_wall_material="brick", column_material="concrete", roof_material="concrete",
                    roof_thickness=0.2)

_FRACTURE = {
    "concrete": {"cell_size": 1.1, "edge_bias": 1.5},
    "brick": {"cell_size": 0.8},
    "wood": {"cell_size": 0.7, "anisotropy": [3, 1]},
    "metal": {"cell_size": 1.2},
    "glass": {"mode": "shatter_on_hit"},
}


def _deep_merge(a: Any, b: Any) -> Any:
    if isinstance(a, dict) and isinstance(b, dict):
        out = dict(a)
        for k, v in b.items():
            out[k] = _deep_merge(a[k], v) if k in a else copy.deepcopy(v)
        return out
    return copy.deepcopy(b)


def _nodes(**over):
    base = [
        ("floors", {}),
        ("exterior_walls", {"bay_width": [3.0, 4.5], "corner_rule": "x_runs_through"}),
        ("columns", {"grid": 6.0, "size": 0.5}),
        ("rooms", {"strategy": "bsp", "min_room": 9.0, "max_room": [25.0, 45.0], "corridor": True,
                   "tags": {"office": 0.6, "meeting": 0.25, "toilet": 0.15}}),
        ("stairs", {"cores": 1, "type": "switchback"}),
        ("openings", {"windows": {"style": "ribbon", "sill": 0.9, "head": 2.6},
                      "doors": {"per_room": 1, "width": 0.9, "height": 2.1}}),
        ("roofs", {"flat": {"parapet": 1.0}, "gable": {"pitch_deg": 35, "eaves": 0.4}}),
        ("balconies", {"facades": [], "every_n_floors": 1}),
        ("manual", {}),
    ]
    out = []
    for name, p in base:
        if name in over:
            if over[name] is None:
                continue
            p = _deep_merge(p, over[name])
            if "tags" in over[name]:
                p["tags"] = copy.deepcopy(over[name]["tags"])
        out.append({name: p})
    return out


PRESETS: dict[str, dict] = {
    "office": dict(
        global_=dict(),
        blockout=dict(shape=["rect", "L", "annex"], width=[24.0, 36.0], depth=[14.0, 18.0], floors=[4, 6]),
        nodes=_nodes(openings={"ground": {"style": "storefront"}}, balconies=None),
        fracture=dict(_FRACTURE, concrete={"cell_size": 1.3, "edge_bias": 1.5}, brick={"cell_size": 1.0})),
    "apartment": dict(
        global_=dict(floor_height=[3.0, 3.2], ext_wall_material="brick", int_wall_thickness=0.12),
        blockout=dict(shape=["rect", "L"], width=[22.0, 32.0], depth=[12.0, 15.0], floors=[4, 6]),
        nodes=_nodes(columns={"grid": 7.0},
                     rooms={"min_room": 8.0, "max_room": [18.0, 30.0], "corridor": True,
                            "tags": {"living_room": 0.35, "bedroom": 0.4, "kitchen": 0.15, "bathroom": 0.1}},
                     openings={"windows": {"style": ["punched", "arched"], "sill": 0.9, "head": 2.3,
                                           "width": [1.0, 1.6]}},
                     balconies={"facades": ["south"], "every_n_floors": 1, "depth": [1.2, 1.5],
                                "probability": 0.6}),
        fracture=dict(_FRACTURE, concrete={"cell_size": 1.3}, brick={"cell_size": 0.9})),
    "warehouse": dict(
        global_=dict(floor_height=[6.0, 7.5], ext_wall_thickness=0.25, roof_material="metal",
                     roof_thickness=0.12),
        blockout=dict(shape="rect", width=[30.0, 44.0], depth=[20.0, 26.0], floors=1, roof=["flat", "gable"]),
        nodes=_nodes(exterior_walls={"bay_width": [5.0, 7.0]},
                     columns={"grid": 9.0, "size": 0.6},
                     rooms={"min_room": 30.0, "max_room": [150.0, 300.0], "corridor": False,
                            "tags": {"storage": 0.7, "office": 0.3}},
                     openings={"windows": {"style": "ribbon", "sill": 3.5, "head": 5.0},
                               "entrance": {"width": 4.0, "height": 4.5}},
                     roofs={"gable": {"pitch_deg": [12, 18], "eaves": 0.5}}, balconies=None),
        fracture=dict(_FRACTURE, concrete={"cell_size": 1.5}, metal={"cell_size": 1.6})),
    "tower": dict(
        global_=dict(floor_height=[3.5, 3.8]),
        blockout=dict(shape=["rect", "podium"], width=[20.0, 30.0], depth=[18.0, 24.0], floors=[8, 15]),
        nodes=_nodes(columns={"grid": 7.0, "size": 0.6},
                     rooms={"min_room": 12.0, "max_room": [40.0, 70.0]},
                     openings={"windows": {"style": "ribbon", "sill": 0.6, "head": 3.0},
                               "ground": {"style": "storefront"}}, balconies=None),
        fracture=dict(_FRACTURE, concrete={"cell_size": 1.8}, brick={"cell_size": 1.4})),
    "house": dict(
        global_=dict(floor_height=[2.9, 3.1], slab_thickness=0.25, ext_wall_thickness=0.25,
                     int_wall_thickness=0.12, ext_wall_material="brick", int_wall_material="wood",
                     slab_material="wood", column_material="wood", roof_material="wood",
                     roof_thickness=0.15),
        blockout=dict(shape="rect", width=[9.0, 12.0], depth=[7.5, 9.0], floors=2, roof=["gable", "hip"]),
        nodes=_nodes(columns=None,
                     rooms={"min_room": 7.0, "max_room": [14.0, 22.0], "corridor": False,
                            "tags": {"living_room": 0.35, "bedroom": 0.35, "kitchen": 0.15, "bathroom": 0.15}},
                     stairs={"flight_width": 1.0},
                     openings={"windows": {"style": "punched", "sill": 0.9, "head": 2.2, "width": [0.9, 1.3]}},
                     roofs={"gable": {"pitch_deg": [30, 40], "eaves": 0.4}, "hip": {"pitch_deg": [25, 32]}},
                     balconies=None),
        fracture=dict(_FRACTURE)),
    "kyoto": dict(
        global_=dict(floor_height=[3.0, 3.2], slab_thickness=0.25, ext_wall_thickness=0.2,
                     int_wall_thickness=0.1, ext_wall_material="wood", int_wall_material="wood",
                     slab_material="wood", column_material="wood", roof_material="wood",
                     roof_thickness=0.15),
        blockout=dict(shape="rect", width=[11.0, 14.0], depth=[8.0, 10.0], floors=[1, 2], roof="kyoto"),
        nodes=_nodes(exterior_walls={"bay_width": [1.8, 2.4]},
                     columns=None,
                     rooms={"min_room": 7.0, "max_room": [15.0, 25.0], "corridor": False,
                            "tags": {"washitsu": 0.6, "engawa": 0.2, "kitchen": 0.2}},
                     stairs={"flight_width": 1.0},
                     openings={"windows": {"style": ["punched", "round"], "sill": 0.8, "head": 2.0,
                                           "width": [0.9, 1.4]}},
                     roofs={"kyoto": {"pitch_deg": [36, 42], "eaves": 0.9, "strips": 4}}, balconies=None),
        fracture=dict(_FRACTURE)),
    # free-form structures (bridges, towers, statues): the `sculpt` node slices CSG primitives
    # into layers and adds its own ground slab; the 1x1 blockout is only a placeholder.
    "structure": dict(
        global_=dict(floor_height=4.0),
        blockout=[dict(footprint=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], floors=1)],
        nodes=[{"sculpt": {"layer": 0.5, "ground": {"pad": 3.0}}}],
        fracture=dict(_FRACTURE, concrete={"cell_size": 1.6}, brick={"cell_size": 1.4},
                      metal={"cell_size": 1.4})),
}


def load_spec(path: str | Path) -> dict:
    p = Path(path)
    spec = yaml.safe_load(p.read_text()) or {}
    spec["_dir"] = str(p.resolve().parent)
    spec.setdefault("name", p.stem)
    return spec


def _node_list(nodes) -> list[tuple[str, dict]]:
    out = []
    for n in nodes:
        if isinstance(n, str):
            out.append((n, {}))
        else:
            (name, p), = n.items()
            out.append((name, dict(p or {})))
    return out


def resolve_spec(spec: dict, seed: int | None = None) -> dict:
    """Merge the preset and sample global/blockout ranges. Returns a fully resolved spec with
    keys: name, preset, seed, global (scalars), blockout (list of masses), nodes
    (list of {name: params}), fracture, indestructible."""
    spec = copy.deepcopy(spec)
    seed = int(spec.get("seed", 0) if seed is None else seed)
    preset_name = spec.get("preset", "office")
    if preset_name not in PRESETS:
        raise ValueError(f"unknown preset {preset_name!r}; have {sorted(PRESETS)}")
    pre = PRESETS[preset_name]
    rng = spec_rng(seed)

    g = _deep_merge(_deep_merge(_BASE_GLOBAL, pre["global_"]), spec.get("global", {}) or {})
    glob = {k: (round(rv(v, rng), 3) if is_range(v) else v) for k, v in sorted(g.items())}

    bo = spec.get("blockout", pre["blockout"])
    if isinstance(bo, dict):
        bo = generate_masses(_deep_merge(pre["blockout"], bo), rng)
    masses = []
    for m in bo:
        mm = dict(m)
        mm["floors"] = int(rv(mm.get("floors", 1), rng))
        mm.setdefault("roof", "flat")
        mm["footprint"] = [[float(x), float(y)] for x, y in mm["footprint"]]
        masses.append(mm)

    pre_nodes = dict(_node_list(pre["nodes"]))
    if "nodes" in spec and spec["nodes"] is not None:
        nodes = []
        for n, p in _node_list(spec["nodes"]):
            m = _deep_merge(pre_nodes.get(n, {}), p)
            for k in ("tags", "facades", "style"):
                if k in p:
                    m[k] = copy.deepcopy(p[k])
            nodes.append({n: m})
    else:
        nodes = [{n: p} for n, p in _node_list(pre["nodes"])]

    frac = _deep_merge(pre["fracture"], spec.get("fracture", {}) or {})
    out = dict(name=str(spec.get("name", preset_name)), preset=preset_name, seed=seed,
               generator=GENERATOR_VERSION, global_=glob, blockout=masses, nodes=nodes,
               fracture=frac, indestructible=list(spec.get("indestructible", ["ground_slab"]) or []),
               stability=dict(spec.get("stability", {}) or {}))
    out["global"] = out.pop("global_")
    if "_dir" in spec:
        out["_dir"] = spec["_dir"]
    return out


def preset_spec(preset: str, name: str | None = None) -> dict:
    return {"name": name or preset, "preset": preset}
