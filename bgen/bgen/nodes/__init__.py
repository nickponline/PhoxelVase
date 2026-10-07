"""Feature-node registry (DESIGN §2.4).

Each node module exposes `node(ctx, params, rng) -> ctx` and optionally
`reserve(ctx, params, rng)`, a pre-pass run for all nodes (in order) before any `node` call,
used to reserve space/decisions later nodes need earlier (stair cores, bays, balcony bays).
Modules are imported lazily to avoid import cycles.
"""
from __future__ import annotations

import importlib
from types import ModuleType

NODE_MODULES = {
    "floors": "floors", "exterior_walls": "exterior_walls", "columns": "columns",
    "rooms": "rooms", "stairs": "stairs", "openings": "openings", "roofs": "roofs",
    "balconies": "balconies", "manual": "manual",
}


def register(name: str, module_path: str) -> None:
    """Register an extra node module (absolute import path or name inside bgen.nodes)."""
    NODE_MODULES[name] = module_path


def get_module(name: str) -> ModuleType:
    if name not in NODE_MODULES:
        raise KeyError(f"unknown feature node {name!r}; have {sorted(NODE_MODULES)}")
    mp = NODE_MODULES[name]
    return importlib.import_module(mp if "." in mp else f"{__name__}.{mp}")


def get_node(name: str):
    return get_module(name).node


def get_reserve(name: str):
    return getattr(get_module(name), "reserve", None)
