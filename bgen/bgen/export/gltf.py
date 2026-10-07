"""`building.glb` preview export via trimesh: one node per chunk with
extras {chunk_id, panel_id, material, flags} (DESIGN §2.8)."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import trimesh

from ..materials import MATERIALS

COLORS = {"concrete": (170, 170, 165, 255), "brick": (165, 90, 70, 255), "wood": (160, 120, 75, 255),
          "metal": (120, 130, 140, 255), "glass": (150, 200, 230, 110)}


def _patch_node_extras(glb: bytes) -> bytes:
    """trimesh writes geometry metadata as *mesh* extras; copy them onto the nodes too."""
    magic, ver, _ = struct.unpack_from("<III", glb, 0)
    jlen, jtype = struct.unpack_from("<II", glb, 12)
    js = json.loads(glb[20:20 + jlen].decode())
    rest = glb[20 + jlen:]
    meshes = js.get("meshes", [])
    for nd in js.get("nodes", []):
        if "mesh" in nd:
            ex = meshes[nd["mesh"]].get("extras")
            if ex:
                nd["extras"] = {k: v for k, v in ex.items() if k in ("chunk_id", "panel_id", "material", "flags")}
    jb = json.dumps(js, separators=(",", ":")).encode()
    jb += b" " * ((4 - len(jb) % 4) % 4)
    out = struct.pack("<III", magic, ver, 12 + 8 + len(jb) + len(rest)) + struct.pack("<II", len(jb), jtype) + jb + rest
    return out


def build_scene(ctx, chunks) -> trimesh.Scene:
    scene = trimesh.Scene()
    mats = {m: trimesh.visual.material.PBRMaterial(name=m, baseColorFactor=COLORS[m],
                                                   alphaMode="BLEND" if m == "glass" else "OPAQUE",
                                                   roughnessFactor=0.9, metallicFactor=0.0)
            for m in MATERIALS}
    for i, c in enumerate(chunks):
        mesh = trimesh.Trimesh(vertices=np.asarray(c.mesh_verts, np.float32), faces=c.mesh_tris.astype(np.int64),
                               vertex_normals=np.asarray(c.mesh_normals, np.float32), process=False)
        mesh.visual = trimesh.visual.TextureVisuals(material=mats[c.material])
        mesh.metadata = {"chunk_id": i, "panel_id": int(c.panel_id), "material": c.material, "flags": int(c.flags)}
        scene.add_geometry(mesh, node_name=f"chunk_{i}", geom_name=f"chunk_{i}")
    return scene


def write_glb(ctx, chunks, path) -> Path:
    scene = build_scene(ctx, chunks)
    data = scene.export(file_type="glb")
    Path(path).write_bytes(_patch_node_extras(data))
    return Path(path)
