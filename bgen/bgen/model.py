"""Core data model shared by all bgen stages (see DESIGN.md §2.3–2.6).

Pipeline:  Spec -> BuildingContext(panels, rooms, sockets) -> [ChunkGeom per panel]
           -> edges -> BuildingData (flat numpy arrays) -> .bld / .glb / renders
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Polygon


@dataclass
class Panel:
    """A structural element: a 2D profile in the panel plane extruded by `thickness`.

    `frame` maps panel-local coords to building coords. Panel plane = local XY,
    the solid occupies local z in [0, thickness].
    """
    id: int
    kind: str
    frame: np.ndarray                 # (4,4) float64
    profile: Polygon                  # local XY, holes allowed
    thickness: float
    material: str
    tags: dict[str, Any] = field(default_factory=dict)  # floor, room ids, facade id, anchor, indestructible, cosmetic...

    def to_world(self, pts_local: np.ndarray) -> np.ndarray:
        """(N,2|3) local -> (N,3) building space."""
        p = np.asarray(pts_local, dtype=np.float64)
        if p.shape[1] == 2:
            p = np.column_stack([p, np.zeros(len(p))])
        return p @ self.frame[:3, :3].T + self.frame[:3, 3]


@dataclass
class Room:
    id: int
    floor: int
    polygon: Polygon                  # building XY
    tag: str


@dataclass
class Socket:
    type: str                         # see materials.SOCKET_TYPES
    xform: np.ndarray                 # (4,4)
    owner_panel: int = -1


@dataclass
class Blockout:
    floor_polys: list[Polygon]        # per floor index, building XY footprint
    floor_z: list[float]              # z of top of slab i (= floor level); floor_z[0] = 0 typically slab top
    roof_types: list[str]             # per floor: roof type used where this floor is exposed above
    facades: list[list[tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)  # per floor: list of (p0,p1) segments


@dataclass
class BuildingContext:
    spec: dict
    params: dict                      # resolved global params (floor_height, slab_thickness, ...)
    blockout: Blockout | None = None
    panels: list[Panel] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    sockets: list[Socket] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def next_panel_id(self) -> int:
        return len(self.panels)

    def add_panel(self, **kw) -> Panel:
        p = Panel(id=self.next_panel_id(), **kw)
        self.panels.append(p)
        return p


@dataclass
class ChunkGeom:
    """One fractured convex chunk (a convex prism) in building space."""
    panel_id: int
    material: str
    poly2d: np.ndarray                # (n,2) convex CCW polygon in panel-local XY
    hull_verts: np.ndarray            # (2n,3) building space
    planes: np.ndarray                # (n+2,4) outward (nx,ny,nz,d) with n·x + d <= 0 inside
    mesh_verts: np.ndarray            # (V,3) building space (flat shaded, unwelded per face)
    mesh_normals: np.ndarray          # (V,3)
    mesh_face_flags: np.ndarray       # (V,) uint16
    mesh_tris: np.ndarray             # (T,3) uint32 local to this chunk
    volume: float
    mass: float
    com: np.ndarray                   # (3,)
    inertia: np.ndarray               # (6,) [Ixx,Iyy,Izz,Ixy,Ixz,Iyz] about com, building axes
    flags: int = 0
    hp: float = 0.0


@dataclass
class Edge:
    a: int
    b: int
    area: float
    strength: float
    centroid: np.ndarray
    normal: np.ndarray                # from a to b


@dataclass
class BuildingData:
    """Everything that goes into a .bld file, as flat arrays (§4)."""
    meta: dict
    elements: np.ndarray              # ELEM_DTYPE
    chunks: np.ndarray                # CHNK_DTYPE
    hull_verts: np.ndarray            # (N,3) f32
    hull_planes: np.ndarray           # (N,4) f32
    mesh_verts: np.ndarray            # MVRT_DTYPE
    mesh_indices: np.ndarray          # (N,) u32
    edges: np.ndarray                 # EDGE_DTYPE
    anchors: np.ndarray               # (ceil(n/64),) u64 bitset
    sockets: np.ndarray               # SOCK_DTYPE
