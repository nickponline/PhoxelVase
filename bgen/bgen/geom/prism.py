"""Convex 2D polygon + thickness -> convex prism ChunkGeom (mesh, hull, planes, mass props)."""
from __future__ import annotations

import numpy as np

from ..materials import FACE_CAP_EDGE, FACE_INNER, FACE_OUTER
from ..model import ChunkGeom


def polygon_moments(poly: np.ndarray) -> tuple[float, np.ndarray, float, float, float]:
    """CCW polygon -> (area, centroid, Jxx=∫y², Jyy=∫x², Jxy=∫xy) about the centroid."""
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2.0
    cx = ((x + x1) * cr).sum() / (6 * A)
    cy = ((y + y1) * cr).sum() / (6 * A)
    Iyy0 = (cr * (y * y + y * y1 + y1 * y1)).sum() / 12.0   # ∫y² about origin
    Ixx0 = (cr * (x * x + x * x1 + x1 * x1)).sum() / 12.0   # ∫x²
    Ixy0 = (cr * (x * y1 + 2 * x * y + 2 * x1 * y1 + x1 * y)).sum() / 24.0
    return A, np.array([cx, cy]), Iyy0 - A * cy * cy, Ixx0 - A * cx * cx, Ixy0 - A * cx * cy


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    a = (x * np.roll(y, -1) - np.roll(x, -1) * y).sum()
    return poly if a > 0 else poly[::-1].copy()


def make_prism(poly2d: np.ndarray, thickness: float, frame: np.ndarray, density: float,
               panel_id: int, material: str, side_flags: np.ndarray | None = None,
               flags: int = 0, hp: float = 0.0) -> ChunkGeom:
    """Extrude convex CCW `poly2d` (panel-local XY) along local +Z by `thickness`.

    side_flags[i] is the FACE_* flag of the side face on edge i -> i+1
    (default FACE_INNER = fracture face). Caps are FACE_OUTER.
    """
    poly = ensure_ccw(np.asarray(poly2d, dtype=np.float64))
    n = len(poly)
    R, T = frame[:3, :3], frame[:3, 3]
    assert np.linalg.det(R) > 0.999, "panel frame must be a proper rotation"
    if side_flags is None:
        side_flags = np.full(n, FACE_INNER, np.uint16)

    t = thickness
    bot = np.column_stack([poly, np.zeros(n)])
    top = np.column_stack([poly, np.full(n, t)])
    hull_local = np.vstack([bot, top])
    W = lambda p: p @ R.T + T
    hull = W(hull_local)

    # planes (outward). local first, then transform: n_w = R n, d_w = d - n_w·T
    pl = []
    for i in range(n):
        e = poly[(i + 1) % n] - poly[i]
        nn = np.array([e[1], -e[0], 0.0]); nn /= np.linalg.norm(nn)
        pl.append((nn, -nn[:2] @ poly[i]))
    pl.append((np.array([0, 0, -1.0]), 0.0))
    pl.append((np.array([0, 0, 1.0]), -t))
    planes = np.array([[*(R @ nl), d - (R @ nl) @ T] for nl, d in pl])

    # mesh (flat shaded, per-face verts)
    verts, nrms, ffl, tris = [], [], [], []

    def add_face(pts, normal, fl):
        b = len(verts)
        verts.extend(pts); nrms.extend([normal] * len(pts)); ffl.extend([fl] * len(pts))
        for k in range(1, len(pts) - 1):
            tris.append((b, b + k, b + k + 1))

    add_face(list(top), np.array([0, 0, 1.0]), FACE_OUTER)
    add_face(list(bot[::-1]), np.array([0, 0, -1.0]), FACE_OUTER)
    for i in range(n):
        j = (i + 1) % n
        add_face([bot[i], bot[j], top[j], top[i]], pl[i][0], int(side_flags[i]))
    mv = W(np.array(verts))
    mn = np.array(nrms) @ R.T

    A, c2, Jxx, Jyy, Jxy = polygon_moments(poly)
    vol = A * t
    m = density * vol
    rho = density
    Il = np.array([
        [rho * (t * Jxx + A * t ** 3 / 12), -rho * t * Jxy, 0.0],
        [-rho * t * Jxy, rho * (t * Jyy + A * t ** 3 / 12), 0.0],
        [0.0, 0.0, rho * t * (Jxx + Jyy)],
    ])
    Iw = R @ Il @ R.T
    com = W(np.array([[c2[0], c2[1], t / 2]]))[0]
    return ChunkGeom(
        panel_id=panel_id, material=material, poly2d=poly, hull_verts=hull, planes=planes,
        mesh_verts=mv, mesh_normals=mn, mesh_face_flags=np.array(ffl, np.uint16),
        mesh_tris=np.array(tris, np.uint32), volume=vol, mass=m, com=com,
        inertia=np.array([Iw[0, 0], Iw[1, 1], Iw[2, 2], Iw[0, 1], Iw[0, 2], Iw[1, 2]]),
        flags=flags, hp=hp,
    )


__all__ = ["make_prism", "polygon_moments", "ensure_ccw", "FACE_CAP_EDGE"]
