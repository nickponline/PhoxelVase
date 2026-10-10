"""Backend-independent render data: flattened mesh, cameras, styles and render jobs.

Both backends (`gl.GLRenderer`, `raster_np.NumpyRenderer`) consume a `RenderMesh` (built once
per building) and a sequence of `RenderJob`s (camera + per-chunk colors / visibility /
transforms). Z-up, meters, right-handed. Matrices are row-major numpy (x' = M @ x).
"""
from __future__ import annotations

import colorsys
import math
from dataclasses import dataclass, field

import numpy as np

from ..materials import ELEMENT_KINDS, F_GLASS, MATERIALS

# sRGB massing-model tints per material id (close to white; edges carry the read)
MATERIAL_COLORS = {
    "concrete": (0.87, 0.87, 0.86),
    "brick": (0.88, 0.80, 0.74),
    "wood": (0.88, 0.80, 0.67),
    "metal": (0.74, 0.77, 0.81),
    "glass": (0.68, 0.83, 0.95),
}


@dataclass
class RenderMesh:
    pos: np.ndarray          # (V,3) f32 building space
    nrm: np.ndarray          # (V,3) f32 unit
    flags: np.ndarray        # (V,) u16 FACE_* flags
    vchunk: np.ndarray       # (V,) i32 chunk id per vertex
    tris: np.ndarray         # (T,3) u32 indices into pos
    tchunk: np.ndarray       # (T,) i32 chunk id per triangle
    n_chunks: int
    com: np.ndarray          # (N,3)
    aabb_min: np.ndarray     # (N,3)
    aabb_max: np.ndarray     # (N,3)
    material: np.ndarray     # (N,) material id
    chunk_flags: np.ndarray  # (N,)
    elem: np.ndarray         # (N,) element id
    elem_kind: np.ndarray    # (N,) element kind name (str array)
    elem_floor: np.ndarray   # (N,) element floor index (-1 = unknown)
    mass: np.ndarray         # (N,)
    # (ground level z, (K,4) xy rects [x0,y0,x1,y1] left open): a sunk foundation / basement
    # sits in pits in the ground; None draws the ground under the lowest point
    ground: tuple[float, np.ndarray] | None = None

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if self.n_chunks == 0:
            return np.zeros(3), np.ones(3)
        return self.aabb_min.min(0).astype(float), self.aabb_max.max(0).astype(float)

    def base_colors(self) -> np.ndarray:
        """(N,3) sRGB colors by material."""
        lut = np.array([MATERIAL_COLORS.get(m, (0.85, 0.85, 0.85)) for m in MATERIALS], np.float32)
        return lut[np.clip(self.material, 0, len(lut) - 1)]


def build_mesh(bd) -> RenderMesh:
    ch = bd.chunks
    n = len(ch)
    mv = bd.mesh_verts
    idx = np.asarray(bd.mesh_indices, np.uint32)
    # chunk id per index via ranges
    ichunk = np.repeat(np.arange(n, dtype=np.int32), ch["mesh_i_cnt"].astype(np.int64)) if n else np.zeros(0, np.int32)
    # mesh index ranges are contiguous & in chunk order (assemble); be robust anyway
    if n and not (np.all(ch["mesh_i_off"][1:] == np.cumsum(ch["mesh_i_cnt"])[:-1]) and ch["mesh_i_off"][0] == 0):
        ichunk = np.full(len(idx), -1, np.int32)
        for i in range(n):
            ichunk[ch["mesh_i_off"][i]:ch["mesh_i_off"][i] + ch["mesh_i_cnt"][i]] = i
    tris = idx[: len(idx) // 3 * 3].reshape(-1, 3)
    tchunk = ichunk[: len(tris) * 3: 3] if len(tris) else np.zeros(0, np.int32)
    vchunk = np.zeros(len(mv), np.int32)
    if len(idx):
        vchunk[idx[: len(ichunk)]] = ichunk
    pos = np.ascontiguousarray(mv["pos"], np.float32)
    nrm = mv["nrm"].astype(np.float32) / 32767.0
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    el = bd.elements
    eidx = np.clip(ch["elem"].astype(np.int64), 0, max(len(el) - 1, 0))
    if len(el):
        kinds = np.array(ELEMENT_KINDS + ["other"] * 64)[el["kind"].astype(np.int64)]
        ekind, efloor = kinds[eidx], el["floor"].astype(np.int32)[eidx]
    else:
        ekind, efloor = np.full(n, "other"), np.full(n, -1, np.int32)
    return RenderMesh(
        pos=pos, nrm=nrm.astype(np.float32), flags=mv["face_flags"].astype(np.uint16), vchunk=vchunk,
        tris=tris.astype(np.uint32), tchunk=tchunk.astype(np.int32), n_chunks=n,
        com=ch["com"].astype(np.float64), aabb_min=ch["aabb_min"].astype(np.float64),
        aabb_max=ch["aabb_max"].astype(np.float64), material=ch["material"].astype(np.int64),
        chunk_flags=ch["flags"].astype(np.int64), elem=ch["elem"].astype(np.int64),
        elem_kind=ekind, elem_floor=efloor, mass=ch["mass"].astype(np.float64),
        ground=ground_of(bd.meta.get("foundation")),
    )


def ground_of(foundation: dict | None, X: np.ndarray | None = None) -> tuple[float, np.ndarray] | None:
    """RenderMesh.ground from a .bld's foundation meta, posed by the 4x4 `X` (world-axis bounds
    of each rect, as in rubble's terrain)."""
    if not foundation:
        return None
    X = np.eye(4) if X is None else np.asarray(X, float)
    R = np.asarray(foundation["rects"], float).reshape(-1, 4)
    out = []
    for x0, y0, x1, y1 in R:
        P = np.array([[x0, y0, 0], [x1, y0, 0], [x1, y1, 0], [x0, y1, 0]]) @ X[:3, :3].T + X[:3, 3]
        out.append([*P[:, :2].min(0), *P[:, :2].max(0)])
    return float(foundation["top"]) + float(X[2, 3]), np.array(out, float).reshape(-1, 4)


def ground_z_of(mesh: RenderMesh) -> float:
    return mesh.ground[0] if mesh.ground is not None else min(0.0, float(mesh.bounds[0][2]))


def ground_cells(c, ext: float, holes: np.ndarray | None) -> np.ndarray:
    """(K,4) xy rects tiling the square of half-size `ext` around `c` minus the `holes`."""
    if holes is None or not len(holes):
        return np.array([[c[0] - ext, c[1] - ext, c[0] + ext, c[1] + ext]])
    xs = np.unique(np.clip(np.r_[holes[:, [0, 2]].ravel(), c[0] - ext, c[0] + ext], c[0] - ext, c[0] + ext))
    ys = np.unique(np.clip(np.r_[holes[:, [1, 3]].ravel(), c[1] - ext, c[1] + ext], c[1] - ext, c[1] + ext))
    out = []
    for j in range(len(ys) - 1):
        ym = 0.5 * (ys[j] + ys[j + 1])
        for i in range(len(xs) - 1):
            xm = 0.5 * (xs[i] + xs[i + 1])
            if not ((holes[:, 0] < xm) & (xm < holes[:, 2]) & (holes[:, 1] < ym) & (ym < holes[:, 3])).any():
                out.append([xs[i], ys[j], xs[i + 1], ys[j + 1]])
    return np.array(out, float).reshape(-1, 4)


# --------------------------------------------------------------------------- cameras

def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    eye, target, up = (np.asarray(v, float) for v in (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f)
    if abs(f @ up) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    s = np.cross(f, up); s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def ortho(l, r, b, t, n, f) -> np.ndarray:
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = 2 / (r - l), 2 / (t - b), -2 / (f - n)
    m[0, 3], m[1, 3], m[2, 3] = -(r + l) / (r - l), -(t + b) / (t - b), -(f + n) / (f - n)
    return m


def perspective(fovy_deg, aspect, n, f) -> np.ndarray:
    t = 1 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0], m[1, 1] = t / aspect, t
    m[2, 2], m[2, 3] = (f + n) / (n - f), 2 * f * n / (n - f)
    m[3, 2] = -1
    return m


@dataclass
class Camera:
    view: np.ndarray
    proj: np.ndarray
    eye: np.ndarray
    ortho: bool
    azimuth: float      # deg, direction from target to eye, CCW from +X
    elevation: float    # deg
    scale: float        # scene radius (for depth thresholds / ground fade)
    target: np.ndarray

    @property
    def view_proj(self) -> np.ndarray:
        return self.proj @ self.view

    def project(self, pts: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """World pts (N,3) -> pixel coords (N,2) (origin top-left) and view depth (N,)."""
        p = np.column_stack([pts, np.ones(len(pts))]) @ self.view_proj.T
        w = p[:, 3:4]
        ndc = p[:, :3] / w
        W, H = size
        px = np.column_stack([(ndc[:, 0] + 1) * 0.5 * W, (1 - ndc[:, 1]) * 0.5 * H])
        vz = -(np.column_stack([pts, np.ones(len(pts))]) @ self.view.T)[:, 2]
        return px, vz


def _dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])


def _corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def fit_camera(lo, hi, azimuth: float, elevation: float, aspect: float, use_ortho: bool = True,
               fovy: float = 30.0, margin: float = 1.10, pts: np.ndarray | None = None) -> Camera:
    """Camera looking at the AABB [lo,hi] from (azimuth, elevation), auto-fitted to the frame."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    P = _corners(lo, hi) if pts is None else np.asarray(pts, float)
    c = 0.5 * (lo + hi)
    R = max(0.5 * np.linalg.norm(hi - lo), 1e-3)
    d = _dir(azimuth, elevation)
    up = (0.0, 1.0, 0.0) if elevation > 89.0 else (0.0, 0.0, 1.0)
    if use_ortho:
        dist = 4 * R
        V = look_at(c + d * dist, c, up)
        q = np.column_stack([P, np.ones(len(P))]) @ V.T
        x0, x1, y0, y1 = q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hw, hh = (x1 - x0) / 2 * margin, (y1 - y0) / 2 * margin
        hw, hh = max(hw, hh * aspect, 1e-3), max(hh, hw / aspect, 1e-3)
        Pm = ortho(cx - hw, cx + hw, cy - hh, cy + hh, 0.05 * R, dist + 12 * R)
        return Camera(V, Pm, c + d * dist, True, azimuth, elevation, R, c)
    # perspective: iterate distance + recenter so the projected corners fit
    th = math.tan(math.radians(fovy) / 2)
    tw = th * aspect
    target = c.copy()
    dist = R / math.sin(math.radians(fovy) / 2) * margin
    for _ in range(8):
        V = look_at(target + d * dist, target, up)
        q = np.column_stack([P, np.ones(len(P))]) @ V.T
        z = np.maximum(-q[:, 2], 1e-3)
        tx, ty = q[:, 0] / z, q[:, 1] / z
        mx, my = (tx.max() + tx.min()) / 2, (ty.max() + ty.min()) / 2
        need = max((tx.max() - tx.min()) / 2 / tw, (ty.max() - ty.min()) / 2 / th) * margin
        # shift target in view plane to center, then scale distance
        inv = np.linalg.inv(V)[:3, :3]
        target = target + inv @ np.array([mx * dist, my * dist, 0.0])
        dist *= max(need, 0.05)
    V = look_at(target + d * dist, target, up)
    Pm = perspective(fovy, aspect, max(0.02 * dist, 0.01), dist + 14 * R)
    return Camera(V, Pm, target + d * dist, False, azimuth, elevation, R, target)


VIEW_ANGLES = {
    # name: (azimuth, elevation, ortho)
    "iso_ne": (45.0, 30.0, True),
    "iso_nw": (135.0, 30.0, True),
    "iso_sw": (225.0, 30.0, True),
    "iso_se": (315.0, 30.0, True),
    "front": (270.0, 0.0, True),     # looking from -Y toward +Y
    "side": (0.0, 0.0, True),        # looking from +X toward -X
    "top": (270.0, 90.0, True),
    "oblique": (300.0, 55.0, False),
}


def named_camera(name: str, lo, hi, size, pts=None, margin=1.10) -> Camera:
    az, el, o = VIEW_ANGLES[name]
    return fit_camera(lo, hi, az, el, size[0] / size[1], use_ortho=o, margin=margin, pts=pts)


# --------------------------------------------------------------------------- style / jobs

def srgb_to_linear(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)


@dataclass
class Style:
    sky_top: tuple = (0.80, 0.85, 0.91)
    sky_bottom: tuple = (0.96, 0.96, 0.95)
    ground: tuple = (0.90, 0.90, 0.89)
    grid_minor: float = 1.0
    grid_major: float = 5.0
    key: float = 1.45          # key light intensity (linear)
    hemi_sky: tuple = (0.50, 0.53, 0.58)
    hemi_ground: tuple = (0.22, 0.21, 0.20)
    inner_factor: float = 0.72  # albedo multiplier for FACE_INNER faces
    line_color: tuple = (0.16, 0.17, 0.19)
    line_alpha: float = 0.85
    normal_thresh: float = 0.92  # dot below this -> crease
    depth_thresh: float = 0.004  # relative depth step
    chunk_lines: float = 0.0     # strength of lines between different chunks (0 = off)
    shadows: bool = True
    ground_on: bool = True
    shadow_strength: float = 0.75


def key_light_dir(cam: Camera) -> np.ndarray:
    """Key light from camera-left-above so the two visible facades read differently."""
    az = cam.azimuth + 55.0 if cam.elevation < 89 else 200.0
    return _dir(az, 50.0)


@dataclass
class RenderJob:
    camera: Camera
    colors: np.ndarray                    # (N,3) sRGB in [0,1]
    visible: np.ndarray | None = None     # (N,) bool
    xforms: np.ndarray | None = None      # (N,4,4) building->world
    style: Style = field(default_factory=Style)
    ground_z: float = 0.0
    ground_center: np.ndarray | None = None  # xy for the fade
    ground_radius: float | None = None
    ground_holes: np.ndarray | None = None   # (K,4) xy rects with no ground (foundation pits)


def random_chunk_colors(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.random(n)
    s = 0.30 + 0.35 * rng.random(n)
    v = 0.78 + 0.18 * rng.random(n)
    return np.array([colorsys.hsv_to_rgb(*x) for x in zip(h, s, v)], np.float32).reshape(n, 3)


def glass_mask(mesh: RenderMesh) -> np.ndarray:
    return ((mesh.chunk_flags & F_GLASS) != 0) | (mesh.material == MATERIALS.index("glass"))


def shadow_camera(lo, hi, light_dir, ground_z=0.0) -> np.ndarray:
    """Light view-proj (ortho along -light_dir) covering the AABB and its ground shadow."""
    P = _corners(lo, hi)
    L = np.asarray(light_dir, float)
    t = (P[:, 2] - ground_z) / max(L[2], 0.05)
    Pg = P - t[:, None] * L
    allp = np.vstack([P, Pg])
    c = allp.mean(0)
    R = np.linalg.norm(allp - c, axis=1).max() + 1e-3
    V = look_at(c + L * 3 * R, c)
    q = np.column_stack([allp, np.ones(len(allp))]) @ V.T
    pad = 0.02 * R
    Pm = ortho(q[:, 0].min() - pad, q[:, 0].max() + pad, q[:, 1].min() - pad, q[:, 1].max() + pad,
               0.1 * R, 6 * R)
    return Pm @ V


def floor_levels(mesh: RenderMesh, meta: dict) -> list[tuple[float, float]]:
    """Per floor index: (floor level z = top of its slab, ceiling z = bottom of the next slab)."""
    params = meta.get("params", {}) or {}
    fh = float(params.get("floor_height", 3.0))
    fl = mesh.elem_floor
    nonroof = ~np.isin(mesh.elem_kind, ["roof", "parapet", "ridge", "fascia"])
    room_floors = [int(r.get("floor", -1)) for r in meta.get("rooms", []) or []]
    nf = max([int(fl[nonroof].max()) if nonroof.any() and len(fl) else -1] + room_floors) + 1
    if nf <= 0:
        return []
    slab = np.isin(mesh.elem_kind, ["floor"])
    lv = []
    for i in range(nf):
        s = slab & (fl == i)
        z0 = float(np.median(mesh.aabb_max[s, 2])) if s.any() else (lv[-1][1] if lv else 0.0)
        lv.append([z0, None])
    for i in range(nf):
        z0 = lv[i][0]
        s = slab & (fl == i + 1)
        z1 = float(np.median(mesh.aabb_min[s, 2])) if s.any() else None
        if z1 is None:
            w = (fl == i) & ~slab & nonroof
            z1 = float(np.percentile(mesh.aabb_max[w, 2], 90)) if w.any() else z0 + fh
        if z1 - z0 < 0.2:
            z1 = z0 + fh
        lv[i][1] = z1
    return [(a, b) for a, b in lv]
