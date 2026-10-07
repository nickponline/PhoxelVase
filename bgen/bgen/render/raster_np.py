"""Pure-numpy z-buffer fallback renderer (same API and look as `gl.GLRenderer`, slower).

Triangles are binned by screen bounding-box size; each bin is rasterized in vectorized batches
(candidate pixels × barycentric test), and depth is resolved with a lexsort per batch. The same
rasterizer produces the shadow map. Ground, sky, lighting and the edge pass mirror the GL shaders.
"""
from __future__ import annotations

import numpy as np

from .scene import RenderJob, RenderMesh, key_light_dir, shadow_camera

_FRAG_BATCH = 1 << 22  # fragments per batch
_TID_BITS = 25
_DEPTH_BITS = 37


def rasterize(xy: np.ndarray, z: np.ndarray, tris: np.ndarray, W: int, H: int,
              tri_ids: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize screen-space triangles. xy: (V,2) pixel coords (pixel centers at i+0.5),
    z: (V,) NDC depth in [-1,1] (smaller = closer). Returns zbuf (H,W) f32 (inf = empty) and
    triangle id (H,W) i32 (-1 = empty).

    Exact scanline spans per triangle row are expanded to fragments with np.repeat; the depth
    test packs (quantized depth, triangle id) into one int64 key resolved with np.minimum.at."""
    INF = np.iinfo(np.int64).max
    kbuf = np.full(H * W, INF, np.int64)
    if len(tris) == 0:
        return np.full((H, W), np.inf, np.float32), np.full((H, W), -1, np.int32)
    ids = np.arange(len(tris), dtype=np.int64) if tri_ids is None else np.asarray(tri_ids, np.int64)
    assert ids.max() < (1 << _TID_BITS)
    P = xy[tris].astype(np.float64)                    # (T,3,2)
    Z = z[tris].astype(np.float64)                     # (T,3)
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    y0 = np.maximum(np.ceil(P[:, :, 1].min(1) - 0.5), 0).astype(np.int64)
    y1 = np.minimum(np.floor(P[:, :, 1].max(1) - 0.5), H - 1).astype(np.int64)
    xmn, xmx = P[:, :, 0].min(1), P[:, :, 0].max(1)
    ok = (y1 >= y0) & (np.abs(det) > 1e-12) & np.isfinite(Z).all(1) & (xmx >= 0) & (xmn <= W) \
        & (Z.min(1) <= 1.0) & (Z.max(1) >= -1.0)
    t_all = np.nonzero(ok)[0]
    if not len(t_all):
        return np.full((H, W), np.inf, np.float32), np.full((H, W), -1, np.int32)
    # screen-space depth plane z = A x + B y + C
    with np.errstate(divide="ignore", invalid="ignore"):
        dz1, dz2 = Z[:, 1] - Z[:, 0], Z[:, 2] - Z[:, 0]
        A = (dz1 * e2[:, 1] - dz2 * e1[:, 1]) / det
        B = (dz2 * e1[:, 0] - dz1 * e2[:, 0]) / det
        C = Z[:, 0] - A * P[:, 0, 0] - B * P[:, 0, 1]
    est = (y1 - y0 + 1) * np.clip(xmx - xmn + 1, 1, W)
    cum = np.cumsum(est[t_all])
    splits = np.searchsorted(cum, np.arange(_FRAG_BATCH, cum[-1] + _FRAG_BATCH, _FRAG_BATCH))
    bounds = np.unique(np.concatenate([[0], splits, [len(t_all)]]))
    qscale = float((1 << _DEPTH_BITS) - 1) / 2.0
    for s, e in zip(bounds[:-1], bounds[1:]):
        t = t_all[s:e]
        if not len(t):
            continue
        nr = y1[t] - y0[t] + 1
        rt = np.repeat(t, nr)
        start = np.cumsum(nr) - nr
        ry = y0[rt] + (np.arange(len(rt)) - np.repeat(start, nr))
        cy = ry + 0.5
        xl = np.full(len(rt), np.inf)
        xr = np.full(len(rt), -np.inf)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            ya, yb = P[rt, a, 1], P[rt, b, 1]
            xa, xb = P[rt, a, 0], P[rt, b, 0]
            v = (np.minimum(ya, yb) <= cy) & (cy <= np.maximum(ya, yb)) & (ya != yb)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = xa + (cy - ya) * (xb - xa) / (yb - ya)
            xl = np.where(v, np.minimum(xl, x), xl)
            xr = np.where(v, np.maximum(xr, x), xr)
        good = np.isfinite(xl)
        xs = np.maximum(np.ceil(np.where(good, xl, 0) - 0.5), 0).astype(np.int64)
        xe = np.minimum(np.floor(np.where(good, xr, -1) - 0.5), W - 1).astype(np.int64)
        n = np.where(good, np.maximum(xe - xs + 1, 0), 0)
        if n.sum() == 0:
            continue
        fr = np.repeat(np.arange(len(rt)), n)
        fstart = np.cumsum(n) - n
        px = xs[fr] + (np.arange(len(fr)) - fstart[fr])
        py = ry[fr]
        ft = rt[fr]
        d = A[ft] * (px + 0.5) + B[ft] * (py + 0.5) + C[ft]
        keep = (d >= -1.0) & (d <= 1.0)
        q = ((d[keep] + 1.0) * qscale).astype(np.int64)
        key = (q << _TID_BITS) | ids[ft[keep]]
        np.minimum.at(kbuf, py[keep] * W + px[keep], key)
    hit = kbuf != INF
    zbuf = np.full(H * W, np.inf, np.float32)
    tbuf = np.full(H * W, -1, np.int32)
    zbuf[hit] = ((kbuf[hit] >> _TID_BITS).astype(np.float64) / qscale - 1.0).astype(np.float32)
    tbuf[hit] = (kbuf[hit] & ((1 << _TID_BITS) - 1)).astype(np.int32)
    return zbuf.reshape(H, W), tbuf.reshape(H, W)


def _to_lin(c):
    return np.power(np.clip(c, 0, 1), 2.2)


def _to_srgb(c):
    return np.power(np.clip(c, 0, 1), 1 / 2.2)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


class NumpyRenderer:
    backend = "numpy"

    def __init__(self, mesh: RenderMesh, size=(1280, 960), ssaa: int = 1, shadow_res: int = 1024):
        self.mesh = mesh
        self.size = (int(size[0]), int(size[1]))
        self.ssaa = int(ssaa)
        self.shadow_res = shadow_res

    def release(self):
        pass

    def _shadow_lookup(self, smap, lvp, wpos, n):
        """PCF 3x3 shadow factor for world points (M,3) with normals (M,3)."""
        S = self.shadow_res
        p = wpos + n * 0.006
        q = np.column_stack([p, np.ones(len(p))]) @ lvp.T
        ndc = q[:, :3] / q[:, 3:4]
        sx, sy = (ndc[:, 0] + 1) * 0.5 * S, (1 - ndc[:, 1]) * 0.5 * S
        sz = ndc[:, 2] - 0.003
        acc = np.zeros(len(p))
        ix0, iy0 = np.floor(sx).astype(np.int64), np.floor(sy).astype(np.int64)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ix, iy = ix0 + di, iy0 + dj
                inb = (ix >= 0) & (ix < S) & (iy >= 0) & (iy < S)
                occ = np.full(len(p), np.inf)
                occ[inb] = smap[iy[inb], ix[inb]]
                acc += (sz <= occ)
        lit = acc / 9.0
        outside = (sx < 0) | (sx >= S) | (sy < 0) | (sy >= S) | (ndc[:, 2] > 1)
        lit[outside] = 1.0
        return lit

    def render(self, job: RenderJob, world_bounds=None):
        mesh, st, cam = self.mesh, job.style, job.camera
        W0, H0 = self.size
        W, H = W0 * self.ssaa, H0 * self.ssaa
        N = mesh.n_chunks
        colors = np.asarray(job.colors, np.float64).reshape(-1, 3) if N else np.zeros((0, 3))
        vis = np.ones(N, bool) if job.visible is None else np.asarray(job.visible, bool)
        # world-space geometry
        if job.xforms is not None and N:
            X = np.asarray(job.xforms, np.float64)[mesh.vchunk]
            wp = np.einsum("vij,vj->vi", X[:, :3, :3], mesh.pos) + X[:, :3, 3]
            wn = np.einsum("vij,vj->vi", X[:, :3, :3], mesh.nrm)
        else:
            wp, wn = mesh.pos.astype(np.float64), mesh.nrm.astype(np.float64)
        tsel = np.nonzero(vis[mesh.tchunk])[0] if len(mesh.tris) else np.zeros(0, np.int64)
        tris = mesh.tris[tsel].astype(np.int64)

        lo, hi = world_bounds if world_bounds is not None else mesh.bounds
        L = key_light_dir(cam)
        lvp = shadow_camera(lo, hi, L, job.ground_z)

        # main camera raster
        VP = cam.view_proj
        hp = np.column_stack([wp, np.ones(len(wp))]) @ VP.T
        wv = hp[:, 3]
        front = wv > 1e-6
        ndc = hp[:, :3] / np.where(front, wv, 1.0)[:, None]
        xy = np.column_stack([(ndc[:, 0] + 1) * 0.5 * W, (1 - ndc[:, 1]) * 0.5 * H])
        tk = front[tris].all(1)
        zv = ndc[:, 2] + 4e-5 * ((mesh.flags & 2) != 0)  # inner faces lose depth ties (as in GL)
        zbuf, tb = rasterize(xy, zv, tris[tk], W, H, tri_ids=tsel[tk].astype(np.int32))
        bmask = tb >= 0

        # shadow map
        smap = None
        if st.shadows and len(tris):
            S = self.shadow_res
            hs = np.column_stack([wp, np.ones(len(wp))]) @ lvp.T
            ns = hs[:, :3] / hs[:, 3:4]
            sxy = np.column_stack([(ns[:, 0] + 1) * 0.5 * S, (1 - ns[:, 1]) * 0.5 * S])
            smap, _ = rasterize(sxy, ns[:, 2], tris, S, S)

        # per-pixel world position (unproject)
        iy, ix = np.mgrid[0:H, 0:W]
        ndx = (ix + 0.5) / W * 2 - 1
        ndy = 1 - (iy + 0.5) / H * 2
        invVP = np.linalg.inv(VP)

        def unproject(zn, m=None):
            if m is None:
                q = np.stack([ndx, ndy, zn, np.ones_like(ndx)], -1) @ invVP.T
            else:
                q = np.stack([ndx[m], ndy[m], zn[m], np.ones(int(m.sum()))], -1) @ invVP.T
            with np.errstate(divide="ignore", invalid="ignore"):
                return q[..., :3] / q[..., 3:4]

        view = cam.view
        fwd = -view[2, :3]
        # sky
        t = np.clip(1 - (iy + 0.5) / H, 0, 1) ** 0.9
        sky = np.array(st.sky_bottom)[None, None] * (1 - t[..., None]) + np.array(st.sky_top)[None, None] * t[..., None]
        img = sky.copy()
        nd_n = np.zeros((H, W, 3))
        depth = np.full((H, W), 1e9)

        # ground (analytic plane)
        if st.ground_on:
            pn, pf = unproject(-np.ones_like(ndx)), unproject(np.ones_like(ndx))
            dz = pf[..., 2] - pn[..., 2]
            with np.errstate(divide="ignore", invalid="ignore"):
                tt = (job.ground_z - pn[..., 2]) / dz
            gmask = (tt >= 0) & (tt <= 1) & np.isfinite(tt)
            tt = np.where(gmask, tt, 0.0)
            gp = pn + tt[..., None] * (pf - pn)
            gq = np.concatenate([gp, np.ones((H, W, 1))], -1) @ VP.T
            gz = gq[..., 2] / gq[..., 3]
            gmask &= ~(bmask & (zbuf < gz))
            if gmask.any():
                c = job.ground_center if job.ground_center is not None else 0.5 * (lo + hi)[:2]
                gr = job.ground_radius if job.ground_radius is not None else max(cam.scale, 1.0)
                gxy = gp[..., :2]
                alb = np.ones((H, W)) * 1.0
                for s, a in ((st.grid_minor, 0.07), (st.grid_major, 0.10)):
                    g = gxy / s
                    fw = np.abs(np.gradient(g[..., 0], axis=1)) + np.abs(np.gradient(g[..., 0], axis=0))
                    fh = np.abs(np.gradient(g[..., 1], axis=1)) + np.abs(np.gradient(g[..., 1], axis=0))
                    fx = np.abs(np.mod(g[..., 0] - 0.5, 1) - 0.5) / np.maximum(fw, 1e-5)
                    fy = np.abs(np.mod(g[..., 1] - 0.5, 1) - 0.5) / np.maximum(fh, 1e-5)
                    line = 1 - np.minimum(np.minimum(fx, fy), 1)
                    line *= 1 - _smoothstep(0.15, 0.5, np.maximum(fw, fh))
                    alb -= a * np.nan_to_num(line)
                gcol = np.array(st.ground)[None, None] * alb[..., None]
                if smap is not None:
                    pts = gp[gmask]
                    sh = np.ones((H, W))
                    sh[gmask] = self._shadow_lookup(smap, lvp, pts, np.tile([0, 0, 1.0], (len(pts), 1)))
                    gcol = gcol * (1 - 0.55 * st.shadow_strength * (1 - sh))[..., None]
                dist = np.linalg.norm(gxy - np.asarray(c)[None, None], axis=-1)
                fade = _smoothstep(1.3 * gr, 3.0 * gr, dist)[..., None]
                gcol = gcol * (1 - fade) + sky * fade
                img[gmask] = gcol[gmask]
                gd = -(np.concatenate([gp, np.ones((H, W, 1))], -1) @ view.T)[..., 2]
                depth[gmask] = gd[gmask]
                nd_n[gmask] = view[:3, :3] @ np.array([0, 0, 1.0])

        # building shading
        if bmask.any():
            tid = tb[bmask]
            v0 = mesh.tris[tid, 0]
            ch = mesh.tchunk[tid]
            n = wn[v0]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            P = unproject(zbuf.astype(np.float64), bmask)
            V = -fwd[None] if cam.ortho else (cam.eye[None] - P)
            flip = (n * V).sum(1) < 0
            n[flip] *= -1
            inner = (mesh.flags[v0] & 2) != 0
            alb = _to_lin(colors[ch]) * np.where(inner, st.inner_factor, 1.0)[:, None]
            ndl = np.maximum(n @ L, 0)
            sh = np.ones(len(n))
            if smap is not None:
                m = ndl > 0
                sh[m] = self._shadow_lookup(smap, lvp, P[m], n[m])
            sh = (1 - st.shadow_strength) + st.shadow_strength * sh
            hemi = np.array(st.hemi_ground)[None] * (1 - (n[:, 2:3] * 0.5 + 0.5)) + \
                np.array(st.hemi_sky)[None] * (n[:, 2:3] * 0.5 + 0.5)
            ao = 0.72 + 0.28 * _smoothstep(0.0, 1.8, P[:, 2] - job.ground_z)
            c = alb * (hemi * ao[:, None] + (st.key * ndl * sh)[:, None] * np.array([1.0, 0.975, 0.94])[None])
            c = c / (1 + 0.18 * c)
            img[bmask] = _to_srgb(c)
            nd_n[bmask] = n @ view[:3, :3].T
            depth[bmask] = -(np.column_stack([P, np.ones(len(P))]) @ view.T)[:, 2]

        # edges
        idb = np.where(bmask, tb, -1)
        cid = np.where(bmask, mesh.tchunk[np.maximum(tb, 0)], -1)
        e = self._edges(bmask, cid, nd_n, depth, cam, st)
        img = img * (1 - (e * st.line_alpha)[..., None]) + np.array(st.line_color)[None, None] * (e * st.line_alpha)[..., None]
        out = np.clip(img, 0, 1)
        cov = bmask.astype(np.float64)
        if self.ssaa > 1:
            s = self.ssaa
            out = out.reshape(H0, s, W0, s, 3).mean((1, 3))
            cov = cov.reshape(H0, s, W0, s).mean((1, 3))
        del idb
        return (out * 255 + 0.5).astype(np.uint8), cov > 0.5

    @staticmethod
    def _edges(b, cid, n, d, cam, st):
        H, W = b.shape
        e = np.zeros((H, W))

        def sh(a, dy, dx):
            # value of neighbor (y+dy, x+dx) with edge clamping
            p = np.pad(a, [(1, 1), (1, 1)] + [(0, 0)] * (a.ndim - 2), mode="edge")
            return p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]

        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            bq, nq, cq = sh(b, dy, dx), sh(n, dy, dx), sh(cid, dy, dx)
            sil = b != bq
            both = b & bq
            crease = both & ((n * nq).sum(-1) < st.normal_thresh)
            e = np.maximum(e, (sil | crease).astype(float))
            if st.chunk_lines > 0:
                e = np.maximum(e, (both & (cid != cq)) * st.chunk_lines)
        aff = d if cam.ortho else 1.0 / np.maximum(d, 1e-6)
        for dy, dx in ((0, 1), (1, 0)):
            ba, bb = sh(b, -dy, -dx), sh(b, dy, dx)
            la, lb = sh(aff, -dy, -dx), sh(aff, dy, dx)
            lap = np.abs(la + lb - 2 * aff)
            thr = st.depth_thresh * cam.scale if cam.ortho else st.depth_thresh * aff
            e = np.maximum(e, (b & ba & bb & (lap > thr)).astype(float))
        return e
