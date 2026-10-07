"""Headless OpenGL renderer (moderngl standalone context, GL 3.3 core).

Pipeline per job:
  1. shadow map: depth-only ortho pass along the key light
  2. G-buffer at `ssaa`× resolution: sky gradient, ground plane (grid + shadow + fade), chunks
     (Lambert key + hemispheric ambient, inner fracture faces darker, contact darkening)
     -> color (sRGB), view normal + linear depth, chunk id
  3. post: normal / depth-laplacian / silhouette edge detection per subsample, line compositing
     and box downsample to the final resolution in one pass (SSAA = anti-aliasing).
"""
from __future__ import annotations

import numpy as np

from .scene import RenderJob, RenderMesh, Style, key_light_dir, shadow_camera

_VS_MESH = """
#version 330
uniform sampler2D xf_tex;
uniform sampler2D col_tex;
uniform int cpr;
uniform int use_xf;
uniform mat4 vp;
in vec3 in_pos; in vec3 in_nrm; in int in_chunk; in float in_flags;
out vec3 v_wpos; out vec3 v_wn; out vec3 v_col; out float v_inner; out float v_id;
void main() {
    ivec2 cc = ivec2(in_chunk % cpr, in_chunk / cpr);
    vec4 col = texelFetch(col_tex, cc, 0);
    mat4 M = mat4(1.0);
    if (use_xf == 1) {
        int x = (in_chunk % cpr) * 4; int y = in_chunk / cpr;
        M = transpose(mat4(texelFetch(xf_tex, ivec2(x, y), 0), texelFetch(xf_tex, ivec2(x + 1, y), 0),
                           texelFetch(xf_tex, ivec2(x + 2, y), 0), texelFetch(xf_tex, ivec2(x + 3, y), 0)));
    }
    vec4 wp = M * vec4(in_pos, 1.0);
    v_wpos = wp.xyz;
    v_wn = mat3(M) * in_nrm;
    v_col = col.rgb;
    v_inner = mod(floor(in_flags / 2.0 + 0.01), 2.0);
    v_id = float(in_chunk + 1);
    gl_Position = vp * wp;
    // inner (fracture) faces lose depth ties against coplanar caps -> no edge-on speckles
    if (v_inner > 0.5) gl_Position.z += 4e-5 * gl_Position.w;
    if (col.a < 0.5) gl_Position = vec4(0.0, 0.0, -10.0, 1.0);
}
"""

_FS_SHADOW = """
#version 330
void main() {}
"""

_COMMON_FS = """
uniform vec3 light_dir;
uniform float key;
uniform vec3 hemi_sky;
uniform vec3 hemi_ground;
uniform sampler2DShadow shadow_tex;
uniform mat4 light_vp;
uniform int use_shadow;
uniform float shadow_texel;
uniform float shadow_bias;
uniform mat4 view;
uniform vec3 eye;
uniform vec3 view_fwd;
uniform int is_ortho;
uniform vec2 viewport;
uniform vec3 sky_top;
uniform vec3 sky_bottom;
vec3 to_lin(vec3 c) { return pow(c, vec3(2.2)); }
vec3 to_srgb(vec3 c) { return pow(clamp(c, 0.0, 1.0), vec3(1.0 / 2.2)); }
vec3 sky_at(vec2 fc) {
    float t = clamp(fc.y / viewport.y, 0.0, 1.0);
    return mix(sky_bottom, sky_top, pow(t, 0.9));
}
float shadow_at(vec3 wp, vec3 n) {
    if (use_shadow == 0) return 1.0;
    vec4 l = light_vp * vec4(wp + n * shadow_bias * 4.0, 1.0);
    vec3 s = l.xyz / l.w * 0.5 + 0.5;
    if (s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0 || s.z > 1.0) return 1.0;
    float acc = 0.0;
    for (int i = -2; i <= 2; ++i)
        for (int j = -2; j <= 2; ++j)
            acc += texture(shadow_tex, vec3(s.xy + vec2(i, j) * shadow_texel, s.z - shadow_bias));
    return acc / 25.0;
}
"""

_FS_MESH = """
#version 330
""" + _COMMON_FS + """
uniform float inner_factor;
uniform float ground_z;
uniform float shadow_strength;
in vec3 v_wpos; in vec3 v_wn; in vec3 v_col; in float v_inner; in float v_id;
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_nd;
layout(location = 2) out float o_id;
void main() {
    vec3 n = normalize(v_wn);
    vec3 V = is_ortho == 1 ? -view_fwd : normalize(eye - v_wpos);
    if (dot(n, V) < 0.0) n = -n;
    vec3 alb = to_lin(v_col) * (v_inner > 0.5 ? inner_factor : 1.0);
    float ndl = max(dot(n, light_dir), 0.0);
    float sh = ndl > 0.0 ? shadow_at(v_wpos, n) : 1.0;
    sh = mix(1.0 - shadow_strength, 1.0, sh);
    vec3 hemi = mix(hemi_ground, hemi_sky, n.z * 0.5 + 0.5);
    float ao = mix(0.72, 1.0, smoothstep(0.0, 1.8, v_wpos.z - ground_z));
    vec3 c = alb * (hemi * ao + key * ndl * sh * vec3(1.0, 0.975, 0.94));
    c = c / (1.0 + 0.18 * c);  // soft shoulder
    o_col = vec4(to_srgb(c), 1.0);
    o_nd = vec4(mat3(view) * n, -(view * vec4(v_wpos, 1.0)).z);
    o_id = v_id;
}
"""

_VS_GROUND = """
#version 330
uniform mat4 vp;
in vec3 in_pos;
out vec3 v_wpos;
void main() { v_wpos = in_pos; gl_Position = vp * vec4(in_pos, 1.0); }
"""

_FS_GROUND = """
#version 330
""" + _COMMON_FS + """
uniform vec3 ground_col;
uniform float grid_minor;
uniform float grid_major;
uniform vec2 fade_center;
uniform vec2 fade_r;
uniform float shadow_strength;
in vec3 v_wpos;
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_nd;
layout(location = 2) out float o_id;
float grid(vec2 p, float s) {
    vec2 g = p / s;
    vec2 w = max(fwidth(g), vec2(1e-5));
    vec2 f = abs(fract(g - 0.5) - 0.5) / w;
    float l = 1.0 - min(min(f.x, f.y), 1.0);
    return l * (1.0 - smoothstep(0.15, 0.5, max(w.x, w.y)));  // fade when too dense
}
void main() {
    vec3 alb = ground_col;
    alb *= 1.0 - 0.07 * grid(v_wpos.xy, grid_minor) - 0.10 * grid(v_wpos.xy, grid_major);
    float sh = shadow_at(v_wpos, vec3(0.0, 0.0, 1.0));
    vec3 c = alb * mix(1.0 - 0.55 * shadow_strength, 1.0, sh);
    float d = length(v_wpos.xy - fade_center);
    float fade = smoothstep(fade_r.x, fade_r.y, d);
    c = mix(c, sky_at(gl_FragCoord.xy), fade);
    o_col = vec4(c, 1.0);
    o_nd = vec4(mat3(view) * vec3(0.0, 0.0, 1.0), -(view * vec4(v_wpos, 1.0)).z);
    o_id = 0.0;
}
"""

_VS_FULL = """
#version 330
in vec2 in_xy;
void main() { gl_Position = vec4(in_xy, 0.0, 1.0); }
"""

_FS_SKY = """
#version 330
""" + _COMMON_FS + """
layout(location = 0) out vec4 o_col;
layout(location = 1) out vec4 o_nd;
layout(location = 2) out float o_id;
void main() {
    o_col = vec4(sky_at(gl_FragCoord.xy), 1.0);
    o_nd = vec4(0.0, 0.0, 0.0, 1e9);
    o_id = 0.0;
}
"""

_FS_POST = """
#version 330
uniform sampler2D t_col;
uniform sampler2D t_nd;
uniform sampler2D t_id;
uniform int ss;
uniform int is_ortho;
uniform float scale;
uniform float n_thr;
uniform float d_thr;
uniform float chunk_lines;
uniform vec3 line_col;
uniform float line_alpha;
out vec4 frag;
ivec2 sz;
float aff(float d) { return is_ortho == 1 ? d : 1.0 / max(d, 1e-6); }
bool bld(ivec2 p) { return texelFetch(t_id, p, 0).r > 0.5; }
float edge_at(ivec2 p) {
    vec4 c = texelFetch(t_nd, p, 0);
    float ic = texelFetch(t_id, p, 0).r;
    bool bc = ic > 0.5;
    float e = 0.0;
    ivec2 offs[4] = ivec2[](ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1));
    for (int k = 0; k < 4; ++k) {
        ivec2 q = clamp(p + offs[k], ivec2(0), sz - 1);
        float iq = texelFetch(t_id, q, 0).r;
        bool bq = iq > 0.5;
        if (!bc && !bq) continue;
        if (bc != bq) { e = 1.0; continue; }
        vec4 n = texelFetch(t_nd, q, 0);
        if (dot(c.xyz, n.xyz) < n_thr) e = 1.0;
        if (abs(ic - iq) > 0.5) e = max(e, chunk_lines);
    }
    if (bc) {
        for (int ax = 0; ax < 2; ++ax) {
            ivec2 o = ax == 0 ? ivec2(1, 0) : ivec2(0, 1);
            ivec2 a = clamp(p - o, ivec2(0), sz - 1), b = clamp(p + o, ivec2(0), sz - 1);
            if (!bld(a) || !bld(b)) continue;
            float la = aff(texelFetch(t_nd, a, 0).w), lb = aff(texelFetch(t_nd, b, 0).w), lc = aff(c.w);
            float lap = abs(la + lb - 2.0 * lc);
            float thr = is_ortho == 1 ? d_thr * scale : d_thr * lc;
            if (lap > thr) e = 1.0;
        }
    }
    return e;
}
void main() {
    sz = textureSize(t_col, 0);
    ivec2 base = ivec2(gl_FragCoord.xy) * ss;
    vec3 acc = vec3(0.0);
    float cov = 0.0;
    for (int i = 0; i < ss; ++i)
        for (int j = 0; j < ss; ++j) {
            ivec2 p = base + ivec2(i, j);
            vec3 col = texelFetch(t_col, p, 0).rgb;
            col = mix(col, line_col, edge_at(p) * line_alpha);
            acc += col;
            cov += bld(p) ? 1.0 : 0.0;
        }
    float k = float(ss * ss);
    frag = vec4(acc / k, cov / k);
}
"""


def _m(a: np.ndarray) -> bytes:
    return np.ascontiguousarray(np.asarray(a, np.float32).T).tobytes()


class GLRenderer:
    """Offscreen renderer bound to one RenderMesh. `render(job) -> (rgb uint8 HxWx3, mask HxW)`."""

    backend = "gl"

    def __init__(self, mesh: RenderMesh, size=(1280, 960), ssaa: int = 2, shadow_res: int = 2048):
        import moderngl
        self.mgl = moderngl
        self.mesh = mesh
        self.size = (int(size[0]), int(size[1]))
        self.ssaa = int(ssaa)
        self.ctx = ctx = moderngl.create_standalone_context(require=330)
        W, H = self.size
        sW, sH = W * self.ssaa, H * self.ssaa
        self.p_mesh = ctx.program(vertex_shader=_VS_MESH, fragment_shader=_FS_MESH)
        self.p_shadow = ctx.program(vertex_shader=_VS_MESH, fragment_shader=_FS_SHADOW)
        self.p_ground = ctx.program(vertex_shader=_VS_GROUND, fragment_shader=_FS_GROUND)
        self.p_sky = ctx.program(vertex_shader=_VS_FULL, fragment_shader=_FS_SKY)
        self.p_post = ctx.program(vertex_shader=_VS_FULL, fragment_shader=_FS_POST)

        # geometry
        V = len(mesh.pos)
        vd = np.zeros(V, dtype=[("p", "f4", 3), ("n", "f4", 3), ("c", "i4"), ("f", "f4")])
        vd["p"], vd["n"], vd["c"], vd["f"] = mesh.pos, mesh.nrm, mesh.vchunk, mesh.flags.astype(np.float32)
        self.n_idx = mesh.tris.size
        if self.n_idx:
            self.vbo = ctx.buffer(vd.tobytes())
            self.ibo = ctx.buffer(np.ascontiguousarray(mesh.tris, np.uint32).tobytes())
            content = [(self.vbo, "3f 3f 1i 1f", "in_pos", "in_nrm", "in_chunk", "in_flags")]
            self.vao_mesh = ctx.vertex_array(self.p_mesh, content, self.ibo)
            self.vao_shadow = ctx.vertex_array(self.p_shadow, [(self.vbo, "3f 12x 1i 4x", "in_pos", "in_chunk")], self.ibo)
        tri = np.array([[-1, -1], [3, -1], [-1, 3]], np.float32)
        self.fs_vbo = ctx.buffer(tri.tobytes())
        self.vao_sky = ctx.vertex_array(self.p_sky, [(self.fs_vbo, "2f", "in_xy")])
        self.vao_post = ctx.vertex_array(self.p_post, [(self.fs_vbo, "2f", "in_xy")])
        self.g_vbo = ctx.buffer(reserve=6 * 12)
        self.vao_ground = ctx.vertex_array(self.p_ground, [(self.g_vbo, "3f", "in_pos")])

        # per-chunk data textures
        N = max(mesh.n_chunks, 1)
        self.cpr = min(N, 1024)
        self.rows = (N + self.cpr - 1) // self.cpr
        self.t_xf = ctx.texture((self.cpr * 4, self.rows), 4, dtype="f4")
        self.t_cc = ctx.texture((self.cpr, self.rows), 4, dtype="f4")
        for t in (self.t_xf, self.t_cc):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)

        # targets
        self.t_col = ctx.texture((sW, sH), 4)
        self.t_nd = ctx.texture((sW, sH), 4, dtype="f4")
        self.t_id = ctx.texture((sW, sH), 1, dtype="f4")
        for t in (self.t_col, self.t_nd, self.t_id):
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.rb_depth = ctx.depth_renderbuffer((sW, sH))
        self.fbo_g = ctx.framebuffer([self.t_col, self.t_nd, self.t_id], self.rb_depth)
        self.t_out = ctx.texture((W, H), 4)
        self.fbo_out = ctx.framebuffer([self.t_out])
        self.shadow_res = shadow_res
        self.t_sh = ctx.depth_texture((shadow_res, shadow_res))
        self.t_sh.compare_func = "<="
        self.t_sh.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.t_sh.repeat_x = self.t_sh.repeat_y = False
        self.fbo_sh = ctx.framebuffer(depth_attachment=self.t_sh)

    # ------------------------------------------------------------------ helpers
    def _upload_chunks(self, job: RenderJob):
        N = self.mesh.n_chunks
        tot = self.cpr * self.rows
        cc = np.zeros((tot, 4), np.float32)
        if N:
            cc[:N, :3] = job.colors
            cc[:N, 3] = 1.0 if job.visible is None else job.visible.astype(np.float32)
        self.t_cc.write(cc.reshape(self.rows, self.cpr, 4).tobytes())
        if job.xforms is not None:
            xf = np.zeros((tot, 4, 4), np.float32)
            xf[:] = np.eye(4, dtype=np.float32)
            xf[:N] = job.xforms
            self.t_xf.write(xf.reshape(self.rows, self.cpr * 4, 4).tobytes())

    def _set(self, prog, **kw):
        for k, v in kw.items():
            if k in prog:
                prog[k].value = v

    def _setm(self, prog, **kw):
        for k, v in kw.items():
            if k in prog:
                prog[k].write(_m(v))

    # ------------------------------------------------------------------ main
    def render(self, job: RenderJob, world_bounds=None):
        mgl, ctx, mesh, st = self.mgl, self.ctx, self.mesh, job.style
        cam = job.camera
        W, H = self.size
        sW, sH = W * self.ssaa, H * self.ssaa
        self._upload_chunks(job)
        lo, hi = world_bounds if world_bounds is not None else mesh.bounds
        L = key_light_dir(cam)
        lvp = shadow_camera(lo, hi, L, job.ground_z)
        R = cam.scale
        use_xf = int(job.xforms is not None)
        common = dict(light_dir=tuple(L), key=st.key, hemi_sky=st.hemi_sky, hemi_ground=st.hemi_ground,
                      use_shadow=int(st.shadows), shadow_texel=1.0 / self.shadow_res,
                      shadow_bias=0.0015, eye=tuple(cam.eye),
                      view_fwd=tuple(-cam.view[2, :3]), is_ortho=int(cam.ortho), viewport=(sW, sH),
                      sky_top=tuple(st.sky_top), sky_bottom=tuple(st.sky_bottom),
                      shadow_strength=st.shadow_strength)
        for p in (self.p_mesh, self.p_ground, self.p_sky):
            self._set(p, **common)
            self._setm(p, light_vp=lvp, view=cam.view)
            if "shadow_tex" in p:
                p["shadow_tex"].value = 2
        for p in (self.p_mesh, self.p_shadow):
            self._set(p, cpr=self.cpr, use_xf=use_xf, xf_tex=0, col_tex=1)

        self.t_xf.use(0)
        self.t_cc.use(1)
        # 1. shadow pass
        if st.shadows and self.n_idx:
            self.fbo_sh.use()
            self.fbo_sh.clear(depth=1.0)
            ctx.enable(mgl.DEPTH_TEST)
            self._setm(self.p_shadow, vp=lvp)
            self.vao_shadow.render(mgl.TRIANGLES)
        # 2. G-buffer
        self.fbo_g.use()
        self.fbo_g.clear(0.0, 0.0, 0.0, 0.0, depth=1.0)
        self.t_sh.use(2)
        ctx.disable(mgl.DEPTH_TEST)
        self.vao_sky.render(mgl.TRIANGLES, vertices=3)
        ctx.enable(mgl.DEPTH_TEST)
        vp = cam.view_proj
        if st.ground_on:
            c = job.ground_center if job.ground_center is not None else 0.5 * (lo + hi)[:2]
            gr = job.ground_radius if job.ground_radius is not None else max(R, 1.0)
            ext = 12 * gr
            z = job.ground_z - 1e-3
            q = np.array([[c[0] - ext, c[1] - ext, z], [c[0] + ext, c[1] - ext, z], [c[0] + ext, c[1] + ext, z],
                          [c[0] - ext, c[1] - ext, z], [c[0] + ext, c[1] + ext, z], [c[0] - ext, c[1] + ext, z]],
                         np.float32)
            self.g_vbo.write(q.tobytes())
            self._set(self.p_ground, ground_col=tuple(st.ground), grid_minor=st.grid_minor,
                      grid_major=st.grid_major, fade_center=(float(c[0]), float(c[1])),
                      fade_r=(1.3 * gr, 3.0 * gr))
            self._setm(self.p_ground, vp=vp)
            self.vao_ground.render(mgl.TRIANGLES)
        if self.n_idx:
            self._set(self.p_mesh, inner_factor=st.inner_factor, ground_z=job.ground_z)
            self._setm(self.p_mesh, vp=vp)
            self.vao_mesh.render(mgl.TRIANGLES)
        # 3. post: edges + downsample
        ctx.disable(mgl.DEPTH_TEST)
        self.fbo_out.use()
        self.t_col.use(3); self.t_nd.use(4); self.t_id.use(5)
        self._set(self.p_post, t_col=3, t_nd=4, t_id=5, ss=self.ssaa, is_ortho=int(cam.ortho), scale=R,
                  n_thr=st.normal_thresh, d_thr=st.depth_thresh, chunk_lines=st.chunk_lines,
                  line_col=tuple(st.line_color), line_alpha=st.line_alpha)
        self.vao_post.render(mgl.TRIANGLES, vertices=3)
        raw = np.frombuffer(self.fbo_out.read(components=4), np.uint8).reshape(H, W, 4)[::-1]
        return raw[..., :3].copy(), raw[..., 3] > 127

    def release(self):
        try:
            self.ctx.release()
        except Exception:
            pass
