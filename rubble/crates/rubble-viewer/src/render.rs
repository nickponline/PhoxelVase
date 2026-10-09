//! Chunk rendering. Every alive chunk is drawn by exactly one *render group*:
//!
//! * `Section(b, s)`: static chunks of building `b` in id range `[s*N, (s+1)*N)`, one mesh,
//!   transform = building pose. Splitting a building into sections keeps rebuilds local
//!   (a 50k-chunk building never rebuilds 50k chunks because one wall chunk died).
//! * `Cluster(id)`: one entity per dynamic cluster; mesh in building space, transform from the
//!   cluster's rigid body every frame.
//! * `Frozen(b, pose)`: settled rubble that shares one world pose.
//!
//! Ownership is reconciled each frame by diffing `Building::state` against a cached copy (cheap,
//! O(chunks) compares), so the viewer never depends on event ordering. A group whose membership
//! changed is marked dirty and rebuilt at most once per frame.
use crate::coords::pose_to_transform;
use crate::{Overlays, Sim};
use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use rubble_core::world::cluster_id;
use rubble_core::{ChunkState, ClusterId, Pose};
use rubble_format::{Bld, MeshVertex, FACE_INNER, FACE_OUTER};
use std::collections::HashMap;

pub const SECTION_CHUNKS: u32 = 2048;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum GroupKey {
    Section(u32, u32),
    Cluster(u64),
    Frozen(u32, u64),
}

pub struct Group {
    pub building: u32,
    pub chunks: Vec<u32>,
    pub dirty: bool,
    pub entity: Option<Entity>,
    pub sleeping: bool,
}

/// Immutable per-building render data: per-chunk vertex/index runs (copied from MVRT/MIDX, or
/// synthesized from the convex hull when a chunk has no render mesh, e.g. testutil buildings).
pub struct BuildingRender {
    pub verts: Vec<MeshVertex>,
    pub idx: Vec<u32>,
    /// per chunk: (first vertex, vertex count, first index, index count); indices are chunk-local
    pub runs: Vec<(u32, u32, u32, u32)>,
}

#[derive(Resource, Default)]
pub struct RenderState {
    pub groups: HashMap<GroupKey, Group>,
    pub owner: Vec<Vec<Option<GroupKey>>>,
    pub cached: Vec<Vec<Option<ChunkState>>>,
    pub buildings: Vec<BuildingRender>,
    /// rebuild everything (colors changed / reset)
    pub all_dirty: bool,
    pub last_rebuild_ms: f32,
    pub last_rebuilt: usize,
    pub last_sync_ms: f32,
}

#[derive(Resource)]
pub struct Materials {
    pub base: Handle<StandardMaterial>,
    pub awake: Handle<StandardMaterial>,
    pub asleep: Handle<StandardMaterial>,
    pub frozen: Handle<StandardMaterial>,
}

#[derive(Component)]
#[allow(dead_code)] // key kept for debugging / inspection
pub struct ChunkGroup(pub GroupKey);

/// `base_color` of each chunk material (multiplied by the per-chunk vertex colour); also used by
/// the dynamic-tint legend in the help box.
pub const TINT_BASE: Color = Color::WHITE;
pub const TINT_AWAKE: Color = Color::srgb(1.0, 0.55, 0.25);
pub const TINT_ASLEEP: Color = Color::srgb(0.35, 0.55, 1.0);
pub const TINT_FROZEN: Color = Color::srgb(0.7, 0.55, 0.8);

impl Materials {
    pub fn new(mats: &mut Assets<StandardMaterial>) -> Self {
        let mk = |c: Color| StandardMaterial { base_color: c, perceptual_roughness: 0.85, reflectance: 0.3, ..default() };
        Materials {
            base: mats.add(mk(TINT_BASE)),
            awake: mats.add(mk(TINT_AWAKE)),
            asleep: mats.add(mk(TINT_ASLEEP)),
            frozen: mats.add(mk(TINT_FROZEN)),
        }
    }
}

fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846ca68b);
    x ^= x >> 16;
    x
}

fn material_srgb(mat: u16) -> [f32; 3] {
    match mat {
        0 => [0.66, 0.65, 0.62], // concrete
        1 => [0.66, 0.34, 0.25], // brick
        2 => [0.60, 0.45, 0.28], // wood
        3 => [0.48, 0.52, 0.58], // metal
        4 => [0.62, 0.80, 0.90], // glass
        _ => [0.7, 0.7, 0.7],
    }
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = Color::hsv(h * 360.0, s, v).to_srgba();
    [c.red, c.green, c.blue]
}

/// sRGB color of a chunk's outer faces (vertex color; inner faces are darkened).
fn chunk_srgb(b: u32, c: u32, bld: &Bld, random: bool) -> [f32; 3] {
    let h = hash32(c.wrapping_mul(2654435761) ^ b.wrapping_mul(97));
    if random {
        return hsv((h & 0xffff) as f32 / 65535.0, 0.55 + 0.35 * ((h >> 16) & 0xff) as f32 / 255.0, 0.9);
    }
    let base = material_srgb(bld.chunks[c as usize].material);
    let j = 0.94 + 0.08 * ((h >> 8) & 0xff) as f32 / 255.0; // subtle per-chunk variation
    [base[0] * j, base[1] * j, base[2] * j]
}

impl BuildingRender {
    pub fn new(bld: &Bld) -> Self {
        let mut r = BuildingRender { verts: vec![], idx: vec![], runs: Vec::with_capacity(bld.chunks.len()) };
        for c in 0..bld.chunks.len() {
            let v0 = r.verts.len() as u32;
            let i0 = r.idx.len() as u32;
            let ids = bld.mesh_indices_of(c);
            if !ids.is_empty() {
                let lo = ids.iter().copied().min().unwrap();
                let hi = ids.iter().copied().max().unwrap();
                r.verts.extend_from_slice(&bld.mesh_verts[lo as usize..=hi as usize]);
                r.idx.extend(ids.iter().map(|&i| i - lo));
            } else {
                hull_mesh(bld.hull_verts_of(c), bld.hull_planes_of(c), &mut r.verts, &mut r.idx, v0);
            }
            r.runs.push((v0, r.verts.len() as u32 - v0, i0, r.idx.len() as u32 - i0));
        }
        r
    }
}

/// Facet a convex hull: for each plane (n·x + d = 0, outward n) gather the on-plane vertices,
/// sort them by angle and fan-triangulate. Indices are relative to `v0`.
fn hull_mesh(hv: &[[f32; 3]], hp: &[[f32; 4]], verts: &mut Vec<MeshVertex>, idx: &mut Vec<u32>, v0: u32) {
    let scale = hv.iter().flat_map(|v| v.iter()).fold(0f32, |m, x| m.max(x.abs())).max(1.0);
    for p in hp {
        let n = Vec3::new(p[0], p[1], p[2]);
        let on: Vec<Vec3> = hv
            .iter()
            .map(|v| Vec3::from_array(*v))
            .filter(|v| (n.dot(*v) + p[3]).abs() < 1e-4 * scale)
            .collect();
        if on.len() < 3 {
            continue;
        }
        let c = on.iter().copied().sum::<Vec3>() / on.len() as f32;
        let u = n.any_orthonormal_vector();
        let w = n.cross(u);
        let mut ring: Vec<(f32, Vec3)> = on.iter().map(|v| ((*v - c).dot(w).atan2((*v - c).dot(u)), *v)).collect();
        ring.sort_by(|a, b| a.0.total_cmp(&b.0));
        let base = verts.len() as u32 - v0;
        let nrm = [(n.x * 32767.0) as i16, (n.y * 32767.0) as i16, (n.z * 32767.0) as i16];
        for (_, v) in &ring {
            verts.push(MeshVertex { pos: v.to_array(), nrm, face_flags: FACE_OUTER });
        }
        for k in 1..ring.len() as u32 - 1 {
            idx.extend_from_slice(&[base, base + k, base + k + 1]);
        }
    }
}

/// Mesh of `chunks` in building space (Z-up; the entity transform rotates it to Y-up).
pub fn build_mesh(b: u32, bld: &Bld, br: &BuildingRender, chunks: &[u32], random_colors: bool) -> Option<Mesh> {
    let nv: usize = chunks.iter().map(|&c| br.runs[c as usize].1 as usize).sum();
    let ni: usize = chunks.iter().map(|&c| br.runs[c as usize].3 as usize).sum();
    if ni == 0 {
        return None;
    }
    let mut pos: Vec<[f32; 3]> = Vec::with_capacity(nv);
    let mut nrm: Vec<[f32; 3]> = Vec::with_capacity(nv);
    let mut col: Vec<[f32; 4]> = Vec::with_capacity(nv);
    let mut idx: Vec<u32> = Vec::with_capacity(ni);
    for &c in chunks {
        let (v0, vn, i0, inn) = br.runs[c as usize];
        if inn == 0 {
            continue;
        }
        let base = pos.len() as u32;
        let s = chunk_srgb(b, c, bld, random_colors);
        let outer = Color::srgb(s[0], s[1], s[2]).to_linear().to_f32_array();
        let inner = Color::srgb(s[0] * 0.62, s[1] * 0.62, s[2] * 0.62).to_linear().to_f32_array();
        for v in &br.verts[v0 as usize..(v0 + vn) as usize] {
            pos.push(v.pos);
            nrm.push([v.nrm[0] as f32 / 32767.0, v.nrm[1] as f32 / 32767.0, v.nrm[2] as f32 / 32767.0]);
            col.push(if v.face_flags & FACE_INNER != 0 { inner } else { outer });
        }
        idx.extend(br.idx[i0 as usize..(i0 + inn) as usize].iter().map(|&i| base + i));
    }
    Some(
        Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
            .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
            .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nrm)
            .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, col)
            .with_inserted_indices(Indices::U32(idx)),
    )
}

pub fn pose_xf(p: &Pose) -> Transform {
    pose_to_transform(p.translation.to_array(), p.rotation.to_array())
}

fn pose_hash(p: &Pose) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for f in p.translation.to_array().iter().chain(p.rotation.to_array().iter()) {
        h ^= f.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl RenderState {
    pub fn reset(&mut self, sim: &Sim) {
        self.groups.clear();
        self.buildings = sim.world.buildings.iter().map(|b| BuildingRender::new(&b.bld)).collect();
        self.owner = sim.world.buildings.iter().map(|b| vec![None; b.n_chunks()]).collect();
        self.cached = sim.world.buildings.iter().map(|b| vec![None; b.n_chunks()]).collect();
        self.all_dirty = true;
    }
}

fn group_key(b: u32, c: u32, st: ChunkState, frozen_pose: &Pose) -> Option<GroupKey> {
    match st {
        ChunkState::Static | ChunkState::Detaching => Some(GroupKey::Section(b, c / SECTION_CHUNKS)),
        ChunkState::InCluster(k) => Some(GroupKey::Cluster(cluster_id(k).0)),
        ChunkState::Frozen => Some(GroupKey::Frozen(b, pose_hash(frozen_pose))),
        ChunkState::Gone => None,
    }
}

/// Reconcile render groups with engine state, rebuild dirty meshes, update transforms.
#[allow(clippy::too_many_arguments)]
pub fn sync_render(
    mut commands: Commands,
    sim: Res<Sim>,
    mut rs: ResMut<RenderState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mats: Res<Materials>,
    overlays: Res<Overlays>,
    mut q: Query<(&mut Transform, &mut MeshMaterial3d<StandardMaterial>), With<ChunkGroup>>,
) {
    let t0 = std::time::Instant::now();
    let rs = &mut *rs;
    let w = &sim.world;
    // 1. diff chunk states -> ownership changes
    for (bi, bd) in w.buildings.iter().enumerate() {
        let b = bi as u32;
        let cached = &mut rs.cached[bi];
        let owner = &mut rs.owner[bi];
        for (ci, st) in bd.state.iter().enumerate() {
            if cached[ci] == Some(*st) {
                continue;
            }
            cached[ci] = Some(*st);
            let c = ci as u32;
            let new = group_key(b, c, *st, &bd.chunk_pose[ci]);
            if new == owner[ci] {
                continue; // Static <-> Detaching: same section
            }
            if let Some(old) = owner[ci] {
                if let Some(g) = rs.groups.get_mut(&old) {
                    g.dirty = true;
                }
            }
            owner[ci] = new;
            if let Some(k) = new {
                let g = rs.groups.entry(k).or_insert_with(|| Group {
                    building: b,
                    chunks: vec![],
                    dirty: true,
                    entity: None,
                    sleeping: false,
                });
                g.chunks.push(c);
                g.dirty = true;
            }
        }
    }
    let t1 = std::time::Instant::now();
    // 2. rebuild dirty groups
    let mut rebuilt = 0;
    let all = std::mem::take(&mut rs.all_dirty);
    let mut dead = vec![];
    for (key, g) in rs.groups.iter_mut() {
        if !(g.dirty || all) {
            continue;
        }
        g.dirty = false;
        let owner = &rs.owner[g.building as usize];
        g.chunks.retain(|&c| owner[c as usize] == Some(*key));
        g.chunks.sort_unstable();
        g.chunks.dedup();
        if g.chunks.is_empty() {
            if let Some(e) = g.entity.take() {
                commands.entity(e).despawn();
            }
            dead.push(*key);
            continue;
        }
        let bd = &w.buildings[g.building as usize];
        let br = &rs.buildings[g.building as usize];
        let Some(mesh) = build_mesh(g.building, &bd.bld, br, &g.chunks, overlays.random_colors) else {
            if let Some(e) = g.entity.take() {
                commands.entity(e).despawn();
            }
            continue;
        };
        let mesh = meshes.add(mesh);
        rebuilt += 1;
        let xf = match key {
            GroupKey::Section(..) => pose_xf(&bd.pose),
            GroupKey::Frozen(..) => pose_xf(&bd.chunk_pose[g.chunks[0] as usize]),
            GroupKey::Cluster(id) => match w.cluster_state(ClusterId(*id)) {
                Some((_, st)) => pose_xf(&st.pose),
                None => pose_xf(&bd.pose),
            },
        };
        let mat = group_material(key, g, &mats, &overlays);
        match g.entity {
            Some(e) => {
                commands.entity(e).insert(Mesh3d(mesh));
            }
            None => {
                g.entity = Some(commands.spawn((Mesh3d(mesh), MeshMaterial3d(mat), xf, ChunkGroup(*key))).id());
            }
        }
    }
    for k in dead {
        rs.groups.remove(&k);
    }
    let t2 = std::time::Instant::now();
    // 3. per-frame transforms + tints
    for (key, g) in rs.groups.iter_mut() {
        let Some(e) = g.entity else { continue };
        let Ok((mut xf, mut mat)) = q.get_mut(e) else { continue };
        if let GroupKey::Cluster(id) = key {
            if let Some((_, st)) = w.cluster_state(ClusterId(*id)) {
                let nx = pose_xf(&st.pose);
                if *xf != nx {
                    *xf = nx;
                }
                g.sleeping = st.sleeping;
            }
        }
        let want = group_material(key, g, &mats, &overlays);
        if mat.0 != want {
            mat.0 = want;
        }
    }
    rs.last_rebuilt = rebuilt;
    if rebuilt > 0 {
        rs.last_rebuild_ms = (t2 - t1).as_secs_f32() * 1e3;
    }
    rs.last_sync_ms = (t1 - t0).as_secs_f32() * 1e3;
}

fn group_material(key: &GroupKey, g: &Group, mats: &Materials, ov: &Overlays) -> Handle<StandardMaterial> {
    if !ov.sleep_tint {
        return mats.base.clone();
    }
    match key {
        GroupKey::Section(..) => mats.base.clone(),
        GroupKey::Frozen(..) => mats.frozen.clone(),
        GroupKey::Cluster(..) => {
            if g.sleeping {
                mats.asleep.clone()
            } else {
                mats.awake.clone()
            }
        }
    }
}

/// Despawn all chunk entities (world reset).
pub fn clear_entities(commands: &mut Commands, rs: &mut RenderState) {
    for g in rs.groups.values_mut() {
        if let Some(e) = g.entity.take() {
            commands.entity(e).despawn();
        }
    }
    rs.groups.clear();
}
