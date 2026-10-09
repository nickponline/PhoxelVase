//! Cosmetic glass shards. The engine removes a broken pane and emits `ChunkShattered`; here the
//! pane bursts into small spinning triangles that fall, bounce on whatever is below them and fade.
//! Purely visual: nothing feeds back into the simulation.
use crate::coords::{to_bevy, zup_to_yup};
use crate::{Sim, DT};
use bevy::asset::RenderAssetUsages;
use bevy::light::NotShadowCaster;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use rubble_core::physics::PhysicsBackend;
use rubble_core::Vec3 as EVec3;

/// Live shards are capped (oldest first make way) so a building full of windows stays cheap.
const MAX_SHARDS: usize = 6000;
/// Shards per m² of pane, and per-pane bounds.
const SHARDS_PER_M2: f32 = 30.0;
const MIN_PER_PANE: usize = 6;
const MAX_PER_PANE: usize = 60;
const LIFETIME: f32 = 6.0;
const FADE: f32 = 1.5;
const GRAVITY: f32 = 9.81;
const MESH_VARIANTS: usize = 8;
/// resting shards re-check the surface under them every this many ticks (staggered)
const RECHECK_TICKS: u64 = 10;

#[derive(Component)]
pub struct Shard {
    /// engine space (Z-up)
    pos: Vec3,
    vel: Vec3,
    rot: Quat,
    /// rotation axis * rad/s (engine space)
    spin: Vec3,
    /// engine z of the surface below the spawn point (floor, balcony, ground)
    floor: f32,
    age: f32,
    resting: bool,
    /// the surface below was re-sampled at this contact (`landing`: please re-sample)
    checked: bool,
    landing: bool,
    scale: f32,
}

#[derive(Resource)]
pub struct ShardAssets {
    meshes: Vec<Handle<Mesh>>,
    material: Handle<StandardMaterial>,
}

/// Engine tick the shards were last advanced to (they move with the simulation, so they pause
/// with it and step exactly once per tick in screenshot mode).
#[derive(Resource, Default)]
pub struct ShardClock(pub u64);

struct Rng(u32);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 5;
        (self.0 >> 8) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
}

pub fn setup_shards(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut mats: ResMut<Assets<StandardMaterial>>) {
    // a few irregular triangles in the local XY plane (unit size, scaled per shard)
    let mut rng = Rng(0x9e37_79b9);
    let handles = (0..MESH_VARIANTS)
        .map(|_| {
            let mut pts = vec![];
            for k in 0..3 {
                let a = (k as f32 + rng.range(-0.3, 0.3)) * std::f32::consts::TAU / 3.0;
                let r = rng.range(0.45, 1.0);
                pts.push([a.cos() * r, a.sin() * r, 0.0]);
            }
            meshes.add(
                Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::RENDER_WORLD)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pts)
                    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0.0, 1.0]; 3])
                    .with_inserted_indices(Indices::U32(vec![0, 1, 2])),
            )
        })
        .collect();
    let material = mats.add(StandardMaterial {
        base_color: Color::srgba(0.85, 0.95, 1.0, 0.7),
        emissive: LinearRgba::rgb(0.25, 0.32, 0.38),
        perceptual_roughness: 0.08,
        reflectance: 0.9,
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        cull_mode: None,
        ..default()
    });
    commands.insert_resource(ShardAssets { meshes: handles, material });
    commands.insert_resource(ShardClock::default());
}

/// Spawn shards for the panes that broke since the last call, then advance all shards to the
/// engine's current tick.
pub fn update_shards(
    mut commands: Commands,
    mut sim: ResMut<Sim>,
    assets: Res<ShardAssets>,
    mut clock: ResMut<ShardClock>,
    mut q: Query<(Entity, &mut Shard, &mut Transform)>,
) {
    let ground = sim.ground;
    let panes = std::mem::take(&mut sim.shattered);
    // advance existing shards by the ticks the engine took
    let ticks = sim.world.tick.saturating_sub(clock.0).min(30);
    clock.0 = sim.world.tick;
    let mut live = 0usize;
    let mut oldest: Vec<(f32, Entity)> = vec![];
    for (e, mut s, mut xf) in &mut q {
        for _ in 0..ticks {
            step(&mut s, DT);
            // a resting shard whose surface fell away (a slab came down) falls again
            if s.resting && sim.world.tick % RECHECK_TICKS == e.to_bits() % RECHECK_TICKS {
                let from = EVec3::new(s.pos.x, s.pos.y, s.pos.z + 0.05);
                let g = ground.unwrap_or(-1e9);
                let below = sim.world.phys.cast_ray(from, EVec3::NEG_Z, 500.0, None).map_or(g, |h| from.z - h.toi).max(g);
                if below < s.floor - 0.02 {
                    s.floor = below;
                    s.resting = false;
                    s.checked = false;
                    s.vel = Vec3::ZERO;
                }
            }
            if s.landing {
                // reached the surface sampled earlier: it may have drifted past an edge (a sill,
                // a slab), so look again from here
                s.landing = false;
                let from = EVec3::new(s.pos.x, s.pos.y, s.pos.z + 0.05);
                let g = ground.unwrap_or(-1e9);
                let below = sim.world.phys.cast_ray(from, EVec3::NEG_Z, 500.0, None).map_or(g, |h| from.z - h.toi).max(g);
                if below < s.floor - 0.02 {
                    s.floor = below; // keep falling
                } else {
                    contact(&mut s);
                }
            }
        }
        if s.age >= LIFETIME {
            commands.entity(e).despawn();
            continue;
        }
        live += 1;
        oldest.push((s.age, e));
        let fade = ((LIFETIME - s.age) / FADE).clamp(0.0, 1.0);
        *xf = Transform {
            translation: to_bevy(s.pos.to_array()),
            rotation: zup_to_yup() * s.rot,
            scale: Vec3::splat(s.scale * fade.max(0.05)),
        };
    }
    if panes.is_empty() {
        return;
    }
    let mut rng = Rng((0x2545_f491 ^ (sim.world.tick as u32).wrapping_mul(0x9e37_79b9)) | 1);
    let mut spawned = vec![];
    for (b, c) in panes {
        let Some(bd) = sim.world.buildings.get(b as usize) else { continue };
        let ch = &bd.bld.chunks[c as usize];
        let pose = bd.chunk_pose[c as usize];
        // engine (glam via rapier) <-> viewer (bevy glam) through arrays
        let xf_point = |v: Vec3| Vec3::from_array(pose.transform_point(EVec3::from_array(v.to_array())).to_array());
        let xf_vec = |v: Vec3| Vec3::from_array((pose.rotation * EVec3::from_array(v.to_array())).to_array());
        let (lo, hi) = (Vec3::from_array(ch.aabb_min), Vec3::from_array(ch.aabb_max));
        let ext = (hi - lo).max(Vec3::splat(1e-3));
        // pane plane: the two largest extents; normal = the thinnest axis
        let thin = if ext.x <= ext.y && ext.x <= ext.z { 0 } else if ext.y <= ext.z { 1 } else { 2 };
        let area = (ext.x * ext.y * ext.z) / ext[thin];
        let n = ((area * SHARDS_PER_M2) as usize).clamp(MIN_PER_PANE, MAX_PER_PANE);
        let mut axis_n = Vec3::ZERO;
        axis_n[thin] = 1.0;
        let nrm = xf_vec(axis_n);
        // burst mostly outward: away from the building's centre
        let centre = xf_point(sim.centres.get(b as usize).copied().unwrap_or(Vec3::ZERO));
        let pane_c = xf_point((lo + hi) * 0.5);
        let out = if (pane_c - centre).dot(nrm) >= 0.0 { nrm } else { -nrm };
        for _ in 0..n {
            let p = xf_point(lo + ext * Vec3::new(rng.next(), rng.next(), rng.next()));
            let side = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-0.2, 0.6));
            let vel = out * rng.range(0.3, 2.5) + side * 0.8;
            // the surface the shard will land on (the pane is already gone from the world)
            let g = ground.unwrap_or(-1e9);
            let floor = sim
                .world
                .phys
                .cast_ray(EVec3::from_array(p.to_array()), EVec3::NEG_Z, 500.0, None)
                .map_or(g, |h| p.z - h.toi)
                .max(g);
            let axis = Vec3::new(rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(-1.0, 1.0)).try_normalize().unwrap_or(Vec3::X);
            spawned.push(Shard {
                pos: p,
                vel,
                rot: Quat::from_axis_angle(axis, rng.range(0.0, std::f32::consts::TAU)),
                spin: axis * rng.range(4.0, 18.0),
                floor,
                age: rng.range(0.0, 0.6), // staggers the fade-out
                resting: false,
                checked: false,
                landing: false,
                scale: rng.range(0.06, 0.2),
            });
        }
    }
    // make room: drop the oldest shards beyond the cap
    let over = (live + spawned.len()).saturating_sub(MAX_SHARDS);
    if over > 0 {
        oldest.sort_by(|a, b| b.0.total_cmp(&a.0));
        for &(_, e) in oldest.iter().take(over) {
            commands.entity(e).despawn();
        }
    }
    let skip = spawned.len().saturating_sub(MAX_SHARDS);
    for (i, s) in spawned.into_iter().skip(skip).enumerate() {
        let xf = Transform {
            translation: to_bevy(s.pos.to_array()),
            rotation: zup_to_yup() * s.rot,
            scale: Vec3::splat(s.scale),
        };
        let mesh = assets.meshes[i % assets.meshes.len()].clone();
        commands.spawn((Mesh3d(mesh), MeshMaterial3d(assets.material.clone()), xf, NotShadowCaster, s));
    }
}

fn step(s: &mut Shard, dt: f32) {
    s.age += dt;
    if s.resting {
        return;
    }
    s.vel.z -= GRAVITY * dt;
    s.vel *= 1.0 - 0.15 * dt; // a little air drag
    s.pos += s.vel * dt;
    let w = s.spin.length();
    if w > 1e-4 {
        s.rot = (Quat::from_axis_angle(s.spin / w, w * dt) * s.rot).normalize();
    }
    if s.pos.z <= s.floor + 0.005 {
        if !s.checked {
            // let the caller re-sample the surface under the current position first
            s.checked = true;
            s.landing = true;
            return;
        }
        contact(s);
    } else {
        s.checked = false;
    }
}

/// Touching the surface below: bounce (losing most energy) or come to rest.
fn contact(s: &mut Shard) {
    s.pos.z = s.floor + 0.005;
    if s.vel.z < -0.8 {
        s.vel.z = -s.vel.z * 0.3;
        s.vel.x *= 0.5;
        s.vel.y *= 0.5;
        s.spin *= 0.4;
    } else {
        land(s);
    }
}

/// Lie flat where it is.
fn land(s: &mut Shard) {
    s.pos.z = s.floor + 0.005;
    s.resting = true;
    let (_, _, yaw) = s.rot.to_euler(EulerRot::XYZ);
    s.rot = Quat::from_rotation_z(yaw);
}

/// Despawn every shard (world reset / building switch).
pub fn clear_shards(commands: &mut Commands, q: &Query<Entity, With<Shard>>) {
    for e in q {
        commands.entity(e).despawn();
    }
}
