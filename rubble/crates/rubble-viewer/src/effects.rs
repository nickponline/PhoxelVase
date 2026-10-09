//! Optional eye candy, both purely visual (nothing feeds back into the simulation):
//!
//! * **Lighting** (key 7): HDR + bloom, SSAO with TAA, screen-space contact shadows, 4K shadow
//!   cascades, a low golden-hour sun with volumetric light shafts through a fog volume over the
//!   scene, a sun glow in the distance fog, and short point-light flashes on explosions.
//! * **Particles** (key 8): debris bits when chunks break or heavy pieces land hard, and flames
//!   (soft camera-facing sprites), sparks and debris on explosions.
//!
//! Particles live in engine space (Z-up) and advance with engine ticks like the glass shards, so
//! they pause with the simulation and are reproducible in screenshot mode.
use crate::coords::{to_bevy, to_engine};
use crate::scene::WorldSpec;
use crate::{Fx, Overlays, Sim, DT};
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Hdr;
use bevy::light::{DirectionalLightShadowMap, FogVolume, NotShadowCaster, VolumetricFog, VolumetricLight};
use bevy::pbr::{ContactShadows, ScreenSpaceAmbientOcclusion, ScreenSpaceAmbientOcclusionQualityLevel};
use bevy::post_process::bloom::Bloom;
use bevy::prelude::*;
use bevy::render::view::Msaa;
use rubble_core::physics::PhysicsBackend;
use rubble_core::Vec3 as EVec3;

/// Something worth a spray of debris, collected by `Sim::step` (engine space).
#[derive(Clone, Copy, Debug)]
pub enum FxEvent {
    /// a chunk was destroyed or crumbled (not glass: that is the shards' job)
    Break { pos: [f32; 3], material: u16 },
    /// a hard contact (N·s) between a falling piece and something
    Impact { pos: [f32; 3], impulse: f32 },
}

/// Most queued events kept per tick (the rest are dropped; a collapse makes thousands).
pub const MAX_FX_EVENTS: usize = 512;
/// Contacts at least this hard (N·s) throw up debris, at most this many per tick, this far apart (m).
pub const IMPACT_MIN: f32 = 2500.0;
pub const IMPACTS_PER_TICK: usize = 12;
pub const IMPACT_SPACING: f32 = 2.0;

const MAX_PARTICLES: usize = 5000;
const MAX_FLASHES: usize = 8;
/// colour/alpha steps over a flame's life (a particle swaps material, never mutates one)
const FIRE_STEPS: usize = 12;
/// distinct flame textures (random ragged outlines)
const FIRE_VARIANTS: usize = 4;
const FIRE_TEX: u32 = 64;
const GRAVITY: f32 = 9.81;
/// sparks / bits lying on their surface re-check it every this many ticks (staggered)
const FLOOR_RECHECK_TICKS: u64 = 6;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// glowing streak, bounces once or twice
    Spark,
    /// small tumbling cube in the chunk's colour
    Bit,
    /// camera-facing flame sprite: grows, rises, cools from white-yellow to dark red and fades
    Fire,
}

#[derive(Component)]
pub struct Particle {
    kind: Kind,
    /// engine space
    pos: Vec3,
    vel: Vec3,
    rot: Quat,
    spin: Vec3,
    age: f32,
    life: f32,
    size0: f32,
    size1: f32,
    /// engine z of the surface below the spawn point
    floor: f32,
    /// drag per second (flames expand to a stop)
    drag: f32,
    step: usize,
    /// flame sprite: texture variant, roll angle about the view axis and its rate (rad, rad/s)
    variant: usize,
    roll: f32,
    roll_rate: f32,
}

#[derive(Component)]
pub struct Flash {
    age: f32,
    life: f32,
    peak: f32,
}

#[derive(Component)]
pub struct SceneFog;

#[derive(Resource)]
pub struct EffectAssets {
    cube: Handle<Mesh>,
    quad: Handle<Mesh>,
    spark: Handle<StandardMaterial>,
    /// per `rubble-format` material id (0..5), opaque
    bits: Vec<Handle<StandardMaterial>>,
    /// flame material per [texture variant][life step]
    fire: Vec<Vec<Handle<StandardMaterial>>>,
}

/// Tick the particles were last advanced to, and a deterministic RNG.
#[derive(Resource)]
pub struct EffectState {
    tick: u64,
    rng: u32,
    live: usize,
}

impl EffectState {
    fn next(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 17;
        self.rng ^= self.rng << 5;
        (self.rng >> 8) as f32 / (1u32 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next()
    }
    /// uniform direction in the hemisphere around `n` (n = 0: full sphere)
    fn dir(&mut self, n: Vec3) -> Vec3 {
        let z = self.range(-1.0, 1.0);
        let a = self.range(0.0, std::f32::consts::TAU);
        let r = (1.0 - z * z).sqrt();
        let d = Vec3::new(r * a.cos(), r * a.sin(), z);
        if d.dot(n) < 0.0 {
            -d
        } else {
            d
        }
    }
}

pub fn setup_effects(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    mut images: ResMut<Assets<Image>>,
    spec: Res<WorldSpec>,
) {
    // glow colours are set for the current lighting by `apply_lighting`
    let spark = mats.add(StandardMaterial { base_color: spark_color(false), unlit: true, ..default() });
    let bits = [[0.62, 0.61, 0.58], [0.6, 0.3, 0.22], [0.55, 0.4, 0.25], [0.45, 0.48, 0.52], [0.7, 0.85, 0.95]]
        .into_iter()
        .map(|[r, g, b]| mats.add(StandardMaterial { base_color: Color::srgb(r, g, b), perceptual_roughness: 0.95, ..default() }))
        .collect();
    let fire = (0..FIRE_VARIANTS)
        .map(|v| {
            let tex = images.add(flame_texture(0x51ed_270b ^ (v as u32 + 1).wrapping_mul(0x9e37_79b9)));
            (0..FIRE_STEPS)
                .map(|s| {
                    mats.add(StandardMaterial {
                        base_color: fire_color(false, fire_t(s)),
                        base_color_texture: Some(tex.clone()),
                        unlit: true,
                        // glows (adds light) but also partly covers what is behind, so flames
                        // still read against a brightly lit wall
                        alpha_mode: AlphaMode::Premultiplied,
                        double_sided: true,
                        cull_mode: None,
                        ..default()
                    })
                })
                .collect()
        })
        .collect();
    commands.insert_resource(EffectAssets {
        cube: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        quad: meshes.add(Rectangle::new(1.0, 1.0)),
        spark,
        bits,
        fire,
    });
    commands.insert_resource(EffectState { tick: 0, rng: 0x2545_f491, live: 0 });
    commands.insert_resource(DirectionalLightShadowMap { size: 2048 });
    commands.spawn((
        FogVolume { density_factor: 0.004, absorption: 0.1, scattering: 0.3, scattering_asymmetry: 0.7, ..default() },
        fog_transform(&spec),
        Visibility::Hidden,
        SceneFog,
    ));
}

/// The fog volume (a unit cube) stretched over the loaded buildings plus some margin.
fn fog_transform(spec: &WorldSpec) -> Transform {
    let (lo, hi) = spec.bounds();
    let pad = 30.0;
    let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5 + pad * 0.25];
    let s = [hi[0] - lo[0] + 2.0 * pad, hi[1] - lo[1] + 2.0 * pad, hi[2] - lo[2] + pad];
    Transform::from_translation(to_bevy(c)).with_scale(Vec3::new(s[0], s[2], s[1]))
}

/// Sun of the plain and the fancy look: (engine direction the light travels, lux, colour).
fn sun(fancy: bool) -> (Vec3, f32, Color) {
    if fancy {
        (to_bevy([0.66, -0.42, -0.42]).normalize(), 8000.0, Color::srgb(1.0, 0.91, 0.80))
    } else {
        (to_bevy([0.62, -0.38, -0.68]).normalize(), 6000.0, Color::WHITE)
    }
}

/// Spark colour: over-bright with the HDR lighting on (so bloom makes it glow), plain orange
/// without it (over-bright values would clip to white).
fn spark_color(hdr: bool) -> Color {
    if hdr {
        Color::LinearRgba(LinearRgba::rgb(12.0, 5.5, 1.5))
    } else {
        Color::srgb(1.0, 0.72, 0.3)
    }
}

/// Life fraction (0 = just born, 1 = gone) of flame colour step `s`.
fn fire_t(s: usize) -> f32 {
    s as f32 / (FIRE_STEPS - 1) as f32
}

/// Flame colour at life fraction `t` (premultiplied by its alpha): white-yellow, orange, red, dark ember, fading
/// out; over-bright with the HDR lighting on, as `spark_color`.
fn fire_color(hdr: bool, t: f32) -> Color {
    const RAMP: [(f32, [f32; 3]); 4] =
        [(0.0, [1.0, 0.62, 0.2]), (0.2, [1.0, 0.4, 0.06]), (0.55, [0.75, 0.15, 0.02]), (1.0, [0.15, 0.02, 0.0])];
    let i = RAMP.iter().rposition(|(t0, _)| *t0 <= t).unwrap_or(0).min(RAMP.len() - 2);
    let ((t0, a), (t1, b)) = (RAMP[i], RAMP[i + 1]);
    let f = ((t - t0) / (t1 - t0)).clamp(0.0, 1.0);
    let c: [f32; 3] = std::array::from_fn(|j| a[j] + (b[j] - a[j]) * f);
    // hot flames glow far more than cooling ones
    let k = if hdr { 1.0 + 1.5 * (1.0 - t).powi(2) } else { 1.0 };
    let alpha = (1.0 - t).powf(0.8);
    // mostly glow: covers only this fraction of what is behind
    const COVER: f32 = 0.4;
    Color::LinearRgba(LinearRgba::new(k * c[0] * alpha, k * c[1] * alpha, k * c[2] * alpha, alpha * COVER))
}

/// A soft flame puff: bright mottled centre fading to a ragged transparent edge (grey levels
/// premultiplied by alpha, tinted by the material). `seed` picks the outline.
fn flame_texture(seed: u32) -> Image {
    use bevy::asset::RenderAssetUsages;
    use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};
    let mut rng = EffectState { tick: 0, rng: seed | 1, live: 0 };
    // edge radius and interior brightness as low-order Fourier series in the angle
    let waves: Vec<(f32, f32, f32)> = (2..7).map(|k| (k as f32, rng.range(0.0, std::f32::consts::TAU), rng.range(0.02, 0.07))).collect();
    let mottle: Vec<(f32, f32, f32)> = (0..6)
        .map(|_| (rng.range(-1.0, 1.0), rng.range(-1.0, 1.0), rng.range(0.0, std::f32::consts::TAU)))
        .collect();
    let n = FIRE_TEX as usize;
    let mut data = vec![0u8; n * n * 4];
    for y in 0..n {
        for x in 0..n {
            let (u, v) = ((x as f32 + 0.5) / n as f32 * 2.0 - 1.0, (y as f32 + 0.5) / n as f32 * 2.0 - 1.0);
            let r = (u * u + v * v).sqrt();
            let a = v.atan2(u);
            let edge = 0.78 + waves.iter().map(|(k, ph, amp)| amp * (k * a + ph).sin()).sum::<f32>();
            let fall = (1.0 - r / edge).clamp(0.0, 1.0);
            let m = mottle.iter().map(|(fx, fy, ph)| (7.0 * (fx * u + fy * v) + ph).sin()).sum::<f32>() / mottle.len() as f32;
            let i = (fall * fall * (3.0 - 2.0 * fall)) * (0.8 + 0.2 * m);
            let b = (i.clamp(0.0, 1.0) * 255.0) as u8;
            data[(y * n + x) * 4..][..4].copy_from_slice(&[b, b, b, b]);
        }
    }
    Image::new(
        Extent3d { width: FIRE_TEX, height: FIRE_TEX, depth_or_array_layers: 1 },
        TextureDimension::D2,
        data,
        TextureFormat::Rgba8Unorm,
        RenderAssetUsages::RENDER_WORLD,
    )
}

/// Switch the camera / sun / ambient between the plain and the fancy look when the toggle
/// changes, and keep the fog volume over the current scene.
pub fn apply_lighting(
    mut commands: Commands,
    ov: Res<Overlays>,
    spec: Res<WorldSpec>,
    mut ambient: ResMut<GlobalAmbientLight>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    assets: Res<EffectAssets>,
    mut shadow_map: ResMut<DirectionalLightShadowMap>,
    cams: Query<Entity, With<Camera3d>>,
    mut fogs: Query<&mut DistanceFog>,
    mut suns: Query<(Entity, &mut DirectionalLight, &mut Transform)>,
    mut volume: Query<(&mut Transform, &mut Visibility), (With<SceneFog>, Without<DirectionalLight>)>,
    mut applied: Local<Option<bool>>,
) {
    if spec.is_changed() {
        if let Ok((mut xf, _)) = volume.single_mut() {
            *xf = fog_transform(&spec);
        }
    }
    let on = ov.lighting;
    if *applied == Some(on) {
        return;
    }
    *applied = Some(on);
    for cam in &cams {
        let mut e = commands.entity(cam);
        if on {
            e.insert((
                Hdr,
                Bloom { intensity: 0.18, ..Bloom::NATURAL },
                Msaa::Off,
                TemporalAntiAliasing::default(),
                ScreenSpaceAmbientOcclusion { quality_level: ScreenSpaceAmbientOcclusionQualityLevel::Medium, ..default() },
                ContactShadows { length: 0.5, ..default() },
                VolumetricFog { ambient_intensity: 0.0, step_count: 32, jitter: 0.5, ..default() },
            ));
        } else {
            e.remove::<(Bloom, TemporalAntiAliasing, ScreenSpaceAmbientOcclusion, ContactShadows, VolumetricFog)>();
            e.insert(Msaa::default());
            e.remove::<Hdr>();
        }
    }
    for mut f in &mut fogs {
        f.directional_light_color = if on { Color::srgba(1.0, 0.85, 0.65, 0.35) } else { Color::NONE };
        f.directional_light_exponent = 40.0;
    }
    let (dir, lux, color) = sun(on);
    for (e, mut l, mut xf) in &mut suns {
        l.illuminance = lux;
        l.color = color;
        l.contact_shadows_enabled = on;
        *xf = Transform::default().looking_to(dir, Vec3::Y);
        if on {
            commands.entity(e).insert(VolumetricLight);
        } else {
            commands.entity(e).remove::<VolumetricLight>();
        }
    }
    ambient.brightness = if on { 180.0 } else { 250.0 };
    ambient.color = if on { Color::srgb(0.7, 0.8, 1.0) } else { Color::srgb(0.85, 0.9, 1.0) };
    shadow_map.size = if on { 4096 } else { 2048 };
    if let Some(mut m) = mats.get_mut(&assets.spark) {
        m.base_color = spark_color(on);
    }
    for steps in &assets.fire {
        for (s, h) in steps.iter().enumerate() {
            if let Some(mut m) = mats.get_mut(h) {
                m.base_color = fire_color(on, fire_t(s));
            }
        }
    }
    if let Ok((_, mut v)) = volume.single_mut() {
        *v = if on { Visibility::Inherited } else { Visibility::Hidden };
    }
}

/// Spawn particles / lights for this frame's happenings and advance everything.
pub fn update_effects(
    mut commands: Commands,
    time: Res<Time<Real>>,
    ov: Res<Overlays>,
    mut sim: ResMut<Sim>,
    fx: Res<Fx>,
    assets: Res<EffectAssets>,
    mut st: ResMut<EffectState>,
    mut parts: Query<(Entity, &mut Particle, &mut Transform, &mut MeshMaterial3d<StandardMaterial>)>,
    mut flashes: Query<(Entity, &mut Flash, &mut PointLight)>,
    cams: Query<&Transform, (With<Camera3d>, Without<Particle>)>,
) {
    let dt = time.delta_secs();
    let events = std::mem::take(&mut sim.fx_events);
    let ticks = sim.world.tick.saturating_sub(st.tick).min(30) as u32;
    st.tick = sim.world.tick;
    let ground = sim.ground.unwrap_or(-1e9);
    let floor_below = |p: Vec3| -> f32 {
        let from = EVec3::new(p.x, p.y, p.z + 0.1);
        sim.world.phys.cast_ray(from, EVec3::NEG_Z, 200.0, None).map_or(ground, |h| from.z - h.toi).max(ground)
    };

    // explosions crossing their fire time this frame (`Fx::blasts` ages are advanced later
    // this frame by the overlay; delayed demolition charges start negative)
    let mut blasts = vec![];
    for &(p, r, age) in &fx.blasts {
        if age <= 0.0 && age + dt > 0.0 {
            blasts.push((Vec3::from(to_engine(p)), r));
        }
    }

    // ---- lights
    let mut n_flash = 0;
    for (e, mut f, mut l) in &mut flashes {
        f.age += dt;
        if f.age >= f.life || !ov.lighting {
            commands.entity(e).despawn();
            continue;
        }
        let k = 1.0 - f.age / f.life;
        l.intensity = f.peak * k * k;
        n_flash += 1;
    }
    if ov.lighting {
        for &(p, r) in &blasts {
            if n_flash >= MAX_FLASHES {
                break;
            }
            n_flash += 1;
            commands.spawn((
                PointLight {
                    color: Color::srgb(1.0, 0.6, 0.25),
                    intensity: 0.8e7 * (r / 6.0),
                    range: r * 8.0,
                    radius: r * 0.3,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_translation(to_bevy(p.to_array())),
                Flash { age: 0.0, life: 0.6, peak: 0.8e7 * (r / 6.0) },
            ));
        }
    }

    // ---- particles
    if !ov.particles {
        for (e, ..) in &parts {
            commands.entity(e).despawn();
        }
        st.live = 0;
        return;
    }
    let budget = MAX_PARTICLES.saturating_sub(st.live);
    let mut spawn = Spawner { commands: &mut commands, assets: &assets, budget };
    for &(p, r) in &blasts {
        let floor = floor_below(p);
        // a short white-hot core, then ragged flames bursting outward, rising and cooling
        for _ in 0..3 {
            let d = st.dir(Vec3::ZERO);
            let s = st.range(0.9, 1.2) * r;
            let life = st.range(0.12, 0.2);
            spawn.fire(&mut st, p + d * r * 0.1, d * r * 0.5, life, s * 0.6, s * 1.2, 4.0, floor);
        }
        for _ in 0..44 {
            let d = st.dir(Vec3::ZERO);
            let o = d * st.range(0.0, 0.35) * r;
            let v = d * st.range(0.6, 1.6) * r + Vec3::Z * st.range(0.0, 0.4) * r;
            let s = st.range(0.32, 0.55) * r;
            let life = st.range(0.35, 0.9);
            let grow = st.range(1.4, 2.2);
            spawn.fire(&mut st, p + o, v, life, s * 0.5, s * grow, 3.5, floor);
        }
        for _ in 0..60 {
            let d = st.dir(Vec3::ZERO);
            let v = d * st.range(8.0, 25.0) + Vec3::Z * 4.0;
            let life = st.range(0.6, 1.4);
            spawn.spark(&mut st, p, v, life, floor);
        }
        for _ in 0..20 {
            let v = st.dir(Vec3::Z) * st.range(5.0, 14.0);
            let size = st.range(0.08, 0.2);
            spawn.bit(&mut st, 0, p, v, size, floor);
        }
    }
    for ev in events {
        match ev {
            FxEvent::Break { pos, material } => {
                let p = Vec3::from(pos);
                let floor = floor_below(p);
                for _ in 0..4 {
                    let v = st.dir(Vec3::Z) * st.range(1.0, 4.0);
                    let size = st.range(0.05, 0.14);
                    spawn.bit(&mut st, material, p, v, size, floor);
                }
            }
            FxEvent::Impact { pos, impulse } => {
                let p = Vec3::from(pos);
                let k = (impulse / IMPACT_MIN).sqrt().clamp(1.0, 4.0);
                for _ in 0..(2.0 * k) as usize {
                    let v = st.dir(Vec3::Z) * st.range(2.0, 5.0);
                    let size = st.range(0.05, 0.12);
                    spawn.bit(&mut st, 0, p, v, size, p.z - 0.05);
                }
            }
        }
    }

    let spawned = budget - spawn.budget;
    // flames face the camera
    let cam_rot = cams.single().map_or(Quat::IDENTITY, |t| t.rotation);
    // advance
    let mut live = 0;
    for (e, mut q, mut xf, mut mat) in &mut parts {
        for _ in 0..ticks {
            step(&mut q, DT);
        }
        // the surface under a particle is sampled once at spawn; one lying on it looks again now
        // and then, since the surface may have been destroyed or the particle slid off its edge.
        // It then falls to whatever is below.
        if ticks > 0 && (st.tick + e.to_bits()).is_multiple_of(FLOOR_RECHECK_TICKS) {
            refloor(&mut q, &floor_below);
        }
        if q.age >= q.life {
            commands.entity(e).despawn();
            continue;
        }
        live += 1;
        let k = q.age / q.life;
        let size = q.size0 + (q.size1 - q.size0) * k.sqrt();
        match q.kind {
            Kind::Spark => {
                let v = to_bevy(q.vel.to_array());
                let len = (v.length() * 0.03).clamp(0.05, 0.5);
                let w = q.size0 * (1.0 - k);
                xf.translation = to_bevy(q.pos.to_array());
                xf.rotation = Quat::from_rotation_arc(Vec3::Z, v.normalize_or(Vec3::Z));
                xf.scale = Vec3::new(w, w, len);
            }
            Kind::Bit => {
                let w = if k > 0.8 { q.size0 * (1.0 - k) * 5.0 } else { q.size0 };
                xf.translation = to_bevy(q.pos.to_array());
                xf.rotation = crate::coords::zup_to_yup() * q.rot;
                xf.scale = Vec3::splat(w);
            }
            Kind::Fire => {
                xf.translation = to_bevy(q.pos.to_array());
                xf.rotation = cam_rot * Quat::from_rotation_z(q.roll);
                xf.scale = Vec3::splat(size);
                let s = ((k * FIRE_STEPS as f32) as usize).min(FIRE_STEPS - 1);
                if s != q.step {
                    q.step = s;
                    mat.0 = assets.fire[q.variant][s].clone();
                }
            }
        }
    }
    st.live = live + spawned;
}

/// A spark / bit lying on its surface takes the surface below it again (`floor_below`, engine
/// space); when that is lower (the old one was destroyed or it slid off), it falls from there.
fn refloor(q: &mut Particle, floor_below: &dyn Fn(Vec3) -> f32) {
    if q.kind != Kind::Fire && q.pos.z <= q.floor + 0.01 {
        let below = floor_below(q.pos);
        if below < q.floor - 0.02 {
            q.floor = below;
        }
    }
}

fn step(q: &mut Particle, dt: f32) {
    q.age += dt;
    match q.kind {
        Kind::Fire => {
            q.vel *= (1.0 - q.drag * dt).max(0.0);
            // hot gas rises
            q.vel.z += 3.0 * dt;
            q.roll += q.roll_rate * dt;
            q.pos += q.vel * dt;
            q.pos.z = q.pos.z.max(q.floor + q.size1 * 0.3);
        }
        Kind::Spark | Kind::Bit => {
            q.vel.z -= GRAVITY * dt;
            q.pos += q.vel * dt;
            q.rot = (Quat::from_scaled_axis(q.spin * dt) * q.rot).normalize();
            if q.pos.z < q.floor {
                q.pos.z = q.floor;
                let (bounce, fric) = if q.kind == Kind::Spark { (0.35, 0.6) } else { (0.25, 0.4) };
                q.vel.z = -q.vel.z * bounce;
                q.vel.x *= fric;
                q.vel.y *= fric;
                q.spin *= 0.5;
            }
        }
    }
}

struct Spawner<'a, 'w, 's> {
    commands: &'a mut Commands<'w, 's>,
    assets: &'a EffectAssets,
    budget: usize,
}

impl Spawner<'_, '_, '_> {
    fn emit(&mut self, p: Particle, mesh: &Handle<Mesh>, mat: Handle<StandardMaterial>) {
        if self.budget == 0 {
            return;
        }
        self.budget -= 1;
        let xf = Transform::from_translation(to_bevy(p.pos.to_array())).with_scale(Vec3::splat(p.size0.max(1e-3)));
        self.commands.spawn((Mesh3d(mesh.clone()), MeshMaterial3d(mat), xf, NotShadowCaster, p));
    }

    fn spark(&mut self, st: &mut EffectState, pos: Vec3, vel: Vec3, life: f32, floor: f32) {
        let size0 = st.range(0.015, 0.03);
        let p = Particle { kind: Kind::Spark, pos, vel, rot: Quat::IDENTITY, spin: Vec3::ZERO, age: 0.0, life, size0, size1: size0, floor, drag: 0.0, step: 0, variant: 0, roll: 0.0, roll_rate: 0.0 };
        self.emit(p, &self.assets.cube.clone(), self.assets.spark.clone());
    }

    fn bit(&mut self, st: &mut EffectState, material: u16, pos: Vec3, vel: Vec3, size: f32, floor: f32) {
        let spin = st.dir(Vec3::ZERO) * st.range(3.0, 12.0);
        let rot = Quat::from_scaled_axis(st.dir(Vec3::ZERO) * 3.0);
        let life = st.range(2.5, 4.5);
        let p = Particle { kind: Kind::Bit, pos, vel, rot, spin, age: 0.0, life, size0: size, size1: size, floor, drag: 0.0, step: 0, variant: 0, roll: 0.0, roll_rate: 0.0 };
        let m = self.assets.bits[(material as usize).min(self.assets.bits.len() - 1)].clone();
        self.emit(p, &self.assets.cube.clone(), m);
    }

    #[allow(clippy::too_many_arguments)]
    fn fire(&mut self, st: &mut EffectState, pos: Vec3, vel: Vec3, life: f32, size0: f32, size1: f32, drag: f32, floor: f32) {
        let variant = (st.next() * FIRE_VARIANTS as f32) as usize % FIRE_VARIANTS;
        let roll = st.range(0.0, std::f32::consts::TAU);
        let roll_rate = st.range(-1.5, 1.5);
        let p = Particle {
            kind: Kind::Fire,
            pos,
            vel,
            rot: Quat::IDENTITY,
            spin: Vec3::ZERO,
            age: 0.0,
            life,
            size0,
            size1,
            floor,
            drag,
            step: 0,
            variant,
            roll,
            roll_rate,
        };
        self.emit(p, &self.assets.quad.clone(), self.assets.fire[variant][0].clone());
    }
}

pub type EffectFilter = Or<(With<Particle>, With<Flash>)>;

/// Remove every particle and flash (reset / building switch).
pub fn clear_effects(commands: &mut Commands, parts: &Query<Entity, EffectFilter>) {
    for e in parts {
        commands.entity(e).despawn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A bit lying on a slab that is then destroyed falls to the ground instead of hanging in
    /// the air where the slab was.
    #[test]
    fn resting_bit_falls_when_its_surface_goes() {
        let mut q = Particle {
            kind: Kind::Bit,
            pos: Vec3::new(1.0, 2.0, 5.5),
            vel: Vec3::ZERO,
            rot: Quat::IDENTITY,
            spin: Vec3::ZERO,
            age: 0.0,
            life: 10.0,
            size0: 0.1,
            size1: 0.1,
            floor: 5.0,
            drag: 0.0,
            step: 0,
            variant: 0,
            roll: 0.0,
            roll_rate: 0.0,
        };
        let mut slab = true;
        let mut run = |q: &mut Particle, slab: bool, secs: f32| {
            for _ in 0..(secs / DT) as usize {
                step(q, DT);
                refloor(q, &|_| if slab { 5.0 } else { 0.0 });
            }
        };
        run(&mut q, slab, 1.0);
        assert!((q.pos.z - 5.0).abs() < 1e-3, "settles on the slab: {}", q.pos.z);
        slab = false;
        run(&mut q, slab, 2.0);
        assert!(q.pos.z.abs() < 1e-3, "falls to the ground once the slab is gone: {}", q.pos.z);
    }
}
