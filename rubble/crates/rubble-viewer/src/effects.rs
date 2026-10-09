//! Optional eye candy, both purely visual (nothing feeds back into the simulation):
//!
//! * **Lighting** (key 7): HDR + bloom, SSAO with TAA, screen-space contact shadows, 4K shadow
//!   cascades, a low golden-hour sun with volumetric light shafts through a fog volume over the
//!   scene, a sun glow in the distance fog, and point lights: a flickering, shadow-casting one
//!   where the beam hits and short flashes on explosions.
//! * **Particles** (key 8): sparks where the beam cuts, debris bits when chunks break or heavy
//!   pieces land hard, and fireballs, sparks and debris on explosions.
//!
//! Particles live in engine space (Z-up) and advance with engine ticks like the glass shards, so
//! they pause with the simulation and are reproducible in screenshot mode.
use crate::coords::{to_bevy, to_engine};
use crate::scene::WorldSpec;
use crate::{Fx, Overlays, Sim, BEAM_RADIUS, BEAM_RANGE, DT};
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
/// alpha steps of the fading materials (a particle swaps material, never mutates one)
const FADE_STEPS: usize = 8;
const GRAVITY: f32 = 9.81;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// glowing streak, bounces once or twice
    Spark,
    /// small tumbling cube in the chunk's colour
    Bit,
    /// growing, fading fireball sphere
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
    /// drag per second (fireballs expand to a stop)
    drag: f32,
    step: usize,
}

#[derive(Component)]
pub struct Flash {
    age: f32,
    life: f32,
    peak: f32,
}

#[derive(Component)]
pub struct BeamLight;
#[derive(Component)]
pub struct SceneFog;

#[derive(Resource)]
pub struct EffectAssets {
    cube: Handle<Mesh>,
    ball: Handle<Mesh>,
    spark: Handle<StandardMaterial>,
    /// per `rubble-format` material id (0..5), opaque
    bits: Vec<Handle<StandardMaterial>>,
    /// fireball material per fade step, step 0 = most opaque
    fire: Vec<Handle<StandardMaterial>>,
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
    spec: Res<WorldSpec>,
) {
    // glow colours are set for the current lighting by `apply_lighting`
    let spark = mats.add(StandardMaterial { base_color: spark_color(false), unlit: true, ..default() });
    let bits = [[0.62, 0.61, 0.58], [0.6, 0.3, 0.22], [0.55, 0.4, 0.25], [0.45, 0.48, 0.52], [0.7, 0.85, 0.95]]
        .into_iter()
        .map(|[r, g, b]| mats.add(StandardMaterial { base_color: Color::srgb(r, g, b), perceptual_roughness: 0.95, ..default() }))
        .collect();
    let fire = (0..FADE_STEPS)
        .map(|s| {
            let a = 1.0 - s as f32 / FADE_STEPS as f32;
            mats.add(StandardMaterial {
                base_color: fire_color(false, a),
                unlit: true,
                alpha_mode: AlphaMode::Add,
                ..default()
            })
        })
        .collect();
    commands.insert_resource(EffectAssets {
        cube: meshes.add(Cuboid::new(1.0, 1.0, 1.0)),
        ball: meshes.add(Sphere::new(1.0).mesh().ico(2).unwrap()),
        spark,
        bits,
        fire,
    });
    commands.insert_resource(EffectState { tick: 0, rng: 0x2545_f491, live: 0 });
    commands.insert_resource(DirectionalLightShadowMap { size: 2048 });
    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.7, 0.35),
            intensity: 0.0,
            range: 25.0,
            radius: 0.1,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default(),
        Visibility::Hidden,
        BeamLight,
    ));
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

/// Fireball colour at opacity `a` (additive), as `spark_color`.
fn fire_color(hdr: bool, a: f32) -> Color {
    let k = if hdr { 9.0 } else { 1.0 };
    Color::LinearRgba(LinearRgba::new(k * a, k * 0.36 * a, k * 0.07 * a, a))
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
    for (s, h) in assets.fire.iter().enumerate() {
        if let Some(mut m) = mats.get_mut(h) {
            m.base_color = fire_color(on, 1.0 - s as f32 / FADE_STEPS as f32);
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
    mut parts: Query<(Entity, &mut Particle, &mut Transform, &mut MeshMaterial3d<StandardMaterial>), Without<BeamLight>>,
    mut flashes: Query<(Entity, &mut Flash, &mut PointLight), Without<BeamLight>>,
    mut beam_light: Query<(&mut PointLight, &mut Transform, &mut Visibility), (With<BeamLight>, Without<Particle>)>,
) {
    let dt = time.delta_secs();
    let events = std::mem::take(&mut sim.fx_events);
    sim.collect_impacts = ov.particles;
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

    // beam: where it is cutting. It bores a clean tunnel of `BEAM_RADIUS` along its line, so a
    // ray down the middle soon finds nothing; a slightly fatter sphere finds the tunnel's rim.
    let beam_hit = sim.beam.and_then(|(o, d)| {
        let d = EVec3::from(d).normalize_or_zero();
        sim.world.phys.cast_sphere(o.into(), d * BEAM_RANGE, BEAM_RADIUS + 0.2).map(|h| {
            let p = EVec3::from(o) + d * (h.toi * BEAM_RANGE + BEAM_RADIUS);
            let n = if h.normal.dot(d) > 0.0 { -h.normal } else { h.normal };
            (Vec3::from(p.to_array()), Vec3::from(n.to_array()), Vec3::from(d.to_array()))
        })
    });

    // ---- lights
    if let Ok((mut l, mut xf, mut v)) = beam_light.single_mut() {
        match beam_hit {
            Some((p, n, _)) if ov.lighting => {
                let t = time.elapsed_secs();
                let flicker = 0.75 + 0.25 * (t * 53.0).sin() * (t * 31.0).cos();
                l.intensity = 600_000.0 * flicker;
                xf.translation = to_bevy((p + n * 0.4).to_array());
                *v = Visibility::Inherited;
            }
            _ => *v = Visibility::Hidden,
        }
    }
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
                    intensity: 4.0e7 * (r / 6.0),
                    range: r * 8.0,
                    radius: r * 0.3,
                    shadow_maps_enabled: false,
                    ..default()
                },
                Transform::from_translation(to_bevy(p.to_array())),
                Flash { age: 0.0, life: 0.6, peak: 4.0e7 * (r / 6.0) },
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
        for _ in 0..8 {
            let d = st.dir(Vec3::ZERO);
            let s = st.range(0.35, 0.6) * r;
            spawn.fire(p + d * r * 0.2, d * r * 0.8, 0.5, s * 0.3, s, 3.0, floor);
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
    if let Some((p, n, d)) = beam_hit {
        // sparks spray off the surface, mostly back towards the shooter
        let floor = floor_below(p);
        let refl = (d - 2.0 * d.dot(n) * n).normalize_or_zero();
        for _ in 0..ticks * 3 {
            let v = (refl * 0.6 + st.dir(n) * 0.8).normalize_or_zero() * st.range(3.0, 9.0);
            let life = st.range(0.25, 0.7);
            spawn.spark(&mut st, p + n * 0.05, v, life, floor);
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
    // advance
    let mut live = 0;
    for (e, mut q, mut xf, mut mat) in &mut parts {
        for _ in 0..ticks {
            step(&mut q, DT);
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
                xf.scale = Vec3::splat(size);
                // fade in over the first 10%, out over the rest
                let a = if k < 0.1 { 1.0 - k / 0.1 } else { (k - 0.1) / 0.9 };
                let s = ((a * FADE_STEPS as f32) as usize).min(FADE_STEPS - 1);
                if s != q.step {
                    q.step = s;
                    mat.0 = assets.fire[s].clone();
                }
            }
        }
    }
    st.live = live + spawned;
}

fn step(q: &mut Particle, dt: f32) {
    q.age += dt;
    match q.kind {
        Kind::Fire => {
            q.vel *= (1.0 - q.drag * dt).max(0.0);
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
        let p = Particle { kind: Kind::Spark, pos, vel, rot: Quat::IDENTITY, spin: Vec3::ZERO, age: 0.0, life, size0, size1: size0, floor, drag: 0.0, step: 0 };
        self.emit(p, &self.assets.cube.clone(), self.assets.spark.clone());
    }

    fn bit(&mut self, st: &mut EffectState, material: u16, pos: Vec3, vel: Vec3, size: f32, floor: f32) {
        let spin = st.dir(Vec3::ZERO) * st.range(3.0, 12.0);
        let rot = Quat::from_scaled_axis(st.dir(Vec3::ZERO) * 3.0);
        let life = st.range(2.5, 4.5);
        let p = Particle { kind: Kind::Bit, pos, vel, rot, spin, age: 0.0, life, size0: size, size1: size, floor, drag: 0.0, step: 0 };
        let m = self.assets.bits[(material as usize).min(self.assets.bits.len() - 1)].clone();
        self.emit(p, &self.assets.cube.clone(), m);
    }

    #[allow(clippy::too_many_arguments)]
    fn fire(&mut self, pos: Vec3, vel: Vec3, life: f32, size0: f32, size1: f32, drag: f32, floor: f32) {
        let p = Particle { kind: Kind::Fire, pos, vel, rot: Quat::IDENTITY, spin: Vec3::ZERO, age: 0.0, life, size0, size1, floor, drag, step: FADE_STEPS - 1 };
        self.emit(p, &self.assets.ball.clone(), self.assets.fire[FADE_STEPS - 1].clone());
    }
}

pub type EffectFilter = Or<(With<Particle>, With<Flash>)>;

/// Remove every particle and flash (reset / building switch).
pub fn clear_effects(commands: &mut Commands, parts: &Query<Entity, EffectFilter>) {
    for e in parts {
        commands.entity(e).despawn();
    }
}
