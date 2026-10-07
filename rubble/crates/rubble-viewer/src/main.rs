//! `rubble-viewer`: interactive Bevy viewer for `rubble-core` (DESIGN §3.2).
//!
//!   cargo run -p rubble-viewer --release -- [FILE.bld ...] [--scenario S.yaml] [--arena N]
//!   cargo run -p rubble-viewer --release -- fixtures/two_box.bld --screenshot out.png --frames 120 \
//!       --explode 1,0.5,1,2 --explode-frame 10
//!
//! The engine world (Z-up) lives in the [`Sim`] resource and is stepped at 60 Hz from
//! `FixedUpdate` (or exactly once per frame in screenshot mode, for reproducible captures).
//! Axis conversion happens only in [`coords`].
#![allow(clippy::too_many_arguments, clippy::type_complexity)] // idiomatic for Bevy systems

mod camera;
mod coords;
mod overlay;
mod render;
mod scene;

use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured};
use bevy::window::{PresentMode, PrimaryWindow, WindowResolution};
use camera::{camera_control, CamCtl};
use coords::{to_bevy, to_engine};
use render::{Materials, RenderState};
use rubble_core::physics::PhysicsBackend;
use rubble_core::{BuildingId, Explosion, Projectile, StepTimings, Weapon, World as EngineWorld};
use scene::{Args, Catalog, Scheduled, WorldSpec};

pub const DT: f32 = 1.0 / 60.0;
/// damage of E / launcher explosions (radius is adjustable with [ ])
pub const EXPLOSION_DAMAGE: f32 = 1500.0;

#[derive(Resource)]
pub struct Sim {
    pub world: EngineWorld,
    pub paused: bool,
    pub step_once: bool,
    pub next_action: usize,
    /// (wall time, events) of recent ticks for events/sec
    pub event_log: std::collections::VecDeque<(f64, usize)>,
    pub tick_ms_avg: f32,
    pub timings_avg: StepTimings,
}

impl Sim {
    fn new(spec: &WorldSpec) -> Self {
        Sim {
            world: spec.build_world(),
            paused: false,
            step_once: false,
            next_action: 0,
            event_log: Default::default(),
            tick_ms_avg: 0.0,
            timings_avg: StepTimings::default(),
        }
    }

    fn step(&mut self, spec: &WorldSpec, now: f64) {
        while let Some((t, a)) = spec.actions.get(self.next_action) {
            if *t > self.world.time + 1e-6 {
                break;
            }
            match a {
                Scheduled::Explode(e) => self.world.explode(*e),
                Scheduled::Fire(p) => self.world.fire(*p),
                Scheduled::Damage(b, cs, amt) => {
                    for &c in cs {
                        self.world.damage_chunk(BuildingId(*b), c, *amt);
                    }
                }
            }
            self.next_action += 1;
        }
        self.world.step(DT);
        let n = self.world.drain_events().len();
        self.event_log.push_back((now, n));
        while self.event_log.front().is_some_and(|(t, _)| now - t > 1.0) {
            self.event_log.pop_front();
        }
        let t = self.world.timings;
        let a = 0.1;
        let ema = |o: f32, n: f32| o + (n - o) * a;
        self.tick_ms_avg = ema(self.tick_ms_avg, t.total_ms);
        let s = &mut self.timings_avg;
        s.projectiles_ms = ema(s.projectiles_ms, t.projectiles_ms);
        s.damage_ms = ema(s.damage_ms, t.damage_ms);
        s.connectivity_ms = ema(s.connectivity_ms, t.connectivity_ms);
        s.stress_ms = ema(s.stress_ms, t.stress_ms);
        s.promotion_ms = ema(s.promotion_ms, t.promotion_ms);
        s.physics_ms = ema(s.physics_ms, t.physics_ms);
        s.impacts_ms = ema(s.impacts_ms, t.impacts_ms);
        s.settle_ms = ema(s.settle_ms, t.settle_ms);
        s.total_ms = self.tick_ms_avg;
    }
}

#[derive(Resource, Default)]
pub struct Overlays {
    pub graph: bool,
    pub anchors: bool,
    pub random_colors: bool,
    pub sleep_tint: bool,
    pub stats: bool,
    pub help: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WeaponSel {
    Ar,
    Sniper,
    Launcher,
}

#[derive(Resource)]
pub struct Controls {
    pub weapon: WeaponSel,
    pub radius: f32,
    pub fire_cooldown: f32,
}

/// Engine-space hit under the cursor this frame (Bevy-space point for the camera).
#[derive(Resource, Default)]
pub struct CursorHit(pub Option<Vec3>);

/// Short-lived visual feedback (tracers, explosion spheres), Bevy space.
#[derive(Resource, Default)]
pub struct Fx {
    pub tracers: Vec<(Vec3, Vec3, f32, Color)>,
    pub blasts: Vec<(Vec3, f32, f32)>,
}

#[derive(Resource)]
struct Shot {
    path: std::path::PathBuf,
    frames: u32,
    explode: Option<[f32; 5]>,
    explode_frame: u32,
    frame: u32,
    started: std::time::Instant,
    sim_ms: f64,
    requested: bool,
    target: Option<Handle<Image>>,
}

fn main() {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };
    let spec = match WorldSpec::from_args(&args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };
    let t0 = std::time::Instant::now();
    let sim = Sim::new(&spec);
    println!(
        "rubble-viewer: {} | {} buildings, {} chunks, loaded in {:.0} ms",
        spec.title,
        spec.buildings.len(),
        spec.total_chunks(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    let mut catalog = Catalog::discover();
    if let [one] = args.blds.as_slice() {
        if args.scenario.is_none() {
            catalog.select_path(one);
        }
    }
    let mut overlays = Overlays { help: true, ..default() };
    for o in &args.overlays {
        match o.as_str() {
            "f1" | "graph" => overlays.graph = true,
            "f2" | "anchors" => overlays.anchors = true,
            "f3" | "colors" => overlays.random_colors = true,
            "f4" | "sleep" => overlays.sleep_tint = true,
            "f5" | "stats" => overlays.stats = true,
            "nohelp" => overlays.help = false,
            _ => eprintln!("unknown overlay {o}"),
        }
    }
    let shot_mode = args.screenshot.is_some();
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: format!("rubble-viewer — {}", spec.title),
            resolution: WindowResolution::new(args.width as u32, args.height as u32),
            present_mode: if shot_mode { PresentMode::AutoNoVsync } else { PresentMode::AutoVsync },
            ..default()
        }),
        ..default()
    }))
    .insert_resource(ClearColor(Color::srgb(0.62, 0.74, 0.86)))
    .insert_resource(Time::<Fixed>::from_hz(60.0))
    .insert_resource(GlobalAmbientLight { color: Color::srgb(0.85, 0.9, 1.0), brightness: 250.0, ..default() })
    .insert_resource(overlays)
    .insert_resource(Controls { weapon: WeaponSel::Ar, radius: 3.0, fire_cooldown: 0.0 })
    .insert_resource(CursorHit::default())
    .insert_resource(Fx::default())
    .insert_resource(RenderState::default())
    .insert_resource(sim)
    .insert_resource(spec)
    .insert_resource(catalog)
    .insert_resource(ArgsRes(args.clone()))
    .add_systems(Startup, setup)
    .add_systems(
        Update,
        (
            cycle_buildings,
            cursor_pick,
            camera_control,
            controls,
            render::sync_render,
            overlay::draw_overlays,
            overlay::update_text,
        )
            .chain(),
    );
    if let Some(path) = &args.screenshot {
        app.insert_resource(Shot {
            path: path.clone(),
            frames: args.frames,
            explode: args.explode,
            explode_frame: args.explode_frame,
            frame: 0,
            started: std::time::Instant::now(),
            sim_ms: 0.0,
            requested: false,
            target: None,
        })
        .add_systems(Update, screenshot_driver.before(render::sync_render).after(controls));
    } else {
        app.add_systems(FixedUpdate, fixed_step);
    }
    app.run();
}

#[derive(Resource)]
struct ArgsRes(Args);

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut mats: ResMut<Assets<StandardMaterial>>,
    spec: Res<WorldSpec>,
    sim: Res<Sim>,
    mut rs: ResMut<RenderState>,
    args: Res<ArgsRes>,
    mut images: ResMut<Assets<Image>>,
    shot: Option<ResMut<Shot>>,
    mut gizmo_store: ResMut<GizmoConfigStore>,
) {
    // debug overlays draw on top of geometry (graph lines live inside the chunks)
    let (gc, _) = gizmo_store.config_mut::<DefaultGizmoConfigGroup>();
    gc.depth_bias = -1.0;
    gc.line.width = 1.5;
    let m = Materials::new(&mut mats);
    commands.insert_resource(m);
    rs.reset(&sim);

    // camera framing (engine Z-up -> Bevy at the boundary)
    let (c, ext) = framing(&spec);
    let (ctl, xf) = camera_pose(&spec, args.0.cam, args.0.look);
    let mut cam = commands.spawn((
        Camera3d::default(),
        xf,
        ctl,
        Projection::Perspective(PerspectiveProjection { fov: 60f32.to_radians(), far: 5000.0, ..default() }),
        DistanceFog {
            color: Color::srgb(0.62, 0.74, 0.86),
            falloff: FogFalloff::Linear { start: ext * 2.0 + 100.0, end: ext * 6.0 + 600.0 },
            ..default()
        },
    ));
    // screenshot mode renders offscreen (works with hidden / occluded windows)
    let cam_id = cam.id();
    let mut ui_cam = None;
    if let Some(mut shot) = shot {
        ui_cam = Some(cam_id);
        let img = Image::new_target_texture(
            args.0.width as u32,
            args.0.height as u32,
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            None,
        );
        let h = images.add(img);
        cam.insert(bevy::camera::RenderTarget::Image(h.clone().into()));
        shot.target = Some(h);
    }

    // sun: engine direction (pointing from the sun) -> Bevy
    let sun_dir = to_bevy([0.62, -0.38, -0.68]).normalize();
    commands.spawn((
        DirectionalLight { illuminance: 6000.0, shadow_maps_enabled: true, ..default() },
        Transform::default().looking_to(sun_dir, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 4,
            first_cascade_far_bound: (ext * 0.15).max(10.0),
            maximum_distance: (ext * 3.0).max(80.0),
            ..default()
        }
        .build(),
    ));

    // ground plane (engine z = ground) — Plane3d is authored Y-up, i.e. already Bevy space
    if let Some(z) = spec.ground {
        let size = (ext * 20.0).max(2000.0);
        commands.spawn((
            Mesh3d(meshes.add(Plane3d::default().mesh().size(size, size))),
            MeshMaterial3d(mats.add(StandardMaterial {
                base_color: Color::srgb(0.42, 0.45, 0.40),
                perceptual_roughness: 1.0,
                reflectance: 0.1,
                ..default()
            })),
            Transform::from_translation(to_bevy([c[0], c[1], z - 0.002])),
        ));
    }
    overlay::spawn_text(&mut commands, ui_cam);
}

/// Centre and diagonal of everything loaded (engine space), for camera, fog and shadows.
fn framing(spec: &WorldSpec) -> ([f32; 3], f32) {
    let (lo, hi) = spec.bounds();
    let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5, (lo[2] + hi[2]) * 0.5];
    let ext = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2) + (hi[2] - lo[2]).powi(2)).sqrt().max(4.0);
    (c, ext)
}

/// Default three-quarter view of the loaded buildings (overridable from the CLI).
fn camera_pose(spec: &WorldSpec, cam: Option<[f32; 3]>, look: Option<[f32; 3]>) -> (CamCtl, Transform) {
    let (c, ext) = framing(spec);
    let eye = cam.unwrap_or([c[0] - 0.55 * ext, c[1] - 0.85 * ext, c[2] + 0.45 * ext + 2.0]);
    let target = look.unwrap_or([c[0], c[1], c[2] * 0.6]);
    CamCtl::looking_at(to_bevy(eye), to_bevy(target))
}

/// + / - : load the next / previous building from `assets/buildings` (wraps around), resetting
/// the world and re-framing the camera.
#[allow(clippy::too_many_arguments)]
fn cycle_buildings(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mut catalog: ResMut<Catalog>,
    mut spec: ResMut<WorldSpec>,
    mut sim: ResMut<Sim>,
    mut rs: ResMut<RenderState>,
    mut fx: ResMut<Fx>,
    mut cams: Query<(&mut Transform, &mut CamCtl, Option<&mut DistanceFog>)>,
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
) {
    let delta = if keys.any_just_pressed([KeyCode::Equal, KeyCode::NumpadAdd]) {
        1
    } else if keys.any_just_pressed([KeyCode::Minus, KeyCode::NumpadSubtract]) {
        -1
    } else {
        return;
    };
    let Some(path) = catalog.step(delta).map(|p| p.to_path_buf()) else {
        eprintln!("rubble-viewer: no buildings found in assets/buildings");
        return;
    };
    let t0 = std::time::Instant::now();
    let new_spec = match WorldSpec::single(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("rubble-viewer: {e}");
            return;
        }
    };
    *sim = Sim::new(&new_spec);
    render::clear_entities(&mut commands, &mut rs);
    rs.reset(&sim);
    fx.tracers.clear();
    fx.blasts.clear();
    println!(
        "rubble-viewer: [{}] {} | {} chunks, loaded in {:.0} ms",
        catalog.label(),
        new_spec.title,
        new_spec.total_chunks(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    let (_, ext) = framing(&new_spec);
    if let Ok((mut xf, mut ctl, fog)) = cams.single_mut() {
        let (c, x) = camera_pose(&new_spec, None, None);
        *ctl = c;
        *xf = x;
        if let Some(mut fog) = fog {
            fog.falloff = FogFalloff::Linear { start: ext * 2.0 + 100.0, end: ext * 6.0 + 600.0 };
        }
    }
    if let Ok(mut w) = windows.single_mut() {
        w.title = format!("rubble-viewer — [{}] {}", catalog.label(), new_spec.title);
    }
    *spec = new_spec;
}

fn fixed_step(mut sim: ResMut<Sim>, spec: Res<WorldSpec>, time: Res<Time<Real>>) {
    if sim.paused && !sim.step_once {
        return;
    }
    sim.step_once = false;
    let now = time.elapsed_secs_f64();
    sim.step(&spec, now);
}

/// Raycast the engine world under the cursor (or screen center while mouse-looking).
fn cursor_pick(
    windows: Query<&Window, With<PrimaryWindow>>,
    cams: Query<(&Camera, &GlobalTransform)>,
    sim: Res<Sim>,
    mut hit: ResMut<CursorHit>,
) {
    hit.0 = None;
    let Some(ray) = view_ray(&windows, &cams) else { return };
    let o = to_engine(ray.0);
    let d = to_engine(ray.1);
    if let Some(h) = sim.world.phys.cast_ray(o.into(), d.into(), 2000.0, None) {
        hit.0 = Some(ray.0 + ray.1 * h.toi);
    }
}

/// Bevy-space (origin, dir) of the pick ray.
fn view_ray(windows: &Query<&Window, With<PrimaryWindow>>, cams: &Query<(&Camera, &GlobalTransform)>) -> Option<(Vec3, Vec3)> {
    let win = windows.single().ok()?;
    let (cam, gt) = cams.single().ok()?;
    let center = Vec2::new(win.width() * 0.5, win.height() * 0.5);
    let p = win.cursor_position().unwrap_or(center);
    let r = cam.viewport_to_world(gt, p).ok()?;
    Some((r.origin, *r.direction))
}

#[allow(clippy::too_many_arguments)]
fn controls(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    time: Res<Time<Real>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cams: Query<(&Camera, &GlobalTransform)>,
    mut sim: ResMut<Sim>,
    spec: Res<WorldSpec>,
    mut ctl: ResMut<Controls>,
    mut ov: ResMut<Overlays>,
    hit: Res<CursorHit>,
    mut fx: ResMut<Fx>,
    mut rs: ResMut<RenderState>,
) {
    let dt = time.delta_secs();
    // toggles
    if keys.just_pressed(KeyCode::F1) {
        ov.graph = !ov.graph;
    }
    if keys.just_pressed(KeyCode::F2) {
        ov.anchors = !ov.anchors;
    }
    if keys.just_pressed(KeyCode::F3) {
        ov.random_colors = !ov.random_colors;
        rs.all_dirty = true;
    }
    if keys.just_pressed(KeyCode::F4) {
        ov.sleep_tint = !ov.sleep_tint;
    }
    if keys.just_pressed(KeyCode::F5) {
        ov.stats = !ov.stats;
    }
    if keys.just_pressed(KeyCode::KeyH) {
        ov.help = !ov.help;
    }
    if keys.just_pressed(KeyCode::KeyP) {
        sim.paused = !sim.paused;
    }
    if keys.just_pressed(KeyCode::Period) {
        sim.paused = true;
        sim.step_once = true;
    }
    if keys.just_pressed(KeyCode::Digit1) {
        ctl.weapon = WeaponSel::Ar;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        ctl.weapon = WeaponSel::Sniper;
    }
    if keys.just_pressed(KeyCode::Digit3) {
        ctl.weapon = WeaponSel::Launcher;
    }
    if keys.just_pressed(KeyCode::BracketLeft) {
        ctl.radius = (ctl.radius - 0.5).max(0.5);
    }
    if keys.just_pressed(KeyCode::BracketRight) {
        ctl.radius = (ctl.radius + 0.5).min(30.0);
    }
    if keys.just_pressed(KeyCode::KeyR) {
        *sim = Sim::new(&spec);
        render::clear_entities(&mut commands, &mut rs);
        rs.reset(&sim);
        fx.tracers.clear();
        fx.blasts.clear();
    }
    if keys.just_pressed(KeyCode::KeyE) {
        if let Some(p) = hit.0 {
            let r = ctl.radius;
            sim.world.explode(Explosion { center: to_engine(p), radius: r, inner_radius: r * 0.3, damage: EXPLOSION_DAMAGE, impulse: 1500.0 * r });
            fx.blasts.push((p, r, 0.0));
        }
    }
    if keys.just_pressed(KeyCode::KeyG) {
        if let Some(p) = hit.0 {
            sim.world.explode(Explosion { center: to_engine(p), radius: 6.0, inner_radius: 2.5, damage: 5000.0, impulse: 30000.0 });
            fx.blasts.push((p, 6.0, 0.0));
        }
    }
    // firing
    ctl.fire_cooldown -= dt;
    let fire = match ctl.weapon {
        WeaponSel::Ar => mouse.pressed(MouseButton::Left) && ctl.fire_cooldown <= 0.0,
        _ => mouse.just_pressed(MouseButton::Left),
    };
    if fire {
        if let Some((o, d)) = view_ray(&windows, &cams) {
            let (oe, de) = (to_engine(o), to_engine(d));
            let end = hit.0.unwrap_or(o + d * 300.0);
            let muzzle = o + d * 0.5 - Vec3::Y * 0.25;
            match ctl.weapon {
                WeaponSel::Ar => {
                    sim.world.fire(Projectile::hitscan(oe, de, Weapon::Ar));
                    ctl.fire_cooldown = 0.1;
                    fx.tracers.push((muzzle, end, 0.0, Color::srgb(1.0, 0.9, 0.4)));
                }
                WeaponSel::Sniper => {
                    sim.world.fire(Projectile::hitscan(oe, de, Weapon::Sniper));
                    fx.tracers.push((muzzle, end, 0.0, Color::srgb(0.5, 0.9, 1.0)));
                }
                WeaponSel::Launcher => {
                    let r = ctl.radius;
                    let e = Explosion { center: [0.0; 3], radius: r, inner_radius: r * 0.3, damage: EXPLOSION_DAMAGE, impulse: 1500.0 * r };
                    sim.world.fire(Projectile::ballistic(oe, de, Weapon::Launcher).with_explosion(e));
                    fx.tracers.push((muzzle, end, 0.0, Color::srgb(1.0, 0.4, 0.2)));
                    if let Some(p) = hit.0 {
                        let delay = (p - o).length() / Weapon::Launcher.params().speed;
                        fx.blasts.push((p, r, -delay));
                    }
                }
            }
        }
    }
}

/// Screenshot mode: one engine tick per rendered frame, scripted explosion, capture, exit.
fn screenshot_driver(
    mut commands: Commands,
    mut shot: ResMut<Shot>,
    mut sim: ResMut<Sim>,
    spec: Res<WorldSpec>,
    mut fx: ResMut<Fx>,
) {
    if shot.requested {
        return;
    }
    if shot.frame == shot.explode_frame {
        if let Some([x, y, z, r, dmg]) = shot.explode {
            sim.world.explode(Explosion { center: [x, y, z], radius: r, inner_radius: r * 0.3, damage: dmg, impulse: 1500.0 * r });
            fx.blasts.push((to_bevy([x, y, z]), r, 0.0));
        }
    }
    let t = std::time::Instant::now();
    let now = shot.started.elapsed().as_secs_f64();
    sim.step(&spec, now);
    shot.sim_ms += t.elapsed().as_secs_f64() * 1e3;
    shot.frame += 1;
    if shot.frame >= shot.frames {
        shot.requested = true;
        let el = shot.started.elapsed().as_secs_f64();
        let s = sim.world.stats();
        println!(
            "screenshot: {} frames in {:.2}s ({:.1} fps incl. sim), sim avg {:.2} ms/tick | static {} frozen {} gone {} in-flight {} clusters {}",
            shot.frames,
            el,
            shot.frames as f64 / el,
            shot.sim_ms / shot.frames as f64,
            s.static_chunks,
            s.frozen_chunks,
            s.gone_chunks,
            s.chunks_in_flight,
            s.clusters
        );
        commands
            .spawn(match &shot.target {
                Some(h) => Screenshot::image(h.clone()),
                None => Screenshot::primary_window(),
            })
            .observe(save_to_disk(shot.path.clone()))
            .observe(|_: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>| {
                exit.write(AppExit::Success);
            });
    }
}
