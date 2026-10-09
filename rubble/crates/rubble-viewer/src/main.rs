//! `rubble-viewer`: interactive Bevy viewer for `rubble-core`.
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
mod beam;
mod demolish;
mod effects;
mod shards;

use bevy::light::CascadeShadowConfigBuilder;
use bevy::prelude::*;
use bevy::render::view::screenshot::{save_to_disk, Screenshot, ScreenshotCaptured};
use bevy::window::{PresentMode, PrimaryWindow, WindowResolution};
use camera::{camera_control, CamCtl, Walker};
use coords::{to_bevy, to_engine};
use render::{Materials, RenderState};
use rubble_core::physics::PhysicsBackend;
use rubble_core::{BuildingId, Explosion, StepTimings, World as EngineWorld};
use scene::{Args, Catalog, Scheduled, WorldSpec};

pub const DT: f32 = 1.0 / 60.0;
/// LMB (hold): a cutting beam. Damage per tick to every chunk within `BEAM_RADIUS` of the line
/// (×material multiplier), range, and push per tick on loose pieces it touches.
pub const BEAM_DAMAGE: f32 = 400.0;
pub const BEAM_RANGE: f32 = 400.0;
/// cutting radius (m); also the radius of the drawn glow
pub const BEAM_RADIUS: f32 = 0.25;
pub const BEAM_IMPULSE: f32 = 60.0;

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
    /// (building, chunk) of glass panes broken since the shard system last ran
    pub shattered: Vec<(u32, u32)>,
    /// building-space centre of each building (shards burst away from it)
    pub centres: Vec<Vec3>,
    /// engine z of the ground plane
    pub ground: Option<f32>,
    /// beam held this frame: engine (origin, direction)
    pub beam: Option<([f32; 3], [f32; 3])>,
    /// timed actions (engine time, action), e.g. demolition charges
    pub queued: Vec<(f32, Scheduled)>,
    /// (building, engine time): demolished buildings collapse without the warning delay until then
    pub hurry: Vec<(u32, f32)>,
    /// breaks / hard landings since the effects system last ran
    pub fx_events: Vec<effects::FxEvent>,
}

fn run_action(w: &mut EngineWorld, a: &Scheduled) {
    match a {
        Scheduled::Explode(e) => w.explode(*e),
        Scheduled::Fire(p) => w.fire(*p),
        Scheduled::Damage(b, cs, amt) => {
            for &c in cs {
                w.damage_chunk(BuildingId(*b), c, *amt);
            }
        }
    }
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
            shattered: vec![],
            centres: vec![],
            ground: spec.ground,
            beam: None,
            queued: vec![],
            hurry: vec![],
            fx_events: vec![],
        }
        .with_centres()
    }

    fn with_centres(mut self) -> Self {
        self.centres = self
            .world
            .buildings
            .iter()
            .map(|b| {
                let n = b.bld.chunks.len().max(1) as f32;
                b.bld.chunks.iter().fold(Vec3::ZERO, |a, c| a + Vec3::from_array(c.com)) / n
            })
            .collect();
        self
    }

    fn step(&mut self, spec: &WorldSpec, now: f64) {
        let now_t = self.world.time + 1e-6;
        while let Some((t, a)) = spec.actions.get(self.next_action) {
            if *t > now_t {
                break;
            }
            run_action(&mut self.world, a);
            self.next_action += 1;
        }
        for (_, a) in self.queued.extract_if(.., |(t, _)| *t <= now_t) {
            run_action(&mut self.world, &a);
        }
        if let Some((o, d)) = self.beam {
            self.world.beam(o, d, BEAM_RANGE, BEAM_RADIUS, BEAM_DAMAGE, BEAM_IMPULSE);
        }
        self.world.step(DT);
        let t = self.world.time;
        self.hurry.retain(|&(_, until)| t <= until);
        for &(b, _) in &self.hurry {
            self.world.hurry_collapse(BuildingId(b));
        }
        let events = self.world.drain_events();
        let n = events.len();
        // hard landings already reported this tick (the engine sends the hardest first)
        let mut landed: Vec<[f32; 3]> = vec![];
        for e in events {
            use effects::FxEvent;
            let fx = match e {
                rubble_core::Event::Impact { pos, impulse, .. } => {
                    let near = |q: &[f32; 3]| (q[0] - pos[0]).powi(2) + (q[1] - pos[1]).powi(2) + (q[2] - pos[2]).powi(2) < effects::IMPACT_SPACING.powi(2);
                    (impulse >= effects::IMPACT_MIN && landed.len() < effects::IMPACTS_PER_TICK && !landed.iter().any(near)).then(|| {
                        landed.push(pos);
                        FxEvent::Impact { pos, impulse }
                    })
                }
                rubble_core::Event::ChunkShattered { building, chunk, pos, material } => {
                    if self.world.buildings[building.0 as usize].glass(chunk as usize) {
                        self.shattered.push((building.0, chunk));
                        None
                    } else {
                        Some(FxEvent::Break { pos, material })
                    }
                }
                rubble_core::Event::ChunkDestroyed { building, chunk, pos } => {
                    let b = &self.world.buildings[building.0 as usize];
                    (!b.glass(chunk as usize)).then(|| FxEvent::Break { pos, material: b.bld.chunks[chunk as usize].material })
                }
                _ => None,
            };
            if let Some(fx) = fx {
                if self.fx_events.len() < effects::MAX_FX_EVENTS {
                    self.fx_events.push(fx);
                }
            }
        }
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
    /// engine `keep_debris` (applied to every world, so it survives reset and building switches)
    pub keep_debris: bool,
    /// fancy lighting / particle effects (see `effects`)
    pub lighting: bool,
    pub particles: bool,
}

/// Engine-space hit under the cursor this frame (Bevy-space point for the camera).
#[derive(Resource, Default)]
pub struct CursorHit(pub Option<Vec3>);

/// Short-lived visual feedback (explosion spheres, the beam), Bevy space.
#[derive(Resource, Default)]
pub struct Fx {
    pub blasts: Vec<(Vec3, f32, f32)>,
    /// beam drawn this frame (Bevy space start, end)
    pub beam: Option<(Vec3, Vec3)>,
}

#[derive(Resource)]
struct Shot {
    path: std::path::PathBuf,
    frames: u32,
    explode: Option<[f32; 5]>,
    explode_frame: u32,
    beam: Option<[f32; 6]>,
    frame: u32,
    started: std::time::Instant,
    sim_ms: f64,
    requested: bool,
    target: Option<Handle<Image>>,
    /// record mode: save a frame every this many ticks into `path` (a directory)
    record_every: Option<u32>,
    demolish_frame: Option<u32>,
    interior_frame: Option<u32>,
    warmup: f32,
    /// frames rendered since the last capture was requested (record mode exits after a few)
    done_frames: u32,
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
    // effects default on interactively, off for screenshots / recordings (opt in with
    // `--overlay lighting,particles`) so captures stay comparable
    let shot_mode = args.screenshot.is_some() || args.record.is_some();
    let mut overlays = Overlays {
        help: true,
        keep_debris: sim.world.cfg.keep_debris,
        lighting: !shot_mode,
        particles: !shot_mode,
        ..default()
    };
    for o in &args.overlays {
        match o.as_str() {
            "f1" | "graph" => overlays.graph = true,
            "f2" | "anchors" => overlays.anchors = true,
            "f3" | "colors" => overlays.random_colors = true,
            "f4" | "sleep" => overlays.sleep_tint = true,
            "f5" | "stats" => overlays.stats = true,
            "f6" | "debris" => overlays.keep_debris = true,
            "f7" | "lighting" => overlays.lighting = true,
            "f8" | "particles" => overlays.particles = true,
            "nolighting" => overlays.lighting = false,
            "noparticles" => overlays.particles = false,
            "nohelp" => overlays.help = false,
            _ => eprintln!("unknown overlay {o}"),
        }
    }
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
    .insert_resource(CursorHit::default())
    .insert_resource(Fx::default())
    .insert_resource(Walker::default())
    .insert_resource(RenderState::default())
    .insert_resource(sim)
    .insert_resource(spec)
    .insert_resource(catalog)
    .insert_resource(ArgsRes(args.clone()))
    .add_systems(Startup, (setup, shards::setup_shards, beam::setup_beam, effects::setup_effects))
    .add_systems(
        Update,
        (
            cycle_buildings,
            cursor_pick,
            camera_control,
            controls,
            effects::apply_lighting,
            render::sync_render,
            shards::update_shards,
            effects::update_effects,
            beam::draw_beam,
            overlay::draw_overlays,
            overlay::update_text,
        )
            .chain(),
    );
    if let Some(path) = args.screenshot.as_ref().or(args.record.as_ref()) {
        if args.record.is_some() {
            if let Err(e) = std::fs::create_dir_all(path) {
                eprintln!("error: {}: {e}", path.display());
                std::process::exit(1);
            }
        }
        app.insert_resource(Shot {
            path: path.clone(),
            record_every: args.record.as_ref().map(|_| args.record_every),
            demolish_frame: args.demolish_frame,
            interior_frame: args.interior_frame,
            warmup: args.warmup,
            done_frames: 0,
            frames: args.frames,
            explode: args.explode,
            explode_frame: args.explode_frame,
            beam: args.beam,
            frame: 0,
            started: std::time::Instant::now(),
            sim_ms: 0.0,
            requested: false,
            target: None,
        })
        .add_systems(Update, screenshot_driver.before(render::sync_render).after(controls))
        // the capture window still gets focus and keystrokes; a scripted run must not react
        .add_systems(PreUpdate, ignore_input.after(bevy::input::InputSystems));
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
        // point-light shadow detail follows this camera (also when it renders offscreen)
        bevy::camera::ShadowLodOrigin,
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
    shard_q: Query<Entity, With<shards::Shard>>,
    effect_q: Query<Entity, effects::EffectFilter>,
    mut walker: ResMut<Walker>,
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
    shards::clear_shards(&mut commands, &shard_q);
    effects::clear_effects(&mut commands, &effect_q);
    rs.reset(&sim);
    fx.blasts.clear();
    println!(
        "rubble-viewer: [{}] {} | {} chunks, loaded in {:.0} ms",
        catalog.label(),
        new_spec.title,
        new_spec.total_chunks(),
        t0.elapsed().as_secs_f64() * 1e3
    );
    let (_, ext) = framing(&new_spec);
    walker.on = false;
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
    walker: Res<Walker>,
    mut hit: ResMut<CursorHit>,
) {
    hit.0 = None;
    let Some(ray) = view_ray(&windows, &cams, walker.on) else { return };
    let o = to_engine(ray.0);
    let d = to_engine(ray.1);
    if let Some(h) = sim.world.phys.cast_ray(o.into(), d.into(), 2000.0, None) {
        hit.0 = Some(ray.0 + ray.1 * h.toi);
    }
}

/// Bevy-space (origin, dir) of the pick ray: through the cursor, or the screen centre
/// (crosshair) when `centred` (walk mode) or the cursor is outside the window.
fn view_ray(windows: &Query<&Window, With<PrimaryWindow>>, cams: &Query<(&Camera, &GlobalTransform)>, centred: bool) -> Option<(Vec3, Vec3)> {
    let win = windows.single().ok()?;
    let (cam, gt) = cams.single().ok()?;
    let center = Vec2::new(win.width() * 0.5, win.height() * 0.5);
    let p = if centred { center } else { win.cursor_position().unwrap_or(center) };
    let r = cam.viewport_to_world(gt, p).ok()?;
    Some((r.origin, *r.direction))
}

#[allow(clippy::too_many_arguments)]
fn controls(
    mut commands: Commands,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    cams: Query<(&Camera, &GlobalTransform)>,
    mut sim: ResMut<Sim>,
    spec: Res<WorldSpec>,
    mut ov: ResMut<Overlays>,
    hit: Res<CursorHit>,
    mut fx: ResMut<Fx>,
    mut rs: ResMut<RenderState>,
    shard_q: Query<Entity, With<shards::Shard>>,
    effect_q: Query<Entity, effects::EffectFilter>,
    walker: Res<Walker>,
) {
    // toggles
    if keys.just_pressed(KeyCode::Digit1) {
        ov.graph = !ov.graph;
    }
    if keys.just_pressed(KeyCode::Digit2) {
        ov.anchors = !ov.anchors;
    }
    if keys.just_pressed(KeyCode::Digit3) {
        ov.random_colors = !ov.random_colors;
        rs.all_dirty = true;
    }
    if keys.just_pressed(KeyCode::Digit4) {
        ov.sleep_tint = !ov.sleep_tint;
    }
    if keys.just_pressed(KeyCode::Digit5) {
        ov.stats = !ov.stats;
    }
    if keys.just_pressed(KeyCode::Digit6) {
        ov.keep_debris = !ov.keep_debris;
    }
    if keys.just_pressed(KeyCode::Digit7) {
        ov.lighting = !ov.lighting;
    }
    if keys.just_pressed(KeyCode::Digit8) {
        ov.particles = !ov.particles;
    }
    // every frame, so a fresh world (reset, next building) picks it up too; affects debris
    // created from now on
    sim.world.cfg.keep_debris = ov.keep_debris;
    // the camera is where the destruction is watched from: debris near it is kept longest
    if let Ok((_, gt)) = cams.single() {
        sim.world.set_focus(&[to_engine(gt.translation())]);
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
    if keys.just_pressed(KeyCode::KeyR) {
        *sim = Sim::new(&spec);
        render::clear_entities(&mut commands, &mut rs);
        shards::clear_shards(&mut commands, &shard_q);
        effects::clear_effects(&mut commands, &effect_q);
        rs.reset(&sim);
        fx.blasts.clear();
    }
    if keys.just_pressed(KeyCode::KeyG) {
        if let Some(p) = hit.0 {
            sim.world.explode(Explosion { center: to_engine(p), radius: 6.0, inner_radius: 2.5, damage: 5000.0, impulse: 30000.0 });
            fx.blasts.push((p, 6.0, 0.0));
        }
    }
    // X: demolish the building under the cursor (all of them if the cursor is on nothing),
    // fall side sideways relative to the view, left or right at random
    if keys.just_pressed(KeyCode::KeyX) {
        let seed = (sim.world.time * 1000.0) as u32 ^ sim.queued.len() as u32;
        let view = view_ray(&windows, &cams, walker.on).map_or([0.0, 1.0], |(_, d)| {
            let e = to_engine(d);
            [e[0], e[1]]
        });
        let side = if seed & 1 == 0 { 1.0 } else { -1.0 };
        let a = side * std::f32::consts::FRAC_PI_2 + ((seed >> 1) % 61) as f32 / 60.0 * 1.0 - 0.5;
        let (sn, cs) = a.sin_cos();
        let fall = [cs * view[0] - sn * view[1], sn * view[0] + cs * view[1]];
        let targets = target_buildings(&sim, hit.0);
        demolish_buildings(&mut sim, &mut fx, &targets, fall, seed);
    }
    // Z: interior demolition of the building under the cursor (all of them if on nothing):
    // random explosions on every floor and the roof
    if keys.just_pressed(KeyCode::KeyZ) {
        let seed = (sim.world.time * 1000.0) as u32 ^ sim.queued.len() as u32 ^ 0x5A5A;
        let targets = target_buildings(&sim, hit.0);
        demolish_interior(&mut sim, &mut fx, &targets, seed);
    }
    // LMB (hold): cutting beam through everything in line, while the button is down
    sim.beam = None;
    fx.beam = None;
    if mouse.pressed(MouseButton::Left) {
        if let Some((o, d)) = view_ray(&windows, &cams, walker.on) {
            sim.beam = Some((to_engine(o), to_engine(d)));
            // walk mode: from a "gun" at the lower right of the view
            let muzzle = match cams.single() {
                Ok((_, gt)) if walker.on => o + d * 0.5 + gt.right() * 0.15 - gt.up() * 0.15,
                _ => o + d * 0.5 - Vec3::Y * 0.25,
            };
            fx.beam = Some((muzzle, o + d * BEAM_RANGE));
        }
    }
}

/// The building nearest (horizontally) to the cursor hit `p`, or every building when the
/// cursor is on nothing.
fn target_buildings(sim: &Sim, p: Option<Vec3>) -> Vec<usize> {
    let world_centre =
        |b: usize| to_bevy(sim.world.buildings[b].pose.transform_point(rubble_core::Vec3::from_array(sim.centres[b].to_array())).to_array());
    match p {
        Some(p) => (0..sim.world.buildings.len())
            .min_by(|&a, &b| world_centre(a).xz().distance(p.xz()).total_cmp(&world_centre(b).xz().distance(p.xz())))
            .into_iter()
            .collect(),
        None => (0..sim.world.buildings.len()).collect(),
    }
}

/// Z: queue random explosions on every floor and the roof of `targets`.
fn demolish_interior(sim: &mut Sim, fx: &mut Fx, targets: &[usize], seed: u32) {
    let t0 = sim.world.time;
    for &b in targets {
        for k in demolish::interior_charges(&sim.world, b, seed.wrapping_add(b as u32)) {
            fx.blasts.push((to_bevy(k.blast.center), k.blast.radius, -k.delay));
            sim.queued.push((t0 + k.delay, Scheduled::Explode(k.blast)));
        }
    }
}

/// X: queue the demolition charges of `targets`, falling roughly along `fall` (engine xy).
fn demolish_buildings(sim: &mut Sim, fx: &mut Fx, targets: &[usize], fall: [f32; 2], seed: u32) {
    let t0 = sim.world.time;
    for &b in targets {
        let charges = demolish::charges(&sim.world, b, fall, seed.wrapping_add(b as u32));
        let last = charges.iter().map(|k| k.delay).fold(0.0, f32::max);
        sim.hurry.push((b as u32, t0 + last + 0.5));
        for k in charges {
            fx.blasts.push((to_bevy(k.blast.center), k.blast.radius, -k.delay));
            sim.queued.push((t0 + k.delay, Scheduled::Explode(k.blast)));
            sim.queued.push((t0 + k.delay, Scheduled::Damage(b as u32, k.chunks, 1e9)));
        }
    }
}

/// Screenshot / record mode: drop this frame's keyboard and mouse input, so typing while a
/// capture runs (its window takes focus) cannot toggle overlays or trigger demolitions.
fn ignore_input(
    mut keys: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
    mut motion: ResMut<bevy::input::mouse::AccumulatedMouseMotion>,
) {
    keys.reset_all();
    mouse.reset_all();
    motion.delta = Vec2::ZERO;
}

/// Screenshot mode: one engine tick per rendered frame, scripted explosion, capture, exit.
fn screenshot_driver(
    mut commands: Commands,
    mut shot: ResMut<Shot>,
    mut sim: ResMut<Sim>,
    spec: Res<WorldSpec>,
    mut fx: ResMut<Fx>,
    mut exit: MessageWriter<AppExit>,
) {
    if shot.requested {
        // record mode: many captures are in flight and the final one's exit hook does not
        // always fire; give the readbacks a few frames, then quit
        if shot.record_every.is_some() {
            shot.done_frames += 1;
            if shot.done_frames > 30 {
                exit.write(AppExit::Success);
            }
        }
        return;
    }
    if shot.frame == 0 && shot.started.elapsed().as_secs_f32() < shot.warmup {
        return;
    }
    if shot.frame == shot.explode_frame {
        if let Some([x, y, z, r, dmg]) = shot.explode {
            sim.world.explode(Explosion { center: [x, y, z], radius: r, inner_radius: r * 0.3, damage: dmg, impulse: 1500.0 * r });
            fx.blasts.push((to_bevy([x, y, z]), r, 0.0));
        }
    }
    if shot.demolish_frame == Some(shot.frame) {
        let targets: Vec<usize> = (0..sim.world.buildings.len()).collect();
        demolish_buildings(&mut sim, &mut fx, &targets, [1.0, 0.0], 7);
    }
    if shot.interior_frame == Some(shot.frame) {
        let targets: Vec<usize> = (0..sim.world.buildings.len()).collect();
        demolish_interior(&mut sim, &mut fx, &targets, 7);
    }
    if let Some(every) = shot.record_every {
        if shot.frame % every == 0 && shot.frame + 1 < shot.frames {
            let p = shot.path.join(format!("frame_{:05}.png", shot.frame / every));
            let shot_of = match &shot.target {
                Some(h) => Screenshot::image(h.clone()),
                None => Screenshot::primary_window(),
            };
            commands.spawn(shot_of).observe(save_to_disk(p));
        }
    }
    if let Some([x, y, z, dx, dy, dz]) = shot.beam {
        if shot.frame >= shot.explode_frame {
            sim.beam = Some(([x, y, z], [dx, dy, dz]));
            let (o, d) = (to_bevy([x, y, z]), to_bevy([dx, dy, dz]).normalize());
            fx.beam = Some((o, o + d * BEAM_RANGE));
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
            .observe(save_to_disk(match shot.record_every {
                Some(every) => shot.path.join(format!("frame_{:05}.png", shot.frame.div_ceil(every))),
                None => shot.path.clone(),
            }))
            .observe(|_: On<ScreenshotCaptured>, mut exit: MessageWriter<AppExit>| {
                exit.write(AppExit::Success);
            });
    }
}
