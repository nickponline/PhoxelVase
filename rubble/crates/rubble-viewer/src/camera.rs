//! Camera (operates in Bevy Y-up space) in one of two modes, toggled with M:
//! - fly: WASD move, Q/E down/up, right mouse held = mouse look.
//! - walk (FPS): a kinematic capsule in the engine world; mouse look is always on (cursor
//!   locked), WASD walk, Shift run, Space jump. It climbs stairs and stands on debris but is
//!   never simulated, so the destruction cannot push it (falling pieces it ends up inside are
//!   resolved by the controller).
use crate::coords::{to_bevy, to_engine};
use crate::Sim;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};
use rubble_core::physics::PhysicsBackend;

/// Capsule radius and total height (m), eye height above the feet.
const RADIUS: f32 = 0.3;
const HEIGHT: f32 = 1.8;
const EYE: f32 = 1.65;
const WALK_SPEED: f32 = 4.5;
const RUN_SPEED: f32 = 8.0;
const JUMP_SPEED: f32 = 5.5;
const GRAVITY: f32 = 18.0;
/// Walk mode: falling this far below where it started puts the walker back.
const FALL_LIMIT: f32 = 60.0;

#[derive(Component)]
pub struct CamCtl {
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
}

impl CamCtl {
    pub fn rotation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }
    /// Build a controller at `eye` looking at `target` (Bevy space).
    pub fn looking_at(eye: Vec3, target: Vec3) -> (Self, Transform) {
        let d = (target - eye).normalize_or(Vec3::NEG_Z);
        let yaw = (-d.x).atan2(-d.z);
        let pitch = d.y.clamp(-1.0, 1.0).asin();
        let c = CamCtl { yaw, pitch, speed: 12.0 };
        let xf = Transform::from_translation(eye).with_rotation(c.rotation());
        (c, xf)
    }
}

/// Walk (FPS) mode state. Positions are engine space (Z-up).
#[derive(Resource, Default)]
pub struct Walker {
    pub on: bool,
    /// capsule centre
    pub centre: [f32; 3],
    pub vz: f32,
    pub grounded: bool,
    /// where walk mode was entered (respawn point after falling off the world)
    pub spawn: [f32; 3],
}

impl Walker {
    /// Enter walk mode under the Bevy-space eye: the capsule stands on the first surface below
    /// it (or stays where it is and falls, if there is none).
    fn enter(&mut self, sim: &Sim, eye: Vec3) {
        let e = to_engine(eye);
        let half = HEIGHT * 0.5;
        let mut c = [e[0], e[1], e[2] - (EYE - half)];
        if let Some(h) = sim.world.phys.cast_sphere(e.into(), [0.0, 0.0, -2000.0].into(), RADIUS * 0.9) {
            c[2] = e[2] - h.toi * 2000.0 + half + 0.05;
        }
        *self = Walker { on: true, centre: c, vz: 0.0, grounded: false, spawn: c };
    }

    fn eye(&self) -> Vec3 {
        let c = self.centre;
        to_bevy([c[0], c[1], c[2] - HEIGHT * 0.5 + EYE])
    }
}

fn set_grab(cursor: &mut Query<&mut CursorOptions, With<PrimaryWindow>>, grab: bool) {
    if let Ok(mut co) = cursor.single_mut() {
        co.grab_mode = if grab { CursorGrabMode::Locked } else { CursorGrabMode::None };
        co.visible = !grab;
    }
}

pub fn camera_control(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    sim: Res<Sim>,
    mut walker: ResMut<Walker>,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut cams: Query<(&mut Transform, &mut CamCtl)>,
    mut was_on: Local<bool>,
) {
    let Ok((mut xf, mut c)) = cams.single_mut() else { return };
    let dt = time.delta_secs().min(0.1);
    if keys.just_pressed(KeyCode::KeyM) {
        if walker.on {
            walker.on = false;
        } else {
            walker.enter(&sim, xf.translation);
        }
    }
    // also catches walk mode being switched off elsewhere (building change)
    if walker.on != *was_on {
        set_grab(&mut cursor, walker.on);
        *was_on = walker.on;
    }
    if walker.on {
        walk(&keys, motion.delta, dt, &sim, &mut walker, &mut c);
        xf.translation = walker.eye();
        xf.rotation = c.rotation();
        return;
    }
    let look = mouse.pressed(MouseButton::Right);
    if mouse.just_pressed(MouseButton::Right) {
        set_grab(&mut cursor, true);
    }
    if mouse.just_released(MouseButton::Right) {
        set_grab(&mut cursor, false);
    }
    if look {
        let d = motion.delta;
        c.yaw -= d.x * 0.0025;
        c.pitch = (c.pitch - d.y * 0.0025).clamp(-1.54, 1.54);
    }
    let rot = c.rotation();
    let mut mv = Vec3::ZERO;
    let fwd = rot * Vec3::NEG_Z;
    let right = rot * Vec3::X;
    if keys.pressed(KeyCode::KeyW) {
        mv += fwd;
    }
    if keys.pressed(KeyCode::KeyS) {
        mv -= fwd;
    }
    if keys.pressed(KeyCode::KeyD) {
        mv += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        mv -= right;
    }
    if keys.pressed(KeyCode::KeyE) {
        mv += Vec3::Y;
    }
    if keys.pressed(KeyCode::KeyQ) {
        mv -= Vec3::Y;
    }
    xf.translation += mv * c.speed * dt;
    xf.rotation = rot;
}

/// One frame of walk mode: mouse look, then move the capsule through the engine world.
fn walk(keys: &ButtonInput<KeyCode>, mouse: Vec2, dt: f32, sim: &Sim, w: &mut Walker, c: &mut CamCtl) {
    c.yaw -= mouse.x * 0.0025;
    c.pitch = (c.pitch - mouse.y * 0.0025).clamp(-1.54, 1.54);
    // horizontal input in Bevy space (yaw only), then to engine
    let (s, co) = c.yaw.sin_cos();
    let fwd = Vec3::new(-s, 0.0, -co);
    let right = Vec3::new(co, 0.0, -s);
    let mut mv = Vec3::ZERO;
    if keys.pressed(KeyCode::KeyW) {
        mv += fwd;
    }
    if keys.pressed(KeyCode::KeyS) {
        mv -= fwd;
    }
    if keys.pressed(KeyCode::KeyD) {
        mv += right;
    }
    if keys.pressed(KeyCode::KeyA) {
        mv -= right;
    }
    let speed = if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) { RUN_SPEED } else { WALK_SPEED };
    let h = to_engine(mv.normalize_or_zero() * speed * dt);
    if w.grounded && keys.just_pressed(KeyCode::Space) {
        w.vz = JUMP_SPEED;
        w.grounded = false;
    }
    w.vz -= GRAVITY * dt;
    let desired = [h[0], h[1], w.vz * dt];
    let half_seg = HEIGHT * 0.5 - RADIUS;
    let (t, grounded) = sim.world.phys.move_capsule(w.centre.into(), half_seg, RADIUS, desired.into(), dt);
    for i in 0..3 {
        w.centre[i] += t[i];
    }
    w.grounded = grounded;
    // landed, or bumped the head
    if (grounded && w.vz < 0.0) || (w.vz > 0.0 && t[2] < desired[2] * 0.5) {
        w.vz = 0.0;
    }
    if w.centre[2] < w.spawn[2] - FALL_LIMIT {
        w.centre = w.spawn;
        w.vz = 0.0;
    }
}
