//! Fly / orbit camera (operates in Bevy Y-up space).
//! Fly: WASD move, Space/C up/down, Shift fast, right mouse held = mouse look, wheel = speed.
//! Orbit (toggle O): right mouse drag orbits the focus, wheel zooms, WASD pans the focus.
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll, MouseScrollUnit};
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CamMode {
    Fly,
    Orbit,
}

#[derive(Component)]
pub struct CamCtl {
    pub yaw: f32,
    pub pitch: f32,
    pub mode: CamMode,
    pub focus: Vec3,
    pub dist: f32,
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
        let c = CamCtl { yaw, pitch, mode: CamMode::Fly, focus: target, dist: (target - eye).length(), speed: 12.0 };
        let xf = Transform::from_translation(eye).with_rotation(c.rotation());
        (c, xf)
    }
}

pub fn camera_control(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut cams: Query<(&mut Transform, &mut CamCtl)>,
    focus_hint: Res<crate::CursorHit>,
) {
    let Ok((mut xf, mut c)) = cams.single_mut() else { return };
    let dt = time.delta_secs().min(0.1);
    let look = mouse.pressed(MouseButton::Right);
    if let Ok(mut co) = cursor.single_mut() {
        if mouse.just_pressed(MouseButton::Right) {
            co.grab_mode = CursorGrabMode::Locked;
            co.visible = false;
        }
        if mouse.just_released(MouseButton::Right) {
            co.grab_mode = CursorGrabMode::None;
            co.visible = true;
        }
    }
    if keys.just_pressed(KeyCode::KeyO) {
        c.mode = match c.mode {
            CamMode::Fly => {
                // orbit around what is under the cursor, else a point ahead
                let f = focus_hint.0.unwrap_or(xf.translation + xf.forward() * 30.0);
                c.focus = f;
                c.dist = (xf.translation - f).length().max(2.0);
                CamMode::Orbit
            }
            CamMode::Orbit => CamMode::Fly,
        };
        if c.mode == CamMode::Orbit {
            // aim at the focus
            let d = (c.focus - xf.translation).normalize_or(Vec3::NEG_Z);
            c.yaw = (-d.x).atan2(-d.z);
            c.pitch = d.y.clamp(-1.0, 1.0).asin();
        }
    }
    if look {
        let d = motion.delta;
        c.yaw -= d.x * 0.0025;
        c.pitch = (c.pitch - d.y * 0.0025).clamp(-1.54, 1.54);
    }
    let wheel = match scroll.unit {
        MouseScrollUnit::Line => scroll.delta.y,
        MouseScrollUnit::Pixel => scroll.delta.y / 40.0,
    };
    let fast = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) { 4.0 } else { 1.0 };
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
    if keys.pressed(KeyCode::Space) {
        mv += Vec3::Y;
    }
    if keys.pressed(KeyCode::KeyC) {
        mv -= Vec3::Y;
    }
    match c.mode {
        CamMode::Fly => {
            if wheel != 0.0 {
                c.speed = (c.speed * 1.15f32.powf(wheel)).clamp(1.0, 500.0);
            }
            xf.translation += mv * c.speed * fast * dt;
            xf.rotation = rot;
        }
        CamMode::Orbit => {
            if wheel != 0.0 {
                c.dist = (c.dist * 0.88f32.powf(wheel)).clamp(1.0, 2000.0);
            }
            // pan the focus on the horizontal plane
            let flat_f = Vec3::new(fwd.x, 0.0, fwd.z).normalize_or_zero();
            let (f, r) = (mv.dot(fwd), mv.dot(right));
            let pan = flat_f * f + right * r + Vec3::Y * mv.y;
            let step = c.dist * 0.6 * fast * dt;
            c.focus += pan * step;
            xf.rotation = rot;
            xf.translation = c.focus - fwd * c.dist;
        }
    }
}
