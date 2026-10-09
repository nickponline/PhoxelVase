//! Fly camera (operates in Bevy Y-up space).
//! WASD move, Q/E down/up, right mouse held = mouse look.
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

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

pub fn camera_control(
    time: Res<Time<Real>>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut cursor: Query<&mut CursorOptions, With<PrimaryWindow>>,
    mut cams: Query<(&mut Transform, &mut CamCtl)>,
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
