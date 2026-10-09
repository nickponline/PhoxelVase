//! Engine (Z-up) <-> Bevy (Y-up) conversion. **This is the only place axes are swapped.**
//!
//! The engine is right-handed Z-up; Bevy is right-handed Y-up. The mapping is a
//! proper rotation of -90° about X:  `(x, y, z)_engine  ->  (x, z, -y)_bevy`.
//!
//! Render meshes stay in engine *building space* (Z-up); an entity's Bevy `Transform` is
//! `R * engine_pose`, so a building-space vertex `v` lands at `R (q v + t)`.
use bevy::math::{Quat, Vec3};
use bevy::transform::components::Transform;

/// Rotation taking engine (Z-up) vectors to Bevy (Y-up) vectors.
pub fn zup_to_yup() -> Quat {
    Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2)
}

/// Engine point/vector -> Bevy.
#[inline]
pub fn to_bevy(p: [f32; 3]) -> Vec3 {
    Vec3::new(p[0], p[2], -p[1])
}

/// Bevy point/vector -> engine.
#[inline]
pub fn to_engine(v: Vec3) -> [f32; 3] {
    [v.x, -v.z, v.y]
}

/// Engine rigid pose (translation, quaternion xyzw; maps building space -> engine world)
/// -> Bevy `Transform` for a mesh authored in building space.
pub fn pose_to_transform(pos: [f32; 3], rot_xyzw: [f32; 4]) -> Transform {
    Transform {
        translation: to_bevy(pos),
        rotation: zup_to_yup() * Quat::from_array(rot_xyzw),
        scale: Vec3::ONE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rotation_matches_point_map() {
        let r = zup_to_yup();
        for p in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]] {
            let a = r * Vec3::from_array(p);
            assert!((a - to_bevy(p)).length() < 1e-6, "{p:?}");
            assert!((Vec3::from_array(to_engine(a)) - Vec3::from_array(p)).length() < 1e-6);
        }
    }
}
