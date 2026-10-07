//! Small math helpers (glam types re-exported through rapier/parry).
pub use rapier3d::math::{Mat3, Pose, Rotation as Quat, Vec3};

/// Rigid placement of a building: translation + yaw about +Z (no scale), DESIGN §1.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Isometry {
    pub pos: [f32; 3],
    pub yaw: f32,
}

impl Isometry {
    pub fn new(pos: [f32; 3], yaw: f32) -> Self {
        Isometry { pos, yaw }
    }
    pub fn identity() -> Self {
        Self::default()
    }
    pub fn to_pose(&self) -> Pose {
        Pose::from_parts(Vec3::from(self.pos), Quat::from_rotation_z(self.yaw))
    }
}

/// Row-major 4x4 matrix (same convention as `.bld` ELEM.frame).
pub fn pose_to_rowmajor(p: &Pose) -> [f32; 16] {
    let m = p.to_mat4().transpose();
    m.to_cols_array()
}

pub fn v3(a: [f32; 3]) -> Vec3 {
    Vec3::from(a)
}

pub fn a3(v: Vec3) -> [f32; 3] {
    v.to_array()
}

/// Symmetric tensor from `.bld` layout [Ixx, Iyy, Izz, Ixy, Ixz, Iyz].
pub fn inertia_mat(i: &[f32; 6]) -> Mat3 {
    Mat3::from_cols(
        Vec3::new(i[0], i[3], i[4]),
        Vec3::new(i[3], i[1], i[5]),
        Vec3::new(i[4], i[5], i[2]),
    )
}

/// Parallel-axis shift term m (|d|^2 E - d d^T).
pub fn parallel_axis(m: f32, d: Vec3) -> Mat3 {
    let dd = d.dot(d);
    Mat3::from_cols(
        Vec3::new(dd - d.x * d.x, -d.y * d.x, -d.z * d.x),
        Vec3::new(-d.x * d.y, dd - d.y * d.y, -d.z * d.y),
        Vec3::new(-d.x * d.z, -d.y * d.z, dd - d.z * d.z),
    ) * m
}
