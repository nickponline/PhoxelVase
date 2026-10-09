//! Engine events.
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, serde::Deserialize)]
pub struct BuildingId(pub u32);

/// Stable cluster id (slotmap key packed as u64).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, serde::Deserialize)]
pub struct ClusterId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakCause {
    Damage,
    Stress,
    Impact,
    /// lost support: the group above the joint has its centre of mass outside its support, or
    /// the chunk was left hanging by slivers
    Tipping,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type")]
pub enum Event {
    ChunkDestroyed { building: BuildingId, chunk: u32, pos: [f32; 3] },
    /// cosmetic: the chunk vanished (renderer spawns particles)
    ChunkShattered { building: BuildingId, chunk: u32, pos: [f32; 3], material: u16 },
    /// debug only (enable with `WorldConfig::emit_edge_events`)
    EdgeBroken { building: BuildingId, edge: u32, cause: BreakCause },
    CollapseWarning { building: BuildingId, chunks: Vec<u32>, delay: f32 },
    ClusterDetached {
        cluster: ClusterId,
        building: BuildingId,
        chunks: Vec<u32>,
        /// row-major 4x4 building-space -> world transform of every chunk in the cluster
        transform: [f32; 16],
        lin_vel: [f32; 3],
        ang_vel: [f32; 3],
    },
    ClusterSplit { parent: ClusterId, children: Vec<ClusterId> },
    ClusterFrozen { cluster: ClusterId, transform: [f32; 16] },
    ClusterDespawned { cluster: ClusterId },
    /// A moving piece hit something (another piece, the building, the ground) hard enough
    /// (`WorldConfig::impact_event_min`). Once per new contact: a piece resting on something
    /// does not repeat it. For sounds, camera shake, dust.
    Impact {
        building: BuildingId,
        /// impulse-weighted contact point
        pos: [f32; 3],
        /// contact normal (unit), pointing from what was hit towards the moving piece
        normal: [f32; 3],
        /// total contact impulse (N·s)
        impulse: f32,
        /// impulse / mass of the lighter moving side (m/s): how hard it hit, independent of size
        speed: f32,
        /// material of the moving piece's chunk at the contact
        material: u16,
    },
}
