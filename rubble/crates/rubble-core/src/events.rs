//! Engine events (DESIGN §3.5).
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
}
