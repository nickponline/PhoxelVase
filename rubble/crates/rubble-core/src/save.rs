//! Persistent damage: what is left of a building, independent of the moving simulation.
//!
//! [`crate::World::save_damage`] captures a building's damage as plain data (serde): destroyed
//! chunks, broken joints, reduced hit points and joint health, and settled rubble with its pose
//! (pieces still moving are saved as rubble where they are). [`crate::World::restore_damage`]
//! applies it to a freshly loaded copy of the same building, so destruction can outlive a world:
//! save games, levels that stay damaged, streaming a district in and out.
use serde::{Deserialize, Serialize};

pub const DAMAGE_STATE_VERSION: u32 = 1;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DamageState {
    pub version: u32,
    /// chunk / joint counts of the building it was saved from (checked on restore)
    pub chunks: u32,
    pub edges: u32,
    /// destroyed chunks
    pub gone: Vec<u32>,
    /// broken joints
    pub broken: Vec<u32>,
    /// (chunk, hit points) of damaged but standing chunks
    pub hp: Vec<(u32, f32)>,
    /// (joint, health) of weakened but intact joints
    pub edge_health: Vec<(u32, f32)>,
    /// settled rubble: chunks that came to rest together share one pose
    pub rubble: Vec<RubbleGroup>,
}

/// Chunks lying at one pose, relative to the building's own pose (so a building restored at
/// another place keeps its rubble with it).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RubbleGroup {
    /// translation and rotation (x, y, z, w) of the chunks' frame in building space
    pub pos: [f32; 3],
    pub rot: [f32; 4],
    pub chunks: Vec<u32>,
}
