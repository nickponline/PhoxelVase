//! `rubble-core`: single-player destruction physics engine (DESIGN §3).
//!
//! Layers: [`building`] (SoA chunk state, CSR graph, incremental connectivity),
//! [`world`] (tick pipeline, damage, promotion, clusters, settling, budgets),
//! [`physics`] (thin [`physics::PhysicsBackend`] boundary; Rapier implementation).

pub mod building;
pub mod config;
pub mod events;
pub mod math;
pub mod physics;
#[cfg(feature = "testutil")]
pub mod testutil;
pub mod world;

pub use building::{Building, ChunkState};
pub use config::{Explosion, Projectile, ProjectileKind, StressSettings, Weapon, WeaponParams, WorldConfig};
pub use events::{BreakCause, BuildingId, ClusterId, Event};
pub use math::{pose_to_rowmajor, Isometry, Pose, Vec3};
pub use world::{BuildingState, Cluster, LoadError, Stats, StepTimings, World};
pub use rapier3d;
pub use rubble_format;
