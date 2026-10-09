//! `WorldConfig::keep_debris`: destroyed chunks stay in the world as falling debris that settles
//! into frozen rubble, instead of vanishing.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;

/// 3×3×3 block of 1 m cells (1 m³: above `debris_max_volume`); destroy the top centre cell.
fn destroy_top_centre(keep: bool) -> (ChunkState, f32) {
    let bld = grid_block(3, 3, 3, 1.0);
    let top = bld
        .chunks
        .iter()
        .position(|c| c.aabb_min[2] > 1.5 && (c.com[0] - 1.5).abs() < 0.1 && (c.com[1] - 1.5).abs() < 0.1)
        .unwrap() as u32;
    let mut cfg = WorldConfig::default();
    cfg.keep_debris = keep;
    let mut w = World::new(cfg);
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    w.step(DT);
    w.damage_chunk(b, top, 1e9);
    for _ in 0..(15.0 / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
    (w.building(b).state[top as usize], w.chunk_world_com(b.0 as usize, top as usize).z)
}

#[test]
fn destroyed_chunk_vanishes_by_default() {
    assert_eq!(destroy_top_centre(false).0, ChunkState::Gone);
}

#[test]
fn destroyed_chunk_stays_as_rubble_with_keep_debris() {
    let (st, z) = destroy_top_centre(true);
    assert_eq!(st, ChunkState::Frozen, "kept debris should settle into frozen rubble");
    assert!(z > 2.0 && z < 3.6, "it should rest on the block (where it was), com z = {z}");
}
