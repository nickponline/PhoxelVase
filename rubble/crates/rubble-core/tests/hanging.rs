//! Regression: rubble must not hang in the air after whatever held it up is removed.
//!
//! Setup: a 6x6 m slab (1 m cells) on four corner columns, and a 2x2x1 m block floating 1 cm
//! above the slab with no bonds. At load the engine keeps unanchored pieces as frozen rubble, so
//! the block starts frozen in mid-air. The support check must let it drop onto the slab, where
//! it freezes again; destroying the slab cells under it must then send it to the ground.
use rubble_core::testutil::*;
use rubble_core::*;
use rubble_format::Bld;

const DT: f32 = 1.0 / 60.0;

fn scene() -> (Bld, Vec<u32>, u32) {
    let mut s = SynthBuilder::new("slab_with_block");
    for (x, y) in [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)] {
        s.add([x, y, 0.0], [x + 1.0, y + 1.0, 3.0], 0, 0);
    }
    let first_slab = s.boxes.len() as u32;
    s.add_grid([0.0, 0.0, 3.0], [6.0, 6.0, 3.3], [6, 6, 1], 0, 0);
    // slab cells under the block (x, y in 1..5)
    let under: Vec<u32> = (0..36u32)
        .filter(|i| (1..5).contains(&(i % 6)) && (1..5).contains(&(i / 6)))
        .map(|i| first_slab + i)
        .collect();
    let block = s.add([2.0, 2.0, 3.31], [4.0, 4.0, 4.31], 0, 0);
    (s.build(), under, block)
}

fn block_z(w: &World, b: BuildingId, block: u32) -> f32 {
    w.chunk_world_com(b.0 as usize, block as usize).z
}

fn run(w: &mut World, seconds: f32) {
    for _ in 0..(seconds / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
}

fn setup(cfg: WorldConfig) -> (World, BuildingId, Vec<u32>, u32) {
    let (bld, under, block) = scene();
    let mut w = World::new(cfg);
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    (w, b, under, block)
}

#[test]
fn rubble_falls_when_its_support_is_destroyed() {
    let (mut w, b, under, block) = setup(WorldConfig::default());
    run(&mut w, 4.0);
    let z = block_z(&w, b, block);
    assert!((z - 3.8).abs() < 0.05, "block should rest on the slab (com z 3.8), got {z}");
    assert_eq!(w.building(b).state[block as usize], ChunkState::Frozen, "settled block should be frozen");

    for c in under {
        w.damage_chunk(b, c, 1e9);
    }
    run(&mut w, 5.0);
    let z = block_z(&w, b, block);
    assert!(z < 0.6, "block must fall to the ground once the slab under it is gone, com z = {z}");
}

#[test]
fn without_thaw_the_block_hangs() {
    // documents the original bug: frozen rubble never re-checks its support
    let mut cfg = WorldConfig::default();
    cfg.thaw_unsupported = false;
    let (mut w, b, under, block) = setup(cfg);
    run(&mut w, 1.0);
    for c in under {
        w.damage_chunk(b, c, 1e9);
    }
    run(&mut w, 3.0);
    let z = block_z(&w, b, block);
    assert!(z > 3.5, "with thaw disabled the frozen block stays in the air (z {z})");
}
