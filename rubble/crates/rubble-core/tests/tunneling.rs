//! Fast debris must not pass through thin slabs.
//!
//! A 0.4 m block falls from 60 m onto a 0.2 m slab on four corner columns. At ~33 m/s it moves
//! ~0.55 m per 1/60 s step: more than the slab and the block together, so without continuous
//! collision detection the slab is never seen overlapping it and the block lands on the ground.
//! (`WorldConfig::ccd_speed` gives fast small pieces predictive contacts; `ccd` sweeps them.)
use rubble_core::testutil::*;
use rubble_core::*;
use rubble_format::Bld;

const DT: f32 = 1.0 / 60.0;

/// (`drop_z` a multiple of 0.4 m.) The block hangs on the side of a 0.4 m pillar standing just outside the slab; destroying
/// the pillar cell it is bonded to drops it straight down onto the slab.
fn scene(drop_z: f32) -> (Bld, u32, u32) {
    let mut s = SynthBuilder::new("thin_slab");
    for (x, y) in [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)] {
        s.add([x, y, 0.0], [x + 1.0, y + 1.0, 3.0], 0, 0);
    }
    s.add_grid([0.0, 0.0, 3.0], [6.0, 6.0, 3.2], [6, 6, 1], 0, 0);
    // pillar in 0.4 m cells up to the block's top
    let n = ((drop_z + 0.4) / 0.4).round() as usize;
    let first = s.boxes.len() as u32;
    s.add_grid([6.0, 2.8, 0.0], [6.4, 3.2, n as f32 * 0.4], [1, 1, n], 0, 0);
    let holder = first + n as u32 - 1;
    let block = s.add([5.6, 2.8, drop_z], [6.0, 3.2, drop_z + 0.4], 0, 0);
    (s.build(), block, holder)
}

/// The block's centre every tick after it is released.
fn drop_track(cfg: WorldConfig, drop_z: f32, secs: f32) -> Vec<Vec3> {
    let (bld, block, holder) = scene(drop_z);
    let mut w = World::new(cfg);
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    w.step(DT);
    w.damage_chunk(b, holder, 1e9);
    (0..(secs / DT) as usize)
        .map(|_| {
            w.step(DT);
            w.drain_events();
            w.chunk_world_com(b.0 as usize, block as usize)
        })
        .collect()
}

/// Where the block (centre) comes to rest.
fn drop_block(cfg: WorldConfig, drop_z: f32) -> f32 {
    drop_track(cfg, drop_z, 6.0).last().unwrap().z
}

/// Lowest centre height while over the slab's footprint: ~3.4 when the slab caught the block,
/// well below the slab (≤ 0.2 on the ground) when it passed through. Rubbing down the pillar
/// gives the block some sideways speed, so after the hit it may skid off the slab's edge; where
/// it finally rests is not the point.
fn lowest_over_slab(cfg: WorldConfig, drop_z: f32, secs: f32) -> f32 {
    let track = drop_track(cfg, drop_z, secs);
    let over = |p: &&Vec3| (0.0..=6.0).contains(&p.x) && (0.0..=6.0).contains(&p.y);
    track.iter().filter(over).map(|p| p.z).fold(f32::INFINITY, f32::min)
}

#[test]
fn fast_block_is_caught_by_thin_slab() {
    // ~33 m/s: ~0.55 m per step
    let low = lowest_over_slab(WorldConfig::default(), 60.0, 5.0);
    assert!(low > 3.3 && low < 3.6, "lowest centre over the slab {low:.2} (on the slab: 3.4)");
}

#[test]
fn slow_block_stops_on_thin_slab() {
    // control: from 4.8 m (~4 m/s) the slab catches it without any help
    let z = drop_block(WorldConfig::default(), 4.8);
    assert!((z - 3.4).abs() < 0.1, "block should rest on the slab (centre z 3.4), got {z:.2}");
}

#[test]
fn very_fast_block_is_caught_by_thin_slab() {
    // ~54 m/s: almost 0.9 m per step
    let low = lowest_over_slab(WorldConfig::default(), 150.0, 7.0);
    assert!(low > 3.3 && low < 3.6, "lowest centre over the slab {low:.2} (on the slab: 3.4)");
}

#[test]
fn without_ccd_the_fast_block_tunnels() {
    // the scene really is a tunneling case: with continuous collision detection off the slab
    // never catches the block
    let cfg = WorldConfig { ccd: false, ccd_speed: 0.0, ..Default::default() };
    let low = lowest_over_slab(cfg, 60.0, 5.0);
    assert!(low < 1.0, "expected the block to pass through the slab, lowest centre over it {low:.2}");
}

#[test]
fn ccd_for_all_bodies_also_stops_it() {
    let cfg = WorldConfig { ccd: true, ccd_speed: 0.0, ..Default::default() };
    let low = lowest_over_slab(cfg, 60.0, 5.0);
    assert!(low > 3.3 && low < 3.6, "lowest centre over the slab {low:.2} (on the slab: 3.4)");
}
