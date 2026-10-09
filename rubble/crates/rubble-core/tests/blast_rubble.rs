//! Explosions push settled rubble.
//!
//! Rubble that has come to rest is frozen into static colliders. A blast next to it must thaw
//! the pieces it pushes hard enough back into moving bodies instead of only damaging them, so a
//! grenade in a rubble pile scatters it. Distant, weak pushes leave the pile frozen.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;

/// A loose 0.8 m concrete block lying on the ground (not anchored: frozen rubble once loaded),
/// and a one-chunk anchored post far away so the building has a root.
fn rubble_block() -> (World, BuildingId, u32) {
    let mut s = SynthBuilder::new("rubble_block");
    s.add([20.0, 20.0, 0.0], [20.5, 20.5, 1.0], 0, 0);
    let block = s.add([0.0, 0.0, 0.001], [0.8, 0.8, 0.801], 0, 0);
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building_bld(s.build(), Isometry::identity());
    w.add_ground_plane(0.0);
    for _ in 0..30 {
        w.step(DT);
    }
    (w, b, block)
}

fn blast(w: &mut World, center: [f32; 3], impulse: f32) {
    // tiny damage: the block must survive, only the push matters
    w.explode(Explosion { center, radius: 4.0, inner_radius: 1.0, damage: 1.0, impulse });
}

fn run(w: &mut World, secs: f32) {
    for _ in 0..(secs / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
}

#[test]
fn blast_scatters_frozen_rubble() {
    let (mut w, b, block) = rubble_block();
    assert_eq!(w.building(b).state[block as usize], ChunkState::Frozen, "the loose block starts as frozen rubble");
    let before = w.chunk_world_com(b.0 as usize, block as usize);
    blast(&mut w, [-1.5, 0.4, 0.4], 6000.0);
    run(&mut w, 0.5);
    let after = w.chunk_world_com(b.0 as usize, block as usize);
    let moved = (after - before).length();
    assert!(moved > 0.5, "the blast should throw the block, it moved {moved:.2} m");
    assert!(after.x > before.x, "away from the blast: {before:?} -> {after:?}");
    // it settles and freezes again
    run(&mut w, 8.0);
    assert_eq!(w.building(b).state[block as usize], ChunkState::Frozen, "settles back into rubble");
}

#[test]
fn weak_distant_blast_leaves_rubble_frozen() {
    let (mut w, b, block) = rubble_block();
    let before = w.chunk_world_com(b.0 as usize, block as usize);
    // edge of the blast radius, small impulse: well under the thaw threshold
    blast(&mut w, [-3.5, 0.4, 0.4], 50.0);
    run(&mut w, 0.5);
    assert_eq!(w.building(b).state[block as usize], ChunkState::Frozen);
    assert!((w.chunk_world_com(b.0 as usize, block as usize) - before).length() < 1e-4);
}
