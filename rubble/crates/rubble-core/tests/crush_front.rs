//! Progressive collapse ("crush front", `WorldConfig::crush_band`).
//!
//! A tall section that falls a storey carries far more energy than the storey below it can
//! absorb, so it must crush its way down floor by floor (the old model let a tall section land
//! rigid and stand on the stubs: building4's top 47 floors did). Checked here on a stiff
//! 30-storey synthetic tube: the front runs all the way down, it never adds energy (nothing
//! falls faster than free fall), an energy gate stops it when a storey can take the hit, and a
//! short section is not affected at all.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;
const STOREY: f32 = 3.3;

/// A stiff tube tower: `floors` storeys of 3.3 m, 6 x 6 m, four 0.3 m perimeter walls in 1.5 m
/// panels per storey and a 0.3 m slab on top of each.
fn tube_tower(floors: usize) -> rubble_format::Bld {
    let mut s = SynthBuilder::new("tube_tower");
    let (side, t, storey, slab) = (6.0f32, 0.3f32, 3.0f32, 0.3f32);
    s.add_grid([0.0, 0.0, 0.0], [side, side, slab], [4, 4, 1], 0, rubble_format::F_INDESTRUCTIBLE);
    for f in 0..floors {
        let z0 = slab + f as f32 * (storey + slab);
        let z1 = z0 + storey;
        s.add_grid([0.0, 0.0, z0], [side, t, z1], [4, 1, 2], 0, 0);
        s.add_grid([0.0, side - t, z0], [side, side, z1], [4, 1, 2], 0, 0);
        s.add_grid([0.0, t, z0], [t, side - t, z1], [1, 4, 2], 0, 0);
        s.add_grid([side - t, t, z0], [side, side - t, z1], [1, 4, 2], 0, 0);
        s.add_grid([0.0, 0.0, z1], [side, side, z1 + slab], [4, 4, 1], 0, 0);
    }
    s.build()
}

fn run(w: &mut World, secs: f32) {
    for _ in 0..(secs / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
}

/// Destroy every chunk with its centre between `z0` and `z1`; returns the chunks above `z1`.
fn cut(w: &mut World, b: BuildingId, z0: f32, z1: f32) -> Vec<usize> {
    let n = w.building(b).n_chunks();
    let mut above = vec![];
    for c in 0..n {
        let z = w.chunk_world_com(b.0 as usize, c).z;
        if z > z0 && z < z1 {
            w.damage_chunk(b, c as u32, 1e9);
        } else if z >= z1 {
            above.push(c);
        }
    }
    above
}

/// (chunks of `above` that were crushed (gone), highest remaining one's centre z)
fn outcome(w: &World, b: BuildingId, above: &[usize]) -> (usize, f32) {
    let bd = w.building(b);
    let gone = above.iter().filter(|&&c| bd.state[c] == ChunkState::Gone).count();
    let top = above
        .iter()
        .filter(|&&c| bd.state[c] != ChunkState::Gone)
        .map(|&c| w.chunk_world_com(b.0 as usize, c).z)
        .fold(0.0f32, f32::max);
    (gone, top)
}

/// The 30-storey tube with the walls of storey `cut` destroyed.
fn cut_tube(cfg: WorldConfig, cut_storey: usize) -> (World, BuildingId, Vec<usize>) {
    let mut w = World::new(cfg);
    let b = w.load_building_bld(tube_tower(30), Isometry::identity());
    w.add_ground_plane(0.0);
    w.step(DT);
    let z0 = 0.3 + cut_storey as f32 * STOREY;
    let above = cut(&mut w, b, z0, z0 + 3.0);
    (w, b, above)
}

#[test]
fn tall_section_crushes_down_without_gaining_energy() {
    let (mut w, b, above) = cut_tube(WorldConfig::default(), 2);
    // the roof: the highest chunk; it rides on top of the front all the way down
    let roof = *above.iter().max_by(|&&a, &&c| w.chunk_world_com(0, a).z.total_cmp(&w.chunk_world_com(0, c).z)).unwrap();
    let z_start = w.chunk_world_com(0, roof).z;
    let mut z_prev = z_start;
    for i in 1..=(8.0 / DT) as usize {
        w.step(DT);
        w.drain_events();
        if w.building(b).state[roof] == ChunkState::Gone {
            break;
        }
        let z = w.chunk_world_com(0, roof).z;
        // energy: never faster than free fall since the cut (plus a step of slack)
        let t = i as f32 * DT;
        let v = (z_prev - z) / DT;
        assert!(v <= 9.81 * t + 1.0, "roof falling at {v:.1} m/s after {t:.2} s: faster than free fall");
        z_prev = z;
    }
    let (gone, top) = outcome(&w, b, &above);
    assert!(gone > above.len() * 3 / 4, "the front should run through the tower: {gone} of {} crushed", above.len());
    assert!(top < 15.0, "the tower should have come down, highest piece at {top:.1} m (roof was {z_start:.1} m)");
}

#[test]
fn crush_needs_the_energy() {
    // a storey that takes far more energy to crush than the falling section has stops the front
    // at once: the outcome is exactly that of the old model
    let after = |cfg: WorldConfig| {
        let (mut w, b, above) = cut_tube(cfg, 2);
        run(&mut w, 8.0);
        outcome(&w, b, &above)
    };
    let gated = after(WorldConfig { crush_distance: 1000.0, ..Default::default() });
    let off = after(WorldConfig { crush_band: 0.0, ..Default::default() });
    assert_eq!(gated, off, "(crushed, highest) with an impossible crush energy vs without the crush front");
}

#[test]
fn short_section_is_not_crushed() {
    // synthetic 20-storey tower, columns of storey 18 destroyed: the ~6 m left above is shorter
    // than `crush_min_extent`, so the outcome is exactly the old one
    let run_cut = |cfg: WorldConfig| {
        let mut w = World::new(cfg);
        let b = w.load_building_bld(tower(20, 9.0, true), Isometry::identity());
        w.add_ground_plane(0.0);
        w.step(DT);
        let z0 = 0.3 + 18.0 * STOREY;
        let above = cut(&mut w, b, z0, z0 + 3.0);
        run(&mut w, 6.0);
        outcome(&w, b, &above)
    };
    let new = run_cut(WorldConfig::default());
    let old = run_cut(WorldConfig { crush_band: 0.0, ..Default::default() });
    assert_eq!(new, old, "(crushed, highest) with and without the crush front");
}
