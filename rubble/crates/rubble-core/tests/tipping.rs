//! Regression: a structure cut so that it stands only on joints off to one side must tip
//! over, not hang in the air. The local stress model alone misses this when the remaining
//! joints lie inside one member (a storey left standing on one wall strip), see
//! `Building::tipping_edges`.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;

/// 6×6×6 m box of four 0.5 m walls (one element each, 1 m cells) under a 0.3 m roof slab.
/// Returns (bld, cells of each wall in the 2..3 m band [front, back, left, right], roof cells).
fn walled_box() -> (rubble_format::Bld, [Vec<u32>; 4], Vec<u32>) {
    let mut s = SynthBuilder::new("walled_box");
    let mut band: [Vec<u32>; 4] = Default::default();
    let walls: [([f32; 3], [f32; 3], [usize; 3]); 4] = [
        ([0.0, 0.0, 0.0], [6.0, 0.5, 6.0], [6, 1, 6]),
        ([0.0, 5.5, 0.0], [6.0, 6.0, 6.0], [6, 1, 6]),
        ([0.0, 0.5, 0.0], [0.5, 5.5, 6.0], [1, 5, 6]),
        ([5.5, 0.5, 0.0], [6.0, 5.5, 6.0], [1, 5, 6]),
    ];
    for (w, (lo, hi, n)) in walls.iter().enumerate() {
        s.begin_element(0);
        let first = s.boxes.len() as u32;
        s.add_grid(*lo, *hi, *n, 0, 0);
        let last = s.boxes.len() as u32;
        band[w] = (first..last).filter(|&c| (s.boxes[c as usize].min[2] - 2.0).abs() < 1e-3).collect();
    }
    s.begin_element(2);
    let r0 = s.boxes.len() as u32;
    s.add_grid([0.0, 0.0, 6.0], [6.0, 6.0, 6.3], [6, 6, 1], 0, 0);
    let roof = (r0..s.boxes.len() as u32).collect();
    s.end_element();
    (s.build(), band, roof)
}

fn roof_z(cfg: WorldConfig, cut: &[usize]) -> f32 {
    let (bld, band, roof) = walled_box();
    let mut w = World::new(cfg);
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    for _ in 0..30 {
        w.step(DT);
    }
    for &i in cut {
        for &c in &band[i] {
            w.damage_chunk(b, c, 1e9);
        }
    }
    for _ in 0..(8.0 / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
    roof.iter().map(|&c| w.chunk_world_com(b.0 as usize, c as usize).z).sum::<f32>() / roof.len() as f32
}

#[test]
fn storey_left_on_one_wall_tips_over() {
    // keep only the front wall in the cut band: the upper box's centre is 2.75 m behind it
    let z = roof_z(WorldConfig::default(), &[1, 2, 3]);
    assert!(z < 4.0, "upper storey must come down, roof com z = {z}");
}

#[test]
fn storey_on_two_opposite_walls_stands() {
    let z = roof_z(WorldConfig::default(), &[2, 3]);
    assert!((z - 6.15).abs() < 0.05, "balanced upper storey must stand, roof com z = {z}");
}

#[test]
fn without_tipping_check_the_storey_hangs() {
    // documents the bug: the stress model alone keeps it up on the single wall strip
    let mut cfg = WorldConfig::default();
    cfg.tipping = false;
    let z = roof_z(cfg, &[1, 2, 3]);
    assert!(z > 6.0, "expected the old behaviour (hanging), roof com z = {z}");
}

#[test]
fn hull_distance() {
    let mut sq = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]];
    assert_eq!(rubble_core::building::dist_outside_hull(&mut sq, [1.0, 1.5]), 0.0);
    assert!((rubble_core::building::dist_outside_hull(&mut sq, [3.0, 1.0]) - 1.0).abs() < 1e-9);
    let mut line = vec![[0.0, 0.0], [4.0, 0.0]];
    assert!((rubble_core::building::dist_outside_hull(&mut line, [2.0, 3.0]) - 3.0).abs() < 1e-9);
}

/// A settled (frozen) slab resting on four corner columns: shoot three of them away and it must
/// tip off the last one, not hang from its corner.
#[test]
fn frozen_slab_left_on_one_corner_comes_down() {
    let mut s = SynthBuilder::new("slab_on_columns");
    let mut cols = vec![];
    for (x, y) in [(0.0, 0.0), (5.0, 0.0), (0.0, 5.0), (5.0, 5.0)] {
        cols.push(s.add([x, y, 0.0], [x + 1.0, y + 1.0, 3.0], 0, 0));
    }
    // 1 cm above the columns, no bonds: frozen rubble at load, settles onto them
    let first = s.boxes.len() as u32;
    s.add_grid([0.0, 0.0, 3.01], [6.0, 6.0, 3.31], [6, 6, 1], 0, 0);
    let slab: Vec<u32> = (first..s.boxes.len() as u32).collect();
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building_bld(s.build(), Isometry::identity());
    w.add_ground_plane(0.0);
    let run = |w: &mut World, secs: f32| {
        for _ in 0..(secs / DT) as usize {
            w.step(DT);
            w.drain_events();
        }
    };
    let slab_z = |w: &World| slab.iter().map(|&c| w.chunk_world_com(b.0 as usize, c as usize).z).sum::<f32>() / slab.len() as f32;
    run(&mut w, 5.0);
    assert!((slab_z(&w) - 3.15).abs() < 0.05, "slab should rest on the columns, z = {}", slab_z(&w));
    assert!(slab.iter().all(|&c| w.building(b).state[c as usize] == ChunkState::Frozen), "slab should be frozen");
    for &c in &cols[1..] {
        w.damage_chunk(b, c, 1e9);
    }
    run(&mut w, 8.0);
    let z = slab_z(&w);
    assert!(z < 2.0, "slab must tip off its last corner column, mean com z = {z}");
}

/// Backstop: joints cut without telling the incremental connectivity search (as a missed code
/// path would) are still caught by the full ground check: the cut-off storey stops being part
/// of the static building (no joints hold it any more; it now just rests on the walls below).
#[test]
fn ground_check_catches_what_incremental_search_missed() {
    let (bld, band, roof) = walled_box();
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    for _ in 0..10 {
        w.step(DT);
    }
    // sever every joint between the 2..3 m band and the wall below, behind the engine's back
    let bd = &mut w.buildings[b.0 as usize];
    let below: std::collections::HashSet<u32> = band.iter().flatten().copied().collect();
    for (e, ed) in bd.bld.edges.iter().enumerate() {
        let (za, zb) = (bd.bld.chunks[ed.a as usize].com[2], bd.bld.chunks[ed.b as usize].com[2]);
        if (below.contains(&ed.a) && zb < za) || (below.contains(&ed.b) && za < zb) {
            bd.edge_alive[e] = false;
        }
    }
    bd.topo_changed = true;
    for _ in 0..(8.0 / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
    assert!(w.stats().ground_check_catches > 0, "the ground check should have fired");
    let bd = w.building(b);
    let still_static = roof.iter().filter(|&&c| bd.state[c as usize] == ChunkState::Static).count();
    assert_eq!(still_static, 0, "the cut-off storey must be released from the static building");
}
