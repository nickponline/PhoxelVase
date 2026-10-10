//! Terrain around sunk foundations: ground level outside the pits, open over them.
use rubble_core::physics::PhysicsBackend;
use rubble_core::testutil::tower;
use rubble_core::*;

/// A tower whose foundation pit is the 2×3 m rectangle at the origin, 4 m deep.
fn sunk_tower() -> rubble_format::Bld {
    let mut b = tower(2, 3.0, true);
    b.meta["foundation"] = serde_json::json!({"rects": [[0.0, 0.0, 2.0, 3.0]], "top": 0.0, "bottom": -4.0});
    b
}

fn ground_below(w: &World, x: f32, y: f32) -> Option<f32> {
    let from = Vec3::new(x, y, 50.0);
    w.phys.cast_ray(from, Vec3::new(0.0, 0.0, -1.0), 100.0, None).map(|h| from.z - h.toi)
}

#[test]
fn terrain_leaves_the_pits_open() {
    let mut w = World::new(WorldConfig::default());
    w.load_building_bld(sunk_tower(), Isometry::new([10.0, 0.0, 0.0], 0.0));
    w.load_building_bld(sunk_tower(), Isometry::new([-10.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2));
    assert_eq!(w.foundation_ground_z(), 0.0); // the test towers carry no ground_z
    assert!(w.add_terrain(50.0));
    assert_eq!(w.terrain_z, (-4.0, 0.0));
    // cells tile the square minus both pits
    let area: f32 = w.terrain.iter().map(|(a, b)| (b[0] - a[0]) * (b[1] - a[1])).sum();
    assert!((area - (100.0 * 100.0 - 2.0 * 6.0)).abs() < 1e-2, "{area}");
    w.phys.sync_queries();
    // ground level well away from the buildings
    assert!((ground_below(&w, 30.0, 30.0).unwrap() - 0.0).abs() < 1e-3);
    // no cell reaches into a pit; the second pit is yawed 90°: x in [-13, -10], y in [0, 2]
    for (lo, hi) in [([10.0f32, 0.0f32], [12.0f32, 3.0f32]), ([-13.0, 0.0], [-10.0, 2.0])] {
        for (a, b) in &w.terrain {
            let overlap = a[0] < hi[0] - 1e-3 && b[0] > lo[0] + 1e-3 && a[1] < hi[1] - 1e-3 && b[1] > lo[1] + 1e-3;
            assert!(!overlap, "terrain cell {a:?}..{b:?} reaches into the pit {lo:?}..{hi:?}");
        }
    }
    // right beside the pit: ground level
    assert!((ground_below(&w, 9.5, 1.5).unwrap() - 0.0).abs() < 1e-3);
}

#[test]
fn no_terrain_without_foundations() {
    let mut w = World::new(WorldConfig::default());
    w.load_building_bld(tower(2, 3.0, true), Isometry::identity());
    assert!(!w.add_terrain(50.0));
    assert!(w.terrain.is_empty());
}
