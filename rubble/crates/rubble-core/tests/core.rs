use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;

fn cfg() -> WorldConfig {
    WorldConfig::default()
}
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/two_box.bld");

fn run(w: &mut World, steps: usize) -> Vec<Event> {
    let mut ev = vec![];
    for _ in 0..steps {
        w.step(DT);
        ev.extend(w.drain_events());
    }
    ev
}

fn count<F: Fn(&Event) -> bool>(ev: &[Event], f: F) -> usize {
    ev.iter().filter(|e| f(e)).count()
}

#[test]
fn two_box_loads_with_4_colliders() {
    let mut w = World::new(cfg());
    let b = w.load_building(FIXTURE, Isometry::identity()).unwrap();
    assert_eq!(w.stats().rapier_colliders, 4);
    assert_eq!(w.building(b).n_chunks(), 4);
    let ev = run(&mut w, 30);
    assert!(ev.is_empty(), "intact building must not change: {ev:?}");
    assert_eq!(w.stats().static_chunks, 4);
}

#[test]
fn two_box_detach_fall_land() {
    let mut w = World::new(cfg());
    let b = w.load_building(FIXTURE, Isometry::new([5.0, 0.0, 0.0], 0.3)).unwrap();
    w.add_ground_plane(0.0);
    w.damage_chunk(b, 0, 1e9);
    w.damage_chunk(b, 1, 1e9);
    let ev = run(&mut w, 1);
    assert_eq!(count(&ev, |e| matches!(e, Event::ChunkDestroyed { .. })), 2);
    let warn = ev.iter().find_map(|e| match e {
        Event::CollapseWarning { chunks, delay, .. } => Some((chunks.clone(), *delay)),
        _ => None,
    });
    let (mut chunks, delay) = warn.expect("collapse warning");
    chunks.sort();
    assert_eq!(chunks, vec![2, 3]);
    assert!(delay >= 0.3 && delay <= 1.5);
    // still static during the delay
    assert!(w.cluster_of(b, 2).is_none());
    let ev = run(&mut w, (delay / DT) as usize + 2);
    let det: Vec<_> = ev
        .iter()
        .filter_map(|e| match e {
            Event::ClusterDetached { chunks, cluster, .. } if chunks.len() == 2 => Some(*cluster),
            _ => None,
        })
        .collect();
    assert_eq!(det.len(), 1, "{ev:?}");
    let cid = det[0];
    assert_eq!(w.cluster_of(b, 2), Some(cid));
    assert_eq!(w.cluster_of(b, 3), Some(cid));
    // falls and lands on the ground plane (bottom face from z=1 to ~0)
    let mut ev = run(&mut w, 60);
    let com = w.chunk_world_com(b.0 as usize, 2);
    assert!(com.z < 0.6 && com.z > 0.3, "chunk com z = {}", com.z);
    assert_eq!(count(&ev, |e| matches!(e, Event::ClusterSplit { .. })), 0, "a 1 m drop must not split");
    // eventually freezes into rubble
    ev.extend(run(&mut w, 240));
    assert!(count(&ev, |e| matches!(e, Event::ClusterFrozen { .. })) >= 1, "{ev:?}");
    let st = w.building_state(b);
    assert_eq!(st.state[2], 3);
    assert_eq!(st.state[3], 3);
    let z = w.chunk_world_com(b.0 as usize, 3).z;
    assert!(z > 0.3 && z < 0.6);
}

#[test]
fn bullet_destroys_one_chunk_no_detach() {
    let mut w = World::new(cfg());
    // 10 m x 6 m wall of 0.5 m cells, 0.3 m thick
    let b = w.load_building_bld(wall(10.0, 6.0, 0.3, 0.5), Isometry::identity());
    w.add_ground_plane(0.0);
    w.step(DT);
    let n = w.building(b).n_chunks();
    let custom = Weapon::Custom(WeaponParams { damage: 1e6, splash_r: 0.0, splash_frac: 0.0, penetration: 0.0, ..Weapon::Ar.params() });
    w.fire(Projectile::hitscan([5.1, -20.0, 3.1], [0.0, 1.0, 0.0], custom));
    let ev = run(&mut w, 120);
    assert_eq!(count(&ev, |e| matches!(e, Event::ChunkDestroyed { .. })), 1, "{ev:?}");
    assert_eq!(count(&ev, |e| matches!(e, Event::CollapseWarning { .. })), 0, "{ev:?}");
    let st = w.stats();
    assert_eq!(st.static_chunks, n - 1);
    // the debris chunk is either a cluster or shattered; nothing else moved
    assert!(st.chunks_in_flight + st.frozen_chunks + st.gone_chunks == 1);
}

#[test]
fn beam_radius_widens_the_cut() {
    // a beam through (5.1, *, 3.1) on a wall of 0.5 m cells: the ray is inside one cell, a 0.3 m
    // radius also reaches the cells left of x=5.0 and below z=3.0 (and the corner at 0.14 m)
    for (radius, expect) in [(0.0, 1), (0.3, 4)] {
        let mut w = World::new(cfg());
        w.load_building_bld(wall(10.0, 6.0, 0.3, 0.5), Isometry::identity());
        w.add_ground_plane(0.0);
        w.step(DT);
        w.beam([5.1, -20.0, 3.1], [0.0, 1.0, 0.0], 100.0, radius, 1e9, 0.0);
        let ev = run(&mut w, 2);
        assert_eq!(count(&ev, |e| matches!(e, Event::ChunkDestroyed { .. })), expect, "radius {radius}: {ev:?}");
    }
}

#[test]
fn ar_bullets_chip_but_dont_destroy_quickly() {
    let mut w = World::new(cfg());
    let b = w.load_building_bld(wall(10.0, 6.0, 0.3, 0.5), Isometry::identity());
    w.step(DT);
    w.fire(Projectile::hitscan([5.1, -20.0, 3.1], [0.0, 1.0, 0.0], Weapon::Ar));
    run(&mut w, 2);
    let st = w.building_state(b);
    let damaged = st.hp.iter().zip(&w.building(b).bld.chunks).filter(|(h, c)| **h < c.hp).count();
    assert!(damaged >= 1);
    assert_eq!(w.stats().static_chunks, w.building(b).n_chunks());
}

#[test]
fn ballistic_projectile_hits() {
    let mut w = World::new(cfg());
    let b = w.load_building_bld(wall(10.0, 6.0, 0.3, 0.5), Isometry::identity());
    w.step(DT);
    w.fire(Projectile::ballistic([5.1, -10.0, 3.5], [0.0, 1.0, 0.0], Weapon::Launcher));
    run(&mut w, 60);
    let st = w.building_state(b);
    let damaged = st.hp.iter().zip(&w.building(b).bld.chunks).filter(|(h, c)| **h < c.hp).count();
    assert!(damaged >= 1);
}

#[test]
fn explosion_knocks_out_tower_base_collapses() {
    let mut w = World::new(cfg());
    let b = w.load_building_bld(tower(3, 6.0, false), Isometry::identity());
    w.add_ground_plane(0.0);
    let ev = run(&mut w, 60);
    assert!(ev.is_empty(), "tower must stand");
    // explosion in the middle of the ground floor, large enough to reach all 4 columns
    w.explode(Explosion { center: [3.0, 3.0, 1.0], radius: 5.0, inner_radius: 4.5, damage: 1e5, impulse: 5000.0 });
    let ev = run(&mut w, 200);
    let big = ev.iter().any(|e| matches!(e, Event::ClusterDetached { chunks, .. } if chunks.len() > 20));
    assert!(big, "tower upper part should detach as a big cluster");
    let _ = b;
}

#[test]
fn single_explosion_on_top_floor_does_not_collapse_tower() {
    let mut w = World::new(cfg());
    let _b = w.load_building_bld(tower(3, 6.0, true), Isometry::identity());
    w.add_ground_plane(0.0);
    run(&mut w, 5);
    w.explode(Explosion { center: [0.3, 0.3, 11.0], radius: 1.0, inner_radius: 0.2, damage: 3000.0, impulse: 1000.0 });
    let ev = run(&mut w, 200);
    assert!(!ev.iter().any(|e| matches!(e, Event::ClusterDetached { chunks, .. } if chunks.len() > 20)));
}

/// Needs the real stress solver: 3 of 5 ground-floor columns removed -> remaining
/// supports overloaded -> structure fails even though still connected.
#[test]
fn stress_collapse_when_columns_removed() {
    let mut w = World::new(cfg());
    let bld = tower(4, 9.0, true);
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    let ev = run(&mut w, 60);
    assert!(ev.is_empty(), "intact tower must stand: {ev:?}");
    // destroy ground-floor column pieces at 3 corners (columns are 3 chunks each, z in 0.3..3.3)
    let cols: Vec<u32> = {
        let bd = w.building(b);
        (0..bd.n_chunks() as u32)
            .filter(|&c| {
                let ch = &bd.bld.chunks[c as usize];
                let corner = (ch.com[0] < 1.0 || ch.com[0] > 8.0) && (ch.com[1] < 1.0 || ch.com[1] > 8.0);
                let lower = ch.com[2] > 0.3 && ch.com[2] < 1.4;
                corner && lower && !(ch.com[0] > 8.0 && ch.com[1] > 8.0)
            })
            .collect()
    };
    assert_eq!(cols.len(), 3);
    for c in cols {
        w.damage_chunk(b, c, 1e9);
    }
    let ev = run(&mut w, 600);
    let big = ev.iter().any(|e| matches!(e, Event::ClusterDetached { chunks, .. } if chunks.len() > 20));
    assert!(big, "overloaded tower should collapse via stress");
}

#[test]
fn cluster_splits_on_impact() {
    let mut w = World::new(cfg());
    // 12 m high column with an 8-cube cantilever at the top
    let b = w.load_building_bld(cantilever(12, 8), Isometry::identity());
    w.add_ground_plane(0.0);
    let mut cfg_ok = true;
    run(&mut w, 2);
    // cut the beam off at the column (chunk 11 is the column top, the beam starts at 12)
    let bd = w.building(b);
    let e = bd.bld.edges.iter().position(|e| (e.a, e.b) == (11, 12)).unwrap();
    let _ = &mut cfg_ok;
    w.damage_chunk(b, 11, 1e9);
    let _ = e;
    let ev = run(&mut w, 300);
    let detached = count(&ev, |e| matches!(e, Event::ClusterDetached { chunks, .. } if chunks.len() == 8));
    assert_eq!(detached, 1, "{ev:?}");
    assert!(count(&ev, |e| matches!(e, Event::ClusterSplit { .. })) >= 1, "beam should split on landing");
}

#[test]
fn freezing_rubble_is_static_and_destructible() {
    let mut w = World::new(cfg());
    let b = w.load_building_bld(two_box(), Isometry::identity());
    w.add_ground_plane(0.0);
    w.damage_chunk(b, 0, 1e9);
    w.damage_chunk(b, 1, 1e9);
    let ev = run(&mut w, 400);
    assert!(count(&ev, |e| matches!(e, Event::ClusterFrozen { .. })) == 1);
    assert_eq!(w.stats().clusters, 0);
    assert_eq!(w.stats().frozen_chunks, 2);
    // frozen rubble stays put and is still shootable
    let z0 = w.chunk_world_com(b.0 as usize, 2).z;
    run(&mut w, 60);
    assert!((w.chunk_world_com(b.0 as usize, 2).z - z0).abs() < 1e-4);
    let p = w.chunk_world_com(b.0 as usize, 2);
    let big = Weapon::Custom(WeaponParams { damage: 1e6, splash_r: 0.0, penetration: 0.0, ..Weapon::Ar.params() });
    w.fire(Projectile::hitscan([p.x, p.y - 10.0, p.z], [0.0, 1.0, 0.0], big));
    let ev = run(&mut w, 2);
    assert!(ev.iter().any(|e| matches!(e, Event::ChunkDestroyed { chunk: 2, .. })), "{ev:?}");
}

#[test]
fn small_debris_despawns_after_ttl() {
    let mut cfg = cfg();
    cfg.debris_ttl = 1.0;
    cfg.freeze_debris = false; // exercise the TTL path (resting debris would otherwise freeze)
    let mut w = World::new(cfg);
    let b = w.load_building_bld(wall(4.0, 4.0, 0.3, 0.5), Isometry::identity());
    w.add_ground_plane(0.0);
    w.damage_chunk(b, 20, 1e9);
    let ev = run(&mut w, 2);
    assert_eq!(count(&ev, |e| matches!(e, Event::ClusterDetached { .. })), 1, "debris spawned");
    let ev = run(&mut w, 90);
    assert_eq!(count(&ev, |e| matches!(e, Event::ClusterDespawned { .. })), 1);
    assert_eq!(w.stats().clusters, 0);
}

#[test]
fn budget_caps_dynamic_clusters() {
    let mut cfg = cfg();
    cfg.max_dynamic_clusters = 5;
    let mut w = World::new(cfg);
    let b = w.load_building_bld(wall(10.0, 6.0, 0.3, 0.5), Isometry::identity());
    w.add_ground_plane(0.0);
    for c in 100..140 {
        w.damage_chunk(b, c, 1e9);
    }
    for _ in 0..120 {
        w.step(DT);
        assert!(w.stats().clusters <= 5);
    }
}

#[test]
fn glass_shatters_on_any_hit_and_indestructible_survives() {
    let mut s = SynthBuilder::new("glass");
    s.add([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0, rubble_format::F_INDESTRUCTIBLE);
    s.add([1.0, 0.0, 0.0], [2.0, 0.05, 1.0], 4, 0);
    let mut w = World::new(cfg());
    let b = w.load_building_bld(s.build(), Isometry::identity());
    w.damage_chunk(b, 1, 0.001);
    w.damage_chunk(b, 0, 1e12);
    let ev = run(&mut w, 2);
    assert!(ev.iter().any(|e| matches!(e, Event::ChunkShattered { chunk: 1, .. })), "{ev:?}");
    assert!(!ev.iter().any(|e| matches!(e, Event::ChunkDestroyed { chunk: 0, .. })));
}

#[test]
fn chunk_transforms() {
    let mut w = World::new(cfg());
    let b = w.load_building_bld(two_box(), Isometry::new([1.0, 2.0, 0.0], 0.0));
    let xf = w.chunk_world_transforms(b);
    assert_eq!(xf.len(), 4);
    assert_eq!(xf[0][3], 1.0); // row-major translation x
    assert_eq!(xf[0][7], 2.0);
    w.step(DT);
    assert_eq!(w.building_state(b).alive[0] & 0xf, 0xf);
}

fn intact_stands(bld: rubble_format::Bld, cfg: WorldConfig) -> Result<(), String> {
    let mut w = World::new(cfg);
    w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    let ev = run(&mut w, 90);
    let st = w.building_state(BuildingId(0));
    let mu = st.utilization.iter().cloned().fold(0.0f32, f32::max);
    if ev.is_empty() { Ok(()) } else { Err(format!("max util {mu}, {} events, first {:?}", ev.len(), &ev[..1])) }
}

#[test]
fn intact_synthetics_stand_without_bending() {
    let mut cfg = cfg();
    cfg.stress.bending = false;
    for (name, bld) in [("tower", tower(4, 9.0, true)), ("grid", grid_block(10, 10, 6, 1.0)), ("wall", wall(10.0, 6.0, 0.3, 0.5)), ("cantilever", cantilever(6, 3))] {
        intact_stands(bld, cfg.clone()).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn intact_synthetics_stand_with_bending() {
    for (name, bld) in [("tower", tower(4, 9.0, true)), ("grid", grid_block(10, 10, 6, 1.0)), ("wall", wall(10.0, 6.0, 0.3, 0.5)), ("cantilever", cantilever(6, 3))] {
        intact_stands(bld, WorldConfig::default()).unwrap_or_else(|e| panic!("{name}: {e}"));
    }
}

#[test]
fn cosmetic_attached_detaches_with_neighbours_and_never_bridges() {
    use rubble_format::F_COSMETIC_ATTACHED;
    let mut s = SynthBuilder::new("cosmetic");
    // two 3-high columns at x=0 and x=2, bridged at the top by a cosmetic chunk at x=1
    for x in [0.0f32, 2.0] {
        s.add_grid([x, 0.0, 0.0], [x + 1.0, 1.0, 3.0], [1, 1, 3], 0, 0);
    }
    let bridge = s.add([1.0, 0.0, 2.0], [2.0, 1.0, 3.0], 2, F_COSMETIC_ATTACHED);
    let mut w = World::new(cfg());
    let b = w.load_building_bld(s.build(), Isometry::identity());
    w.add_ground_plane(0.0);
    assert!(run(&mut w, 10).is_empty());
    // knock out the base of the left column (chunk 0): its top 2 chunks must fall even though the
    // cosmetic bridge still touches the right column
    w.damage_chunk(b, 0, 1e9);
    let ev = run(&mut w, 120);
    let det: Vec<Vec<u32>> = ev
        .iter()
        .filter_map(|e| match e {
            Event::ClusterDetached { chunks, .. } if chunks.len() > 1 => Some(chunks.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(det.len(), 1, "{ev:?}");
    let mut c = det[0].clone();
    c.sort();
    assert_eq!(c, vec![1, 2], "cosmetic bridge stays with the grounded column");
    assert_eq!(w.building_state(b).state[bridge as usize], 0);
    // now take out the right column's base: the bridge detaches with it
    w.damage_chunk(b, 3, 1e9);
    let ev = run(&mut w, 120);
    let det: Vec<Vec<u32>> = ev
        .iter()
        .filter_map(|e| match e {
            Event::ClusterDetached { chunks, .. } if chunks.len() > 1 => Some(chunks.clone()),
            _ => None,
        })
        .collect();
    let mut c = det.concat();
    c.sort();
    assert_eq!(c, vec![4, 5, bridge]);
}
