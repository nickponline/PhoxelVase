//! `World::set_focus`: when debris has to go, the debris the player is looking at stays.
//!
//! Two identical walls 120 m apart are blasted in the same tick. With a tight piece budget the
//! debris near the focus is kept and the far wall's debris dropped, whichever wall the focus is
//! on (without a focus, the wall processed first simply takes the free slots). Far debris also
//! despawns sooner.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;
const GAP: f32 = 120.0;

/// Moving debris pieces near wall A and near wall B after `secs`.
fn blast_two_walls(cfg: WorldConfig, focus: Option<[f32; 3]>, secs: f32) -> (usize, usize) {
    let mut w = World::new(cfg);
    w.load_building_bld(wall(10.0, 6.0, 0.3, 0.5), Isometry::identity());
    w.load_building_bld(wall(10.0, 6.0, 0.3, 0.5), Isometry::new([GAP, 0.0, 0.0], 0.0));
    w.add_ground_plane(0.0);
    if let Some(f) = focus {
        w.set_focus(&[f]);
    }
    w.step(DT);
    for x in [5.0, GAP + 5.0] {
        w.explode(Explosion { center: [x, -0.5, 2.0], radius: 3.0, inner_radius: 1.5, damage: 5000.0, impulse: 30000.0 });
    }
    for _ in 0..(secs / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
    let (mut a, mut b) = (0, 0);
    for (_, cl) in w.clusters.iter() {
        let x = w.chunk_world_com(cl.building as usize, cl.chunks[0] as usize).x;
        if x < GAP / 2.0 {
            a += 1;
        } else {
            b += 1;
        }
    }
    (a, b)
}

const AT_A: [f32; 3] = [5.0, -15.0, 2.0];
const AT_B: [f32; 3] = [GAP + 5.0, -15.0, 2.0];

#[test]
fn budget_keeps_debris_near_the_focus() {
    let cfg = WorldConfig { max_dynamic_clusters: 30, ..Default::default() };
    let (a, b) = blast_two_walls(cfg.clone(), Some(AT_A), 0.5);
    assert!(a + b <= 30, "the budget holds: {a} + {b}");
    assert!(a > 2 * b, "focus on A: {a} pieces at A vs {b} at B");
    // the same with the focus on B, which without a focus loses every slot to A (blasted first)
    let (a, b) = blast_two_walls(cfg.clone(), Some(AT_B), 0.5);
    assert!(a + b <= 30, "the budget holds: {a} + {b}");
    assert!(b > 2 * a, "focus on B: {b} pieces at B vs {a} at A");
    let (a0, b0) = blast_two_walls(cfg, None, 0.5);
    assert!(a0 > b0, "(control) no focus: the wall processed first takes the slots, {a0} vs {b0}");
}

#[test]
fn far_debris_despawns_sooner() {
    // generous budget: only the lifetimes differ. After 4 s debris (10 s lifetime) still moving
    // near the focus is kept; far away it lives only 2.5 s
    let cfg = WorldConfig { max_dynamic_clusters: 10_000, freeze_debris: false, ..Default::default() };
    let (a, b) = blast_two_walls(cfg, Some(AT_A), 4.0);
    assert!(a > 0, "debris near the focus is still there");
    assert_eq!(b, 0, "far debris is gone after 4 s");
}
