//! Blast waves blow out windows well beyond the blast radius (`WorldConfig::glass_blast_range`),
//! but not windows shielded by something solid.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;
const GLASS: u16 = 4;

/// Blast of radius 2 m at the origin. Panes (1 x 0.02 x 1 m on short anchored posts): one in the
/// open 4 m away, one 4 m away behind a concrete wall, one in the open 8 m away (beyond 3 radii).
/// Returns which of (open, shielded, far) shattered.
fn blast(cfg: WorldConfig) -> (bool, bool, bool) {
    let mut s = SynthBuilder::new("panes");
    let pane = |s: &mut SynthBuilder, x: f32, y: f32| {
        s.add([x, y, 0.0], [x + 1.0, y + 0.2, 1.0], 0, 0); // post
        s.add([x, y + 0.09, 1.0], [x + 1.0, y + 0.11, 2.0], GLASS, 0)
    };
    let open = pane(&mut s, -0.5, 4.0);
    let shielded = pane(&mut s, -0.5, -4.2);
    let far = pane(&mut s, 7.5, 0.0);
    // concrete wall between the blast and the shielded pane
    s.add([-3.0, -2.5, 0.0], [3.0, -2.2, 4.0], 0, 0);
    let mut w = World::new(cfg);
    let b = w.load_building_bld(s.build(), Isometry::identity());
    w.add_ground_plane(0.0);
    w.step(DT);
    w.explode(Explosion { center: [0.0, 0.0, 1.5], radius: 2.0, inner_radius: 0.5, damage: 50.0, impulse: 0.0 });
    for _ in 0..3 {
        w.step(DT);
    }
    let gone = |c: u32| w.building(b).state[c as usize] == ChunkState::Gone;
    (gone(open), gone(shielded), gone(far))
}

#[test]
fn blast_wave_breaks_exposed_windows_only() {
    let (open, shielded, far) = blast(WorldConfig::default());
    assert!(open, "a pane in the open at 2 blast radii shatters");
    assert!(!shielded, "a pane behind a concrete wall survives");
    assert!(!far, "a pane beyond 3 blast radii survives");
}

#[test]
fn without_blast_wave_only_the_radius_counts() {
    let (open, _, _) = blast(WorldConfig { glass_blast_range: 0.0, ..Default::default() });
    assert!(!open, "(control) without the blast wave the pane at 2 radii survives");
}

#[test]
fn office_facade_windows_blow_out() {
    // a G-sized blast (6 m) in front of office_1: count panes that shatter, with and without
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../assets/buildings/office_1/building.bld");
    if !std::path::Path::new(p).exists() {
        return;
    }
    let count = |cfg: WorldConfig| {
        let mut w = World::new(cfg);
        w.load_building(p, Isometry::identity()).unwrap();
        w.add_ground_plane(0.0);
        w.step(DT);
        w.explode(Explosion { center: [12.0, -3.0, 4.0], radius: 6.0, inner_radius: 2.5, damage: 5000.0, impulse: 30000.0 });
        let mut n = 0;
        for _ in 0..10 {
            w.step(DT);
            n += w.drain_events().iter().filter(|e| matches!(e, Event::ChunkShattered { building, chunk, .. } if w.building(*building).glass(*chunk as usize))).count();
        }
        n
    };
    let with = count(WorldConfig::default());
    let without = count(WorldConfig { glass_blast_range: 0.0, ..Default::default() });
    eprintln!("office: {with} panes shattered with the blast wave, {without} without");
    assert!(with >= 2 * without.max(1), "the blast wave should blow out many more windows: {with} vs {without}");
}
