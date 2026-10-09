//! `Event::Impact`: an event when a falling piece hits (and one per bounce, each a new contact),
//! none once it rests.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;

/// A 0.8 m concrete block bonded to the side of a pillar of 0.8 m cells, its bottom at `drop_z`
/// (a multiple of 0.8 m); destroying the pillar cell beside it drops it onto the ground.
fn drop_block(cfg: WorldConfig, drop_z: f32, secs: f32) -> (Vec<(f32, Event)>, World, u32) {
    let mut s = SynthBuilder::new("drop_block");
    let n = ((drop_z + 0.8) / 0.8).round() as usize;
    let first = s.boxes.len() as u32;
    s.add_grid([10.0, 0.0, 0.0], [10.8, 0.8, n as f32 * 0.8], [1, 1, n], 0, 0);
    let holder = first + n as u32 - 1;
    let block = s.add([9.2, 0.0, drop_z], [10.0, 0.8, drop_z + 0.8], 0, 0);
    let mut w = World::new(cfg);
    let b = w.load_building_bld(s.build(), Isometry::identity());
    w.add_ground_plane(0.0);
    w.step(DT);
    w.drain_events();
    w.damage_chunk(b, holder, 1e9);
    let mut events = vec![];
    for i in 0..(secs / DT) as usize {
        w.step(DT);
        let t = i as f32 * DT;
        events.extend(w.drain_events().into_iter().filter(|e| matches!(e, Event::Impact { .. })).map(|e| (t, e)));
    }
    (events, w, block)
}

#[test]
fn landing_then_bounces_then_silence() {
    // 4.8 m fall: ~9.7 m/s, then it tumbles and comes to rest
    let (events, w, block) = drop_block(WorldConfig::default(), 4.8, 5.0);
    assert!(!events.is_empty(), "the landing must be reported");
    let (t0, first) = &events[0];
    let Event::Impact { pos, normal, impulse, speed, material, .. } = first else { unreachable!() };
    assert!((pos[0] - 9.6).abs() < 1.0 && pos[2].abs() < 0.2, "landing contact on the ground under the block: {pos:?}");
    assert!(normal[2] > 0.9, "normal from the ground up to the block: {normal:?}");
    assert!((*speed - 9.7).abs() < 3.0, "speed ~9.7 m/s, got {speed:.2}");
    assert_eq!(*material, 0, "concrete");
    // each bounce / tumble is a new, weaker contact
    for (_, e) in &events[1..] {
        let Event::Impact { impulse: j, .. } = e else { unreachable!() };
        assert!(j < impulse, "bounces are weaker than the landing: {j} vs {impulse}");
    }
    // resting: silence for the last 2 s
    let last = events.last().unwrap().0;
    assert!(last < 3.0, "no impacts while it rests (last at {last:.2} s, landing at {t0:.2} s)");
    assert!(w.chunk_world_com(0, block as usize).z < 0.5, "it came to rest on the ground");
}

#[test]
fn no_events_when_disabled() {
    let cfg = WorldConfig { impact_event_min: 0.0, ..Default::default() };
    let (events, _, _) = drop_block(cfg, 4.8, 2.0);
    assert!(events.is_empty(), "{events:?}");
}
