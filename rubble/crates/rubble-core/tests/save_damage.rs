//! Persistent damage: `World::save_damage` / `World::restore_damage`.
//!
//! office_1 is blasted and left to settle; its damage is saved, sent through JSON and restored
//! onto a freshly loaded office in a new world. The restored ruin must match the original chunk
//! for chunk (standing, destroyed, rubble at the same place), stay put when simulated, follow
//! the building when it is loaded somewhere else, and refuse to go onto the wrong building.
use rubble_core::testutil::*;
use rubble_core::*;

const DT: f32 = 1.0 / 60.0;
const OFFICE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../assets/buildings/office_1/building.bld");

fn run(w: &mut World, secs: f32) -> Vec<Event> {
    let mut ev = vec![];
    for _ in 0..(secs / DT) as usize {
        w.step(DT);
        ev.extend(w.drain_events());
    }
    ev
}

/// office_1 hit by five blasts along its front, then 12 s to settle.
fn ruined_office() -> Option<World> {
    if !std::path::Path::new(OFFICE).exists() {
        eprintln!("skip: no office_1 asset");
        return None;
    }
    let mut w = World::new(WorldConfig::default());
    w.load_building(OFFICE, Isometry::identity()).unwrap();
    w.add_ground_plane(0.0);
    w.step(DT);
    for x in [3.0, 8.0, 13.0, 18.0, 22.0] {
        w.explode(Explosion { center: [x, -0.5, 5.5], radius: 3.5, inner_radius: 1.5, damage: 6000.0, impulse: 20000.0 });
    }
    run(&mut w, 12.0);
    Some(w)
}

fn category(s: ChunkState) -> u8 {
    match s {
        ChunkState::Static | ChunkState::Detaching => 0,
        ChunkState::Frozen | ChunkState::InCluster(_) => 1,
        ChunkState::Gone => 2,
    }
}

fn restored(state: &DamageState, at: Isometry) -> World {
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building(OFFICE, at).unwrap();
    w.add_ground_plane(0.0);
    w.restore_damage(b, state).unwrap();
    w.step(DT);
    w
}

#[test]
fn round_trip_reproduces_the_ruin() {
    let Some(w) = ruined_office() else { return };
    let b = BuildingId(0);
    let saved = w.save_damage(b);
    assert!(!saved.gone.is_empty() && !saved.broken.is_empty() && !saved.rubble.is_empty(), "the office should be damaged");
    let json = serde_json::to_string(&saved).unwrap();
    let back: DamageState = serde_json::from_str(&json).unwrap();
    assert_eq!(back, saved, "JSON round trip");
    eprintln!(
        "saved: {} gone, {} broken joints, {} rubble groups ({} chunks), {} damaged chunks; {} KB JSON",
        saved.gone.len(),
        saved.broken.len(),
        saved.rubble.len(),
        saved.rubble.iter().map(|g| g.chunks.len()).sum::<usize>(),
        saved.hp.len(),
        json.len() / 1024
    );

    let r = restored(&back, Isometry::identity());
    let (a, z) = (w.building(b), r.building(b));
    let ta = w.chunk_world_transforms(b);
    let tz = r.chunk_world_transforms(b);
    for c in 0..a.n_chunks() {
        assert_eq!(category(a.state[c]), category(z.state[c]), "chunk {c}: {:?} vs {:?}", a.state[c], z.state[c]);
        if category(a.state[c]) != 2 {
            for i in 0..16 {
                assert!((ta[c][i] - tz[c][i]).abs() < 1e-4, "chunk {c} moved: {:?} vs {:?}", ta[c], tz[c]);
            }
        }
    }
    assert_eq!(a.edge_alive, z.edge_alive, "same joints");
}

#[test]
fn restored_ruin_stays_put() {
    let Some(w) = ruined_office() else { return };
    let saved = w.save_damage(BuildingId(0));
    let mut r = restored(&saved, Isometry::identity());
    let before = r.stats();
    let ev = run(&mut r, 4.0);
    let after = r.stats();
    let falling = ev.iter().filter(|e| matches!(e, Event::ClusterDetached { .. } | Event::ChunkDestroyed { .. })).count();
    assert_eq!(falling, 0, "nothing should fall or break after a restore: {:?}", &ev[..ev.len().min(5)]);
    assert_eq!((before.static_chunks, before.gone_chunks), (after.static_chunks, after.gone_chunks));
}

#[test]
fn rubble_follows_a_moved_building() {
    let Some(w) = ruined_office() else { return };
    let saved = w.save_damage(BuildingId(0));
    let at = Isometry::new([50.0, -20.0, 0.0], std::f32::consts::FRAC_PI_2);
    let r = restored(&saved, at);
    let pose = at.to_pose();
    let b = BuildingId(0);
    for g in saved.rubble.iter().take(20) {
        for &c in g.chunks.iter().take(3) {
            let want = pose.transform_point(w.chunk_world_com(0, c as usize));
            let got = r.chunk_world_com(0, c as usize);
            assert!((want - got).length() < 1e-3, "rubble chunk {c}: {got:?}, expected {want:?}");
        }
    }
    assert_eq!(r.building(b).state.iter().filter(|s| **s == ChunkState::Gone).count(), saved.gone.len());
}

#[test]
fn refuses_the_wrong_building() {
    let Some(w) = ruined_office() else { return };
    let saved = w.save_damage(BuildingId(0));
    let mut other = World::new(WorldConfig::default());
    let t = other.load_building_bld(tower(4, 6.0, true), Isometry::identity());
    assert!(other.restore_damage(t, &saved).is_err(), "different building");
    let mut r = restored(&saved, Isometry::identity());
    assert!(r.restore_damage(BuildingId(0), &saved).is_err(), "already damaged");
}

#[test]
fn restored_collapse_stays_put() {
    // the whole office brought down (ground floor destroyed) and settled: big rubble groups
    // lying on each other must stay as they are after a restore
    if !std::path::Path::new(OFFICE).exists() {
        return;
    }
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building(OFFICE, Isometry::identity()).unwrap();
    w.add_ground_plane(0.0);
    w.step(DT);
    for c in 0..w.building(b).n_chunks() {
        let z = w.chunk_world_com(0, c).z;
        if z > 0.4 && z < 3.4 {
            w.damage_chunk(b, c as u32, 1e9);
        }
    }
    run(&mut w, 20.0);
    let saved = w.save_damage(b);
    let rubble: usize = saved.rubble.iter().map(|g| g.chunks.len()).sum();
    eprintln!("collapse saved: {} gone, {} rubble chunks in {} groups", saved.gone.len(), rubble, saved.rubble.len());
    assert!(rubble > 1000, "the office should be a rubble pile");
    let mut r = restored(&saved, Isometry::identity());
    let t0 = r.chunk_world_transforms(b);
    let ev = run(&mut r, 4.0);
    let moved = ev.iter().filter(|e| matches!(e, Event::ClusterDetached { .. })).count();
    let t1 = r.chunk_world_transforms(b);
    let shifted = (0..t0.len()).filter(|&c| (0..16).any(|i| (t0[c][i] - t1[c][i]).abs() > 0.05)).count();
    eprintln!("after restore: {moved} pieces set moving, {shifted} chunks shifted > 5 cm");
    assert!(shifted <= rubble / 100, "the pile should stay put: {shifted} of {rubble} rubble chunks moved");
}
