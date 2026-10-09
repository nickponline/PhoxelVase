//! Several worlds stepped at the same time on different threads (a server running matches, a
//! test runner, an editor with previews) must never deadlock. They share the physics thread
//! pool; rapier's step parks a pool worker on a channel while its deferred broad-phase task
//! runs elsewhere in the pool, which used to hang when several steps were inside the pool.
use rubble_core::testutil::*;
use rubble_core::*;
use std::sync::mpsc;
use std::time::Duration;

const DT: f32 = 1.0 / 60.0;

const OFFICE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../assets/buildings/office_1/building.bld");

/// A collapse that keeps the physics pool busy: office_1 (or a synthetic tower when the asset
/// is missing) with its ground floor destroyed.
fn collapse(seed: u32, ticks: usize, physics_threads: usize) -> usize {
    let mut w = World::new(WorldConfig { physics_threads, ..Default::default() });
    let b = if std::path::Path::new(OFFICE).exists() {
        w.load_building(OFFICE, Isometry::identity()).unwrap()
    } else {
        w.load_building_bld(tower(10, 9.0, true), Isometry::identity())
    };
    w.add_ground_plane(0.0);
    w.step(DT);
    let n = w.building(b).n_chunks();
    for c in 0..n {
        let z = w.chunk_world_com(0, c).z;
        if z > 0.4 && z < 3.4 && !(c as u32 + seed).is_multiple_of(7) {
            w.damage_chunk(b, c as u32, 1e9);
        }
    }
    for _ in 0..ticks {
        w.step(DT);
        w.drain_events();
    }
    w.stats().gone_chunks
}

/// `threads` worlds collapsing at once, each on its own thread, sharing a physics pool of
/// `pool` threads; fails instead of hanging.
fn concurrent(threads: u32, rounds: u32, pool: usize) {
    let (tx, rx) = mpsc::channel();
    for t in 0..threads {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for round in 0..rounds {
                collapse(t * 10 + round, 300, pool);
            }
            let _ = tx.send(t);
        });
    }
    drop(tx);
    for _ in 0..threads {
        if let Err(e) = rx.recv_timeout(Duration::from_secs(120)) {
            panic!("worlds stepping concurrently hung (deadlock): {e:?}");
        }
    }
}

#[test]
fn many_worlds_on_many_threads() {
    concurrent(16, 2, 0);
}

#[test]
fn many_worlds_on_a_small_pool() {
    // the smallest pool rapier defers work in (2 threads): the easiest to exhaust
    concurrent(16, 2, 2);
}
