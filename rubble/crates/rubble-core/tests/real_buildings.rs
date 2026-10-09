//! Idle-stability on bgen-generated buildings (skipped when assets are missing).
use rubble_core::*;

const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../assets/buildings");

fn idle(name: &str, cfg: WorldConfig, secs: f32) -> Option<(usize, usize, usize, f32)> {
    let p = format!("{ASSETS}/{name}/building.bld");
    if !std::path::Path::new(&p).exists() {
        eprintln!("skip {name}: no asset");
        return None;
    }
    let mut w = World::new(cfg);
    let b = w.load_building(&p, Isometry::identity()).unwrap();
    w.add_ground_plane(0.0);
    let mut detached = 0;
    let mut breaks = 0;
    let mut destroyed = 0;
    for _ in 0..(secs * 60.0) as usize {
        w.step(1.0 / 60.0);
        for e in w.drain_events() {
            match &e {
                Event::ClusterDetached { chunks, .. } => detached += chunks.len(),
                Event::EdgeBroken { .. } => breaks += 1,
                Event::ChunkDestroyed { .. } | Event::ChunkShattered { .. } => {
                    destroyed += 1;
                    eprintln!("  {name}: {e:?}");
                }
                Event::CollapseWarning { ref chunks, .. } => eprintln!("  {name}: collapse warning {} chunks {:?}", chunks.len(), &chunks[..chunks.len().min(5)]),
                _ => {}
            }
        }
    }
    let mu = w.building_state(b).utilization.iter().cloned().fold(0.0f32, f32::max);
    let bd = w.building(b);
    eprintln!("  {name}: load-time pre-cracked joints {} / {}, floating chunks frozen {}", bd.load_cracked_edges, bd.bld.edges.len(), bd.load_floating);
    Some((detached, breaks, destroyed, mu))
}

fn presets() -> Vec<String> {
    let mut v = vec![];
    for p in ["office", "apartment", "warehouse", "tower", "house", "kyoto"] {
        for s in [1, 2, 3] {
            v.push(format!("{p}_{s}"));
        }
    }
    v.push("building4_4".into());
    v
}

fn check(bending: bool) {
    let mut bad = vec![];
    for name in presets() {
        let mut cfg = WorldConfig::default();
        cfg.stress.bending = bending;
        cfg.emit_edge_events = true;
        if let Some((d, b, x, mu)) = idle(&name, cfg, 10.0) {
            eprintln!("{name:12} bending={bending} detached={d} edge_breaks={b} destroyed={x} max_util={mu:.3}");
            if d + b + x > 0 {
                bad.push(name);
            }
        }
    }
    assert!(bad.is_empty(), "unstable idle buildings: {bad:?}");
}

#[test]
fn presets_idle_axial() {
    check(false);
}

#[test]
fn presets_idle_bending() {
    check(true);
}
