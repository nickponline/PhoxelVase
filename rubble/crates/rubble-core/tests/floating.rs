//! Nothing may hang in the air after destruction: real buildings are cut with the viewer's beam
//! and blasted, then `World::floating_report` must find no unsupported rubble once the dust
//! settles. Skips buildings whose assets are missing.
use rubble_core::*;
use rubble_format::Bld;

const DT: f32 = 1.0 / 60.0;

fn load_asset(name: &str) -> Option<Bld> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/buildings").join(name).join("building.bld");
    Bld::load(&p).ok()
}

fn bounds(b: &Bld) -> ([f32; 3], [f32; 3]) {
    let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
    for c in &b.chunks {
        for k in 0..3 {
            lo[k] = lo[k].min(c.aabb_min[k]);
            hi[k] = hi[k].max(c.aabb_max[k]);
        }
    }
    (lo, hi)
}

fn summarize(r: &FloatingReport) -> String {
    let f: Vec<String> = r.frozen.iter().take(6).map(|g| format!("frozen x{} z{:.1} touching {}", g.chunks.len(), g.min_com_z, g.touching)).collect();
    let c: Vec<String> = r.clusters.iter().take(6).map(|(_, g)| format!("cluster x{} z{:.1} touching {}", g.chunks.len(), g.min_com_z, g.touching)).collect();
    format!("{} frozen, {} clusters: {:?} {:?}", r.frozen.len(), r.clusters.len(), f, c)
}

/// Sweep the beam through the building at height `z` (fraction of its height) for `secs`,
/// like holding LMB and panning across, then let it settle.
fn beam_and_settle(name: &str, zf: &[f32], blast: bool) -> Option<FloatingReport> {
    let bld = load_asset(name)?;
    let (lo, hi) = bounds(&bld);
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
    let r = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt();
    for &f in zf {
        let z = lo[2] + (hi[2] - lo[2]) * f;
        // eye outside the building, sweep the aim across its whole width over 1.5 s
        let eye = [c[0] - r, c[1] - r * 0.3, z];
        let n = 90;
        for i in 0..n {
            let t = i as f32 / (n - 1) as f32;
            let target = [c[0] + (t - 0.5) * r * 1.2, c[1] + (0.5 - t) * r * 0.6, z];
            let d = [target[0] - eye[0], target[1] - eye[1], 0.0];
            let l = (d[0] * d[0] + d[1] * d[1]).sqrt();
            w.beam(eye, [d[0] / l, d[1] / l, 0.0], 400.0, 0.25, 400.0, 60.0);
            w.step(DT);
            w.hurry_collapse(b);
            w.drain_events();
        }
    }
    if blast {
        w.explode(Explosion { center: [c[0], c[1], lo[2] + 1.0], radius: 6.0, inner_radius: 2.5, damage: 5000.0, impulse: 30000.0 });
    }
    for _ in 0..(40.0 / DT) as usize {
        w.step(DT);
        w.drain_events();
    }
    let r = w.floating_report();
    let st = w.stats();
    eprintln!("{name} beam@{zf:?} blast={blast}: {} | clusters {} frozen_chunks {}", summarize(&r), w.clusters.len(), st.frozen_chunks);
    Some(r)
}

#[test]
#[ignore] // ~15 s; run with --ignored --nocapture
fn real_buildings_nothing_floats() {
    let mut bad = vec![];
    for name in ["house_1", "kyoto_1", "warehouse_1", "office_1", "apartment_1", "tower_1"] {
        for (zf, blast) in [(vec![0.08], false), (vec![0.5], false), (vec![0.08, 0.45], true)] {
            if let Some(r) = beam_and_settle(name, &zf, blast) {
                // a piece or two wedged sideways into the pile (friction) is fine; anything in
                // mid-air, or a bigger group without support, is not
                let real = r.frozen.iter().chain(r.clusters.iter().map(|(_, g)| g)).any(|g| g.touching == 0 || g.chunks.len() > 2);
                if real {
                    bad.push(format!("{name} {zf:?} {blast}: {}", summarize(&r)));
                }
            }
        }
    }
    assert!(bad.is_empty(), "floating rubble:\n{}", bad.join("\n"));
}


/// No intact building may fail the tipping check at any height (it would collapse at the
/// first scratch). Checks every 0.25 m plane of every asset present.
#[test]
fn intact_buildings_pass_tipping_everywhere() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/buildings");
    let Ok(rd) = std::fs::read_dir(&dir) else { return };
    let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    let mut bad = vec![];
    for name in names {
        let Some(bld) = load_asset(&name) else { continue };
        let (lo, hi) = bounds(&bld);
        let w = {
            let mut w = World::new(WorldConfig::default());
            w.load_building_bld(bld, Isometry::identity());
            w
        };
        let b = &w.buildings[0];
        let mut z = lo[2].max(0.0) + 0.125;
        while z < hi[2] {
            let e = b.tipping_edges(z, WorldConfig::default().tip_margin);
            if !e.is_empty() {
                bad.push(format!("{name} z {z:.2}: {} edges", e.len()));
                break;
            }
            z += 0.25;
        }
    }
    assert!(bad.is_empty(), "intact buildings failing the tipping check:\n{}", bad.join("\n"));
}
