//! Randomized destruction (beam sweeps, blasts, point damage) on real buildings, auditing
//! every 0.5 s that nothing hangs in the air:
//! * every static chunk still reaches an anchor through the joint graph (`unanchored_static`);
//! * no frozen rubble / resting cluster is in mid-air (`floating_report`, touching nothing).
//! Run with `--ignored --nocapture`; `FUZZ_SEEDS=n` runs per building (default 2).
use rubble_core::*;
use rubble_format::Bld;

const DT: f32 = 1.0 / 60.0;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

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

/// Returns human-readable violations.
fn fuzz(name: &str, seed: u64, secs: f32) -> Vec<String> {
    let Some(bld) = load_asset(name) else { return vec![] };
    let (lo, hi) = bounds(&bld);
    let mut cfg = WorldConfig::default();
    cfg.keep_debris = std::env::var_os("FUZZ_KEEP").is_some();
    let mut w = World::new(cfg);
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    let (mut late_ms, mut late_n) = (0.0f32, 0usize);
    let thin0: Vec<bool> = {
        let (bd, cfg) = (w.building(b), WorldConfig::default());
        (0..bd.n_chunks()).map(|k| bd.hangs_by_slivers(k, cfg.sliver_area, cfg.sliver_seat_area)).collect()
    };
    let mut rng = Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) ^ 0xABCDEF);
    let c = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
    let r = ((hi[0] - lo[0]).powi(2) + (hi[1] - lo[1]).powi(2)).sqrt() * 0.5 + 5.0;
    let mut out = vec![];
    let mut log: Vec<String> = vec![];
    let watch: Option<u32> = std::env::var("FUZZ_WATCH").ok().and_then(|v| v.parse().ok());
    let mut prev_st: Option<ChunkState> = None;
    let mut beam: Option<([f32; 3], f32, f32, usize)> = None; // eye, yaw0, yaw rate, ticks left
    let n = (secs / DT) as usize;
    for t in 0..n {
        let busy = t < n * 2 / 3;
        if busy && beam.is_none() && t % 60 == 0 {
            match (rng.f() * 3.0) as u32 {
                0 => {
                    let z = rng.r(lo[2].max(0.3), hi[2]);
                    let a = rng.r(0.0, std::f32::consts::TAU);
                    let eye = [c[0] + r * a.cos(), c[1] + r * a.sin(), z];
                    let yaw0 = a + std::f32::consts::PI + rng.r(-0.6, 0.6);
                    let rate = rng.r(-0.03, 0.03);
                    log.push(format!("t{t} beam z{z:.1}"));
                    beam = Some((eye, yaw0, rate, (rng.r(20.0, 90.0)) as usize));
                }
                1 => {
                    let p = [rng.r(lo[0], hi[0]), rng.r(lo[1], hi[1]), rng.r(lo[2].max(0.5), hi[2])];
                    let rad = rng.r(2.0, 6.0);
                    log.push(format!("t{t} blast r{rad:.1} at {p:.1?}"));
                    w.explode(Explosion { center: p, radius: rad, inner_radius: rad * 0.4, damage: 5000.0, impulse: 5000.0 * rad });
                }
                _ => {
                    let nch = w.building(b).n_chunks();
                    let k = (rng.f() * nch as f32) as u32 % nch as u32;
                    log.push(format!("t{t} kill around c{k}"));
                    for d in 0..12u32 {
                        w.damage_chunk(b, (k + d * 7) % nch as u32, 1e9);
                    }
                }
            }
        }
        if let Some((eye, yaw, rate, left)) = beam.as_mut() {
            *yaw += *rate;
            w.beam(*eye, [yaw.cos(), yaw.sin(), 0.0], 400.0, 0.25, 400.0, 60.0);
            *left -= 1;
            if *left == 0 {
                beam = None;
            }
        }
        let t0 = std::time::Instant::now();
        w.step(DT);
        if t >= n - 300 {
            late_ms += t0.elapsed().as_secs_f32() * 1e3;
            late_n += 1;
        }
        let evs = w.drain_events();
        if let Some(wc) = watch {
            let st = w.building(b).state[wc as usize];
            let z = w.chunk_world_com(b.0 as usize, wc as usize).z;
            for e in &evs {
                let txt = format!("{e:?}");
                if (txt.contains(&format!(" {wc},")) || txt.contains(&format!("[{wc},")) || txt.contains(&format!(" {wc}]"))) || (matches!(st, ChunkState::InCluster(_)) && txt.contains("Cluster") && txt.contains(&format!("{:?}", st).replace("InCluster(ClusterKey(", "").split('v').next().unwrap_or("#").to_string()) && false) {
                    eprintln!("    t{t} {}", &txt[..txt.len().min(160)]);
                }
            }
            if t % 15 == 0 || prev_st != Some(st) {
                let rep = w.floating_report();
                let fl = rep.frozen.iter().any(|g| g.chunks.contains(&wc));
                eprintln!("    t{t} c{wc} {st:?} z {z:.2} floating {fl} clusters {}", w.clusters.len());
            }
            prev_st = Some(st);
        }
        if t % 30 == 29 {
            let (s, loose) = w.unanchored_static(b);
            if !s.is_empty() || !loose.is_empty() {
                let zs: Vec<f32> = s.iter().take(5).map(|&k| w.building(b).bld.chunks[k as usize].com[2]).collect();
                out.push(format!("{name} seed{seed} t{t}: {} static structural + {} non-structural chunks unanchored (z {zs:.1?}); last actions {:?}",
                    s.len(), loose.len(), &log[log.len().saturating_sub(3)..]));
            }
            // mid-air rubble: only judged once settled (end of run) to avoid flagging things mid-flight
        }
    }
    {
        // static chunks left hanging by slivers by the destruction (not thin parts of the intact model)
        let bd = w.building(b);
        let cfg = WorldConfig::default();
        let thin: Vec<usize> = (0..bd.n_chunks())
            .filter(|&k| !thin0[k] && bd.hangs_by_slivers(k, cfg.sliver_area, cfg.sliver_seat_area))
            .collect();
        if !thin.is_empty() {
            out.push(format!("{name} seed{seed} end: {} static chunks hanging by slivers, e.g. {:?}", thin.len(), &thin[..thin.len().min(4)]));
        }
    }
    let rep = w.floating_report();
    for (kind, g) in rep.frozen.iter().map(|g| ("frozen", g)).chain(rep.clusters.iter().map(|(_, g)| ("cluster", g))) {
        if g.touching == 0 || g.chunks.len() > 2 {
            let st: Vec<String> = g.chunks.iter().take(3).map(|&c| {
                let bd = w.building(b);
                format!("c{c} {:?} vol {:.3}", bd.state[c as usize], bd.bld.chunks[c as usize].volume)
            }).collect();
            out.push(format!("{name} seed{seed} end: {kind} {} chunk group unsupported at z {:.1}, touching {} {st:?}", g.chunks.len(), g.min_com_z, g.touching));
        }
    }
    let st = w.stats();
    eprintln!("{name} seed{seed}: {} violations; frozen {} gone {} clusters {} | last 5 s tick avg {:.3} ms",
        out.len(), st.frozen_chunks, st.gone_chunks, w.clusters.len(), late_ms / late_n.max(1) as f32);
    out
}

#[test]
#[ignore]
fn fuzz_nothing_hangs() {
    let seeds: u64 = std::env::var("FUZZ_SEEDS").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let only = std::env::var("FUZZ_ONLY").ok();
    let names = ["house_1", "kyoto_1", "warehouse_1", "office_1", "apartment_1", "tower_1", "eiffel_1", "suspension_bridge_1", "aqueduct_1", "colossus_1"];
    let mut bad = vec![];
    for name in names {
        if only.as_deref().is_some_and(|o| !name.starts_with(o)) {
            continue;
        }
        let first: u64 = std::env::var("FUZZ_FIRST").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        for s in first..first + seeds {
            bad.extend(fuzz(name, s, 30.0));
        }
    }
    for b in &bad {
        eprintln!("{b}");
    }
    assert!(bad.is_empty(), "{} violations", bad.len());
}

#[test]
#[ignore]
fn load_frozen_counts() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/buildings");
    let mut names: Vec<String> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    for name in names {
        let Some(bld) = load_asset(&name) else { continue };
        let mut w = World::new(WorldConfig::default());
        let b = w.load_building_bld(bld, Isometry::identity());
        let bd = w.building(b);
        if bd.load_floating > 0 || bd.load_cracked_edges > 0 {
            eprintln!("{name:24} floating/frozen at load {:5}  pre-cracked edges {:5}", bd.load_floating, bd.load_cracked_edges);
        }
    }
}
