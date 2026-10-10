//! `rubble-sim`: headless scenario runner and benchmark.
//!
//!   rubble-sim run scenario.yaml [--out DIR]
//!   rubble-sim bench [--buildings N] [--no-bending] [--ticks N]
use rubble_core::testutil;
use rubble_core::*;
use serde::Deserialize;
use std::io::{BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ---------------------------------------------------------------- scenario format

#[derive(Deserialize)]
struct Scenario {
    buildings: Vec<BuildingSpec>,
    steps: usize,
    #[serde(default = "default_dt")]
    dt: f32,
    #[serde(default)]
    actions: Vec<Action>,
    #[serde(default)]
    record: Record,
    /// ground plane height; `null` disables the ground, `auto` (default) puts it at the lowest
    /// foundation bottom of the loaded buildings and lays ground level around their foundations
    #[serde(default = "default_ground")]
    ground: serde_yaml::Value,
    /// optional WorldConfig overrides (any subset of fields)
    #[serde(default)]
    config: Option<serde_yaml::Value>,
}
fn default_dt() -> f32 {
    1.0 / 60.0
}
/// half-size (m) of the ground laid around buildings with sunk foundations
const TERRAIN_EXTENT: f32 = 1000.0;

fn default_ground() -> serde_yaml::Value {
    serde_yaml::Value::String("auto".into())
}

#[derive(Deserialize)]
struct BuildingSpec {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    synthetic: Option<Synthetic>,
    #[serde(default)]
    pos: [f32; 3],
    #[serde(default)]
    yaw: f32,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Synthetic {
    Tower { floors: usize, side: f32, #[serde(default)] center_column: bool },
    Grid { nx: usize, ny: usize, nz: usize, #[serde(default = "one")] size: f32 },
    Wall { w: f32, h: f32, #[serde(default = "thick")] t: f32, #[serde(default = "half")] cell: f32 },
    Cantilever { h: usize, len: usize },
    TwoBox,
}
fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}
fn thick() -> f32 {
    0.3
}

impl Synthetic {
    fn build(&self) -> rubble_format::Bld {
        match *self {
            Synthetic::Tower { floors, side, center_column } => testutil::tower(floors, side, center_column),
            Synthetic::Grid { nx, ny, nz, size } => testutil::grid_block(nx, ny, nz, size),
            Synthetic::Wall { w, h, t, cell } => testutil::wall(w, h, t, cell),
            Synthetic::Cantilever { h, len } => testutil::cantilever(h, len),
            Synthetic::TwoBox => testutil::two_box(),
        }
    }
}

#[derive(Deserialize)]
struct Action {
    t: f32,
    #[serde(default)]
    explode: Option<Explosion>,
    #[serde(default)]
    fire: Option<Fire>,
    #[serde(default)]
    damage: Option<DamageAct>,
}

#[derive(Deserialize, Clone)]
struct Fire {
    origin: [f32; 3],
    dir: [f32; 3],
    #[serde(default = "default_weapon")]
    weapon: Weapon,
    #[serde(default = "one_u")]
    count: u32,
    #[serde(default = "ten")]
    rate: f32,
    #[serde(default)]
    ballistic: bool,
}
fn default_weapon() -> Weapon {
    Weapon::Ar
}
fn one_u() -> u32 {
    1
}
fn ten() -> f32 {
    10.0
}

#[derive(Deserialize, Clone)]
struct DamageAct {
    #[serde(default)]
    building: u32,
    chunks: Vec<u32>,
    #[serde(default = "big")]
    amount: f32,
}
fn big() -> f32 {
    1e9
}

#[derive(Deserialize)]
struct Record {
    #[serde(default = "yes")]
    transforms: bool,
    #[serde(default = "yes")]
    events: bool,
    #[serde(default = "default_out")]
    out: String,
    #[serde(default = "one_u")]
    every: u32,
}
impl Default for Record {
    fn default() -> Self {
        Record { transforms: true, events: true, out: default_out(), every: 1 }
    }
}
fn yes() -> bool {
    true
}
fn default_out() -> String {
    "runs/out".into()
}

enum Scheduled {
    Explode(Explosion),
    Fire(Projectile),
    Damage(DamageAct),
}

// ---------------------------------------------------------------- transforms.bin

struct TransformWriter {
    f: BufWriter<std::fs::File>,
    n_frames: u32,
    n_total: usize,
    buf: Vec<[f32; 16]>,
}

impl TransformWriter {
    fn create(path: &Path, w: &World, names: &[String], dt: f32, every: u32) -> std::io::Result<Self> {
        let mut f = BufWriter::new(std::fs::File::create(path)?);
        let n_total: usize = w.buildings.iter().map(|b| b.n_chunks()).sum();
        f.write_all(b"RBTF")?;
        for v in [1u32, 0, n_total as u32, w.buildings.len() as u32] {
            f.write_all(&v.to_le_bytes())?;
        }
        f.write_all(&dt.to_le_bytes())?;
        f.write_all(&every.to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?;
        let mut off = 0u32;
        for (b, name) in w.buildings.iter().zip(names) {
            f.write_all(&off.to_le_bytes())?;
            f.write_all(&(b.n_chunks() as u32).to_le_bytes())?;
            let bytes = name.as_bytes();
            f.write_all(&(bytes.len() as u32).to_le_bytes())?;
            f.write_all(bytes)?;
            f.write_all(&[0u8; 3][..(4 - bytes.len() % 4) % 4])?;
            off += b.n_chunks() as u32;
        }
        Ok(TransformWriter { f, n_frames: 0, n_total, buf: Vec::new() })
    }

    fn frame(&mut self, w: &World) -> std::io::Result<()> {
        self.f.write_all(&(w.tick as u32).to_le_bytes())?;
        self.f.write_all(&w.time.to_le_bytes())?;
        let mut alive = Vec::with_capacity(self.n_total);
        for (i, b) in w.buildings.iter().enumerate() {
            self.buf.resize(b.n_chunks(), [0.0; 16]);
            w.chunk_world_transforms_into(BuildingId(i as u32), &mut self.buf);
            for m in &self.buf {
                for v in m {
                    self.f.write_all(&v.to_le_bytes())?;
                }
            }
            alive.extend(b.state.iter().map(|s| (*s != ChunkState::Gone) as u8));
        }
        self.f.write_all(&alive)?;
        self.f.write_all(&[0u8; 3][..(4 - alive.len() % 4) % 4])?;
        self.n_frames += 1;
        Ok(())
    }

    fn finish(mut self) -> std::io::Result<()> {
        self.f.flush()?;
        let mut file = self.f.into_inner().map_err(|e| e.into_error())?;
        file.seek(SeekFrom::Start(8))?;
        file.write_all(&self.n_frames.to_le_bytes())?;
        Ok(())
    }
}

// ---------------------------------------------------------------- run

fn run_scenario(path: &Path, out_override: Option<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    let sc: Scenario = serde_yaml::from_str(&text)?;
    let base = path.parent().unwrap_or(Path::new("."));
    let mut cfg = WorldConfig::default();
    if let Some(over) = &sc.config {
        let mut v = serde_yaml::to_value(&cfg)?;
        merge_yaml(&mut v, over);
        cfg = serde_yaml::from_value(v)?;
    }
    let mut w = World::new(cfg);
    let mut names = vec![];
    let t_load = Instant::now();
    for b in &sc.buildings {
        let iso = Isometry::new(b.pos, b.yaw);
        if let Some(p) = &b.path {
            let mut pp = PathBuf::from(p);
            if !pp.exists() {
                pp = base.join(p);
            }
            let pp = pp.canonicalize().unwrap_or(pp);
            w.load_building(&pp, iso)?;
            names.push(pp.to_string_lossy().into_owned());
        } else if let Some(s) = &b.synthetic {
            w.load_building_bld(s.build(), iso);
            names.push(format!("synthetic:{s:?}"));
        } else {
            return Err("building needs `path` or `synthetic`".into());
        }
    }
    let load_ms = t_load.elapsed().as_secs_f64() * 1e3;
    match &sc.ground {
        serde_yaml::Value::Null => {}
        serde_yaml::Value::String(s) if s == "auto" => {
            w.add_terrain(TERRAIN_EXTENT);
            w.add_ground_plane(w.foundation_ground_z());
        }
        v => w.add_ground_plane(v.as_f64().ok_or("ground must be a number, null or auto")? as f32),
    }
    let out = out_override.unwrap_or_else(|| {
        let o = PathBuf::from(&sc.record.out);
        if o.is_absolute() { o } else { std::env::current_dir().unwrap().join(o) }
    });
    std::fs::create_dir_all(&out)?;
    // schedule
    let mut sched: Vec<(f32, Scheduled)> = vec![];
    for a in &sc.actions {
        if let Some(e) = a.explode {
            sched.push((a.t, Scheduled::Explode(e)));
        }
        if let Some(f) = &a.fire {
            for i in 0..f.count {
                let p = if f.ballistic {
                    Projectile::ballistic(f.origin, f.dir, f.weapon)
                } else {
                    Projectile::hitscan(f.origin, f.dir, f.weapon)
                };
                sched.push((a.t + i as f32 / f.rate.max(1e-3), Scheduled::Fire(p)));
            }
        }
        if let Some(d) = &a.damage {
            sched.push((a.t, Scheduled::Damage(d.clone())));
        }
    }
    sched.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut sched = sched.into_iter().peekable();

    let mut ev_out = if sc.record.events {
        Some(BufWriter::new(std::fs::File::create(out.join("events.jsonl"))?))
    } else {
        None
    };
    let every = sc.record.every.max(1);
    let mut tw = if sc.record.transforms {
        let mut t = TransformWriter::create(&out.join("transforms.bin"), &w, &names, sc.dt, every)?;
        t.frame(&w)?;
        Some(t)
    } else {
        None
    };
    let mut tick_ms = Vec::with_capacity(sc.steps);
    let mut worst = StepTimings::default();
    let mut worst_tick = 0u64;
    let mut tick_ms_nowait = Vec::with_capacity(sc.steps);
    // per-phase totals over the run (ms)
    let mut phase = StepTimings::default();
    let mut n_events = 0usize;
    let t_run = Instant::now();
    for step in 0..sc.steps {
        while let Some((t, _)) = sched.peek() {
            if *t > w.time + 1e-6 {
                break;
            }
            match sched.next().unwrap().1 {
                Scheduled::Explode(e) => w.explode(e),
                Scheduled::Fire(p) => w.fire(p),
                Scheduled::Damage(d) => {
                    for c in d.chunks {
                        w.damage_chunk(BuildingId(d.building), c, d.amount);
                    }
                }
            }
        }
        w.step(sc.dt);
        tick_ms.push(w.timings.total_ms);
        tick_ms_nowait.push(w.timings.total_ms - w.timings.impact_wait_ms);
        let t = &w.timings;
        phase.projectiles_ms += t.projectiles_ms;
        phase.damage_ms += t.damage_ms;
        phase.connectivity_ms += t.connectivity_ms;
        phase.stress_ms += t.stress_ms;
        phase.promotion_ms += t.promotion_ms;
        phase.physics_ms += t.physics_ms;
        phase.impacts_ms += t.impacts_ms;
        phase.settle_ms += t.settle_ms;
        phase.impact_wait_ms += t.impact_wait_ms;
        phase.total_ms += t.total_ms;
        if w.timings.total_ms > worst.total_ms {
            worst = w.timings;
            worst_tick = w.tick;
        }
        let evs = w.drain_events();
        n_events += evs.len();
        if let Some(f) = ev_out.as_mut() {
            for e in evs {
                let mut v = serde_json::to_value(&e)?;
                if let Some(o) = v.as_object_mut() {
                    o.insert("tick".into(), (w.tick - 1).into());
                    o.insert("time".into(), (w.time - sc.dt).into());
                }
                serde_json::to_writer(&mut *f, &v)?;
                f.write_all(b"\n")?;
            }
        }
        if let Some(t) = tw.as_mut() {
            if (step + 1) % every as usize == 0 {
                t.frame(&w)?;
            }
        }
    }
    let wall_s = t_run.elapsed().as_secs_f64();
    if let Some(mut f) = ev_out {
        f.flush()?;
    }
    if let Some(t) = tw {
        t.finish()?;
    }
    let (avg, p99, max) = summarize(&mut tick_ms.clone());
    let (_, p99_nw, max_nw) = summarize(&mut tick_ms_nowait);
    let st = w.stats();
    let summary = serde_json::json!({
        "steps": sc.steps, "dt": sc.dt, "load_ms": load_ms, "wall_s": wall_s,
        "tick_ms": {"avg": avg, "p99": p99, "max": max, "p99_excl_impact_wait": p99_nw, "max_excl_impact_wait": max_nw}, "worst_tick": worst_tick, "worst_timings": worst, "events": n_events, "final_stats": st,
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary)?)?;
    println!(
        "ran {} steps in {:.2}s  (load {:.1} ms, tick avg {:.3} ms, p99 {:.3} ms, max {:.3} ms), {} events -> {}",
        sc.steps, wall_s, load_ms, avg, p99, max, n_events, out.display()
    );
    println!("excluding blocking waits on background impact solves: p99 {p99_nw:.3} ms, max {max_nw:.3} ms");
    println!("final: {st:?}");
    println!("worst tick {worst_tick}: {worst:?}");
    println!("phase totals: {phase:?}");
    Ok(())
}

fn merge_yaml(dst: &mut serde_yaml::Value, src: &serde_yaml::Value) {
    match (dst, src) {
        (serde_yaml::Value::Mapping(d), serde_yaml::Value::Mapping(s)) => {
            for (k, v) in s {
                match d.get_mut(k) {
                    Some(dv) => merge_yaml(dv, v),
                    None => {
                        d.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (d, s) => *d = s.clone(),
    }
}

fn summarize(v: &mut [f32]) -> (f32, f32, f32) {
    if v.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let avg = v.iter().sum::<f32>() / v.len() as f32;
    v.sort_by(|a, b| a.total_cmp(b));
    let p99 = v[((v.len() as f32 * 0.99) as usize).min(v.len() - 1)];
    (avg, p99, *v.last().unwrap())
}

// ---------------------------------------------------------------- bench

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

fn bench(n_buildings: usize, bending: bool, ticks: usize) {
    let mut cfg = WorldConfig::default();
    cfg.stress.bending = bending;
    let dt = 1.0 / 60.0;
    println!("rubble-sim bench  (threads: {}, stress bending: {bending})", rayon::current_num_threads());
    println!();
    println!("{:<44} {:>12}", "item", "value");
    println!("{}", "-".repeat(58));

    // load times
    let block = testutil::grid_block(20, 20, 10, 1.0);
    let tower = testutil::tower_cols(8, 10.0, true, 1.2);
    for (name, bld) in [("grid 20x20x10", &block), ("tower 8 floors", &tower)] {
        let mut best = f64::MAX;
        for _ in 0..5 {
            let mut w = World::new(cfg.clone());
            let b = bld.clone();
            let t = Instant::now();
            w.load_building_bld(b, Isometry::identity());
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        println!("{:<44} {:>9.2} ms", format!("load {name} ({} chunks)", bld.chunks.len()), best);
    }

    // arena
    let mut w = World::new(cfg.clone());
    let spacing = 32.0;
    let side = (n_buildings as f32).sqrt().ceil() as usize;
    let t = Instant::now();
    let mut kinds = vec![];
    for i in 0..n_buildings {
        let pos = [(i % side) as f32 * spacing, (i / side) as f32 * spacing, 0.0];
        let is_tower = i % 2 == 1;
        let bld = if is_tower { tower.clone() } else { block.clone() };
        w.load_building_bld(bld, Isometry::new(pos, 0.0));
        kinds.push((pos, is_tower));
    }
    w.add_ground_plane(0.0);
    let load_ms = t.elapsed().as_secs_f64() * 1e3;
    let st = w.stats();
    println!("{:<44} {:>9.1} ms", format!("load arena ({n_buildings} bldgs, {} chunks)", st.chunks_total), load_ms);
    let t = Instant::now();
    w.step(dt);
    println!("{:<44} {:>9.2} ms", "first tick (initial stress solve)", t.elapsed().as_secs_f64() * 1e3);
    for _ in 0..60 {
        w.step(dt);
    }
    let ev = w.drain_events();
    if !ev.is_empty() {
        println!("WARNING: intact arena produced {} events (first: {:?})", ev.len(), ev[0]);
    }

    // steady state: 12 AR shots per tick at random buildings
    let mut rng = Lcg(7);
    let mut times = vec![];
    let mut proj_ms = vec![];
    for _ in 0..ticks {
        for _ in 0..12 {
            let (pos, _) = kinds[(rng.f() * kinds.len() as f32) as usize % kinds.len()];
            let target = [pos[0] + 2.0 + rng.f() * 8.0, pos[1] + 2.0 + rng.f() * 8.0, 0.5 + rng.f() * 9.0];
            let origin = [target[0] - 15.0, target[1] - 25.0, 1.7];
            let d = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
            w.fire(Projectile::hitscan(origin, d, Weapon::Ar));
        }
        w.step(dt);
        times.push(w.timings.total_ms);
        proj_ms.push(w.timings.projectiles_ms);
        w.drain_events();
    }
    let (avg, p99, max) = summarize(&mut times);
    println!("{:<44} {:>9.3} ms", "steady tick avg (12 AR shots/tick)", avg);
    println!("{:<44} {:>9.3} ms", "steady tick p99", p99);
    println!("{:<44} {:>9.3} ms", "steady tick max", max);
    println!("{:<44} {:>9.3} ms", "  of which projectiles avg", summarize(&mut proj_ms).0);

    // worst case: blow out the base of 4 towers and cut 4 blocks off their bottom layer
    let mut towers = 0;
    let mut blocks = 0;
    for (i, (pos, is_tower)) in kinds.iter().enumerate() {
        if *is_tower && towers < 4 {
            towers += 1;
            w.explode(Explosion { center: [pos[0] + 5.0, pos[1] + 5.0, 1.5], radius: 8.0, inner_radius: 7.5, damage: 1e5, impulse: 2e4 });
        } else if !*is_tower && blocks < 4 {
            blocks += 1;
            let b = BuildingId(i as u32);
            for c in 0..400u32 {
                w.damage_chunk(b, c, 1e9);
            }
        }
    }
    let mut times = vec![];
    let mut worst = StepTimings::default();
    let mut worst_stats = w.stats();
    let mut n_ev = 0;
    let trace = std::env::var("RUBBLE_TRACE").is_ok();
    for i in 0..600 {
        w.step(dt);
        n_ev += w.drain_events().len();
        times.push(w.timings.total_ms);
        if trace && i % 20 == 0 {
            let s = w.stats();
            eprintln!("t{i:4} {:7.2} ms phys {:7.2} conn {:5.2} stress {:5.2} | clusters {:4} flight {:5} frozen {:5} bodies {} colliders {}",
                w.timings.total_ms, w.timings.physics_ms, w.timings.connectivity_ms, w.timings.stress_ms, s.clusters, s.chunks_in_flight, s.frozen_chunks, s.rapier_bodies, s.rapier_colliders);
        }
        if w.timings.total_ms >= worst.total_ms {
            worst = w.timings;
            worst_stats = w.stats();
        }
    }
    let (avg, p99, max) = summarize(&mut times);
    println!("{:<44} {:>9.3} ms", "collapse tick avg (600 ticks)", avg);
    println!("{:<44} {:>9.3} ms", "collapse tick p99", p99);
    println!("{:<44} {:>9.3} ms", "collapse tick max (worst)", max);
    println!(
        "  worst breakdown: proj {:.2} dmg {:.2} conn {:.2} stress {:.2} promo {:.2} phys {:.2} impacts {:.2} settle {:.2}",
        worst.projectiles_ms, worst.damage_ms, worst.connectivity_ms, worst.stress_ms, worst.promotion_ms,
        worst.physics_ms, worst.impacts_ms, worst.settle_ms
    );
    println!(
        "  worst-tick state: {} clusters, {} chunks in flight, {} frozen; {} events total",
        worst_stats.clusters, worst_stats.chunks_in_flight, worst_stats.frozen_chunks, n_ev
    );
    let st = w.stats();
    println!("  end: {} clusters, {} in flight, {} frozen, {} gone", st.clusters, st.chunks_in_flight, st.frozen_chunks, st.gone_chunks);
}

fn usage() -> ! {
    eprintln!("usage:\n  rubble-sim run <scenario.yaml> [--out DIR]\n  rubble-sim bench [--buildings N] [--ticks N] [--no-bending]");
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    match args.get(1).map(String::as_str) {
        Some("run") => {
            let Some(p) = args.get(2) else { usage() };
            if let Err(e) = run_scenario(Path::new(p), flag("--out").map(PathBuf::from)) {
                eprintln!("error: {e}");
                std::process::exit(1);
            }
        }
        Some("bench") => {
            let n = flag("--buildings").and_then(|v| v.parse().ok()).unwrap_or(40);
            let ticks = flag("--ticks").and_then(|v| v.parse().ok()).unwrap_or(300);
            bench(n, !args.iter().any(|a| a == "--no-bending"), ticks);
        }
        _ => usage(),
    }
}
