//! Loss of support in real and synthetic buildings: crushing, catch-up solve, buckling,
//! tipping, damage-weakened joints. Real-building cases skip when assets are missing.
use rubble_core::rubble_format::Bld;
use rubble_core::testutil::*;
use rubble_core::*;

const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../assets/buildings");
const DT: f32 = 1.0 / 60.0;
const KIND_FLOOR: u16 = 2;
const KIND_COLUMN: u16 = 3;

/// The pre-change stress model (for A/B numbers): no crushing realism, no bearing sections,
/// no buckling, no catch-up, no damage weakening, 4096-node windows.
pub fn legacy(mut c: WorldConfig) -> WorldConfig {
    c.stress.compression_factor = 10.0;
    c.stress.bearing_sections = false;
    c.stress.buckling = false;
    c.stress.catchup_frac = 0.0;
    c.stress.damage_weakens = false;
    c.stress.region_max_nodes = 4096;
    c
}

#[derive(Debug, Default)]
struct Outcome {
    /// chunks in detached clusters (excluding debris of the destroyed chunks)
    detached: usize,
    /// edges broken by stress
    breaks: usize,
    /// (time s, max util over the kept chunks' edges, max util overall)
    trace: Vec<(f32, f32, f32)>,
    first_detach: Option<f32>,
}

/// Load `bld`, settle, destroy `kill` chunks, run `secs`; util traced on edges of `watch`.
fn run_scenario(bld: Bld, cfg: WorldConfig, kill: &[u32], watch: &[u32], secs: f32) -> Outcome {
    let mut cfg = cfg;
    cfg.emit_edge_events = true;
    let mut w = World::new(cfg);
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    for _ in 0..30 {
        w.step(DT);
    }
    let pre = w.drain_events();
    assert!(
        !pre.iter().any(|e| matches!(e, Event::ClusterDetached { .. } | Event::EdgeBroken { .. })),
        "intact building must stand: {:?}",
        &pre[..pre.len().min(3)]
    );
    for &c in kill {
        w.damage_chunk(b, c, 1e12);
    }
    let mut out = Outcome::default();
    let marks = [0.02f32, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0];
    let steps = (secs / DT).round() as usize;
    for s in 1..=steps {
        w.step(DT);
        let t = s as f32 * DT;
        for e in w.drain_events() {
            match e {
                // debris of the destroyed chunks themselves doesn't count
                Event::ClusterDetached { chunks, .. } if chunks.iter().any(|c| !kill.contains(c)) => {
                    out.detached += chunks.len();
                    out.first_detach.get_or_insert(t);
                }
                Event::EdgeBroken { cause: BreakCause::Stress, .. } => out.breaks += 1,
                _ => {}
            }
        }
        if marks.iter().any(|&m| (m - t).abs() < 0.5 * DT) {
            let bd = w.building(b);
            let mut uw = 0f32;
            for &c in watch {
                for (_, e) in bd.neighbors(c) {
                    uw = uw.max(bd.utilization[e as usize]);
                }
            }
            let ua = bd.utilization.iter().cloned().fold(0.0f32, f32::max);
            out.trace.push((t, uw, ua));
        }
    }
    out
}

fn load_asset(name: &str) -> Option<Bld> {
    let p = format!("{ASSETS}/{name}/building.bld");
    match Bld::load(&p) {
        Ok(b) => Some(b),
        Err(_) => {
            eprintln!("skip: no {name} asset");
            None
        }
    }
}

/// Ground-floor chunks (aabb within z 0..3.5, not floor slabs); column chunks grouped by element.
fn ground_floor(bld: &Bld) -> (Vec<u32>, Vec<(u32, [f32; 2], Vec<u32>)>) {
    let mut ground = vec![];
    let mut cols: Vec<(u32, [f32; 2], Vec<u32>)> = vec![];
    for (i, c) in bld.chunks.iter().enumerate() {
        let kind = bld.elements[c.elem as usize].kind;
        if c.aabb_min[2] >= -0.01 && c.aabb_max[2] <= 3.51 && kind != KIND_FLOOR {
            ground.push(i as u32);
            if kind == KIND_COLUMN {
                match cols.iter_mut().find(|x| x.0 == c.elem) {
                    Some(x) => x.2.push(i as u32),
                    None => cols.push((c.elem, [c.com[0], c.com[1]], vec![i as u32])),
                }
            }
        }
    }
    (ground, cols)
}

fn office_case(name: &str, keep: impl Fn(&[(u32, [f32; 2], Vec<u32>)], &Bld) -> Vec<u32>, cfg: WorldConfig, secs: f32) -> Option<Outcome> {
    let bld = load_asset(name)?;
    let (ground, cols) = ground_floor(&bld);
    let keep_set = keep(&cols, &bld);
    let kill: Vec<u32> = ground.iter().copied().filter(|c| !keep_set.contains(c)).collect();
    let o = run_scenario(bld, cfg, &kill, &keep_set, secs);
    println!("{name}: kept {} chunks, killed {}: {o:?}", keep_set.len(), kill.len());
    Some(o)
}

fn corner_column(cols: &[(u32, [f32; 2], Vec<u32>)], _: &Bld) -> Vec<u32> {
    // the column nearest the footprint's min corner, ignoring outliers outside the walls
    let c = cols
        .iter()
        .filter(|c| c.1[0] > 0.0 && c.1[1] > 0.0)
        .min_by(|a, b| (a.1[0] + a.1[1]).total_cmp(&(b.1[0] + b.1[1])))
        .unwrap();
    c.2.clone()
}

fn interior_columns(cols: &[(u32, [f32; 2], Vec<u32>)], bld: &Bld) -> Vec<u32> {
    // footprint of the building proper: the foundation below z = 0 is padded beyond it
    let (lo, hi) = bld.chunks.iter().filter(|c| c.aabb_min[2] >= -0.01).fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), c| {
        ([lo[0].min(c.com[0]), lo[1].min(c.com[1])], [hi[0].max(c.com[0]), hi[1].max(c.com[1])])
    });
    cols.iter()
        .filter(|c| c.1[0] > lo[0] + 2.0 && c.1[0] < hi[0] - 2.0 && c.1[1] > lo[1] + 2.0 && c.1[1] < hi[1] - 2.0)
        .flat_map(|c| c.2.clone())
        .collect()
}

fn all_columns(cols: &[(u32, [f32; 2], Vec<u32>)], _: &Bld) -> Vec<u32> {
    cols.iter().flat_map(|c| c.2.clone()).collect()
}

#[test]
fn office_one_corner_column_collapses() {
    if let Some(o) = office_case("office_1", corner_column, WorldConfig::default(), 4.0) {
        assert!(o.first_detach.is_some_and(|t| t <= 3.0) && o.detached > 500, "{o:?}");
    }
}

#[test]
fn office_two_interior_columns_collapse() {
    if let Some(o) = office_case("office_1", interior_columns, WorldConfig::default(), 4.0) {
        assert!(o.first_detach.is_some_and(|t| t <= 3.0) && o.detached > 500, "{o:?}");
    }
}

#[test]
fn office_all_columns_no_walls_stands() {
    if let Some(o) = office_case("office_1", all_columns, WorldConfig::default(), 10.0) {
        // a few slab joints may crack while the load re-routes, but nothing comes down
        assert!(o.detached == 0 && o.breaks < 50, "{o:?}");
    }
}

/// A/B numbers for the report (run with --ignored --nocapture).
#[test]
#[ignore]
fn office_pillar_cases_legacy() {
    for (label, keep) in [
        ("1 corner column", corner_column as fn(&[(u32, [f32; 2], Vec<u32>)], &Bld) -> Vec<u32>),
        ("2 interior columns", interior_columns),
        ("all columns, no walls", all_columns),
    ] {
        println!("--- legacy: {label}");
        office_case("office_1", keep, legacy(WorldConfig::default()), 10.0);
    }
}

// ---------------------------------------------------------------- synthetic members

const KIND_EXT_WALL: u16 = 0;
const KIND_BEAM: u16 = 4;

/// Anchored 3×3 m pad, a `t`×`t` column of `h` 1 m chunks (ELEM kind column) and a heavy
/// concrete block (3×3×`top_h` m) on top. With `brace_at` = Some(z), two thin struts tie the
/// column at height z to anchored side walls. Returns (bld, column chunks, strut chunks).
fn braced_column(t: f32, h: usize, top_h: f32, brace_at: Option<f32>) -> (Bld, Vec<u32>, Vec<u32>) {
    let mut s = SynthBuilder::new("braced_column");
    let (c0, c1) = (1.5 - t / 2.0, 1.5 + t / 2.0);
    s.add_grid([0.0, 0.0, 0.0], [3.0, 3.0, 0.3], [2, 2, 1], 0, F_INDESTRUCTIBLE_FLAG);
    s.begin_element(KIND_COLUMN);
    let first = s.boxes.len() as u32;
    s.add_grid([c0, c0, 0.3], [c1, c1, 0.3 + h as f32], [1, 1, h], 0, 0);
    let col: Vec<u32> = (first..s.boxes.len() as u32).collect();
    s.begin_element(KIND_FLOOR);
    let ztop = 0.3 + h as f32;
    s.add_grid([0.0, 0.0, ztop], [3.0, 3.0, ztop + top_h], [2, 2, 1], 0, 0);
    let mut struts = vec![];
    if let Some(z) = brace_at {
        // side walls at x = -3.3..-3.0 and x = 6.0..6.3 (anchored), struts along x
        s.begin_element(KIND_EXT_WALL);
        s.add_grid([-3.3, c0, 0.0], [-3.0, c1, z + 0.1], [1, 1, 1], 0, 0);
        s.begin_element(KIND_EXT_WALL);
        s.add_grid([6.0, c0, 0.0], [6.3, c1, z + 0.1], [1, 1, 1], 0, 0);
        s.begin_element(KIND_BEAM);
        let f = s.boxes.len() as u32;
        s.add([-3.0, 1.4, z - 0.1], [c0, 1.6, z + 0.1], 0, 0);
        s.add([c1, 1.4, z - 0.1], [6.0, 1.6, z + 0.1], 0, 0);
        struts = (f..s.boxes.len() as u32).collect();
    }
    s.end_element();
    (s.build(), col, struts)
}
const F_INDESTRUCTIBLE_FLAG: u16 = rubble_core::rubble_format::F_INDESTRUCTIBLE;

/// Buckling: a slender column holding a heavy block stands while braced at mid-height and
/// fails (block comes down) once the bracing struts are destroyed.
#[test]
fn column_buckles_when_mid_height_bracing_destroyed() {
    let (bld, col, struts) = braced_column(0.3, 6, 1.0, Some(3.3));
    assert_eq!(struts.len(), 2);
    let braced = run_scenario(bld.clone(), WorldConfig::default(), &[], &col, 2.0);
    println!("braced: {braced:?}");
    assert!(braced.detached == 0 && braced.breaks == 0 && braced.trace.last().unwrap().1 < 1.0);
    let o = run_scenario(bld, WorldConfig::default(), &struts, &col, 4.0);
    println!("bracing destroyed: {o:?}");
    assert!(o.trace[0].1 > 1.0, "unbraced column must be overloaded: {o:?}");
    assert!(o.first_detach.is_some_and(|t| t < 3.0), "{o:?}");
    // without buckling the same column holds
    let mut cfg = WorldConfig::default();
    cfg.stress.buckling = false;
    let (bld, _, struts) = braced_column(0.3, 6, 1.0, Some(3.3));
    let nb = run_scenario(bld, cfg, &struts, &col, 2.0);
    assert!(nb.first_detach.is_none(), "{nb:?}");
}

/// Tipping: an 8 m tall 6×4 m block standing on a 1 m wide anchored strip under one edge
/// (centroid 2.5 m outside the support) breaks the far side of its support and comes down;
/// the same block on a centred strip stands.
fn block_on_strip(strip_x: f32) -> Bld {
    let mut s = SynthBuilder::new("block_on_strip");
    s.add_grid([strip_x, 0.0, 0.0], [strip_x + 1.0, 4.0, 1.0], [1, 4, 1], 0, 0);
    s.add_grid([0.0, 0.0, 1.0], [6.0, 4.0, 9.0], [6, 4, 8], 0, 0);
    s.build()
}

#[test]
fn eccentric_block_tips_centered_block_stands() {
    let centred = run_scenario(block_on_strip(2.5), WorldConfig::default(), &[], &[], 3.0);
    println!("centred: {centred:?}");
    assert!(centred.detached == 0 && centred.breaks == 0, "{centred:?}");
    // the edge strip version can't even be loaded intact: build it, run, expect it to fail
    let mut cfg = WorldConfig::default();
    cfg.stress.settle_overloads_at_load = false;
    cfg.emit_edge_events = true;
    let mut w = World::new(cfg);
    let b = w.load_building_bld(block_on_strip(0.0), Isometry::identity());
    w.add_ground_plane(0.0);
    let (mut detached, mut breaks, mut first) = (0, 0, None);
    for s in 0..(3.0 / DT) as usize {
        w.step(DT);
        for e in w.drain_events() {
            match e {
                Event::ClusterDetached { chunks, .. } => {
                    detached += chunks.len();
                    first.get_or_insert(s as f32 * DT);
                }
                Event::EdgeBroken { cause: BreakCause::Stress, .. } => breaks += 1,
                _ => {}
            }
        }
    }
    let mu = w.building(b).utilization.iter().cloned().fold(0.0f32, f32::max);
    println!("eccentric: detached {detached} breaks {breaks} first {first:?} max util {mu}");
    assert!(detached >= 150 && first.is_some_and(|t| t < 2.0), "eccentric block should tip off its support");
}

/// Damage weakens supports: a heavily loaded column shot repeatedly (10 % hp per shot) gives
/// way well before its hp reaches 0.
#[test]
fn shot_column_fails_before_hp_runs_out() {
    let (bld, col, _) = braced_column(0.3, 3, 0.9, None);
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    for _ in 0..30 {
        w.step(DT);
    }
    assert!(w.drain_events().is_empty());
    let target = col[1];
    let hp0 = w.building(b).hp[target as usize];
    let u0 = w.building(b).utilization.iter().cloned().fold(0.0f32, f32::max);
    let mut failed_at = None;
    'outer: for shot in 1..10 {
        w.damage_chunk(b, target, 0.1 * hp0);
        for _ in 0..30 {
            w.step(DT);
            if w.drain_events().iter().any(|e| matches!(e, Event::ClusterDetached { chunks, .. } if chunks.len() >= 4)) {
                failed_at = Some((shot, w.building(b).hp[target as usize] / hp0));
                break 'outer;
            }
        }
    }
    println!("intact util {u0:.2}; failed at {failed_at:?} (shot, hp fraction)");
    let (_, frac) = failed_at.expect("column should give way under the load");
    assert!(frac > 0.2, "failed only at hp fraction {frac}");
}

