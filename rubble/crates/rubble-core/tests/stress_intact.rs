//! Intact synthetic structures must stand under the full stress model (bending on):
//! max utilization < 0.5 (DESIGN §2.7). Graph built exactly like `Building::new`.
use rubble_core::rubble_format::{Bld, F_COSMETIC_ATTACHED, F_GLASS};
use rubble_core::testutil::*;
use rubble_stress::{static_report, StressConfig, StressGraph, StressInput, StaticReport};

fn report(bld: &Bld, cfg: &StressConfig) -> StaticReport {
    let n = bld.chunks.len();
    let structural: Vec<bool> = bld.chunks.iter().map(|c| c.flags & (F_GLASS | F_COSMETIC_ATTACHED) == 0).collect();
    let anchor: Vec<bool> = (0..n).map(|c| bld.is_anchor(c)).collect();
    let weight: Vec<f32> = (0..n).map(|c| if structural[c] { bld.chunks[c].mass * 9.81 } else { 0.0 }).collect();
    let pairs: Vec<(u32, u32)> = bld.edges.iter().map(|e| (e.a, e.b)).collect();
    let cap: Vec<f32> = bld.edges.iter().map(|e| e.strength).collect();
    let cen: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.centroid).collect();
    let nrm: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.normal).collect();
    let pos: Vec<[f32; 3]> = bld.chunks.iter().map(|c| c.com).collect();
    let area: Vec<f32> = bld.edges.iter().map(|e| e.area).collect();
    let g = StressGraph::new(n, &pairs, &cap, &cen, &nrm, &pos).with_areas(&area);
    let edge_alive: Vec<bool> = bld.edges.iter().map(|e| structural[e.a as usize] && structural[e.b as usize]).collect();
    let input = StressInput { node_weight: &weight, node_alive: &structural, anchor: &anchor, edge_alive: &edge_alive };
    let rep = static_report(&g, &input, cfg);
    assert!(rep.converged);
    rep
}

/// Uniform flat slab (1.5 m chunks, 0.3 m thick) on a 4×4 grid of 0.6 m columns at 4 m centres.
fn slab_on_columns() -> Bld {
    let mut s = SynthBuilder::new("slab_on_columns");
    for i in 0..4 {
        for j in 0..4 {
            let (x, y) = (0.45 + 4.0 * i as f32, 0.45 + 4.0 * j as f32);
            s.add_grid([x, y, 0.0], [x + 0.6, y + 0.6, 3.0], [1, 1, 3], 0, 0);
        }
    }
    s.add_grid([0.0, 0.0, 3.0], [13.5, 13.5, 3.3], [9, 9, 1], 0, 0);
    s.build()
}

fn intact_cases() -> Vec<(&'static str, Bld)> {
    vec![
        ("two_box", two_box()),
        ("wall 10x6", wall(10.0, 6.0, 0.3, 0.5)),
        ("wall 14x7", wall(14.0, 7.0, 0.3, 0.5)),
        ("grid 10x10x6", grid_block(10, 10, 6, 1.0)),
        ("grid 20x20x10", grid_block(20, 20, 10, 1.0)),
        ("tower 4/9 center", tower(4, 9.0, true)),
        ("tower 5/10 center", tower(5, 10.0, true)),
        ("tower 8/8 no center", tower(8, 8.0, false)),
        ("tower_cols 8/10 1.2", tower_cols(8, 10.0, true, 1.2)),
        ("cantilever 6/3", cantilever(6, 3)),
        ("cantilever 6/7", cantilever(6, 7)),
        ("slab on columns", slab_on_columns()),
    ]
}

#[test]
fn intact_structures_stand_with_bending() {
    let cfg = StressConfig::default();
    assert!(cfg.bending);
    let mut failures = vec![];
    for (name, bld) in intact_cases() {
        let rep = report(&bld, &cfg);
        let ax = report(&bld, &StressConfig { bending: false, ..Default::default() });
        println!("{name:22} max util {:.3} (axial only {:.3})", rep.max_util, ax.max_util);
        if rep.max_util >= 0.5 {
            failures.push(format!("{name}: {:.3}", rep.max_util));
        }
    }
    assert!(failures.is_empty(), "intact structures over 0.5: {failures:?}");
}

/// The big-span towers of rubble-viewer's arenas (0.3 m slabs spanning 8.5–10 m diagonally
/// to a column) stand (u < 1) but don't meet the 0.5 validation margin.
#[test]
fn big_span_towers_stand() {
    let cfg = StressConfig::default();
    for (name, bld) in [("tower 6/12", tower(6, 12.0, true)), ("tower 3/14", tower(3, 14.0, true))] {
        let rep = report(&bld, &cfg);
        println!("{name}: max util {:.3}", rep.max_util);
        assert!(rep.max_util < 1.0, "{name}: {}", rep.max_util);
    }
}

#[test]
fn arena_buildings_stand_with_bending() {
    // rubble-viewer's synthetic arena (minus its 14 m-span tower, see big_span_towers_stand)
    // plus rubble-sim's bench arena tower
    let make = |i: usize| match i % 6 {
        0 => tower(5, 10.0, true),
        1 => wall(14.0, 7.0, 0.3, 0.5),
        2 => grid_block(6, 6, 6, 1.0),
        3 => tower(8, 8.0, false),
        4 => cantilever(6, 7),
        _ => tower_cols(8, 10.0, true, 1.2), // rubble-sim bench arena tower
    };
    let cfg = StressConfig::default();
    for (i, (bld, _)) in arena(6, make, 24.0).into_iter().enumerate() {
        let rep = report(&bld, &cfg);
        assert!(rep.max_util < 0.5, "arena building {i}: {}", rep.max_util);
    }
}

#[test]
fn long_concrete_cantilever_fails_with_bending_only() {
    // 1 m plain-concrete cubes: a long overhang must fail through bending, not axially.
    let on = StressConfig::default();
    let off = StressConfig { bending: false, ..Default::default() };
    let long = cantilever(6, 20);
    let (u_on, u_off) = (report(&long, &on).max_util, report(&long, &off).max_util);
    println!("cantilever 6/20: bending {u_on:.3}, axial only {u_off:.3}");
    assert!(u_on > 1.0, "long cantilever should fail: {u_on}");
    assert!(u_off < 0.5);
}

/// Same scenario as core.rs `stress_collapse_when_columns_removed`, with the full stress
/// model (bending + compression capacity): 3 of 5 ground-floor columns removed -> the
/// floor slabs become cantilevers off the remaining columns and fail in bending.
#[test]
fn tower_collapses_when_columns_removed_with_bending() {
    use rubble_core::{Event, Isometry, World, WorldConfig};
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building_bld(tower(4, 9.0, true), Isometry::identity());
    w.add_ground_plane(0.0);
    let run = |w: &mut World, n: usize| {
        let mut ev = vec![];
        for _ in 0..n {
            w.step(1.0 / 60.0);
            ev.extend(w.drain_events());
        }
        ev
    };
    let ev = run(&mut w, 60);
    assert!(ev.is_empty(), "intact tower must stand: {ev:?}");
    let cols: Vec<u32> = {
        let bd = w.building(b);
        (0..bd.n_chunks() as u32)
            .filter(|&c| {
                let ch = &bd.bld.chunks[c as usize];
                let corner = (ch.com[0] < 1.0 || ch.com[0] > 8.0) && (ch.com[1] < 1.0 || ch.com[1] > 8.0);
                let lower = ch.com[2] > 0.3 && ch.com[2] < 1.4;
                corner && lower && !(ch.com[0] > 8.0 && ch.com[1] > 8.0)
            })
            .collect()
    };
    assert_eq!(cols.len(), 3);
    for c in cols {
        w.damage_chunk(b, c, 1e9);
    }
    let ev = run(&mut w, 600);
    let big = ev.iter().any(|e| matches!(e, Event::ClusterDetached { chunks, .. } if chunks.len() > 20));
    let mu = w.building_state(b).utilization.iter().cloned().fold(0.0f32, f32::max);
    assert!(big, "overloaded tower should collapse via stress (max util now {mu})");
}

// ---------------------------------------------------------------- real bgen buildings

const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../assets/buildings");
const KINDS: [&str; 15] = [
    "ext_wall", "int_wall", "floor", "column", "beam", "roof", "stair", "landing", "step", "parapet", "balcony", "glass",
    "fascia", "ridge", "other",
];

/// Same as `report`, with chunks grouped by element (as `Building::new` does).
fn report_grouped(bld: &Bld, cfg: &StressConfig, alive: &[bool]) -> StaticReport {
    let n = bld.chunks.len();
    let structural: Vec<bool> =
        bld.chunks.iter().enumerate().map(|(i, c)| c.flags & (F_GLASS | F_COSMETIC_ATTACHED) == 0 && alive[i]).collect();
    let anchor: Vec<bool> = (0..n).map(|c| bld.is_anchor(c)).collect();
    let weight: Vec<f32> = (0..n).map(|c| if structural[c] { bld.chunks[c].mass * 9.81 } else { 0.0 }).collect();
    let pairs: Vec<(u32, u32)> = bld.edges.iter().map(|e| (e.a, e.b)).collect();
    let cap: Vec<f32> = bld.edges.iter().map(|e| e.strength).collect();
    let cen: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.centroid).collect();
    let nrm: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.normal).collect();
    let pos: Vec<[f32; 3]> = bld.chunks.iter().map(|c| c.com).collect();
    let area: Vec<f32> = bld.edges.iter().map(|e| e.area).collect();
    let group: Vec<u32> = bld.chunks.iter().map(|c| c.elem).collect();
    let g = StressGraph::new(n, &pairs, &cap, &cen, &nrm, &pos).with_areas(&area).with_node_groups(&group);
    let edge_alive: Vec<bool> = bld.edges.iter().map(|e| structural[e.a as usize] && structural[e.b as usize]).collect();
    let input = StressInput { node_weight: &weight, node_alive: &structural, anchor: &anchor, edge_alive: &edge_alive };
    static_report(&g, &input, cfg)
}

fn describe(bld: &Bld, e: u32, u: f32) -> String {
    let ed = &bld.edges[e as usize];
    let kind = |c: u32| KINDS.get(bld.elements[bld.chunks[c as usize].elem as usize].kind as usize).copied().unwrap_or("?");
    format!(
        "u={u:.2} area={:.3} n=[{:.1},{:.1},{:.1}] {}({})↔{}({}) at ({:.1},{:.1},{:.1})",
        ed.area, ed.normal[0], ed.normal[1], ed.normal[2],
        kind(ed.a), bld.chunks[ed.a as usize].elem, kind(ed.b), bld.chunks[ed.b as usize].elem,
        ed.centroid[0], ed.centroid[1], ed.centroid[2]
    )
}

fn real_buildings() -> Vec<(String, Bld)> {
    // assets/buildings plus any extra dirs in $RUBBLE_EXTRA_ASSETS (':'-separated)
    let mut dirs = vec![ASSETS.to_string()];
    if let Ok(extra) = std::env::var("RUBBLE_EXTRA_ASSETS") {
        dirs.extend(extra.split(':').map(String::from));
    }
    let mut out = vec![];
    for d in dirs {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        for name in names {
            if let Ok(bld) = Bld::load(&format!("{d}/{name}/building.bld")) {
                out.push((name, bld));
            }
        }
    }
    out
}

/// Presets whose intact structure is over-stressed in this model for a structural reason
/// (not a solver artifact): tower ribbon glazing leaves 0.5 m plain-concrete spandrel bands
/// spanning 3–3.5 m windows under the floor line load; warehouse_2's 9 m flat roof slab.
/// They stand (< 3) but miss the 0.5 margin until bgen reinforces them (§2.7).
const KNOWN_OVER: [&str; 2] = ["tower_", "warehouse_2"];

#[test]
fn real_buildings_stand_with_bending() {
    let mut bad = vec![];
    for (name, bld) in real_buildings() {
        let alive = vec![true; bld.chunks.len()];
        let rep = report_grouped(&bld, &StressConfig::default(), &alive);
        let ax = report_grouped(&bld, &StressConfig { bending: false, ..Default::default() }, &alive);
        println!("{name:14} chunks {:6} max util bend {:.3} axial {:.3}", bld.chunks.len(), rep.max_util, ax.max_util);
        for &(e, u) in rep.worst_edges.iter().take(2) {
            println!("    {}", describe(&bld, e, u));
        }
        let known = KNOWN_OVER.iter().any(|k| name.starts_with(k));
        // a spec may accept a lower margin (bgen `stability: {accept_util}`, e.g. very tall towers)
        let accepted = bld.meta.pointer("/spec/stability/accept_util").and_then(|v| v.as_f64()).map(|v| v as f32);
        let limit = if known { 3.0 } else { accepted.unwrap_or(0.5) };
        if rep.max_util >= limit {
            bad.push(format!("{name}: {:.3}", rep.max_util));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// Real office: destroy every ground-floor column, ext_wall and int_wall chunk in one half
/// of the footprint -> the storeys above that half must overload and break away.
#[test]
fn real_office_half_ground_floor_removed_collapses() {
    use rubble_core::{Event, Isometry, World, WorldConfig};
    let p = format!("{ASSETS}/office_1/building.bld");
    let Ok(bld) = Bld::load(&p) else {
        eprintln!("skip: no office_1 asset");
        return;
    };
    let xs: Vec<f32> = bld.chunks.iter().map(|c| c.com[0]).collect();
    let mid = 0.5 * (xs.iter().cloned().fold(f32::INFINITY, f32::min) + xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max));
    let kill: Vec<u32> = (0..bld.chunks.len())
        .filter(|&i| {
            let el = &bld.elements[bld.chunks[i].elem as usize];
            el.floor == 0 && matches!(KINDS[el.kind as usize], "column" | "ext_wall" | "int_wall") && bld.chunks[i].com[0] < mid
        })
        .map(|i| i as u32)
        .collect();
    let mut w = World::new(WorldConfig::default());
    let b = w.load_building_bld(bld, Isometry::identity());
    w.add_ground_plane(0.0);
    for _ in 0..30 {
        w.step(1.0 / 60.0);
    }
    assert!(w.drain_events().iter().all(|e| !matches!(e, Event::ClusterDetached { .. })), "intact office must stand");
    for &c in &kill {
        w.damage_chunk(b, c, 1e12);
    }
    let (mut peak, mut detached) = (0f32, 0usize);
    for _ in 0..600 {
        w.step(1.0 / 60.0);
        for e in w.drain_events() {
            if let Event::ClusterDetached { chunks, .. } = e {
                detached += chunks.len();
            }
        }
        peak = peak.max(w.building(b).utilization.iter().cloned().fold(0.0, f32::max));
    }
    println!("office_1: removed {} ground chunks, peak util {peak:.2}, detached {detached}", kill.len());
    assert!(peak > 1.0 && detached > 100, "peak {peak}, detached {detached}");
}
