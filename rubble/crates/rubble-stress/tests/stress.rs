use rubble_stress::testgraphs::*;
use rubble_stress::*;

fn cfg_full() -> StressConfig {
    StressConfig { max_iters: 10_000, tol: 1e-4, ..Default::default() }
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * b.abs().max(1e-9)
}

#[test]
fn column_flow_equals_weight_above() {
    let n = 12;
    let t = column(n, 2.0, 100.0);
    let g = t.graph();
    let mut st = StressState::new(&g);
    let r = solve_step(&g, &mut st, &t.input(), &cfg_full(), 1.0 / 60.0);
    assert!(r.converged);
    for e in 0..n as u32 {
        // edge e joins node e (below) and e+1 (above): load flows downward = b→a, negative
        let above = (n - e as usize) as f64 * 2.0;
        let f = -st.edge_flow(&g, e);
        assert!(close(f, above, 1e-3), "edge {e}: flow {f} vs {above}");
        // downward load through a vertical contact is bearing: capacity × compression_factor
        assert!(close(r.utilization[e as usize] as f64, above / 1000.0, 1e-3));
    }
}

#[test]
fn two_columns_split_load_evenly() {
    let (t, [ea, eb]) = two_columns_slab(4, 3, 1.0, 1000.0);
    let g = t.graph();
    let mut st = StressState::new(&g);
    let r = solve_step(&g, &mut st, &t.input(), &StressConfig { bending: false, ..cfg_full() }, 0.0);
    assert!(r.converged);
    let (fa, fb) = (st.edge_flow(&g, ea), st.edge_flow(&g, eb));
    // slab has 7 cubes of weight 1 -> 3.5 each; flow is top(column)->slab orientation, so negative
    assert!(close(-fa, 3.5, 1e-3) && close(-fb, 3.5, 1e-3), "{fa} {fb}");
    assert!(close(r.utilization[ea as usize] as f64, r.utilization[eb as usize] as f64, 1e-4));
}

#[test]
fn table_leg_removal_hysteresis() {
    // 3x3 slab, weight 1 each => 9 N; legs cap 0.5 (×10 in compression) => 2.25/5 = 0.45
    // intact, 9/5 = 1.8 on one leg
    let tb = table(3, 2, 1.0, 1000.0, 0.5);
    let mut t = tb.g.clone();
    let g = t.graph();
    let cfg = StressConfig { hold_time: 0.25, ..Default::default() };
    let dt = 1.0 / 60.0;
    let mut st = StressState::new(&g);
    let mut r = solve_step(&g, &mut st, &t.input(), &cfg, dt);
    for _ in 0..5 {
        r = solve_step(&g, &mut st, &t.input(), &cfg, dt);
    }
    assert!(r.converged);
    let u0 = r.utilization[tb.leg_top_edges[3] as usize];
    assert!((u0 - 0.45).abs() < 1e-3, "intact leg util {u0}");
    assert!(r.to_break.is_empty());
    for leg in &tb.legs[0..3] {
        for &i in leg {
            t.node_alive[i as usize] = false;
        }
    }
    let hold_ticks = (0.25f32 / dt).round() as usize; // 15
    let mut first_break = None;
    for tick in 1..=hold_ticks + 3 {
        let r = solve_step(&g, &mut st, &t.input(), &cfg, dt);
        let u = r.utilization[tb.leg_top_edges[3] as usize];
        if tick >= 2 {
            assert!((u - 1.8).abs() < 1e-2, "tick {tick}: remaining leg util {u}");
        }
        if !r.to_break.is_empty() && first_break.is_none() {
            first_break = Some(tick);
            assert!(r.to_break.contains(&tb.leg_top_edges[3]) || r.to_break.iter().all(|&e| r.utilization[e as usize] > 1.0));
            // only remaining-leg edges are overloaded
            for &e in &r.to_break {
                let (a, b) = g.edge_nodes(e);
                assert!(tb.legs[3].contains(&a) || tb.legs[3].contains(&b), "unexpected break {e}");
            }
        }
    }
    let fb = first_break.expect("leg never broke");
    assert!(fb >= hold_ticks - 1 && fb <= hold_ticks + 1, "broke at tick {fb}, hold {hold_ticks}");
}

#[test]
fn cantilever_bending_scales_with_length_squared() {
    // returns (u with bending, u axial only) at the root edge
    let root_util = |len: usize| {
        let (t, root) = cantilever(3, len, 1.0, 100.0, 1.0);
        let g = t.graph();
        let mut st = StressState::new(&g);
        let r = solve_step(&g, &mut st, &t.input(), &cfg_full(), 0.0);
        assert!(r.converged);
        let m = st.edge_moment(&g, root);
        let expect = (len * len) as f64 / 2.0; // sum_{x=1..L} (x - 0.5)
        assert!(close(m, expect, 1e-3), "moment {m} vs {expect}");
        let mut st2 = StressState::new(&g);
        let r2 = solve_step(&g, &mut st2, &t.input(), &StressConfig { bending: false, ..cfg_full() }, 0.0);
        (r.utilization[root as usize], r2.utilization[root as usize])
    };
    let (u4, a4) = root_util(4);
    let (u8, a8) = root_util(8);
    let (u16, a16) = root_util(16);
    let (b4, b8, b16) = (u4 - a4, u8 - a8, u16 - a16);
    assert!((b8 / b4 - 4.0).abs() < 0.01 && (b16 / b8 - 4.0).abs() < 0.01, "bending ratios {} {}", b8 / b4, b16 / b8);
    assert!((a8 / a4 - 2.0).abs() < 1e-2, "axial ratio {}", a8 / a4);
    assert!(u16 / u8 > 3.0, "total ratio {}", u16 / u8);
    // long overhang fails with bending, not without
    let (u20, a20) = root_util(20);
    assert!(u20 > 1.0 && a20 < 0.5, "{u20} {a20}");
}

/// Smallest per-call iteration budget whose single solve_step gets every edge's utilization
/// within `target` of the exact answer.
fn iters_to_accuracy(g: &StressGraph, st0: &StressState, t: &TestGraph, exact: &[f32], target: f32, bending: bool) -> usize {
    for k in 1..=400 {
        let mut st = st0.clone();
        let cfg = StressConfig { max_iters: k, bending, region_max_nodes: usize::MAX, ..Default::default() };
        let r = solve_step(g, &mut st, &t.input(), &cfg, 0.0);
        let err = r.utilization.iter().zip(exact).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        if err <= target {
            return k;
        }
    }
    usize::MAX
}

#[test]
fn warm_start_far_fewer_iterations() {
    // tall tower: cold solves are expensive (condition number ~ height²)
    let base = grid_building(8, 8, 40, 1.0, 1000.0);
    let g = base.graph();
    for bending in [false, true] {
        let full = StressConfig { bending, ..cfg_full() };
        let mut warm = StressState::new(&g);
        assert!(solve_step(&g, &mut warm, &base.input(), &full, 0.0).converged);
        let cases: Vec<(&str, TestGraph)> = vec![
            ("chunk shot out at floor 10", {
                let mut t = base.clone();
                t.node_alive[(10 * 8 + 4) * 8 + 4] = false;
                t
            }),
            ("ground-floor chunk removed", {
                let mut t = base.clone();
                t.node_alive[(8 + 3) * 8 + 3] = false;
                t
            }),
            ("debris lands on floor 20", {
                let mut t = base.clone();
                for x in 2..6 {
                    t.node_weight[(20 * 8 + 2) * 8 + x] = 3.0;
                }
                t
            }),
        ];
        for (what, t) in cases {
            let rep = static_report(&g, &t.input(), &full);
            assert!(rep.converged);
            let target = 0.01 * rep.max_util; // 1% of the peak utilization
            let cold = iters_to_accuracy(&g, &StressState::new(&g), &t, &rep.utilization, target, bending);
            let warm_k = iters_to_accuracy(&g, &warm, &t, &rep.utilization, target, bending);
            println!("bending={bending} {what}: iterations to 1% accuracy: cold {cold}, warm {warm_k}");
            assert!(warm_k * 3 <= cold, "{what}: warm {warm_k} vs cold {cold}");
        }
    }
}

#[test]
fn warm_local_window_is_cheap() {
    // In a graph larger than region_max_nodes, a local change is solved on a small window:
    // the first warm tick touches a few thousand rows instead of the whole graph.
    let mut t = grid_building(24, 24, 24, 1.0, 1000.0);
    let g = t.graph();
    let cfg = StressConfig { bending: false, region_max_nodes: 2048, ..Default::default() };
    let full = StressConfig { max_iters: 10_000, ..cfg.clone() };
    let mut st = StressState::new(&g);
    let cold = solve_step(&g, &mut st, &t.input(), &full, 0.0);
    assert!(cold.converged);
    t.node_alive[(6 * 24 + 12) * 24 + 12] = false;
    let mut work = 0usize;
    let mut ticks = 0;
    loop {
        let r = solve_step(&g, &mut st, &t.input(), &cfg, 0.0);
        ticks += 1;
        work += r.iters;
        if r.converged || ticks > 200 {
            break;
        }
    }
    let rep = static_report(&g, &t.input(), &full);
    let err = st.utilization().iter().zip(&rep.utilization).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("cold global iters {} (x{} rows); warm: {ticks} ticks, {work} window iters; err {err}", cold.iters, g.n_nodes());
    assert!(ticks <= 200);
    assert!(err < 2e-3 * rep.max_util.max(1e-3) + 1e-4);
}

#[test]
fn local_mode_matches_global_solution() {
    // force local (windowed) mode with a small region cap, then compare to a static solve
    let mut t = grid_building(16, 16, 16, 1.0, 1000.0);
    let g = t.graph();
    let cfg = StressConfig { region_max_nodes: 600, max_iters: 30, tol: 1e-4, bending: true, ..Default::default() };
    let mut st = StressState::new(&g);
    let mut r = solve_step(&g, &mut st, &t.input(), &cfg, 0.0);
    let mut ticks = 0;
    while !r.converged {
        r = solve_step(&g, &mut st, &t.input(), &cfg, 0.0);
        ticks += 1;
        assert!(ticks < 1000);
    }
    // remove a ground-floor-ish 2x2 block of supports
    for (x, y) in [(7, 7), (8, 7), (7, 8), (8, 8)] {
        for z in 1..3 {
            t.node_alive[(z * 16 + y) * 16 + x] = false;
        }
    }
    let mut ticks = 0;
    loop {
        r = solve_step(&g, &mut st, &t.input(), &cfg, 0.0);
        ticks += 1;
        if r.converged {
            break;
        }
        assert!(ticks < 2000, "local mode did not converge");
    }
    let rep = static_report(&g, &t.input(), &cfg);
    let mut max_err = 0f32;
    for e in 0..g.n_edges() {
        max_err = max_err.max((r.utilization[e] - rep.utilization[e]).abs());
    }
    println!("local-mode ticks {ticks}, max util err {max_err}, max util {}", rep.max_util);
    assert!(max_err < 0.01 * rep.max_util.max(1e-3), "max err {max_err}");
}

#[test]
fn static_report_finds_unsupported() {
    let mut t = column(10, 1.0, 100.0);
    t.edge_alive[4] = false; // between node 4 and 5
    let g = t.graph();
    let rep = static_report(&g, &t.input(), &cfg_full());
    assert!(rep.converged);
    assert_eq!(rep.unsupported_nodes, (5..=10).collect::<Vec<u32>>());
    assert!(close(rep.total_load, 4.0, 1e-9));
    assert!(close(rep.anchor_flow, 4.0, 1e-3));
    assert!((rep.max_util - 0.004).abs() < 1e-5);
    assert_eq!(rep.worst_edges[0].0, 0);

    // incremental path detects the same thing
    let mut t2 = column(10, 1.0, 100.0);
    let mut st = StressState::new(&g);
    solve_step(&g, &mut st, &t2.input(), &cfg_full(), 0.0);
    t2.edge_alive[4] = false;
    let r = solve_step(&g, &mut st, &t2.input(), &cfg_full(), 0.0);
    for i in 5..=10 {
        assert!(!st.is_supported(i));
    }
    assert!(r.utilization[5..].iter().all(|&u| u == 0.0));
    assert!((r.utilization[0] - 0.004).abs() < 1e-5);
}

#[test]
fn flow_conservation_at_anchors() {
    let mut t = grid_building(10, 8, 9, 3.0, 500.0);
    // irregular damage: remove some nodes and edges
    for i in (100..t.n_nodes()).step_by(37) {
        t.node_alive[i] = false;
    }
    for e in (0..t.n_edges()).step_by(11) {
        t.edge_alive[e] = false;
    }
    let g = t.graph();
    let rep = static_report(&g, &t.input(), &cfg_full());
    assert!(rep.converged);
    println!("load {} anchor flow {} unsupported {}", rep.total_load, rep.anchor_flow, rep.unsupported_nodes.len());
    assert!(close(rep.anchor_flow, rep.total_load, 1e-3));

    // table after losing 3 legs: everything goes through one foot
    let tb = table(4, 3, 2.0, 1e4, 1e4);
    let mut t = tb.g.clone();
    for leg in &tb.legs[1..4] {
        for &i in leg {
            t.node_alive[i as usize] = false;
        }
    }
    let g = t.graph();
    let rep = static_report(&g, &t.input(), &cfg_full());
    assert!(close(rep.anchor_flow, 32.0, 1e-3), "{}", rep.anchor_flow);
}

#[test]
fn no_breaks_when_intact_and_deterministic() {
    let t = grid_building(12, 12, 10, 1.0, 1000.0);
    let g = t.graph();
    let cfg = StressConfig::default();
    let run = || {
        let mut st = StressState::new(&g);
        let mut out = Vec::new();
        for _ in 0..10 {
            out.push(solve_step(&g, &mut st, &t.input(), &cfg, 1.0 / 60.0));
        }
        out
    };
    let (a, b) = (run(), run());
    for (x, y) in a.iter().zip(&b) {
        assert!(x.to_break.is_empty());
        assert_eq!(x.utilization, y.utilization);
        assert_eq!(x.iters, y.iters);
    }
}

#[test]
fn rayon_path_bit_identical() {
    let mut t = grid_building(20, 20, 20, 1.0, 1000.0);
    let g = t.graph();
    let seq = StressConfig { par_threshold: usize::MAX, ..Default::default() };
    let par = StressConfig { par_threshold: 1, ..Default::default() };
    let (mut a, mut b) = (StressState::new(&g), StressState::new(&g));
    for tick in 0..12 {
        if tick == 4 {
            t.node_alive[(2 * 20 + 10) * 20 + 10] = false;
        }
        let ra = solve_step(&g, &mut a, &t.input(), &seq, 1.0 / 60.0);
        let rb = solve_step(&g, &mut b, &t.input(), &par, 1.0 / 60.0);
        assert_eq!(ra.iters, rb.iters);
        assert_eq!(ra.utilization, rb.utilization);
    }
}

#[test]
fn uniform_wall_has_no_bending() {
    // Regression: a uniformly supported wall/block has no net lateral load transfer, so its
    // side-by-side joints must see no moment (the old first-moment potentials gave u≈1.5).
    let t = grid_building(20, 1, 12, 1766.0, 3.0e5);
    let g = t.graph();
    let mut st = StressState::new(&g);
    let r = solve_step(&g, &mut st, &t.input(), &cfg_full(), 0.0);
    for e in 0..g.n_edges() as u32 {
        if g.is_horizontal(e) {
            assert!(st.edge_moment(&g, e) < 1e-3 * 1766.0, "edge {e}: {}", st.edge_moment(&g, e));
        }
    }
    assert!(r.utilization.iter().all(|&u| u < 0.05));
}
