//! cargo bench -p rubble-stress
//! 100k-node / ~300k-edge lattice (47×47×46, ground layer anchored).
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rubble_stress::testgraphs::grid_building;
use rubble_stress::*;

const NX: usize = 47;
const NZ: usize = 46;

fn node(x: usize, y: usize, z: usize) -> usize {
    (z * NX + y) * NX + x
}

fn bench(c: &mut Criterion) {
    let t = grid_building(NX, NX, NZ, 23544.0, 5.0e6);
    eprintln!("graph: {} nodes, {} edges", t.n_nodes(), t.n_edges());

    c.bench_function("build_100k", |b| {
        b.iter(|| StressGraph::new(t.n_nodes(), &t.edges, &t.capacity, &t.centroid, &t.normal, &t.node_pos))
    });

    let g = t.graph();
    for bending in [false, true] {
        let single = StressConfig { bending, par_threshold: usize::MAX, ..Default::default() };
        let full = StressConfig { max_iters: 100_000, ..single.clone() };
        let mut st = StressState::new(&g);
        let r = solve_step(&g, &mut st, &t.input(), &full, 0.0);
        assert!(r.converged);
        let tag = if bending { "bend" } else { "axial" };

        // 1) a bullet removes one chunk near the ground (largest redistribution per chunk)
        let mut t1 = t.clone();
        t1.node_alive[node(NX / 2, NX / 2, 1)] = false;
        c.bench_function(&format!("warm_step_remove_ground_node_{tag}"), |b| {
            b.iter_batched(
                || st.clone(),
                |mut s| solve_step(&g, &mut s, &t1.input(), &single, 1.0 / 60.0),
                BatchSize::LargeInput,
            )
        });
        // 2) an explosion breaks the bonds of a 3×3×3 block mid-building
        let mut t2 = t.clone();
        for z in 20..23 {
            for y in 22..25 {
                for x in 22..25 {
                    t2.node_alive[node(x, y, z)] = false;
                }
            }
        }
        c.bench_function(&format!("warm_step_remove_block27_{tag}"), |b| {
            b.iter_batched(
                || st.clone(),
                |mut s| solve_step(&g, &mut s, &t2.input(), &single, 1.0 / 60.0),
                BatchSize::LargeInput,
            )
        });
        // 3) no change (steady state)
        c.bench_function(&format!("warm_step_nochange_{tag}"), |b| {
            b.iter_batched(
                || st.clone(),
                |mut s| solve_step(&g, &mut s, &t.input(), &single, 1.0 / 60.0),
                BatchSize::LargeInput,
            )
        });
        // 4) one global 30-iteration step (first tick after load), single vs rayon
        for (name, cfg) in [
            ("single", single.clone()),
            ("rayon", StressConfig { par_threshold: 32768, ..single.clone() }),
        ] {
            c.bench_function(&format!("cold_global_step30_{tag}_{name}"), |b| {
                b.iter_batched(
                    || StressState::new(&g),
                    |mut s| solve_step(&g, &mut s, &t.input(), &cfg, 1.0 / 60.0),
                    BatchSize::LargeInput,
                )
            });
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20).measurement_time(std::time::Duration::from_secs(3)).warm_up_time(std::time::Duration::from_secs(1));
    targets = bench
}
criterion_main!(benches);
