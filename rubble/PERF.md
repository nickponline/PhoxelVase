# rubble performance notes

Machine: Apple M3 Max (12 P + 4 E cores), macOS, release build (`debug = 1`, thin LTO).
Harness: `stages.py`-style runner (scenario actions via `bgen.sim`, `World.step(1/60)` back to back,
per-stage times from `stats()['timings']`, SHA1 of the final `chunk_world_transforms` of every
building, total event count). Before/after numbers are medians of 3 runs, **interleaved** baseline/new
on the same machine state (the baseline is a saved copy of the old `rubble.abi3.so`).

All optimizations are **bit-identical**: same final-state hash and event count for all six
scenarios (each run twice), identical `random_play.py office_1,apartment_1 3` output, identical
hashes for any physics thread count (1, 3, 4, 6, 8, 12, 16), and all tests pass
(`cargo test --release --workspace --exclude rubble-viewer`, `pytest bgen/tests rubble/crates/rubble-py/tests`).

## Baseline (step 0, before any change)

| scenario | hash | events | avg ms | p99 ms | max ms | per-stage avg ms |
|---|---|---|---|---|---|---|
| two_box_drop | f0a8e12da2 | 6 | 0.01 | 0.22 | 0.40 | physics 0.009 |
| office_columns_out | e3a7556446 | 11088 | 3.09 | 6.00 | 10.0 | physics 2.38, impacts 0.63, stress 0.047, settle 0.017, impact_wait 0.014, promotion 0.012 |
| office_top_blast | 88909b55e0 | 622 | 0.42 | 1.07 | 6.24 | physics 0.366, impacts 0.033, stress 0.014 |
| tower_columns_out | b1609a9bff | 13661 | 2.14 | 5.12 | 10.4 | physics 1.74, impacts 0.33, stress 0.044, settle 0.013, promotion 0.012 |
| apartment_ground_out | 5b3d5cb63f | 10703 | 2.67 | 5.57 | 7.8 | physics 2.14, impacts 0.44, stress 0.054, settle 0.016 |
| building4_shaft_cut | cb0ec3b62e | 10086 | 13.96 | 27.2 | 82–87 | physics 7.26, impacts 5.14, stress 1.50, impact_wait 0.105, promotion 0.028 |

Repeated runs gave identical hash and event count. Idle (building4_4 + office_1 loaded, 600 ticks,
no damage): avg 0.076 ms, p99 0.098 ms (first tick 21 ms: initial broad-phase build).
`rubble-sim bench`: load arena 126 ms, first tick 40.0 ms, steady avg 0.382 / p99 0.678 ms,
collapse avg 1.99 / p99 5.73 / max 30.0 ms.

## After

Interleaved A/B, 3 runs each (baseline re-measured in the same session):

| scenario | avg ms (base → new) | p99 ms | max ms | physics | impacts | stress |
|---|---|---|---|---|---|---|
| two_box_drop | 0.01 → 0.01 | 0.04 → 0.05 | 0.10 → 0.09 | ~0 | – | – |
| office_columns_out | 2.88 → **1.98 (−31%)** | 5.39 → 3.70 | 9.8 → 12.3* | 2.25 → 1.58 | 0.56 → 0.34 | 0.043 → 0.028 |
| office_top_blast | 0.42 → **0.27 (−36%)** | 1.07 → 0.72 | 7.0 → 6.1 | 0.37 → 0.23 | 0.031 → 0.023 | 0.013 → 0.012 |
| tower_columns_out | 2.01 → **1.41 (−30%)** | 5.15 → 3.67 | 10.5 → 12.1* | 1.65 → 1.18 | 0.29 → 0.19 | 0.040 → 0.024 |
| apartment_ground_out | 2.48 → **1.72 (−31%)** | 4.82 → 3.53 | 7.7 → 5.7 | 2.02 → 1.44 | 0.39 → 0.24 | 0.037 → 0.023 |
| building4_shaft_cut | 12.60 → **6.66 (−47%)** | 24.9 → **13.1 (−47%)** | 82.7 → 27.7 | 6.70 → 4.01 | 4.45 → 1.50 | 1.38 → 1.11 |

impact_wait (building4) 0.116 → 0.038 ms; promotion 0.024 → 0.013 ms; settle 0.015 → 0.008 ms.
Hashes/events unchanged everywhere.

\* office/tower max: the worst tick is a tick that blocks on a background landing solve
(`impact_wait`). The solve takes the same ~17 ms, but it is due `impact_latency_ticks` (6) ticks
after launch, and back-to-back ticks are now faster, so less of it is hidden. At a real 60 Hz
tick rate (100 ms for 6 ticks) there is no wait in either version.

Idle (600 ticks, two buildings): avg 0.063 → 0.063 ms, p99 0.046 → 0.047 ms (unchanged).
`rubble-sim bench` (2 runs each): load arena 116–119 → 109–116 ms, first tick 37 → 37 ms,
steady avg 0.36–0.39 → 0.32 ms, collapse avg 1.80 → 1.28–1.31 ms, collapse p99 5.1–5.3 → 3.0–3.2 ms,
collapse max 27–28 → 10.2 ms (same events and end state).

## Where the time was (profiles: `sample` on rubble-sim / the Python harness)

- building4: rebuilding the ~30k-part compound of the falling tower every time a piece broke off
  (255 rebuilds, ~9 ms each: per-part AABB + binned BVH build + CCD thickness), the ground
  half-space generating one manifold per compound part every step, rebuild bookkeeping (BFS,
  edge lists), a 37 ms thaw query burst when the tower detached (32k support boxes), 100–150 ms
  landing solves the tick blocked on, and a global stress re-solve every tick.
- office/tower/apartment: dominated by the Rapier step; the global rayon pool (16 threads incl.
  E-cores) spent much of it in fork/join wake-ups and spinning.

## Optimizations (individual gains from A/B steps during the work; noisy ±5%)

1. **Physics on a dedicated 6-thread pool** (`WorldConfig::physics_threads`, 0 = auto =
   min(6, cores/2), env `RUBBLE_PHYSICS_THREADS`; `physics.rs`). Rapier's results do not depend
   on the thread count (verified 1–16). office physics 2.22 → 1.69 ms (−24%), tower −22%,
   apartment −24%, building4 −4%. 6 beat 4/8/12/16 on this machine. Idle ticks with no active
   body step on the caller thread (the pool round trip cost ~20 µs).
2. **Vendored parry3d: parallel compound construction** (`vendor/parry3d`, see below): part AABBs in
   parallel, binned BVH built in parallel with the identical node layout, parallel CCD-thickness
   min. building4 impacts 5.2 → 3.25 ms.
3. **Cached compound part data**: per-chunk `compute_aabb(identity)` and `ccd_thickness()` are
   computed once at load (`Building::shape_aabb/shape_ccd`) and passed to
   `Compound::new_with_part_data`. building4 impacts 2.08 → 1.52 ms; office impacts 0.48 → 0.35.
4. **Composite vs half-space: skip far parts** (parry): a half-space's AABB is all of space, so every
   part of every cluster got a manifold vs the ground each step. Parts provably beyond the
   prediction distance (part AABB vs plane, 1 cm margin) get exactly the state the general code
   leaves (no points, same normals) without computing support features. building4 physics
   6.7 → 5.9 ms.
5. **Composite manifolds: reuse when the query reaches the same leaves** (parry): skip the
   per-leaf hash-map bookkeeping when the leaf list equals the previous call's (always true for
   the ground pair of a resting compound). building4 physics 5.27 → 4.33 ms.
6. **Cluster rebuild**: one pass for the parent's edge list instead of two, compact `edge_ab`,
   reused scratch, compound parts moved instead of cloned twice, big removed compounds dropped
   on a background thread (each drop = 30k cache-missing `Arc` decrements). building4 impacts
   3.25 → 2.70 ms, p99 25 → 19 ms.
7. **"Still one piece" shortcut** in `rebuild_cluster`: a cluster is connected after every full
   rebuild; if it lost no chunk and both ends of every edge broken since then are still
   connected (bidirectional BFS, capped), the full walk would change nothing, so skip it
   (`Cluster::broken/connected`, `Building::still_connected`). 158 of 413 giant rebuilds in
   building4. ~−0.1–0.2 ms/tick there.
8. **Impact edge check**: only edges that received load this tick can cross the break threshold
   (loads only decay afterwards), so check those instead of every edge of every touched
   cluster. building4 impacts −0.28 ms.
9. **Landing solves on a persistent 4-thread pool** (no OS thread per job) with parallel CG
   kernels for ≥4096 rows (CG reductions use fixed chunks: bit-identical to sequential).
   building4 giant jobs 150/100 ms → 63/46 ms; max tick 92 → 60 ms; impact_wait 0.17 → 0.02 ms.
10. **Thaw after big removals**: support-lost AABBs cached at collider insertion (no recompute on
    removal); the box queries are skipped when there is no sleeping body / no frozen chunk, and
    otherwise pre-tested hierarchically over Morton-sorted groups (a box finds nothing its group's
    box does not; boxes are still processed in the original order); wake + seed collection share
    one traversal. Collapse tick of building4: thaw 37 → 2.6 ms (max tick 59 → 33 ms).
11. **Stress**: `solve_step_lean` (no per-tick clone of the 126k-edge utilization vector; copied into
    the existing buffer), `any_overloaded()` from the hot list instead of a full scan, the second
    utilization refresh skipped when it would recompute the same edges (full moment update after
    a global solve), the full moment update sorts integer keys of the `total_cmp` order (same
    order, unit-tested), compact `edge_ab`/`hp_max` reads in `solve_building`. building4 stress
    1.41 → 1.16 ms; office 0.045 → 0.031 ms.
12. **Contacts / impacts plumbing**: pairs with no point above the impulse threshold are skipped
    before any collider lookup; reused hits buffer; the hits sort compares one packed integer
    (same comparison outcomes, same algorithm, so the same permutation); the
    `RUBBLE_DEBUG_IMPACT` env lookup (took a lock every tick) is read once. office impacts
    0.64 → 0.60 ms.
13. Load: hull AABB/CCD data computed in the same parallel pass that builds the hulls (load time
    unchanged: building4_4 load 493 → 486 ms median).

### Vendored parry3d

`rubble/vendor/parry3d` is parry3d 0.31.1 (Apache-2.0) with examples/tests/dev-deps stripped,
wired in with `[patch.crates-io]` in `rubble/Cargo.toml`. Changes (see
`diff -ru ~/.cargo/registry/src/*/parry3d-0.31.1/src rubble/vendor/parry3d/src`):

- `partitioning/bvh/bvh_binned_build.rs`, `bvh_tree.rs`, `utils/vec_map.rs`: split/partition code
  factored into `binned_split` (unchanged); `rebuild_binned_parallel` builds both halves of
  ranges ≥ 4096 leaves with `rayon::join` into precomputed slot ranges (a subtree with k leaves
  uses k−1 nodes, allocated depth-first), so nodes/parents/leaf indices are identical;
  `Bvh::from_leaves_sequential` (doc-hidden) for the equality test
  (`rubble-core/tests/vendored_parry.rs`).
- `shape/compound.rs`, `shape/shape.rs`: part AABBs in parallel (≥ 1024 parts);
  `Compound::new_with_part_data` (caller-supplied part AABBs and CCD min); parallel / cached
  `ccd_thickness`.
- `query/contact_manifolds/contact_manifolds_composite_shape_shape.rs`,
  `contact_manifolds_halfspace_pfm.rs`, `query_dispatcher.rs`, `default_query_dispatcher.rs`:
  the far-from-half-space fast path (convex polyhedron parts, default dispatcher only) and the
  same-leaves manifold reuse (off when `serde-serialize` is enabled).

rubble-core calls `Compound::new_with_part_data`, so it needs the vendored copy; to go back to
stock parry, drop the `[patch]` line and build compounds with `SharedShape::compound` again
(`physics.rs::compound_of`).

## Rejected / not applied

Bit-identical but slower:
- CG kernels of landing solves on the **global** rayon pool: contended with the physics step;
  building4 max tick 85 → 200 ms. (Kept with a dedicated pool, item 9.)
- Physics pool sizes 4/8/12/16 and the global pool: all slower than 6 here (office avg 2.17–2.69
  vs 2.01–2.12 ms; building4 6.96–8.23 vs 6.71–6.94 ms).
- Thaw pre-test with flat groups of 8/32/128 boxes (13 / 7.9 / 5.5 ms on the collapse tick) —
  replaced by the hierarchical pre-test (2.6 ms).

Would change results (not applied):
- **Finite ground instead of a half-space** (a 4 km × 4 km × 10 m cuboid): building4 avg
  6.67 → 4.92 ms (physics 4.0 → 2.4 ms), office 1.98 → 2.08 ms (slower); different hash/events
  (contact sets and manifold order change).
- **Stop the perpetual global stress re-solve**: after building4 collapses the stress solve never
  converges (the f32 CG stalls above `tol`), so the building stays `stress_active` and re-solves
  globally every tick: ~1.1 ms/tick (the whole remaining stress stage) for no visible effect.
  Fixing it changes the warm-started potentials/utilizations.
- **Keep the giant compound's BVH and tombstone removed parts** (or split huge clusters into
  several bodies) instead of rebuilding a 30k-part compound per split: would remove most of the
  remaining impacts stage (~1 ms/tick in building4) but changes part order / BVH layout, hence
  contact order.
- Lazily applied `edge_impulse` decay (one scale factor per cluster): changes float rounding.

## Determinism hazard (pre-existing, not changed)

Two stress settings are wall-clock based: `stress.budget_ms` (iterations per tick shrink when a
stress solve takes > 1.5 ms) and `stress.catchup_max_ms`. The six scenarios happen to be invariant
to the iteration budget (same hashes with `budget_ms` 0.01 and 1000), but `random_play.py
apartment_1` seed 0 is not: `budget_ms = 0.01` changes its outcome, and one baseline-equivalent run
under heavy machine load (load average ~44) produced that different outcome. A work-based budget
(e.g. CG iterations × rows) would make results independent of machine load.
