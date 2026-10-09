# Vendored rapier3d 0.36.0

Unmodified crates.io rapier3d 0.36.0 (`src/`, `README.md`, `Cargo.toml` without its example and
test targets), except for one change:

`src/pipeline/physics_pipeline/mod.rs`, `PhysicsPipeline::join_deferred_bvh_optimize`: the join
of the deferred broad-phase optimization task waits with `try_recv` + `rayon::yield_now()`
instead of a blocking `rx.recv()`. The task is `rayon::spawn`ed into the pool the step runs in;
a blocking `recv` parks that pool worker, and with several worlds stepping in a shared pool
(rubble's physics pool is shared between worlds) every worker could end up parked or waiting
on a parked one: a rare deadlock. Waiting the rayon way lets the waiting worker run pending
pool work, including the task itself. Results are unchanged (same task, same data).

Regression test: `crates/rubble-core/tests/concurrent_worlds.rs`.
