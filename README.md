# PhoxelVase

Procedurally generated, pre-fractured buildings and a real-time destruction engine to knock them down.

![office_1 demolished with X](recordings/office_1.gif)
![office_xcoarse_2 demolished with X](recordings/office_xcoarse_2.gif)

- **`bgen/`** (Python) generates buildings from YAML specs. Each building is a layout of walls, floors,
  columns, stairs, openings and roofs, cut into convex chunks joined by a graph of bonded joints.
  It is checked for static stability before export as `.bld` (plus `.glb` and renders).
- **`rubble/`** (Rust) runs the destruction. Damage breaks chunks and joints. A stress solver on the
  joint graph decides what gives way. Anything no longer connected to the ground, or tipping off its
  support, falls as rigid bodies (rapier) and settles into rubble.
  Crates: `rubble-core` (engine), `rubble-stress` (solver), `rubble-format` (`.bld`),
  `rubble-viewer` (Bevy app), `rubble-sim` (headless runs and benchmarks), `rubble-py` (Python bindings).

Building presets: office, apartment, warehouse, tower, house, kyoto. Free-form structures: eiffel,
suspension_bridge, aqueduct, colossus. `assets/buildings/` holds three seeds of each, plus coarse and xcoarse fracture variants.
`docs/explainer.html` is an illustrated walkthrough.

## Run the viewer
```sh
cd rubble && cargo run --release -p rubble-viewer -- ../assets/buildings/office_1/building.bld
```
Press H in the viewer for the controls. `recordings/` has a demolition GIF for every office variant.

## Tests
```sh
.venv/bin/python -m pytest -q bgen/tests rubble/crates/rubble-py/tests
cd rubble && cargo test --release --workspace --exclude rubble-viewer
cd rubble && cargo test --release -p rubble-core --test fuzz_hang -- --ignored   # randomized "nothing floats" check
```
