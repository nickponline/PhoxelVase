# PhoxelVase

Physics engine focused on real-time destruction.

## Run the viewer
```sh
cd rubble && cargo run --release -p rubble-viewer -- ../assets/buildings/office_1/building.bld
```

Press H in the viewer for the controls.

![tower_1 demolished from the inside with Z](recordings/tower_1_z.gif)
![colossus_1 demolished with X](recordings/colossus_1_x.gif)
![apartment_1 demolished with X](recordings/apartment_1_x.gif)

- **`bgen/`** (Python) generates buildings from YAML specs. Each building is a layout of walls, floors,
  columns, stairs, openings and roofs, cut into convex chunks joined by a graph of bonded joints.
  It is checked for static stability before export as `.bld` (plus `.glb` and renders).
  Each building has a basement below ground level (spec key `basement`, the lowest floor's
  outline; empty for colossus) on a layered concrete foundation (`foundation`), so blasts on
  the ground floor break through into it. The viewer and sims lay the ground at z = 0 around it.
- **`rubble/`** (Rust) runs the destruction. Damage breaks chunks and joints. A stress solver on the
  joint graph decides what gives way. Anything no longer connected to the ground, or tipping off its
  support, falls as rigid bodies (rapier) and settles into rubble.
  Crates: `rubble-core` (engine), `rubble-stress` (solver), `rubble-format` (`.bld`),
  `rubble-viewer` (Bevy app), `rubble-sim` (headless runs and benchmarks), `rubble-py` (Python bindings).

Building presets: office, apartment, warehouse, tower, house, kyoto.
Free-form structures: colossus.

`docs/explainer.html` is an illustrated walkthrough.
