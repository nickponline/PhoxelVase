# the-finals — destructible buildings

Procedurally generated, pre-fractured buildings and a real-time destruction engine to knock them down.

![office_1 demolished with X](recordings/office_1.gif)

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
The design is in `DESIGN.md`, and `docs/explainer.html` is an illustrated walkthrough.

## Setup
```sh
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -e bgen
cd rubble/crates/rubble-py && env -u CONDA_PREFIX VIRTUAL_ENV=$PWD/../../../.venv \
  PATH=$PWD/../../../.venv/bin:$PATH maturin develop --release && cd -   # `import rubble` (stability checks)
```

## Run the viewer
```sh
cd rubble && cargo run --release -p rubble-viewer -- ../assets/buildings/office_1/building.bld
```

| key | action | key | toggle |
|---|---|---|---|
| LMB (hold) | beam: cuts through everything in line | 1 | stress graph |
| G | demolish at the cursor (blast) | 2 | anchors |
| X | demolish: charges through the bottom level | 3 | chunk colours |
| RMB + WASD, Q/E | fly, down/up | 4 | dynamic tint (with legend) |
| +/- | next/prev building in `assets/buildings` | 5 | stats panel |
| R / P / . | reset / pause / single step | 6 | keep debris as rubble |
| H | help | | |

## Generate buildings
```sh
.venv/bin/bgen build bgen/specs/office.yaml --seed 1 --out assets/buildings     # one building + renders/
.venv/bin/bgen batch bgen/specs --seeds 1..3 --jobs 6 --out assets/buildings --no-render
```

## Headless runs and recordings
```sh
rubble/target/release/rubble-sim run scenarios/office_columns_out.yaml    # from the repo root
cd rubble && cargo run --release -p rubble-sim -- bench
# 10 s recording: X at 0.5 s, a frame every 4 ticks, then a 15 fps GIF
rubble/target/release/rubble-viewer assets/buildings/office_1/building.bld --record /tmp/rec \
  --frames 600 --demolish-frame 30 --overlay nohelp --size 960x540
ffmpeg -framerate 15 -i /tmp/rec/frame_%05d.png -vf scale=640:-1 office_1.gif
```
`recordings/` has one of these for every office variant.

## Tests
```sh
.venv/bin/python -m pytest -q bgen/tests rubble/crates/rubble-py/tests
cd rubble && cargo test --release --workspace --exclude rubble-viewer
cd rubble && cargo test --release -p rubble-core --test fuzz_hang -- --ignored   # randomized "nothing floats" check
```
