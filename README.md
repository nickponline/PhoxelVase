# the-finals — procedural destructible buildings + destruction engine

See `DESIGN.md` for the design. Two parts sharing the `.bld` format (§4):

- `bgen/` (Python) — procedural, pre-fractured building generator (replaces Houdini + Building Creator).
- `rubble/` (Rust) — single-player destruction physics engine: rubble-format, rubble-stress, rubble-core,
  rubble-sim (headless CLI/bench), rubble-viewer (Bevy), rubble-py (Python bindings).

## Setup
```
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -e bgen
cd rubble/crates/rubble-py && env -u CONDA_PREFIX VIRTUAL_ENV=$PWD/../../../.venv \
  PATH=$PWD/../../../.venv/bin:$PATH maturin develop --release   # `import rubble`
```

## Generate buildings
```
.venv/bin/bgen build bgen/specs/office.yaml --seed 1 --out assets/buildings   # + renders/
.venv/bin/bgen batch bgen/specs --seeds 1..3 --jobs 6 --out assets/buildings --no-render
.venv/bin/bgen render assets/buildings/office_1 ; .venv/bin/bgen validate assets/buildings/office_1
```
Presets: office, apartment, warehouse, tower, house, kyoto. Output per building:
`building.bld`, `building.glb`, `manifest.json` (stats, validation, stability), `renders/*.png`.
Validation includes static stability via rubble's stress solver with automatic edge reinforcement.

## Destroy them
```
.venv/bin/python -m bgen.sim scenarios/office_columns_out.yaml   # -> runs/<name>/{sim.gif,contact_sheet.png,summary.json}
cd rubble && cargo run --release -p rubble-sim -- run ../scenarios/office_columns_out.yaml
cd rubble && cargo run --release -p rubble-sim -- bench
cd rubble && cargo run --release -p rubble-viewer -- ../assets/buildings/office_1/building.bld
```
Viewer: LMB fire, 1/2/3 weapon (AR/sniper/launcher), E explosion at cursor ([ ] radius), G big blast,
RMB+WASD fly, O orbit, R reset, P pause, +/- next/prev building from assets/buildings (wraps), F1 stress graph,
F2 anchors, F3 fracture colors, F4 cluster tint, F5 stats, H help.

## Tests
```
.venv/bin/python -m pytest -q bgen/tests rubble/crates/rubble-py/tests
cd rubble && cargo test --release --workspace --exclude rubble-viewer
```

## Explainer
Open `docs/explainer.html` in a browser for a step-by-step illustrated explanation of the whole system.
Rebuild its figures with `.venv/bin/python docs/explainer/make_figures.py`.
