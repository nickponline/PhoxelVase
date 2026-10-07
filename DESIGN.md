# Procedural Destructible Buildings + Destruction Physics — Design

Status: draft v1 (2026-10-06). This is the build plan that agents will implement.

Two deliverables that share one file format:

1. **`bgen`** (Python): a procedural building generator that replaces Houdini plus Embark's Building Creator. It produces pre-fractured, watertight, collision-ready buildings, along with their connection graph, metadata, and preview images.
2. **`rubble`** (Rust): a high-performance, single-player destruction physics engine for a local demo. It loads `bgen` buildings and destroys them with projectiles and explosions, including structural collapse.

```
 spec.yaml + seed                                     scenario / viewer      
      │                                                        │
      ▼                                                        ▼
 ┌──────────────┐  building.bld (binary, canonical)  ┌───────────────────┐
 │  bgen (Py)   │ ─────────────────────────────────▶ │  rubble-core (Rs) │
 │ blockout →   │  building.glb (preview / DCC)      │ damage → graph →  │
 │ feature nodes│  manifest.json, renders/*.png      │ stress → collapse │
 │ → fracture → │                                    │ → rigid bodies    │
 │ graph → export│◀── stability check (pyo3) ────────│ → events          │
 └──────────────┘                                    └───────────────────┘
```

---

## 0. What we know about THE FINALS, and what we copy from it

| THE FINALS (public info) | Our design decision |
|---|---|
| Destruction is fully physically simulated, with structural collapse. | `rubble-core` is a headless simulation library. The viewer, the scenario runner and the Python bindings drive it directly in-process and render its state; cosmetic particles come from events. This is a single-player demo, so there is no networking. |
| Assets are **pre-fractured offline**. Real-time fracturing is too expensive and unpredictable. | `bgen` fractures everything at generation time. The engine never cuts geometry. It only removes chunks, breaks connections, and turns groups of chunks into bodies. |
| Building Creator is an object-level HDA with global parameters (wall thickness, floor height). Inside it, a SOP chain of **feature nodes** each adds one architectural element. Every node has the same I/O: (geometry, blockout+metadata) → (geometry, blockout+metadata). | `bgen` uses the same model. A `BuildingContext` (global params + blockout + element list + metadata) flows through an ordered list of `FeatureNode`s. All nodes share one signature. |
| The nodes are Exterior Walls (horizontal edge loops split floors, vertical loops split the facade), Floors, Roofs (plane/ridge/fascia/eaves), Rooms (interior walls from volume intersections, tagged with room names), Manual Module (snapped placement), and Decals. | We implement these nodes except decals (no materials). Rooms carry tags such as `living_room`. Manual edits are stored as world-space overrides that persist across regenerations. |
| Input must be **watertight**, so fracturing can generate interior faces automatically. | Every element is built as a closed extruded prism, so it is watertight by construction. Fracture faces are flagged `inner`. |
| Collision uses a **2D convex decomposition extruded to 3D**, refined iteratively. Convex ∩ convex = convex, so hulls stay valid after fracturing. Adjacent hulls are merged afterwards. | This is our core geometry representation (§2.3). Every element is a planar **profile polygon plus thickness**. Profile → convex decomposition → Voronoi cells clip the convex parts → each chunk is a convex prism that is also its own collision hull → merge small adjacent pieces when the union is still convex. |
| Modules carry visual meshes, boolean meshes (arched or round openings), and **sockets** (balconies, windows, and curtains are spawned in-engine). | Openings are boolean cutters on the 2D profile. Sockets are exported as transforms plus a type, and the engine spawns props from them. |
| GDC 2024 "Engineering Mayhem" (Måns Isaksson) covered **connection graphs between pieces, structural analysis, the physics state of debris, and simulation performance**. | `rubble` has a chunk connection graph, ground anchors, an incremental connectivity pass, an amortized stress solver, cluster → rigid-body promotion, and debris settling and freezing. |
| Collapsing debris **stays and reshapes the map**. | Large debris that settles is *frozen* back into static, still-destructible world geometry. Small debris despawns. |
| Blockout → fractured asset takes about 4–6 minutes per change in Houdini. | Target: **< 30 s** per building on a laptop for typical sizes (3–6 floors, 2–8k chunks). |

The stress-solver internals from the GDC talk are not public, so those parts are our own design. The closest public reference is **NVIDIA Blast**: a support graph, bonds with area and strength, and an iterative stress solver. We borrow ideas from it, not code.

---

## 1. Conventions (shared contract)

- Units are **meters, kilograms, seconds**. The world is **Z-up** and right-handed. Gravity is `(0, 0, -9.81)`.
- A building's local origin is at ground level (z = 0), at the footprint's bounding-box min corner. The engine places buildings with a rigid transform (yaw + translation; no scale).
- IDs are dense `u32` indices into arrays (chunks, edges, elements). No hash maps cross the boundary.
- All binary data is little-endian, with sections 16-byte aligned (§4).
- Both projects pin the **format version** (`BLD_VERSION`). A change requires updating the Python writer, the Rust reader, and the golden fixtures together.

---

## 2. `bgen`: procedural building generator (Python)

### 2.1 Dependencies

`numpy`, `scipy` (Voronoi, KD-tree), `shapely>=2` (2D booleans and offsets), `mapbox_earcut` (triangulation), `trimesh` (glb export, mesh checks), `pyyaml`, `moderngl` (headless rendering), `Pillow`, `matplotlib` (floor plans and graph plots). Optional: `manifold3d` to validate watertightness of unions in tests.

No Houdini and no Blender. Everything runs headless from the CLI.

### 2.2 Inputs: building spec

The input is a YAML spec plus a seed. Everything not given is sampled from the preset's ranges using the seed, so `(spec, seed)` fully determines the output.

```yaml
name: corner_office
preset: office            # office | apartment | warehouse | tower | house | kyoto
seed: 1234
global:
  floor_height: 3.6       # floor-to-floor height
  slab_thickness: 0.3
  ext_wall_thickness: 0.3
  int_wall_thickness: 0.15
blockout:                 # list of masses; unioned per floor
  - footprint: [[0,0],[24,0],[24,16],[0,16]]
    floors: 5
  - footprint: [[24,0],[32,0],[32,8],[24,8]]
    floors: 2
    roof: gable
nodes:                    # ordered feature-node chain, Building-Creator style
  - floors: {}
  - exterior_walls: {bay_width: [3.0, 4.5], corner_rule: x_runs_through}
  - columns: {grid: 6.0, size: 0.5}
  - rooms: {strategy: bsp, min_room: 9.0, tags: [office, meeting, toilet, corridor]}
  - stairs: {cores: 1, type: switchback}
  - openings: {windows: {style: ribbon, sill: 0.9, head: 2.6}, doors: {per_room: 1}}
  - roofs: {flat: {parapet: 1.0}, gable: {pitch_deg: 35, eaves: 0.4}}
  - balconies: {facades: [south], every_n_floors: 1}   # emitted as sockets + slabs
  - manual: {file: corner_office.overrides.yaml}      # optional
fracture:
  concrete: {cell_size: 0.9, edge_bias: 2.0}
  brick:    {cell_size: 0.6}
  wood:     {cell_size: 0.5, anisotropy: [3,1]}
  glass:    {mode: shatter_on_hit}
indestructible: [ground_slab]   # element tags the engine must never break
```

### 2.3 Core representation: `Panel` (profile + thickness)

Every structural element (wall, floor slab, column, beam, roof plane, stair flight, parapet, balcony slab) is a **Panel**:

```python
@dataclass
class Panel:
    id: int
    kind: str                 # "ext_wall" | "int_wall" | "floor" | "column" | "beam" | "roof" | "stair" | ...
    frame: np.ndarray         # 4x4 local→building transform; panel plane = local XY, thickness along +Z
    profile: shapely.Polygon  # 2D outline in local XY, holes allowed (openings)
    thickness: float
    material: str             # concrete | brick | wood | metal | glass
    tags: dict                # floor index, room ids on each side, facade id, "indestructible", ...
```

Why this representation:

- It is **watertight by construction**: an extruded simple polygon is a closed solid.
- Openings (windows, doors, arches, round windows) are 2D boolean differences. Curves are polygonized at a configurable segment count.
- Collision and fracture reduce to **2D** problems, which are robust and fast with shapely. This mirrors THE FINALS' 2D convex decomposition extruded to 3D.
- A sloped roof or inclined stair is a panel whose frame is tilted.

**Junction rules** (so that panels never overlap and contacts are clean coplanar faces):

1. Floor slabs span the full floor footprint, including under exterior walls.
2. Walls sit between slabs: they start at the top of slab *n* and end at the bottom of slab *n+1*.
3. At exterior corners, walls whose direction is closest to X run through the corner, and Y-ish walls butt into them (`corner_rule`). Interior walls butt into exterior walls.
4. Columns occupy their own footprint. Walls are clipped around columns.

Under these rules, every structural contact is a pair of anti-parallel coplanar faces. That makes connection detection exact (§2.6).

### 2.4 Pipeline

```
spec ─▶ Blockout ─▶ [FeatureNode]* ─▶ Panels ─▶ Decompose ─▶ Fracture ─▶ Merge ─▶ Graph ─▶ Validate ─▶ Export ─▶ Render
```

```python
class FeatureNode(Protocol):
    name: str
    def __call__(self, ctx: BuildingContext, params: dict, rng: np.random.Generator) -> BuildingContext: ...

@dataclass
class BuildingContext:
    spec: Spec
    blockout: Blockout        # per-floor footprint polygons, floor z-levels, facade edge loops
    panels: list[Panel]
    rooms: list[Room]         # per floor: polygon, tag, id
    sockets: list[Socket]     # type, 4x4 transform, owner panel
    meta: dict                # free-form data passed downstream (Building Creator's "output 2")
```

Each node gets its own RNG stream, derived from `hash(seed, node_index, node_name)`. Editing one node's parameters therefore does not reshuffle the others.

**Blockout.** Union the masses per floor level to get `floor_polys[i]` (shapely). Derive the **facade loops**: the exterior ring of each floor, split into straight facade segments. Store the z-levels for each floor.

**Feature nodes:**

| Node | Behavior |
|---|---|
| `floors` | One slab panel per floor: `floor_polys[i]` at `z_i`, thickness `slab_thickness`. The ground slab is tagged `anchor` (and `indestructible` if listed). Floor *i+1*'s slab covers only `floor_polys[i+1]`; setbacks leave the rest as roof. |
| `exterior_walls` | For each facade segment on each floor, create one wall panel whose profile is a rectangle `segment_length × clear_height`. Split it into **bays** (vertical loops) with widths sampled from `bay_width`. Bay boundaries are stored in `meta` so that openings and balconies align to them. |
| `columns` | Place columns on a structural grid inside each floor polygon, plus at facade corners. Each column is a narrow vertical panel. Columns raise the global stiffness that the stress check relies on. |
| `rooms` | BSP-split each floor polygon (excluding stair cores) into rooms of at least `min_room` m². Room tags are assigned with weights (corridor gets the long thin cells). Each shared room edge becomes an interior wall panel. Each room keeps its polygon and tag. |
| `stairs` | Reserve a core rectangle that is the same on every floor and cut it out of the slabs above. Add an inclined slab panel per flight, a landing slab, and step prisms (small panels tagged `cosmetic_attached`). This guarantees vertical traversal. |
| `openings` | Subtract window and door shapes from wall profiles. Windows follow a style (`punched`, `ribbon`, `arched`, `round`, `storefront`) aligned to bays. Each room gets at least one door to the corridor or a neighbor, so the graph is fully walkable. Each window emits a **glass panel** (material glass, non-structural) and a `window_frame` socket. |
| `roofs` | `flat`: the top slab plus parapet wall panels. `gable`/`hip`: inclined panels built from the footprint's straight skeleton (simple rectangles first, `shapely`-based skeleton later), plus ridge, fascia and eaves strips as narrow panels. `kyoto`: a curved-eave variant that polygonizes the curve into several tilted strips. |
| `balconies` | A cantilevered slab panel plus parapet at bay positions. It also emits a `balcony_rail` socket. Cantilevers are the main test case for the stress solver. |
| `manual` | Loads overrides: add, move or delete a panel or opening, keyed by **world-space position and kind**, not by ID. Edits survive regeneration as long as the targeted feature still exists at that location, which matches Building Creator's behavior. |

Adding a node means writing one file in `bgen/nodes/` and registering it. Nodes must not mutate earlier panels except through documented operations (`cut_opening`, `remove_panel`, `split_panel`).

### 2.5 Decompose → Fracture → Merge (per panel, in 2D)

1. **Convex decomposition.** Triangulate the profile with its holes (earcut), then apply **Hertel–Mehlhorn** merging: greedily remove diagonals while the merged polygon stays convex. Result: at most 4× the optimal number of convex parts, which is plenty.
2. **Seed points.** Material `cell_size` gives the target density. Optional `edge_bias` adds extra seeds near opening edges and panel borders, so damage around windows looks good. `anisotropy` scales the seed distribution (wood splinters along its grain). Each panel gets a deterministic RNG.
3. **Voronoi clip.** Compute the 2D Voronoi of the seeds, with mirrored boundary points so all cells are bounded. Intersect each cell (convex) with each convex part (convex); the result is convex. Drop slivers below `min_area`.
4. **Merge.** Merge adjacent convex pieces whose union's convex hull equals their union (area check with tolerance) when either piece is under `min_chunk_area`. This is the "adjacent hulls merged" step.
5. **Extrude.** Each 2D convex piece becomes a convex prism of panel thickness, transformed by `panel.frame`. Faces get flags: `outer` (part of the original panel surface), `inner` (a fracture or cut face), `cap_edge` (on the panel border).
6. **Glass** is not Voronoi-fractured into chunks. It is a single chunk with `flags = GLASS` that the engine shatters cosmetically on any hit.

Each chunk is now simultaneously its **render mesh**, its **convex collision hull** (same vertices, ≤ 2·n_poly verts), and its **mass element**. Mass, center of mass and inertia tensor are computed analytically for the prism from the material density.

### 2.6 Connection graph

Nodes are chunks. An edge exists when two chunks share a contact face with area ≥ `min_contact` (default 0.01 m²).

- **Intra-panel edges:** shared Voronoi/decomposition edges in 2D. The contact area is the shared edge length × thickness.
- **Inter-panel edges:** use a broadphase (AABB grid / `scipy.spatial.cKDTree` on face centroids). For candidate pairs, find faces with anti-parallel normals (dot < −0.999) whose planes are within `1e-4` m. Project both faces into the plane and intersect them with shapely; the overlap area becomes the contact area. Junction rules (§2.3) make these contacts exact.
- **Edge data:** `area`, `centroid`, `normal` (from a to b), and `strength = area × bond_strength(material_a, material_b)`. The bond strength table is symmetric, e.g. concrete–concrete is high, wood–concrete is lower, and glass is never structural.
- **Anchors:** chunks with a face at z ≤ 0 + eps, or chunks in elements tagged `anchor`, are ground-anchored.
- **Hierarchy:** each chunk stores `panel_id`. Panels belong to floors. This lets the engine do coarse reasoning per panel or per floor (LOD, collapse grouping).

### 2.7 Validation (fails the build)

- Every hull is convex: all vertices lie on the inner side of all face planes (within 1e-5).
- Every chunk mesh is watertight (`trimesh.is_watertight`) with consistent winding and positive volume.
- Per panel, the sum of chunk volumes equals profile area × thickness (within 0.1%).
- No two chunks interpenetrate (sampled SAT test on graph-neighbor and AABB-overlapping pairs).
- **Every non-cosmetic chunk is connected to an anchor.** A floating chunk means a generator bug.
- **Static stability:** call `rubble`'s stress solver through the `rubble-py` bindings (§3.6). The undamaged building must stand with a utilization below 0.5 on every edge. If not, the generator reinforces: it adds columns or thickens the cantilever and retries up to N times, then fails with a report.
- Room connectivity: all rooms are reachable through doors or stairs.

### 2.8 Outputs

```
assets/buildings/<name>_<seed>/
  building.bld          # canonical binary bundle for the engine (§4)
  building.glb          # preview: one node per chunk, extras={chunk_id, panel_id, material, flags}
  manifest.json         # spec echo, seed, version, stats, materials table, rooms, sockets, timing
  graph.json            # optional, human-readable connection graph (debug flag)
  renders/
    iso_{ne,nw,se,sw}.png   # 4 corner isometric views, 30° elevation, uniform shading
    front.png side.png top.png
    fractured_iso.png       # random color per chunk: shows the fracture pattern
    exploded.png            # chunks pushed outward from their panel centroid ×0.15: shows the interior
    cutaway_floor_{i}.png   # roof and higher floors hidden, oblique view into floor i
    plan_floor_{i}.png      # matplotlib floor plan: rooms colored by tag, doors, stairs, columns
    graph.png               # connection graph overlay: edges colored by strength, anchors highlighted
    stress.png              # (if rubble-py available) static utilization heatmap per edge
```

**Renderer** (`bgen/render/`): `moderngl.create_standalone_context()` (headless; works on macOS via CGL and on Linux via EGL), flat or Lambert shading with a single key light plus ambient, edge outlines from a normal and depth discontinuity pass, and orthographic or perspective cameras auto-fitted to the building's AABB. A pure-numpy z-buffer fallback rasterizer is used when there is no GL context. It is slower, but is fine for previews.

### 2.9 CLI

```
bgen build specs/corner_office.yaml --seed 1234 --out assets/buildings/
bgen batch specs/ --seeds 1..20 --jobs 8         # multiprocessing, one building per process
bgen render assets/buildings/corner_office_1234   # re-render only
bgen validate assets/buildings/corner_office_1234
bgen district --grid 4x4 --seed 7                 # optional: a block of buildings + ground → arena.json
```

### 2.10 Package layout

```
bgen/
  pyproject.toml
  bgen/
    spec.py  context.py  rng.py  cli.py
    blockout.py
    nodes/ floors.py exterior_walls.py columns.py rooms.py stairs.py openings.py roofs.py balconies.py manual.py
    geom/  panel.py convex2d.py voronoi2d.py fracture.py prism.py massprops.py skeleton.py
    graph.py  validate.py  materials.py
    export/ bld.py gltf.py manifest.py
    render/ gl.py raster_np.py views.py plots.py
  specs/ office.yaml apartment.yaml warehouse.yaml tower.yaml house.yaml kyoto.yaml
  tests/
```

---

## 3. `rubble`: destruction physics engine (Rust)

### 3.1 Why Rust, and what we build versus reuse

Rust gives native performance, SIMD, safe parallelism (`rayon`), easy Python bindings (`pyo3`), and a mature physics ecosystem.

- **We reuse** `rapier3d` for rigid-body dynamics (broadphase, narrowphase, contact solver, sleeping, CCD) and `parry3d` for geometric queries. A production-grade contact solver is not where the value is, and Rapier is fast (SIMD and parallel features) and has optional cross-platform determinism.
- **We build** the destruction layer, which is everything THE FINALS-specific: the chunk world, damage, the connection graph, connectivity, the stress solver, cluster promotion and splitting, debris lifecycle, budgets, and events.
- The rigid-body backend sits behind a thin `PhysicsBackend` trait, so a custom solver specialized for convex prisms can replace Rapier later without touching the destruction layer.

### 3.2 Crates (Cargo workspace `rubble/`)

| Crate | Purpose |
|---|---|
| `rubble-format` | `.bld` reader/writer, `#[repr(C)]` records + `bytemuck` zero-copy casts, version checks. Shared by every other crate. |
| `rubble-core` | The engine library: `World`, destruction layer, Rapier integration, events. No rendering, no IO except loading. |
| `rubble-sim` | Headless CLI: runs scripted scenarios (`scenario.yaml`), benchmarks, and recording. Writes event logs and per-frame transform dumps. |
| `rubble-viewer` | Bevy app: load buildings, fly camera, click to shoot, keys for explosives, debug overlays (graph, stress heatmap, anchors, sleeping bodies, budgets). Uses `rubble-core` directly. |
| `rubble-py` | `pyo3` + `maturin` bindings: load, apply damage, step, read state. Used by `bgen` validation, by Python scenario rendering, and by tests. |

### 3.3 World model

```
World
 ├─ buildings: Vec<Building>            // immutable data from .bld + mutable state
 │    ├─ chunks (SoA): hp[], alive bitset, state[] {Static, InCluster(cid), Frozen, Gone}
 │    ├─ graph: CSR adjacency (offsets[], neighbors[], edge_ids[]), edge_health[], edge_alive bitset
 │    ├─ anchor bitset
 │    └─ static_body: one fixed Rapier body; one collider per live static chunk (convex hull)
 ├─ clusters: SlotMap<ClusterId, Cluster>   // detached groups simulated as ONE dynamic compound body
 │    └─ chunk list, internal edge subset, rapier body handle, age, accumulated impulse per internal edge
 ├─ projectiles: Vec<Projectile>
 ├─ pending: collapse queue (delayed detaches), damage queue
 └─ budgets / stats
```

Key performance idea: an **intact building is static**. Its chunks are colliders on one fixed body and cost nothing in the solver. Only detached clusters become dynamic bodies, and one cluster is one compound body regardless of how many chunks it has. A 5-storey section falling is therefore one body, which matches how THE FINALS collapses look: a big piece falls as one and breaks on impact.

### 3.4 Per-tick pipeline (fixed 60 Hz)

```
1. Projectiles      advance; ray/shape-cast against static chunks + dynamic clusters (CCD by sweep)
2. Damage           apply queued damage (bullets, explosions, impacts) → chunk HP, edge health
3. Removal          chunks with hp ≤ 0 → removed (event ChunkDestroyed); edges with health ≤ 0 → broken
4. Connectivity     incremental: from the dirty set, find components with no anchor → detach candidates
5. Stress           amortized solver on dirty buildings (≤ budget ms); overloaded edges → broken → back to 4
6. Promotion        detached components → after collapse_delay → new Cluster (dynamic compound body)
7. Rapier step      integrate dynamic clusters
8. Cluster impacts  contact impulses → internal-edge damage → split clusters along broken edges
9. Settling         sleeping clusters: big → Frozen static rubble (re-anchored, still destructible); small → despawn timer
10. Budgets         enforce caps (max dynamic bodies, max chunks in flight); LOD-out smallest/oldest
11. Output          events (for rendering, FX and logging)
```

#### Damage

- **Bullets** are hitscan (raycast) or ballistic (substepped swept sphere). A hit applies `damage × material_multiplier` to the hit chunk, and a fraction to its graph neighbors within `splash_r`. Optional penetration: the projectile continues with energy reduced by `thickness × material_resistance`. Glass shatters on any hit.
- **Explosions** `{center, radius, inner_radius, damage, impulse}` sphere-query the chunks. Chunk damage falls off linearly from `inner_radius` to `radius`. Occlusion: a coarse ray to each chunk centroid reduces damage by the number of intact chunks crossed (cap 3 rays per chunk, ≤ 256 chunks). Explosions also damage **edges** out to `crack_radius = 1.5 × radius`. This breaks bonds without destroying chunks, so intact slabs detach and fall, which is the signature effect.
- **Impacts:** contact impulses from falling clusters (step 8) are converted into damage on both the struck static chunks and the cluster's internal edges.

#### Connectivity (incremental)

- Maintain a dirty set: the endpoints of removed chunks and broken edges.
- From each dirty node that has not already been visited, run a BFS over alive edges with early-out on reaching an anchor. If the BFS exhausts without finding an anchor, the visited set is a detached component. Visited marks use an epoch counter, so they never need clearing.
- The cost is proportional to the size of the affected component, not the building. A full recompute (union-find, parallel per building) runs only on load and on debug assertions.

#### Stress solver (structural analysis)

This is what makes buildings fall when you take out their ground-floor supports, even while they are still technically connected.

- **Model:** each chunk has a load (weight `m·g`). Load must flow along alive edges to anchors. Treat the graph as a resistive network: edge conductance `k_e = strength_e`, sources = chunk weights, sinks = anchors (potential 0). Solve the weighted graph Laplacian `L φ = w` with anchors as Dirichlet nodes. The edge flow is `f_e = k_e (φ_a − φ_b)`, and its **utilization** is `u_e = |f_e| / capacity_e`.
- **Bending/overhang term:** for edges whose contact normal is roughly horizontal (wall-to-wall or slab-to-wall junctions carrying cantilevers), add a moment estimate: carried load × lever arm (horizontal distance from the edge centroid to the downstream load centroid), divided by a section modulus derived from the contact area. Without this, cantilevers and balconies never fail.
- **Solve:** conjugate gradient with a Jacobi preconditioner, **warm-started** from the previous φ. Run a fixed number of iterations per tick within a time budget (default 1.5 ms total, parallel across buildings with rayon). After damage, the solution converges over a few ticks, which also produces a natural creak delay before collapse.
- **Failure:** edges with `u_e > 1` (after a short hysteresis, `u_e > 1` for at least `t_hold`) break, starting with the worst first, at most `K` per tick. Each break feeds back into connectivity, so a collapse progresses over several frames instead of everything popping at once.
- **Scope:** only buildings with a dirty flag are solved. Structural components below `min_component_mass` skip stress and use connectivity only.
- `bgen` calls the same solver offline for static validation (§2.7), so a generated building and the engine agree on what stands.

#### Promotion, clusters, and splitting

- A detached component waits `collapse_delay` (0.3–1.5 s, size-scaled; it emits `CollapseWarning` for dust and creak FX). It then becomes a `Cluster`: its static colliders are removed, and one dynamic Rapier body is created with a compound shape of the chunk hulls. Mass and inertia are summed from the precomputed per-chunk props. Initial velocity comes from the explosion impulse if one caused the detach.
- **Splitting:** the cluster keeps its internal edges. Contact impulses at step 8 are distributed to the internal edges near the contact. An edge breaks when its accumulated impulse exceeds `strength × impact_factor`. Connected components inside the cluster are then recomputed, and any new component spawns a new body, inheriting the linear and angular velocity at its center of mass.
- **Single destroyed chunks** (HP ≤ 0) either become small dynamic debris (if their volume is above `debris_min_volume` and the budget allows) or vanish, with a cosmetic `ChunkShattered` event that the renderer turns into particles.

#### Settling and freezing (debris reshapes the map)

- A cluster asleep for at least `freeze_time` (1 s) with mass ≥ `freeze_min_mass` becomes **Frozen**. Its body is removed and its chunks are re-added as static colliders at their current world transforms, under a per-building "rubble" fixed body. They are re-anchored (rubble rests on what it landed on) and remain shootable with their remaining HP.
- Small clusters despawn after `debris_ttl` (10 s), or immediately when over budget.

#### Budgets and performance targets

| Item | Target |
|---|---|
| Arena scale | 30–60 buildings, 150k–400k chunks total |
| Tick at 60 Hz, worst case (big collapse) | ≤ 8 ms on 8 cores |
| Steady state (no destruction) | ≤ 1 ms (everything static; projectile queries only) |
| Max simultaneously dynamic clusters | 512 (configurable); over budget → smallest/oldest despawn or freeze early |
| Stress solve budget | 1.5 ms/tick across all dirty buildings |
| Load a 5k-chunk building | ≤ 20 ms (zero-copy `.bld` + collider creation) |

Implementation rules for agents:
- SoA arrays and bitsets for chunk and edge state. No per-chunk heap allocations in the tick.
- CSR graph, built once at load and never rebuilt. Breaking an edge flips a bit.
- Use rayon over buildings for stress and over dirty components for connectivity.
- Use Rapier features `simd-stable`, `parallel`. `enhanced-determinism` is a build feature for reproducible test runs.
- Rapier risk: hundreds of thousands of static colliders in one BVH. Mitigation: give each building its own static body (spatially coherent subtree). If profiling shows a problem, keep far-from-action static chunks out of Rapier: a custom per-building BVH (from `parry3d::partitioning::Qbvh`) handles projectile and explosion queries, and colliders are activated only inside an "awake region" around dynamic clusters.

### 3.5 Public API (`rubble-core`)

```rust
let mut world = World::new(WorldConfig::default());
let b: BuildingId = world.load_building("assets/buildings/corner_office_1234/building.bld", Isometry::new(pos, yaw))?;
world.add_ground_plane(0.0);

world.fire(Projectile::hitscan(origin, dir, Weapon::AR));         // or Projectile::ballistic(...)
world.explode(Explosion { center, radius: 4.0, inner_radius: 1.0, damage: 400.0, impulse: 3000.0 });
world.damage_chunk(b, chunk_id, 1e9);                              // scripted / tests

world.step(1.0 / 60.0);

for ev in world.drain_events() { match ev {
    Event::ChunkDestroyed { building, chunk, pos } => {}
    Event::ChunkShattered { .. } => {}           // cosmetic
    Event::EdgeBroken { .. } => {}               // debug only
    Event::CollapseWarning { building, chunks, delay } => {}
    Event::ClusterDetached { cluster, building, chunks, transform, lin_vel, ang_vel } => {}
    Event::ClusterSplit { parent, children } => {}
    Event::ClusterFrozen { cluster, transform } => {}
    Event::ClusterDespawned { cluster } => {}
}}

let state = world.building_state(b);  // alive bitset, utilization per edge (debug)
```

### 3.6 Python bindings (`rubble-py`) and integration with `bgen`

```python
import rubble
w = rubble.World()
b = w.load_building(".../building.bld")
report = w.static_stress_report(b)        # used by bgen validate: max utilization, worst edges
w.explode(center=(10, 4, 1), radius=4)
for _ in range(240): w.step(1/60)
xf = w.chunk_world_transforms(b)          # (N, 4, 4) numpy, zero-copy
```

This closes the loop:
- `bgen validate` calls the real solver, so generated buildings stand.
- `bgen render-sim scenario.yaml` runs a scripted destruction in `rubble` and renders frames with the `bgen` renderer, using per-chunk transforms. This yields destruction preview images and gifs without opening the viewer.

### 3.7 Scenario format (`rubble-sim`, tests, and renders)

```yaml
buildings:
  - {path: assets/buildings/corner_office_1234/building.bld, pos: [0,0,0], yaw: 0}
steps: 600
dt: 0.016666
actions:
  - {t: 0.5, explode: {center: [2,2,0.5], radius: 3, damage: 800, impulse: 4000}}
  - {t: 0.5, explode: {center: [22,2,0.5], radius: 3, damage: 800, impulse: 4000}}
  - {t: 1.0, fire: {origin: [-20,8,1.6], dir: [1,0,0], weapon: ar, count: 30, rate: 10}}
record: {transforms: true, events: true, out: runs/office_collapse/}
```

---

## 4. The `.bld` file format (the contract)

```
Header (64 B)
  magic        [u8;4]  = "BLD\0"
  version      u32     = 1
  flags        u32
  section_count u32
  content_hash [u8;32] (blake3 of all sections)
  reserved     ...
Section table: section_count × { tag [u8;4], offset u64, size u64, count u32, stride u32 }
Sections (16-byte aligned):
```

| Tag | Record (`#[repr(C)]`, stride) | Content |
|---|---|---|
| `META` | utf-8 JSON | name, seed, spec echo, units, bounds, material table (density, hp, bond strengths, cell size), room list, generator version |
| `ELEM` | `{id u32, kind u16, material u16, floor i16, flags u16, first_chunk u32, chunk_count u32, frame [f32;16], thickness f32}` | panels |
| `CHNK` | `{elem u32, material u16, flags u16, mass f32, volume f32, hp f32, com [f32;3], inertia [f32;6], aabb_min [f32;3], aabb_max [f32;3], hull_v_off u32, hull_v_cnt u32, hull_p_off u32, hull_p_cnt u32, mesh_i_off u32, mesh_i_cnt u32}` | chunks |
| `HVRT` | `[f32;3]` | hull vertices (building space) |
| `HPLN` | `[f32;4]` (n, d) | hull face planes: fast containment and SAT |
| `MVRT` | `{pos [f32;3], nrm [i16;3], face_flags u16}` | render vertices (`outer`/`inner`/`cap_edge`) |
| `MIDX` | `u32` | render indices, per chunk range |
| `EDGE` | `{a u32, b u32, area f32, strength f32, centroid [f32;3], normal [f32;3]}` | connection graph (a < b) |
| `ANCH` | bitset `u64[]` | anchor chunks |
| `SOCK` | `{type u32, owner_elem u32, xform [f32;16]}` | sockets (balcony, window_frame, door, prop) |

`CHNK.flags` bits are: `ANCHOR`, `INDESTRUCTIBLE`, `GLASS`, `COSMETIC_ATTACHED` (no structural role; detaches with its parent), and `NO_DEBRIS`.

The Python writer uses numpy structured dtypes that match the Rust `repr(C)` layouts exactly. A **golden-fixture test** writes a tiny two-box building in Python, reads it in Rust, and compares field by field. This test lives in CI for both sides.

---

## 5. Repository layout

```
the-finals/
  DESIGN.md                 ← this file
  docs/format.md            ← .bld spec (extracted from §4 when frozen)
  bgen/                     ← Python package (§2.10)
  rubble/                   ← Cargo workspace (§3.2)
  fixtures/                 ← golden .bld files + expected stats
  scenarios/                ← rubble-sim yaml
  assets/buildings/         ← generated output (gitignored except a few samples)
```

---

## 6. Implementation plan for agents

Milestones gate the dependencies. Workstreams within a milestone can run in parallel, each in its own worktree.

**M0 – Contract (one agent, must land first)**
- `rubble-format` crate with record structs, reader, and validation.
- `bgen/export/bld.py` writer with matching numpy dtypes.
- A hand-built fixture (two stacked boxes, 4 chunks, 3 edges, 2 anchors) and a round-trip test in both languages.
- *Done when:* Python writes it, Rust reads it, and the field-by-field test passes.

**M1 – Parallel core (5 agents)**

| Agent | Scope | Done when |
|---|---|---|
| A: geometry | `geom/*`: Panel, earcut + Hertel–Mehlhorn, bounded 2D Voronoi, clip, merge, prism extrude, mass props | Property tests: convexity, volume conservation, watertight, deterministic for the same seed |
| B: architecture | `blockout.py`, all `nodes/*`, presets | Generates panels for all 6 presets. Rooms reachable. No panel overlaps (junction rules). |
| C: render | `render/*`: GL views, numpy fallback, plots (plan, graph, exploded) | All §2.8 images produced for the fixture and a stub building |
| D: engine core | `rubble-core`: load, static colliders, damage, incremental connectivity, promotion, clusters, splitting, freezing, events | Unit tests on synthetic graphs: remove the base → detach; a bullet doesn't detach; cluster splits on impact |
| E: viewer | `rubble-viewer` (Bevy): load `.bld`, camera, shoot, explode, overlays | Can load the fixture and blow it up interactively |

**M2 – Integration (2–3 agents)**
- Graph construction (§2.6) and validation (§2.7) on real generated buildings. Generator-to-engine end-to-end.
- Stress solver (§3.4) plus `rubble-py` bindings. `bgen validate` uses them, with an auto-reinforce loop.
- `rubble-sim` scenarios, `bgen render-sim`, and benchmarks (criterion): load time, steady-state tick, worst-case collapse tick.
- *Done when:* 20 seeds × 6 presets generate, validate, and stand. The scripted "blow out the ground-floor columns" scenario collapses the building in each preset. The "single explosion on the top floor" scenario does *not* collapse it. Performance targets in §3.4 are met on the benchmark arena.

**M3 – Polish / stretch**
- `bgen district` arena generation. Curved roofs (kyoto) and straight-skeleton hip roofs.
- Optional: second-level micro-fracture of chunks for richer debris (cosmetic only, render-side).

---

## 7. Tuning parameters and risks

| Risk | Mitigation |
|---|---|
| The stress model looks wrong: buildings are too stable or too fragile | Every coefficient lives in the materials table in `META`. Viewer overlays show live utilization. Scripted benchmark scenarios have expected outcomes (collapse / no collapse) as regression tests. |
| Cantilevers ignored by the pure-flow model | Explicit bending term (§3.4), with balconies as test cases |
| Rapier with 300k static colliders | Per-building static bodies; fallback to a custom per-building QBVH with an awake region (§3.4) |
| Floating-point mismatch in inter-panel contact detection | Junction rules give exact coplanar contacts. Snap panel frames to a 0.1 mm grid. |
| Slivers from Voronoi/opening intersections | `min_area` drop plus a merge pass; validation rejects chunks with an aspect ratio > 20 |
| Generation time | Per-panel fracture is independent, so `multiprocessing` runs over panels; Voronoi/clip is vectorized where possible |

## Sources

- [Making the Procedural Buildings of THE FINALS (SideFX)](https://www.sidefx.com/community/making-the-procedural-buildings-of-the-finals-using-houdini/)
- [Engineering Mayhem: Technical Deep-Dive into Environmental Destruction in THE FINALS (GDC 2024 schedule)](https://schedule.gdconf.com/session/engineering-mayhem-technical-deep-dive-into-environmental-destruction-in-the-finals/900179)
- [Engineering Mayhem (GDC Vault)](https://gdcvault.com/play/1034280/Engineering-Mayhem-Technical-Deep-Dive)
- NVIDIA Blast (support graph + stress solver): conceptual reference for §3.4

---

## Implementation status (2026-10-06)

M0–M2 implemented (see README.md). Notable deviations from the design above:
- Stress model: conductance = capacity / centre distance; near-vertical bearing contacts get
  `compression_factor` (10×); bending uses net-flow moment about a propagated upstream load centroid,
  aggregated per **section** (inter-panel interface or intra-panel cut band) with `flexural_factor` (5×).
- Material bond strengths raised to 8/5/3/20 MPa (concrete/brick/wood/metal); bgen reinforces any edge
  with intact utilization ≥ 0.5 (scales its strength), and demotes tiny structural slivers that only touch
  glass to cosmetic.
- Engine: static chunk colliders are parentless (Rapier child removal is O(n)); CCD off by default;
  explosion evaluation parallel; landing break-up runs a one-shot stress solve on the falling cluster
  with deceleration as the load (applied after `impact_latency_ticks` for determinism); resting debris
  freezes into static rubble; only small destroyed chunks become debris.
- Frozen rubble re-checks its support (`thaw_unsupported`): when geometry is removed (destroyed,
  detached, despawned) the engine wakes sleeping bodies around it and thaws any frozen group that
  no longer rests on anything (a 3 cm down/up probe). A slow round-robin sweep catches supports
  that moved away while anything is still dynamic. Rubble also waits to freeze until nothing under
  it is moving.
- Renderer: 2× SSAA instead of MSAA (edge pass needs full-res buffers).
- Not done: `bgen district`, door-swing plans.
