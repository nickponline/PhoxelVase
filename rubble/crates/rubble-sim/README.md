# rubble-sim

Headless runner for `rubble-core`.

```
cargo run --release -p rubble-sim -- run scenarios/tower_collapse.yaml [--out DIR]
cargo run --release -p rubble-sim -- bench [--buildings 40] [--ticks 300] [--no-bending]
```

## Scenario format

```yaml
buildings:
  - {path: ../fixtures/two_box.bld, pos: [0,0,0], yaw: 0}         # path: cwd-relative, else scenario-relative
  - {synthetic: {kind: tower, floors: 4, side: 6.0}, pos: [20,0,0]} # kinds: tower|grid|wall|cantilever|two_box
steps: 600
dt: 0.016666
ground: auto           # ground plane height; null = no ground; auto (default) = lowest foundation bottom
config:                # optional partial WorldConfig override (merged into defaults)
  stress: {max_iters: 30}
actions:
  - {t: 0.5, explode: {center: [2,2,0.5], radius: 3, inner_radius: 0.5, damage: 800, impulse: 4000}}
  - {t: 1.0, fire: {origin: [-20,8,1.6], dir: [1,0,0], weapon: ar, count: 30, rate: 10, ballistic: false}}
  - {t: 2.0, damage: {building: 0, chunks: [0, 1], amount: 1.0e9}}
record: {transforms: true, events: true, out: runs/office_collapse/, every: 1}
```

Weapons: `ar`, `smg`, `sniper`, `shotgun`, `launcher`, or `{custom: {damage, splash_r, splash_frac, penetration, range, speed, radius, impulse, gravity_scale}}`.

## Outputs (in `out/`)

* `events.jsonl` — one JSON object per engine event: the serialized `Event`
  (`"type": "ChunkDestroyed" | "ChunkShattered" | "EdgeBroken" | "CollapseWarning" |
  "ClusterDetached" | "ClusterSplit" | "ClusterFrozen" | "ClusterDespawned"`, plus its fields)
  with `tick` and `time` added. Cluster ids are u64; matrices are row-major 4x4.
* `transforms.bin` — per-frame chunk transforms (layout below).
* `summary.json` — load time, tick timing (avg/p99/max, plus the same excluding blocking waits on
  background impact solves — see `WorldConfig::impact_latency_ticks`), worst-tick stage breakdown,
  event count, final stats.

## `transforms.bin` layout (little-endian, every field 4 bytes)

```
Header (32 bytes)
  magic        [u8;4] = "RBTF"
  version      u32    = 1
  n_frames     u32           (patched when the run finishes)
  n_chunks     u32           total chunks over all buildings
  n_buildings  u32
  dt           f32           simulation step
  every        u32           a frame is written every `every` steps
  reserved     u32
Building table: n_buildings x
  chunk_offset u32           index of the building's first chunk in the per-frame arrays
  n_chunks     u32
  name_len     u32
  name         [u8; name_len] utf-8 (absolute .bld path or "synthetic:..."), zero-padded to 4 bytes
Frames: n_frames x
  tick         u32
  time         f32
  transforms   f32[n_chunks][16]   row-major 4x4, maps the chunk's *building-space* hull/mesh
                                   vertices (as stored in the .bld) to world space
  alive        u8[n_chunks]        1 = chunk exists (static, falling or frozen), 0 = gone
  padding      0..3 bytes to a multiple of 4
```

Frame 0 is the state before the first step; frame i (i >= 1) follows step `i*every`.
Frame size = `8 + 65*n_chunks + pad(n_chunks)`. Chunks that are gone keep their last transform.
Synthetic buildings have no render mesh (`MIDX` empty); draw their hulls (`HVRT`) instead.

Python reader:

```python
import struct, numpy as np
def read_transforms(path):
    raw = open(path, "rb").read()
    magic, ver, n_frames, n, n_bld, dt, every, _ = struct.unpack_from("<4sIIIIfII", raw, 0)
    assert magic == b"RBTF" and ver == 1
    off, blds = 32, []
    for _ in range(n_bld):
        c0, nc, ln = struct.unpack_from("<III", raw, off); off += 12
        blds.append((c0, nc, raw[off:off + ln].decode())); off += (ln + 3) // 4 * 4
    fsz = 8 + n * 65 + (4 - n % 4) % 4
    xf = np.empty((n_frames, n, 4, 4), np.float32); alive = np.empty((n_frames, n), bool); ticks = []
    for f in range(n_frames):
        ticks.append(struct.unpack_from("<If", raw, off))
        xf[f] = np.frombuffer(raw, np.float32, n * 16, off + 8).reshape(n, 4, 4)
        alive[f] = np.frombuffer(raw, np.uint8, n, off + 8 + n * 64) != 0
        off += fsz
    return dict(dt=dt, every=every, buildings=blds, ticks=ticks, xf=xf, alive=alive)
```

World position of a vertex `v` of chunk `c` at frame `f`: `xf[f, c] @ [v, 1]`.

## bench

Builds an arena of synthetic buildings (alternating 20x20x10 blocks of 1 m cubes = 4000 chunks,
and 8-floor towers), then reports: single-building load time, arena load, first tick (initial
stress solve), steady-state tick with 12 AR shots per tick, and the collapse window (600 ticks
after blowing out 4 tower bases and the bottom layer of 4 blocks) with a per-stage breakdown of
the worst tick. `--no-bending` disables the stress bending term (axial-only).
Set `RUBBLE_TRACE=1` to print the collapse timeline. Use `RAYON_NUM_THREADS=8` to emulate 8 cores.
