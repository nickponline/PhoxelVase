//! `rubble` Python module (DESIGN.md §3.6): pyo3 bindings over `rubble-core` / `rubble-stress`.
//!
//! ```python
//! import rubble
//! w = rubble.World({"stress": {"bending": True}})
//! b = w.load_building("fixtures/two_box.bld")
//! w.add_ground_plane(0.0)
//! w.explode((1, 0.5, 0.5), 3.0, damage=1e6)
//! w.step(1/60, n=120)
//! xf = w.chunk_world_transforms(b)   # (N,4,4) f32, building-space -> world
//! rep = rubble.static_stress_report("fixtures/two_box.bld")
//! ```
//!
//! Structured values (config, weapons, events, stats) cross the boundary as JSON so they follow
//! the serde layouts of `rubble-core` exactly (same as `rubble-sim`'s events.jsonl).
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{PyIndexError, PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rubble_core::{
    BuildingId, ChunkState, Explosion, Isometry, Projectile, Weapon, WorldConfig,
};
use rubble_format::{Bld, F_COSMETIC_ATTACHED, F_GLASS};
use rubble_stress::{static_report, StaticReport, StressConfig, StressGraph, StressInput};

fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let json = py.import("json")?;
    json.call_method1("dumps", (obj,))?.extract()
}

fn json_to_py<'py>(py: Python<'py>, s: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("json")?.call_method1("loads", (s,))
}

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Stress graph exactly as `rubble_core::Building::new` builds it.
fn stress_graph_of(bld: &Bld) -> StressGraph {
    let pairs: Vec<(u32, u32)> = bld.edges.iter().map(|e| (e.a, e.b)).collect();
    let cap: Vec<f32> = bld.edges.iter().map(|e| e.strength).collect();
    let cen: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.centroid).collect();
    let nrm: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.normal).collect();
    let pos: Vec<[f32; 3]> = bld.chunks.iter().map(|c| c.com).collect();
    let area: Vec<f32> = bld.edges.iter().map(|e| e.area).collect();
    let elem: Vec<u32> = bld.chunks.iter().map(|c| c.elem).collect();
    StressGraph::new(bld.chunks.len(), &pairs, &cap, &cen, &nrm, &pos).with_areas(&area).with_node_groups(&elem)
}

fn report_to_dict<'py>(py: Python<'py>, r: StaticReport, g: &StressGraph) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("max_util", r.max_util)?;
    let worst = PyList::empty(py);
    for (e, u) in &r.worst_edges {
        let (a, b) = g.edge_nodes(*e);
        worst.append((*e, *u, a, b))?;
    }
    d.set_item("worst_edges", worst)?; // (edge, utilization, chunk_a, chunk_b), descending
    d.set_item("unsupported_nodes", r.unsupported_nodes)?;
    d.set_item("converged", r.converged)?;
    d.set_item("iters", r.iters)?;
    d.set_item("total_load", r.total_load)?;
    d.set_item("anchor_flow", r.anchor_flow)?;
    d.set_item("utilization", PyArray1::from_vec(py, r.utilization))?;
    Ok(d)
}

fn stress_cfg(bending: bool, bend_scale: f32) -> StressConfig {
    StressConfig { bending, bend_scale, ..StressConfig::default() }
}

/// Offline static analysis of an undamaged building (bgen validation, DESIGN §2.7).
///
/// Returns dict(max_util, worst_edges=[(edge, util, a, b)], unsupported_nodes, converged, iters,
/// total_load, anchor_flow, utilization=ndarray f32 (E,)). Loads are m·g of structural chunks
/// (glass / cosmetic chunks carry nothing, as in the engine).
#[pyfunction]
#[pyo3(signature = (path, bending=true, bend_scale=1.0, gravity=9.81))]
fn static_stress_report<'py>(
    py: Python<'py>,
    path: &str,
    bending: bool,
    bend_scale: f32,
    gravity: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let bld = Bld::load(path).map_err(|e| PyIOError::new_err(format!("{path}: {e}")))?;
    let n = bld.chunks.len();
    let structural: Vec<bool> =
        bld.chunks.iter().map(|c| c.flags & (F_GLASS | F_COSMETIC_ATTACHED) == 0).collect();
    let weight: Vec<f32> =
        (0..n).map(|c| if structural[c] { bld.chunks[c].mass * gravity } else { 0.0 }).collect();
    let anchor: Vec<bool> = (0..n).map(|c| bld.is_anchor(c)).collect();
    let edge_alive: Vec<bool> =
        bld.edges.iter().map(|e| structural[e.a as usize] && structural[e.b as usize]).collect();
    let g = stress_graph_of(&bld);
    let input = StressInput { node_weight: &weight, node_alive: &structural, anchor: &anchor, edge_alive: &edge_alive };
    let r = static_report(&g, &input, &stress_cfg(bending, bend_scale));
    report_to_dict(py, r, &g)
}

/// Destruction world (wraps `rubble_core::World`).
#[pyclass(unsendable, module = "rubble")]
struct World {
    w: rubble_core::World,
}

impl World {
    fn bidx(&self, b: u32) -> PyResult<usize> {
        if (b as usize) < self.w.buildings.len() {
            Ok(b as usize)
        } else {
            Err(PyIndexError::new_err(format!("building {b} out of range ({} loaded)", self.w.buildings.len())))
        }
    }
}

#[pymethods]
impl World {
    /// `config`: optional (partial) dict merged over `WorldConfig::default()` (serde field names,
    /// e.g. {"collapse_delay_min": 0.2, "stress": {"bending": False}}).
    #[new]
    #[pyo3(signature = (config=None))]
    fn new(py: Python<'_>, config: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let cfg: WorldConfig = match config {
            Some(c) if !c.is_none() => serde_json::from_str(&py_to_json(py, c)?).map_err(to_py_err)?,
            _ => WorldConfig::default(),
        };
        Ok(World { w: rubble_core::World::new(cfg) })
    }

    /// Effective configuration as a dict.
    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &serde_json::to_string(&self.w.cfg).map_err(to_py_err)?)
    }

    #[getter]
    fn time(&self) -> f32 {
        self.w.time
    }

    #[getter]
    fn tick(&self) -> u64 {
        self.w.tick
    }

    #[getter]
    fn n_buildings(&self) -> usize {
        self.w.buildings.len()
    }

    /// Load a `.bld`, placed by yaw (radians, about +Z) then translation. Returns the building id.
    #[pyo3(signature = (path, pos=(0.0, 0.0, 0.0), yaw=0.0))]
    fn load_building(&mut self, path: &str, pos: (f32, f32, f32), yaw: f32) -> PyResult<u32> {
        let iso = Isometry::new([pos.0, pos.1, pos.2], yaw);
        self.w
            .load_building(path, iso)
            .map(|b| b.0)
            .map_err(|e| PyIOError::new_err(format!("{path}: {e}")))
    }

    #[pyo3(signature = (z=0.0))]
    fn add_ground_plane(&mut self, z: f32) {
        self.w.add_ground_plane(z);
    }

    /// Fire a projectile. `weapon`: 'ar' | 'smg' | 'sniper' | 'shotgun' | 'launcher' or
    /// {'custom': {...WeaponParams}}. Hitscan unless `ballistic=True`.
    #[pyo3(signature = (origin, dir, weapon=None, ballistic=false))]
    fn fire(
        &mut self,
        py: Python<'_>,
        origin: (f32, f32, f32),
        dir: (f32, f32, f32),
        weapon: Option<&Bound<'_, PyAny>>,
        ballistic: bool,
    ) -> PyResult<()> {
        let wp: Weapon = match weapon {
            Some(o) if !o.is_none() => serde_json::from_str(&py_to_json(py, o)?)
                .map_err(|e| PyValueError::new_err(format!("bad weapon: {e}")))?,
            _ => Weapon::Ar,
        };
        let (o, d) = ([origin.0, origin.1, origin.2], [dir.0, dir.1, dir.2]);
        self.w.fire(if ballistic { Projectile::ballistic(o, d, wp) } else { Projectile::hitscan(o, d, wp) });
        Ok(())
    }

    #[pyo3(signature = (center, radius=4.0, damage=400.0, impulse=3000.0, inner_radius=1.0))]
    fn explode(&mut self, center: (f32, f32, f32), radius: f32, damage: f32, impulse: f32, inner_radius: f32) {
        self.w.explode(Explosion { center: [center.0, center.1, center.2], radius, inner_radius, damage, impulse });
    }

    /// Queue damage on a chunk (applied in the next step). `chunk` may be an int or a list of ints.
    #[pyo3(signature = (b, chunk, amount=1.0e9))]
    fn damage_chunk(&mut self, b: u32, chunk: &Bound<'_, PyAny>, amount: f32) -> PyResult<()> {
        let bi = self.bidx(b)?;
        let n = self.w.buildings[bi].n_chunks();
        let ids: Vec<u32> = if let Ok(c) = chunk.extract::<u32>() { vec![c] } else { chunk.extract()? };
        for c in ids {
            if c as usize >= n {
                return Err(PyIndexError::new_err(format!("chunk {c} out of range ({n})")));
            }
            self.w.damage_chunk(BuildingId(b), c, amount);
        }
        Ok(())
    }

    /// Advance `n` fixed steps of `dt` seconds. Events accumulate until `drain_events`.
    #[pyo3(signature = (dt=1.0/60.0, n=1))]
    fn step(&mut self, dt: f32, n: usize) {
        for _ in 0..n {
            self.w.step(dt);
        }
    }

    /// Engine events as dicts (`type` key + fields, same as rubble-sim events.jsonl).
    fn drain_events<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let ev = self.w.drain_events();
        json_to_py(py, &serde_json::to_string(&ev).map_err(to_py_err)?)
    }

    fn n_chunks(&self, b: u32) -> PyResult<usize> {
        Ok(self.w.buildings[self.bidx(b)?].n_chunks())
    }

    /// (N,4,4) float32 row-major building-space -> world transform per chunk.
    fn chunk_world_transforms<'py>(&self, py: Python<'py>, b: u32) -> PyResult<Bound<'py, PyAny>> {
        let n = self.w.buildings[self.bidx(b)?].n_chunks();
        let xf = self.w.chunk_world_transforms(BuildingId(b));
        let flat: Vec<f32> = xf.into_iter().flatten().collect();
        Ok(PyArray1::from_vec(py, flat).reshape([n, 4, 4])?.into_any())
    }

    /// (N,) bool: chunk exists (static, detaching, falling or frozen).
    fn alive<'py>(&self, py: Python<'py>, b: u32) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let bd = &self.w.buildings[self.bidx(b)?];
        Ok(PyArray1::from_vec(py, bd.state.iter().map(|s| *s != ChunkState::Gone).collect()))
    }

    /// (N,) u8 chunk state: 0 static, 1 detaching, 2 in cluster, 3 frozen, 4 gone (ChunkState::code).
    fn chunk_state<'py>(&self, py: Python<'py>, b: u32) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let bd = &self.w.buildings[self.bidx(b)?];
        Ok(PyArray1::from_vec(py, bd.state.iter().map(|s| s.code()).collect()))
    }

    /// (N,) f32 chunk hit points.
    fn chunk_hp<'py>(&self, py: Python<'py>, b: u32) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_vec(py, self.w.buildings[self.bidx(b)?].hp.clone()))
    }

    /// (E,) bool edge alive.
    fn edge_alive<'py>(&self, py: Python<'py>, b: u32) -> PyResult<Bound<'py, PyArray1<bool>>> {
        Ok(PyArray1::from_vec(py, self.w.buildings[self.bidx(b)?].edge_alive.clone()))
    }

    /// (E,) f32 latest stress utilization per edge (0 for dead / unsolved edges).
    fn edge_utilization<'py>(&self, py: Python<'py>, b: u32) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_vec(py, self.w.buildings[self.bidx(b)?].utilization.clone()))
    }

    /// Full static solve of building `b` in its *current* state (static structural chunks and
    /// alive edges only). Same dict as the module-level `static_stress_report`.
    #[pyo3(signature = (b, bending=None))]
    fn static_stress_report<'py>(&self, py: Python<'py>, b: u32, bending: Option<bool>) -> PyResult<Bound<'py, PyDict>> {
        let bd = &self.w.buildings[self.bidx(b)?];
        let node_alive: Vec<bool> =
            (0..bd.n_chunks()).map(|c| bd.state[c] == ChunkState::Static && bd.structural[c]).collect();
        let edge_alive: Vec<bool> = bd
            .bld
            .edges
            .iter()
            .enumerate()
            .map(|(e, ed)| bd.edge_alive[e] && node_alive[ed.a as usize] && node_alive[ed.b as usize])
            .collect();
        let input = StressInput { node_weight: &bd.weight, node_alive: &node_alive, anchor: &bd.anchor, edge_alive: &edge_alive };
        let s = &self.w.cfg.stress;
        let r = static_report(&bd.stress_graph, &input, &stress_cfg(bending.unwrap_or(s.bending), s.bend_scale));
        report_to_dict(py, r, &bd.stress_graph)
    }

    /// World counters (Stats) plus `time` and per-stage `timings` (ms) of the last step.
    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut v = serde_json::to_value(self.w.stats()).map_err(to_py_err)?;
        if let Some(o) = v.as_object_mut() {
            o.insert("time".into(), self.w.time.into());
            o.insert("timings".into(), serde_json::to_value(self.w.timings).map_err(to_py_err)?);
        }
        json_to_py(py, &v.to_string())
    }

    fn __repr__(&self) -> String {
        let s = self.w.stats();
        format!(
            "rubble.World(buildings={}, chunks={}, clusters={}, tick={}, time={:.3})",
            s.buildings, s.chunks_total, s.clusters, s.tick, self.w.time
        )
    }
}

#[pymodule]
fn rubble(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<World>()?;
    m.add_function(wrap_pyfunction!(static_stress_report, m)?)?;
    m.add("BLD_VERSION", rubble_format::BLD_VERSION)?;
    m.add("CHUNK_STATES", ("static", "detaching", "in_cluster", "frozen", "gone"))?;
    Ok(())
}
