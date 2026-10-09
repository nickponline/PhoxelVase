//! The destruction world and its fixed-step tick pipeline (DESIGN §3.3–3.5).
use crate::building::{Building, ChunkState, ClusterKey};
use crate::config::*;
use crate::events::*;
use crate::math::*;
use crate::physics::*;
use rayon::prelude::*;
use rubble_format::{Bld, BldError, F_NO_DEBRIS};
use rubble_stress::{solve_step_lean, StressConfig, StressInput};
use slotmap::{Key, KeyData, SlotMap};
use std::path::Path;
use std::time::Instant;

const KIND_CHUNK: u128 = 1;
/// What holds a piece of rubble up (see `World::rests_on_something`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Support {
    /// nothing underneath
    None,
    /// only things that may move away: dynamic bodies, chunks waiting to collapse
    Transient,
    /// the ground, the static building or frozen rubble
    Solid,
}

/// Rubble with nothing under it (see `World::floating_report`).
#[derive(Clone, Debug)]
pub struct FloatingGroup {
    pub building: BuildingId,
    pub chunks: Vec<u32>,
    /// lowest chunk centre of mass (world z)
    pub min_com_z: f32,
    /// colliders outside the group it touches where it is (0: really in mid-air; otherwise it
    /// is wedged between things beside it, held by friction)
    pub touching: usize,
}

#[derive(Clone, Debug, Default)]
pub struct FloatingReport {
    /// frozen (static) rubble groups with no support
    pub frozen: Vec<FloatingGroup>,
    /// resting dynamic clusters with no support (e.g. asleep in mid-air)
    pub clusters: Vec<(ClusterId, FloatingGroup)>,
}

impl FloatingReport {
    pub fn is_empty(&self) -> bool {
        self.frozen.is_empty() && self.clusters.is_empty()
    }
}

/// Smallest group of support-lost boxes pre-tested together (see `thaw_unsupported`).
const THAW_BOX_GROUP: usize = 16;
/// Row count from which background impact solves run their CG kernels on rayon.
const IMPACT_PAR_THRESHOLD: usize = 4096;
const KIND_CLUSTER: u128 = 2;
const KIND_GROUND: u128 = 3;

fn chunk_tag(b: u32, c: u32) -> u128 {
    (KIND_CHUNK << 64) | ((b as u128) << 32) | c as u128
}
fn cluster_tag(k: ClusterKey) -> u128 {
    (KIND_CLUSTER << 64) | k.data().as_ffi() as u128
}
fn tag_kind(t: u128) -> u128 {
    t >> 64
}
fn tag_chunk(t: u128) -> (u32, u32) {
    (((t >> 32) & 0xffff_ffff) as u32, (t & 0xffff_ffff) as u32)
}
fn tag_cluster(t: u128) -> ClusterKey {
    KeyData::from_ffi(t as u64).into()
}
pub fn cluster_id(k: ClusterKey) -> ClusterId {
    ClusterId(k.data().as_ffi())
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Bld(#[from] BldError),
}

/// A detached group of chunks simulated as one dynamic compound body.
pub struct Cluster {
    pub building: u32,
    /// chunk ids; also the compound sub-shape order
    pub chunks: Vec<u32>,
    /// internal (still alive) edge ids
    pub edges: Vec<u32>,
    pub edge_impulse: Vec<f32>,
    pub body: BodyId,
    pub collider: ColliderId,
    pub mass: f32,
    pub local_com: Vec3,
    pub age: f32,
    pub rest_time: f32,
    pub debris: bool,
    pub dirty: bool,
    /// world time of the last impact-stress evaluation
    pub last_impact: f32,
    /// accumulated landing velocity change (decays over `impact_window`)
    pub impact_dv: Vec3,
    /// internal edges broken since the last rebuild (lets a rebuild prove "still one piece"
    /// without walking the whole cluster)
    pub broken: Vec<u32>,
    /// known to be one connected piece (set by a full rebuild; a new cluster may not be, e.g.
    /// after its glass shattered)
    pub connected: bool,
}

struct ImpactResult {
    building: u32,
    broken: Vec<u32>,
    crushed: Vec<u32>,
}

struct PendingImpact {
    due_tick: u64,
    rx: std::sync::Mutex<std::sync::mpsc::Receiver<ImpactResult>>,
}

/// Self-contained impact solve (runs on a worker thread).
struct ImpactJob {
    building: u32,
    graph: std::sync::Arc<rubble_stress::StressGraph>,
    inter_panel: std::sync::Arc<Vec<bool>>,
    weight: Vec<f32>,
    alive: Vec<bool>,
    anchor: Vec<bool>,
    edge_alive: Vec<bool>,
    edges: Vec<u32>,
    scfg: StressConfig,
    joint: f32,
    rounds: usize,
    max_breaks: usize,
    debug: Option<String>,
}

/// `RUBBLE_DEBUG_IMPACT` is set (read once; the environment lookup takes a lock).
fn debug_impact() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("RUBBLE_DEBUG_IMPACT").is_ok())
}

/// Workers for background impact solves (persistent: no thread spawn per job). A dedicated
/// pool keeps them off the global rayon pool, which the physics step uses.
fn impact_pool() -> &'static rayon::ThreadPool {
    static POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| rayon::ThreadPoolBuilder::new().num_threads(4).thread_name(|i| format!("rubble-impact-{i}")).build().unwrap())
}

fn run_impact_job(mut j: ImpactJob) -> ImpactResult {
    let t0 = Instant::now();
    let mut broken: Vec<u32> = vec![];
    let mut st = rubble_stress::StressState::new(&j.graph);
    for round in 0..j.rounds {
        let input = StressInput {
            node_weight: &j.weight,
            node_alive: &j.alive,
            anchor: &j.anchor,
            edge_alive: &j.edge_alive,
            ..Default::default()
        };
        let mut r = solve_step_lean(&j.graph, &mut st, &input, &j.scfg, 0.0);
        for _ in 0..2 {
            if r.converged {
                break;
            }
            r = solve_step_lean(&j.graph, &mut st, &input, &j.scfg, 0.0);
        }
        let util = st.utilization();
        let mut over: Vec<(f32, u32)> = j
            .edges
            .iter()
            .filter(|&&e| j.edge_alive[e as usize])
            .filter_map(|&e| {
                let u = util[e as usize] / if j.inter_panel[e as usize] { j.joint } else { 1.0 };
                (u > 1.0).then_some((u, e))
            })
            .collect();
        over.sort_by(|a, b| b.0.total_cmp(&a.0));
        if let Some(d) = &j.debug {
            eprintln!("impact {d} round {round} conv={} max_u={:.2} over={}", r.converged, over.first().map_or(0.0, |x| x.0), over.len());
        }
        if over.is_empty() {
            break;
        }
        for &(_, e) in over.iter().take(j.max_breaks.saturating_sub(broken.len())) {
            j.edge_alive[e as usize] = false;
            broken.push(e);
        }
        if broken.len() >= j.max_breaks {
            break;
        }
    }
    if let Some(d) = &j.debug {
        eprintln!("impact {d} solved in {:.2} ms, {} bonds broken", t0.elapsed().as_secs_f32() * 1e3, broken.len());
    }
    ImpactResult { building: j.building, broken, crushed: vec![] }
}

#[derive(Default)]
struct ExplosionFx {
    damage: Vec<(u32, u32, f32)>,
    edge_damage: Vec<(u32, u32, f32)>,
    /// (building, chunk, impulse, world point)
    impulses: Vec<(u32, u32, Vec3, Vec3)>,
}

struct PendingCollapse {
    building: u32,
    chunks: Vec<u32>,
    timer: f32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct StepTimings {
    pub projectiles_ms: f32,
    pub damage_ms: f32,
    pub connectivity_ms: f32,
    pub stress_ms: f32,
    pub promotion_ms: f32,
    pub physics_ms: f32,
    pub impacts_ms: f32,
    pub settle_ms: f32,
    /// time spent blocked on a due background impact solve (headless back-to-back runs only;
    /// at real-time tick rates the solve finishes within its latency)
    pub impact_wait_ms: f32,
    pub total_ms: f32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct Stats {
    pub tick: u64,
    pub buildings: usize,
    pub chunks_total: usize,
    pub static_chunks: usize,
    pub frozen_chunks: usize,
    pub gone_chunks: usize,
    pub clusters: usize,
    pub chunks_in_flight: usize,
    pub pending_collapses: usize,
    pub rapier_bodies: usize,
    pub rapier_colliders: usize,
    pub stress_iters: usize,
    /// chunks in the largest dynamic cluster
    pub largest_cluster: usize,
    /// chunks the full ground check had to send down because the incremental search missed
    /// them (should stay 0; anything else is a bug worth reporting)
    pub ground_check_catches: usize,
}

/// Debug/inspection view of a building.
#[derive(Clone, Debug)]
pub struct BuildingState {
    pub alive: Vec<u64>,
    pub state: Vec<u8>,
    pub hp: Vec<f32>,
    pub edge_alive: Vec<u64>,
    pub edge_health: Vec<f32>,
    pub utilization: Vec<f32>,
}

/// A queued cutting beam, see [`World::beam`].
#[derive(Clone, Copy, Debug)]
struct Beam {
    o: Vec3,
    /// unit direction
    d: Vec3,
    range: f32,
    radius: f32,
    dmg: f32,
    imp: f32,
}

pub struct World {
    pub cfg: WorldConfig,
    pub phys: RapierBackend,
    pub buildings: Vec<Building>,
    pub clusters: SlotMap<ClusterKey, Cluster>,
    projectiles: Vec<Projectile>,
    /// queued beams
    beams: Vec<Beam>,
    explosions: Vec<Explosion>,
    damage: Vec<(u32, u32, f32)>,
    edge_damage: Vec<(u32, u32, f32)>,
    pending: Vec<PendingCollapse>,
    events: Vec<Event>,
    pub time: f32,
    pub tick: u64,
    pub timings: StepTimings,
    stress_iters: usize,
    needs_sync: bool,
    pending_impacts: Vec<PendingImpact>,
    last_dt: f32,
    /// world AABBs of geometry removed this tick: whatever rested on it must be re-checked
    support_lost: Vec<(Vec3, Vec3)>,
    /// round-robin cursor (building, chunk) of the frozen-support sweep
    sweep_cursor: (usize, usize),
    /// frozen groups (building, any chunk) resting on something that may move away (a dynamic
    /// body, a collapsing piece): re-checked every tick until their support is solid
    support_watch: Vec<(u32, u32)>,
    ground_check_catches: usize,
    // scratch
    contacts: Vec<ContactImpulse>,
    scratch_u32: Vec<u32>,
    scratch_hits: Vec<(ClusterKey, u32, Vec3)>,
}

/// Interleave the low 10 bits of x, y, z (Morton order).
fn morton3(x: u32, y: u32, z: u32) -> u32 {
    fn spread(mut v: u32) -> u32 {
        v &= 0x3ff;
        v = (v | (v << 16)) & 0x0300_00ff;
        v = (v | (v << 8)) & 0x0300_f00f;
        v = (v | (v << 4)) & 0x030c_30c3;
        (v | (v << 2)) & 0x0924_9249
    }
    spread(x) | (spread(y) << 1) | (spread(z) << 2)
}

pub(crate) fn bits_of(v: impl Iterator<Item = bool>, n: usize) -> Vec<u64> {
    let mut out = vec![0u64; (n + 63) / 64];
    for (i, b) in v.enumerate() {
        if b {
            out[i / 64] |= 1 << (i % 64);
        }
    }
    out
}

impl World {
    pub fn new(cfg: WorldConfig) -> Self {
        let mut phys = RapierBackend::new(v3(cfg.gravity));
        phys.set_ccd(cfg.ccd);
        phys.set_solver_iterations(cfg.solver_iterations);
        phys.set_threads(cfg.physics_threads);
        let stress_iters = cfg.stress.max_iters;
        World {
            cfg,
            phys,
            buildings: vec![],
            clusters: SlotMap::with_key(),
            projectiles: vec![],
            beams: vec![],
            explosions: vec![],
            damage: vec![],
            edge_damage: vec![],
            pending: vec![],
            events: vec![],
            time: 0.0,
            tick: 0,
            timings: StepTimings::default(),
            stress_iters,
            needs_sync: false,
            pending_impacts: vec![],
            last_dt: 1.0 / 60.0,
            support_lost: vec![],
            sweep_cursor: (0, 0),
            support_watch: vec![],
            ground_check_catches: 0,
            contacts: vec![],
            scratch_u32: vec![],
            scratch_hits: vec![],
        }
    }

    // ------------------------------------------------------------------ loading

    pub fn load_building(&mut self, path: impl AsRef<Path>, iso: Isometry) -> Result<BuildingId, LoadError> {
        let bld = Bld::load(path)?;
        Ok(self.load_building_bld(bld, iso))
    }

    pub fn load_building_bld(&mut self, bld: Bld, iso: Isometry) -> BuildingId {
        let pose = iso.to_pose();
        let g = v3(self.cfg.gravity).length();
        let mut b = Building::new(bld, pose, self.cfg.edge_health_per_newton, g);
        let bi = self.buildings.len() as u32;
        for c in 0..b.n_chunks() {
            let (h, aabb) = self.phys.add_static_collider(b.shapes[c].clone(), pose, chunk_tag(bi, c as u32));
            b.collider[c] = Some(h);
            b.collider_aabb[c] = aabb;
        }
        // full connectivity check once at load
        b.dirty = (0..b.n_chunks() as u32).collect();
        let det = b.find_detached();
        b.stress_active = true;
        self.buildings.push(b);
        // Chunks not connected to any anchor in the *intact* file are generator defects (e.g.
        // slivers hanging only on glass). Keep them as anchored static rubble (state Frozen) so an
        // untouched building never sheds pieces; `Building::load_floating` records how many.
        {
            let b = &mut self.buildings[bi as usize];
            for comp in det {
                for c in comp {
                    b.state[c as usize] = ChunkState::Frozen;
                    b.load_floating += 1;
                }
            }
        }
        self.settle_stress_at_load(bi as usize);
        self.needs_sync = true;
        BuildingId(bi)
    }

    pub fn add_ground_plane(&mut self, z: f32) {
        self.phys.add_ground_plane(z, KIND_GROUND << 64);
        self.needs_sync = true;
    }

    // ------------------------------------------------------------------ inputs

    pub fn fire(&mut self, p: Projectile) {
        self.projectiles.push(p);
    }
    /// A cutting beam for one tick (call it every tick while the beam is held). Every chunk the
    /// capsule of `radius` around the segment `origin + t·dir, t ∈ [0, range]` touches takes
    /// `damage` (× its material multiplier), with no penetration limit; dynamic pieces it touches
    /// get `impulse` (N·s) along the beam. `radius` 0 is a plain ray. Applied in the next tick's
    /// projectile stage.
    pub fn beam(&mut self, origin: [f32; 3], dir: [f32; 3], range: f32, radius: f32, damage: f32, impulse: f32) {
        let d = v3(dir).normalize_or_zero();
        if d != Vec3::ZERO && range > 0.0 {
            self.beams.push(Beam { o: v3(origin), d, range, radius: radius.max(0.0), dmg: damage, imp: impulse });
        }
    }
    pub fn explode(&mut self, e: Explosion) {
        self.explosions.push(e);
    }
    pub fn damage_chunk(&mut self, b: BuildingId, chunk: u32, amount: f32) {
        self.damage.push((b.0, chunk, amount));
    }
    /// Release every pending collapse of building `b` on the next tick instead of after its
    /// `collapse_delay` (e.g. a scripted demolition, where the warning delay only reads as lag).
    pub fn hurry_collapse(&mut self, b: BuildingId) {
        for p in self.pending.iter_mut().filter(|p| p.building == b.0) {
            p.timer = 0.0;
        }
    }
    pub fn drain_events(&mut self) -> Vec<Event> {
        std::mem::take(&mut self.events)
    }

    // ------------------------------------------------------------------ queries

    pub fn chunk_world_pose(&self, b: usize, c: usize) -> Pose {
        let bd = &self.buildings[b];
        match bd.state[c] {
            ChunkState::Static | ChunkState::Detaching => bd.pose,
            ChunkState::InCluster(k) => self.phys.body_state(self.clusters[k].body).pose,
            ChunkState::Frozen | ChunkState::Gone => bd.chunk_pose[c],
        }
    }

    pub fn chunk_world_com(&self, b: usize, c: usize) -> Vec3 {
        self.chunk_world_pose(b, c).transform_point(v3(self.buildings[b].bld.chunks[c].com))
    }

    /// Row-major 4x4 building-space -> world transform per chunk.
    pub fn chunk_world_transforms(&self, b: BuildingId) -> Vec<[f32; 16]> {
        let mut out = vec![[0.0; 16]; self.buildings[b.0 as usize].n_chunks()];
        self.chunk_world_transforms_into(b, &mut out);
        out
    }

    pub fn chunk_world_transforms_into(&self, b: BuildingId, out: &mut [[f32; 16]]) {
        let bd = &self.buildings[b.0 as usize];
        let base = pose_to_rowmajor(&bd.pose);
        for c in 0..bd.n_chunks() {
            out[c] = match bd.state[c] {
                ChunkState::Static | ChunkState::Detaching => base,
                _ => pose_to_rowmajor(&self.chunk_world_pose(b.0 as usize, c)),
            };
        }
    }

    pub fn building_state(&self, b: BuildingId) -> BuildingState {
        let bd = &self.buildings[b.0 as usize];
        BuildingState {
            alive: bits_of(bd.state.iter().map(|s| *s != ChunkState::Gone), bd.n_chunks()),
            state: bd.state.iter().map(|s| s.code()).collect(),
            hp: bd.hp.clone(),
            edge_alive: bits_of(bd.edge_alive.iter().copied(), bd.edge_alive.len()),
            edge_health: bd.edge_health.clone(),
            utilization: bd.utilization.clone(),
        }
    }

    pub fn building(&self, b: BuildingId) -> &Building {
        &self.buildings[b.0 as usize]
    }

    pub fn cluster_of(&self, b: BuildingId, chunk: u32) -> Option<ClusterId> {
        match self.buildings[b.0 as usize].state[chunk as usize] {
            ChunkState::InCluster(k) => Some(cluster_id(k)),
            _ => None,
        }
    }

    pub fn cluster_state(&self, id: ClusterId) -> Option<(&Cluster, BodyState)> {
        let k: ClusterKey = KeyData::from_ffi(id.0).into();
        self.clusters.get(k).map(|c| (c, self.phys.body_state(c.body)))
    }

    pub fn stats(&self) -> Stats {
        let mut s = Stats {
            tick: self.tick,
            buildings: self.buildings.len(),
            clusters: self.clusters.len(),
            pending_collapses: self.pending.len(),
            rapier_bodies: self.phys.num_bodies(),
            rapier_colliders: self.phys.num_colliders(),
            stress_iters: self.stress_iters,
            largest_cluster: self.clusters.values().map(|c| c.chunks.len()).max().unwrap_or(0),
            ground_check_catches: self.ground_check_catches,
            ..Default::default()
        };
        for b in &self.buildings {
            s.chunks_total += b.n_chunks();
            for st in &b.state {
                match st {
                    ChunkState::Static | ChunkState::Detaching => s.static_chunks += 1,
                    ChunkState::Frozen => s.frozen_chunks += 1,
                    ChunkState::Gone => s.gone_chunks += 1,
                    ChunkState::InCluster(_) => s.chunks_in_flight += 1,
                }
            }
        }
        s
    }

    // ------------------------------------------------------------------ tick

    pub fn step(&mut self, dt: f32) {
        let t_start = Instant::now();
        self.last_dt = dt;
        self.timings.impact_wait_ms = 0.0;
        if self.needs_sync {
            self.phys.sync_queries();
            self.needs_sync = false;
        }
        let mut t = Instant::now();
        let lap = |t: &mut Instant| {
            let e = t.elapsed().as_secs_f32() * 1e3;
            *t = Instant::now();
            e
        };
        // 1. projectiles + explosions
        self.step_projectiles(dt);
        self.process_explosions();
        self.timings.projectiles_ms = lap(&mut t);
        // 2-3. damage + removal
        self.apply_damage();
        self.timings.damage_ms = lap(&mut t);
        // 4. connectivity
        self.connectivity();
        self.timings.connectivity_ms = lap(&mut t);
        // 5. stress (+ connectivity again)
        if self.step_stress(dt) {
            self.connectivity();
        }
        if self.step_tipping(dt) | self.step_slivers() {
            self.connectivity();
        }
        self.step_ground_check(dt);
        self.timings.stress_ms = lap(&mut t);
        // 6. promotion
        self.promote(dt);
        self.rebuild_dirty_clusters();
        self.timings.promotion_ms = lap(&mut t);
        // 7. physics (first wake / thaw anything whose support was removed this tick)
        self.thaw_unsupported();
        self.phys.step(dt);
        self.timings.physics_ms = lap(&mut t);
        // 8. impacts
        self.impacts();
        self.timings.impacts_ms = lap(&mut t);
        // 9-10. settling + budgets
        self.settle(dt);
        self.budgets();
        self.timings.settle_ms = lap(&mut t);
        self.time += dt;
        self.tick += 1;
        self.timings.total_ms = t_start.elapsed().as_secs_f32() * 1e3;
    }

    // ------------------------------------------------------------------ 1. projectiles

    fn step_projectiles(&mut self, dt: f32) {
        let mut ps = std::mem::take(&mut self.projectiles);
        let g = v3(self.cfg.gravity);
        ps.retain_mut(|p| match p.kind {
            ProjectileKind::Hitscan => {
                let dir = v3(p.vel).normalize_or_zero();
                if dir != Vec3::ZERO {
                    self.trace_hitscan(v3(p.pos), dir, &p.weapon);
                }
                false
            }
            ProjectileKind::Ballistic => {
                let n = self.cfg.ballistic_substeps.max(1);
                let h = dt / n as f32;
                let mut pos = v3(p.pos);
                let mut vel = v3(p.vel);
                for _ in 0..n {
                    vel += g * p.weapon.gravity_scale * h;
                    let delta = vel * h;
                    if let Some(hit) = self.phys.cast_sphere(pos, delta, p.weapon.radius) {
                        let point = pos + delta * hit.toi;
                        self.apply_hit(&hit, point, vel.normalize_or_zero(), p.weapon.damage, &p.weapon);
                        if let Some(mut e) = p.explode {
                            e.center = a3(point);
                            self.explosions.push(e);
                        }
                        return false;
                    }
                    pos += delta;
                }
                p.pos = a3(pos);
                p.vel = a3(vel);
                p.age += dt;
                p.age < 15.0 && pos.z > -100.0
            }
        });
        ps.extend(self.projectiles.drain(..));
        self.projectiles = ps;
        let beams = std::mem::take(&mut self.beams);
        for b in beams {
            self.apply_beam(b);
        }
    }

    /// See [`World::beam`].
    fn apply_beam(&mut self, Beam { o, d, range, radius, dmg, imp }: Beam) {
        // static / frozen chunks are one collider each; a cluster is one compound collider,
        // so its chunks are tested individually below
        let mut tags: Vec<u128> = vec![];
        if radius > 0.0 {
            // overlap the capsule in short pieces: one long diagonal capsule has a huge AABB and
            // the broad phase would hand back half the scene
            let piece = (radius * 8.0).max(4.0);
            let n = (range / piece).ceil().max(1.0) as usize;
            for i in 0..n {
                let (t0, t1) = (range * i as f32 / n as f32, range * (i + 1) as f32 / n as f32);
                let cap = Shape::capsule(o + d * t0, o + d * t1, radius);
                self.phys.shape_overlaps(&cap, Pose::IDENTITY, &mut tags);
            }
            tags.sort_unstable();
            tags.dedup();
        } else {
            let mut hits = vec![];
            self.phys.ray_all(o, d, range, usize::MAX, &mut hits);
            tags.extend(hits.iter().map(|h| h.tag));
        }
        let ray = rapier3d::prelude::Ray::new(o, d);
        let capsule = Shape::capsule(o, o + d * range, radius);
        let mut struck: Vec<(u32, u32)> = vec![];
        let mut pushes: Vec<(BodyId, Vec3)> = vec![];
        for &tag in &tags {
            match tag_kind(tag) {
                KIND_CHUNK => struck.push(tag_chunk(tag)),
                KIND_CLUSTER => {
                    let Some(cl) = self.clusters.get(tag_cluster(tag)) else { continue };
                    let pose = self.phys.body_state(cl.body).pose;
                    let bd = &self.buildings[cl.building as usize];
                    // beam axis in building space, cheap reject against chunk AABBs (grown by
                    // the radius) first
                    let inv = pose.inverse();
                    let (lo_o, lo_d) = (inv.transform_point(o), inv.transform_vector(d));
                    let inv_d = Vec3::ONE / lo_d;
                    // (along-beam distance, chunk world COM) of the nearest chunk struck
                    let mut first = (f32::MAX, Vec3::ZERO);
                    for &c in &cl.chunks {
                        let ch = &bd.bld.chunks[c as usize];
                        let t1 = (v3(ch.aabb_min) - Vec3::splat(radius) - lo_o) * inv_d;
                        let t2 = (v3(ch.aabb_max) + Vec3::splat(radius) - lo_o) * inv_d;
                        let (tn, tf) = (t1.min(t2).max_element().max(0.0), t1.max(t2).min_element().min(range));
                        if tn > tf {
                            continue;
                        }
                        let shape = &*bd.shapes[c as usize];
                        let t = if radius > 0.0 {
                            let hit = rapier3d::parry::query::intersection_test(&pose, shape, &Pose::IDENTITY, &*capsule)
                                .map_or(false, |r| r.intersecting);
                            hit.then_some(tn)
                        } else {
                            shape.cast_ray(&pose, &ray, range, true)
                        };
                        if let Some(t) = t {
                            struck.push((cl.building, c));
                            if t < first.0 {
                                first = (t, pose.transform_point(v3(ch.com)));
                            }
                        }
                    }
                    if imp > 0.0 && first.0 < f32::MAX {
                        // a ray pushes where it enters; a fat beam at the nearest chunk it grazes
                        let at = if radius > 0.0 { first.1 } else { o + d * first.0 };
                        pushes.push((cl.body, at));
                    }
                }
                _ => {}
            }
        }
        for (b, c) in struck {
            let mat = self.buildings[b as usize].bld.chunks[c as usize].material;
            self.damage.push((b, c, dmg * self.cfg.damage_mult(mat)));
        }
        for (body, p) in pushes {
            self.phys.apply_impulse_at_point(body, d * imp, p);
        }
    }

    fn trace_hitscan(&mut self, mut origin: Vec3, dir: Vec3, w: &WeaponParams) {
        let mut energy = w.penetration;
        let mut range = w.range;
        let mut dmg = w.damage;
        for _ in 0..=self.cfg.max_penetrations {
            let Some(hit) = self.phys.cast_ray(origin, dir, range, None) else { return };
            let point = origin + dir * hit.toi;
            let Some((b, c)) = self.apply_hit(&hit, point, dir, dmg, w) else { return };
            if energy <= 0.0 {
                return;
            }
            // thickness along the ray: cast back from beyond the chunk
            let bd = &self.buildings[b as usize];
            let pose = self.chunk_world_pose(b as usize, c as usize);
            let ch = &bd.bld.chunks[c as usize];
            let ext = (v3(ch.aabb_max) - v3(ch.aabb_min)).length() + 0.01;
            let far = point + dir * ext;
            let back = rapier3d::prelude::Ray::new(far, -dir);
            let thick = match bd.shapes[c as usize].cast_ray(&pose, &back, ext, true) {
                Some(t2) => (ext - t2).max(0.0),
                None => 0.0,
            };
            let cost = thick * self.cfg.resistance(ch.material);
            if cost >= energy {
                return;
            }
            dmg *= (energy - cost) / energy;
            energy -= cost;
            origin = point + dir * (thick + 1e-3);
            range -= hit.toi + thick;
            if range <= 0.0 {
                return;
            }
        }
    }

    /// Map a ray/sweep hit to a chunk; for a cluster, the chunk nearest to the point.
    fn resolve_hit(&self, tag: u128, point: Vec3) -> Option<(u32, u32)> {
        match tag_kind(tag) {
            KIND_CHUNK => Some(tag_chunk(tag)),
            KIND_CLUSTER => {
                let k = tag_cluster(tag);
                let cl = self.clusters.get(k)?;
                if cl.chunks.len() == 1 {
                    return Some((cl.building, cl.chunks[0]));
                }
                let pose = self.phys.body_state(cl.body).pose;
                let bd = &self.buildings[cl.building as usize];
                let mut best = (f32::MAX, cl.chunks[0]);
                for &c in &cl.chunks {
                    let d = bd.shapes[c as usize].distance_to_point(&pose, point, true);
                    if d < best.0 {
                        best = (d, c);
                    }
                }
                Some((cl.building, best.1))
            }
            _ => None,
        }
    }

    fn apply_hit(&mut self, hit: &RayHit, point: Vec3, dir: Vec3, dmg: f32, w: &WeaponParams) -> Option<(u32, u32)> {
        let (b, c) = self.resolve_hit(hit.tag, point)?;
        let mat = self.buildings[b as usize].bld.chunks[c as usize].material;
        self.damage.push((b, c, dmg * self.cfg.damage_mult(mat)));
        if w.splash_r > 0.0 && w.splash_frac > 0.0 {
            let bd = &self.buildings[b as usize];
            let mut splash = vec![];
            for (nb, e) in bd.neighbors(c) {
                if bd.edge_alive[e as usize] && bd.alive(nb as usize) {
                    let p = self.chunk_world_com(b as usize, nb as usize);
                    if (p - point).length() <= w.splash_r {
                        let m = self.cfg.damage_mult(bd.bld.chunks[nb as usize].material);
                        splash.push((b, nb, dmg * w.splash_frac * m));
                    }
                }
            }
            self.damage.extend(splash);
        }
        if let ChunkState::InCluster(k) = self.buildings[b as usize].state[c as usize] {
            let body = self.clusters[k].body;
            self.phys.apply_impulse_at_point(body, dir * w.impulse, point);
        }
        Some((b, c))
    }

    fn process_explosions(&mut self) {
        let exps = std::mem::take(&mut self.explosions);
        if exps.is_empty() {
            return;
        }
        // queries are read-only: evaluate all explosions in parallel, then apply in order
        let fx: Vec<ExplosionFx> = {
            let this = &*self;
            exps.par_iter().map(|e| this.explosion_effects(e)).collect()
        };
        for f in fx {
            self.damage.extend(f.damage);
            self.edge_damage.extend(f.edge_damage);
            for (b, c, j, com) in f.impulses {
                let bd = &mut self.buildings[b as usize];
                match bd.state[c as usize] {
                    ChunkState::InCluster(k) => {
                        let body = self.clusters[k].body;
                        self.phys.apply_impulse_at_point(body, j, com);
                    }
                    ChunkState::Static | ChunkState::Detaching => {
                        let cu = c as usize;
                        if self.time - bd.pending_impulse_t[cu] > self.cfg.impulse_memory {
                            bd.pending_impulse[cu] = [0.0; 3];
                        }
                        let p = v3(bd.pending_impulse[cu]) + j;
                        bd.pending_impulse[cu] = a3(p);
                        bd.pending_impulse_t[cu] = self.time;
                    }
                    _ => {}
                }
            }
        }
    }

    fn explosion_effects(&self, e: &Explosion) -> ExplosionFx {
        let mut fx = ExplosionFx::default();
        let center = v3(e.center);
        let r = e.radius.max(1e-3);
        let inner = e.inner_radius.clamp(0.0, r * 0.999);
        let crack = r * self.cfg.crack_radius_factor;
        let mut qbuf = Vec::new();
        self.phys.query_aabb(center - Vec3::splat(crack), center + Vec3::splat(crack), &mut qbuf);
        // candidates: (building, chunk, pose, distance)
        let mut cand: Vec<(u32, u32, Pose, f32)> = vec![];
        for &(_, tag) in &qbuf {
            match tag_kind(tag) {
                KIND_CHUNK => {
                    let (b, c) = tag_chunk(tag);
                    let pose = self.chunk_world_pose(b as usize, c as usize);
                    let d = self.buildings[b as usize].shapes[c as usize].distance_to_point(&pose, center, true);
                    if d <= crack {
                        cand.push((b, c, pose, d));
                    }
                }
                KIND_CLUSTER => {
                    let Some(cl) = self.clusters.get(tag_cluster(tag)) else { continue };
                    let pose = self.phys.body_state(cl.body).pose;
                    let bd = &self.buildings[cl.building as usize];
                    for &c in &cl.chunks {
                        let d = bd.shapes[c as usize].distance_to_point(&pose, center, true);
                        if d <= crack {
                            cand.push((cl.building, c, pose, d));
                        }
                    }
                }
                _ => {}
            }
        }
        cand.sort_by(|a, b| a.3.total_cmp(&b.3));
        let falloff = |d: f32, outer: f32| {
            if d <= inner {
                1.0
            } else {
                (1.0 - (d - inner) / (outer - inner).max(1e-3)).max(0.0)
            }
        };
        // chunk damage with occlusion
        let mut weights: Vec<(u32, u32, Pose, f32)> = vec![];
        let mut wsum = 0.0;
        let mut hits = Vec::new();
        for (i, &(b, c, pose, d)) in cand.iter().enumerate() {
            if d > r {
                continue;
            }
            let mut f = falloff(d, r);
            if f <= 0.0 {
                continue;
            }
            if i < self.cfg.occlusion_max_chunks && d > 1e-3 {
                let target = self.buildings[b as usize].shapes[c as usize].project_point(&pose, center, true).point;
                let dv = target - center;
                let dist = dv.length();
                if dist > 1e-3 {
                    hits.clear();
                    // the target itself may be among the hits: collect one extra
                    self.phys.ray_all(center, dv / dist, dist - 1e-2, self.cfg.occlusion_max_hits + 1, &mut hits);
                    let me = chunk_tag(b, c);
                    let n_occ = hits
                        .iter()
                        .filter(|h| h.tag != me && tag_kind(h.tag) == KIND_CHUNK && h.toi < dist - 1e-2)
                        .count()
                        .min(self.cfg.occlusion_max_hits);
                    f *= self.cfg.occlusion_factor.powi(n_occ as i32);
                }
            }
            let mat = self.buildings[b as usize].bld.chunks[c as usize].material;
            fx.damage.push((b, c, e.damage * f * self.cfg.damage_mult(mat)));
            weights.push((b, c, pose, f));
            wsum += f;
        }
        // impulse: distributed over affected chunks, radial
        if e.impulse > 0.0 && wsum > 0.0 {
            for &(b, c, pose, f) in &weights {
                let com = pose.transform_point(v3(self.buildings[b as usize].bld.chunks[c as usize].com));
                let dir = (com - center).try_normalize().unwrap_or(Vec3::Z);
                fx.impulses.push((b, c, dir * (e.impulse * f / wsum), com));
            }
        }
        // edge damage out to crack radius
        let mut edges: Vec<(u32, u32)> = vec![];
        for &(b, c, _, _) in &cand {
            let bd = &self.buildings[b as usize];
            for (_, ei) in bd.neighbors(c) {
                if bd.edge_alive[ei as usize] {
                    edges.push((b, ei));
                }
            }
        }
        edges.sort_unstable();
        edges.dedup();
        for (b, ei) in edges {
            let bd = &self.buildings[b as usize];
            let ed = &bd.bld.edges[ei as usize];
            if bd.indestructible(ed.a as usize) && bd.indestructible(ed.b as usize) {
                continue;
            }
            let pose = self.chunk_world_pose(b as usize, ed.a as usize);
            let d = (pose.transform_point(v3(ed.centroid)) - center).length();
            let f = falloff(d, crack);
            if f > 0.0 {
                fx.edge_damage.push((b, ei, e.damage * f * self.cfg.explosion_edge_damage_scale));
            }
        }
        fx
    }

    // ------------------------------------------------------------------ 2-3. damage

    fn apply_damage(&mut self) {
        let dmg = std::mem::take(&mut self.damage);
        let mut removals: Vec<(u32, u32)> = vec![];
        let weakens = self.cfg.stress.damage_weakens;
        for (b, c, amt) in dmg {
            let Some(bd) = self.buildings.get_mut(b as usize) else { continue };
            let cu = c as usize;
            if cu >= bd.n_chunks() || !bd.alive(cu) || bd.indestructible(cu) || amt <= 0.0 {
                continue;
            }
            if bd.hp[cu] <= 0.0 {
                continue; // already queued
            }
            if bd.glass(cu) {
                bd.hp[cu] = 0.0;
            } else {
                bd.hp[cu] -= amt;
            }
            if bd.hp[cu] <= 0.0 {
                removals.push((b, c));
            } else if weakens {
                bd.stress_active = true; // weakened joints: re-check utilization
            }
        }
        let ed = std::mem::take(&mut self.edge_damage);
        for (b, e, amt) in ed {
            let bd = &mut self.buildings[b as usize];
            if !bd.edge_alive[e as usize] {
                continue;
            }
            bd.edge_health[e as usize] -= amt;
            if bd.edge_health[e as usize] <= 0.0 {
                self.break_edge(b, e, BreakCause::Damage);
            }
        }
        for (b, c) in removals {
            self.remove_chunk(b, c);
        }
    }

    fn break_edge(&mut self, b: u32, e: u32, cause: BreakCause) {
        let bd = &mut self.buildings[b as usize];
        if !bd.edge_alive[e as usize] {
            return;
        }
        bd.edge_alive[e as usize] = false;
        bd.topo_changed = true;
        let (ea, eb) = (bd.bld.edges[e as usize].a, bd.bld.edges[e as usize].b);
        if cause != BreakCause::Tipping && bd.state[ea as usize] == ChunkState::Static && bd.state[eb as usize] == ChunkState::Static {
            bd.damage_z.push(bd.bld.edges[e as usize].centroid[2]);
        }
        bd.sliver_check.extend_from_slice(&[ea, eb]);
        for x in [ea, eb] {
            match bd.state[x as usize] {
                ChunkState::Static => bd.mark_dirty(x),
                ChunkState::InCluster(k) => {
                    if let Some(cl) = self.clusters.get_mut(k) {
                        cl.dirty = true;
                    }
                }
                _ => {}
            }
        }
        if let ChunkState::InCluster(k) = bd.state[ea as usize] {
            if bd.state[eb as usize] == ChunkState::InCluster(k) {
                if let Some(cl) = self.clusters.get_mut(k) {
                    cl.broken.push(e);
                }
            }
        }
        if self.cfg.emit_edge_events {
            self.events.push(Event::EdgeBroken { building: BuildingId(b), edge: e, cause });
        }
    }

    /// Chunk HP reached zero: remove it from the static set / its cluster, then either
    /// spawn small dynamic debris or shatter it.
    fn remove_chunk(&mut self, b: u32, c: u32) {
        let cu = c as usize;
        let min_vol = self.debris_min_volume();
        let (keep, max_clusters) = (self.cfg.keep_debris, self.max_clusters());
        let st = self.buildings[b as usize].state[cu];
        if st == ChunkState::Gone {
            return;
        }
        let pose = self.chunk_world_pose(b as usize, cu);
        let mut vel = (Vec3::ZERO, Vec3::ZERO);
        let com_local = v3(self.buildings[b as usize].bld.chunks[cu].com);
        let com = pose.transform_point(com_local);
        match st {
            ChunkState::InCluster(k) => {
                if let Some(cl) = self.clusters.get_mut(k) {
                    cl.dirty = true;
                    let s = self.phys.body_state(cl.body);
                    vel = (s.linvel + s.angvel.cross(com - s.world_com), s.angvel);
                }
            }
            _ => {
                self.drop_static_collider(b, c);
            }
        }
        let bd = &mut self.buildings[b as usize];
        if st == ChunkState::Static {
            bd.damage_z.push(bd.bld.chunks[cu].com[2]);
        }
        bd.topo_changed = true;
        bd.state[cu] = ChunkState::Gone;
        bd.chunk_pose[cu] = pose;
        bd.stress_active = true;
        let (o0, o1) = (bd.csr_off[cu] as usize, bd.csr_off[cu + 1] as usize);
        for k in o0..o1 {
            let (nb, e) = (bd.csr_nbr[k], bd.csr_edge[k] as usize);
            if bd.edge_alive[e] {
                bd.edge_alive[e] = false;
                if bd.state[nb as usize] == ChunkState::Static {
                    bd.dirty.push(nb);
                    bd.sliver_check.push(nb);
                }
            }
        }
        self.events.push(Event::ChunkDestroyed { building: BuildingId(b), chunk: c, pos: a3(com) });
        let ch = bd.bld.chunks[cu];
        // debris that is destroyed again shatters (no debris-of-debris)
        let was_debris = matches!(st, ChunkState::InCluster(k) if self.clusters.get(k).map_or(false, |c| c.debris));
        let can_debris = !was_debris
            && !bd.glass(cu)
            && ch.flags & F_NO_DEBRIS == 0
            && ch.volume >= min_vol
            && (keep || ch.volume <= self.cfg.debris_max_volume)
            && self.clusters.len() < max_clusters
            && self.clusters.values().map(|c| c.chunks.len()).sum::<usize>() < self.cfg.max_chunks_in_flight;
        if can_debris {
            if !matches!(st, ChunkState::InCluster(_)) {
                // pick up a recent explosion impulse
                if self.time - bd.pending_impulse_t[cu] <= self.cfg.impulse_memory {
                    vel.0 += v3(bd.pending_impulse[cu]) / ch.mass.max(1e-3);
                    bd.pending_impulse_t[cu] = f32::NEG_INFINITY;
                }
            }
            bd.hp[cu] = 1.0; // one more hit shatters the debris
            let k = self.create_cluster(b, vec![c], pose, vel.0, vel.1, true);
            self.emit_detached(k);
        } else {
            self.events.push(Event::ChunkShattered { building: BuildingId(b), chunk: c, pos: a3(com), material: ch.material });
        }
    }

    // ------------------------------------------------------------------ 4. connectivity

    fn connectivity(&mut self) {
        let detached: Vec<(u32, Vec<Vec<u32>>)> = self
            .buildings
            .par_iter_mut()
            .enumerate()
            .filter(|(_, b)| !b.dirty.is_empty())
            .map(|(i, b)| (i as u32, b.find_detached()))
            .collect();
        for (b, comps) in detached {
            for comp in comps {
                self.begin_collapse(b, comp);
            }
        }
    }

    fn begin_collapse(&mut self, b: u32, comp: Vec<u32>) {
        let bd = &mut self.buildings[b as usize];
        let mut mass = 0.0;
        for &c in &comp {
            bd.state[c as usize] = ChunkState::Detaching;
            mass += bd.bld.chunks[c as usize].mass;
        }
        bd.stress_active = true;
        let cfg = &self.cfg;
        let t = ((mass / 100.0).max(1.0).ln() / (cfg.collapse_delay_ref_mass / 100.0).max(1.0001).ln()).clamp(0.0, 1.0);
        let delay = cfg.collapse_delay_min + (cfg.collapse_delay_max - cfg.collapse_delay_min) * t;
        self.events.push(Event::CollapseWarning { building: BuildingId(b), chunks: comp.clone(), delay });
        self.pending.push(PendingCollapse { building: b, chunks: comp, timer: delay });
    }

    // ------------------------------------------------------------------ 5. stress

    /// Returns true if any edge broke.
    fn stress_config(&self, max_iters: usize) -> StressConfig {
        let s = &self.cfg.stress;
        StressConfig {
            max_iters,
            tol: s.tol,
            hold_time: s.hold_time,
            max_breaks_per_tick: s.max_breaks_per_tick,
            bending: s.bending,
            bend_scale: s.bend_scale,
            compression_factor: s.compression_factor,
            tension_factor: s.tension_factor,
            bearing_sections: s.bearing_sections,
            buckling: s.buckling,
            rankine_k: s.rankine_k,
            catchup_frac: s.catchup_frac,
            catchup_max_ms: s.catchup_max_ms,
            region_max_nodes: s.region_max_nodes,
            ..StressConfig::default()
        }
    }

    /// One stress solve on a building. Edges are only reported for breaking when the solve has
    /// converged, or when it has stayed unconverged for `max_unconverged_ticks` (so a damage-induced
    /// overload still breaks within a bounded time even if the solver keeps chasing residual).
    fn solve_building(b: &mut Building, scfg: &StressConfig, dt: f32, max_unconverged: u32, damage_weakens: bool) -> Vec<u32> {
        for ((na, st), &s) in b.stress_node_alive.iter_mut().zip(&b.state).zip(&b.structural) {
            *na = *st == ChunkState::Static && s;
        }
        let na = &b.stress_node_alive;
        for ((ea, &alive), &[a, bb]) in b.stress_edge_alive.iter_mut().zip(&b.edge_alive).zip(&b.edge_ab) {
            *ea = alive && na[a as usize] && na[bb as usize];
        }
        if damage_weakens {
            for ((s, &hp), &max) in b.stress_strength.iter_mut().zip(&b.hp).zip(&b.hp_max) {
                *s = if max > 0.0 { (hp / max).clamp(0.0, 1.0) } else { 1.0 };
            }
        }
        let input = StressInput {
            node_weight: &b.weight,
            node_alive: &b.stress_node_alive,
            anchor: &b.anchor,
            edge_alive: &b.stress_edge_alive,
            node_strength: if damage_weakens { &b.stress_strength } else { &[] },
        };
        let r = solve_step_lean(&b.stress_graph, &mut b.stress_state, &input, scfg, dt);
        let any_over = b.stress_state.any_overloaded();
        if r.converged {
            b.stress_unconverged = 0;
        } else {
            b.stress_unconverged += 1;
        }
        if r.converged && r.to_break.is_empty() && !any_over {
            b.stress_active = false;
        }
        b.utilization.clear();
        b.utilization.extend_from_slice(b.stress_state.utilization());
        if r.converged || b.stress_unconverged >= max_unconverged {
            r.to_break
        } else {
            Vec::new()
        }
    }

    /// Solve the intact building to convergence (no hold-time accumulation, no breaks).
    fn settle_stress_at_load(&mut self, bi: usize) {
        if !self.cfg.stress.enabled {
            return;
        }
        let scfg = self.stress_config(self.cfg.stress.load_iters.max(1));
        let settle = self.cfg.stress.settle_overloads_at_load;
        let damage_weakens = self.cfg.stress.damage_weakens;
        let b = &mut self.buildings[bi];
        for _round in 0..self.cfg.stress.load_max_rounds.max(1) {
            for _ in 0..self.cfg.stress.load_max_calls {
                Self::solve_building(b, &scfg, 0.0, u32::MAX, damage_weakens);
                if b.stress_unconverged == 0 {
                    break;
                }
            }
            if !settle {
                break;
            }
            // Joints already overloaded in the *intact* building are treated as pre-cracked:
            // break them silently (worst first, in batches) and re-solve, so the building starts
            // in equilibrium and an untouched building never sheds pieces in play.
            let mut over: Vec<(f32, usize)> =
                b.utilization.iter().enumerate().filter(|(e, &u)| u > 1.0 && b.edge_alive[*e]).map(|(e, &u)| (u, e)).collect();
            if over.is_empty() {
                break;
            }
            over.sort_by(|x, y| y.0.total_cmp(&x.0));
            let n = (over.len() / 8).max(4).min(over.len());
            for &(_, e) in &over[..n] {
                b.edge_alive[e] = false;
                b.load_cracked_edges += 1;
                let ed = b.bld.edges[e];
                b.dirty.push(ed.a);
                b.dirty.push(ed.b);
            }
            // anything that lost its support becomes anchored static rubble, as for floaters
            for comp in b.find_detached() {
                for c in comp {
                    b.state[c as usize] = ChunkState::Frozen;
                    b.load_floating += 1;
                }
            }
        }
        b.dirty.clear();
        b.stress_unconverged = 0;
        b.stress_active = b.utilization.iter().any(|&u| u > 1.0);
    }

    /// Returns true if any edge broke.
    fn step_stress(&mut self, dt: f32) -> bool {
        if !self.cfg.stress.enabled {
            return false;
        }
        let s = self.cfg.stress.clone();
        let scfg = self.stress_config(self.stress_iters);
        let min_mass = s.min_component_mass;
        let max_unconv = s.max_unconverged_ticks;
        let damage = s.damage_weakens;
        let t0 = Instant::now();
        let results: Vec<(u32, Vec<u32>)> = self
            .buildings
            .par_iter_mut()
            .enumerate()
            .filter(|(_, b)| b.stress_active)
            .filter_map(|(i, b)| {
                if min_mass > 0.0 && b.total_static_structural_mass() < min_mass {
                    b.stress_active = false;
                    return None;
                }
                Some((i as u32, Self::solve_building(b, &scfg, dt, max_unconv, damage)))
            })
            .collect();
        let ms = t0.elapsed().as_secs_f32() * 1e3;
        if !results.is_empty() {
            if ms > s.budget_ms {
                self.stress_iters = ((self.stress_iters as f32 * 0.7) as usize).max(4);
            } else if ms < s.budget_ms * 0.5 {
                self.stress_iters = (self.stress_iters + self.stress_iters / 4 + 1).min(s.max_iters);
            }
        }
        let mut broke = false;
        for (b, edges) in results {
            for e in edges {
                if self.buildings[b as usize].edge_alive[e as usize] {
                    self.break_edge(b, e, BreakCause::Stress);
                    broke = true;
                }
            }
        }
        broke
    }

    /// Backstop for the incremental connectivity: a full search of the load-carrying graph of
    /// every building whose joints changed (throttled), collapsing whatever reaches no anchor.
    fn step_ground_check(&mut self, dt: f32) {
        if !self.cfg.ground_check {
            return;
        }
        let interval = self.cfg.ground_check_interval;
        let found: Vec<(u32, Vec<Vec<u32>>)> = self
            .buildings
            .par_iter_mut()
            .enumerate()
            .filter_map(|(i, b)| {
                b.ground_timer += dt;
                if !b.topo_changed || b.ground_timer < interval {
                    return None;
                }
                b.topo_changed = false;
                b.ground_timer = 0.0;
                let comps = b.ungrounded_components();
                (!comps.is_empty()).then_some((i as u32, comps))
            })
            .collect();
        for (b, comps) in found {
            for comp in comps {
                self.ground_check_catches += comp.len();
                if std::env::var_os("RUBBLE_DEBUG_GROUND").is_some() {
                    eprintln!("ground check: building {b}: {} chunks reach no anchor, collapsing", comp.len());
                }
                self.begin_collapse(b, comp);
            }
        }
    }

    /// Cut loose static chunks next to recent damage that hang by slivers. Their neighbours are
    /// re-checked on the next tick (breaking marks them), so a dangling chain comes off piece
    /// by piece. Returns true if any edge broke.
    fn step_slivers(&mut self) -> bool {
        let (max_area, seat) = (self.cfg.sliver_area, self.cfg.sliver_seat_area);
        let mut broke = false;
        for b in 0..self.buildings.len() {
            if self.buildings[b].sliver_check.is_empty() {
                continue;
            }
            let mut cand = std::mem::take(&mut self.buildings[b].sliver_check);
            if max_area <= 0.0 {
                continue;
            }
            cand.sort_unstable();
            cand.dedup();
            let mut cut = vec![];
            {
                let bd = &self.buildings[b];
                for &c in &cand {
                    if bd.hangs_by_slivers(c as usize, max_area, seat) {
                        for k in bd.csr_off[c as usize] as usize..bd.csr_off[c as usize + 1] as usize {
                            let e = bd.csr_edge[k];
                            if bd.edge_alive[e as usize] {
                                cut.push(e);
                            }
                        }
                    }
                }
            }
            for e in cut {
                self.break_edge(b as u32, e, BreakCause::Tipping);
                broke = true;
            }
        }
        broke
    }

    /// Tipping check on the planes of recent damage (see `Building::tipping_edges`).
    /// Returns true if any edge broke.
    fn step_tipping(&mut self, dt: f32) -> bool {
        let cfg = &self.cfg;
        let (band, margin, maxp) = (cfg.tip_band.max(0.05), cfg.tip_margin, cfg.tip_max_planes.max(1));
        let (enabled, interval) = (cfg.tipping, cfg.tip_interval);
        let work: Vec<(u32, Vec<u32>)> = self
            .buildings
            .par_iter_mut()
            .enumerate()
            .filter_map(|(i, b)| {
                if !enabled {
                    b.damage_z.clear();
                    return None;
                }
                if b.damage_z.is_empty() {
                    b.tip_timer = 0.0;
                    return None;
                }
                b.tip_timer += dt;
                if b.tip_timer < interval {
                    return None;
                }
                b.tip_timer = 0.0;
                let mut planes: Vec<i64> = vec![];
                for &z in b.damage_z.iter().rev() {
                    let k = (z / band).floor() as i64;
                    if !planes.contains(&k) {
                        planes.push(k);
                        if planes.len() >= maxp {
                            break;
                        }
                    }
                }
                b.damage_z.clear();
                let mut edges: Vec<u32> = planes.iter().flat_map(|&k| b.tipping_edges((k as f32 + 0.5) * band, margin)).collect();
                edges.sort_unstable();
                edges.dedup();
                (!edges.is_empty()).then_some((i as u32, edges))
            })
            .collect();
        let mut broke = false;
        for (b, edges) in work {
            for e in edges {
                if self.buildings[b as usize].edge_alive[e as usize] {
                    self.break_edge(b, e, BreakCause::Tipping);
                    broke = true;
                }
            }
        }
        broke
    }

    // ------------------------------------------------------------------ 6. promotion

    fn promote(&mut self, dt: f32) {
        for p in &mut self.pending {
            p.timer -= dt;
        }
        let (ready, keep): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending).into_iter().partition(|p| p.timer <= 0.0);
        self.pending = keep;
        for p in ready {
            let b = p.building;
            let comps = {
                let bd = &mut self.buildings[b as usize];
                bd.epoch = bd.epoch.wrapping_add(1).max(1);
                let ep = bd.epoch;
                let members: Vec<u32> =
                    p.chunks.iter().copied().filter(|&c| bd.state[c as usize] == ChunkState::Detaching).collect();
                for &c in &members {
                    bd.comp_ep[c as usize] = ep;
                }
                // split into connected components among members
                let mut comps: Vec<Vec<u32>> = vec![];
                for &s in &members {
                    if bd.visit[s as usize] == ep {
                        continue;
                    }
                    bd.visit[s as usize] = ep;
                    let mut comp = vec![s];
                    let mut head = 0;
                    while head < comp.len() {
                        let n = comp[head];
                        head += 1;
                        let (o0, o1) = (bd.csr_off[n as usize] as usize, bd.csr_off[n as usize + 1] as usize);
                        for k in o0..o1 {
                            let nb = bd.csr_nbr[k] as usize;
                            if bd.edge_alive[bd.csr_edge[k] as usize] && bd.comp_ep[nb] == ep && bd.visit[nb] != ep {
                                bd.visit[nb] = ep;
                                comp.push(nb as u32);
                            }
                        }
                    }
                    comps.push(comp);
                }
                comps
            };
            for comp in comps {
                self.promote_component(b, comp);
            }
        }
    }

    fn promote_component(&mut self, b: u32, comp: Vec<u32>) {
        // glass shatters instead of falling
        let mut keep = Vec::with_capacity(comp.len());
        for c in comp {
            if self.buildings[b as usize].glass(c as usize) {
                self.shatter_static(b, c);
            } else {
                keep.push(c);
            }
        }
        if keep.is_empty() {
            return;
        }
        let bd = &self.buildings[b as usize];
        let mass: f32 = keep.iter().map(|&c| bd.bld.chunks[c as usize].mass).sum();
        let vol: f32 = keep.iter().map(|&c| bd.bld.chunks[c as usize].volume).sum();
        let small = mass < self.cfg.freeze_min_mass;
        if vol < self.debris_min_volume() || (small && self.clusters.len() >= self.max_clusters()) {
            for c in keep {
                self.shatter_static(b, c);
            }
            return;
        }
        // initial velocity from recent explosion impulses
        let pose = bd.pose;
        let mp = self.mass_props(b, &keep);
        let com_w = pose.transform_point(mp.local_com);
        let (mut jl, mut ja) = (Vec3::ZERO, Vec3::ZERO);
        for &c in &keep {
            let cu = c as usize;
            if self.time - bd.pending_impulse_t[cu] <= self.cfg.impulse_memory {
                let j = v3(bd.pending_impulse[cu]);
                let p = pose.transform_point(v3(bd.bld.chunks[cu].com));
                jl += j;
                ja += (p - com_w).cross(j);
            }
        }
        let linvel = jl / mp.mass.max(1e-3);
        let r = Mat3::from_quat(pose.rotation);
        let iw = r * mp.inertia * r.transpose();
        let angvel = if iw.determinant().abs() > 1e-9 { iw.inverse() * ja } else { Vec3::ZERO };
        let k = self.create_cluster(b, keep, pose, linvel, angvel, false);
        self.emit_detached(k);
    }

    fn shatter_static(&mut self, b: u32, c: u32) {
        let com = self.chunk_world_com(b as usize, c as usize);
        let cu = c as usize;
        self.drop_static_collider(b, c);
        let bd = &mut self.buildings[b as usize];
        bd.chunk_pose[cu] = bd.pose;
        bd.state[cu] = ChunkState::Gone;
        bd.hp[cu] = 0.0;
        bd.topo_changed = true;
        for k in bd.csr_off[cu] as usize..bd.csr_off[cu + 1] as usize {
            let (nb, e) = (bd.csr_nbr[k], bd.csr_edge[k] as usize);
            if bd.edge_alive[e] {
                bd.edge_alive[e] = false;
                if bd.state[nb as usize] == ChunkState::Static {
                    bd.dirty.push(nb);
                    bd.sliver_check.push(nb);
                }
            }
        }
        let material = bd.bld.chunks[cu].material;
        self.events.push(Event::ChunkShattered { building: BuildingId(b), chunk: c, pos: a3(com), material });
    }

    fn mass_props(&self, b: u32, chunks: &[u32]) -> MassProps {
        let bd = &self.buildings[b as usize];
        let mut m = 0.0;
        let mut mc = Vec3::ZERO;
        for &c in chunks {
            let ch = &bd.bld.chunks[c as usize];
            m += ch.mass;
            mc += v3(ch.com) * ch.mass;
        }
        let com = mc / m.max(1e-6);
        let mut i = Mat3::ZERO;
        for &c in chunks {
            let ch = &bd.bld.chunks[c as usize];
            i += inertia_mat(&ch.inertia) + parallel_axis(ch.mass, v3(ch.com) - com);
        }
        MassProps { mass: m, local_com: com, inertia: i }
    }

    /// Create a dynamic cluster from chunks (all in the same body frame `pose`).
    fn create_cluster(&mut self, b: u32, chunks: Vec<u32>, pose: Pose, linvel: Vec3, angvel: Vec3, debris: bool) -> ClusterKey {
        let mp = self.mass_props(b, &chunks);
        let key = self.clusters.insert(Cluster {
            building: b,
            chunks: vec![],
            edges: vec![],
            edge_impulse: vec![],
            body: BodyId(Default::default()),
            collider: ColliderId(Default::default()),
            mass: mp.mass,
            local_com: mp.local_com,
            age: 0.0,
            rest_time: 0.0,
            debris,
            dirty: false,
            last_impact: f32::NEG_INFINITY,
            impact_dv: Vec3::ZERO,
            broken: vec![],
            connected: false,
        });
        for &c in &chunks {
            let cu = c as usize;
            self.drop_static_collider(b, c);
            self.buildings[b as usize].state[cu] = ChunkState::InCluster(key);
        }
        let bd = &mut self.buildings[b as usize];
        let mut edges = vec![];
        for &c in &chunks {
            let cu = c as usize;
            for k in bd.csr_off[cu] as usize..bd.csr_off[cu + 1] as usize {
                let (nb, e) = (bd.csr_nbr[k], bd.csr_edge[k]);
                if !bd.edge_alive[e as usize] {
                    continue;
                }
                if bd.state[nb as usize] == ChunkState::InCluster(key) {
                    if c < nb {
                        bd.edge_slot[e as usize] = edges.len() as u32;
                        edges.push(e);
                    }
                } else {
                    // cut ties with whatever stays behind (and re-check what it held up)
                    bd.edge_alive[e as usize] = false;
                    bd.topo_changed = true;
                    if bd.state[nb as usize] == ChunkState::Static {
                        bd.dirty.push(nb);
                        bd.sliver_check.push(nb);
                    }
                }
            }
        }
        bd.stress_active = true;
        let parts = bd.compound_parts(&chunks);
        let (body, col) = self.phys.add_dynamic_compound(pose, parts, &mp, linvel, angvel, cluster_tag(key));
        if self.cfg.keep_debris && (debris || mp.mass < self.cfg.freeze_min_mass) {
            self.phys.set_light_debris(col, true);
        }
        let cl = &mut self.clusters[key];
        cl.edge_impulse = vec![0.0; edges.len()];
        cl.edges = edges;
        cl.chunks = chunks;
        cl.body = body;
        cl.collider = col;
        key
    }

    fn emit_detached(&mut self, k: ClusterKey) {
        let cl = &self.clusters[k];
        let s = self.phys.body_state(cl.body);
        self.events.push(Event::ClusterDetached {
            cluster: cluster_id(k),
            building: BuildingId(cl.building),
            chunks: cl.chunks.clone(),
            transform: pose_to_rowmajor(&s.pose),
            lin_vel: a3(s.linvel),
            ang_vel: a3(s.angvel),
        });
    }

    /// Recompute membership / components of clusters that lost chunks or edges.
    fn rebuild_dirty_clusters(&mut self) {
        let dirty: Vec<ClusterKey> = self.clusters.iter().filter(|(_, c)| c.dirty).map(|(k, _)| k).collect();
        for k in dirty {
            self.rebuild_cluster(k);
        }
    }

    fn rebuild_cluster(&mut self, k: ClusterKey) {
        let Some(cl) = self.clusters.get_mut(k) else { return };
        cl.dirty = false;
        let broken = std::mem::take(&mut cl.broken);
        let b = cl.building;
        let bd = &mut self.buildings[b as usize];
        let mut members = std::mem::take(&mut self.scratch_u32);
        members.clear();
        members.extend(cl.chunks.iter().copied().filter(|&c| bd.state[c as usize] == ChunkState::InCluster(k)));
        // A cluster is connected after every rebuild. If it lost no chunk and every edge broken
        // since then still has its ends connected, it is still one piece: the full walk below
        // would find one component equal to `chunks` and change nothing.
        if cl.connected && !broken.is_empty() && members.len() == cl.chunks.len() && bd.still_connected(k, &broken) {
            bd.epoch = bd.epoch.wrapping_add(1).max(1); // keep the epoch sequence of the full walk
            self.scratch_u32 = members;
            return;
        }
        if members.is_empty() {
            self.scratch_u32 = members;
            let body = cl.body;
            self.clusters.remove(k);
            self.drop_body(body);
            self.events.push(Event::ClusterDespawned { cluster: cluster_id(k) });
            return;
        }
        // components over alive internal edges
        bd.epoch = bd.epoch.wrapping_add(1).max(1);
        let ep = bd.epoch;
        let mut comps: Vec<Vec<u32>> = vec![];
        for &s in &members {
            if bd.visit[s as usize] == ep {
                continue;
            }
            bd.visit[s as usize] = ep;
            // the first component is usually (nearly) the whole cluster
            let mut comp = Vec::with_capacity(if comps.is_empty() { members.len() } else { 0 });
            comp.push(s);
            let mut head = 0;
            while head < comp.len() {
                let n = comp[head] as usize;
                head += 1;
                for kk in bd.csr_off[n] as usize..bd.csr_off[n + 1] as usize {
                    let nb = bd.csr_nbr[kk] as usize;
                    if bd.edge_alive[bd.csr_edge[kk] as usize]
                        && bd.visit[nb] != ep
                        && bd.state[nb] == ChunkState::InCluster(k)
                    {
                        bd.visit[nb] = ep;
                        comp.push(nb as u32);
                    }
                }
            }
            if comp.capacity() > 2 * comp.len() + 64 {
                comp.shrink_to_fit();
            }
            comps.push(comp);
        }
        let n_members = members.len();
        self.scratch_u32 = members;
        cl.connected = true; // it keeps one component below
        let changed = comps.len() > 1 || comps[0].len() != cl.chunks.len();
        if !changed {
            return;
        }
        let mass_of = |comp: &Vec<u32>| comp.iter().map(|&c| bd.bld.chunks[c as usize].mass).sum::<f32>();
        if comps.len() > 1 {
            comps.sort_by(|a, b| mass_of(b).total_cmp(&mass_of(a)));
        }
        debug_assert!(comps.iter().map(|c| c.len()).sum::<usize>() == n_members);
        let parent_state = self.phys.body_state(cl.body);
        // parent keeps the heaviest component
        let main = comps.remove(0);
        let mp = self.mass_props(b, &main);
        let bd = &self.buildings[b as usize];
        let cl = &mut self.clusters[k];
        cl.chunks = main;
        let parts = bd.compound_parts(&cl.chunks);
        cl.collider = self.phys.set_dynamic_compound(cl.body, cl.collider, parts, &mp);
        cl.mass = mp.mass;
        cl.local_com = mp.local_com;
        let debris = cl.debris;
        let mut children = vec![];
        for comp in comps {
            let mpc = self.mass_props(b, &comp);
            let com_w = parent_state.pose.transform_point(mpc.local_com);
            let v = parent_state.linvel + parent_state.angvel.cross(com_w - parent_state.world_com);
            let ck = self.create_cluster(b, comp, parent_state.pose, v, parent_state.angvel, debris);
            self.clusters[ck].connected = true; // a component
            children.push(ck);
        }
        // parent's internal edges: still alive with both ends in the parent (children took theirs)
        let bd = &mut self.buildings[b as usize];
        let cl = &mut self.clusters[k];
        let mut edges = std::mem::take(&mut cl.edges);
        edges.retain(|&e| {
            let [ea, eb] = bd.edge_ab[e as usize];
            bd.edge_alive[e as usize]
                && bd.state[ea as usize] == ChunkState::InCluster(k)
                && bd.state[eb as usize] == ChunkState::InCluster(k)
        });
        for (i, &e) in edges.iter().enumerate() {
            bd.edge_slot[e as usize] = i as u32;
        }
        cl.edge_impulse.clear();
        cl.edge_impulse.resize(edges.len(), 0.0);
        cl.edges = edges;
        if !children.is_empty() {
            self.events.push(Event::ClusterSplit { parent: cluster_id(k), children: children.iter().map(|&c| cluster_id(c)).collect() });
            for c in children {
                self.emit_detached(c);
            }
        }
    }

    // ------------------------------------------------------------------ 8. impacts

    fn impacts(&mut self) {
        let decay = (-self.last_dt / self.cfg.impact_window.max(1e-3)).exp();
        for (_, cl) in self.clusters.iter_mut() {
            cl.impact_dv *= decay;
        }
        self.contacts.clear();
        let mut contacts = std::mem::take(&mut self.contacts);
        self.phys.contacts(1.0, &mut contacts);
        // (cluster, chunk, impulse vector on the cluster)
        let mut hits = std::mem::take(&mut self.scratch_hits);
        hits.clear();
        let mut crush: Vec<(u32, u32)> = vec![];
        for ct in &contacts {
            for (tag, sub, sign) in [(ct.tag1, ct.sub1, -1.0f32), (ct.tag2, ct.sub2, 1.0)] {
                match tag_kind(tag) {
                    KIND_CLUSTER => {
                        let k = tag_cluster(tag);
                        let Some(cl) = self.clusters.get(k) else { continue };
                        let chunk = match sub {
                            Some(s) if (s as usize) < cl.chunks.len() && cl.chunks.len() > 1 => cl.chunks[s as usize],
                            _ => match self.resolve_hit(tag, ct.point) {
                                Some((_, c)) => c,
                                None => continue,
                            },
                        };
                        // small debris crushed by a hard hit (e.g. under a falling section)
                        if cl.chunks.len() <= self.cfg.crush_max_chunks
                            && ct.impulse / cl.mass.max(1e-3) > self.cfg.crush_dv
                        {
                            crush.push((cl.building, chunk));
                        }
                        hits.push((k, chunk, ct.normal * (ct.impulse * sign)));
                    }
                    KIND_CHUNK => {
                        if ct.impulse > self.cfg.impact_min_impulse {
                            let (b, c) = tag_chunk(tag);
                            self.damage.push((b, c, (ct.impulse - self.cfg.impact_min_impulse) * self.cfg.impact_damage_scale));
                        }
                    }
                    _ => {}
                }
            }
        }
        // Bond load model: a rigid cluster responds to the total impulse with dv = J/M; the part of
        // a chunk's contact impulse not explained by its own share (J_c - m_c dv) must be carried
        // by its bonds.
        // same order as comparing (key, chunk); KeyData orders by (idx, version)
        hits.sort_unstable_by_key(|h| {
            let f = h.0.data().as_ffi();
            ((f & 0xffff_ffff) as u128) << 64 | ((f >> 32) as u128) << 32 | h.1 as u128
        });
        let mut touched: Vec<ClusterKey> = vec![];
        let mut impact_jobs: Vec<(ClusterKey, Vec<u32>, Vec3)> = vec![];
        // per touched cluster (same order as `touched`): the edge slots that received load this
        // tick. Only those can cross the break threshold now: every alive edge is checked in the
        // tick its load rises, and accumulated loads only decay afterwards.
        let mut loaded: Vec<(usize, usize)> = vec![];
        let mut loaded_slots: Vec<u32> = vec![];
        let mut xs: Vec<(u32, Vec3)> = vec![];
        let mut adds: Vec<(u32, f32)> = vec![];
        let mut i = 0;
        while i < hits.len() {
            let k = hits[i].0;
            let mut j = i;
            let mut jt = Vec3::ZERO;
            while j < hits.len() && hits[j].0 == k {
                jt += hits[j].2;
                j += 1;
            }
            let cl = &self.clusters[k];
            let bd = &self.buildings[cl.building as usize];
            let dv = jt / cl.mass.max(1e-3);
            let cl_chunks_len = cl.chunks.len();
            let cl_last_impact = cl.last_impact;
            // per contacted chunk: excess impulse X_c = J_c - m_c dv
            xs.clear();
            let mut p = i;
            while p < j {
                let c = hits[p].1;
                let mut jc = Vec3::ZERO;
                while p < j && hits[p].1 == c {
                    jc += hits[p].2;
                    p += 1;
                }
                xs.push((c, jc - dv * bd.bld.chunks[c as usize].mass));
            }
            // bond load = half the relative excess across the bond, minus its compressive part
            // (concrete does not fail in compression; a flat landing only compresses bonds).
            let x_of = |c: u32| match xs.binary_search_by_key(&c, |x| x.0) {
                Ok(i) => (xs[i].1, true),
                Err(_) => (-dv * bd.bld.chunks[c as usize].mass, false),
            };
            adds.clear();
            for &(c, xc) in &xs {
                for (nb, e) in bd.neighbors(c) {
                    if !bd.edge_alive[e as usize] || bd.state[nb as usize] != ChunkState::InCluster(k) {
                        continue;
                    }
                    let (xn, contacted) = x_of(nb);
                    if contacted && nb < c {
                        continue; // counted from the other side
                    }
                    let ed = &bd.bld.edges[e as usize];
                    let mut n = v3(ed.normal);
                    if ed.a != c {
                        n = -n; // normal from c to nb
                    }
                    let l = (xc - xn) * 0.5;
                    let comp = l.dot(n).max(0.0);
                    let load = (l - n * comp).length();
                    adds.push((bd.edge_slot[e as usize], load));
                }
            }
            let cl = &mut self.clusters[k];
            let l0 = loaded_slots.len();
            for &(s, v) in &adds {
                if let Some(acc) = cl.edge_impulse.get_mut(s as usize) {
                    *acc += v;
                    loaded_slots.push(s);
                }
            }
            loaded.push((l0, loaded_slots.len()));
            // landing deceleration accumulated over a short window, net of the steady
            // gravity-support impulse (a resting cluster accumulates ~0)
            let net = dv + v3(self.cfg.gravity) * self.last_dt;
            if net.dot(dv) > 0.0 {
                cl.impact_dv += net;
            }
            let acc = cl.impact_dv.length();
            if cl_chunks_len >= self.cfg.impact_stress_min_chunks
                && acc >= self.cfg.impact_stress_min_dv
                && self.time - cl_last_impact >= self.cfg.impact_stress_cooldown
            {
                let mut supports: Vec<u32> = xs.iter().map(|x| x.0).collect();
                supports.dedup();
                impact_jobs.push((k, supports, cl.impact_dv));
                cl.impact_dv = Vec3::ZERO;
            }
            touched.push(k);
            i = j;
        }
        self.contacts = contacts;
        self.scratch_hits = hits;
        crush.sort_unstable();
        crush.dedup();
        for (b, c) in crush {
            self.damage.push((b, c, f32::MAX));
        }
        // collect impact solves that are due (fixed latency => deterministic), then launch new ones
        let due: Vec<PendingImpact> = {
            let (d, keep): (Vec<_>, Vec<_>) =
                std::mem::take(&mut self.pending_impacts).into_iter().partition(|p| p.due_tick <= self.tick);
            self.pending_impacts = keep;
            d
        };
        for p in due {
            let tw = Instant::now();
            let r = p.rx.into_inner().unwrap().recv().expect("impact job");
            self.timings.impact_wait_ms += tw.elapsed().as_secs_f32() * 1e3;
            for e in r.broken {
                self.break_edge(r.building, e, BreakCause::Impact);
            }
            for c in r.crushed {
                self.damage.push((r.building, c, f32::MAX));
            }
        }
        if !impact_jobs.is_empty() {
            let t = self.time;
            for (k, sup, dv) in impact_jobs {
                self.clusters[k].last_impact = t;
                let job = self.impact_job(k, &sup, dv);
                let (tx, rx) = std::sync::mpsc::channel();
                if self.cfg.impact_latency_ticks == 0 {
                    let _ = tx.send(run_impact_job(job));
                } else {
                    impact_pool().spawn(move || {
                        let _ = tx.send(run_impact_job(job));
                    });
                }
                self.pending_impacts.push(PendingImpact { due_tick: self.tick + self.cfg.impact_latency_ticks as u64, rx: std::sync::Mutex::new(rx) });
            }
            if self.cfg.impact_latency_ticks == 0 {
                // apply immediately
                let due = std::mem::take(&mut self.pending_impacts);
                for p in due {
                    let r = p.rx.into_inner().unwrap().recv().expect("impact job");
                    for e in r.broken {
                        self.break_edge(r.building, e, BreakCause::Impact);
                    }
                    for c in r.crushed {
                        self.damage.push((r.building, c, f32::MAX));
                    }
                }
            }
        }
        // `touched` is sorted and unique (hits are grouped by cluster key)
        debug_assert!(touched.windows(2).all(|w| w[0] < w[1]));
        let factor = self.cfg.impact_factor;
        let decay = self.cfg.impact_decay;
        let mut to_break = vec![];
        for (ti, k) in touched.into_iter().enumerate() {
            let Some(cl) = self.clusters.get(k) else { continue };
            let b = cl.building;
            let (l0, l1) = loaded[ti];
            let slots = &mut loaded_slots[l0..l1];
            slots.sort_unstable();
            to_break.clear();
            let mut last = u32::MAX;
            for &s in slots.iter() {
                if s == last {
                    continue;
                }
                last = s;
                let e = cl.edges[s as usize];
                let strength = self.buildings[b as usize].bld.edges[e as usize].strength;
                if cl.edge_impulse[s as usize] > strength * factor {
                    to_break.push(e);
                }
            }
            for &e in &to_break {
                self.break_edge(b, e, BreakCause::Impact);
            }
        }
        for (_, cl) in self.clusters.iter_mut() {
            for a in cl.edge_impulse.iter_mut() {
                *a *= decay;
            }
        }
        let t0 = Instant::now();
        self.rebuild_dirty_clusters();
        if debug_impact() && t0.elapsed().as_secs_f32() > 1e-3 {
            eprintln!("rebuild clusters {:.2} ms ({} clusters)", t0.elapsed().as_secs_f32() * 1e3, self.clusters.len());
        }
    }

    /// Impact stress (input side): a hard landing decelerates the whole cluster by `dv` within
    /// roughly `impact_duration`; every chunk's inertial load `m·dv/impact_duration` must flow
    /// through the bonds into the chunks touching the ground. The flow is solved with the
    /// structural solver on a worker thread (see [`run_impact_job`]); overloaded bonds break
    /// (inter-panel joints at `impact_joint_factor` of their capacity), cascading over a few
    /// rounds. This makes a falling section break up / pancake instead of landing rigid.
    fn impact_job(&self, k: ClusterKey, supports: &[u32], dvv: Vec3) -> ImpactJob {
        let dv = dvv.length();
        let cl = &self.clusters[k];
        let b = cl.building;
        let bd = &self.buildings[b as usize];
        let n = bd.n_chunks();
        let g_eff = dv / self.cfg.impact_duration.max(1e-3);
        let mut weight = vec![0.0f32; n];
        let mut alive = vec![false; n];
        let mut anchor = vec![false; n];
        for &c in &cl.chunks {
            let cu = c as usize;
            if bd.state[cu] == ChunkState::InCluster(k) {
                alive[cu] = true;
                weight[cu] = bd.bld.chunks[cu].mass * g_eff;
            }
        }
        for &c in supports {
            anchor[c as usize] = alive[c as usize];
        }
        // The cluster lands on a rubble bed rather than on a few points: every chunk whose COM is
        // within `impact_support_band` (along the impact direction) of the contacts also bears load.
        if self.cfg.impact_support_band > 0.0 {
            let pose = self.phys.body_state(cl.body).pose;
            let up = dvv.try_normalize().unwrap_or(Vec3::Z);
            let h = |c: u32| pose.transform_point(v3(bd.bld.chunks[c as usize].com)).dot(up);
            let base = supports.iter().map(|&c| h(c)).fold(f32::INFINITY, f32::min);
            for &c in &cl.chunks {
                if alive[c as usize] && h(c) <= base + self.cfg.impact_support_band {
                    anchor[c as usize] = true;
                }
            }
        }
        let mut edge_alive = vec![false; bd.edge_alive.len()];
        let mut edges = Vec::with_capacity(cl.edges.len());
        for &e in &cl.edges {
            let ed = &bd.bld.edges[e as usize];
            if bd.edge_alive[e as usize] && alive[ed.a as usize] && alive[ed.b as usize] {
                edge_alive[e as usize] = true;
                edges.push(e);
            }
        }
        let mut scfg = self.stress_config(self.cfg.impact_iters);
        // impact stress keeps the joint model it was tuned with (no crushing/tipping/buckling;
        // the landing load is transient)
        scfg.compression_factor = 10.0;
        scfg.bearing_sections = false;
        scfg.buckling = false;
        scfg.catchup_frac = 0.0;
        scfg.hold_time = 0.0;
        scfg.tol = self.cfg.impact_tol;
        // big landing solves dominate the worst ticks (the tick that collects them blocks on
        // them): run their CG kernels on rayon. The CG reductions use fixed chunks, so the result
        // is bit-identical to the sequential solve.
        scfg.par_threshold = IMPACT_PAR_THRESHOLD;
        ImpactJob {
            building: b,
            graph: bd.stress_graph.clone(),
            inter_panel: bd.edge_inter_panel.clone(),
            weight,
            alive,
            anchor,
            edge_alive,
            edges,
            scfg,
            joint: self.cfg.impact_joint_factor.max(1e-3),
            rounds: self.cfg.impact_rounds.max(1),
            max_breaks: self.cfg.impact_max_breaks,
            debug: debug_impact().then(|| format!("t={:.2} chunks={} dv={dv:.2} g_eff={g_eff:.0}", self.time, cl.chunks.len())),
        }
    }

    // ------------------------------------------------------------------ 9-10. settle + budgets

    fn settle(&mut self, dt: f32) {
        let mut freeze = vec![];
        let mut despawn = vec![];
        for (k, cl) in self.clusters.iter_mut() {
            let s = self.phys.body_state(cl.body);
            cl.age += dt;
            let resting = s.sleeping || (s.linvel.length() < self.cfg.rest_lin_vel && s.angvel.length() < self.cfg.rest_ang_vel);
            cl.rest_time = if resting { cl.rest_time + dt } else { 0.0 };
            let big = !cl.debris && cl.mass >= self.cfg.freeze_min_mass;
            if s.world_com.z < -100.0 {
                despawn.push(k);
            } else if (big || self.cfg.freeze_debris || self.cfg.keep_debris)
                && cl.rest_time >= self.cfg.freeze_time
                && cl.age >= self.cfg.freeze_min_age
                // a body resting on something that still moves would be left hanging when that
                // support moves away, so wait until everything under it has settled too
                && !self.phys.touches_moving_dynamic(cl.body, self.cfg.rest_lin_vel, self.cfg.rest_ang_vel)
            {
                freeze.push(k);
            } else if !big && cl.age >= if self.cfg.keep_debris { self.cfg.keep_debris_max_age } else { self.cfg.debris_ttl } {
                despawn.push(k);
            } else if big && cl.age >= self.cfg.max_dynamic_time {
                freeze.push(k);
            } else if !big && !self.cfg.keep_debris && cl.age >= self.cfg.max_dynamic_time {
                despawn.push(k);
            }
        }
        for k in freeze {
            self.freeze_cluster(k);
        }
        for k in despawn {
            self.despawn_cluster(k);
        }
    }

    fn budgets(&mut self) {
        let mut in_flight: usize = self.clusters.values().map(|c| c.chunks.len()).sum();
        let mut over = self.clusters.len().saturating_sub(self.max_clusters());
        if over == 0 && in_flight <= self.cfg.max_chunks_in_flight {
            return;
        }
        let mut order: Vec<(f32, f32, ClusterKey)> = self.clusters.iter().map(|(k, c)| (c.mass, -c.age, k)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        for (_, _, k) in order {
            if over == 0 && in_flight <= self.cfg.max_chunks_in_flight {
                break;
            }
            let cl = &self.clusters[k];
            in_flight -= cl.chunks.len();
            over = over.saturating_sub(1);
            if !cl.debris && cl.mass >= self.cfg.freeze_min_mass && cl.rest_time > 0.2 {
                self.freeze_cluster(k);
            } else {
                self.despawn_cluster(k);
            }
        }
    }

    /// Smallest destroyed chunk / detached piece kept as debris (smaller ones shatter).
    fn debris_min_volume(&self) -> f32 {
        if self.cfg.keep_debris { self.cfg.keep_debris_min_volume } else { self.cfg.debris_min_volume }
    }

    /// Cap on moving pieces (see `WorldConfig::keep_debris`).
    fn max_clusters(&self) -> usize {
        if self.cfg.keep_debris { self.cfg.keep_debris_max_clusters } else { self.cfg.max_dynamic_clusters }
    }

    /// Remove a static (building or frozen) collider and remember where it was.
    fn drop_static_collider(&mut self, b: u32, c: u32) {
        let bd = &mut self.buildings[b as usize];
        let Some(h) = bd.collider[c as usize].take() else { return };
        if self.cfg.thaw_unsupported {
            self.support_lost.push(bd.collider_aabb[c as usize]);
        }
        self.phys.remove_collider(h);
    }

    /// Remove a dynamic body and remember where it was.
    fn drop_body(&mut self, body: BodyId) {
        if self.cfg.thaw_unsupported {
            if let Some(a) = self.phys.body_aabb(body) {
                self.support_lost.push(a);
            }
        }
        self.phys.remove_body(body);
    }

    /// Frozen rubble is static, so nothing re-checks its support by itself. Two triggers do:
    /// - geometry removed this tick (destroyed, detached, despawned): wake sleeping bodies around
    ///   it (a removed support never wakes what sleeps on it) and test the frozen groups there;
    /// - a slow round-robin sweep over all frozen chunks, for supports that moved away.
    /// A group with nothing under it thaws back into a dynamic cluster; it falls and freezes
    /// again once it has settled. Thawing removes colliders too, so stacks come down layer by layer.
    fn thaw_unsupported(&mut self) {
        if !self.cfg.thaw_unsupported {
            self.support_lost.clear();
            self.support_watch.clear();
            return;
        }
        let boxes = std::mem::take(&mut self.support_lost);
        let m = self.cfg.thaw_margin;
        let mut hits = vec![];
        let mut seeds: Vec<(u32, u32)> = vec![];
        // The box queries only matter through sleeping bodies (woken) and frozen chunks (seeds);
        // skip the half that has nothing to find (neither set changes during the loop).
        let (wake, seek) = if boxes.is_empty() {
            (false, false)
        } else {
            (
                self.clusters.values().any(|cl| self.phys.body_state(cl.body).sleeping),
                self.buildings.iter().any(|b| b.state.contains(&ChunkState::Frozen)),
            )
        };
        let dilate = |(lo, hi): (Vec3, Vec3)| (Vec3::new(lo.x - m, lo.y - m, lo.z - m), Vec3::new(hi.x + m, hi.y + m, hi.z + m));
        // A box only matters if it finds a sleeping body or a frozen chunk. Pre-test spatially
        // grouped boxes (a box finds nothing its group's bounding box does not); boxes of groups
        // that find nothing are skipped. Exact: sleeping bodies only get fewer and frozen chunks
        // do not change while the boxes are processed, which happens in the original order.
        let mut live = vec![true; if wake || seek { boxes.len() } else { 0 }];
        if live.len() > THAW_BOX_GROUP {
            let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
            for &(a, b) in &boxes {
                lo = lo.min(a + b);
                hi = hi.max(a + b);
            }
            let scale = Vec3::splat(1023.0) / (hi - lo).max(Vec3::splat(1e-6));
            let mut order: Vec<(u32, u32)> = boxes
                .iter()
                .enumerate()
                .map(|(i, &(a, b))| {
                    let q = ((a + b - lo) * scale).clamp(Vec3::ZERO, Vec3::splat(1023.0));
                    (morton3(q.x as u32, q.y as u32, q.z as u32), i as u32)
                })
                .collect();
            order.sort_unstable();
            let buildings = &self.buildings;
            let frozen = |tag: u128| {
                seek && tag_kind(tag) == KIND_CHUNK && {
                    let (b, c) = tag_chunk(tag);
                    buildings[b as usize].state[c as usize] == ChunkState::Frozen
                }
            };
            // hierarchical: test a Morton range's bounding box, recurse into halves that touch
            let mut stack = vec![(0usize, order.len())];
            while let Some((a, b)) = stack.pop() {
                let (mut glo, mut ghi) = dilate(boxes[order[a].1 as usize]);
                for &(_, i) in &order[a + 1..b] {
                    let (l, h) = dilate(boxes[i as usize]);
                    glo = glo.min(l);
                    ghi = ghi.max(h);
                }
                if !self.phys.aabb_touches(glo, ghi, wake, &frozen) {
                    for &(_, i) in &order[a..b] {
                        live[i as usize] = false;
                    }
                } else if b - a > THAW_BOX_GROUP {
                    let mid = (a + b) / 2;
                    stack.push((mid, b));
                    stack.push((a, mid));
                }
            }
        }
        for (i, &bx) in boxes.iter().enumerate() {
            if !live.get(i).copied().unwrap_or(false) {
                continue;
            }
            let (qlo, qhi) = dilate(bx);
            hits.clear();
            match (wake, seek) {
                (true, true) => self.phys.wake_and_query_aabb(qlo, qhi, &mut hits),
                (true, false) => self.phys.wake_bodies_in_aabb(qlo, qhi),
                _ => self.phys.query_aabb(qlo, qhi, &mut hits),
            }
            for &(_, tag) in &hits {
                if tag_kind(tag) == KIND_CHUNK {
                    let (b, c) = tag_chunk(tag);
                    if self.buildings[b as usize].state[c as usize] == ChunkState::Frozen {
                        seeds.push((b, c));
                    }
                }
            }
        }
        // round-robin sweep for supports that moved away. Only moving bodies can move away,
        // so with no dynamic clusters there is nothing to find.
        let mut budget = if self.clusters.is_empty() { 0 } else { self.cfg.thaw_sweep_per_tick };
        let mut scanned = 0usize;
        let total: usize = self.buildings.iter().map(|b| b.n_chunks()).sum();
        while budget > 0 && scanned < total {
            let (bi, ci) = self.sweep_cursor;
            if bi >= self.buildings.len() {
                self.sweep_cursor = (0, 0);
                if self.buildings.is_empty() {
                    break;
                }
                continue;
            }
            if ci >= self.buildings[bi].n_chunks() {
                self.sweep_cursor = (bi + 1, 0);
                continue;
            }
            self.sweep_cursor = (bi, ci + 1);
            scanned += 1;
            if self.buildings[bi].state[ci] == ChunkState::Frozen {
                seeds.push((bi as u32, ci as u32));
                budget -= 1;
            }
        }
        seeds.append(&mut self.support_watch);
        let mut checked = std::collections::HashSet::new();
        for (b, c) in seeds {
            if self.buildings[b as usize].state[c as usize] != ChunkState::Frozen || !checked.insert((b, c)) {
                continue;
            }
            let group = self.frozen_group(b, c);
            for &g in &group {
                checked.insert((b, g));
            }
            match self.group_supported(b, &group) {
                Support::Solid => continue,
                Support::Transient => {
                    // resting on something that may move away: look again next tick
                    self.support_watch.push((b, c));
                    continue;
                }
                Support::None => {}
            }
            let bd = &self.buildings[b as usize];
            let pose = bd.chunk_pose[c as usize];
            let mass: f32 = group.iter().map(|&g| bd.bld.chunks[g as usize].mass).sum();
            let debris = mass < self.cfg.freeze_min_mass;
            let k = self.create_cluster(b, group, pose, Vec3::ZERO, Vec3::ZERO, debris);
            self.emit_detached(k);
        }
    }

    /// Does something outside the group hold it up from below? A collider counts as support if
    /// the group touches it when moved down by `thaw_probe` but not when lifted by the same
    /// amount. Side contacts persist both ways, so two piles leaning on each other in mid-air
    /// do not hold each other up.
    fn group_supported(&self, b: u32, group: &[u32]) -> Support {
        let bd = &self.buildings[b as usize];
        let members: std::collections::HashSet<u32> = group.iter().copied().collect();
        let outside = |tag: &u128| {
            !(tag_kind(*tag) == KIND_CHUNK && {
                let (tb, tc) = tag_chunk(*tag);
                tb == b && members.contains(&tc)
            })
        };
        self.rests_on_something(b, group, |c| bd.chunk_pose[c as usize], outside, Some(self.cfg.tip_margin))
    }

    /// Shared by the thaw check and `floating_report`: do the chunks (at `pose_of(c)`) touch
    /// anything accepted by `outside` when moved down by `thaw_probe` but not when lifted?
    ///
    /// With `com_margin`, a group of several chunks only counts as solidly supported if its
    /// centre of mass lies over (within the margin of) the footprint of its chunks that rest on
    /// something solid: a frozen floor whose supports were shot away except under one corner
    /// must come down and tip, not hang off that corner.
    fn rests_on_something(
        &self, b: u32, chunks: &[u32], pose_of: impl Fn(u32) -> Pose, outside: impl Fn(&u128) -> bool, com_margin: Option<f32>,
    ) -> Support {
        let com_margin = com_margin.filter(|_| chunks.len() > 1);
        let mut patches: Vec<[f64; 2]> = vec![];
        let bd = &self.buildings[b as usize];
        // lowest chunks first: they are the ones that rest on something; stop at the first solid hit
        let mut order: Vec<(f32, u32)> =
            chunks.iter().map(|&c| (pose_of(c).transform_point(v3(bd.bld.chunks[c as usize].aabb_min)).z, c)).collect();
        order.sort_by(|a, b| a.0.total_cmp(&b.0));
        let d = Vec3::Z * self.cfg.thaw_probe;
        let (mut below, mut above) = (vec![], vec![]);
        // static building or frozen rubble stays put; a body or a collapsing piece may move away
        let solid = |t: u128| match tag_kind(t) {
            KIND_GROUND => true,
            KIND_CHUNK => {
                let (tb, tc) = tag_chunk(t);
                matches!(self.buildings[tb as usize].state[tc as usize], ChunkState::Static | ChunkState::Frozen)
            }
            _ => false,
        };
        let mut transient = false;
        let mut judge = |hits: &mut dyn Iterator<Item = u128>| {
            for t in hits {
                if solid(t) {
                    return true;
                }
                transient = true;
            }
            false
        };
        for (_, c) in order {
            let p = pose_of(c);
            let shape = &bd.shapes[c as usize];
            below.clear();
            self.phys.shape_overlaps(shape, Pose::from_parts(p.translation - d, p.rotation), &mut below);
            below.retain(|t| outside(t));
            if below.is_empty() {
                continue;
            }
            // the ground is never beside you; debris sunk into it by more than the probe still rests on it
            let mut on_solid = below.iter().any(|t| tag_kind(*t) == KIND_GROUND);
            above.clear();
            self.phys.shape_overlaps(shape, Pose::from_parts(p.translation + d, p.rotation), &mut above);
            let mut under: Vec<u128> = below.iter().copied().filter(|t| !above.contains(t)).collect();
            if under.is_empty() {
                // wedged into a pile (penetration deeper than the probe): what it rests on lets go
                // within a short lift, a wall beside it does not
                above.clear();
                self.phys.shape_overlaps(shape, Pose::from_parts(p.translation + d * 8.0, p.rotation), &mut above);
                under = below.iter().copied().filter(|t| !above.contains(t)).collect();
            }
            on_solid = on_solid || judge(&mut under.into_iter());
            if on_solid {
                if com_margin.is_none() {
                    return Support::Solid;
                }
                let bb = shape.compute_aabb(&p);
                let (lo, hi) = (bb.mins, bb.maxs);
                patches.extend_from_slice(&[
                    [lo.x as f64, lo.y as f64],
                    [hi.x as f64, lo.y as f64],
                    [hi.x as f64, hi.y as f64],
                    [lo.x as f64, hi.y as f64],
                ]);
            }
        }
        if let (Some(margin), false) = (com_margin, patches.is_empty()) {
            let (mut m, mut cx, mut cy) = (0f64, 0f64, 0f64);
            for &c in chunks {
                let ch = &bd.bld.chunks[c as usize];
                let w = pose_of(c).transform_point(v3(ch.com));
                m += ch.mass as f64;
                cx += ch.mass as f64 * w.x as f64;
                cy += ch.mass as f64 * w.y as f64;
            }
            if m > 0.0 && crate::building::dist_outside_hull(&mut patches, [cx / m, cy / m]) <= margin as f64 {
                return Support::Solid;
            }
        }
        if transient { Support::Transient } else { Support::None }
    }

    fn touching(&self, b: u32, chunks: &[u32], pose_of: impl Fn(u32) -> Pose, outside: impl Fn(&u128) -> bool) -> usize {
        let bd = &self.buildings[b as usize];
        let mut all = std::collections::HashSet::new();
        let mut hits = vec![];
        for &c in chunks {
            hits.clear();
            self.phys.shape_overlaps(&bd.shapes[c as usize], pose_of(c), &mut hits);
            all.extend(hits.iter().copied().filter(|t| outside(t)));
        }
        all.len()
    }

    /// Diagnostic: static chunks of the intact structure that no longer reach an anchor through
    /// alive joints (a full search from every anchor, independent of the incremental dirty-node
    /// search) and are not already waiting to collapse. Structural chunks, then non-structural
    /// (glass, cosmetic) ones that hang on nothing grounded. Should always be empty after `step`.
    pub fn unanchored_static(&self, b: BuildingId) -> (Vec<u32>, Vec<u32>) {
        let bd = &self.buildings[b.0 as usize];
        let (mut structural, mut loose) = (vec![], vec![]);
        for comp in bd.ungrounded_components() {
            for c in comp {
                if bd.structural[c as usize] { structural.push(c) } else { loose.push(c) }
            }
        }
        (structural, loose)
    }

    /// Diagnostic: rubble that should be falling but is not. Lists frozen groups and resting
    /// (sleeping or slow) dynamic clusters that have nothing under them. O(chunks) shape
    /// queries: for tests and debug overlays, not for every tick.
    pub fn floating_report(&self) -> FloatingReport {
        let mut rep = FloatingReport::default();
        for (bi, bd) in self.buildings.iter().enumerate() {
            let b = bi as u32;
            let mut seen = std::collections::HashSet::new();
            for c in 0..bd.n_chunks() as u32 {
                if bd.state[c as usize] != ChunkState::Frozen || seen.contains(&c) {
                    continue;
                }
                let group = self.frozen_group(b, c);
                seen.extend(group.iter().copied());
                if self.group_supported(b, &group) == Support::None {
                    let z = group.iter().map(|&g| self.chunk_world_com(bi, g as usize).z).fold(f32::INFINITY, f32::min);
                    let members: std::collections::HashSet<u32> = group.iter().copied().collect();
                    let touching = self.touching(b, &group, |g| bd.chunk_pose[g as usize], |t| {
                        !(tag_kind(*t) == KIND_CHUNK && tag_chunk(*t).0 == b && members.contains(&tag_chunk(*t).1))
                    });
                    rep.frozen.push(FloatingGroup { building: BuildingId(b), chunks: group, min_com_z: z, touching });
                }
            }
        }
        for (k, cl) in self.clusters.iter() {
            let s = self.phys.body_state(cl.body);
            let resting = s.sleeping || (s.linvel.length() < self.cfg.rest_lin_vel && s.angvel.length() < self.cfg.rest_ang_vel);
            if !resting {
                continue;
            }
            let me = cluster_tag(k);
            let b = cl.building;
            let bi = b as usize;
            if self.rests_on_something(b, &cl.chunks, |c| self.chunk_world_pose(bi, c as usize), |t| *t != me, None) == Support::None {
                let z = cl.chunks.iter().map(|&g| self.chunk_world_com(bi, g as usize).z).fold(f32::INFINITY, f32::min);
                let touching = self.touching(b, &cl.chunks, |c| self.chunk_world_pose(bi, c as usize), |t| *t != me);
                rep.clusters.push((cluster_id(k), FloatingGroup { building: BuildingId(b), chunks: cl.chunks.clone(), min_com_z: z, touching }));
            }
        }
        rep
    }

    /// The frozen chunks that froze together with `seed`: connected over alive internal edges
    /// and sharing its pose.
    fn frozen_group(&self, b: u32, seed: u32) -> Vec<u32> {
        let bd = &self.buildings[b as usize];
        let p0 = bd.chunk_pose[seed as usize];
        let same = |p: &Pose| (p.translation - p0.translation).length() < 1e-4 && p.rotation.dot(p0.rotation).abs() > 1.0 - 1e-6;
        let mut out = vec![seed];
        let mut seen = std::collections::HashSet::from([seed]);
        let mut i = 0;
        while i < out.len() {
            let cu = out[i] as usize;
            i += 1;
            for k in bd.csr_off[cu] as usize..bd.csr_off[cu + 1] as usize {
                let (nb, e) = (bd.csr_nbr[k], bd.csr_edge[k] as usize);
                if bd.edge_alive[e]
                    && bd.state[nb as usize] == ChunkState::Frozen
                    && same(&bd.chunk_pose[nb as usize])
                    && seen.insert(nb)
                {
                    out.push(nb);
                }
            }
        }
        out
    }

    /// Settled big cluster -> static, anchored, still-destructible rubble.
    fn freeze_cluster(&mut self, k: ClusterKey) {
        let Some(cl) = self.clusters.remove(k) else { return };
        let s = self.phys.body_state(cl.body);
        self.phys.remove_body(cl.body);
        let b = cl.building;
        for &c in &cl.chunks {
            let bd = &mut self.buildings[b as usize];
            let cu = c as usize;
            if bd.state[cu] != ChunkState::InCluster(k) {
                continue;
            }
            bd.state[cu] = ChunkState::Frozen;
            bd.chunk_pose[cu] = s.pose;
            let (h, aabb) = self.phys.add_static_collider(bd.shapes[cu].clone(), s.pose, chunk_tag(b, c));
            bd.collider[cu] = Some(h);
            bd.collider_aabb[cu] = aabb;
        }
        self.needs_sync = true;
        // classify its support next tick (it may rest on rubble that is still moving)
        if let Some(&c) = cl.chunks.first() {
            self.support_watch.push((b, c));
        }
        self.events.push(Event::ClusterFrozen { cluster: cluster_id(k), transform: pose_to_rowmajor(&s.pose) });
    }

    fn despawn_cluster(&mut self, k: ClusterKey) {
        let Some(cl) = self.clusters.remove(k) else { return };
        let s = self.phys.body_state(cl.body);
        self.drop_body(cl.body);
        let bd = &mut self.buildings[cl.building as usize];
        for &c in &cl.chunks {
            if bd.state[c as usize] == ChunkState::InCluster(k) {
                bd.state[c as usize] = ChunkState::Gone;
                bd.chunk_pose[c as usize] = s.pose;
            }
        }
        self.events.push(Event::ClusterDespawned { cluster: cluster_id(k) });
    }
}
