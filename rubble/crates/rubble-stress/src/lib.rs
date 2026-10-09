#![allow(clippy::needless_range_loop, clippy::manual_memcpy)]
//! # rubble-stress — structural stress solver
//!
//! ## Model
//! The building's connection graph is treated as a **resistive (electrical) network**:
//! * every chunk is a node with a load source `w_i = m_i·g` (Newtons),
//! * every alive anchor is a Dirichlet sink with potential `φ = 0`,
//! * every alive edge has conductance `k_e = capacity_e / L_e` (axial stiffness E·A/L with
//!   `L_e` the centre-to-centre distance of its chunks).
//!
//! We solve the weighted graph Laplacian `L φ = w` over the *supported* nodes (alive,
//! non-anchor, connected to an anchor through conducting edges). The edge "load flow" is
//! `f_e = k_e (φ_a − φ_b)` (positive = load travels a→b) and the axial utilization is
//! `u_e = |f_e| / capacity_e`. Kirchhoff's current law makes flow conservative: the total
//! flow into anchors equals the total weight of supported nodes, and a column's edge carries
//! exactly the weight above it.
//!
//! **Approximation / limitations.** Load splits across parallel paths proportional to their
//! conductance (like current), not according to elastic stiffness/compatibility; there is no
//! notion of compression vs tension, no friction, no lateral loads, and a beam spanning two
//! supports sends each half of its load to the nearer support (so the mid-span edge carries
//! ~0 flow; real mid-span bending is not captured). It's a stable, cheap, monotone indicator
//! of "how much of the structure hangs off this bond", which is what gameplay needs.
//!
//! ## Capacities
//! `capacity` is the joint's tensile/shear strength (area × bond strength), scaled by the
//! weaker endpoint's remaining strength (`StressInput::node_strength`, e.g. hp/hp_max).
//! * **Bearing** (load moving down across a near-vertical-normal joint): × `compression_factor`
//!   (3.5: bgen concrete bond 8 MPa → ~28 MPa crushing) × the buckling factor φ.
//! * **Buckling** ([`StressGraph::with_columns`]): column chunks joined by vertical joints form
//!   stacks; any joint to a non-column member (slab, beam, wall) or an anchor braces the stack
//!   at that height. A chunk's unbraced length L is the distance between the nearest braces
//!   below and above it (2× its height above the last brace for a free top), and
//!   `φ = 1/(1 + (L/r)²/k)` (Rankine) with `r = thickness/√12`, `k = rankine_k` (1200): a 0.5 m
//!   column braced every 3.5 m keeps 0.67, losing a bracing level drops it to ~0.33.
//!   Recomputed whenever the topology changes.
//! * **Bearing sections** (eccentric load, tipping): vertical-normal contacts between two
//!   members (or, in an ungrouped structure, all those in a 0.5 m height band) form a section.
//!   The load passing down through it (`N`) and its moment about the section centroid
//!   (`m = Σ f·(C_up − c)`) give a linear stress field; each joint is checked at the extreme
//!   fibres of its own patch, compression against `bond·compression_factor·φ`, tension against
//!   `bond·tension_factor` (5: the members' reinforcement, as for flexure). A load centroid
//!   outside the support puts the far side into tension: it breaks and the structure tips.
//! * Lateral transfer through side joints and hanging load use the plain capacity.
//!
//! ## Catch-up after big damage
//! The tick's removed joints are summed by the flow they carried; if that exceeds
//! `catchup_frac` (5 %) of the total supported load, the building is solved globally to
//! convergence within the same call (≤ `catchup_max_ms`), so utilization reflects the damage
//! at once instead of creeping up over seconds of warm-started windowed iterations.

//! ## Bending term (`cfg.bending`)
//! For edges with near-horizontal contact normal (`|n·z| < 0.5`) we add a moment estimate
//! `M_e = |f_e| · |(C_up − c_e)·n̂_h|`: the **net** lateral load through the joint times the
//! lever arm, the offset along the joint normal between the joint centroid `c_e` and the
//! centroid `C_up` of the load carried by the upstream node (the node the load comes from).
//! Carried loads are propagated down the acyclic net-flow graph (upstream = higher φ):
//! `F_i = w_i + Σ_in f`, `C_i = (w_i x_i + Σ_in f·C_j) / F_i` — a convex combination of the
//! positions of the loads that end up passing through `i`, with every weight positive. This
//! is the "weighted downstream centroid" approximation (perfect mixing at each node).
//! Properties: a uniformly supported wall or slab has no net lateral flow, hence no moment
//! (an earlier first-moment-potential formulation let opposite lateral flows cancel in `f`
//! but not in the moment, giving intact walls u ≈ 1.5); a cantilever of length L gets
//! `M_root = wL²/2`; a balanced T cancels. The in-plane offset (torsion) is ignored.
//! Incremental: after a solve, carried loads are re-propagated from the solved nodes,
//! upstream first, stopping wherever a node's carried load/centroid doesn't change.
//!
//! Stress: `σ = M / S`, `S = A^1.5/6` (square-joint section), and the joint's flexural
//! capacity is `flexural_factor · bond · S` with `bond = capacity / A` — the factor models
//! reinforcement (default 5; plain concrete = 1). Hence
//! `u_bend = bend_scale · 6|M| / (flexural_factor · capacity · √A)`; `u = u_axial + u_bend`.
//! Areas come from [`StressGraph::with_areas`]; without it `A = capacity / DEFAULT_BOND_STRENGTH`.
//! ### Member sections ([`StressGraph::with_node_groups`], chunk → element id)
//! Without groups every horizontal joint is its own section (fine for single-member
//! synthetic structures). With groups, bending is evaluated per **section**, not per contact:
//! * *interface*: all horizontal contacts between two members (e.g. ext_wall↔column);
//! * *intra-member cut*: joints of one member whose horizontal normal falls in the same 30°
//!   direction bin and whose centroids lie in the same 1 m slab across that direction (two
//!   staggered slabbings; each joint takes the stronger of its two cuts).
//!
//! Section moment `M = |Σ_e |f_e|·(C_up,e − c_S)·n̂_S|` about the section centroid (opposite
//! rotations cancel), section modulus `S = A·h/6` from the alive contact area `A` and its
//! vertical extent `h` (gravity moments bend about a horizontal axis), so
//! `u_bend = bend_scale · 6M / (flexural_factor · Σcap · h)`, shared by all its joints
//! (a crack runs through the whole section; dead joints shrink it). Load handed over from
//! another member is placed at the contact centroid (members are checked one by one, with
//! supported members' loads applied at the connections). A single group = no grouping.
//!
//! Limitations: when a joint breaks the network re-routes load through any remaining path
//! (redundancy is generous); load attraction follows axial stiffness, so slender spans that
//! receive load (spandrels over ribbon windows, slab corners) can be over-stressed.

//! ## Solver
//! Jacobi-PCG in `f32` on a compact CSR of the active region, solving for the *correction*
//! `δ` to the stored (`f64`) potentials, so warm starts are free. Convergence criterion: every
//! node's force imbalance `|r_i| ≤ tol · w_rms`.
//! * **Global mode** (first call, or graph small: supported nodes ≤
//!   `region_max_nodes`): all supported nodes; the system matrix is cached until the
//!   topology changes.
//! * **Local mode** (after local changes in a large graph): BFS from the dirty nodes (rows
//!   whose equations changed, or with leftover residual) up to `region_max_nodes`, solve with
//!   the region's boundary ring held at its current potentials (Dirichlet), then any ring node
//!   whose residual exceeds the threshold becomes dirty for the next tick. This is a moving-window
//!   Schwarz iteration: a removal is resolved locally at once and its far-field effect diffuses
//!   out over subsequent ticks (the "creak delay"). Window seeds are the
//!   worst-residual dirty nodes; while work remains the window grows ×1.5 per tick (up to
//!   `max_region_growth × region_max_nodes`), and after 8 stalled ticks at that size the solver
//!   escalates to global steps, so `converged` is guaranteed to become true eventually.
//!   Boundary leftovers below `local_tol_factor · tol · w_rms` are accepted, not chased.
//!
//! Warm starting matters for *accuracy per iteration*, not for the strict tolerance: a removal
//! creates an O(chunk weight) residual, so reaching `tol` takes a similar number of CG
//! iterations either way, but the warm solution is within ~1% utilization after a handful of
//! iterations whereas a cold one needs ~the full solve (see tests).
//!
//! Changes are detected by diffing the inputs against the previous call (memcmp-speed).
//! Removals re-check support with a best-first search toward anchors guided by `φ` (cost ~
//! path length unless the component really detached). Additions (alive/anchor/edge turned
//! on) trigger an O(N+E) reclassification.
//!
//! ## Breaking
//! An edge must have `u > 1` continuously for `hold_time` seconds (timer accumulates `dt`
//! per tick while over, resets otherwise) before it is reported in `to_break`, worst first,
//! at most `max_breaks_per_tick`. The solver never mutates inputs: the caller breaks the edge
//! by clearing `edge_alive` next tick.

mod cg;
pub mod testgraphs;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Assumed bond strength (Pa) used to guess contact area from capacity when no areas given.
pub const DEFAULT_BOND_STRENGTH: f32 = 1.0e6;

/// Unconverged local ticks at the maximum window size before escalating to global steps.
const STALL_TICKS: usize = 8;
/// PCG iterations per catch-up round (rounds repeat until converged or out of time).
const CATCHUP_ROUND_ITERS: usize = 250;

/// Intra-member cuts: horizontal-normal joints are grouped by direction (CUT_BINS bins over
/// 180°) and by position along that direction (CUT_BIN m slabs).
const CUT_BINS: usize = 6;
const CUT_ANGLE: f64 = std::f64::consts::PI / CUT_BINS as f64;
const CUT_BIN: f64 = 1.0;
/// Height band (m) of intra-member bearing sections (two staggered bandings).
const BEAR_BAND: f64 = 0.5;
const SEC_BEND: u8 = 0;
const SEC_BEAR: u8 = 1;
/// cut key -> (edge, slot (2 = inter-member), orientation sign)
type CutBuckets = std::collections::HashMap<(u32, u32, i64), Vec<(u32, u8, f32)>>;

const DEAD: u8 = 0;
const UNKNOWN: u8 = 1; // supported, solved for
const ANCHOR: u8 = 2;
const FLOATING: u8 = 3; // alive, not connected to any anchor

/// Immutable graph data in CSR form, built once per building.
#[derive(Clone, Debug)]
pub struct StressGraph {
    n: usize,
    offsets: Vec<u32>,
    adj_node: Vec<u32>,
    adj_edge: Vec<u32>,
    edge_ab: Vec<[u32; 2]>,
    capacity: Vec<f32>,
    /// Conductance per edge: capacity / centre-to-centre distance (axial stiffness E·A/L).
    cond: Vec<f32>,
    /// Edge centroid xy relative to `origin`.
    centroid_xy: Vec<[f32; 2]>,
    /// Edge centroid z (absolute).
    edge_z: Vec<f32>,
    /// Node group (element id); empty = all nodes in one group.
    group: Vec<u32>,
    /// Interface id per edge (u32::MAX = none); see [`StressGraph::with_node_groups`].
    iface_of: Vec<[u32; 2]>,
    iface_off: Vec<u32>,
    iface_edges: Vec<u32>,
    /// Oriented unit horizontal normal per interface (from lower to higher group id).
    iface_n: Vec<[f32; 2]>,
    /// Section kind: SEC_BEND (horizontal-normal joints: bending cut / interface) or
    /// SEC_BEAR (vertical-normal joints: bearing section with eccentricity, i.e. tipping).
    sec_kind: Vec<u8>,
    /// Node centre z (absolute).
    node_z: Vec<f32>,
    /// Column section thickness per node (0 = not a column); see [`StressGraph::with_columns`].
    col_thick: Vec<f32>,
    /// Node position xy relative to `origin`.
    node_xy: Vec<[f32; 2]>,
    horizontal: Vec<bool>,
    /// For near-vertical normals (|n̂·z| ≥ 0.5): sign of n_z (+1 = b is above a), else 0.
    vert_sign: Vec<i8>,
    /// Unit horizontal part of the contact normal (for horizontal edges).
    nrm_h: Vec<[f32; 2]>,
    area: Vec<f32>,
    /// `6 / (capacity · √area)` for horizontal edges, else 0.
    bend_coef: Vec<f32>,
    origin: [f32; 3],
    length_scale: f32,
}

impl StressGraph {
    /// Build the CSR. `edges[e] = (a, b)`, `normal[e]` is the contact normal (a→b),
    /// `centroid[e]` the contact centroid, `node_pos[i]` the chunk centre of mass.
    pub fn new(
        n_nodes: usize,
        edges: &[(u32, u32)],
        capacity: &[f32],
        centroid: &[[f32; 3]],
        normal: &[[f32; 3]],
        node_pos: &[[f32; 3]],
    ) -> Self {
        let m = edges.len();
        assert_eq!(capacity.len(), m, "capacity len");
        assert_eq!(centroid.len(), m, "centroid len");
        assert_eq!(normal.len(), m, "normal len");
        assert_eq!(node_pos.len(), n_nodes, "node_pos len");
        let mut deg = vec![0u32; n_nodes + 1];
        for &(a, b) in edges {
            assert!((a as usize) < n_nodes && (b as usize) < n_nodes, "edge endpoint out of range");
            deg[a as usize] += 1;
            deg[b as usize] += 1;
        }
        let mut offsets = vec![0u32; n_nodes + 1];
        for i in 0..n_nodes {
            offsets[i + 1] = offsets[i] + deg[i];
        }
        let mut fill: Vec<u32> = offsets[..n_nodes].to_vec();
        let mut adj_node = vec![0u32; 2 * m];
        let mut adj_edge = vec![0u32; 2 * m];
        for (e, &(a, b)) in edges.iter().enumerate() {
            let pa = fill[a as usize] as usize;
            fill[a as usize] += 1;
            adj_node[pa] = b;
            adj_edge[pa] = e as u32;
            let pb = fill[b as usize] as usize;
            fill[b as usize] += 1;
            adj_node[pb] = a;
            adj_edge[pb] = e as u32;
        }
        // origin = bbox centre (keeps moment sources well-conditioned in f32)
        let (mut lo, mut hi) = ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]);
        for p in node_pos {
            for c in 0..3 {
                lo[c] = lo[c].min(p[c]);
                hi[c] = hi[c].max(p[c]);
            }
        }
        let (origin, length_scale) = if n_nodes == 0 {
            ([0.0; 3], 1.0)
        } else {
            let o = [0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])];
            let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(hi[2] - lo[2]);
            (o, ext.max(1e-3))
        };
        let node_xy = node_pos.iter().map(|p| [p[0] - origin[0], p[1] - origin[1]]).collect();
        let centroid_xy = centroid.iter().map(|c| [c[0] - origin[0], c[1] - origin[1]]).collect();
        let horizontal = normal
            .iter()
            .map(|n| {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                len > 0.0 && (n[2] / len).abs() < 0.5
            })
            .collect();
        let vert_sign = normal
            .iter()
            .map(|n| {
                let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
                if len > 0.0 && (n[2] / len).abs() >= 0.5 {
                    if n[2] > 0.0 {
                        1
                    } else {
                        -1
                    }
                } else {
                    0
                }
            })
            .collect();
        let nrm_h = normal
            .iter()
            .map(|n| {
                let l = (n[0] * n[0] + n[1] * n[1]).sqrt();
                if l > 0.0 {
                    [n[0] / l, n[1] / l]
                } else {
                    [0.0, 0.0]
                }
            })
            .collect();
        let area = capacity.iter().map(|&c| c.max(0.0) / DEFAULT_BOND_STRENGTH).collect();
        let mut g = StressGraph {
            n: n_nodes,
            offsets,
            adj_node,
            adj_edge,
            edge_ab: edges.iter().map(|&(a, b)| [a, b]).collect(),
            capacity: capacity.to_vec(),
            cond: edges
                .iter()
                .zip(capacity)
                .map(|(&(a, b), &c)| {
                    let (pa, pb) = (node_pos[a as usize], node_pos[b as usize]);
                    let l = ((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt();
                    // axial stiffness ∝ E·A/L: capacity (∝ A) over centre-to-centre distance
                    c.max(0.0) / l.max(0.05)
                })
                .collect(),
            centroid_xy,
            edge_z: centroid.iter().map(|c| c[2]).collect(),
            group: Vec::new(),
            iface_of: vec![[u32::MAX; 2]; m],
            iface_off: vec![0],
            iface_edges: Vec::new(),
            iface_n: Vec::new(),
            sec_kind: Vec::new(),
            node_z: node_pos.iter().map(|p| p[2]).collect(),
            col_thick: Vec::new(),
            node_xy,
            horizontal,
            vert_sign,
            nrm_h,
            area,
            bend_coef: Vec::new(),
            origin,
            length_scale,
        };
        g.update_bend_coef();
        let zeros = vec![0u32; n_nodes];
        g.build_sections(&zeros, false);
        g
    }

    /// Provide true contact areas (m²) for the bending section modulus.
    pub fn with_areas(mut self, area: &[f32]) -> Self {
        assert_eq!(area.len(), self.edge_ab.len(), "area len");
        self.area = area.to_vec();
        self.update_bend_coef();
        self
    }

    /// Group nodes (chunk → element/panel id). Joints are then evaluated per *section*:
    /// * horizontal-normal joints (bending): all contacts between two members form an
    ///   **interface**; joints inside a member form planar **cuts** (see crate docs);
    /// * vertical-normal joints (bearing): contacts between two members (e.g. a wall standing
    ///   on a slab, a column under a slab) form one bearing section; joints inside a member
    ///   only get the per-joint axial check.
    ///
    /// Without groups, horizontal joints are their own sections and bearing sections are the
    /// height bands of the whole structure.
    pub fn with_node_groups(mut self, group: &[u32]) -> Self {
        assert_eq!(group.len(), self.n, "group len");
        if group.iter().all(|&x| x == group.first().copied().unwrap_or(0)) {
            return self; // a single member carries no structure information
        }
        self.build_sections(group, true);
        self.group = group.to_vec();
        self
    }

    /// Mark column nodes with their section thickness (m; min horizontal size, 0 = not a
    /// column). Column stacks get a buckling (Rankine) reduction of their compression capacity
    /// from the unbraced length between lateral bracing (see crate docs).
    pub fn with_columns(mut self, thickness: &[f32]) -> Self {
        assert_eq!(thickness.len(), self.n, "thickness len");
        self.col_thick = if thickness.iter().any(|&t| t > 0.0) { thickness.to_vec() } else { Vec::new() };
        self
    }

    fn build_sections(&mut self, group: &[u32], horizontal_cuts: bool) {
        let m = self.edge_ab.len();
        // 1. bucket (edge, slot, orientation sign) entries by section key
        let mut buckets: CutBuckets = std::collections::HashMap::new();
        for e in 0..m {
            let [a, b] = self.edge_ab[e];
            let (ga, gb) = (group[a as usize], group[b as usize]);
            if self.vert_sign[e] != 0 {
                // bearing sections
                if ga != gb {
                    buckets.entry((ga.min(gb), ga.max(gb), i64::MIN + 1)).or_default().push((e as u32, 2, 1.0));
                } else if !horizontal_cuts {
                    // ungrouped structure: height bands of the whole structure. (Within a real
                    // member the Voronoi joints of a band are a partial, noisy section and the
                    // member's own load centroid is a mixture; members are checked at their
                    // interfaces instead.)
                    for slot in 0..2u8 {
                        let zb = (self.edge_z[e] as f64 / BEAR_BAND + 0.5 * slot as f64).floor() as i64;
                        buckets.entry((ga, u32::MAX - 100 - slot as u32, zb)).or_default().push((e as u32, slot, 1.0));
                    }
                }
                continue;
            }
            if !horizontal_cuts || !self.horizontal[e] {
                continue;
            }
            let n = self.nrm_h[e];
            if ga != gb {
                // inter-member interface: all contacts between the two members
                let sgn = if ga < gb { 1.0 } else { -1.0 };
                buckets.entry((ga.min(gb), ga.max(gb), i64::MIN)).or_default().push((e as u32, 2, sgn));
            } else {
                // intra-member cut: joints of one member with similar horizontal normal
                // direction (CUT_BINS bins over 180°) within a CUT_BIN-wide slab across that
                // direction. Two staggered slabbings; an edge takes the stronger of its two cuts
                // (no artifacts at slab boundaries; Voronoi joints of one cut aren't coplanar).
                let th = (n[1] as f64).atan2(n[0] as f64).rem_euclid(std::f64::consts::PI);
                let bin = ((th / CUT_ANGLE) as usize).min(CUT_BINS - 1);
                let ang = (bin as f64 + 0.5) * CUT_ANGLE;
                let dir = [ang.cos(), ang.sin()];
                let c = self.centroid_xy[e];
                let pos = c[0] as f64 * dir[0] + c[1] as f64 * dir[1];
                let dot = n[0] as f64 * dir[0] + n[1] as f64 * dir[1];
                let sgn = if dot >= 0.0 { 1.0 } else { -1.0 };
                for slot in 0..2u8 {
                    let pbin = (pos / CUT_BIN + 0.5 * slot as f64).floor() as i64;
                    buckets.entry((ga, u32::MAX - (2 * bin as u32 + slot as u32), pbin)).or_default().push((e as u32, slot, sgn));
                }
            }
        }
        // 2. one section per key
        let mut keys: Vec<(u32, u32, i64)> = buckets.keys().copied().collect();
        keys.sort_unstable();
        let mut iface_of = vec![[u32::MAX; 2]; m];
        let mut off = vec![0u32];
        let mut edges: Vec<u32> = Vec::new();
        let mut iface_n: Vec<[f32; 2]> = Vec::new();
        let mut kind: Vec<u8> = Vec::new();
        for key in keys {
            let list = buckets.remove(&key).unwrap();
            let id = iface_n.len() as u32;
            let mut ns = [0f64; 2];
            let bearing = self.vert_sign[list[0].0 as usize] != 0;
            for &(e, slot, sgn) in &list {
                let eu = e as usize;
                if slot == 2 {
                    iface_of[eu] = [id, id];
                } else {
                    iface_of[eu][slot as usize] = id;
                }
                edges.push(e);
                let (n, w) = (self.nrm_h[eu], self.area[eu].max(1e-12) as f64);
                ns[0] += sgn as f64 * w * n[0] as f64;
                ns[1] += sgn as f64 * w * n[1] as f64;
            }
            let l = (ns[0] * ns[0] + ns[1] * ns[1]).sqrt();
            iface_n.push(if l > 0.0 && !bearing { [(ns[0] / l) as f32, (ns[1] / l) as f32] } else { [0.0, 0.0] });
            kind.push(if bearing { SEC_BEAR } else { SEC_BEND });
            off.push(edges.len() as u32);
        }
        self.iface_n = iface_n;
        self.sec_kind = kind;
        self.iface_of = iface_of;
        self.iface_off = off;
        self.iface_edges = edges;
    }

    /// Number of inter-group bending interfaces.
    pub fn n_interfaces(&self) -> usize {
        self.iface_n.len()
    }

    fn update_bend_coef(&mut self) {
        self.bend_coef = (0..self.edge_ab.len())
            .map(|e| {
                let (c, a) = (self.capacity[e], self.area[e]);
                if self.horizontal[e] && c > 0.0 && a > 0.0 {
                    6.0 / (c * a.sqrt())
                } else {
                    0.0
                }
            })
            .collect();
    }

    pub fn n_nodes(&self) -> usize {
        self.n
    }
    /// Reference point (node bbox centre); internal xy coordinates are relative to it.
    pub fn origin(&self) -> [f32; 3] {
        self.origin
    }
    pub fn n_edges(&self) -> usize {
        self.edge_ab.len()
    }
    pub fn edge_nodes(&self, e: u32) -> (u32, u32) {
        let [a, b] = self.edge_ab[e as usize];
        (a, b)
    }
    pub fn capacity(&self, e: u32) -> f32 {
        self.capacity[e as usize]
    }
    /// True if the edge gets the bending term (|n·z| < 0.5).
    pub fn is_horizontal(&self, e: u32) -> bool {
        self.horizontal[e as usize]
    }
    /// `(neighbor, edge)` pairs of node `i`.
    pub fn neighbors(&self, i: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
        let (s, e) = (self.offsets[i as usize] as usize, self.offsets[i as usize + 1] as usize);
        self.adj_node[s..e].iter().copied().zip(self.adj_edge[s..e].iter().copied())
    }
    #[inline]
    fn row(&self, i: usize) -> std::ops::Range<usize> {
        self.offsets[i] as usize..self.offsets[i + 1] as usize
    }
}

/// Per-tick inputs (all indexed by node / edge id).
#[derive(Default, Clone, Copy)]
pub struct StressInput<'a> {
    /// `m·g` per node (N).
    pub node_weight: &'a [f32],
    pub node_alive: &'a [bool],
    pub anchor: &'a [bool],
    pub edge_alive: &'a [bool],
    /// Remaining strength fraction per node (e.g. hp / hp_max; empty = all 1). An edge's
    /// capacity is scaled by the weaker of its two nodes, so damaged supports fail under load.
    pub node_strength: &'a [f32],
}

#[derive(Clone, Debug)]
pub struct StressConfig {
    /// Max PCG iterations (SpMVs) per `solve_step` call.
    pub max_iters: usize,
    /// Convergence: every node's force imbalance ≤ tol · RMS(node weight).
    pub tol: f32,
    /// Seconds an edge must stay at u > 1 before it is reported in `to_break`.
    pub hold_time: f32,
    pub max_breaks_per_tick: usize,
    /// Add the cantilever bending term on horizontal-normal edges.
    pub bending: bool,
    /// Multiplier on the bending utilization (tuning knob; 1 = model as documented).
    pub bend_scale: f32,
    /// Flexural capacity of a joint relative to an unreinforced section of the same bond
    /// strength (`M_cap = flexural_factor · bond · A^1.5/6`). Structural members (slabs,
    /// beams) are reinforced: rebar gives RC sections roughly an order of magnitude more
    /// bending capacity than plain-concrete cracking. Default 5 (calibrated: intact synthetic
    /// towers stay < 0.5 while losing 3 of 5 ground-floor columns collapses a tower).
    pub flexural_factor: f32,
    /// Capacity multiplier when an edge with near-vertical normal carries load downward
    /// (bearing/compression), also used on the compression side of bearing sections.
    /// `capacity` (= area × bond strength) is the tensile/shear value; with bgen's concrete
    /// bond (8 MPa) the default 3.5 gives ~28 MPa crushing strength.
    pub compression_factor: f32,
    /// Tension capacity multiplier on the far side of an eccentrically loaded bearing section:
    /// the same reinforcement that gives members their flexural capacity carries tension
    /// across horizontal joints (a moment taken by a beam root continues down the column).
    /// Default 5 (= `flexural_factor`). A load centroid outside the support still tips: the
    /// required tension grows with the eccentricity while the section stays small.
    pub tension_factor: f32,
    /// Evaluate bearing sections (vertical-normal joints: eccentric load, tipping). Needs
    /// `bending` (uses the carried-load centroids).
    pub bearing_sections: bool,
    /// Rankine buckling of column stacks (see [`StressGraph::with_columns`]).
    pub buckling: bool,
    /// Rankine constant k in `φ = 1 / (1 + (L/r)²/k)`. 1200: a 0.5 m column braced every
    /// 3.5 m keeps φ ≈ 0.67; losing one bracing level (L doubled) gives ≈ 0.33.
    pub rankine_k: f32,
    /// Catch-up: when one tick removes support carrying more than this fraction of the total
    /// load (sum of flows through removed joints), solve globally to convergence right away
    /// (time-capped by `catchup_max_ms`). 0 disables.
    pub catchup_frac: f32,
    pub catchup_max_ms: f32,
    /// Max nodes in a local (warm) solve region; graphs with ≤ this many supported nodes
    /// are always solved globally.
    pub region_max_nodes: usize,
    /// While residual remains after a local solve, the window grows ×1.5 per tick up to
    /// `region_max_nodes · max_region_growth` (switching to a global solve if that covers
    /// every supported node). If residual still remains after 8 more ticks at that size, the
    /// solver escalates to global steps (rayon-parallel above `par_threshold`) until converged.
    pub max_region_growth: usize,
    /// In local mode, residual left on the window's boundary (or in an unfinished window)
    /// is only chased if it exceeds `local_tol_factor · tol · w_rms`; smaller leftovers are
    /// accepted. Keeps windowed solves from chasing negligible far-field noise forever.
    pub local_tol_factor: f32,
    /// Use rayon for SpMV/vector kernels when the solved system has ≥ this many rows.
    pub par_threshold: usize,
}

impl Default for StressConfig {
    fn default() -> Self {
        StressConfig {
            max_iters: 30,
            tol: 1e-3,
            hold_time: 0.25,
            max_breaks_per_tick: 8,
            bending: true,
            bend_scale: 1.0,
            flexural_factor: 5.0,
            compression_factor: 3.5,
            tension_factor: 5.0,
            bearing_sections: true,
            buckling: true,
            rankine_k: 1200.0,
            catchup_frac: 0.05,
            catchup_max_ms: 20.0,
            region_max_nodes: 16384,
            max_region_growth: 4,
            local_tol_factor: 10.0,
            par_threshold: 32768,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StressResult {
    /// Per-edge utilization (0 for dead / unsupported edges).
    pub utilization: Vec<f32>,
    /// Edges that have been overloaded for ≥ hold_time, worst first, ≤ max_breaks_per_tick.
    pub to_break: Vec<u32>,
    /// True when no residual work is pending (solution within tolerance everywhere).
    pub converged: bool,
    /// PCG iterations spent in this call.
    pub iters: usize,
}

#[derive(Clone, Debug)]
pub struct StaticReport {
    pub max_util: f32,
    /// Up to 20 worst edges `(edge, utilization)`, descending.
    pub worst_edges: Vec<(u32, f32)>,
    pub converged: bool,
    pub iters: usize,
    /// Alive, non-anchor nodes not connected (through conducting edges) to any anchor.
    pub unsupported_nodes: Vec<u32>,
    /// Sum of weights of supported (solved) nodes.
    pub total_load: f64,
    /// Sum of flow from supported nodes into anchors (== total_load at convergence).
    pub anchor_flow: f64,
    pub utilization: Vec<f32>,
}

#[derive(PartialEq, Clone, Debug)]
struct HeapItem(f64, u32);
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    // min-heap on potential, ties by id
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// Mutable solver state: warm-start potentials, classification, hysteresis timers.
#[derive(Clone, Debug)]
pub struct StressState {
    init: bool,
    /// Carried-load centroids are valid (bending was on last tick).
    moments_valid: bool,
    class: Vec<u8>,
    /// Load potential φ per node (0 for non-UNKNOWN nodes).
    pot: Vec<f64>,
    /// Carried load F_i = w_i + Σ inflow (N) and its centroid C_i (xy rel. origin).
    load_f: Vec<f32>,
    load_c: Vec<[f32; 2]>,
    /// Nodes whose carried load changed this tick (util refresh list).
    moved: Vec<u32>,
    /// Interfaces needing their bending term recomputed this tick.
    iface_dirty: Vec<u32>,
    iface_mark: Vec<bool>,
    /// Latest bending utilization per cut/interface.
    cut_bend: Vec<f32>,
    /// Latest bearing-section utilization per edge and slot.
    sec_u: Vec<[f32; 2]>,
    /// Buckling factor per node (1 = none) and the topology version it was computed for.
    phi: Vec<f32>,
    phi_version: u64,
    /// Strength fraction per node (from `StressInput::node_strength`).
    strength: Vec<f32>,
    /// Nodes whose capacities changed without a re-solve (damage, buckling): util refresh.
    cap_moved: Vec<u32>,
    /// Sum of |flow| through joints removed since the last solve (catch-up trigger).
    removed_flow: f64,
    total_load: f64,
    edge_k: Vec<f32>,
    prev_alive: Vec<bool>,
    prev_anchor: Vec<bool>,
    prev_weight: Vec<u32>,
    prev_edge_alive: Vec<bool>,
    pending: Vec<u32>,
    pend_mark: Vec<bool>,
    need_global: bool,
    /// Current local-region cap; grows ×1.5 per unconverged local tick, resets on convergence.
    local_cap: usize,
    /// Consecutive unconverged local ticks at the window-size limit.
    stall: usize,
    visit: Vec<u32>,
    search_id: u32,
    util: Vec<f32>,
    timer: Vec<f32>,
    hot: Vec<u32>,
    is_hot: Vec<bool>,
    w_rms: f64,
    n_unknown: usize,
    stats_dirty: bool,
    version: u64,
    sys: cg::LocalSys,
    sys_version: u64,
    loc: Vec<u32>,
    work: cg::Work,
    ring: Vec<u32>,
    scratch: Vec<u32>,
    heap: BinaryHeap<HeapItem>,
}

impl StressState {
    pub fn new(g: &StressGraph) -> Self {
        let (n, m) = (g.n, g.edge_ab.len());
        StressState {
            init: false,
            moments_valid: false,
            class: vec![DEAD; n],
            pot: vec![0.0; n],
            load_f: vec![0.0; n],
            load_c: g.node_xy.clone(),
            moved: Vec::new(),
            iface_dirty: Vec::new(),
            iface_mark: vec![false; g.iface_n.len()],
            cut_bend: vec![0.0; g.iface_n.len()],
            sec_u: vec![[0.0; 2]; m],
            phi: vec![1.0; n],
            phi_version: 0,
            strength: vec![1.0; n],
            cap_moved: Vec::new(),
            removed_flow: 0.0,
            total_load: 0.0,
            edge_k: vec![0.0; m],
            prev_alive: vec![false; n],
            prev_anchor: vec![false; n],
            prev_weight: vec![0; n],
            prev_edge_alive: vec![false; m],
            pending: Vec::new(),
            pend_mark: vec![false; n],
            need_global: true,
            local_cap: 0,
            stall: 0,
            visit: vec![0; n],
            search_id: 0,
            util: vec![0.0; m],
            timer: vec![0.0; m],
            hot: Vec::new(),
            is_hot: vec![false; m],
            w_rms: 0.0,
            n_unknown: 0,
            stats_dirty: true,
            version: 1,
            sys: cg::LocalSys::default(),
            sys_version: 0,
            loc: vec![u32::MAX; n],
            work: cg::Work::default(),
            ring: Vec::new(),
            scratch: Vec::new(),
            heap: BinaryHeap::new(),
        }
    }

    /// Current utilization (same as the last `StressResult::utilization`), without a copy.
    pub fn utilization(&self) -> &[f32] {
        &self.util
    }
    /// True if any edge has utilization > 1 (same as `utilization().iter().any(|&u| u > 1.0)`
    /// after a solve step: the hysteresis pass leaves exactly those edges on the hot list).
    pub fn any_overloaded(&self) -> bool {
        debug_assert_eq!(!self.hot.is_empty(), self.util.iter().any(|&u| u > 1.0));
        !self.hot.is_empty()
    }
    /// Load potential φ of node i (0 for anchors / dead / unsupported).
    pub fn potential(&self, i: u32) -> f64 {
        self.pot[i as usize]
    }
    /// Signed load flow through edge e (positive = from a to b), in Newtons.
    pub fn edge_flow(&self, g: &StressGraph, e: u32) -> f64 {
        let k = self.edge_k[e as usize] as f64;
        let [a, b] = g.edge_ab[e as usize];
        k * (self.pot[a as usize] - self.pot[b as usize])
    }
    /// Bending moment (N·m) estimated at a horizontal-normal edge: `|f| · |(C_up − c_e)·n̂|`
    /// (0 for other edges; only meaningful when solved with bending on).
    pub fn edge_moment(&self, g: &StressGraph, e: u32) -> f64 {
        self.moment(g, e as usize, self.edge_flow(g, e))
    }
    /// Carried load through node i (its weight plus all inflow), and that load's centroid
    /// (world xy). Only maintained with bending on.
    pub fn carried_load(&self, g: &StressGraph, i: u32) -> (f32, [f32; 2]) {
        let c = self.load_c[i as usize];
        (self.load_f[i as usize], [c[0] + g.origin[0], c[1] + g.origin[1]])
    }
    #[inline]
    fn moment(&self, g: &StressGraph, e: usize, f: f64) -> f64 {
        if !g.horizontal[e] || f == 0.0 {
            return 0.0;
        }
        let [a, b] = g.edge_ab[e];
        let up = if f > 0.0 { a } else { b } as usize;
        let (cu, ce, n) = (self.load_c[up], g.centroid_xy[e], g.nrm_h[e]);
        let d = (cu[0] - ce[0]) * n[0] + (cu[1] - ce[1]) * n[1];
        f.abs() * d.abs() as f64
    }
    #[doc(hidden)]
    /// (edges in cut, alive edges, alive area, z_lo, z_hi, |moment|, Σcap) of the cut holding edge e.
    pub fn debug_cut(&self, g: &StressGraph, e: u32, slot: usize) -> Option<(usize, usize, f64, f32, f32, f64, f64)> {
        let id = g.iface_of[e as usize][slot];
        if id == u32::MAX {
            return None;
        }
        let edges = &g.iface_edges[g.iface_off[id as usize] as usize..g.iface_off[id as usize + 1] as usize];
        let (mut cap, mut area, mut cx, mut cy, mut alive) = (0f64, 0f64, 0f64, 0f64, 0usize);
        let (mut zlo, mut zhi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &e in edges {
            let e = e as usize;
            if self.edge_k[e] == 0.0 {
                continue;
            }
            alive += 1;
            let a = g.area[e].max(1e-9) as f64;
            cap += g.capacity[e] as f64;
            area += a;
            cx += a * g.centroid_xy[e][0] as f64;
            cy += a * g.centroid_xy[e][1] as f64;
            let half = 0.5 * g.area[e].max(0.0).sqrt();
            zlo = zlo.min(g.edge_z[e] - half);
            zhi = zhi.max(g.edge_z[e] + half);
        }
        let (cx, cy) = (cx / area, cy / area);
        let n = g.iface_n[id as usize];
        let mut m = 0f64;
        for &e in edges {
            let e = e as usize;
            let k = self.edge_k[e];
            if k == 0.0 {
                continue;
            }
            let [a, b] = g.edge_ab[e];
            let f = k as f64 * (self.pot[a as usize] - self.pot[b as usize]);
            if f == 0.0 {
                continue;
            }
            let upn = if f > 0.0 { a } else { b } as usize;
            let c = self.load_c[upn];
            m += f.abs() * ((c[0] as f64 - cx) * n[0] as f64 + (c[1] as f64 - cy) * n[1] as f64);
        }
        Some((edges.len(), alive, area, zlo, zhi, m.abs(), cap))
    }
    #[doc(hidden)]
    pub fn debug_pending(&self) -> (usize, bool) {
        (self.pending.len(), self.need_global)
    }
    /// Node is alive, non-anchor and connected to an anchor (i.e. solved for).
    pub fn is_supported(&self, i: u32) -> bool {
        self.class[i as usize] == UNKNOWN
    }

    #[inline]
    fn pend(&mut self, i: u32) {
        if !self.pend_mark[i as usize] {
            self.pend_mark[i as usize] = true;
            self.pending.push(i);
        }
    }

    fn zero_edges_of(&mut self, g: &StressGraph, i: usize) {
        for p in g.row(i) {
            let e = g.adj_edge[p] as usize;
            self.util[e] = 0.0;
            self.timer[e] = 0.0;
        }
    }

    fn set_class(&mut self, g: &StressGraph, i: usize, c: u8) {
        let old = self.class[i];
        if old == c {
            return;
        }
        self.class[i] = c;
        self.version += 1;
        self.stats_dirty = true;
        if c == UNKNOWN {
            self.pend(i as u32);
        } else {
            self.pot[i] = 0.0;
            if old == UNKNOWN {
                self.zero_edges_of(g, i);
            }
        }
    }

    fn calc_k(g: &StressGraph, input: &StressInput, e: usize) -> f32 {
        let [a, b] = g.edge_ab[e];
        let c = g.cond[e];
        if input.edge_alive[e] && input.node_alive[a as usize] && input.node_alive[b as usize] && c > 0.0 {
            c
        } else {
            0.0
        }
    }

    /// Returns `Some(added)` if k changed.
    fn update_edge_k(&mut self, g: &StressGraph, input: &StressInput, e: usize) -> Option<bool> {
        let k = Self::calc_k(g, input, e);
        let old = self.edge_k[e];
        if k == old {
            return None;
        }
        if old > 0.0 {
            let [a, b] = g.edge_ab[e];
            self.removed_flow += (old as f64 * (self.pot[a as usize] - self.pot[b as usize])).abs();
        }
        self.edge_k[e] = k;
        self.version += 1;
        let [a, b] = g.edge_ab[e];
        self.pend(a);
        self.pend(b);
        if k == 0.0 {
            self.util[e] = 0.0;
            self.timer[e] = 0.0;
        }
        Some(old == 0.0)
    }

    /// O(N+E) classification: BFS from all alive anchors over conducting edges.
    fn full_classify(&mut self, g: &StressGraph, input: &StressInput) {
        let mut q = std::mem::take(&mut self.scratch);
        q.clear();
        let mut target = vec![DEAD; g.n];
        for i in 0..g.n {
            target[i] = if !input.node_alive[i] {
                DEAD
            } else if input.anchor[i] {
                q.push(i as u32);
                ANCHOR
            } else {
                FLOATING
            };
        }
        let mut head = 0;
        while head < q.len() {
            let i = q[head] as usize;
            head += 1;
            for p in g.row(i) {
                let j = g.adj_node[p] as usize;
                if target[j] == FLOATING && self.edge_k[g.adj_edge[p] as usize] > 0.0 {
                    target[j] = UNKNOWN;
                    q.push(j as u32);
                }
            }
        }
        for (i, &t) in target.iter().enumerate() {
            self.set_class(g, i, t);
        }
        self.scratch = q;
    }

    fn next_search_id(&mut self) -> u32 {
        if self.search_id == u32::MAX {
            self.visit.iter_mut().for_each(|v| *v = 0);
            self.search_id = 0;
        }
        self.search_id += 1;
        self.search_id
    }

    /// Best-first search (by φ) from `seed` toward any anchor. If none is reachable, the
    /// whole component is marked FLOATING. `tick_base`: first search id of this batch;
    /// nodes visited by an earlier (anchored) search in the batch count as anchored.
    fn check_support(&mut self, g: &StressGraph, seed: usize, tick_base: u32) {
        if self.class[seed] != UNKNOWN || self.visit[seed] >= tick_base {
            return;
        }
        let cur = self.next_search_id();
        let mut comp = std::mem::take(&mut self.scratch);
        comp.clear();
        self.heap.clear();
        self.visit[seed] = cur;
        comp.push(seed as u32);
        self.heap.push(HeapItem(self.pot[seed], seed as u32));
        let mut anchored = false;
        'outer: while let Some(HeapItem(_, i)) = self.heap.pop() {
            for p in g.row(i as usize) {
                if self.edge_k[g.adj_edge[p] as usize] == 0.0 {
                    continue;
                }
                let j = g.adj_node[p] as usize;
                match self.class[j] {
                    ANCHOR => {
                        anchored = true;
                        break 'outer;
                    }
                    UNKNOWN => {
                        let v = self.visit[j];
                        if v == cur {
                            continue;
                        }
                        if v >= tick_base {
                            anchored = true; // reached a node proven anchored this batch
                            break 'outer;
                        }
                        self.visit[j] = cur;
                        comp.push(j as u32);
                        self.heap.push(HeapItem(self.pot[j], j as u32));
                    }
                    _ => {}
                }
            }
        }
        if !anchored {
            for idx in 0..comp.len() {
                let i = comp[idx] as usize;
                self.set_class(g, i, FLOATING);
            }
        }
        self.scratch = comp;
    }

    /// Diff inputs against the previous call and update edge conductances/classes/pending.
    fn sync(&mut self, g: &StressGraph, input: &StressInput) {
        let (n, m) = (g.n, g.edge_ab.len());
        assert_eq!(input.node_weight.len(), n, "node_weight len");
        assert_eq!(input.node_alive.len(), n, "node_alive len");
        assert_eq!(input.anchor.len(), n, "anchor len");
        assert_eq!(input.edge_alive.len(), m, "edge_alive len");
        if !self.init {
            self.init = true;
            self.prev_alive.copy_from_slice(input.node_alive);
            self.prev_anchor.copy_from_slice(input.anchor);
            for (d, s) in self.prev_weight.iter_mut().zip(input.node_weight) {
                *d = s.to_bits();
            }
            self.prev_edge_alive.copy_from_slice(input.edge_alive);
            for e in 0..m {
                self.edge_k[e] = Self::calc_k(g, input, e);
            }
            self.full_classify(g, input);
            self.need_global = true;
            self.version += 1;
            self.stats_dirty = true;
            return;
        }
        const B: usize = 256;
        let mut additions = false;
        let mut removal_seeds = Vec::new();
        // ---- nodes
        let mut start = 0;
        while start < n {
            let end = (start + B).min(n);
            let same = self.prev_alive[start..end] == input.node_alive[start..end]
                && self.prev_anchor[start..end] == input.anchor[start..end]
                && self.prev_weight[start..end]
                    .iter()
                    .zip(&input.node_weight[start..end])
                    .all(|(a, b)| *a == b.to_bits());
            if !same {
                for i in start..end {
                    let (alive, anchor) = (input.node_alive[i], input.anchor[i]);
                    let wbits = input.node_weight[i].to_bits();
                    if self.prev_alive[i] != alive {
                        self.prev_alive[i] = alive;
                        if alive {
                            additions = true;
                        }
                        for p in g.row(i) {
                            let e = g.adj_edge[p] as usize;
                            if let Some(added) = self.update_edge_k(g, input, e) {
                                if added {
                                    additions = true;
                                } else {
                                    let [a, b] = g.edge_ab[e];
                                    removal_seeds.push(a);
                                    removal_seeds.push(b);
                                }
                            }
                        }
                        if !alive {
                            self.set_class(g, i, DEAD);
                        }
                    }
                    if self.prev_anchor[i] != anchor {
                        self.prev_anchor[i] = anchor;
                        if alive {
                            if anchor {
                                self.set_class(g, i, ANCHOR);
                                additions = true;
                                for p in g.row(i) {
                                    let j = g.adj_node[p];
                                    self.pend(j);
                                }
                            } else {
                                self.set_class(g, i, UNKNOWN);
                                removal_seeds.push(i as u32);
                            }
                        }
                    }
                    if self.prev_weight[i] != wbits {
                        self.prev_weight[i] = wbits;
                        self.stats_dirty = true;
                        if self.class[i] == UNKNOWN {
                            self.pend(i as u32);
                        }
                    }
                }
            }
            start = end;
        }
        // ---- edges
        let mut start = 0;
        while start < m {
            let end = (start + B).min(m);
            if self.prev_edge_alive[start..end] != input.edge_alive[start..end] {
                for e in start..end {
                    if self.prev_edge_alive[e] != input.edge_alive[e] {
                        self.prev_edge_alive[e] = input.edge_alive[e];
                        if let Some(added) = self.update_edge_k(g, input, e) {
                            if added {
                                additions = true;
                            } else {
                                let [a, b] = g.edge_ab[e];
                                removal_seeds.push(a);
                                removal_seeds.push(b);
                            }
                        }
                    }
                }
            }
            start = end;
        }
        if additions {
            self.full_classify(g, input);
        } else if !removal_seeds.is_empty() {
            let base = self.next_search_id();
            for &s in &removal_seeds {
                self.check_support(g, s as usize, base);
            }
        }
        self.sync_strength(input);
    }

    fn sync_strength(&mut self, input: &StressInput) {
        if input.node_strength.is_empty() {
            return;
        }
        assert_eq!(input.node_strength.len(), self.strength.len(), "node_strength len");
        for (i, (&s, cur)) in input.node_strength.iter().zip(self.strength.iter_mut()).enumerate() {
            let s = s.clamp(0.0, 1.0);
            if s != *cur {
                *cur = s;
                self.cap_moved.push(i as u32);
            }
        }
    }

    /// Rankine buckling factor of column nodes from the unbraced length of their stack.
    fn update_buckling(&mut self, g: &StressGraph, cfg: &StressConfig) {
        if g.col_thick.is_empty() {
            return;
        }
        if !cfg.buckling {
            for i in 0..g.n {
                if self.phi[i] != 1.0 {
                    self.phi[i] = 1.0;
                    self.cap_moved.push(i as u32);
                }
            }
            self.phi_version = 0;
            return;
        }
        if self.phi_version == self.version {
            return;
        }
        self.phi_version = self.version;
        let alive = |c: u8| c == UNKNOWN || c == ANCHOR;
        let tag = self.next_search_id();
        let mut stack: Vec<u32> = Vec::new();
        let mut braces: Vec<f32> = Vec::new();
        for s0 in 0..g.n {
            if g.col_thick[s0] <= 0.0 || !alive(self.class[s0]) || self.visit[s0] == tag {
                continue;
            }
            // collect the stack: column nodes joined by conducting vertical joints
            stack.clear();
            braces.clear();
            stack.push(s0 as u32);
            self.visit[s0] = tag;
            let mut head = 0;
            while head < stack.len() {
                let i = stack[head] as usize;
                head += 1;
                if self.class[i] == ANCHOR {
                    braces.push(g.node_z[i]);
                }
                for p in g.row(i) {
                    let e = g.adj_edge[p] as usize;
                    if self.edge_k[e] == 0.0 {
                        continue;
                    }
                    let j = g.adj_node[p] as usize;
                    if !alive(self.class[j]) {
                        continue;
                    }
                    if g.col_thick[j] > 0.0 && g.vert_sign[e] != 0 {
                        if self.visit[j] != tag {
                            self.visit[j] = tag;
                            stack.push(j as u32);
                        }
                    } else if g.col_thick[j] <= 0.0 {
                        braces.push(g.edge_z[e]); // lateral support from another member
                    }
                }
            }
            braces.sort_by(f32::total_cmp);
            let ztop = stack.iter().map(|&i| g.node_z[i as usize]).fold(f32::NEG_INFINITY, f32::max);
            for &i in &stack {
                let i = i as usize;
                let z = g.node_z[i];
                let below = braces.iter().rev().find(|&&b| b <= z + 1e-4).copied();
                let above = braces.iter().find(|&&b| b >= z - 1e-4).copied();
                let phi = match (below, above) {
                    (Some(lo), Some(hi)) if hi > lo => Self::rankine(hi - lo, g.col_thick[i], cfg.rankine_k),
                    (Some(lo), None) => Self::rankine(2.0 * (ztop - lo).max(0.0), g.col_thick[i], cfg.rankine_k),
                    _ => 1.0, // braced at the node itself, or hanging (no compression)
                };
                if (phi - self.phi[i]).abs() > 1e-4 {
                    self.phi[i] = phi;
                    self.cap_moved.push(i as u32);
                }
            }
        }
    }

    fn rankine(l: f32, thick: f32, k: f32) -> f32 {
        let r = thick.max(1e-3) / 12f32.sqrt();
        let s = l / r;
        1.0 / (1.0 + s * s / k.max(1e-3))
    }

    /// Buckling factor of node i (1 = not a column / no reduction).
    pub fn buckling_factor(&self, i: u32) -> f32 {
        self.phi[i as usize]
    }

    fn update_stats(&mut self, input: &StressInput) {
        if !self.stats_dirty {
            return;
        }
        self.stats_dirty = false;
        let (mut s2, mut cnt) = (0f64, 0usize);
        for (i, &c) in self.class.iter().enumerate() {
            if c == UNKNOWN {
                let w = input.node_weight[i] as f64;
                s2 += w * w;
                cnt += 1;
            }
        }
        self.n_unknown = cnt;
        self.w_rms = if cnt > 0 { (s2 / cnt as f64).sqrt() } else { 0.0 };
        self.total_load = (0..self.class.len()).filter(|&i| self.class[i] == UNKNOWN).map(|i| input.node_weight[i] as f64).sum();
    }

    #[inline]
    fn compute_util(&self, g: &StressGraph, e: usize, up: &UtilParams) -> f32 {
        let k = self.edge_k[e];
        if k == 0.0 {
            return 0.0;
        }
        let [a, b] = g.edge_ab[e];
        let f = k as f64 * (self.pot[a as usize] - self.pot[b as usize]);
        let s = self.edge_strength(g, e);
        // load moving down across a horizontal-ish contact = bearing (compression)
        let vs = g.vert_sign[e] as f64;
        let cap = if vs * f < 0.0 {
            g.capacity[e] * s * up.compression * self.edge_phi(g, e)
        } else {
            g.capacity[e] * s
        };
        let mut u = f.abs() / cap as f64;
        if up.bending && g.bend_coef[e] > 0.0 && g.iface_of[e][0] == u32::MAX {
            u += (up.bend_k * g.bend_coef[e] / s) as f64 * self.moment(g, e, f);
        }
        u as f32
    }

    /// Recompute carried load F_i and its centroid C_i from current inflows (net flow from
    /// higher-potential supported neighbours). Returns true if it changed noticeably.
    #[inline]
    fn recompute_load(&mut self, g: &StressGraph, input: &StressInput, i: usize, eps_c: f32) -> bool {
        let w = input.node_weight[i].max(0.0) as f64;
        let xy = g.node_xy[i];
        let (mut f_tot, mut cx, mut cy) = (w, w * xy[0] as f64, w * xy[1] as f64);
        let pi = self.pot[i];
        let grouped = !g.group.is_empty();
        for p in g.row(i) {
            let e = g.adj_edge[p] as usize;
            let k = self.edge_k[e];
            let j = g.adj_node[p] as usize;
            if k == 0.0 || self.class[j] != UNKNOWN {
                continue;
            }
            let d = self.pot[j] - pi;
            if d > 0.0 {
                let f = k as f64 * d;
                f_tot += f;
                // Load handed over from another member acts on this member at the contact.
                let c = if grouped && g.group[j] != g.group[i] { g.centroid_xy[e] } else { self.load_c[j] };
                cx += f * c[0] as f64;
                cy += f * c[1] as f64;
            }
        }
        let c = if f_tot > 0.0 { [(cx / f_tot) as f32, (cy / f_tot) as f32] } else { xy };
        let f_new = f_tot as f32;
        let (oc, of) = (self.load_c[i], self.load_f[i]);
        self.load_c[i] = c;
        self.load_f[i] = f_new;
        (c[0] - oc[0]).abs() > eps_c || (c[1] - oc[1]).abs() > eps_c || (f_new - of).abs() > 1e-5 * f_new.max(of)
    }

    /// Propagate carried loads down the (acyclic) net-flow graph, upstream first. `seeds` had
    /// their potentials changed; propagation continues downstream only while a node's
    /// carried load/centroid changes by more than a small epsilon. Fills `self.moved`.
    fn update_moments(&mut self, g: &StressGraph, input: &StressInput, seeds: &[u32], full: bool) {
        let eps_c = 1e-4 * g.length_scale;
        self.moved.clear();
        if full {
            // descending potential, ties by id (a total order: any sort gives this sequence).
            // Keyed on the `total_cmp` bit pattern so the sort compares plain integers.
            let mut keyed: Vec<(u64, u32)> = (0..g.n as u32)
                .filter(|&i| self.class[i as usize] == UNKNOWN)
                .map(|i| (desc_total_key(self.pot[i as usize]), i))
                .collect();
            keyed.sort_unstable();
            let order: Vec<u32> = keyed.into_iter().map(|(_, i)| i).collect();
            for &i in &order {
                self.recompute_load(g, input, i as usize, eps_c);
            }
            self.moved = order;
            return;
        }
        let seed_tag = self.next_search_id();
        let tag = self.next_search_id();
        let mut heap = std::mem::take(&mut self.heap);
        heap.clear();
        for &s in seeds {
            let si = s as usize;
            if self.class[si] == UNKNOWN && self.visit[si] < seed_tag {
                self.visit[si] = seed_tag;
                heap.push(HeapItem(-self.pot[si], s)); // min-heap on −φ = upstream first
            }
        }
        while let Some(HeapItem(_, i)) = heap.pop() {
            let i = i as usize;
            let changed = self.recompute_load(g, input, i, eps_c);
            if !(changed || self.visit[i] == seed_tag) {
                continue;
            }
            self.moved.push(i as u32);
            let pi = self.pot[i];
            for p in g.row(i) {
                let j = g.adj_node[p] as usize;
                if self.edge_k[g.adj_edge[p] as usize] != 0.0
                    && self.class[j] == UNKNOWN
                    && self.pot[j] < pi
                    && self.visit[j] < seed_tag
                {
                    self.visit[j] = tag;
                    heap.push(HeapItem(-self.pot[j], j as u32));
                }
            }
        }
        self.heap = heap;
    }

    /// Capacity fraction of edge e from damage (weaker endpoint).
    #[inline]
    fn edge_strength(&self, g: &StressGraph, e: usize) -> f32 {
        let [a, b] = g.edge_ab[e];
        self.strength[a as usize].min(self.strength[b as usize]).max(0.02)
    }
    #[inline]
    fn edge_phi(&self, g: &StressGraph, e: usize) -> f32 {
        let [a, b] = g.edge_ab[e];
        self.phi[a as usize].min(self.phi[b as usize])
    }

    fn refresh_util_of(&mut self, g: &StressGraph, i: usize, up: &UtilParams) {
        for p in g.row(i) {
            let e = g.adj_edge[p] as usize;
            let ids = g.iface_of[e];
            let routed = ids[0] != u32::MAX
                && up.bending
                && (g.sec_kind[ids[0] as usize] == SEC_BEND || up.bearing);
            if routed {
                if self.iface_mark.len() != g.iface_n.len() {
                    self.iface_mark = vec![false; g.iface_n.len()];
                    self.cut_bend = vec![0.0; g.iface_n.len()];
                }
                for id in ids {
                    if !self.iface_mark[id as usize] {
                        self.iface_mark[id as usize] = true;
                        self.iface_dirty.push(id);
                    }
                }
                continue; // set together with its cuts in flush_interfaces
            }
            let u = self.compute_util(g, e, up);
            self.set_util(e, u);
        }
    }

    #[inline]
    fn set_util(&mut self, e: usize, u: f32) {
        self.util[e] = u;
        if u > 1.0 && !self.is_hot[e] {
            self.is_hot[e] = true;
            self.hot.push(e as u32);
        }
    }

    /// Bending of each dirty interface as one cut section: the moment about the interface
    /// centroid axis `M_I = |Σ_e |f_e| · (C_up,e − c_I)·n̂_I|` (opposite rotations cancel),
    /// section `S_I = A_I · h_I / 6` with `A_I` the alive contact area and `h_I` its vertical
    /// extent (gravity moments bend about a horizontal axis, so depth = height), capacity
    /// `flexural_factor · bond · S_I` with `bond = Σcap / A_I`
    /// ⇒ `u_bend = bend_k · 6 M_I / (Σcap · h_I)`. Every interface edge gets axial_e + u_bend.
    fn flush_interfaces(&mut self, g: &StressGraph, up: &UtilParams) {
        let dirty = std::mem::take(&mut self.iface_dirty);
        for &id in &dirty {
            if g.sec_kind[id as usize] == SEC_BEND {
                self.cut_bend[id as usize] = self.cut_bending(g, id as usize, up.bend_k);
            } else {
                self.bearing_section(g, id as usize, up);
            }
        }
        for &id in &dirty {
            self.iface_mark[id as usize] = false;
            let bend = g.sec_kind[id as usize] == SEC_BEND;
            let (s0, s1) = (g.iface_off[id as usize] as usize, g.iface_off[id as usize + 1] as usize);
            for &e in &g.iface_edges[s0..s1] {
                let e = e as usize;
                let mut u = self.compute_util(g, e, up);
                if self.edge_k[e] != 0.0 {
                    let [c0, c1] = g.iface_of[e];
                    if bend {
                        u += self.cut_bend[c0 as usize].min(self.cut_bend[c1 as usize]);
                    } else {
                        u = u.max(self.sec_u[e][0].min(self.sec_u[e][1]));
                    }
                }
                self.set_util(e, u);
            }
        }
        self.iface_dirty = dirty;
        self.iface_dirty.clear();
    }

    /// Bearing section (vertical-normal joints of one member pair / height band): axial load
    /// `N` of the load passing down through it and its moment `m = Σ f·(C_up − c)` about the
    /// section centroid give a linear stress field `σ(p) = N/A + (p − c)·J⁻¹m` (J: second
    /// moments of the contact areas, each contact also counting its own `A²/12`). Each joint
    /// is checked at the extreme fibres of its own patch (±√A/2 along the stress gradient):
    /// compression against `bond·compression_factor·φ`, tension against `bond·tension_factor`
    /// (`bond = capacity/area`, damage-scaled). A load centroid outside the support makes the
    /// far side go into tension and break: the structure tips.
    fn bearing_section(&mut self, g: &StressGraph, id: usize, up: &UtilParams) {
        let edges = &g.iface_edges[g.iface_off[id] as usize..g.iface_off[id + 1] as usize];
        let (mut area, mut cx, mut cy) = (0f64, 0f64, 0f64);
        for &e in edges {
            let e = e as usize;
            if self.edge_k[e] != 0.0 {
                let a = g.area[e].max(1e-9) as f64;
                area += a;
                cx += a * g.centroid_xy[e][0] as f64;
                cy += a * g.centroid_xy[e][1] as f64;
            }
        }
        let slot_of = |e: usize| if g.iface_of[e][0] as usize == id { 0 } else { 1 };
        if area <= 0.0 {
            for &e in edges {
                let e = e as usize;
                self.sec_u[e][slot_of(e)] = 0.0;
            }
            return;
        }
        let (cx, cy) = (cx / area, cy / area);
        let (mut jxx, mut jyy, mut jxy) = (0f64, 0f64, 0f64);
        let (mut nload, mut mx, mut my) = (0f64, 0f64, 0f64);
        for &e in edges {
            let e = e as usize;
            let k = self.edge_k[e];
            if k == 0.0 {
                continue;
            }
            let a = g.area[e].max(1e-9) as f64;
            let (dx, dy) = (g.centroid_xy[e][0] as f64 - cx, g.centroid_xy[e][1] as f64 - cy);
            jxx += a * dx * dx + a * a / 12.0;
            jyy += a * dy * dy + a * a / 12.0;
            jxy += a * dx * dy;
            let [na, nb] = g.edge_ab[e];
            let f = k as f64 * (self.pot[na as usize] - self.pot[nb as usize]);
            let down = -f * g.vert_sign[e] as f64; // > 0: load moves down across the joint
            if down > 0.0 {
                let upn = if f > 0.0 { na } else { nb } as usize;
                let c = self.load_c[upn];
                nload += down;
                mx += down * (c[0] as f64 - cx);
                my += down * (c[1] as f64 - cy);
            }
        }
        let det = jxx * jyy - jxy * jxy;
        let (gx, gy) = if nload > 0.0 && det > 1e-18 {
            ((jyy * mx - jxy * my) / det, (jxx * my - jxy * mx) / det)
        } else {
            (0.0, 0.0)
        };
        let gn = (gx * gx + gy * gy).sqrt();
        let s0 = nload / area;
        for &e in edges {
            let e = e as usize;
            let slot = slot_of(e);
            if self.edge_k[e] == 0.0 || nload <= 0.0 {
                self.sec_u[e][slot] = 0.0;
                continue;
            }
            let a = g.area[e].max(1e-9) as f64;
            let (dx, dy) = (g.centroid_xy[e][0] as f64 - cx, g.centroid_xy[e][1] as f64 - cy);
            let sc = s0 + dx * gx + dy * gy;
            let half = 0.5 * a.sqrt() * gn;
            let bond = (g.capacity[e] * self.edge_strength(g, e)) as f64 / a;
            let uc = (sc + half).max(0.0) / (bond * (up.compression * self.edge_phi(g, e)) as f64);
            let ut = (half - sc).max(0.0) / (bond * up.tension as f64);
            self.sec_u[e][slot] = uc.max(ut) as f32;
        }
    }

    fn cut_bending(&self, g: &StressGraph, id: usize, bend_k: f32) -> f32 {
        let edges = &g.iface_edges[g.iface_off[id] as usize..g.iface_off[id + 1] as usize];
        let (mut cap, mut area, mut cx, mut cy) = (0f64, 0f64, 0f64, 0f64);
        let (mut zlo, mut zhi) = (f32::INFINITY, f32::NEG_INFINITY);
        for &e in edges {
            let e = e as usize;
            if self.edge_k[e] == 0.0 {
                continue;
            }
            let a = g.area[e].max(1e-9) as f64;
            cap += (g.capacity[e] * self.edge_strength(g, e)) as f64;
            area += a;
            cx += a * g.centroid_xy[e][0] as f64;
            cy += a * g.centroid_xy[e][1] as f64;
            let half = 0.5 * g.area[e].max(0.0).sqrt();
            zlo = zlo.min(g.edge_z[e] - half);
            zhi = zhi.max(g.edge_z[e] + half);
        }
        if cap <= 0.0 || zhi <= zlo {
            return 0.0;
        }
        let (cx, cy) = (cx / area, cy / area);
        let n = g.iface_n[id];
        let mut m = 0f64;
        for &e in edges {
            let e = e as usize;
            let k = self.edge_k[e];
            if k == 0.0 {
                continue;
            }
            let [a, b] = g.edge_ab[e];
            let f = k as f64 * (self.pot[a as usize] - self.pot[b as usize]);
            if f == 0.0 {
                continue;
            }
            let upn = if f > 0.0 { a } else { b } as usize;
            let c = self.load_c[upn];
            m += f.abs() * ((c[0] as f64 - cx) * n[0] as f64 + (c[1] as f64 - cy) * n[1] as f64);
        }
        (bend_k as f64 * 6.0 * m.abs() / (cap * (zhi - zlo) as f64)) as f32
    }


    /// Full nodal residual `w_i − (L φ)_i` (f64).
    #[inline]
    fn residual(&self, g: &StressGraph, input: &StressInput, i: usize) -> f64 {
        let pi = self.pot[i];
        let mut r = input.node_weight[i] as f64;
        for p in g.row(i) {
            let k = self.edge_k[g.adj_edge[p] as usize] as f64;
            if k != 0.0 {
                r -= k * (pi - self.pot[g.adj_node[p] as usize]);
            }
        }
        r
    }

    /// Build the compact system for `self.sys.nodes` (with `loc` set) and the residual RHS.
    /// Collects the boundary ring (UNKNOWN neighbours outside the region) when `collect_ring`.
    fn build_sys(&mut self, g: &StressGraph, input: &StressInput, collect_ring: bool) {
        let nodes = std::mem::take(&mut self.sys.nodes);
        self.sys.clear();
        self.sys.row_ptr.push(0);
        self.work.b.clear();
        self.ring.clear();
        let ring_tag = if collect_ring { self.next_search_id() } else { 0 };
        for &gi in &nodes {
            let i = gi as usize;
            let mut d = 0f32;
            for p in g.row(i) {
                let k = self.edge_k[g.adj_edge[p] as usize];
                if k == 0.0 {
                    continue;
                }
                d += k;
                let j = g.adj_node[p] as usize;
                let lj = self.loc[j];
                if lj != u32::MAX {
                    self.sys.col.push(lj);
                    self.sys.val.push(k);
                } else if collect_ring && self.class[j] == UNKNOWN && self.visit[j] != ring_tag {
                    self.visit[j] = ring_tag;
                    self.ring.push(j as u32);
                }
            }
            self.sys.row_ptr.push(self.sys.col.len() as u32);
            self.sys.diag.push(d);
            self.sys.inv_diag.push(if d > 0.0 { 1.0 / d } else { 0.0 });
        }
        self.sys.nodes = nodes;
        self.fill_rhs(g, input);
    }

    fn fill_rhs(&mut self, g: &StressGraph, input: &StressInput) {
        let mut b = std::mem::take(&mut self.work.b);
        b.clear();
        for &gi in &self.sys.nodes {
            b.push(self.residual(g, input, gi as usize) as f32);
        }
        self.work.b = b;
    }

    fn solve(&mut self, g: &StressGraph, input: &StressInput, cfg: &StressConfig, max_iters: usize) -> usize {
        let thr = (cfg.tol as f64 * self.w_rms).max(1e-30) as f32;
        let cap = self.local_cap.max(cfg.region_max_nodes).max(1);
        let global = self.need_global || self.n_unknown <= cap;
        if global {
            if self.sys_version != self.version {
                let mut nodes = std::mem::take(&mut self.sys.nodes);
                nodes.clear();
                nodes.extend((0..g.n as u32).filter(|&i| self.class[i as usize] == UNKNOWN));
                for (li, &i) in nodes.iter().enumerate() {
                    self.loc[i as usize] = li as u32;
                }
                self.sys.nodes = nodes;
                self.build_sys(g, input, false);
                for &i in &self.sys.nodes {
                    self.loc[i as usize] = u32::MAX;
                }
                self.sys_version = self.version;
            } else {
                self.fill_rhs(g, input);
            }
            // global covers everything pending
            for &i in &self.pending {
                self.pend_mark[i as usize] = false;
            }
            self.pending.clear();
        } else {
            self.sys_version = 0; // local system overwrites the cache
            let mut nodes = std::mem::take(&mut self.sys.nodes);
            nodes.clear();
            // Seeds: the worst-residual pending nodes (at most cap/8), so the region is a
            // coherent ball around where the error is, not a scatter of pending nodes.
            let mut pending = std::mem::take(&mut self.pending);
            let mut scored: Vec<(f64, u32)> = Vec::with_capacity(pending.len());
            for &s in &pending {
                self.pend_mark[s as usize] = false;
                if self.class[s as usize] != UNKNOWN {
                    continue;
                }
                let sc = self.residual(g, input, s as usize).abs();
                scored.push((sc, s));
            }
            scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
            let max_seeds = (cap / 8).max(1);
            let mut leftover = Vec::new();
            for &(_, s) in &scored {
                let si = s as usize;
                if nodes.len() < max_seeds {
                    self.loc[si] = nodes.len() as u32;
                    nodes.push(s);
                } else {
                    leftover.push(s);
                }
            }
            pending.clear();
            let mut head = 0;
            'bfs: while head < nodes.len() && nodes.len() < cap {
                let i = nodes[head] as usize;
                head += 1;
                for p in g.row(i) {
                    let j = g.adj_node[p] as usize;
                    if self.class[j] == UNKNOWN && self.loc[j] == u32::MAX && self.edge_k[g.adj_edge[p] as usize] > 0.0 {
                        if nodes.len() >= cap {
                            break 'bfs;
                        }
                        self.loc[j] = nodes.len() as u32;
                        nodes.push(j as u32);
                    }
                }
            }
            self.pending = pending;
            for s in leftover {
                if self.loc[s as usize] == u32::MAX {
                    self.pend(s);
                }
            }
            self.sys.nodes = nodes;
            self.build_sys(g, input, true);
            for &i in &self.sys.nodes {
                self.loc[i as usize] = u32::MAX;
            }
        }
        let par = self.sys.n() >= cfg.par_threshold;
        let (iters, conv) = cg::pcg::<1>(&self.sys, &mut self.work, [thr], max_iters, par);
        // apply correction
        for (li, &gi) in self.sys.nodes.iter().enumerate() {
            self.pot[gi as usize] += self.work.x[li] as f64;
        }
        if global {
            self.need_global = !conv;
            self.local_cap = 0;
        } else {
            let lf = cfg.local_tol_factor.max(1.0);
            if !conv {
                for li in 0..self.sys.n() {
                    if self.work.r[li].abs() > lf * thr {
                        let gi = self.sys.nodes[li];
                        self.pend(gi);
                    }
                }
            }
            // ring nodes absorb the correction's coupling; re-check them
            let ring = std::mem::take(&mut self.ring);
            for &j in &ring {
                if self.residual(g, input, j as usize).abs() > (lf * thr) as f64 {
                    self.pend(j);
                }
            }
            self.ring = ring;
            // grow the window while work remains so far-field effects get absorbed
            let limit = cfg.region_max_nodes.max(1).saturating_mul(cfg.max_region_growth.max(1));
            if self.pending.is_empty() {
                self.local_cap = 0;
                self.stall = 0;
            } else {
                self.local_cap = (cap + cap / 2).min(limit);
                if cap >= limit {
                    self.stall += 1;
                    if self.stall > STALL_TICKS {
                        // windowed sweeps aren't finishing: fall back to global steps
                        self.need_global = true;
                        self.stall = 0;
                    }
                }
            }
        }
        iters
    }
}

/// Unsigned key that sorts ascending in *descending* `f64::total_cmp` order.
#[inline]
fn desc_total_key(x: f64) -> u64 {
    let b = x.to_bits() as i64;
    // `total_cmp`'s signed key, then shifted to unsigned
    let t = (b ^ ((((b >> 63) as u64) >> 1) as i64)) as u64 ^ (1u64 << 63);
    !t
}

#[cfg(test)]
mod key_tests {
    #[test]
    fn desc_total_key_matches_total_cmp() {
        let v = [
            0.0, -0.0, 1.0, -1.0, 1e-300, -1e-300, f64::MIN_POSITIVE, f64::INFINITY, f64::NEG_INFINITY, 3.5, -3.5,
            1e300, -1e300, f64::NAN, -f64::NAN, 2.0, 2.0000000000000004,
        ];
        for &a in &v {
            for &b in &v {
                assert_eq!(super::desc_total_key(a).cmp(&super::desc_total_key(b)), b.total_cmp(&a), "{a} {b}");
            }
        }
    }
}

struct UtilParams {
    bending: bool,
    bearing: bool,
    bend_k: f32,
    compression: f32,
    tension: f32,
}

/// One amortized solver tick (see crate docs).
pub fn solve_step(g: &StressGraph, st: &mut StressState, input: &StressInput, cfg: &StressConfig, dt: f32) -> StressResult {
    let mut r = solve_step_lean(g, st, input, cfg, dt);
    r.utilization = st.util.clone();
    r
}

/// [`solve_step`] without the utilization copy: `StressResult::utilization` is left empty, read
/// it from [`StressState::utilization`] instead.
pub fn solve_step_lean(g: &StressGraph, st: &mut StressState, input: &StressInput, cfg: &StressConfig, dt: f32) -> StressResult {
    st.sync(g, input);
    let bend_full = cfg.bending && !st.moments_valid;
    st.moments_valid = cfg.bending;
    st.update_stats(input);
    let mut iters = 0;
    let solving = st.n_unknown > 0 && (st.need_global || !st.pending.is_empty());
    if st.n_unknown == 0 {
        st.need_global = false;
        for &i in &st.pending {
            st.pend_mark[i as usize] = false;
        }
        st.pending.clear();
    }
    // Catch-up: a tick that removed support carrying a large share of the load gets a full
    // global solve to convergence now (time-capped), so utilization reflects the damage within
    // a tick instead of creeping up over seconds of warm-started local iterations.
    let catchup = cfg.catchup_frac > 0.0 && st.removed_flow > cfg.catchup_frac as f64 * st.total_load && solving;
    st.removed_flow = 0.0;
    let mut region = Vec::new();
    if solving {
        if catchup {
            st.need_global = true;
            let t0 = std::time::Instant::now();
            loop {
                iters += st.solve(g, input, cfg, cfg.max_iters.max(CATCHUP_ROUND_ITERS));
                if !st.need_global || t0.elapsed().as_secs_f32() * 1e3 > cfg.catchup_max_ms {
                    break;
                }
            }
        } else {
            iters = st.solve(g, input, cfg, cfg.max_iters);
        }
        region = std::mem::take(&mut st.sys.nodes);
    }
    st.update_buckling(g, cfg);
    let up = UtilParams {
        bending: cfg.bending,
        bearing: cfg.bearing_sections,
        bend_k: cfg.bend_scale / cfg.flexural_factor.max(1e-6),
        compression: cfg.compression_factor.max(1e-6),
        tension: cfg.tension_factor.max(1e-6),
    };
    // a full moment update over a global solve moves exactly the solved nodes: their edges are
    // refreshed once below (a second pass would recompute the same values)
    let mut moved_is_region = false;
    if cfg.bending && (solving || bend_full) {
        let full = bend_full || region.len() * 2 > st.n_unknown;
        st.update_moments(g, input, &region, full);
        // region ⊆ supported nodes and `moved` = all supported nodes after a full update
        moved_is_region = full && region.len() == st.moved.len();
    } else {
        st.moved.clear();
    }
    // Refresh utilization of every edge touching the solved set (only their potentials
    // changed; in global mode this is every conducting edge with load on it) and of edges
    // leaving nodes whose carried-load centroid moved (bending term).
    for &i in &region {
        st.refresh_util_of(g, i as usize, &up);
    }
    if cfg.bending && !moved_is_region {
        let moved = std::mem::take(&mut st.moved);
        for &i in &moved {
            st.refresh_util_of(g, i as usize, &up);
        }
        st.moved = moved;
    }
    let cap_moved = std::mem::take(&mut st.cap_moved);
    for &i in &cap_moved {
        if st.class[i as usize] == UNKNOWN || st.class[i as usize] == ANCHOR {
            st.refresh_util_of(g, i as usize, &up);
        }
    }
    st.cap_moved = cap_moved;
    st.cap_moved.clear();
    st.flush_interfaces(g, &up);
    if solving {
        st.sys.nodes = region;
    }
    // hysteresis
    let mut cands: Vec<(f32, u32)> = Vec::new();
    let mut idx = 0;
    while idx < st.hot.len() {
        let e = st.hot[idx] as usize;
        if st.util[e] > 1.0 {
            st.timer[e] += dt;
            if st.timer[e] >= cfg.hold_time {
                cands.push((st.util[e], e as u32));
            }
            idx += 1;
        } else {
            st.timer[e] = 0.0;
            st.is_hot[e] = false;
            st.hot.swap_remove(idx);
        }
    }
    cands.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
    cands.truncate(cfg.max_breaks_per_tick);
    StressResult {
        utilization: Vec::new(),
        to_break: cands.into_iter().map(|(_, e)| e).collect(),
        converged: !st.need_global && st.pending.is_empty(),
        iters,
    }
}

/// One-shot full solve (iterate to convergence) for offline validation (bgen): no
/// hysteresis, no breaking.
pub fn static_report(g: &StressGraph, input: &StressInput, cfg: &StressConfig) -> StaticReport {
    let mut st = StressState::new(g);
    let mut c = cfg.clone();
    c.hold_time = f32::INFINITY;
    c.max_breaks_per_tick = 0;
    c.max_iters = cfg.max_iters.max(5000);
    let mut iters = 0;
    let mut converged = false;
    for _ in 0..20 {
        let r = solve_step(g, &mut st, input, &c, 0.0);
        iters += r.iters;
        if r.converged {
            converged = true;
            break;
        }
    }
    let mut order: Vec<u32> = (0..g.n_edges() as u32).filter(|&e| st.util[e as usize] > 0.0).collect();
    order.sort_by(|&a, &b| st.util[b as usize].total_cmp(&st.util[a as usize]).then(a.cmp(&b)));
    let worst_edges: Vec<(u32, f32)> = order.iter().take(20).map(|&e| (e, st.util[e as usize])).collect();
    let unsupported_nodes = (0..g.n as u32).filter(|&i| st.class[i as usize] == FLOATING).collect();
    let mut total_load = 0f64;
    for i in 0..g.n {
        if st.class[i] == UNKNOWN {
            total_load += input.node_weight[i] as f64;
        }
    }
    let mut anchor_flow = 0f64;
    for e in 0..g.n_edges() {
        let [a, b] = g.edge_ab[e];
        let (ca, cb) = (st.class[a as usize], st.class[b as usize]);
        if ca == UNKNOWN && cb == ANCHOR {
            anchor_flow += st.edge_flow(g, e as u32);
        } else if cb == UNKNOWN && ca == ANCHOR {
            anchor_flow -= st.edge_flow(g, e as u32);
        }
    }
    StaticReport {
        max_util: worst_edges.first().map(|w| w.1).unwrap_or(0.0),
        worst_edges,
        converged,
        iters,
        unsupported_nodes,
        total_load,
        anchor_flow,
        utilization: st.util,
    }
}
