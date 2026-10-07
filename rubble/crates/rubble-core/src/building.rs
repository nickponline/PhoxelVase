//! Per-building immutable data + SoA mutable state + CSR graph (DESIGN §3.3).
use crate::math::{Pose, Vec3};
use crate::physics::{ColliderId, PhysicsBackend, RapierBackend, Shape};
use rayon::prelude::*;
use rubble_format::{Bld, F_COSMETIC_ATTACHED, F_GLASS, F_INDESTRUCTIBLE};
use rubble_stress::{StressGraph, StressState};

slotmap::new_key_type! {
    /// Internal cluster key.
    pub struct ClusterKey;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkState {
    /// part of the intact static building
    Static,
    /// detached from anchors, waiting `collapse_delay` (still a static collider)
    Detaching,
    InCluster(ClusterKey),
    /// settled rubble: static, anchored, still destructible
    Frozen,
    Gone,
}

impl ChunkState {
    pub fn code(&self) -> u8 {
        match self {
            ChunkState::Static => 0,
            ChunkState::Detaching => 1,
            ChunkState::InCluster(_) => 2,
            ChunkState::Frozen => 3,
            ChunkState::Gone => 4,
        }
    }
}

pub struct Building {
    pub name: String,
    pub pose: Pose,
    pub bld: Bld,
    /// convex hull per chunk, in building space
    pub shapes: Vec<Shape>,
    pub structural: Vec<bool>,
    pub anchor: Vec<bool>,
    pub weight: Vec<f32>,
    // CSR
    pub csr_off: Vec<u32>,
    pub csr_nbr: Vec<u32>,
    pub csr_edge: Vec<u32>,
    pub edge_max_health: Vec<f32>,
    // mutable SoA
    pub state: Vec<ChunkState>,
    pub hp: Vec<f32>,
    /// last world pose for Frozen/Gone chunks
    pub chunk_pose: Vec<Pose>,
    pub collider: Vec<Option<ColliderId>>,
    pub edge_alive: Vec<bool>,
    pub edge_health: Vec<f32>,
    pub pending_impulse: Vec<[f32; 3]>,
    pub pending_impulse_t: Vec<f32>,
    // connectivity scratch
    pub dirty: Vec<u32>,
    pub visit: Vec<u32>,
    pub comp_mark: Vec<u32>,
    pub comp_ep: Vec<u32>,
    pub grounded: Vec<u32>,
    pub epoch: u32,
    // stress
    pub stress_graph: std::sync::Arc<StressGraph>,
    /// per edge: joins two different elements (panels)
    pub edge_inter_panel: std::sync::Arc<Vec<bool>>,
    pub stress_state: StressState,
    pub stress_active: bool,
    /// consecutive stress ticks without convergence
    pub stress_unconverged: u32,
    pub utilization: Vec<f32>,
    pub stress_node_alive: Vec<bool>,
    pub stress_edge_alive: Vec<bool>,
    /// index of an edge inside its cluster's internal edge list
    pub edge_slot: Vec<u32>,
    /// chunks found unanchored at load (kept as static rubble)
    pub load_floating: u32,
    /// joints found overloaded in the intact building and pre-cracked at load
    pub load_cracked_edges: u32,
}

impl Building {
    pub fn new(bld: Bld, pose: Pose, edge_health_per_newton: f32, gravity: f32) -> Self {
        let n = bld.chunks.len();
        let shapes: Vec<Shape> = (0..n)
            .into_par_iter()
            .map(|c| {
                let pts: Vec<Vec3> = bld.hull_verts_of(c).iter().map(|v| Vec3::from(*v)).collect();
                RapierBackend::convex_hull_shape(&pts).unwrap_or_else(|| {
                    let ch = &bld.chunks[c];
                    let lo = Vec3::from(ch.aabb_min);
                    let hi = Vec3::from(ch.aabb_max);
                    let corners: Vec<Vec3> = (0..8)
                        .map(|i| {
                            Vec3::new(
                                if i & 1 == 0 { lo.x } else { hi.x },
                                if i & 2 == 0 { lo.y } else { hi.y },
                                if i & 4 == 0 { lo.z } else { hi.z.max(lo.z + 1e-3) },
                            )
                        })
                        .collect();
                    RapierBackend::convex_hull_shape(&corners).expect("degenerate chunk hull")
                })
            })
            .collect();
        let structural: Vec<bool> =
            bld.chunks.iter().map(|c| c.flags & (F_GLASS | F_COSMETIC_ATTACHED) == 0).collect();
        let anchor: Vec<bool> = (0..n).map(|c| bld.is_anchor(c)).collect();
        let weight: Vec<f32> =
            (0..n).map(|c| if structural[c] { bld.chunks[c].mass * gravity } else { 0.0 }).collect();
        // CSR
        let mut deg = vec![0u32; n + 1];
        for e in &bld.edges {
            deg[e.a as usize] += 1;
            deg[e.b as usize] += 1;
        }
        let mut csr_off = vec![0u32; n + 1];
        for i in 0..n {
            csr_off[i + 1] = csr_off[i] + deg[i];
        }
        let m = csr_off[n] as usize;
        let mut fill = csr_off.clone();
        let mut csr_nbr = vec![0u32; m];
        let mut csr_edge = vec![0u32; m];
        for (ei, e) in bld.edges.iter().enumerate() {
            for (x, y) in [(e.a, e.b), (e.b, e.a)] {
                let k = fill[x as usize] as usize;
                csr_nbr[k] = y;
                csr_edge[k] = ei as u32;
                fill[x as usize] += 1;
            }
        }
        let edge_max_health: Vec<f32> =
            bld.edges.iter().map(|e| (e.strength * edge_health_per_newton).max(1.0)).collect();
        let pairs: Vec<(u32, u32)> = bld.edges.iter().map(|e| (e.a, e.b)).collect();
        let cap: Vec<f32> = bld.edges.iter().map(|e| e.strength).collect();
        let cen: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.centroid).collect();
        let nrm: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.normal).collect();
        let pos: Vec<[f32; 3]> = bld.chunks.iter().map(|c| c.com).collect();
        let area: Vec<f32> = bld.edges.iter().map(|e| e.area).collect();
        let elem: Vec<u32> = bld.chunks.iter().map(|c| c.elem).collect();
        let stress_graph = StressGraph::new(n, &pairs, &cap, &cen, &nrm, &pos).with_areas(&area).with_node_groups(&elem);
        let stress_state = StressState::new(&stress_graph);
        let ne = bld.edges.len();
        Building {
            name: bld.name().to_string(),
            pose,
            shapes,
            structural,
            anchor,
            weight,
            csr_off,
            csr_nbr,
            csr_edge,
            edge_health: edge_max_health.clone(),
            edge_max_health,
            state: vec![ChunkState::Static; n],
            hp: bld.chunks.iter().map(|c| c.hp).collect(),
            chunk_pose: vec![pose; n],
            collider: vec![None; n],
            edge_alive: vec![true; ne],
            pending_impulse: vec![[0.0; 3]; n],
            pending_impulse_t: vec![f32::NEG_INFINITY; n],
            dirty: Vec::new(),
            visit: vec![0; n],
            comp_mark: vec![0; n],
            comp_ep: vec![0; n],
            grounded: vec![0; n],
            epoch: 0,
            stress_graph: std::sync::Arc::new(stress_graph),
            edge_inter_panel: std::sync::Arc::new(
                bld.edges.iter().map(|e| bld.chunks[e.a as usize].elem != bld.chunks[e.b as usize].elem).collect(),
            ),
            stress_state,
            stress_active: true,
            stress_unconverged: 0,
            utilization: vec![0.0; ne],
            stress_node_alive: vec![false; n],
            stress_edge_alive: vec![false; ne],
            edge_slot: vec![u32::MAX; ne],
            load_floating: 0,
            load_cracked_edges: 0,
            bld,
        }
    }

    #[inline]
    pub fn n_chunks(&self) -> usize {
        self.state.len()
    }
    #[inline]
    pub fn neighbors(&self, c: u32) -> impl Iterator<Item = (u32, u32)> + '_ {
        let (a, b) = (self.csr_off[c as usize] as usize, self.csr_off[c as usize + 1] as usize);
        self.csr_nbr[a..b].iter().copied().zip(self.csr_edge[a..b].iter().copied())
    }
    #[inline]
    pub fn alive(&self, c: usize) -> bool {
        self.state[c] != ChunkState::Gone
    }
    #[inline]
    pub fn indestructible(&self, c: usize) -> bool {
        self.bld.chunks[c].flags & F_INDESTRUCTIBLE != 0
    }
    #[inline]
    pub fn glass(&self, c: usize) -> bool {
        self.bld.chunks[c].flags & F_GLASS != 0
    }
    pub fn mark_dirty(&mut self, c: u32) {
        self.dirty.push(c);
        self.stress_active = true;
    }

    /// Incremental connectivity (DESIGN §3.4): BFS from dirty nodes over alive edges between
    /// `Static` chunks, early-out on anchors, epoch-stamped visits. Non-structural chunks
    /// (glass, cosmetic) never bridge structural components; they detach with their neighbours.
    pub fn find_detached(&mut self) -> Vec<Vec<u32>> {
        if self.dirty.is_empty() {
            return Vec::new();
        }
        self.epoch = self.epoch.wrapping_add(1).max(1);
        let ep = self.epoch;
        let dirty = std::mem::take(&mut self.dirty);
        let mut out: Vec<Vec<u32>> = Vec::new();
        let mut cosmetic_seeds: Vec<u32> = Vec::new();
        for &d in &dirty {
            let du = d as usize;
            if self.state[du] != ChunkState::Static || self.visit[du] == ep {
                continue;
            }
            if !self.structural[du] {
                cosmetic_seeds.push(d);
                continue;
            }
            if let Some(comp) = self.bfs_structural(d, ep) {
                out.push(comp);
            }
        }
        for (i, comp) in out.iter().enumerate() {
            for &c in comp {
                self.comp_mark[c as usize] = i as u32;
                self.comp_ep[c as usize] = ep;
                for (nb, e) in self.neighbors(c) {
                    if self.edge_alive[e as usize]
                        && !self.structural[nb as usize]
                        && self.state[nb as usize] == ChunkState::Static
                    {
                        cosmetic_seeds.push(nb);
                    }
                }
            }
        }
        // non-structural pass
        let mut group: Vec<u32> = Vec::new();
        for s in cosmetic_seeds {
            let su = s as usize;
            if self.state[su] != ChunkState::Static || self.visit[su] == ep {
                continue;
            }
            group.clear();
            group.push(s);
            self.visit[su] = ep;
            let mut head = 0;
            let mut grounded = false;
            let mut attach: Option<usize> = None;
            while head < group.len() {
                let n = group[head];
                head += 1;
                if self.anchor[n as usize] {
                    grounded = true;
                }
                let (a, b) = (self.csr_off[n as usize] as usize, self.csr_off[n as usize + 1] as usize);
                for k in a..b {
                    let (nb, e) = (self.csr_nbr[k], self.csr_edge[k]);
                    let nu = nb as usize;
                    if !self.edge_alive[e as usize] || self.state[nu] != ChunkState::Static {
                        continue;
                    }
                    if !self.structural[nu] {
                        if self.visit[nu] != ep {
                            self.visit[nu] = ep;
                            group.push(nb);
                        }
                        continue;
                    }
                    // structural neighbour: grounded or detached?
                    if self.visit[nu] == ep {
                        if self.comp_ep[nu] == ep {
                            attach.get_or_insert(self.comp_mark[nu] as usize);
                        } else {
                            grounded = true;
                        }
                    } else if let Some(comp) = self.bfs_structural(nb, ep) {
                        let idx = out.len();
                        for &c in &comp {
                            self.comp_mark[c as usize] = idx as u32;
                            self.comp_ep[c as usize] = ep;
                        }
                        out.push(comp);
                        attach.get_or_insert(idx);
                    } else {
                        grounded = true;
                    }
                }
            }
            if grounded {
                continue;
            }
            match attach {
                Some(i) => out[i].extend_from_slice(&group),
                None => out.push(group.clone()),
            }
        }
        out
    }

    /// BFS over structural Static chunks; returns the component if it has no anchor.
    fn bfs_structural(&mut self, start: u32, ep: u32) -> Option<Vec<u32>> {
        let mut comp = vec![start];
        self.visit[start as usize] = ep;
        let mut found = self.anchor[start as usize];
        let mut head = 0;
        'outer: while !found && head < comp.len() {
            let n = comp[head] as usize;
            head += 1;
            let (a, b) = (self.csr_off[n] as usize, self.csr_off[n + 1] as usize);
            for k in a..b {
                let nb = self.csr_nbr[k] as usize;
                if !self.edge_alive[self.csr_edge[k] as usize]
                    || self.state[nb] != ChunkState::Static
                    || !self.structural[nb]
                {
                    continue;
                }
                if self.visit[nb] == ep {
                    if self.grounded[nb] == ep {
                        found = true;
                        break 'outer;
                    }
                    continue;
                }
                self.visit[nb] = ep;
                comp.push(nb as u32);
                if self.anchor[nb] {
                    found = true;
                    break 'outer;
                }
            }
        }
        if found {
            for &c in &comp {
                self.grounded[c as usize] = ep;
            }
            None
        } else {
            Some(comp)
        }
    }

    pub fn total_static_structural_mass(&self) -> f32 {
        (0..self.n_chunks())
            .filter(|&c| self.state[c] == ChunkState::Static && self.structural[c])
            .map(|c| self.bld.chunks[c].mass)
            .sum()
    }
}

impl RapierBackend {
    pub fn convex_hull_shape(points: &[Vec3]) -> Option<Shape> {
        <RapierBackend as PhysicsBackend>::convex_hull(points)
    }
}
