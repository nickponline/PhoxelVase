//! Per-building immutable data + SoA mutable state + CSR graph (DESIGN §3.3).
use crate::math::{Pose, Vec3};
use crate::physics::{ColliderId, CompoundParts, PhysicsBackend, RapierBackend, Shape};
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
    /// `shapes[c].compute_aabb(identity)` and `shapes[c].ccd_thickness()` (compound part data)
    pub shape_aabb: Vec<rapier3d::parry::bounding_volume::Aabb>,
    pub shape_ccd: Vec<f32>,
    pub structural: Vec<bool>,
    pub anchor: Vec<bool>,
    pub weight: Vec<f32>,
    // CSR
    pub csr_off: Vec<u32>,
    pub csr_nbr: Vec<u32>,
    pub csr_edge: Vec<u32>,
    /// endpoints (a, b) per edge (compact copy of `bld.edges[e].a/b`)
    pub edge_ab: Vec<[u32; 2]>,
    pub edge_max_health: Vec<f32>,
    // mutable SoA
    pub state: Vec<ChunkState>,
    pub hp: Vec<f32>,
    /// initial (max) hp per chunk (compact copy of `bld.chunks[c].hp`)
    pub hp_max: Vec<f32>,
    /// last world pose for Frozen/Gone chunks
    pub chunk_pose: Vec<Pose>,
    pub collider: Vec<Option<ColliderId>>,
    /// world AABB of each live static collider (valid while `collider[c]` is `Some`)
    pub collider_aabb: Vec<(Vec3, Vec3)>,
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
    /// marks / epoch of `still_connected` (separate from `visit`/`epoch`)
    visit2: Vec<u32>,
    epoch2: u32,
    scratch_q: (Vec<u32>, Vec<u32>),
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
    /// per-chunk remaining strength fraction (hp / hp_max) fed to the stress solver
    pub stress_strength: Vec<f32>,
    pub stress_edge_alive: Vec<bool>,
    /// index of an edge inside its cluster's internal edge list
    pub edge_slot: Vec<u32>,
    /// chunks found unanchored at load (kept as static rubble)
    pub load_floating: u32,
    /// joints found overloaded in the intact building and pre-cracked at load
    pub load_cracked_edges: u32,
    /// building-space heights of damage since the last tipping check (see `tipping_edges`)
    pub damage_z: Vec<f32>,
    /// time since the last tipping check
    pub tip_timer: f32,
    /// static chunks next to recent damage, to test for hanging by slivers
    pub sliver_check: Vec<u32>,
    /// joints or chunks changed since the last full ground check (`ungrounded_components`)
    pub topo_changed: bool,
    /// time since the last full ground check
    pub ground_timer: f32,
}

impl Building {
    pub fn new(bld: Bld, pose: Pose, edge_health_per_newton: f32, gravity: f32) -> Self {
        let n = bld.chunks.len();
        // hull per chunk, with its compound part data computed while it is in cache
        let parts: Vec<(Shape, rapier3d::parry::bounding_volume::Aabb, f32)> = (0..n)
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
            .map(|s: Shape| {
                let aabb = s.compute_aabb(&Pose::IDENTITY);
                let ccd = s.ccd_thickness();
                (s, aabb, ccd)
            })
            .collect();
        let mut shapes = Vec::with_capacity(n);
        let mut shape_aabb = Vec::with_capacity(n);
        let mut shape_ccd = Vec::with_capacity(n);
        for (s, a, t) in parts {
            shapes.push(s);
            shape_aabb.push(a);
            shape_ccd.push(t);
        }
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
        let stress_graph = stress_graph_for(&bld);
        let stress_state = StressState::new(&stress_graph);
        let ne = bld.edges.len();
        Building {
            name: bld.name().to_string(),
            pose,
            shapes,
            shape_aabb,
            shape_ccd,
            structural,
            anchor,
            weight,
            csr_off,
            csr_nbr,
            csr_edge,
            edge_ab: bld.edges.iter().map(|e| [e.a, e.b]).collect(),
            edge_health: edge_max_health.clone(),
            edge_max_health,
            state: vec![ChunkState::Static; n],
            hp: bld.chunks.iter().map(|c| c.hp).collect(),
            hp_max: bld.chunks.iter().map(|c| c.hp).collect(),
            chunk_pose: vec![pose; n],
            collider: vec![None; n],
            collider_aabb: vec![(Vec3::ZERO, Vec3::ZERO); n],
            edge_alive: vec![true; ne],
            pending_impulse: vec![[0.0; 3]; n],
            pending_impulse_t: vec![f32::NEG_INFINITY; n],
            dirty: Vec::new(),
            visit: vec![0; n],
            comp_mark: vec![0; n],
            comp_ep: vec![0; n],
            grounded: vec![0; n],
            epoch: 0,
            visit2: vec![0; n],
            epoch2: 0,
            scratch_q: (Vec::new(), Vec::new()),
            stress_graph: std::sync::Arc::new(stress_graph),
            edge_inter_panel: std::sync::Arc::new(
                bld.edges.iter().map(|e| bld.chunks[e.a as usize].elem != bld.chunks[e.b as usize].elem).collect(),
            ),
            stress_state,
            stress_active: true,
            stress_unconverged: 0,
            utilization: vec![0.0; ne],
            stress_node_alive: vec![false; n],
            stress_strength: vec![1.0; n],
            stress_edge_alive: vec![false; ne],
            edge_slot: vec![u32::MAX; ne],
            load_floating: 0,
            load_cracked_edges: 0,
            damage_z: Vec::new(),
            tip_timer: 0.0,
            sliver_check: Vec::new(),
            topo_changed: false,
            ground_timer: 0.0,
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

    /// Are both ends of every edge in `edges` still connected through alive edges between chunks
    /// of cluster `k`? Bidirectional BFS per edge with a work cap; `false` also means "gave up".
    pub fn still_connected(&mut self, k: ClusterKey, edges: &[u32]) -> bool {
        const MAX_VISITS: usize = 4096;
        let me = ChunkState::InCluster(k);
        let (mut qa, mut qb) = std::mem::take(&mut self.scratch_q);
        let mut ok = true;
        for &e in edges {
            let [a, b] = self.edge_ab[e as usize];
            if self.state[a as usize] != me || self.state[b as usize] != me {
                ok = false;
                break;
            }
            if self.epoch2 >= u32::MAX - 2 {
                self.visit2.iter_mut().for_each(|v| *v = 0);
                self.epoch2 = 0;
            }
            let (ta, tb) = (self.epoch2 + 1, self.epoch2 + 2);
            self.epoch2 += 2;
            qa.clear();
            qb.clear();
            qa.push(a);
            qb.push(b);
            self.visit2[a as usize] = ta;
            self.visit2[b as usize] = tb;
            let (mut ha, mut hb) = (0usize, 0usize);
            let mut met = false;
            'search: while ha < qa.len() && hb < qb.len() && qa.len() + qb.len() < MAX_VISITS {
                // expand one node from the side with the smaller frontier
                let (q, h, mine, other) =
                    if qa.len() - ha <= qb.len() - hb { (&mut qa, &mut ha, ta, tb) } else { (&mut qb, &mut hb, tb, ta) };
                let n = q[*h] as usize;
                *h += 1;
                for kk in self.csr_off[n] as usize..self.csr_off[n + 1] as usize {
                    let nb = self.csr_nbr[kk] as usize;
                    if !self.edge_alive[self.csr_edge[kk] as usize] || self.state[nb] != me {
                        continue;
                    }
                    let v = self.visit2[nb];
                    if v == other {
                        met = true;
                        break 'search;
                    }
                    if v != mine {
                        self.visit2[nb] = mine;
                        q.push(nb as u32);
                    }
                }
            }
            if !met {
                ok = false;
                break;
            }
        }
        self.scratch_q = (qa, qb);
        ok
    }

    /// Rigid-body tipping across the horizontal plane at building-space height `z`.
    ///
    /// Every connected group of static structural chunks whose centres lie above the plane
    /// (and that holds no anchor itself) reaches the ground only through the alive joints
    /// crossing the plane, so it can only stand if its centre of mass lies over the footprint
    /// of those joints (their contact patches' convex hull). The local stress model cannot see
    /// this when the remaining joints sit inside one member, e.g. a building cut through
    /// except for one wall: the solver hands load over between members at the contact points,
    /// so a whole storey standing on a strip of one wall reads as lightly loaded. Returns the
    /// crossing joints of every group whose centre of mass is more than `margin` outside its
    /// support; breaking them lets the group fall and the rigid-body simulation tip it over.
    pub fn tipping_edges(&self, z: f32, margin: f32) -> Vec<u32> {
        let n = self.n_chunks();
        let above = |c: usize| self.state[c] == ChunkState::Static && self.structural[c] && self.bld.chunks[c].com[2] > z;
        let mut seen = vec![false; n];
        let mut out = Vec::new();
        let (mut stack, mut crossing, mut pts) = (Vec::new(), Vec::new(), Vec::new());
        for s in 0..n {
            if seen[s] || !above(s) {
                continue;
            }
            seen[s] = true;
            stack.push(s);
            crossing.clear();
            let (mut anchored, mut m, mut mx, mut my) = (false, 0f64, 0f64, 0f64);
            while let Some(c) = stack.pop() {
                anchored |= self.anchor[c];
                let ch = &self.bld.chunks[c];
                m += ch.mass as f64;
                mx += ch.mass as f64 * ch.com[0] as f64;
                my += ch.mass as f64 * ch.com[1] as f64;
                for k in self.csr_off[c] as usize..self.csr_off[c + 1] as usize {
                    let (nb, e) = (self.csr_nbr[k] as usize, self.csr_edge[k]);
                    if !self.edge_alive[e as usize] || self.state[nb] != ChunkState::Static || !self.structural[nb] {
                        continue;
                    }
                    if above(nb) {
                        if !seen[nb] {
                            seen[nb] = true;
                            stack.push(nb);
                        }
                    } else {
                        crossing.push(e);
                    }
                }
            }
            if anchored || crossing.is_empty() || m <= 0.0 {
                continue;
            }
            pts.clear();
            for &e in &crossing {
                let ed = &self.bld.edges[e as usize];
                let h = 0.5 * ed.area.max(0.0).sqrt() as f64;
                let (x, y) = (ed.centroid[0] as f64, ed.centroid[1] as f64);
                pts.extend_from_slice(&[[x - h, y - h], [x + h, y - h], [x + h, y + h], [x - h, y + h]]);
            }
            if dist_outside_hull(&mut pts, [mx / m, my / m]) > margin as f64 {
                out.extend_from_slice(&crossing);
            }
        }
        out
    }

    /// Full ground check over the load-carrying graph (the stress solver's: alive joints with
    /// positive capacity between alive static structural chunks): every connected component
    /// that reaches no anchor, independent of the incremental dirty-node search. Non-structural
    /// (glass, cosmetic) chunks hanging only on such a component, or on nothing grounded, join
    /// it (or form their own). Chunks already detaching are not included.
    pub fn ungrounded_components(&self) -> Vec<Vec<u32>> {
        let n = self.n_chunks();
        let st = |c: usize| self.state[c] == ChunkState::Static;
        let carries = |e: usize| self.edge_alive[e] && self.bld.edges[e].strength > 0.0;
        // 1. reach from every anchor
        let mut reached = vec![false; n];
        let mut q: Vec<usize> = (0..n).filter(|&c| st(c) && self.structural[c] && self.anchor[c]).collect();
        for &c in &q {
            reached[c] = true;
        }
        while let Some(c) = q.pop() {
            for k in self.csr_off[c] as usize..self.csr_off[c + 1] as usize {
                let (nb, e) = (self.csr_nbr[k] as usize, self.csr_edge[k] as usize);
                if carries(e) && st(nb) && self.structural[nb] && !reached[nb] {
                    reached[nb] = true;
                    q.push(nb);
                }
            }
        }
        // 2. components of what was not reached
        let mut comp_of = vec![u32::MAX; n];
        let mut comps: Vec<Vec<u32>> = vec![];
        for s0 in 0..n {
            if !st(s0) || !self.structural[s0] || reached[s0] || comp_of[s0] != u32::MAX {
                continue;
            }
            let id = comps.len() as u32;
            let mut comp = vec![s0 as u32];
            comp_of[s0] = id;
            let mut i = 0;
            while i < comp.len() {
                let c = comp[i] as usize;
                i += 1;
                for k in self.csr_off[c] as usize..self.csr_off[c + 1] as usize {
                    let (nb, e) = (self.csr_nbr[k] as usize, self.csr_edge[k] as usize);
                    if carries(e) && st(nb) && self.structural[nb] && comp_of[nb] == u32::MAX {
                        comp_of[nb] = id;
                        comp.push(nb as u32);
                    }
                }
            }
            comps.push(comp);
        }
        // 3. non-structural groups: grounded if they touch reached structure (or an anchor)
        let mut seen = vec![false; n];
        for s0 in 0..n {
            if !st(s0) || self.structural[s0] || seen[s0] {
                continue;
            }
            let mut group = vec![s0 as u32];
            seen[s0] = true;
            let (mut i, mut grounded, mut attach) = (0, false, None);
            while i < group.len() {
                let c = group[i] as usize;
                i += 1;
                grounded |= self.anchor[c];
                for k in self.csr_off[c] as usize..self.csr_off[c + 1] as usize {
                    let (nb, e) = (self.csr_nbr[k] as usize, self.csr_edge[k] as usize);
                    if !self.edge_alive[e] || !st(nb) {
                        continue;
                    }
                    if self.structural[nb] {
                        if reached[nb] {
                            grounded = true;
                        } else if comp_of[nb] != u32::MAX {
                            attach.get_or_insert(comp_of[nb] as usize);
                        }
                    } else if !seen[nb] {
                        seen[nb] = true;
                        group.push(nb as u32);
                    }
                }
            }
            if !grounded {
                match attach {
                    Some(i) => comps[i].extend_from_slice(&group),
                    None => comps.push(group),
                }
            }
        }
        comps
    }

    /// Is static chunk `c` hanging by slivers (see `WorldConfig::sliver_area`)? Anchors and
    /// non-structural chunks never are.
    pub fn hangs_by_slivers(&self, c: usize, max_area: f32, seat_area: f32) -> bool {
        if self.state[c] != ChunkState::Static || self.anchor[c] || !self.structural[c] {
            return false;
        }
        let zc = self.bld.chunks[c].com[2];
        let mut area = 0.0;
        for k in self.csr_off[c] as usize..self.csr_off[c + 1] as usize {
            let (nb, e) = (self.csr_nbr[k] as usize, self.csr_edge[k] as usize);
            if !self.edge_alive[e] || self.state[nb] != ChunkState::Static {
                continue;
            }
            let ed = &self.bld.edges[e];
            if ed.area >= seat_area && ed.centroid[2] < zc - 0.05 {
                return false; // sits on something
            }
            area += ed.area;
            if area >= max_area {
                return false;
            }
        }
        true
    }

    pub fn total_static_structural_mass(&self) -> f32 {
        (0..self.n_chunks())
            .filter(|&c| self.state[c] == ChunkState::Static && self.structural[c])
            .map(|c| self.bld.chunks[c].mass)
            .sum()
    }
}

impl Building {
    /// Compound parts for these chunks (in this order).
    pub fn compound_parts(&self, chunks: &[u32]) -> CompoundParts {
        CompoundParts {
            shapes: chunks.iter().map(|&c| self.shapes[c as usize].clone()).collect(),
            aabbs: chunks.iter().map(|&c| self.shape_aabb[c as usize]).collect(),
            ccd: chunks.iter().map(|&c| self.shape_ccd[c as usize]).collect(),
        }
    }
}

impl RapierBackend {
    pub fn convex_hull_shape(points: &[Vec3]) -> Option<Shape> {
        <RapierBackend as PhysicsBackend>::convex_hull(points)
    }
}

/// ELEM kind id of columns (bgen `ELEMENT_KINDS`).
pub const KIND_COLUMN: u16 = 3;

/// The stress graph of a building (shared by `Building::new` and rubble-py's static report):
/// edge capacity = strength, contact areas, chunks grouped by element (member sections) and
/// column chunks (ELEM kind `column`) marked with their section thickness for buckling.
pub fn stress_graph_for(bld: &Bld) -> StressGraph {
    let n = bld.chunks.len();
    let pairs: Vec<(u32, u32)> = bld.edges.iter().map(|e| (e.a, e.b)).collect();
    let cap: Vec<f32> = bld.edges.iter().map(|e| e.strength).collect();
    let cen: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.centroid).collect();
    let nrm: Vec<[f32; 3]> = bld.edges.iter().map(|e| e.normal).collect();
    let pos: Vec<[f32; 3]> = bld.chunks.iter().map(|c| c.com).collect();
    let area: Vec<f32> = bld.edges.iter().map(|e| e.area).collect();
    let elem: Vec<u32> = bld.chunks.iter().map(|c| c.elem).collect();
    let col: Vec<f32> = bld
        .chunks
        .iter()
        .map(|c| {
            let is_col = bld.elements.get(c.elem as usize).is_some_and(|e| e.kind == KIND_COLUMN);
            if is_col {
                (c.aabb_max[0] - c.aabb_min[0]).min(c.aabb_max[1] - c.aabb_min[1]).max(0.0)
            } else {
                0.0
            }
        })
        .collect();
    StressGraph::new(n, &pairs, &cap, &cen, &nrm, &pos).with_areas(&area).with_node_groups(&elem).with_columns(&col)
}

/// Distance from `p` to the convex hull of `pts` (0 inside). Reorders `pts`.
pub fn dist_outside_hull(pts: &mut [[f64; 2]], p: [f64; 2]) -> f64 {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    // Andrew's monotone chain, CCW
    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(pts.len() + 1);
    for pass in 0..2 {
        let start = hull.len();
        let iter: Box<dyn Iterator<Item = &[f64; 2]>> = if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &q in iter {
            while hull.len() >= start + 2 && cross(hull[hull.len() - 2], hull[hull.len() - 1], q) <= 0.0 {
                hull.pop();
            }
            hull.push(q);
        }
        hull.pop();
    }
    let seg = |a: [f64; 2], b: [f64; 2]| {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 { (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        ((p[0] - a[0] - t * dx).powi(2) + (p[1] - a[1] - t * dy).powi(2)).sqrt()
    };
    match hull.len() {
        0 => f64::INFINITY,
        1 => seg(hull[0], hull[0]),
        2 => seg(hull[0], hull[1]),
        k => {
            let inside = (0..k).all(|i| cross(hull[i], hull[(i + 1) % k], p) >= 0.0);
            if inside {
                0.0
            } else {
                (0..k).map(|i| seg(hull[i], hull[(i + 1) % k])).fold(f64::INFINITY, f64::min)
            }
        }
    }
}
