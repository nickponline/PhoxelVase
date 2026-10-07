//! Small synthetic structures for tests, benches and downstream crates (rubble-core).
//! Units: metres, Newtons. Nodes are unit-ish cubes; edge centroid = midpoint of the two
//! node centres, normal = unit vector a→b, area given per edge.

use crate::{StressGraph, StressInput};

#[derive(Clone, Debug, Default)]
pub struct TestGraph {
    pub node_pos: Vec<[f32; 3]>,
    pub node_weight: Vec<f32>,
    pub anchor: Vec<bool>,
    pub node_alive: Vec<bool>,
    pub edges: Vec<(u32, u32)>,
    pub capacity: Vec<f32>,
    pub centroid: Vec<[f32; 3]>,
    pub normal: Vec<[f32; 3]>,
    pub area: Vec<f32>,
    pub edge_alive: Vec<bool>,
}

impl TestGraph {
    pub fn add_node(&mut self, pos: [f32; 3], weight: f32, anchor: bool) -> u32 {
        self.node_pos.push(pos);
        self.node_weight.push(weight);
        self.anchor.push(anchor);
        self.node_alive.push(true);
        (self.node_pos.len() - 1) as u32
    }
    pub fn add_edge(&mut self, a: u32, b: u32, capacity: f32, area: f32) -> u32 {
        let (pa, pb) = (self.node_pos[a as usize], self.node_pos[b as usize]);
        let d = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-12);
        self.edges.push((a, b));
        self.capacity.push(capacity);
        self.centroid.push([0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])]);
        self.normal.push([d[0] / l, d[1] / l, d[2] / l]);
        self.area.push(area);
        self.edge_alive.push(true);
        (self.edges.len() - 1) as u32
    }
    pub fn n_nodes(&self) -> usize {
        self.node_pos.len()
    }
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
    /// Build the solver graph (with areas).
    pub fn graph(&self) -> StressGraph {
        StressGraph::new(self.n_nodes(), &self.edges, &self.capacity, &self.centroid, &self.normal, &self.node_pos)
            .with_areas(&self.area)
    }
    pub fn input(&self) -> StressInput<'_> {
        StressInput {
            node_weight: &self.node_weight,
            node_alive: &self.node_alive,
            anchor: &self.anchor,
            edge_alive: &self.edge_alive,
        }
    }
    /// Find the edge between a and b (either orientation).
    pub fn edge_between(&self, a: u32, b: u32) -> Option<u32> {
        self.edges.iter().position(|&(x, y)| (x, y) == (a, b) || (x, y) == (b, a)).map(|e| e as u32)
    }
}

/// Anchor (node 0, z=0) + `n` stacked cubes (nodes 1..=n at z=1..n), each weighing `w`.
/// Edge i-1 joins node i-1 and i and carries `(n - i + 1)·w`.
pub fn column(n: usize, w: f32, cap: f32) -> TestGraph {
    let mut t = TestGraph::default();
    let mut prev = t.add_node([0.0, 0.0, 0.0], w, true);
    for i in 1..=n {
        let cur = t.add_node([0.0, 0.0, i as f32], w, false);
        t.add_edge(prev, cur, cap, 1.0);
        prev = cur;
    }
    t
}

/// Two columns of height `h` (anchored bases at x=0 and x=2·half_span) under a slab row of
/// `2·half_span+1` cubes at z=h+1, symmetric about the middle.
/// Returns (graph, [top edge of column A, top edge of column B]).
pub fn two_columns_slab(h: usize, half_span: usize, w: f32, cap: f32) -> (TestGraph, [u32; 2]) {
    let mut t = TestGraph::default();
    let span = 2 * half_span;
    let mut tops = [0u32; 2];
    for (ci, x) in [0.0, span as f32].into_iter().enumerate() {
        let mut prev = t.add_node([x, 0.0, 0.0], w, true);
        for z in 1..=h {
            let cur = t.add_node([x, 0.0, z as f32], w, false);
            t.add_edge(prev, cur, cap, 1.0);
            prev = cur;
        }
        tops[ci] = prev;
    }
    let mut slab = Vec::new();
    for i in 0..=span {
        slab.push(t.add_node([i as f32, 0.0, (h + 1) as f32], w, false));
    }
    for i in 0..span {
        t.add_edge(slab[i], slab[i + 1], cap, 1.0);
    }
    let ea = t.add_edge(tops[0], slab[0], cap, 1.0);
    let eb = t.add_edge(tops[1], slab[span], cap, 1.0);
    (t, [ea, eb])
}

pub struct Table {
    pub g: TestGraph,
    /// Leg node ids (excluding the anchored foot) per leg.
    pub legs: [Vec<u32>; 4],
    /// Edge from the top of each leg into the slab.
    pub leg_top_edges: [u32; 4],
}

/// `n×n` slab of cubes (weight `w_slab` each) at z = leg_h+1 on four legs of `leg_h` cubes at
/// the corners, each with an anchored foot at z=0. All edges have capacity `cap_slab`,
/// except leg edges which have `cap_leg`.
pub fn table(n: usize, leg_h: usize, w_slab: f32, cap_slab: f32, cap_leg: f32) -> Table {
    assert!(n >= 2);
    let mut t = TestGraph::default();
    let z = (leg_h + 1) as f32;
    let mut slab = vec![0u32; n * n];
    for y in 0..n {
        for x in 0..n {
            slab[y * n + x] = t.add_node([x as f32, y as f32, z], w_slab, false);
        }
    }
    for y in 0..n {
        for x in 0..n {
            if x + 1 < n {
                t.add_edge(slab[y * n + x], slab[y * n + x + 1], cap_slab, 1.0);
            }
            if y + 1 < n {
                t.add_edge(slab[y * n + x], slab[(y + 1) * n + x], cap_slab, 1.0);
            }
        }
    }
    let corners = [(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1)];
    let mut legs: [Vec<u32>; 4] = Default::default();
    let mut tops = [0u32; 4];
    for (li, &(x, y)) in corners.iter().enumerate() {
        let mut prev = t.add_node([x as f32, y as f32, 0.0], 0.0, true);
        for k in 1..=leg_h {
            let cur = t.add_node([x as f32, y as f32, k as f32], 0.0, false);
            t.add_edge(prev, cur, cap_leg, 1.0);
            legs[li].push(cur);
            prev = cur;
        }
        tops[li] = t.add_edge(prev, slab[y * n + x], cap_leg, 1.0);
    }
    Table { g: t, legs, leg_top_edges: tops }
}

/// Anchored wall column of `wall_h` cubes (x=0), plus a horizontal beam of `len` cubes
/// sticking out in +x from the wall top (z = wall_h). Returns (graph, root edge wall→beam).
pub fn cantilever(wall_h: usize, len: usize, w: f32, cap: f32, area: f32) -> (TestGraph, u32) {
    let mut t = TestGraph::default();
    let mut prev = t.add_node([0.0, 0.0, 0.0], w, true);
    for z in 1..=wall_h {
        let cur = t.add_node([0.0, 0.0, z as f32], w, false);
        t.add_edge(prev, cur, cap, area);
        prev = cur;
    }
    let mut root = u32::MAX;
    for x in 1..=len {
        let cur = t.add_node([x as f32, 0.0, wall_h as f32], w, false);
        let e = t.add_edge(prev, cur, cap, area);
        if x == 1 {
            root = e;
        }
        prev = cur;
    }
    (t, root)
}

/// Solid `nx×ny×nz` lattice of unit cubes with 6-neighbour edges; layer z=0 anchored.
/// Node id = (z·ny + y)·nx + x. ~3N edges (47×47×46 ≈ 100k nodes / 300k edges).
pub fn grid_building(nx: usize, ny: usize, nz: usize, w: f32, cap: f32) -> TestGraph {
    let mut t = TestGraph::default();
    let id = |x: usize, y: usize, z: usize| ((z * ny + y) * nx + x) as u32;
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                t.add_node([x as f32, y as f32, z as f32], w, z == 0);
            }
        }
    }
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if x + 1 < nx {
                    t.add_edge(id(x, y, z), id(x + 1, y, z), cap, 1.0);
                }
                if y + 1 < ny {
                    t.add_edge(id(x, y, z), id(x, y + 1, z), cap, 1.0);
                }
                if z + 1 < nz {
                    t.add_edge(id(x, y, z), id(x, y, z + 1), cap, 1.0);
                }
            }
        }
    }
    t
}
