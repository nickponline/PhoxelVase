//! Synthetic in-memory buildings (axis-aligned boxes with face-contact edges) shared by
//! tests, `rubble-sim bench` and synthetic scenarios.
use rubble_format::*;
use std::collections::HashMap;

pub fn default_materials() -> Vec<MaterialInfo> {
    let m = |id, name: &str, density, hp_per_m3, min_hp, bond_strength, cell_size, structural| MaterialInfo {
        id,
        name: name.into(),
        density,
        hp_per_m3,
        min_hp,
        bond_strength,
        cell_size,
        structural,
    };
    vec![
        m(0, "concrete", 2400.0, 2000.0, 60.0, 2.0e6, 0.9, true),
        m(1, "brick", 1900.0, 1400.0, 40.0, 1.2e6, 0.6, true),
        m(2, "wood", 600.0, 900.0, 25.0, 6.0e5, 0.5, true),
        m(3, "metal", 7800.0, 6000.0, 150.0, 4.0e6, 1.0, true),
        m(4, "glass", 2500.0, 10.0, 1.0, 0.0, 10.0, false),
    ]
}

#[derive(Clone, Copy, Debug)]
pub struct SBox {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub material: u16,
    pub flags: u16,
    /// element id (0 = the default "other" element)
    pub elem: u32,
}

/// Builds a `Bld` from axis-aligned boxes. Edges are created between boxes sharing a face
/// (coplanar touching faces with overlap area >= `min_contact`).
pub struct SynthBuilder {
    pub name: String,
    pub boxes: Vec<SBox>,
    pub materials: Vec<MaterialInfo>,
    pub min_contact: f32,
    /// boxes whose min z <= this are anchors
    pub anchor_z: f32,
    /// element kinds (bgen ELEMENT_KINDS ids); element 0 is the default "other" (14)
    pub elem_kinds: Vec<u16>,
    /// element assigned to boxes added from now on (see [`SynthBuilder::begin_element`])
    pub cur_elem: u32,
}

impl SynthBuilder {
    pub fn new(name: &str) -> Self {
        SynthBuilder {
            name: name.into(),
            boxes: vec![],
            materials: default_materials(),
            min_contact: 0.01,
            anchor_z: 1e-4,
            elem_kinds: vec![14],
            cur_elem: 0,
        }
    }

    pub fn add(&mut self, min: [f32; 3], max: [f32; 3], material: u16, flags: u16) -> u32 {
        self.boxes.push(SBox { min, max, material, flags, elem: self.cur_elem });
        self.boxes.len() as u32 - 1
    }

    /// Start a new element (member) of the given kind (e.g. 2 floor, 3 column); boxes added
    /// until the next `begin_element`/`end_element` belong to it. Returns its id.
    pub fn begin_element(&mut self, kind: u16) -> u32 {
        self.elem_kinds.push(kind);
        self.cur_elem = (self.elem_kinds.len() - 1) as u32;
        self.cur_elem
    }

    /// Back to the default element 0.
    pub fn end_element(&mut self) {
        self.cur_elem = 0;
    }

    /// Fill a box region with a grid of `n` cells.
    pub fn add_grid(&mut self, min: [f32; 3], max: [f32; 3], n: [usize; 3], material: u16, flags: u16) {
        let d = [(max[0] - min[0]) / n[0] as f32, (max[1] - min[1]) / n[1] as f32, (max[2] - min[2]) / n[2] as f32];
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let lo = [min[0] + d[0] * i as f32, min[1] + d[1] * j as f32, min[2] + d[2] * k as f32];
                    self.add(lo, [lo[0] + d[0], lo[1] + d[1], lo[2] + d[2]], material, flags);
                }
            }
        }
    }

    pub fn build(&self) -> Bld {
        let n = self.boxes.len();
        let mut chunks = Vec::with_capacity(n);
        let mut hull_verts = Vec::with_capacity(n * 8);
        let mut hull_planes = Vec::with_capacity(n * 6);
        let mut anchors = vec![0u64; (n + 63) / 64];
        for (i, b) in self.boxes.iter().enumerate() {
            let mat = &self.materials[b.material as usize];
            let s = [b.max[0] - b.min[0], b.max[1] - b.min[1], b.max[2] - b.min[2]];
            let vol = s[0] * s[1] * s[2];
            let mass = vol * mat.density;
            let com = [(b.min[0] + b.max[0]) * 0.5, (b.min[1] + b.max[1]) * 0.5, (b.min[2] + b.max[2]) * 0.5];
            let k = mass / 12.0;
            let inertia =
                [k * (s[1] * s[1] + s[2] * s[2]), k * (s[0] * s[0] + s[2] * s[2]), k * (s[0] * s[0] + s[1] * s[1]), 0.0, 0.0, 0.0];
            let mut flags = b.flags;
            if b.min[2] <= self.anchor_z {
                flags |= F_ANCHOR;
                anchors[i / 64] |= 1 << (i % 64);
            }
            if mat.name == "glass" {
                flags |= F_GLASS;
            }
            let hv = hull_verts.len() as u32;
            for c in 0..8 {
                hull_verts.push([
                    if c & 1 == 0 { b.min[0] } else { b.max[0] },
                    if c & 2 == 0 { b.min[1] } else { b.max[1] },
                    if c & 4 == 0 { b.min[2] } else { b.max[2] },
                ]);
            }
            let hp_off = hull_planes.len() as u32;
            hull_planes.extend_from_slice(&[
                [-1.0, 0.0, 0.0, b.min[0]],
                [1.0, 0.0, 0.0, -b.max[0]],
                [0.0, -1.0, 0.0, b.min[1]],
                [0.0, 1.0, 0.0, -b.max[1]],
                [0.0, 0.0, -1.0, b.min[2]],
                [0.0, 0.0, 1.0, -b.max[2]],
            ]);
            chunks.push(ChunkRecord {
                elem: b.elem,
                material: b.material,
                flags,
                mass,
                volume: vol,
                hp: (mat.hp_per_m3 * vol).max(mat.min_hp),
                com,
                inertia,
                aabb_min: b.min,
                aabb_max: b.max,
                hull_v_off: hv,
                hull_v_cnt: 8,
                hull_p_off: hp_off,
                hull_p_cnt: 6,
                mesh_i_off: 0,
                mesh_i_cnt: 0,
            });
        }
        let edges = self.find_edges();
        let mut frame = [0.0f32; 16];
        frame[0] = 1.0;
        frame[5] = 1.0;
        frame[10] = 1.0;
        frame[15] = 1.0;
        let elements: Vec<ElemRecord> = self
            .elem_kinds
            .iter()
            .enumerate()
            .map(|(id, &kind)| {
                let mine: Vec<usize> = (0..n).filter(|&i| self.boxes[i].elem == id as u32).collect();
                ElemRecord {
                    id: id as u32,
                    kind,
                    material: 0,
                    floor: 0,
                    flags: 0,
                    first_chunk: mine.first().copied().unwrap_or(0) as u32,
                    chunk_count: mine.len() as u32,
                    frame,
                    thickness: 1.0,
                }
            })
            .collect();
        Bld {
            meta: serde_json::json!({"name": self.name, "materials": self.materials, "synthetic": true}),
            materials: self.materials.clone(),
            elements,
            chunks,
            hull_verts,
            hull_planes,
            mesh_verts: vec![],
            mesh_indices: vec![],
            edges,
            anchors,
            sockets: vec![],
            content_hash: [0; 32],
        }
    }

    fn find_edges(&self) -> Vec<EdgeRecord> {
        const EPS: f32 = 1e-4;
        let cell = 1.0f32;
        let key = |x: f32| (x / cell).floor() as i32;
        let mut grid: HashMap<(i32, i32, i32), Vec<u32>> = HashMap::new();
        for (i, b) in self.boxes.iter().enumerate() {
            for z in key(b.min[2] - EPS)..=key(b.max[2] + EPS) {
                for y in key(b.min[1] - EPS)..=key(b.max[1] + EPS) {
                    for x in key(b.min[0] - EPS)..=key(b.max[0] + EPS) {
                        grid.entry((x, y, z)).or_default().push(i as u32);
                    }
                }
            }
        }
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        for v in grid.values() {
            for (ii, &a) in v.iter().enumerate() {
                for &b in &v[ii + 1..] {
                    pairs.push((a.min(b), a.max(b)));
                }
            }
        }
        pairs.sort_unstable();
        pairs.dedup();
        let mut edges = Vec::new();
        for (a, b) in pairs {
            let (ba, bb) = (&self.boxes[a as usize], &self.boxes[b as usize]);
            for ax in 0..3 {
                let (o1, o2) = ((ax + 1) % 3, (ax + 2) % 3);
                let ov = |k: usize| (ba.max[k].min(bb.max[k]) - ba.min[k].max(bb.min[k])).max(0.0);
                let (l1, l2) = (ov(o1), ov(o2));
                let area = l1 * l2;
                if area < self.min_contact {
                    continue;
                }
                let sign = if (ba.max[ax] - bb.min[ax]).abs() < EPS {
                    1.0
                } else if (bb.max[ax] - ba.min[ax]).abs() < EPS {
                    -1.0
                } else {
                    continue;
                };
                let mut c = [0.0; 3];
                c[ax] = if sign > 0.0 { ba.max[ax] } else { ba.min[ax] };
                c[o1] = 0.5 * (ba.min[o1].max(bb.min[o1]) + ba.max[o1].min(bb.max[o1]));
                c[o2] = 0.5 * (ba.min[o2].max(bb.min[o2]) + ba.max[o2].min(bb.max[o2]));
                let mut nrm = [0.0; 3];
                nrm[ax] = sign;
                let bond = self.materials[ba.material as usize]
                    .bond_strength
                    .min(self.materials[bb.material as usize].bond_strength);
                edges.push(EdgeRecord { a, b, area, strength: (area * bond).max(1.0), centroid: c, normal: nrm });
                break;
            }
        }
        edges
    }
}

/// The two_box fixture rebuilt synthetically (4 x 1 m cubes; 0,1 anchored, 2,3 on top).
pub fn two_box() -> Bld {
    let mut s = SynthBuilder::new("two_box_synth");
    s.add_grid([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], [2, 1, 1], 0, 0);
    s.add_grid([0.0, 0.0, 1.0], [2.0, 1.0, 2.0], [2, 1, 1], 0, 0);
    s.build()
}

/// Solid block of `nx*ny*nz` cubes of edge `size`.
pub fn grid_block(nx: usize, ny: usize, nz: usize, size: f32) -> Bld {
    let mut s = SynthBuilder::new(&format!("grid_{nx}x{ny}x{nz}"));
    s.add_grid([0.0; 3], [nx as f32 * size, ny as f32 * size, nz as f32 * size], [nx, ny, nz], 0, 0);
    s.build()
}

/// Thin wall `w` x `h` m of `cell` m chunks, `t` thick, along X.
pub fn wall(w: f32, h: f32, t: f32, cell: f32) -> Bld {
    let mut s = SynthBuilder::new("wall");
    let nx = (w / cell).round().max(1.0) as usize;
    let nz = (h / cell).round().max(1.0) as usize;
    s.add_grid([0.0; 3], [w, t, h], [nx, 1, nz], 0, 0);
    s.build()
}

/// Tower: anchored ground slab, `floors` storeys of 4 corner columns (+ optional center
/// column) carrying fractured slabs. Footprint `side` x `side` m, storey height 3 m.
pub fn tower(floors: usize, side: f32, center_column: bool) -> Bld {
    tower_cols(floors, side, center_column, 0.6)
}

/// [`tower`] with a custom column cross-section `col` x `col` m.
pub fn tower_cols(floors: usize, side: f32, center_column: bool, col: f32) -> Bld {
    let mut s = SynthBuilder::new(&format!("tower_{floors}"));
    let slab_t = 0.3;
    let story = 3.0;
    let nslab = (side / 1.5).round().max(1.0) as usize;
    s.add_grid([0.0, 0.0, 0.0], [side, side, slab_t], [nslab, nslab, 1], 0, F_INDESTRUCTIBLE);
    for f in 0..floors {
        let z0 = slab_t + f as f32 * (story + slab_t);
        let z1 = z0 + story;
        let mut cols = vec![[0.0, 0.0], [side - col, 0.0], [0.0, side - col], [side - col, side - col]];
        if center_column {
            cols.push([(side - col) * 0.5, (side - col) * 0.5]);
        }
        for c in cols {
            s.add_grid([c[0], c[1], z0], [c[0] + col, c[1] + col, z1], [1, 1, 3], 0, 0);
        }
        s.add_grid([0.0, 0.0, z1], [side, side, z1 + slab_t], [nslab, nslab, 1], 0, 0);
    }
    s.build()
}

/// Anchored column with a horizontal cantilever beam of `len` cubes at height `h`.
pub fn cantilever(h: usize, len: usize) -> Bld {
    let mut s = SynthBuilder::new("cantilever");
    s.add_grid([0.0, 0.0, 0.0], [1.0, 1.0, h as f32], [1, 1, h], 0, 0);
    s.add_grid([1.0, 0.0, h as f32 - 1.0], [1.0 + len as f32, 1.0, h as f32], [len, 1, 1], 0, 0);
    s.build()
}

/// Arena of `n` blocks laid out on a grid with `gap` spacing; returns (bld, position) pairs.
pub fn arena(n: usize, make: impl Fn(usize) -> Bld, spacing: f32) -> Vec<(Bld, [f32; 3])> {
    let side = (n as f32).sqrt().ceil() as usize;
    (0..n).map(|i| (make(i), [(i % side) as f32 * spacing, (i / side) as f32 * spacing, 0.0])).collect()
}
