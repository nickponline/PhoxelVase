//! `.bld` building bundle: the contract between `bgen` (Python) and `rubble`.
//! Record layouts are `#[repr(C)]` and must match `bgen/bgen/export/bld.py` dtypes byte-for-byte.

use bytemuck::{Pod, Zeroable};
use std::path::Path;

pub const MAGIC: [u8; 4] = *b"BLD\0";
pub const BLD_VERSION: u32 = 1;
pub const HEADER_SIZE: usize = 64;
pub const SECTION_ENTRY_SIZE: usize = 32;

// chunk flags
pub const F_ANCHOR: u16 = 1;
pub const F_INDESTRUCTIBLE: u16 = 2;
pub const F_GLASS: u16 = 4;
pub const F_COSMETIC_ATTACHED: u16 = 8;
pub const F_NO_DEBRIS: u16 = 16;
// render-vertex face flags
pub const FACE_OUTER: u16 = 1;
pub const FACE_INNER: u16 = 2;
pub const FACE_CAP_EDGE: u16 = 4;

pub const MATERIALS: [&str; 5] = ["concrete", "brick", "wood", "metal", "glass"];
pub const ELEMENT_KINDS: [&str; 15] = [
    "ext_wall", "int_wall", "floor", "column", "beam", "roof", "stair", "landing", "step",
    "parapet", "balcony", "glass", "fascia", "ridge", "other",
];
pub const SOCKET_TYPES: [&str; 5] = ["balcony_rail", "window_frame", "door", "prop", "light"];

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct ElemRecord {
    pub id: u32,
    pub kind: u16,
    pub material: u16,
    pub floor: i16,
    pub flags: u16,
    pub first_chunk: u32,
    pub chunk_count: u32,
    /// row-major 4x4 local->building transform
    pub frame: [f32; 16],
    pub thickness: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct ChunkRecord {
    pub elem: u32,
    pub material: u16,
    pub flags: u16,
    pub mass: f32,
    pub volume: f32,
    pub hp: f32,
    pub com: [f32; 3],
    /// inertia tensor about com in building axes: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] (tensor entries)
    pub inertia: [f32; 6],
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub hull_v_off: u32,
    pub hull_v_cnt: u32,
    pub hull_p_off: u32,
    pub hull_p_cnt: u32,
    /// range into MIDX (indices are global into MVRT)
    pub mesh_i_off: u32,
    pub mesh_i_cnt: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    /// normal * 32767
    pub nrm: [i16; 3],
    pub face_flags: u16,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct EdgeRecord {
    pub a: u32,
    pub b: u32,
    pub area: f32,
    /// capacity in N
    pub strength: f32,
    pub centroid: [f32; 3],
    /// unit normal from a to b
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, PartialEq)]
pub struct SocketRecord {
    pub ty: u32,
    pub owner_elem: u32,
    pub xform: [f32; 16],
}

const _: () = {
    assert!(std::mem::size_of::<ElemRecord>() == 88);
    assert!(std::mem::size_of::<ChunkRecord>() == 104);
    assert!(std::mem::size_of::<MeshVertex>() == 20);
    assert!(std::mem::size_of::<EdgeRecord>() == 40);
    assert!(std::mem::size_of::<SocketRecord>() == 72);
};

#[derive(Debug, thiserror::Error)]
pub enum BldError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("bad magic")]
    BadMagic,
    #[error("unsupported version {0} (expected {BLD_VERSION})")]
    Version(u32),
    #[error("truncated file")]
    Truncated,
    #[error("missing section {0}")]
    Missing(&'static str),
    #[error("section {0}: bad stride {1}")]
    Stride(String, u32),
    #[error("content hash mismatch")]
    Hash,
    #[error("meta json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid: {0}")]
    Invalid(String),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct MaterialInfo {
    pub id: u32,
    pub name: String,
    pub density: f32,
    pub hp_per_m3: f32,
    pub min_hp: f32,
    pub bond_strength: f32,
    pub cell_size: f32,
    pub structural: bool,
}

/// Fully-owned, decoded building bundle.
#[derive(Clone, Debug)]
pub struct Bld {
    pub meta: serde_json::Value,
    pub materials: Vec<MaterialInfo>,
    pub elements: Vec<ElemRecord>,
    pub chunks: Vec<ChunkRecord>,
    pub hull_verts: Vec<[f32; 3]>,
    pub hull_planes: Vec<[f32; 4]>,
    pub mesh_verts: Vec<MeshVertex>,
    pub mesh_indices: Vec<u32>,
    pub edges: Vec<EdgeRecord>,
    pub anchors: Vec<u64>,
    pub sockets: Vec<SocketRecord>,
    pub content_hash: [u8; 32],
}

fn rd_u32(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn rd_u64(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

fn cast_vec<T: Pod>(data: &[u8]) -> Vec<T> {
    // copy into aligned storage (input slice alignment is only guaranteed to 16 within file)
    let n = data.len() / std::mem::size_of::<T>();
    let mut v = vec![T::zeroed(); n];
    bytemuck::cast_slice_mut::<T, u8>(&mut v).copy_from_slice(&data[..n * std::mem::size_of::<T>()]);
    v
}

impl Bld {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, BldError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self, BldError> {
        if raw.len() < HEADER_SIZE { return Err(BldError::Truncated); }
        if raw[0..4] != MAGIC { return Err(BldError::BadMagic); }
        let version = rd_u32(raw, 4);
        if version != BLD_VERSION { return Err(BldError::Version(version)); }
        let n = rd_u32(raw, 12) as usize;
        let mut content_hash = [0u8; 32];
        content_hash.copy_from_slice(&raw[16..48]);
        let mut hasher = blake3::Hasher::new();
        let mut secs: std::collections::HashMap<[u8; 4], (&[u8], u32, u32)> = Default::default();
        for i in 0..n {
            let o = HEADER_SIZE + i * SECTION_ENTRY_SIZE;
            if raw.len() < o + SECTION_ENTRY_SIZE { return Err(BldError::Truncated); }
            let tag: [u8; 4] = raw[o..o + 4].try_into().unwrap();
            let count = rd_u32(raw, o + 4);
            let stride = rd_u32(raw, o + 8);
            let off = rd_u64(raw, o + 16) as usize;
            let size = rd_u64(raw, o + 24) as usize;
            if raw.len() < off + size { return Err(BldError::Truncated); }
            let data = &raw[off..off + size];
            hasher.update(data);
            secs.insert(tag, (data, count, stride));
        }
        if *hasher.finalize().as_bytes() != content_hash { return Err(BldError::Hash); }

        fn sec<'a, T: Pod>(
            secs: &std::collections::HashMap<[u8; 4], (&'a [u8], u32, u32)>, tag: &'static str,
        ) -> Result<Vec<T>, BldError> {
            let key: [u8; 4] = tag.as_bytes().try_into().unwrap();
            let (data, count, stride) = secs.get(&key).ok_or(BldError::Missing(tag))?;
            if *stride as usize != std::mem::size_of::<T>() {
                return Err(BldError::Stride(tag.into(), *stride));
            }
            if data.len() != *count as usize * *stride as usize {
                return Err(BldError::Truncated);
            }
            Ok(cast_vec(data))
        }

        let meta_raw = secs.get(b"META").ok_or(BldError::Missing("META"))?.0;
        let meta: serde_json::Value = serde_json::from_slice(meta_raw)?;
        let materials: Vec<MaterialInfo> = match meta.get("materials") {
            Some(m) => serde_json::from_value(m.clone())?,
            None => Vec::new(),
        };
        let bld = Bld {
            meta,
            materials,
            elements: sec(&secs, "ELEM")?,
            chunks: sec(&secs, "CHNK")?,
            hull_verts: sec(&secs, "HVRT")?,
            hull_planes: sec(&secs, "HPLN")?,
            mesh_verts: sec(&secs, "MVRT")?,
            mesh_indices: sec(&secs, "MIDX")?,
            edges: sec(&secs, "EDGE")?,
            anchors: sec(&secs, "ANCH")?,
            sockets: sec(&secs, "SOCK")?,
            content_hash,
        };
        bld.validate()?;
        Ok(bld)
    }

    /// Structural sanity checks on ranges and indices.
    pub fn validate(&self) -> Result<(), BldError> {
        let nc = self.chunks.len() as u32;
        for (i, c) in self.chunks.iter().enumerate() {
            let bad = |s: &str| Err(BldError::Invalid(format!("chunk {i}: {s}")));
            if (c.hull_v_off + c.hull_v_cnt) as usize > self.hull_verts.len() { return bad("hull verts range"); }
            if (c.hull_p_off + c.hull_p_cnt) as usize > self.hull_planes.len() { return bad("hull planes range"); }
            if (c.mesh_i_off + c.mesh_i_cnt) as usize > self.mesh_indices.len() { return bad("mesh range"); }
            if c.elem as usize >= self.elements.len() { return bad("elem index"); }
            if !(c.mass > 0.0) { return bad("mass"); }
        }
        if self.mesh_indices.iter().any(|&i| i as usize >= self.mesh_verts.len()) {
            return Err(BldError::Invalid("mesh index out of range".into()));
        }
        for (i, e) in self.edges.iter().enumerate() {
            if e.a >= nc || e.b >= nc || e.a >= e.b {
                return Err(BldError::Invalid(format!("edge {i}: bad endpoints {} {}", e.a, e.b)));
            }
        }
        if self.anchors.len() != (self.chunks.len() + 63) / 64 {
            return Err(BldError::Invalid("anchor bitset length".into()));
        }
        Ok(())
    }

    pub fn is_anchor(&self, chunk: usize) -> bool {
        (self.anchors[chunk / 64] >> (chunk % 64)) & 1 == 1
    }
    pub fn hull_verts_of(&self, chunk: usize) -> &[[f32; 3]] {
        let c = &self.chunks[chunk];
        &self.hull_verts[c.hull_v_off as usize..(c.hull_v_off + c.hull_v_cnt) as usize]
    }
    pub fn hull_planes_of(&self, chunk: usize) -> &[[f32; 4]] {
        let c = &self.chunks[chunk];
        &self.hull_planes[c.hull_p_off as usize..(c.hull_p_off + c.hull_p_cnt) as usize]
    }
    pub fn mesh_indices_of(&self, chunk: usize) -> &[u32] {
        let c = &self.chunks[chunk];
        &self.mesh_indices[c.mesh_i_off as usize..(c.mesh_i_off + c.mesh_i_cnt) as usize]
    }
    pub fn name(&self) -> &str {
        self.meta.get("name").and_then(|v| v.as_str()).unwrap_or("building")
    }
}
