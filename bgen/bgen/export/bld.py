"""`.bld` binary bundle writer/reader. Layouts must match rubble-format exactly."""
from __future__ import annotations

import json
import struct
from pathlib import Path

import blake3
import numpy as np

from ..materials import (DEFAULT_MATERIALS, KIND_ID, MATERIAL_ID, SOCKET_ID, F_ANCHOR,
                         materials_json)
from ..model import BuildingContext, BuildingData, ChunkGeom, Edge

MAGIC = b"BLD\0"
BLD_VERSION = 1
HEADER_SIZE = 64
SECTION_ENTRY_SIZE = 32
ALIGN = 16

ELEM_DTYPE = np.dtype([
    ("id", "<u4"), ("kind", "<u2"), ("material", "<u2"), ("floor", "<i2"), ("flags", "<u2"),
    ("first_chunk", "<u4"), ("chunk_count", "<u4"), ("frame", "<f4", (16,)), ("thickness", "<f4"),
])
CHNK_DTYPE = np.dtype([
    ("elem", "<u4"), ("material", "<u2"), ("flags", "<u2"), ("mass", "<f4"), ("volume", "<f4"),
    ("hp", "<f4"), ("com", "<f4", (3,)), ("inertia", "<f4", (6,)), ("aabb_min", "<f4", (3,)),
    ("aabb_max", "<f4", (3,)), ("hull_v_off", "<u4"), ("hull_v_cnt", "<u4"), ("hull_p_off", "<u4"),
    ("hull_p_cnt", "<u4"), ("mesh_i_off", "<u4"), ("mesh_i_cnt", "<u4"),
])
MVRT_DTYPE = np.dtype([("pos", "<f4", (3,)), ("nrm", "<i2", (3,)), ("face_flags", "<u2")])
EDGE_DTYPE = np.dtype([
    ("a", "<u4"), ("b", "<u4"), ("area", "<f4"), ("strength", "<f4"),
    ("centroid", "<f4", (3,)), ("normal", "<f4", (3,)),
])
SOCK_DTYPE = np.dtype([("type", "<u4"), ("owner_elem", "<u4"), ("xform", "<f4", (16,))])

assert ELEM_DTYPE.itemsize == 88 and CHNK_DTYPE.itemsize == 104 and MVRT_DTYPE.itemsize == 20
assert EDGE_DTYPE.itemsize == 40 and SOCK_DTYPE.itemsize == 72

SECTION_ORDER = ["META", "ELEM", "CHNK", "HVRT", "HPLN", "MVRT", "MIDX", "EDGE", "ANCH", "SOCK"]


def _sections(bd: BuildingData) -> list[tuple[str, bytes, int, int]]:
    meta = json.dumps(bd.meta, separators=(",", ":"), default=_json_default).encode()
    hv = np.ascontiguousarray(bd.hull_verts, dtype="<f4").reshape(-1, 3)
    hp = np.ascontiguousarray(bd.hull_planes, dtype="<f4").reshape(-1, 4)
    mi = np.ascontiguousarray(bd.mesh_indices, dtype="<u4").reshape(-1)
    an = np.ascontiguousarray(bd.anchors, dtype="<u8").reshape(-1)
    return [
        ("META", meta, len(meta), 1),
        ("ELEM", bd.elements.astype(ELEM_DTYPE).tobytes(), len(bd.elements), ELEM_DTYPE.itemsize),
        ("CHNK", bd.chunks.astype(CHNK_DTYPE).tobytes(), len(bd.chunks), CHNK_DTYPE.itemsize),
        ("HVRT", hv.tobytes(), len(hv), 12),
        ("HPLN", hp.tobytes(), len(hp), 16),
        ("MVRT", bd.mesh_verts.astype(MVRT_DTYPE).tobytes(), len(bd.mesh_verts), MVRT_DTYPE.itemsize),
        ("MIDX", mi.tobytes(), len(mi), 4),
        ("EDGE", bd.edges.astype(EDGE_DTYPE).tobytes(), len(bd.edges), EDGE_DTYPE.itemsize),
        ("ANCH", an.tobytes(), len(an), 8),
        ("SOCK", bd.sockets.astype(SOCK_DTYPE).tobytes(), len(bd.sockets), SOCK_DTYPE.itemsize),
    ]


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    raise TypeError(type(o))


def _pad(n: int) -> int:
    return (ALIGN - n % ALIGN) % ALIGN


def write_bld(bd: BuildingData, path: str | Path) -> str:
    """Write bundle; returns hex content hash."""
    secs = _sections(bd)
    h = blake3.blake3()
    for _, data, _, _ in secs:
        h.update(data)
    digest = h.digest()
    table_size = SECTION_ENTRY_SIZE * len(secs)
    off = HEADER_SIZE + table_size
    off += _pad(off)
    entries, blobs = [], []
    for tag, data, count, stride in secs:
        entries.append(struct.pack("<4sIIIQQ", tag.encode(), count, stride, 0, off, len(data)))
        blobs.append(data + b"\0" * _pad(len(data)))
        off += len(data) + _pad(len(data))
    header = struct.pack("<4sIII32s16s", MAGIC, BLD_VERSION, 0, len(secs), digest, b"\0" * 16)
    assert len(header) == HEADER_SIZE
    out = bytearray(header)
    out += b"".join(entries)
    out += b"\0" * _pad(len(out))
    for b in blobs:
        out += b
    Path(path).write_bytes(bytes(out))
    return digest.hex()


def read_bld(path: str | Path) -> BuildingData:
    raw = Path(path).read_bytes()
    magic, version, _flags, n, digest, _ = struct.unpack_from("<4sIII32s16s", raw, 0)
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != BLD_VERSION:
        raise ValueError(f"version {version} != {BLD_VERSION}")
    secs = {}
    h = blake3.blake3()
    for i in range(n):
        tag, count, stride, _, off, size = struct.unpack_from("<4sIIIQQ", raw, HEADER_SIZE + i * SECTION_ENTRY_SIZE)
        data = raw[off:off + size]
        h.update(data)
        secs[tag.decode()] = (data, count, stride)
    if h.digest() != digest:
        raise ValueError("content hash mismatch")
    f = lambda t, dt: np.frombuffer(secs[t][0], dtype=dt).copy()
    return BuildingData(
        meta=json.loads(secs["META"][0].decode()),
        elements=f("ELEM", ELEM_DTYPE), chunks=f("CHNK", CHNK_DTYPE),
        hull_verts=f("HVRT", "<f4").reshape(-1, 3), hull_planes=f("HPLN", "<f4").reshape(-1, 4),
        mesh_verts=f("MVRT", MVRT_DTYPE), mesh_indices=f("MIDX", "<u4"),
        edges=f("EDGE", EDGE_DTYPE), anchors=f("ANCH", "<u8"), sockets=f("SOCK", SOCK_DTYPE),
    )


def anchors_to_bitset(mask: np.ndarray) -> np.ndarray:
    n = len(mask)
    words = np.zeros((n + 63) // 64, dtype=np.uint64)
    idx = np.nonzero(mask)[0]
    np.bitwise_or.at(words, idx // 64, (np.uint64(1) << (idx % 64).astype(np.uint64)))
    return words


def bitset_to_mask(words: np.ndarray, n: int) -> np.ndarray:
    i = np.arange(n)
    return ((words[i // 64] >> (i % 64).astype(np.uint64)) & np.uint64(1)).astype(bool)


def assemble(ctx: BuildingContext, chunks: list[ChunkGeom], edges: list[Edge],
             extra_meta: dict | None = None) -> BuildingData:
    """Flatten panels + chunks + edges into BuildingData. `chunks` must be grouped by panel_id
    in ascending panel order (ELEM.first_chunk/chunk_count ranges rely on it)."""
    nC = len(chunks)
    panel_ids = [c.panel_id for c in chunks]
    assert panel_ids == sorted(panel_ids), "chunks must be grouped by panel id"

    elements = np.zeros(len(ctx.panels), ELEM_DTYPE)
    counts = np.bincount(np.array(panel_ids, dtype=np.int64), minlength=len(ctx.panels)) if nC else np.zeros(len(ctx.panels), int)
    firsts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(ctx.panels) else np.zeros(0, int)
    for p in ctx.panels:
        e = elements[p.id]
        e["id"] = p.id
        e["kind"] = KIND_ID.get(p.kind, KIND_ID["other"])
        e["material"] = MATERIAL_ID[p.material]
        e["floor"] = int(p.tags.get("floor", -1))
        e["flags"] = (1 if p.tags.get("anchor") else 0) | (2 if p.tags.get("indestructible") else 0) \
            | (8 if p.tags.get("cosmetic") else 0)
        e["first_chunk"] = firsts[p.id]
        e["chunk_count"] = counts[p.id]
        e["frame"] = np.asarray(p.frame, dtype=np.float32).reshape(16)  # row-major
        e["thickness"] = p.thickness

    ch = np.zeros(nC, CHNK_DTYPE)
    hv, hp, mv, mi = [], [], [], []
    hv_off = hp_off = mv_off = mi_off = 0
    for i, c in enumerate(chunks):
        r = ch[i]
        r["elem"] = c.panel_id
        r["material"] = MATERIAL_ID[c.material]
        r["flags"] = c.flags
        r["mass"], r["volume"], r["hp"] = c.mass, c.volume, c.hp
        r["com"], r["inertia"] = c.com, c.inertia
        r["aabb_min"], r["aabb_max"] = c.hull_verts.min(0), c.hull_verts.max(0)
        r["hull_v_off"], r["hull_v_cnt"] = hv_off, len(c.hull_verts)
        r["hull_p_off"], r["hull_p_cnt"] = hp_off, len(c.planes)
        r["mesh_i_off"], r["mesh_i_cnt"] = mi_off, c.mesh_tris.size
        hv.append(c.hull_verts); hp.append(c.planes)
        v = np.zeros(len(c.mesh_verts), MVRT_DTYPE)
        v["pos"] = c.mesh_verts
        v["nrm"] = np.clip(np.round(c.mesh_normals * 32767), -32767, 32767).astype(np.int16)
        v["face_flags"] = c.mesh_face_flags
        mv.append(v)
        mi.append(c.mesh_tris.reshape(-1).astype(np.uint32) + mv_off)  # global vertex index
        hv_off += len(c.hull_verts); hp_off += len(c.planes)
        mv_off += len(c.mesh_verts); mi_off += c.mesh_tris.size

    ed = np.zeros(len(edges), EDGE_DTYPE)
    for i, e in enumerate(edges):
        a, b, n = (e.a, e.b, e.normal) if e.a < e.b else (e.b, e.a, -np.asarray(e.normal))
        ed[i] = (a, b, e.area, e.strength, e.centroid, n)
    if len(ed):
        ed = ed[np.lexsort((ed["b"], ed["a"]))]

    anchor_mask = np.array([bool(c.flags & F_ANCHOR) for c in chunks], dtype=bool)
    sk = np.zeros(len(ctx.sockets), SOCK_DTYPE)
    for i, s in enumerate(ctx.sockets):
        sk[i] = (SOCKET_ID[s.type], s.owner_panel if s.owner_panel >= 0 else 0xFFFFFFFF,
                 np.asarray(s.xform, dtype=np.float32).reshape(16))

    allv = np.concatenate(hv) if hv else np.zeros((0, 3))
    meta = {
        "name": ctx.spec.get("name", "building"),
        "seed": ctx.spec.get("seed", 0),
        "units": "m", "up": "z",
        "bld_version": BLD_VERSION,
        "bounds": [allv.min(0).tolist(), allv.max(0).tolist()] if len(allv) else [[0, 0, 0], [0, 0, 0]],
        "materials": materials_json(DEFAULT_MATERIALS),
        "params": ctx.params,
        "rooms": [{"id": r.id, "floor": r.floor, "tag": r.tag,
                   "polygon": [list(map(float, p)) for p in r.polygon.exterior.coords]} for r in ctx.rooms],
        "spec": ctx.spec,
    }
    if extra_meta:
        meta.update(extra_meta)
    return BuildingData(
        meta=meta, elements=elements, chunks=ch,
        hull_verts=allv.astype(np.float32),
        hull_planes=(np.concatenate(hp) if hp else np.zeros((0, 4))).astype(np.float32),
        mesh_verts=np.concatenate(mv) if mv else np.zeros(0, MVRT_DTYPE),
        mesh_indices=np.concatenate(mi) if mi else np.zeros(0, np.uint32),
        edges=ed, anchors=anchors_to_bitset(anchor_mask), sockets=sk,
    )
