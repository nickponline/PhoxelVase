//! X: controlled demolition. A plane of charges near the bottom of a building cuts clean through
//! that level, the charges going off in random order. The cut is a wedge in height: a tall band on the fall side,
//! a thin slice on the far side.
//!
//! The building mostly drops straight down: the engine keeps it static until its last support
//! goes (nothing can hinge while standing), and the freed top breaks up on its first impact
//! before it can pivot on the stubs. Toppling needs engine-side hinging.
use rubble_core::{rubble_format::F_INDESTRUCTIBLE, Explosion, Vec3 as EVec3, World as EngineWorld};

/// grid spacing of the charges (m)
const SPACING: f32 = 3.0;
/// height of the cut's base above the building's lowest chunk (m)
const HEIGHT: f32 = 1.0;
/// height of the cut above its base on the fall side (m); the far side cuts just the chunks at
/// the base
const WEDGE: f32 = 2.5;
/// the charges go off at random times within this window (s)
const SPREAD: f32 = 0.8;
/// blast radius (m) of the charges (visual + push; the cut itself is exact)
const BLAST: f32 = 3.0;
/// blast radii vary by ± this fraction
const SIZE_JITTER: f32 = 0.2;
/// chance that a charge is a big one, twice the radius
const BIG_CHANCE: f32 = 0.15;

pub struct Charge {
    /// seconds after the trigger
    pub delay: f32,
    pub blast: Explosion,
    /// chunks of the building this charge cuts
    pub chunks: Vec<u32>,
}

/// Charges for building `b`, with the fall side along the building axis nearest the
/// engine-space horizontal direction `fall`. `seed` varies timings and blast sizes.
pub fn charges(w: &EngineWorld, b: usize, fall: [f32; 2], seed: u32) -> Vec<Charge> {
    let bd = &w.buildings[b];
    if bd.bld.chunks.is_empty() {
        return vec![];
    }
    // building-space AABB
    let (mut lo, mut hi) = (EVec3::splat(f32::MAX), EVec3::splat(f32::MIN));
    for c in &bd.bld.chunks {
        lo = lo.min(EVec3::from_array(c.aabb_min));
        hi = hi.max(EVec3::from_array(c.aabb_max));
    }
    // fall along the building axis nearest the requested direction, so the far side is a whole
    // wall line rather than one corner
    let d = bd.pose.rotation.inverse() * EVec3::new(fall[0], fall[1], 0.0);
    let dir = if d.x.abs() >= d.y.abs() { EVec3::X * d.x.signum() } else { EVec3::Y * d.y.signum() };
    let mid = (lo + hi) * 0.5;
    // extent of the footprint along the fall direction
    let half = ((hi.x - lo.x) * dir.x.abs() + (hi.y - lo.y) * dir.y.abs()) * 0.5;
    // 0 on the fall side .. 1 on the far side
    let side = |p: EVec3| if half > 1e-3 { (0.5 - (p - mid).dot(dir) / (2.0 * half)).clamp(0.0, 1.0) } else { 0.0 };
    let base = (lo.z + HEIGHT).min(mid.z);
    let n = |a: f32, b: f32| ((b - a) / SPACING).ceil().max(1.0) as usize;
    let (nx, ny) = (n(lo.x, hi.x), n(lo.y, hi.y));
    let mut rng = seed.wrapping_mul(0x9E37_79B9) | 1;
    let mut rand = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng as f32 / u32::MAX as f32
    };
    let mut out = vec![];
    for i in 0..=nx {
        for j in 0..=ny {
            let x = lo.x + (hi.x - lo.x) * i as f32 / nx as f32;
            let y = lo.y + (hi.y - lo.y) * j as f32 / ny as f32;
            let p = EVec3::new(x, y, base);
            let big = if rand() < BIG_CHANCE { 2.0 } else { 1.0 };
            let radius = BLAST * big * (1.0 + SIZE_JITTER * (2.0 * rand() - 1.0));
            out.push(Charge {
                delay: SPREAD * rand(),
                blast: Explosion {
                    center: bd.pose.transform_point(p).to_array(),
                    radius,
                    inner_radius: radius * 0.3,
                    damage: 300.0,
                    impulse: 2000.0 * radius,
                },
                chunks: vec![],
            });
        }
    }
    // the wedge: each chunk in it goes with the nearest charge
    for (c, ch) in bd.bld.chunks.iter().enumerate() {
        if ch.flags & F_INDESTRUCTIBLE != 0 {
            continue;
        }
        let com = EVec3::from_array(ch.com);
        let top = base + WEDGE * (1.0 - side(com));
        if ch.aabb_min[2] > top || ch.aabb_max[2] < base {
            continue;
        }
        let at = |k: &Charge| {
            let p = bd.pose.inverse_transform_point(EVec3::from_array(k.blast.center));
            (p.x - com.x).powi(2) + (p.y - com.y).powi(2)
        };
        if let Some(k) = out.iter_mut().min_by(|a, b| at(a).total_cmp(&at(b))) {
            k.chunks.push(c as u32);
        }
    }
    out
}

/// bgen element kinds (`ELEMENT_KINDS`) used to find levels
const KIND_FLOOR: u16 = 2;
const KIND_ROOF: u16 = 5;
/// Z (interior demolition): charges per level (floor slab or roof), inclusive range
const INTERIOR_PER_LEVEL: (u32, u32) = (2, 3);
/// height above the slab (m)
const INTERIOR_HEIGHT: f32 = 1.2;
/// they go off at random times within this window (s)
const INTERIOR_SPREAD: f32 = 1.5;
const INTERIOR_BLAST: f32 = 4.5;
/// slab tops closer than this (m) are one level
const LEVEL_GAP: f32 = 1.0;
/// buildings without floor elements: one level per this much height (m)
const LEVEL_FALLBACK: f32 = 3.5;

/// Z: a few explosions inside building `b` on every floor and on the roof, at random spots
/// above the slabs, going off at random times. Unlike `charges` nothing is cut: the blasts do
/// the damage and the engine decides what comes down.
pub fn interior_charges(w: &EngineWorld, b: usize, seed: u32) -> Vec<Charge> {
    let bd = &w.buildings[b];
    let bld = &bd.bld;
    let kind = |c: &rubble_core::rubble_format::ChunkRecord| bld.elements.get(c.elem as usize).map_or(u16::MAX, |e| e.kind);
    // levels: groups of floor-slab chunks with about the same top; the roof is one more.
    // Indestructible slabs count too: the ground floor is often a foundation slab.
    let mut slabs: Vec<(f32, u32)> =
        (0..bld.chunks.len()).filter(|&c| kind(&bld.chunks[c]) == KIND_FLOOR).map(|c| (bld.chunks[c].aabb_max[2], c as u32)).collect();
    if slabs.is_empty() {
        // no floor elements (bridges, towers of beams): bands of height over everything
        slabs = (0..bld.chunks.len())
            .map(|c| ((bld.chunks[c].aabb_max[2] / LEVEL_FALLBACK).floor() * LEVEL_FALLBACK, c as u32))
            .collect();
    }
    slabs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut levels: Vec<Vec<u32>> = vec![];
    let mut last = f32::NEG_INFINITY;
    for (z, c) in slabs {
        if z - last > LEVEL_GAP || levels.is_empty() {
            levels.push(vec![]);
        }
        last = z;
        levels.last_mut().unwrap().push(c);
    }
    let roof: Vec<u32> = (0..bld.chunks.len()).filter(|&c| kind(&bld.chunks[c]) == KIND_ROOF).map(|c| c as u32).collect();
    if !roof.is_empty() {
        levels.push(roof);
    }
    let mut rng = seed.wrapping_mul(0x9E37_79B9) ^ 0x7F4A_7C15 | 1;
    let mut rand = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        rng as f32 / u32::MAX as f32
    };
    let mut out = vec![];
    for level in &levels {
        let (lo, hi) = INTERIOR_PER_LEVEL;
        let n = lo + ((hi - lo + 1) as f32 * rand()) as u32;
        for _ in 0..n.min(hi) {
            // above a random chunk of the level, so the charge is over the slab (inside)
            let ch = &bld.chunks[level[((level.len() as f32 * rand()) as usize).min(level.len() - 1)] as usize];
            let p = EVec3::new(ch.com[0], ch.com[1], ch.aabb_max[2] + INTERIOR_HEIGHT);
            let big = if rand() < BIG_CHANCE { 1.6 } else { 1.0 };
            let radius = INTERIOR_BLAST * big * (1.0 + SIZE_JITTER * (2.0 * rand() - 1.0));
            out.push(Charge {
                delay: INTERIOR_SPREAD * rand(),
                blast: Explosion {
                    center: bd.pose.transform_point(p).to_array(),
                    radius,
                    inner_radius: radius * 0.4,
                    damage: 5000.0,
                    impulse: 5000.0 * radius,
                },
                chunks: vec![],
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rubble_core::testutil::tower;
    use rubble_core::{BuildingId, ChunkState, Isometry, WorldConfig};

    /// The charge plane is cut clean through and the upper floors come down without a pause,
    /// whichever side the fall side is.
    /// Z: charges on every floor and the roof, all inside the building's footprint.
    #[test]
    fn interior_charges_on_every_level() {
        let mut w = EngineWorld::new(WorldConfig::default());
        w.load_building_bld(tower(6, 9.0, true), Isometry::new([5.0, -3.0, 0.0], 0.3));
        let ks = interior_charges(&w, 0, 7);
        let bd = &w.buildings[0];
        let (mut lo, mut hi) = (EVec3::splat(f32::MAX), EVec3::splat(f32::MIN));
        for c in &bd.bld.chunks {
            lo = lo.min(EVec3::from_array(c.aabb_min));
            hi = hi.max(EVec3::from_array(c.aabb_max));
        }
        let local: Vec<EVec3> = ks.iter().map(|k| bd.pose.inverse_transform_point(EVec3::from_array(k.blast.center))).collect();
        for p in &local {
            assert!(p.x >= lo.x && p.x <= hi.x && p.y >= lo.y && p.y <= hi.y, "charge {p:?} outside the footprint");
            assert!(p.z <= hi.z + INTERIOR_HEIGHT + 0.01, "charge {p:?} above the building");
        }
        // one band of charges per storey (3 m) from the ground up to the top
        let mut bands: Vec<i32> = local.iter().map(|p| (p.z / 3.0).floor() as i32).collect();
        bands.sort();
        bands.dedup();
        assert!(bands.len() >= 6, "levels with charges: {bands:?}");
        assert!(ks.iter().all(|k| k.delay >= 0.0 && k.delay <= INTERIOR_SPREAD));
    }

    #[test]
    fn tower_cut_through_and_comes_down() {
        let yaw = 0.4f32;
        // requested world direction (along the tower's own axes, so no snapping)
        for (k, a) in [0.0f32, 90.0, 180.0, 270.0].into_iter().enumerate() {
            let a = yaw + a.to_radians();
            demolish_tower(yaw, [a.cos(), a.sin()], k as u32);
        }
    }

    fn demolish_tower(yaw: f32, fall: [f32; 2], seed: u32) {
        let mut w = EngineWorld::new(WorldConfig::default());
        w.load_building_bld(tower(8, 9.0, true), Isometry::new([0.0, 0.0, 0.0], yaw));
        w.add_ground_plane(0.0);
        w.step(1.0 / 60.0);
        let upper: Vec<usize> = (0..w.buildings[0].n_chunks()).filter(|&c| w.chunk_world_com(0, c).z > 15.0).collect();
        // gone chunks count where they were destroyed, so shattering doesn't bias the mean
        let mean = |w: &EngineWorld| upper.iter().fold(EVec3::ZERO, |a, &c| a + w.chunk_world_com(0, c)) / upper.len() as f32;
        let before = mean(&w);
        let lo_z = (0..w.buildings[0].n_chunks()).map(|c| w.buildings[0].bld.chunks[c].aabb_min[2]).fold(f32::MAX, f32::min);
        let plane = lo_z + HEIGHT;
        let cut: Vec<usize> = (0..w.buildings[0].n_chunks())
            .filter(|&c| {
                let ch = &w.buildings[0].bld.chunks[c];
                ch.aabb_min[2] < plane && ch.aabb_max[2] > plane && ch.flags & F_INDESTRUCTIBLE == 0
            })
            .collect();
        assert!(!cut.is_empty());
        let mut queued = charges(&w, 0, fall, seed);
        assert!(!queued.is_empty());
        let last = queued.iter().map(|k| k.delay).fold(0.0, f32::max);
        let t0 = w.time;
        // when the top starts to fall
        let mut falling = None;
        for _ in 0..(8 * 60) {
            queued.retain(|k| {
                let due = t0 + k.delay <= w.time + 1e-6;
                if due {
                    w.explode(k.blast);
                    for &c in &k.chunks {
                        w.damage_chunk(BuildingId(0), c, 1e9);
                    }
                }
                !due
            });
            w.step(1.0 / 60.0);
            w.hurry_collapse(BuildingId(0));
            if w.clusters.values().any(|c| c.chunks.len() > 100) {
                falling.get_or_insert(w.time - t0);
            }
            if w.time - t0 > SPREAD + 0.2 {
                for &c in &cut {
                    assert!(w.buildings[0].state[c] == ChunkState::Gone, "fall {fall:?}: chunk {c} on the charge plane survived");
                }
            }
        }
        // no collapse-warning pause: the top is free by the time the last charge has gone off
        let falling = falling.expect("the top never came free");
        assert!(falling <= last + 0.1, "fall {fall:?}: top fell at {falling:.2} s, last charge at {last:.2} s");
        let shift = mean(&w) - before;
        assert!(shift.z < -8.0, "fall {fall:?}: upper floors should come down: {shift:?}");
    }
}
