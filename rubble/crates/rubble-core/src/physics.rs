//! Rigid-body backend boundary. The destruction layer only talks to
//! [`PhysicsBackend`]; [`RapierBackend`] is the current implementation.
//!
//! Conventions: every dynamic body's local frame equals the *building frame* of the
//! building its chunks came from, so chunk hulls (stored in building space) are used as
//! compound sub-shapes with identity sub-poses, and a chunk's world transform is simply
//! the body pose.

use crate::math::{Mat3, Pose, Vec3};
use rapier3d::parry::query::ShapeCastOptions;
use rapier3d::parry::shape::SharedShape;
use rapier3d::prelude::*;

pub type Shape = SharedShape;

/// Collision group of light debris (see `PhysicsBackend::set_light_debris`).
const LIGHT_DEBRIS: Group = Group::GROUP_2;

/// Parts of a dynamic compound (identity sub-poses) with per-part data the caller has cached:
/// `aabbs[i] == shapes[i].compute_aabb(&Pose::IDENTITY)`, `ccd[i] == shapes[i].ccd_thickness()`.
pub struct CompoundParts {
    pub shapes: Vec<Shape>,
    pub aabbs: Vec<Aabb>,
    pub ccd: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BodyId(pub(crate) RigidBodyHandle);
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColliderId(pub(crate) ColliderHandle);

/// Mass properties in the body's local frame (= building frame).
#[derive(Clone, Copy, Debug)]
pub struct MassProps {
    pub mass: f32,
    pub local_com: Vec3,
    /// inertia tensor about `local_com`, body axes
    pub inertia: Mat3,
}

#[derive(Clone, Copy, Debug)]
pub struct BodyState {
    pub pose: Pose,
    pub linvel: Vec3,
    pub angvel: Vec3,
    pub world_com: Vec3,
    pub sleeping: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RayHit {
    pub collider: ColliderId,
    pub tag: u128,
    pub toi: f32,
    pub normal: Vec3,
}

/// One solver contact point with its applied impulse (N·s) for the last step.
#[derive(Clone, Copy, Debug)]
pub struct ContactImpulse {
    pub tag1: u128,
    pub tag2: u128,
    /// compound sub-shape index on each side when known
    pub sub1: Option<u32>,
    pub sub2: Option<u32>,
    pub point: Vec3,
    pub normal: Vec3,
    pub impulse: f32,
}

pub trait PhysicsBackend {
    fn set_gravity(&mut self, g: Vec3);
    /// Continuous collision detection for fast dynamic bodies (expensive against many static hulls).
    fn set_ccd(&mut self, enabled: bool);
    /// Continuous collision detection for one body (only runs while `set_ccd(true)`).
    fn set_body_ccd(&mut self, b: BodyId, enabled: bool);
    /// Predictive ("soft") collision detection for one body: contacts are looked for this far
    /// ahead along its path (m), 0 = off. Much cheaper than `set_body_ccd`.
    fn set_body_soft_ccd(&mut self, b: BodyId, distance: f32);
    fn set_solver_iterations(&mut self, iters: usize);
    fn convex_hull(points: &[Vec3]) -> Option<Shape>
    where
        Self: Sized;
    /// Returns the collider and its world AABB (what `collider_aabb` would return).
    fn add_static_collider(&mut self, shape: Shape, pose: Pose, tag: u128) -> (ColliderId, (Vec3, Vec3));
    fn add_ground_plane(&mut self, z: f32, tag: u128) -> ColliderId;
    fn remove_collider(&mut self, c: ColliderId);
    fn add_dynamic_compound(
        &mut self, pose: Pose, parts: CompoundParts, mass: &MassProps, linvel: Vec3, angvel: Vec3, tag: u128,
    ) -> (BodyId, ColliderId);
    /// Light debris: the collider still hits everything else, but not other light debris
    /// (piles of small bits are the expensive part of a collapse). Kept across `set_dynamic_compound`.
    fn set_light_debris(&mut self, col: ColliderId, on: bool);
    /// Replace the body's compound shape and mass; returns the (possibly new) collider id.
    fn set_dynamic_compound(&mut self, body: BodyId, col: ColliderId, parts: CompoundParts, mass: &MassProps) -> ColliderId;
    fn remove_body(&mut self, b: BodyId);
    fn body_state(&self, b: BodyId) -> BodyState;
    fn set_body_velocity(&mut self, b: BodyId, linvel: Vec3, angvel: Vec3);
    fn apply_impulse_at_point(&mut self, b: BodyId, j: Vec3, p: Vec3);
    fn step(&mut self, dt: f32);
    /// Refresh acceleration structures so colliders inserted since the last step are queryable.
    fn sync_queries(&mut self);
    fn cast_ray(&self, origin: Vec3, dir: Vec3, max_toi: f32, exclude: Option<ColliderId>) -> Option<RayHit>;
    /// All colliders crossed by the segment, unordered (at most `max` collected).
    fn ray_all(&self, origin: Vec3, dir: Vec3, max_toi: f32, max: usize, out: &mut Vec<RayHit>);
    /// Sweep a sphere from `origin` along `delta` (toi in [0,1]).
    fn cast_sphere(&self, origin: Vec3, delta: Vec3, radius: f32) -> Option<RayHit>;
    /// Colliders whose (broad-phase) AABB intersects the box.
    fn query_aabb(&self, min: Vec3, max: Vec3, out: &mut Vec<(ColliderId, u128)>);
    fn contacts(&self, min_impulse: f32, out: &mut Vec<ContactImpulse>);
    /// World AABB of a collider (None if it no longer exists).
    fn collider_aabb(&self, c: ColliderId) -> Option<(Vec3, Vec3)>;
    /// World AABB of all colliders of a body (None if it no longer exists).
    fn body_aabb(&self, b: BodyId) -> Option<(Vec3, Vec3)>;
    /// Wake every dynamic body with a collider whose AABB intersects the box. Removing a
    /// support never wakes what sleeps on it, so callers do this after removing geometry.
    fn wake_bodies_in_aabb(&mut self, min: Vec3, max: Vec3);
    /// Does any collider found by `query_aabb(min, max)` belong to a sleeping dynamic body
    /// (when `sleeping`), or have a tag accepted by `tag_pred`?
    fn aabb_touches(&self, min: Vec3, max: Vec3, sleeping: bool, tag_pred: &dyn Fn(u128) -> bool) -> bool;
    /// `wake_bodies_in_aabb` followed by `query_aabb` on the same box, in one traversal.
    fn wake_and_query_aabb(&mut self, min: Vec3, max: Vec3, out: &mut Vec<(ColliderId, u128)>);
    /// True if the body has an active contact with another dynamic body that is still moving
    /// (awake and faster than the given rest thresholds).
    fn touches_moving_dynamic(&self, b: BodyId, rest_lin: f32, rest_ang: f32) -> bool;
    /// user_data of every collider that `shape` placed at `pose` intersects.
    fn shape_overlaps(&self, shape: &Shape, pose: Pose, out: &mut Vec<u128>);
    fn num_bodies(&self) -> usize;
    fn num_colliders(&self) -> usize;
}

pub struct RapierBackend {
    pub world: PhysicsWorld,
    /// rayon pool the step runs in (None: the caller's / global pool)
    pool: Option<&'static rayon::ThreadPool>,
    scratch_parents: Vec<RigidBodyHandle>,
}

impl RapierBackend {
    pub fn new(gravity: Vec3) -> Self {
        let mut world = PhysicsWorld::default();
        world.gravity = gravity;
        // keep per-sub-shape manifolds so impacts map to chunks without a spatial search
        world.integration_parameters.contact_clustering = false;
        RapierBackend { world, pool: None, scratch_parents: Vec::new() }
    }
}

fn hit_from(world: &PhysicsWorld, h: ColliderHandle, toi: f32, normal: Vec3) -> RayHit {
    RayHit { collider: ColliderId(h), tag: world.colliders[h].user_data, toi, normal }
}

impl RapierBackend {
    /// Run the rigid-body step on a dedicated pool of `n` threads (0 = auto, see
    /// `WorldConfig::physics_threads`). Results are identical for any thread count.
    pub fn set_threads(&mut self, n: usize) {
        let n = if n > 0 {
            n
        } else if let Some(e) = std::env::var("RUBBLE_PHYSICS_THREADS").ok().and_then(|v| v.parse::<usize>().ok()) {
            e
        } else {
            let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
            (cores / 2).clamp(1, 6)
        };
        self.pool = Some(physics_pool(n));
    }
}

impl RapierBackend {
    /// Kinematic Z-up capsule walker (the viewer's FPS mode; not part of the backend trait, it
    /// never touches the simulation). `centre` is the capsule centre, `half_height` the half
    /// length of its segment. Slides along walls, climbs stairs (autostep + slopes up to 50°)
    /// and snaps down them; light debris is ignored so rubble cannot trap the walker.
    /// Returns (translation actually applied, grounded afterwards).
    pub fn move_capsule(&self, centre: Vec3, half_height: f32, radius: f32, desired: Vec3, dt: f32) -> (Vec3, bool) {
        use rapier3d::control::{CharacterAutostep, CharacterLength, KinematicCharacterController};
        let kcc = KinematicCharacterController {
            up: Vec3::Z,
            offset: CharacterLength::Absolute(0.02),
            autostep: Some(CharacterAutostep {
                max_height: CharacterLength::Absolute(0.3),
                min_width: CharacterLength::Absolute(0.1),
                include_dynamic_bodies: true,
            }),
            max_slope_climb_angle: 50f32.to_radians(),
            min_slope_slide_angle: 55f32.to_radians(),
            snap_to_ground: Some(CharacterLength::Absolute(0.3)),
            ..Default::default()
        };
        let shape = Capsule::new_z(half_height, radius);
        let filter = QueryFilter::default().groups(InteractionGroups::new(Group::GROUP_1, !LIGHT_DEBRIS, InteractionTestMode::And));
        let qp = self.world.query_pipeline_with_filter(filter);
        let m = kcc.move_shape(dt, &qp, &shape, &Pose::from_translation(centre), desired, |_| {});
        (m.translation, m.grounded)
    }
}

impl PhysicsBackend for RapierBackend {
    fn set_gravity(&mut self, g: Vec3) {
        self.world.gravity = g;
    }

    fn set_ccd(&mut self, enabled: bool) {
        self.world.integration_parameters.max_ccd_substeps = enabled as usize;
    }

    fn set_body_ccd(&mut self, b: BodyId, enabled: bool) {
        if let Some(rb) = self.world.bodies.get_mut(b.0) {
            rb.enable_ccd(enabled);
        }
    }

    fn set_body_soft_ccd(&mut self, b: BodyId, distance: f32) {
        if let Some(rb) = self.world.bodies.get_mut(b.0) {
            rb.set_soft_ccd_prediction(distance);
        }
    }

    fn set_solver_iterations(&mut self, iters: usize) {
        self.world.integration_parameters.num_solver_iterations = iters.max(1);
    }

    fn convex_hull(points: &[Vec3]) -> Option<Shape> {
        SharedShape::convex_hull(points)
    }

    fn add_static_collider(&mut self, shape: Shape, pose: Pose, tag: u128) -> (ColliderId, (Vec3, Vec3)) {
        // Parent-less colliders behave as fixed geometry and are O(1) to remove
        // (a fixed body with thousands of children has O(n) child removal).
        let c = ColliderBuilder::new(shape).position(pose).user_data(tag).friction(0.8).density(0.0).build();
        let aabb = c.compute_aabb();
        let h = self.world.colliders.insert(c);
        // make it visible to scene queries immediately (before the next step)
        self.world.broad_phase.set_aabb(&self.world.integration_parameters, h, aabb);
        (ColliderId(h), (aabb.mins, aabb.maxs))
    }

    fn add_ground_plane(&mut self, z: f32, tag: u128) -> ColliderId {
        let c = ColliderBuilder::new(SharedShape::halfspace(Vec3::Z))
            .translation(Vec3::new(0.0, 0.0, z))
            .user_data(tag)
            .friction(0.9)
            .build();
        let aabb = c.compute_aabb();
        let h = self.world.colliders.insert(c);
        self.world.broad_phase.set_aabb(&self.world.integration_parameters, h, aabb);
        ColliderId(h)
    }

    fn remove_collider(&mut self, c: ColliderId) {
        let w = &mut self.world;
        if let Some(col) = w.colliders.remove(c.0, &mut w.islands, &mut w.bodies, &mut w.soft_bodies, true) {
            drop_in_background(col);
        }
    }

    fn add_dynamic_compound(
        &mut self, pose: Pose, parts: CompoundParts, mass: &MassProps, linvel: Vec3, angvel: Vec3, tag: u128,
    ) -> (BodyId, ColliderId) {
        let mp = MassProperties::with_inertia_matrix(mass.local_com, mass.mass, mass.inertia);
        let rb = RigidBodyBuilder::dynamic()
            .pose(pose)
            .additional_mass_properties(mp)
            .linvel(linvel)
            .angvel(angvel)
            .user_data(tag)
            .build();
        let shape = compound_of(parts);
        let col = ColliderBuilder::new(shape).density(0.0).friction(0.8).restitution(0.0).user_data(tag).build();
        let (b, c) = self.world.insert(rb, col);
        (BodyId(b), ColliderId(c))
    }

    fn set_dynamic_compound(&mut self, body: BodyId, col: ColliderId, parts: CompoundParts, mass: &MassProps) -> ColliderId {
        // NOTE: `Collider::set_shape` on a collider with live contacts trips rapier 0.36's
        // solver-graph validation (stale contact entries), so swap the collider instead.
        let (tag, groups) =
            self.world.colliders.get(col.0).map(|c| (c.user_data, c.collision_groups())).unwrap_or((0, InteractionGroups::all()));
        self.remove_collider(col);
        let c = ColliderBuilder::new(compound_of(parts))
            .density(0.0)
            .friction(0.8)
            .restitution(0.0)
            .user_data(tag)
            .collision_groups(groups)
            .build();
        let h = self.world.insert_collider(c, Some(body.0));
        if let Some(b) = self.world.bodies.get_mut(body.0) {
            let mp = MassProperties::with_inertia_matrix(mass.local_com, mass.mass, mass.inertia);
            b.set_additional_mass_properties(mp, true);
        }
        ColliderId(h)
    }

    fn set_light_debris(&mut self, col: ColliderId, on: bool) {
        if let Some(c) = self.world.colliders.get_mut(col.0) {
            c.set_collision_groups(if on {
                InteractionGroups::new(LIGHT_DEBRIS, !LIGHT_DEBRIS, InteractionTestMode::And)
            } else {
                InteractionGroups::all()
            });
        }
    }

    fn remove_body(&mut self, b: BodyId) {
        self.world.remove_body(b.0);
    }

    fn body_state(&self, b: BodyId) -> BodyState {
        let rb = &self.world.bodies[b.0];
        BodyState {
            pose: *rb.position(),
            linvel: rb.linvel(),
            angvel: rb.angvel(),
            world_com: rb.center_of_mass(),
            sleeping: rb.is_sleeping(),
        }
    }

    fn set_body_velocity(&mut self, b: BodyId, linvel: Vec3, angvel: Vec3) {
        if let Some(rb) = self.world.bodies.get_mut(b.0) {
            rb.set_linvel(linvel, true);
            rb.set_angvel(angvel, true);
        }
    }

    fn apply_impulse_at_point(&mut self, b: BodyId, j: Vec3, p: Vec3) {
        if let Some(rb) = self.world.bodies.get_mut(b.0) {
            rb.apply_impulse_at_point(j, p, true);
        }
    }

    fn step(&mut self, dt: f32) {
        self.world.integration_parameters.dt = dt;
        // Handing the step to the pool costs a cross-thread round trip (~20 µs): not worth it
        // when nothing moves (results do not depend on where the step runs).
        let idle = self.world.islands.num_active_bodies() == 0;
        match self.pool {
            Some(pool) if !idle => pool.install(|| self.world.step()),
            _ => self.world.step(),
        }
    }

    fn sync_queries(&mut self) {
        // Static colliders are pushed into the BVH on insertion (`set_aabb`). Running the
        // collision pipeline between physics steps corrupts rapier 0.36's incremental
        // solver-graph bookkeeping, so this is intentionally a no-op.
    }

    fn cast_ray(&self, origin: Vec3, dir: Vec3, max_toi: f32, exclude: Option<ColliderId>) -> Option<RayHit> {
        let ray = Ray::new(origin, dir);
        let mut f = QueryFilter::default();
        if let Some(e) = exclude {
            f = f.exclude_collider(e.0);
        }
        self.world
            .cast_ray_and_get_normal(&ray, max_toi, true, f)
            .map(|(h, i)| hit_from(&self.world, h, i.time_of_impact, i.normal))
    }

    fn ray_all(&self, origin: Vec3, dir: Vec3, max_toi: f32, max: usize, out: &mut Vec<RayHit>) {
        let ray = Ray::new(origin, dir);
        for (h, c, i) in self.world.intersect_ray(ray, max_toi, true, QueryFilter::default()) {
            out.push(RayHit { collider: ColliderId(h), tag: c.user_data, toi: i.time_of_impact, normal: i.normal });
            if out.len() >= max {
                break;
            }
        }
    }

    fn cast_sphere(&self, origin: Vec3, delta: Vec3, radius: f32) -> Option<RayHit> {
        let ball = Ball::new(radius);
        let pose = Pose::from_translation(origin);
        self.world
            .cast_shape(&pose, delta, &ball, ShapeCastOptions::with_max_time_of_impact(1.0), QueryFilter::default())
            .map(|(h, hit)| {
                let n = self.world.colliders[h].position().transform_vector(hit.normal2);
                hit_from(&self.world, h, hit.time_of_impact, n)
            })
    }

    fn query_aabb(&self, min: Vec3, max: Vec3, out: &mut Vec<(ColliderId, u128)>) {
        let aabb = Aabb::new(min, max);
        for (h, c) in self.world.intersect_aabb_conservative(aabb, QueryFilter::default()) {
            out.push((ColliderId(h), c.user_data));
        }
    }

    fn contacts(&self, min_impulse: f32, out: &mut Vec<ContactImpulse>) {
        let w = &self.world;
        for pair in w.narrow_phase.contact_pairs() {
            let Some(rigid) = pair.rigid() else { continue };
            let clustered = !rigid.solver_clusters.is_empty();
            let manifolds: &[ContactManifold] = if clustered { &rigid.solver_clusters } else { &rigid.manifolds };
            // most pairs carry no impulse above the threshold: skip them before any lookup
            if !manifolds.iter().any(|m| m.points.iter().any(|p| p.data.impulse > min_impulse)) {
                continue;
            }
            let (Some(c1), Some(c2)) = (w.colliders.get(pair.collider1), w.colliders.get(pair.collider2)) else {
                continue;
            };
            for m in manifolds {
                let n = c1.position().transform_vector(m.local_n1);
                for p in &m.points {
                    if p.data.impulse <= min_impulse {
                        continue;
                    }
                    out.push(ContactImpulse {
                        tag1: c1.user_data,
                        tag2: c2.user_data,
                        sub1: (!clustered).then_some(m.subshape1),
                        sub2: (!clustered).then_some(m.subshape2),
                        point: c1.position().transform_point(p.local_p1),
                        normal: n,
                        impulse: p.data.impulse,
                    });
                }
            }
        }
    }

    fn collider_aabb(&self, c: ColliderId) -> Option<(Vec3, Vec3)> {
        let a = self.world.colliders.get(c.0)?.compute_aabb();
        Some((a.mins, a.maxs))
    }

    fn body_aabb(&self, b: BodyId) -> Option<(Vec3, Vec3)> {
        let rb = self.world.bodies.get(b.0)?;
        let mut out: Option<(Vec3, Vec3)> = None;
        for h in rb.colliders() {
            if let Some(c) = self.world.colliders.get(*h) {
                let a = c.compute_aabb();
                out = Some(match out {
                    None => (a.mins, a.maxs),
                    Some((lo, hi)) => (lo.min(a.mins), hi.max(a.maxs)),
                });
            }
        }
        out
    }

    fn wake_bodies_in_aabb(&mut self, min: Vec3, max: Vec3) {
        let aabb = Aabb::new(min, max);
        let parents: Vec<RigidBodyHandle> = self
            .world
            .intersect_aabb_conservative(aabb, QueryFilter::default())
            .filter_map(|(_, c)| c.parent())
            .collect();
        for p in parents {
            if let Some(rb) = self.world.bodies.get_mut(p) {
                if rb.is_dynamic() && rb.is_sleeping() {
                    rb.wake_up(true);
                }
            }
        }
    }

    fn aabb_touches(&self, min: Vec3, max: Vec3, sleeping: bool, tag_pred: &dyn Fn(u128) -> bool) -> bool {
        let w = &self.world;
        w.intersect_aabb_conservative(Aabb::new(min, max), QueryFilter::default()).any(|(_, c)| {
            tag_pred(c.user_data)
                || (sleeping
                    && c.parent().and_then(|p| w.bodies.get(p)).is_some_and(|rb| rb.is_dynamic() && rb.is_sleeping()))
        })
    }

    fn wake_and_query_aabb(&mut self, min: Vec3, max: Vec3, out: &mut Vec<(ColliderId, u128)>) {
        let aabb = Aabb::new(min, max);
        let mut parents = std::mem::take(&mut self.scratch_parents);
        parents.clear();
        for (h, c) in self.world.intersect_aabb_conservative(aabb, QueryFilter::default()) {
            if let Some(p) = c.parent() {
                parents.push(p);
            }
            out.push((ColliderId(h), c.user_data));
        }
        for &p in &parents {
            if let Some(rb) = self.world.bodies.get_mut(p) {
                if rb.is_dynamic() && rb.is_sleeping() {
                    rb.wake_up(true);
                }
            }
        }
        self.scratch_parents = parents;
    }

    fn touches_moving_dynamic(&self, b: BodyId, rest_lin: f32, rest_ang: f32) -> bool {
        let w = &self.world;
        let Some(rb) = w.bodies.get(b.0) else { return false };
        for &h in rb.colliders() {
            for pair in w.narrow_phase.contact_pairs_with(h) {
                if !pair.has_any_active_contact() {
                    continue;
                }
                let other = if pair.collider1 == h { pair.collider2 } else { pair.collider1 };
                let dynamic = w
                    .colliders
                    .get(other)
                    .and_then(|c| c.parent())
                    .and_then(|p| w.bodies.get(p))
                    .is_some_and(|o| {
                        o.is_dynamic()
                            && !o.is_sleeping()
                            && (o.linvel().length() >= rest_lin || o.angvel().length() >= rest_ang)
                    });
                if dynamic {
                    return true;
                }
            }
        }
        false
    }

    fn shape_overlaps(&self, shape: &Shape, pose: Pose, out: &mut Vec<u128>) {
        out.extend(self.world.intersect_shape(pose, &**shape, QueryFilter::default()).map(|(_, c)| c.user_data));
    }

    fn num_bodies(&self) -> usize {
        self.world.bodies.len()
    }
    fn num_colliders(&self) -> usize {
        self.world.colliders.len()
    }
}

fn compound_of(mut parts: CompoundParts) -> SharedShape {
    debug_assert!(parts.shapes.len() == parts.aabbs.len() && parts.shapes.len() == parts.ccd.len());
    if parts.shapes.len() == 1 {
        return parts.shapes.pop().unwrap();
    }
    // same compound as `SharedShape::compound` (identity sub-poses), minus recomputing the
    // per-part AABBs and CCD thicknesses (`Compound::ccd_thickness` folds the same min)
    let ccd = parts.ccd.iter().fold(f32::MAX, |c, &t| c.min(t));
    let shapes = parts.shapes.into_iter().map(|s| (Pose::IDENTITY, s)).collect();
    SharedShape::new(rapier3d::parry::shape::Compound::new_with_part_data(shapes, parts.aabbs, Some(ccd)))
}

/// Process-wide rayon pool with `n` threads (created on first use, shared by all worlds).
fn physics_pool(n: usize) -> &'static rayon::ThreadPool {
    static POOLS: std::sync::Mutex<Vec<(usize, &'static rayon::ThreadPool)>> = std::sync::Mutex::new(Vec::new());
    let mut pools = POOLS.lock().unwrap();
    if let Some(&(_, p)) = pools.iter().find(|(k, _)| *k == n) {
        return p;
    }
    let p: &'static rayon::ThreadPool = Box::leak(Box::new(
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .thread_name(|i| format!("rubble-physics-{i}"))
            .build()
            .expect("physics thread pool"),
    ));
    pools.push((n, p));
    p
}

/// Compounds at least this big are dropped on a background thread.
const BACKGROUND_DROP_PARTS: usize = 1024;

/// Dropping a big compound releases one `Arc` per part (cache-missing atomic decrements, and
/// the part shapes are still shared with the building, so nothing is actually freed but the
/// compound's own buffers). Hand big ones to a background thread instead of blocking the tick.
fn drop_in_background(col: Collider) {
    let big = col.shape().as_compound().is_some_and(|c| c.shapes().len() >= BACKGROUND_DROP_PARTS);
    if !big {
        return;
    }
    static DROPPER: std::sync::OnceLock<std::sync::Mutex<std::sync::mpsc::Sender<Collider>>> = std::sync::OnceLock::new();
    let tx = DROPPER.get_or_init(|| {
        let (tx, rx) = std::sync::mpsc::channel::<Collider>();
        std::thread::Builder::new()
            .name("rubble-dropper".into())
            .spawn(move || {
                for c in rx {
                    drop(c);
                }
            })
            .expect("spawn dropper thread");
        std::sync::Mutex::new(tx)
    });
    if let Err(e) = tx.lock().unwrap().send(col) {
        drop(e.0);
    }
}
