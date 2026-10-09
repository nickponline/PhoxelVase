use alloc::{boxed::Box, vec::Vec};

use crate::bounding_volume::BoundingVolume;
use crate::math::{Pose, Real, Vector};
use crate::query::contact_manifolds::contact_manifolds_halfspace_pfm::contact_manifold_halfspace_far;
use crate::shape::ShapeType;
use crate::query::contact_manifolds::contact_manifolds_workspace::{
    TypedWorkspaceData, WorkspaceData,
};
use crate::query::contact_manifolds::ContactManifoldsWorkspace;
use crate::query::query_dispatcher::PersistentQueryDispatcher;
use crate::query::ContactManifold;
use crate::shape::{CompositeShape, Shape};
use crate::utils::hashmap::{Entry, HashMap};
use crate::utils::PoseOpt;

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Deserialize, rkyv::Serialize)
)]
#[derive(Clone)]
struct SubDetector {
    manifold_id: usize,
    timestamp: bool,
}

#[cfg_attr(feature = "serde-serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Default)]
pub struct CompositeShapeShapeContactManifoldsWorkspace {
    timestamp: bool,
    sub_detectors: HashMap<u32, SubDetector>,
    /// The leaves of the last query, in order (`sub_detectors[last_leaves[i]].manifold_id == i`
    /// and no other key is present).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    last_leaves: Vec<u32>,
    /// Every sub-detector's timestamp is logically `timestamp`, whatever is stored (set when a
    /// query reached exactly the same leaves as the last one; see `REUSE_SAME_LEAVES`).
    #[cfg_attr(feature = "serde-serialize", serde(skip))]
    stamps_current: bool,
}

/// When a query reaches exactly the leaves of the previous query, in the same order (a resting
/// composite, or any composite against a half-space whose AABB is all of space), the
/// bookkeeping keeps every sub-detector with the same manifold id: skip it and reuse the
/// manifolds in place. (Off with serialization, which stores the per-detector timestamps.)
#[cfg(feature = "parallel")]
const REUSE_SAME_LEAVES: bool = cfg!(not(feature = "serde-serialize"));

impl CompositeShapeShapeContactManifoldsWorkspace {
    pub fn new() -> Self {
        Self::default()
    }
}

fn ensure_workspace_exists(workspace: &mut Option<ContactManifoldsWorkspace>) {
    if workspace
        .as_ref()
        .and_then(|w| {
            w.0.downcast_ref::<CompositeShapeShapeContactManifoldsWorkspace>()
        })
        .is_some()
    {
        return;
    }

    *workspace = Some(ContactManifoldsWorkspace(Box::new(
        CompositeShapeShapeContactManifoldsWorkspace::new(),
    )));
}

/// Leaves reached by the query from which the sub-shape manifolds are computed in parallel
/// (`parallel` feature).
#[cfg(feature = "parallel")]
const PARALLEL_LEAVES: usize = 256;

/// Computes the contact manifolds between a composite shape and an abstract shape.
///
/// The manifolds are in the order of the composite's leaves reached by the query. Under the
/// `parallel` feature, a query reaching many leaves computes their manifolds across threads
/// (same manifolds, same order).
pub fn contact_manifolds_composite_shape_shape<ManifoldData, ContactData>(
    dispatcher: &dyn PersistentQueryDispatcher<ManifoldData, ContactData>,
    pos12: &Pose,
    composite1: &(dyn CompositeShape + Sync),
    shape2: &dyn Shape,
    prediction: Real,
    manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
    workspace: &mut Option<ContactManifoldsWorkspace>,
    flipped: bool,
) where
    ManifoldData: Default + Clone + Send + Sync,
    ContactData: Default + Copy + Send + Sync,
{
    ensure_workspace_exists(workspace);
    let workspace: &mut CompositeShapeShapeContactManifoldsWorkspace =
        workspace.as_mut().unwrap().0.downcast_mut().unwrap();
    let new_timestamp = !workspace.timestamp;
    workspace.timestamp = new_timestamp;

    /*
     * Compute interferences.
     */

    let pos12 = *pos12;
    let pos21 = pos12.inverse();
    let deformable = composite1.is_deformable();

    // Traverse bvh1 first.
    let ls_aabb2_1 = shape2.compute_aabb(&pos12).loosened(prediction);
    let mut old_manifolds = core::mem::take(manifolds);

    #[cfg(feature = "parallel")]
    let leaves: Vec<u32> = composite1.bvh().intersect_aabb(&ls_aabb2_1).collect();
    #[cfg(feature = "parallel")]
    let reuse = REUSE_SAME_LEAVES && !deformable && !leaves.is_empty() && leaves == workspace.last_leaves;
    #[cfg(not(feature = "parallel"))]
    let reuse = false;
    if reuse {
        // exactly what the bookkeeping below would produce: every leaf keeps its manifold, at
        // the same index, and every sub-detector is refreshed
        *manifolds = core::mem::take(&mut old_manifolds);
        workspace.stamps_current = true;
    } else if workspace.stamps_current {
        // materialize the timestamps the skipped bookkeeping would have written
        workspace.stamps_current = false;
        for detector in workspace.sub_detectors.values_mut() {
            detector.timestamp = !new_timestamp;
        }
    }

    // The manifold of a leaf: the one kept from the last query, or a fresh one (its sub-shape
    // ids and pose set by the first computation), pushed in leaf order.
    let mut bookkeep =
        |leaf1: u32, manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>| {
            match workspace.sub_detectors.entry(leaf1) {
                Entry::Occupied(entry) => {
                    let sub_detector = entry.into_mut();
                    let mut manifold = old_manifolds[sub_detector.manifold_id].take();
                    sub_detector.manifold_id = manifolds.len();
                    sub_detector.timestamp = new_timestamp;
                    if deformable {
                        manifold.mark_shapes_deformed();
                    }
                    manifolds.push(manifold);
                    false
                }
                Entry::Vacant(entry) => {
                    let _ = entry.insert(SubDetector {
                        manifold_id: manifolds.len(),
                        timestamp: new_timestamp,
                    });
                    let mut manifold = ContactManifold::new();
                    if flipped {
                        manifold.subshape1 = 0;
                        manifold.subshape2 = leaf1;
                    } else {
                        manifold.subshape1 = leaf1;
                        manifold.subshape2 = 0;
                    }
                    if deformable {
                        manifold.mark_shapes_deformed();
                    }
                    manifolds.push(manifold);
                    true
                }
            }
        };
    // A half-space's AABB is all of space, so every part reaches the narrow phase; parts that
    // are provably beyond `prediction` from the plane get the (empty) manifold the general path
    // would produce without computing support features.
    let halfspace2 = shape2.as_halfspace();
    let far_from_halfspace = |leaf1: u32| -> bool {
        let (Some(hs), Some(node)) = (halfspace2, composite1.bvh().leaf_node(leaf1)) else {
            return false;
        };
        let aabb = node.aabb();
        let (lo, hi) = (aabb.mins, aabb.maxs);
        let mut min_d = Real::MAX;
        let mut max_abs: Real = 0.0;
        for i in 0..8 {
            let c = Vector::new(
                if i & 1 == 0 { lo.x } else { hi.x },
                if i & 2 == 0 { lo.y } else { hi.y },
                if i & 4 == 0 { lo.z } else { hi.z },
            );
            let p = pos21 * c;
            min_d = min_d.min(p.dot(hs.normal));
            max_abs = max_abs.max(p.abs().max_element());
        }
        // generous margin over the rounding of the exact per-vertex test
        min_d.is_finite() && min_d > prediction + 1.0e-2 + 1.0e-4 * max_abs
    };
    // The manifold's sub-shape pose (a fresh manifold) and contacts.
    let compute =
        |leaf1: u32, fresh: bool, manifold: &mut ContactManifold<ManifoldData, ContactData>| {
            composite1.map_part_at(leaf1, &mut |part_pos1, part_shape1, normal_constraints1| {
                if fresh {
                    if flipped {
                        manifold.set_subshape_pos2(part_pos1.copied());
                    } else {
                        manifold.set_subshape_pos1(part_pos1.copied());
                    }
                }
                if let Some(hs) = halfspace2 {
                    // same arguments as the default dispatcher's half-space arms
                    if part_shape1.shape_type() == ShapeType::ConvexPolyhedron
                        && dispatcher.is_default()
                        && far_from_halfspace(leaf1)
                    {
                        if flipped {
                            contact_manifold_halfspace_far(&part_pos1.prepend_to(&pos21), hs, manifold, false);
                        } else {
                            contact_manifold_halfspace_far(&part_pos1.inv_mul(&pos12).inverse(), hs, manifold, true);
                        }
                        return;
                    }
                }
                if flipped {
                    let _ = dispatcher.contact_manifold_convex_convex(
                        &part_pos1.prepend_to(&pos21),
                        shape2,
                        part_shape1,
                        None,
                        normal_constraints1,
                        prediction,
                        manifold,
                    );
                } else {
                    let _ = dispatcher.contact_manifold_convex_convex(
                        &part_pos1.inv_mul(&pos12),
                        part_shape1,
                        shape2,
                        normal_constraints1,
                        None,
                        prediction,
                        manifold,
                    );
                }
            });
        };

    #[cfg(feature = "parallel")]
    if reuse {
        if leaves.len() >= PARALLEL_LEAVES {
            use rayon::prelude::*;
            manifolds
                .par_iter_mut()
                .zip(leaves.par_iter())
                .for_each(|(manifold, &leaf1)| compute(leaf1, false, manifold));
        } else {
            for (manifold, &leaf1) in manifolds.iter_mut().zip(&leaves) {
                compute(leaf1, false, manifold);
            }
        }
    } else {
        if leaves.len() >= PARALLEL_LEAVES {
            use rayon::prelude::*;
            let fresh: Vec<bool> = leaves
                .iter()
                .map(|&leaf1| bookkeep(leaf1, manifolds))
                .collect();
            manifolds
                .par_iter_mut()
                .zip(leaves.par_iter())
                .zip(fresh.par_iter())
                .for_each(|((manifold, &leaf1), &fresh)| compute(leaf1, fresh, manifold));
        } else {
            for &leaf1 in &leaves {
                let fresh = bookkeep(leaf1, manifolds);
                compute(leaf1, fresh, manifolds.last_mut().unwrap());
            }
        }
    }
    #[cfg(not(feature = "parallel"))]
    for leaf1 in composite1.bvh().intersect_aabb(&ls_aabb2_1) {
        let fresh = bookkeep(leaf1, manifolds);
        compute(leaf1, fresh, manifolds.last_mut().unwrap());
    }

    if !reuse {
        workspace
            .sub_detectors
            .retain(|_, detector| detector.timestamp == new_timestamp);
    }
    #[cfg(feature = "parallel")]
    {
        workspace.last_leaves = leaves;
    }
}

impl WorkspaceData for CompositeShapeShapeContactManifoldsWorkspace {
    fn as_typed_workspace_data(&self) -> TypedWorkspaceData<'_> {
        TypedWorkspaceData::CompositeShapeShapeContactManifoldsWorkspace(self)
    }

    fn clone_dyn(&self) -> Box<dyn WorkspaceData> {
        Box::new(self.clone())
    }
}
