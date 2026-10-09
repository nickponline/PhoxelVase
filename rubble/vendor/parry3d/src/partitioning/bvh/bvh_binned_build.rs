use super::bvh_tree::{BvhBuildStrategy, BvhNodeIndex, BvhNodeWide};
use super::{Bvh, BvhNode, BvhWorkspace};
use crate::bounding_volume::{Aabb, BoundingVolume};
use crate::math::{Real, VectorExt};

impl Bvh {
    /// Fully rebuilds this BVH using the given strategy.
    ///
    /// This rebuilds a BVH with the same leaves, but different intermediate nodes and depth using
    /// the specified building strategy.
    pub fn rebuild(&mut self, workspace: &mut BvhWorkspace, strategy: BvhBuildStrategy) {
        if self.nodes.len() < 2 {
            // Nothing to rebuild if the tree is empty or only contains the root.
            // This takes care of the case where we have a partial root too.
            return;
        }

        workspace.rebuild_leaves.clear();
        for node in self.nodes.iter() {
            if node.left.is_leaf() {
                workspace.rebuild_leaves.push(node.left);
            }
            if node.right.is_leaf() {
                workspace.rebuild_leaves.push(node.right);
            }
        }

        self.nodes.clear();
        self.parents.clear();
        self.free_wide_nodes.clear();
        self.nodes.push(BvhNodeWide::zeros());
        self.parents.push(BvhNodeIndex::default());

        match strategy {
            BvhBuildStrategy::Binned => self.rebuild_range_binned(0, &mut workspace.rebuild_leaves),
            BvhBuildStrategy::Ploc => self.rebuild_range_ploc(0, &mut workspace.rebuild_leaves),
        }
    }

    /// Bins `leaves` (len > 1), picks the SAH split plane and partitions them in place.
    /// Returns the size of the left part. (Split out of `rebuild_range_binned` unchanged, so
    /// the sequential and the parallel builders produce the same tree.)
    fn binned_split(leaves: &mut [BvhNode]) -> usize {
        // PERF: calculate an optimal bin count dynamically based on the number of leaves to split?
        //       The paper suggests (4 + 2 * sqrt(num_leaves).floor()).min(16)
        const NUM_BINS: usize = 8;
        const BIN_EPSILON: Real = 1.0e-5;

        let mut bins = [BvhBin::default(); NUM_BINS];

        assert!(leaves.len() > 1);

        let centroid_aabb = Aabb::from_points(leaves.iter().map(|node| node.center()));
        let bins_axis = centroid_aabb.extents().max_position();
        let bins_range = [
            centroid_aabb.mins.vget(bins_axis),
            centroid_aabb.maxs.vget(bins_axis),
        ];

        // Compute bins characteristics.
        let k1 = NUM_BINS as Real * (1.0 - BIN_EPSILON) / (bins_range[1] - bins_range[0]);
        let k0 = bins_range[0];
        // NOTE: the clamp guards against degenerate leaf AABBs, whose non-finite or very
        //       large coordinates would push the bin index outside [0, NUM_BINS - 1]
        //       (`as usize` saturates).
        let bin_id_unclamped = |center: Real| (k1 * (center - k0)) as usize;
        for leaf in &*leaves {
            let bin_id = bin_id_unclamped(leaf.center().vget(bins_axis)).min(NUM_BINS - 1);
            let bin = &mut bins[bin_id];
            bin.aabb.merge(&leaf.aabb());
            bin.leaf_count += 1;
        }

        // Select the best splitting plane (there are NUM_BINS - 1 splitting planes) based on SAH.
        let mut right_merges = bins;
        let mut right_acc = bins[NUM_BINS - 1];

        for i in 1..NUM_BINS - 1 {
            right_acc.aabb.merge(&right_merges[NUM_BINS - 1 - i].aabb);
            right_acc.leaf_count += &right_merges[NUM_BINS - 1 - i].leaf_count;
            right_merges[NUM_BINS - 1 - i] = right_acc;
        }

        let mut best_cost = Real::MAX;
        let mut best_plane = 0;
        let mut left_merge = bins[0];
        let mut best_leaf_count = bins[0].leaf_count;

        for i in 0..NUM_BINS - 1 {
            let right = &right_merges[i + 1];

            let cost = left_merge.aabb.volume() * left_merge.leaf_count as Real
                + right.aabb.volume() * right.leaf_count as Real;
            if cost < best_cost {
                best_cost = cost;
                best_plane = i;
                best_leaf_count = left_merge.leaf_count;
            }

            left_merge.aabb.merge(&bins[i + 1].aabb);
            left_merge.leaf_count += bins[i + 1].leaf_count;
        }

        // With the splitting plane selected, sort & split the leaves in place.
        let mut mid = best_leaf_count as usize;

        // In degenerate cases where all the node end up on the same bin,
        // just split the range in two.
        if mid == 0 || mid == leaves.len() {
            mid = leaves.len() / 2;
        } else {
            // Sort in-place.
            // TODO PERF: try with using teh leaves_tmp instead of in-place sorting.
            let bin = |leaves: &mut [BvhNode], id: usize| {
                let node = &leaves[id];
                bin_id_unclamped(node.center().vget(bins_axis)).min(NUM_BINS - 1)
            };

            let mut left_id = 0;
            let mut right_id = mid;

            'outer: while left_id != mid && right_id != leaves.len() {
                while bin(leaves, left_id) <= best_plane {
                    left_id += 1;

                    if left_id == mid {
                        break 'outer;
                    }
                }

                while bin(leaves, right_id) > best_plane {
                    right_id += 1;

                    if right_id == leaves.len() {
                        break 'outer;
                    }
                }

                leaves.swap(left_id, right_id);
                left_id += 1;
                right_id += 1;
            }
        }

        mid
    }

    pub(crate) fn rebuild_range_binned(&mut self, target_node_id: u32, leaves: &mut [BvhNode]) {
        let mid = Self::binned_split(leaves);

        // Recurse.
        let (left_leaves, right_leaves) = leaves.split_at_mut(mid);

        assert!(!left_leaves.is_empty() && !right_leaves.is_empty());

        // Recurse.
        if left_leaves.len() == 1 {
            let target = &mut self.nodes[target_node_id as usize];
            target.left = left_leaves[0];

            if target.left.is_leaf() {
                self.leaf_node_indices[target.left.children as usize] =
                    BvhNodeIndex::left(target_node_id);
            } else {
                self.parents[target.left.children as usize] = BvhNodeIndex::left(target_node_id);
            }
        } else {
            let left_id = self.nodes.len() as u32;
            self.nodes.push(BvhNodeWide::zeros());
            self.parents.push(BvhNodeIndex::left(target_node_id));
            self.rebuild_range_binned(left_id, left_leaves);
            self.nodes[target_node_id as usize].left = self.nodes[left_id as usize].merged(left_id);
        }

        if right_leaves.len() == 1 {
            let target = &mut self.nodes[target_node_id as usize];
            target.right = right_leaves[0];

            if target.right.is_leaf() {
                self.leaf_node_indices[target.right.children as usize] =
                    BvhNodeIndex::right(target_node_id);
            } else {
                self.parents[target.right.children as usize] = BvhNodeIndex::right(target_node_id);
            }
        } else {
            let right_id = self.nodes.len() as u32;
            self.nodes.push(BvhNodeWide::zeros());
            self.parents.push(BvhNodeIndex::right(target_node_id));
            self.rebuild_range_binned(right_id, right_leaves);
            self.nodes[target_node_id as usize].right =
                self.nodes[right_id as usize].merged(right_id);
        }
    }
}

/// Raw pointers to the BVH buffers, shared by the parallel binned build tasks.
///
/// Safety: tasks write disjoint node slots / leaf entries (see `rebuild_binned_parallel`).
#[cfg(feature = "parallel")]
#[derive(Copy, Clone)]
struct BuildPtrs {
    nodes: *mut BvhNodeWide,
    parents: *mut BvhNodeIndex,
    /// Slots of `leaf_node_indices` (every leaf id is already present).
    leaf_slots: *mut Option<BvhNodeIndex>,
}

#[cfg(feature = "parallel")]
unsafe impl Send for BuildPtrs {}
#[cfg(feature = "parallel")]
unsafe impl Sync for BuildPtrs {}

#[cfg(feature = "parallel")]
impl Bvh {
    /// Ranges with fewer leaves are built sequentially.
    pub(crate) const PAR_BINNED_MIN_LEAVES: usize = 4096;

    /// Parallel equivalent of `rebuild_range_binned(0, leaves)` on a tree holding only its
    /// (zeroed) root: produces exactly the same nodes, parents and leaf indices.
    ///
    /// The sequential build allocates node slots depth-first (left subtree first) and a subtree
    /// with `k > 1` leaves uses exactly `k - 1` wide nodes, so every subtree's slot range is known
    /// up front and both halves can be built concurrently into disjoint ranges.
    pub(crate) fn rebuild_binned_parallel(&mut self, leaves: &mut [BvhNode]) {
        assert!(leaves.len() > 1);
        debug_assert!(self.nodes.len() == 1 && self.parents.len() == 1);
        let n = leaves.len();
        self.nodes.resize(n - 1, BvhNodeWide::zeros());
        self.parents.resize(n - 1, BvhNodeIndex::default());
        let ptrs = BuildPtrs {
            nodes: self.nodes.as_mut_ptr(),
            parents: self.parents.as_mut_ptr(),
            leaf_slots: self.leaf_node_indices.slots_mut_ptr(),
        };
        // SAFETY: the buffers are sized for the whole tree and outlive the call; see
        // `binned_parallel_recurse` for the disjointness of the writes.
        unsafe { Self::binned_parallel_recurse(ptrs, 0, 1, leaves) };
    }

    /// Builds the subtree of `leaves` rooted at node `target` (already allocated), allocating
    /// its descendants from slot `next_free` on.
    ///
    /// Safety: writes go to `nodes[target].left/right` (each half writes only its own field),
    /// slots in `[next_free, next_free + leaves.len() - 2]` and the parent / leaf-index entries
    /// of this subtree's nodes and leaves, which no other task touches.
    unsafe fn binned_parallel_recurse(p: BuildPtrs, target: u32, next_free: u32, leaves: &mut [BvhNode]) {
        let parallel = leaves.len() >= Self::PAR_BINNED_MIN_LEAVES;
        let mid = Self::binned_split(leaves);
        let (left_leaves, right_leaves) = leaves.split_at_mut(mid);
        assert!(!left_leaves.is_empty() && !right_leaves.is_empty());
        let left_id = next_free;
        let right_id = next_free + if left_leaves.len() == 1 { 0 } else { left_leaves.len() as u32 - 1 };
        let mut left = move || {
            let tgt = p.nodes.add(target as usize);
            if left_leaves.len() == 1 {
                let node = left_leaves[0];
                core::ptr::addr_of_mut!((*tgt).left).write(node);
                if node.is_leaf() {
                    let slot = &mut *p.leaf_slots.add(node.children as usize);
                    *slot.as_mut().expect("key not present") = BvhNodeIndex::left(target);
                } else {
                    *p.parents.add(node.children as usize) = BvhNodeIndex::left(target);
                }
            } else {
                *p.parents.add(left_id as usize) = BvhNodeIndex::left(target);
                Self::binned_parallel_recurse(p, left_id, left_id + 1, left_leaves);
                let merged = (*p.nodes.add(left_id as usize)).merged(left_id);
                core::ptr::addr_of_mut!((*tgt).left).write(merged);
            }
        };
        let mut right = move || {
            let tgt = p.nodes.add(target as usize);
            if right_leaves.len() == 1 {
                let node = right_leaves[0];
                core::ptr::addr_of_mut!((*tgt).right).write(node);
                if node.is_leaf() {
                    let slot = &mut *p.leaf_slots.add(node.children as usize);
                    *slot.as_mut().expect("key not present") = BvhNodeIndex::right(target);
                } else {
                    *p.parents.add(node.children as usize) = BvhNodeIndex::right(target);
                }
            } else {
                *p.parents.add(right_id as usize) = BvhNodeIndex::right(target);
                Self::binned_parallel_recurse(p, right_id, right_id + 1, right_leaves);
                let merged = (*p.nodes.add(right_id as usize)).merged(right_id);
                core::ptr::addr_of_mut!((*tgt).right).write(merged);
            }
        };
        if parallel {
            rayon::join(left, right);
        } else {
            left();
            right();
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct BvhBin {
    aabb: Aabb,
    leaf_count: u32,
}

impl Default for BvhBin {
    fn default() -> Self {
        Self {
            aabb: Aabb::new_invalid(),
            leaf_count: 0,
        }
    }
}
