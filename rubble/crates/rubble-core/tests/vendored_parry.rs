//! The vendored parry3d's parallel binned BVH build must produce exactly the sequential tree
//! (bit-identical simulation depends on it).
use parry3d::bounding_volume::Aabb;
use parry3d::math::Vector;
use parry3d::partitioning::{Bvh, BvhBuildStrategy};

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*s >> 40) as f32) / (1u64 << 24) as f32
}

#[test]
fn parallel_binned_build_matches_sequential() {
    let mut seed = 7u64;
    for &n in &[2usize, 3, 100, 4095, 4096, 4097, 20_000, 70_000] {
        let leaves: Vec<Aabb> = (0..n)
            .map(|i| {
                // clustered + duplicated centres exercise the degenerate split path
                let c = if i % 7 == 0 {
                    Vector::new(1.0, 2.0, 3.0)
                } else {
                    Vector::new(lcg(&mut seed) * 50.0, lcg(&mut seed) * 5.0, lcg(&mut seed) * 200.0)
                };
                let h = Vector::new(lcg(&mut seed), lcg(&mut seed), lcg(&mut seed)) * 0.5;
                Aabb::new(c - h, c + h)
            })
            .collect();
        let par = Bvh::from_leaves(BvhBuildStrategy::Binned, &leaves);
        let seq = Bvh::from_leaves_sequential(BvhBuildStrategy::Binned, &leaves);
        assert_eq!(format!("{par:?}"), format!("{seq:?}"), "n = {n}");
    }
}
