use rubble_format::*;

fn fixture() -> Bld {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/two_box.bld");
    Bld::load(p).expect("load two_box.bld (generate with bgen fixtures)")
}

#[test]
fn two_box_fields() {
    let b = fixture();
    assert_eq!(b.name(), "two_box");
    assert_eq!(b.elements.len(), 2);
    assert_eq!(b.chunks.len(), 4);
    assert_eq!(b.edges.len(), 4);
    assert_eq!(b.sockets.len(), 1);
    assert_eq!(b.materials.len(), 5);
    assert_eq!(b.materials[0].name, "concrete");
    for (i, c) in b.chunks.iter().enumerate() {
        assert!((c.volume - 1.0).abs() < 1e-6);
        assert!((c.mass - 2400.0).abs() < 1e-3);
        assert!((c.inertia[0] - 400.0).abs() < 1e-2);
        assert_eq!(c.hull_v_cnt, 8);
        assert_eq!(c.hull_p_cnt, 6);
        assert_eq!(c.mesh_i_cnt, 36);
        assert_eq!(b.is_anchor(i), i < 2);
        let ez = if i < 2 { 0.5 } else { 1.5 };
        assert!((c.com[2] - ez).abs() < 1e-6);
        for v in b.hull_verts_of(i) {
            for p in b.hull_planes_of(i) {
                assert!(p[0] * v[0] + p[1] * v[1] + p[2] * v[2] + p[3] <= 1e-5);
            }
        }
    }
    let pairs: Vec<(u32, u32)> = b.edges.iter().map(|e| (e.a, e.b)).collect();
    assert_eq!(pairs, vec![(0, 1), (0, 2), (1, 3), (2, 3)]);
    assert_eq!(b.elements[1].frame[11], 1.0); // row-major translation z
    assert_eq!(b.sockets[0].ty, 3);
}

#[test]
fn rejects_corruption() {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../fixtures/two_box.bld");
    let mut raw = std::fs::read(p).unwrap();
    let n = raw.len();
    raw[n - 20] ^= 0xff;
    assert!(matches!(Bld::from_bytes(&raw), Err(BldError::Hash) | Err(BldError::Invalid(_))));
}
