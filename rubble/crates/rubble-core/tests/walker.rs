//! `RapierBackend::move_capsule` (the viewer's walk mode) climbs a bgen switchback stair:
//! walk the first flight, cross the mid landing, walk the second flight and arrive one floor up
//! (skipped when assets are missing).
use rubble_core::*;

const ASSETS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../assets/buildings");
const RADIUS: f32 = 0.3;
const HALF_SEG: f32 = 0.6;
const DT: f32 = 1.0 / 60.0;

/// Walk the capsule (centre `c`) horizontally towards `(x, y)` at 4.5 m/s under gravity.
fn walk_to(w: &World, c: &mut Vec3, vz: &mut f32, x: f32, y: f32) {
    for _ in 0..600 {
        let d = Vec3::new(x - c.x, y - c.y, 0.0);
        if d.length() < 0.05 {
            return;
        }
        let h = d.normalize() * (4.5 * DT).min(d.length());
        *vz -= 18.0 * DT;
        let (t, grounded) = w.phys.move_capsule(*c, HALF_SEG, RADIUS, Vec3::new(h.x, h.y, *vz * DT), DT);
        *c += t;
        if grounded && *vz < 0.0 {
            *vz = 0.0;
        }
    }
    panic!("stuck at {c:?} walking to ({x}, {y})");
}

#[test]
fn walks_up_office_stairs() {
    let dir = format!("{ASSETS}/office_1");
    let Ok(man) = std::fs::read_to_string(format!("{dir}/manifest.json")) else {
        eprintln!("skip: no asset");
        return;
    };
    let man: serde_json::Value = serde_json::from_str(&man).unwrap();
    let fh = man["params"]["floor_height"].as_f64().unwrap() as f32;
    let core = man["rooms"].as_array().unwrap().iter().find(|r| r["tag"] == "stair" && r["floor"] == 0).unwrap();
    let pts: Vec<(f32, f32)> =
        core["polygon"].as_array().unwrap().iter().map(|p| (p[0].as_f64().unwrap() as f32, p[1].as_f64().unwrap() as f32)).collect();
    let (x0, x1) = pts.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.0), a.1.max(p.0)));
    let (y0, y1) = pts.iter().fold((f32::MAX, f32::MIN), |a, p| (a.0.min(p.1), a.1.max(p.1)));
    let mut wd = World::new(WorldConfig::default());
    wd.load_building(&format!("{dir}/building.bld"), Isometry::identity()).unwrap();
    wd.add_ground_plane(0.0);
    wd.step(DT);

    // core-local (u along the long axis from the min corner, v across); landing u < 1.4,
    // flight 1 on v < 1.15 rising +u, flight 2 on the far side rising -u
    let along_x = x1 - x0 >= y1 - y0;
    let (l, wdt) = if along_x { (x1 - x0, y1 - y0) } else { (y1 - y0, x1 - x0) };
    let at = |u: f32, v: f32| if along_x { (x0 + u, y0 + v) } else { (x0 + v, y0 + u) };
    let (sx, sy) = at(0.7, 0.575);
    let mut c = Vec3::new(sx, sy, 1.5);
    let mut vz = 0.0;
    // settle onto the ground-floor landing
    for _ in 0..120 {
        vz -= 18.0 * DT;
        let (t, grounded) = wd.phys.move_capsule(c, HALF_SEG, RADIUS, Vec3::new(0.0, 0.0, vz * DT), DT);
        c += t;
        if grounded && vz < 0.0 {
            vz = 0.0;
        }
    }
    let start = c.z;
    for (u, v) in [(l - 0.6, 0.575), (l - 0.6, wdt - 0.575), (0.7, wdt - 0.575)] {
        let (x, y) = at(u, v);
        walk_to(&wd, &mut c, &mut vz, x, y);
        eprintln!("at ({:.2}, {:.2}) z {:.2}", c.x, c.y, c.z);
    }
    let rise = c.z - start;
    assert!((rise - fh).abs() < 0.3, "climbed {rise:.2} m, floor height {fh:.2}");
}
