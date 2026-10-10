//! What to load: CLI args, `.bld` files, synthetic arenas, scenario yaml (same schema as
//! `rubble-sim`; `record` is ignored). Builds fresh `World`s for start/reset.
use bevy::prelude::Resource;
use rubble_core::testutil;
use rubble_core::{Explosion, Isometry, Projectile, Weapon, World, WorldConfig};
use rubble_format::Bld;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Default, Debug, Clone)]
pub struct Args {
    pub blds: Vec<PathBuf>,
    pub scenario: Option<PathBuf>,
    pub arena: Option<usize>,
    pub screenshot: Option<PathBuf>,
    pub frames: u32,
    pub explode: Option<[f32; 5]>,
    pub explode_frame: u32,
    /// screenshot mode: hold the beam (engine origin, direction) from `explode_frame` on
    pub beam: Option<[f32; 6]>,
    pub cam: Option<[f32; 3]>,
    pub look: Option<[f32; 3]>,
    pub overlays: Vec<String>,
    pub width: f32,
    pub height: f32,
    /// record mode: save a PNG every `record_every` sim ticks into this directory
    pub record: Option<PathBuf>,
    pub record_every: u32,
    /// screenshot/record mode: press X (demolish every building) at this frame
    pub demolish_frame: Option<u32>,
    /// screenshot/record mode: press Z (interior demolition of every building) at this frame
    pub interior_frame: Option<u32>,
    /// screenshot/record mode: render this many seconds before the simulation starts (shader
    /// pipelines compile asynchronously; frames captured before that are black)
    pub warmup: f32,
}

fn parse_vec<const N: usize>(s: &str) -> Result<[f32; N], String> {
    let v: Vec<f32> = s.split(',').map(|x| x.trim().parse::<f32>()).collect::<Result<_, _>>().map_err(|e| format!("{s}: {e}"))?;
    v.try_into().map_err(|_| format!("expected {N} comma-separated numbers, got `{s}`"))
}

pub const USAGE: &str = "usage: rubble-viewer [FILE.bld ...] [--scenario S.yaml] [--arena N]
       [--screenshot OUT.png --frames N [--explode x,y,z,r[,damage]] [--explode-frame K]
        [--beam x,y,z,dx,dy,dz] [--cam x,y,z] [--look x,y,z] [--overlay f1,..,f8|lighting|particles|nolighting|noparticles] [--size WxH]]
       [--record DIR --frames N [--record-every K]]   frames DIR/frame_00000.png ... every K ticks (default 4)
       [--demolish-frame K]   screenshot/record mode: demolish (X) every building at tick K
       [--interior-frame K]   screenshot/record mode: interior demolition (Z) of every building at tick K
       [--warmup SECS]   render before simulating (default 3 s when recording)
  no inputs: synthetic testutil arena.  --arena N: N synthetic 6-storey towers (perf test).
  coordinates are engine Z-up meters.";

impl Args {
    pub fn parse() -> Result<Self, String> {
        let mut a = Args { frames: 120, explode_frame: 10, width: 1600.0, height: 900.0, record_every: 4, warmup: -1.0, ..Default::default() };
        let mut it = std::env::args().skip(1);
        while let Some(s) = it.next() {
            let mut val = || it.next().ok_or_else(|| format!("{s} needs a value"));
            match s.as_str() {
                "-h" | "--help" => return Err(USAGE.into()),
                "--scenario" => a.scenario = Some(val()?.into()),
                "--arena" => a.arena = Some(val()?.parse().map_err(|e| format!("--arena: {e}"))?),
                "--screenshot" => a.screenshot = Some(val()?.into()),
                "--record" => a.record = Some(val()?.into()),
                "--record-every" => a.record_every = val()?.parse::<u32>().map_err(|e| format!("--record-every: {e}"))?.max(1),
                "--warmup" => a.warmup = val()?.parse().map_err(|e| format!("--warmup: {e}"))?,
                "--demolish-frame" => a.demolish_frame = Some(val()?.parse().map_err(|e| format!("--demolish-frame: {e}"))?),
                "--interior-frame" => a.interior_frame = Some(val()?.parse().map_err(|e| format!("--interior-frame: {e}"))?),
                "--frames" => a.frames = val()?.parse().map_err(|e| format!("--frames: {e}"))?,
                "--explode" => {
                    let v = val()?;
                    a.explode = Some(match v.split(',').count() {
                        4 => {
                            let [x, y, z, r] = parse_vec::<4>(&v)?;
                            [x, y, z, r, 2000.0]
                        }
                        _ => parse_vec::<5>(&v)?,
                    })
                }
                "--explode-frame" => a.explode_frame = val()?.parse().map_err(|e| format!("--explode-frame: {e}"))?,
                "--beam" => a.beam = Some(parse_vec::<6>(&val()?)?),
                "--cam" => a.cam = Some(parse_vec::<3>(&val()?)?),
                "--look" => a.look = Some(parse_vec::<3>(&val()?)?),
                "--overlay" => a.overlays = val()?.split(',').map(|x| x.trim().to_lowercase()).collect(),
                "--size" => {
                    let v = val()?;
                    let (w, h) = v.split_once('x').ok_or("--size WxH")?;
                    a.width = w.parse().map_err(|_| "--size WxH")?;
                    a.height = h.parse().map_err(|_| "--size WxH")?;
                }
                f if f.starts_with("--") => return Err(format!("unknown flag {f}\n{USAGE}")),
                f => a.blds.push(f.into()),
            }
        }
        if a.warmup < 0.0 {
            a.warmup = if a.record.is_some() { 3.0 } else { 0.0 };
        }
        Ok(a)
    }
}

// ---------------------------------------------------------------- scenario (rubble-sim schema)

#[derive(Deserialize)]
struct Scenario {
    buildings: Vec<BuildingSpec>,
    #[serde(default)]
    actions: Vec<Action>,
    /// ground plane height, `null` for none, or `auto` (default): the lowest foundation bottom
    #[serde(default = "default_ground")]
    ground: serde_yaml::Value,
    #[serde(default)]
    config: Option<serde_yaml::Value>,
}
fn default_ground() -> serde_yaml::Value {
    serde_yaml::Value::String("auto".into())
}

#[derive(Deserialize)]
struct BuildingSpec {
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    synthetic: Option<Synthetic>,
    #[serde(default)]
    pos: [f32; 3],
    #[serde(default)]
    yaw: f32,
}

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Synthetic {
    Tower { floors: usize, side: f32, #[serde(default)] center_column: bool },
    Grid { nx: usize, ny: usize, nz: usize, #[serde(default = "one")] size: f32 },
    Wall { w: f32, h: f32, #[serde(default = "thick")] t: f32, #[serde(default = "half")] cell: f32 },
    Cantilever { h: usize, len: usize },
    TwoBox,
}
fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}
fn thick() -> f32 {
    0.3
}
impl Synthetic {
    fn build(&self) -> Bld {
        match *self {
            Synthetic::Tower { floors, side, center_column } => testutil::tower(floors, side, center_column),
            Synthetic::Grid { nx, ny, nz, size } => testutil::grid_block(nx, ny, nz, size),
            Synthetic::Wall { w, h, t, cell } => testutil::wall(w, h, t, cell),
            Synthetic::Cantilever { h, len } => testutil::cantilever(h, len),
            Synthetic::TwoBox => testutil::two_box(),
        }
    }
}

#[derive(Deserialize)]
struct Action {
    t: f32,
    #[serde(default)]
    explode: Option<Explosion>,
    #[serde(default)]
    fire: Option<Fire>,
    #[serde(default)]
    damage: Option<DamageAct>,
}
#[derive(Deserialize, Clone)]
struct Fire {
    origin: [f32; 3],
    dir: [f32; 3],
    #[serde(default = "default_weapon")]
    weapon: Weapon,
    #[serde(default = "one_u")]
    count: u32,
    #[serde(default = "ten")]
    rate: f32,
    #[serde(default)]
    ballistic: bool,
}
fn default_weapon() -> Weapon {
    Weapon::Ar
}
fn one_u() -> u32 {
    1
}
fn ten() -> f32 {
    10.0
}
#[derive(Deserialize, Clone)]
struct DamageAct {
    #[serde(default)]
    building: u32,
    chunks: Vec<u32>,
    #[serde(default = "big")]
    amount: f32,
}
fn big() -> f32 {
    1e9
}

#[derive(Clone)]
pub enum Scheduled {
    Explode(Explosion),
    Fire(Projectile),
    Damage(u32, Vec<u32>, f32),
}

fn merge_yaml(base: &mut serde_yaml::Value, over: &serde_yaml::Value) {
    match (base, over) {
        (serde_yaml::Value::Mapping(b), serde_yaml::Value::Mapping(o)) => {
            for (k, v) in o {
                match b.get_mut(k) {
                    Some(bv) => merge_yaml(bv, v),
                    None => {
                        b.insert(k.clone(), v.clone());
                    }
                }
            }
        }
        (b, o) => *b = o.clone(),
    }
}

/// Half-size (m) of the ground laid around buildings with sunk foundations.
pub const TERRAIN_EXTENT: f32 = 1000.0;

/// Everything needed to (re)build the world from scratch (R = reset).
#[derive(Resource, Clone)]
pub struct WorldSpec {
    pub cfg: WorldConfig,
    pub buildings: Vec<(Bld, Isometry)>,
    pub ground: Option<f32>,
    /// `ground` follows the buildings' foundations (see `fit_ground`)
    pub ground_auto: bool,
    /// scripted actions, sorted by time
    pub actions: Vec<(f32, Scheduled)>,
    pub title: String,
}

fn bld_extent(b: &Bld) -> ([f32; 3], [f32; 3]) {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for c in &b.chunks {
        for k in 0..3 {
            lo[k] = lo[k].min(c.aabb_min[k]);
            hi[k] = hi[k].max(c.aabb_max[k]);
        }
    }
    (lo, hi)
}

impl WorldSpec {
    pub fn from_args(a: &Args) -> Result<Self, String> {
        if let Some(sc) = &a.scenario {
            let mut spec = Self::from_scenario(sc)?;
            // extra .bld files on the command line are appended to the right of the scenario
            spec.append_row(&a.blds)?;
            spec.fit_ground();
            return Ok(spec);
        }
        let mut spec = WorldSpec {
            cfg: WorldConfig::default(),
            buildings: vec![],
            ground: Some(0.0),
            ground_auto: true,
            actions: vec![],
            title: String::new(),
        };
        if !a.blds.is_empty() {
            spec.append_row(&a.blds)?;
            spec.title = a.blds.iter().map(|p| building_name(p)).collect::<Vec<_>>().join(" ");
        } else if let Some(n) = a.arena {
            for (bld, pos) in testutil::arena(n, |_| testutil::tower(6, 12.0, true), 20.0) {
                spec.buildings.push((bld, Isometry::new(pos, 0.0)));
            }
            spec.title = format!("synthetic arena: {n} towers");
        } else {
            let make = |i: usize| match i % 6 {
                0 => testutil::tower(5, 10.0, true),
                1 => testutil::wall(14.0, 7.0, 0.3, 0.5),
                2 => testutil::grid_block(6, 6, 6, 1.0),
                3 => testutil::tower(8, 8.0, false),
                4 => testutil::cantilever(6, 7),
                _ => testutil::tower(3, 14.0, true),
            };
            for (bld, pos) in testutil::arena(9, make, 24.0) {
                spec.buildings.push((bld, Isometry::new(pos, 0.0)));
            }
            spec.title = "synthetic testutil arena".into();
        }
        spec.fit_ground();
        Ok(spec)
    }

    /// With `ground_auto`, put the ground plane at the lowest foundation bottom (0 without any).
    fn fit_ground(&mut self) {
        if self.ground_auto {
            let z = self.buildings.iter().map(|(b, iso)| iso.pos[2] + b.ground_z()).reduce(f32::min);
            self.ground = Some(z.unwrap_or(0.0));
        }
    }

    /// Lay `.bld` files out in a row along +X after whatever is already loaded.
    fn append_row(&mut self, paths: &[PathBuf]) -> Result<(), String> {
        let mut x = self
            .buildings
            .iter()
            .map(|(b, iso)| iso.pos[0] + bld_extent(b).1[0])
            .fold(f32::NEG_INFINITY, f32::max);
        if !x.is_finite() {
            x = 0.0;
        } else {
            x += 10.0;
        }
        for p in paths {
            let bld = Bld::load(p).map_err(|e| format!("{}: {e}", p.display()))?;
            let (lo, hi) = bld_extent(&bld);
            self.buildings.push((bld, Isometry::new([x - lo[0], 0.0, 0.0], 0.0)));
            x += hi[0] - lo[0] + 10.0;
        }
        Ok(())
    }

    fn from_scenario(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let sc: Scenario = serde_yaml::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let base = path.parent().unwrap_or(Path::new("."));
        let mut cfg = WorldConfig::default();
        if let Some(over) = &sc.config {
            let mut v = serde_yaml::to_value(&cfg).map_err(|e| e.to_string())?;
            merge_yaml(&mut v, over);
            cfg = serde_yaml::from_value(v).map_err(|e| format!("config: {e}"))?;
        }
        let mut buildings = vec![];
        for b in &sc.buildings {
            let iso = Isometry::new(b.pos, b.yaw);
            if let Some(p) = &b.path {
                let mut pp = PathBuf::from(p);
                if !pp.exists() {
                    pp = base.join(p);
                }
                if !pp.exists() {
                    // scenarios usually live in <repo>/scenarios with repo-relative paths
                    pp = base.join("..").join(p);
                }
                buildings.push((Bld::load(&pp).map_err(|e| format!("{}: {e}", pp.display()))?, iso));
            } else if let Some(s) = &b.synthetic {
                buildings.push((s.build(), iso));
            } else {
                return Err("building needs `path` or `synthetic`".into());
            }
        }
        let mut actions = vec![];
        for a in &sc.actions {
            if let Some(e) = a.explode {
                actions.push((a.t, Scheduled::Explode(e)));
            }
            if let Some(f) = &a.fire {
                for i in 0..f.count {
                    let p = if f.ballistic {
                        Projectile::ballistic(f.origin, f.dir, f.weapon)
                    } else {
                        Projectile::hitscan(f.origin, f.dir, f.weapon)
                    };
                    actions.push((a.t + i as f32 / f.rate.max(1e-3), Scheduled::Fire(p)));
                }
            }
            if let Some(d) = &a.damage {
                actions.push((a.t, Scheduled::Damage(d.building, d.chunks.clone(), d.amount)));
            }
        }
        actions.sort_by(|a, b| a.0.total_cmp(&b.0));
        let (ground, ground_auto) = match &sc.ground {
            serde_yaml::Value::Null => (None, false),
            serde_yaml::Value::String(s) if s == "auto" => (None, true),
            v => (Some(v.as_f64().ok_or_else(|| format!("{}: ground must be a number, null or auto", path.display()))? as f32), false),
        };
        let mut spec = WorldSpec { cfg, buildings, ground, ground_auto, actions, title: path.display().to_string() };
        spec.fit_ground();
        Ok(spec)
    }

    pub fn build_world(&self) -> World {
        let mut w = World::new(self.cfg.clone());
        for (bld, iso) in &self.buildings {
            w.load_building_bld(bld.clone(), *iso);
        }
        if let Some(z) = self.ground {
            // ground level around sunk foundations / basements (the plane catches the pits)
            if self.ground_auto {
                w.add_terrain(TERRAIN_EXTENT);
            }
            w.add_ground_plane(z);
        }
        w
    }

    /// World-space (engine) AABB of all buildings, for camera framing.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let mut lo = [f32::MAX; 3];
        let mut hi = [f32::MIN; 3];
        for (b, iso) in &self.buildings {
            let (l, h) = bld_extent(b);
            let (s, c) = iso.yaw.sin_cos();
            for corner in 0..8 {
                let p = [
                    if corner & 1 == 0 { l[0] } else { h[0] },
                    if corner & 2 == 0 { l[1] } else { h[1] },
                    if corner & 4 == 0 { l[2] } else { h[2] },
                ];
                let w = [c * p[0] - s * p[1] + iso.pos[0], s * p[0] + c * p[1] + iso.pos[1], p[2] + iso.pos[2]];
                for k in 0..3 {
                    lo[k] = lo[k].min(w[k]);
                    hi[k] = hi[k].max(w[k]);
                }
            }
        }
        if lo[0] > hi[0] {
            return ([-10.0; 3], [10.0; 3]);
        }
        (lo, hi)
    }

    pub fn total_chunks(&self) -> usize {
        self.buildings.iter().map(|(b, _)| b.chunks.len()).sum()
    }
}

// ---------------------------------------------------------------- building catalog (+ / -)

/// Every `assets/buildings/*/building.bld`, sorted by folder name, for cycling with + / -.
#[derive(Resource, Default, Debug)]
pub struct Catalog {
    pub paths: Vec<PathBuf>,
    /// index of the building currently shown, if it came from the catalog
    pub current: Option<usize>,
}

impl Catalog {
    /// Look for `assets/buildings` next to the cwd, one level up (running from `rubble/`), and
    /// relative to this crate's source tree.
    pub fn discover() -> Self {
        let candidates = [
            PathBuf::from("assets/buildings"),
            PathBuf::from("../assets/buildings"),
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../assets/buildings"),
        ];
        let Some(dir) = candidates.iter().find(|d| d.is_dir()) else { return Self::default() };
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path().join("building.bld"))
            .filter(|p| p.is_file())
            .collect();
        paths.sort();
        Catalog { paths, current: None }
    }

    /// Mark `p` as current if it is one of the catalog's buildings.
    pub fn select_path(&mut self, p: &Path) {
        let canon = |q: &Path| q.canonicalize().ok();
        let want = canon(p);
        self.current = self.paths.iter().position(|q| want.is_some() && canon(q) == want);
    }

    /// Step `delta` entries with wrap-around; from "no current building" + goes to the first
    /// and - to the last.
    pub fn step(&mut self, delta: i32) -> Option<&Path> {
        let n = self.paths.len() as i32;
        if n == 0 {
            return None;
        }
        let i = match self.current {
            Some(i) => (i as i32 + delta).rem_euclid(n),
            None if delta >= 0 => 0,
            None => n - 1,
        };
        self.current = Some(i as usize);
        Some(&self.paths[i as usize])
    }

    pub fn label(&self) -> String {
        match self.current {
            Some(i) => format!("{}/{}", i + 1, self.paths.len()),
            None => format!("-/{}", self.paths.len()),
        }
    }
}

/// Short display name for a catalog entry: its folder (e.g. `office_1`).
pub fn building_name(p: &Path) -> String {
    p.parent().and_then(|d| d.file_name()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| p.display().to_string())
}

impl WorldSpec {
    /// A world holding just one `.bld` at the origin (default config, ground under its foundation).
    pub fn single(path: &Path) -> Result<Self, String> {
        let mut spec = WorldSpec {
            cfg: WorldConfig::default(),
            buildings: vec![],
            ground: Some(0.0),
            ground_auto: true,
            actions: vec![],
            title: building_name(path),
        };
        spec.append_row(&[path.to_path_buf()])?;
        spec.fit_ground();
        Ok(spec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_wraps() {
        let mut c = Catalog { paths: vec!["a".into(), "b".into(), "c".into()], current: None };
        assert_eq!(c.step(-1).unwrap(), Path::new("c"));
        assert_eq!(c.step(1).unwrap(), Path::new("a"));
        assert_eq!(c.step(1).unwrap(), Path::new("b"));
        assert_eq!(c.step(1).unwrap(), Path::new("c"));
        assert_eq!(c.step(1).unwrap(), Path::new("a"));
        assert_eq!(c.step(-1).unwrap(), Path::new("c"));
        assert_eq!(c.label(), "3/3");
        let mut e = Catalog::default();
        assert!(e.step(1).is_none());
    }
}
