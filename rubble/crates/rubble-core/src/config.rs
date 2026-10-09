//! World configuration, weapons and projectile/explosion descriptors.
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct StressSettings {
    pub enabled: bool,
    pub max_iters: usize,
    pub tol: f32,
    pub hold_time: f32,
    pub max_breaks_per_tick: usize,
    pub bending: bool,
    pub bend_scale: f32,
    /// total wall-clock budget per tick across buildings (ms); iterations adapt to it
    pub budget_ms: f32,
    /// buildings whose structural mass is below this skip stress (connectivity only)
    pub min_component_mass: f32,
    /// edges only break on converged solves, or after this many unconverged ticks
    pub max_unconverged_ticks: u32,
    /// at load the intact building is solved to convergence: iterations per call / max calls
    pub load_iters: usize,
    pub load_max_calls: usize,
    /// pre-crack joints that are overloaded in the intact building (see `World::load_building`)
    pub settle_overloads_at_load: bool,
    pub load_max_rounds: usize,
    /// bearing (compression) capacity multiplier on joint strength (see rubble-stress)
    pub compression_factor: f32,
    /// tension capacity multiplier on the far side of eccentrically loaded bearing sections
    pub tension_factor: f32,
    /// eccentric bearing sections (tipping) on vertical-normal joints
    pub bearing_sections: bool,
    /// Rankine buckling of column stacks (unbraced length between slab connections)
    pub buckling: bool,
    pub rankine_k: f32,
    /// a tick removing support that carried more than this share of the load triggers a full
    /// solve to convergence (≤ catchup_max_ms); 0 disables
    pub catchup_frac: f32,
    pub catchup_max_ms: f32,
    /// joint capacity scales with the remaining hp fraction of its weaker chunk
    pub damage_weakens: bool,
    /// buildings with at most this many supported chunks are always solved globally
    pub region_max_nodes: usize,
}

impl Default for StressSettings {
    fn default() -> Self {
        StressSettings {
            enabled: true,
            max_iters: 30,
            tol: 1e-3,
            hold_time: 0.25,
            max_breaks_per_tick: 8,
            bending: true,
            bend_scale: 1.0,
            budget_ms: 1.5,
            min_component_mass: 0.0,
            max_unconverged_ticks: 20,
            load_iters: 200,
            load_max_calls: 50,
            settle_overloads_at_load: true,
            load_max_rounds: 64,
            compression_factor: 3.5,
            tension_factor: 5.0,
            bearing_sections: true,
            buckling: true,
            rankine_k: 1200.0,
            catchup_frac: 0.05,
            catchup_max_ms: 20.0,
            damage_weakens: true,
            region_max_nodes: 16384,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct WorldConfig {
    pub gravity: [f32; 3],
    // promotion
    pub collapse_delay_min: f32,
    pub collapse_delay_max: f32,
    /// mass (kg) at which the collapse delay reaches its maximum (log-scaled from 100 kg)
    pub collapse_delay_ref_mass: f32,
    // debris
    pub debris_min_volume: f32,
    /// destroyed chunks larger than this pulverize (ChunkShattered) instead of becoming debris
    pub debris_max_volume: f32,
    pub debris_ttl: f32,
    pub freeze_time: f32,
    /// small clusters / debris that come to rest also freeze into static rubble (instead of
    /// staying dynamic until `debris_ttl`)
    pub freeze_debris: bool,
    /// clusters younger than this never freeze (debris stays pushable while things land on it)
    pub freeze_min_age: f32,
    /// clusters of at most `crush_max_chunks` chunks hit with impulse/mass > `crush_dv` (m/s) are crushed
    pub crush_dv: f32,
    pub crush_max_chunks: usize,
    pub freeze_min_mass: f32,
    /// Keep debris around as rubble instead of removing it: destroyed chunks of any size from
    /// `keep_debris_min_volume` up (instead of `debris_min_volume`..`debris_max_volume`) become
    /// debris, up to `keep_debris_max_clusters` moving pieces (instead of
    /// `max_dynamic_clusters`), resting debris
    /// freezes into rubble instead of despawning after `debris_ttl` / `max_dynamic_time`
    /// (only debris still moving after `keep_debris_max_age` is removed), and light pieces
    /// (debris, or under `freeze_min_mass`) do not collide with each other while moving.
    pub keep_debris: bool,
    pub keep_debris_min_volume: f32,
    pub keep_debris_max_age: f32,
    pub keep_debris_max_clusters: usize,
    /// clusters still moving after this long are force-frozen (big) or despawned (small)
    pub max_dynamic_time: f32,
    /// Focus (see `World::set_focus`, e.g. the player's camera): small debris farther than
    /// `lod_far` (m) from every focus point lives only `lod_far_ttl` of its usual time, and when
    /// the piece budgets are exceeded small debris goes farthest first. No focus: no effect.
    pub lod_far: f32,
    pub lod_far_ttl: f32,
    pub rest_lin_vel: f32,
    pub rest_ang_vel: f32,
    /// Rigid-body tipping check (`Building::tipping_edges`) on horizontal planes at the heights
    /// of recent damage: a group of the intact structure whose centre of mass is more than
    /// `tip_margin` (m) outside the joints it stands on is cut loose and falls.
    pub tipping: bool,
    pub tip_margin: f32,
    /// seconds between checks per damaged building (damage heights accumulate meanwhile)
    pub tip_interval: f32,
    /// damage heights are merged into planes on this grid (m)
    pub tip_band: f32,
    /// at most this many planes per check (the most recently damaged first)
    pub tip_max_planes: usize,
    /// Full ground check (`Building::ungrounded_components`) at most every
    /// `ground_check_interval` s per building whose joints changed: anything the load-carrying
    /// graph no longer connects to an anchor collapses, whatever the incremental search found.
    pub ground_check: bool,
    pub ground_check_interval: f32,
    /// After damage, a static chunk next to it that is left hanging by slivers (alive joints
    /// totalling less than `sliver_area` m², none of them a joint of at least
    /// `sliver_seat_area` m² under its centre of mass) is cut loose: the joint graph still
    /// connects it to the ground, but nothing visibly holds it up.
    pub sliver_area: f32,
    pub sliver_seat_area: f32,
    /// When geometry is removed (destroyed, detached, despawned), wake sleeping bodies around it
    /// and turn frozen rubble that rested on it back into falling clusters.
    pub thaw_unsupported: bool,
    /// how far (m) around removed geometry to look for things that rested on it
    pub thaw_margin: f32,
    /// support test: a frozen group is supported if its chunks, moved down by this much (m),
    /// touch anything outside the group (ground, building, other rubble, debris)
    pub thaw_probe: f32,
    /// safety net: frozen chunks re-checked for support per tick (round robin) while anything is
    /// still moving, 0 = off
    pub thaw_sweep_per_tick: usize,
    /// an explosion thaws frozen rubble it pushes by more than this (m/s, impulse / chunk mass)
    /// back into moving pieces, so blasts scatter rubble piles; 0 = rubble ignores blasts
    pub blast_thaw_dv: f32,
    // budgets
    pub max_dynamic_clusters: usize,
    pub max_chunks_in_flight: usize,
    // impacts / splitting
    /// internal edge breaks when accumulated impulse > strength * impact_factor
    pub impact_factor: f32,
    /// per-step decay of accumulated edge impulse
    pub impact_decay: f32,
    /// impact stress (big clusters landing hard): inertial load m·dv/impact_duration is routed
    /// through the bonds into the ground-contact chunks; bonds with utilization > 1 break.
    pub impact_duration: f32,
    pub impact_stress_min_dv: f32,
    /// time constant (s) of the landing-deceleration accumulator
    pub impact_window: f32,
    pub impact_stress_min_chunks: usize,
    pub impact_stress_cooldown: f32,
    pub impact_max_breaks: usize,
    /// inter-panel joints fail under impact at this fraction of their static capacity
    pub impact_joint_factor: f32,
    /// chunks within this distance (m, along the impact direction) of the contacts bear load too
    pub impact_support_band: f32,
    /// solve/break rounds per impact (lets failures cascade)
    pub impact_rounds: usize,
    /// impact solves run on a worker thread and are applied exactly this many ticks later
    /// (fixed latency keeps the simulation deterministic); 0 = solve inline
    pub impact_latency_ticks: u32,
    /// solver tolerance / iterations per call for impact solves
    pub impact_tol: f32,
    pub impact_iters: usize,
    /// Progressive collapse ("crush front"): a falling piece taller than `crush_min_extent` (m,
    /// along the impact) that lands with more kinetic energy than it takes to crush the
    /// `crush_band` (m, about a storey) at its base crushes that band; the rest keeps falling
    /// with the energy left over, lands on the next storey and so on, so a tall section
    /// pancakes itself floor by floor instead of landing rigid. The crush energy of the band is
    /// the capacity (N) of its load-bearing bonds (normal within ~45° of the impact) times
    /// `crush_distance` (m): the resisting force drops sharply once columns buckle, so the work
    /// to crush a storey is about its peak capacity over ~1/7 of the storey height (0.5 m for
    /// 3.6 m). Gravity's work over that distance counts on the falling side. `crush_band` 0 = off.
    pub crush_band: f32,
    pub crush_min_extent: f32,
    pub crush_distance: f32,
    /// of the crushed chunks, every n-th stays as debris (the rest is pulverized); 0 = none.
    /// Solid debris under the front would cushion and hold up the falling section.
    pub crush_debris_every: usize,
    /// `Event::Impact` for new contacts with at least this impulse (N·s); 0 = no impact events
    pub impact_event_min: f32,
    /// at most this many `Event::Impact` per tick (the hardest)
    pub impact_events_max: usize,
    /// contact impulse (N·s) below which impacts do not damage static chunks
    pub impact_min_impulse: f32,
    /// hp damage per N·s above `impact_min_impulse`
    pub impact_damage_scale: f32,
    // edges
    /// edge hit points = strength (N) * this
    pub edge_health_per_newton: f32,
    /// edge damage = explosion damage * falloff * this
    pub explosion_edge_damage_scale: f32,
    pub crack_radius_factor: f32,
    // explosions
    pub occlusion_factor: f32,
    pub occlusion_max_hits: usize,
    pub occlusion_max_chunks: usize,
    /// explosion impulse memory: detaching chunks pick up impulses younger than this (s)
    pub impulse_memory: f32,
    // projectiles
    pub ballistic_substeps: usize,
    pub max_penetrations: usize,
    /// per material id: damage multiplier
    pub material_damage_mult: Vec<f32>,
    /// per material id: penetration resistance (energy per meter)
    pub material_resistance: Vec<f32>,
    pub emit_edge_events: bool,
    /// full (swept) continuous collision detection for every moving body, whatever its speed.
    /// Expensive: rapier's CCD pass costs about as much as the rest of a collapse tick.
    pub ccd: bool,
    /// predictive ("soft") collision detection for pieces faster than this (m/s), looking one
    /// step's travel ahead, switched per body every tick; 0 = off. Without it a piece moving
    /// more than a slab's thickness per tick (0.2 m slab at 60 Hz: ~12 m/s, a ~7 m fall) can
    /// pass through it.
    pub ccd_speed: f32,
    /// ... for pieces of at most this many chunks. Small debris is what visibly tunnels; a big
    /// falling section is far thicker than any slab it lands on.
    pub ccd_max_chunks: usize,
    pub solver_iterations: usize,
    /// Worker threads for the rigid-body step (a process-wide pool per size). Rapier's results do
    /// not depend on the thread count; a few threads beat the whole machine for this workload
    /// (less fork/join and spin overhead; on an M3 Max 6 was fastest). 0 = auto
    /// (`RUBBLE_PHYSICS_THREADS` if set, else min(6, cores / 2)).
    pub physics_threads: usize,
    pub stress: StressSettings,
}

impl Default for WorldConfig {
    fn default() -> Self {
        WorldConfig {
            gravity: [0.0, 0.0, -9.81],
            collapse_delay_min: 0.3,
            collapse_delay_max: 1.0,
            collapse_delay_ref_mass: 1.0e6,
            debris_min_volume: 0.05,
            debris_max_volume: 0.25,
            debris_ttl: 10.0,
            freeze_time: 0.5,
            freeze_debris: true,
            freeze_min_age: 2.0,
            crush_dv: 6.0,
            crush_max_chunks: 4,
            freeze_min_mass: 500.0,
            keep_debris: false,
            keep_debris_min_volume: 0.01,
            keep_debris_max_age: 120.0,
            keep_debris_max_clusters: 2048,
            max_dynamic_time: 30.0,
            lod_far: 60.0,
            lod_far_ttl: 0.25,
            rest_lin_vel: 0.15,
            rest_ang_vel: 0.25,
            ground_check: true,
            ground_check_interval: 0.1,
            sliver_area: 0.05,
            sliver_seat_area: 0.02,
            tipping: true,
            tip_margin: 0.5,
            tip_interval: 0.1,
            tip_band: 0.5,
            tip_max_planes: 8,
            thaw_unsupported: true,
            thaw_margin: 0.06,
            thaw_probe: 0.03,
            thaw_sweep_per_tick: 16,
            blast_thaw_dv: 0.5,
            max_dynamic_clusters: 512,
            max_chunks_in_flight: 50_000,
            impact_factor: 0.01,
            impact_decay: 0.5,
            impact_duration: 0.03,
            impact_stress_min_dv: 2.0,
            impact_window: 0.15,
            impact_stress_min_chunks: 8,
            impact_stress_cooldown: 0.1,
            impact_max_breaks: 8192,
            impact_joint_factor: 0.1,
            impact_support_band: 1.0,
            impact_rounds: 3,
            impact_latency_ticks: 6,
            impact_event_min: 1000.0,
            impact_events_max: 64,
            crush_band: 3.6,
            crush_min_extent: 8.0,
            crush_distance: 0.5,
            crush_debris_every: 10,
            impact_tol: 1e-3,
            impact_iters: 200,
            impact_min_impulse: 3000.0,
            impact_damage_scale: 0.05,
            edge_health_per_newton: 1e-3,
            explosion_edge_damage_scale: 1.0,
            crack_radius_factor: 1.5,
            occlusion_factor: 0.5,
            occlusion_max_hits: 3,
            occlusion_max_chunks: 256,
            impulse_memory: 3.0,
            ballistic_substeps: 4,
            max_penetrations: 4,
            //                     concrete brick wood metal glass
            material_damage_mult: vec![1.0, 1.1, 1.5, 0.6, 10.0],
            material_resistance: vec![400.0, 300.0, 100.0, 1500.0, 1.0],
            emit_edge_events: false,
            ccd: false,
            ccd_speed: 9.0,
            ccd_max_chunks: 16,
            solver_iterations: 4,
            physics_threads: 0,
            stress: StressSettings::default(),
        }
    }
}

impl WorldConfig {
    pub fn damage_mult(&self, mat: u16) -> f32 {
        self.material_damage_mult.get(mat as usize).copied().unwrap_or(1.0)
    }
    pub fn resistance(&self, mat: u16) -> f32 {
        self.material_resistance.get(mat as usize).copied().unwrap_or(400.0)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct WeaponParams {
    pub damage: f32,
    pub splash_r: f32,
    pub splash_frac: f32,
    /// penetration energy (resistance·m); 0 = stops at first hit
    pub penetration: f32,
    pub range: f32,
    /// muzzle speed for ballistic projectiles (m/s)
    pub speed: f32,
    /// swept sphere radius for ballistic projectiles
    pub radius: f32,
    /// impulse applied to dynamic clusters on hit (N·s)
    pub impulse: f32,
    pub gravity_scale: f32,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Weapon {
    Ar,
    Smg,
    Sniper,
    Shotgun,
    Launcher,
    Custom(WeaponParams),
}

impl Weapon {
    pub const AR: Weapon = Weapon::Ar;
    pub fn params(&self) -> WeaponParams {
        let base = WeaponParams {
            damage: 25.0,
            splash_r: 0.0,
            splash_frac: 0.0,
            penetration: 0.0,
            range: 500.0,
            speed: 800.0,
            radius: 0.02,
            impulse: 50.0,
            gravity_scale: 1.0,
        };
        match *self {
            Weapon::Ar => WeaponParams { damage: 30.0, splash_r: 0.6, splash_frac: 0.2, penetration: 20.0, ..base },
            Weapon::Smg => WeaponParams { damage: 18.0, ..base },
            Weapon::Sniper => WeaponParams { damage: 120.0, penetration: 150.0, speed: 1200.0, ..base },
            Weapon::Shotgun => WeaponParams { damage: 12.0, splash_r: 0.3, splash_frac: 0.3, range: 60.0, ..base },
            Weapon::Launcher => WeaponParams { damage: 100.0, speed: 60.0, radius: 0.1, impulse: 500.0, ..base },
            Weapon::Custom(p) => p,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectileKind {
    Hitscan,
    Ballistic,
}

#[derive(Clone, Copy, Debug)]
pub struct Projectile {
    pub kind: ProjectileKind,
    pub pos: [f32; 3],
    /// unit direction for hitscan; velocity (m/s) for ballistic
    pub vel: [f32; 3],
    pub weapon: WeaponParams,
    pub age: f32,
    /// optional explosion on impact (rockets)
    pub explode: Option<Explosion>,
}

impl Projectile {
    pub fn hitscan(origin: [f32; 3], dir: [f32; 3], weapon: Weapon) -> Self {
        Projectile { kind: ProjectileKind::Hitscan, pos: origin, vel: dir, weapon: weapon.params(), age: 0.0, explode: None }
    }
    /// `dir` is normalized and scaled by the weapon muzzle speed.
    pub fn ballistic(origin: [f32; 3], dir: [f32; 3], weapon: Weapon) -> Self {
        let p = weapon.params();
        let l = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt().max(1e-9);
        let v = [dir[0] / l * p.speed, dir[1] / l * p.speed, dir[2] / l * p.speed];
        Projectile { kind: ProjectileKind::Ballistic, pos: origin, vel: v, weapon: p, age: 0.0, explode: None }
    }
    pub fn with_explosion(mut self, e: Explosion) -> Self {
        self.explode = Some(e);
        self
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub struct Explosion {
    pub center: [f32; 3],
    pub radius: f32,
    #[serde(default)]
    pub inner_radius: f32,
    pub damage: f32,
    #[serde(default)]
    pub impulse: f32,
}
