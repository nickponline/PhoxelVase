//! Debug overlays (gizmos) and the help / stats text.
use crate::camera::{CamCtl, CamMode};
use crate::coords::to_bevy;
use crate::render::RenderState;
use crate::{Controls, Fx, Overlays, Sim, WeaponSel};
use bevy::prelude::*;
use rubble_core::ChunkState;

#[derive(Component)]
pub struct HelpText;
#[derive(Component)]
pub struct StatsText;

pub fn spawn_text(commands: &mut Commands, ui_cam: Option<Entity>) {
    let panel = |left: bool| Node {
        position_type: PositionType::Absolute,
        top: px(8.0),
        left: if left { px(8.0) } else { auto() },
        right: if left { auto() } else { px(8.0) },
        padding: UiRect::all(px(8.0)),
        ..default()
    };
    let font = TextFont::from_font_size(FontSize::Px(13.0));
    let h = commands.spawn((
        Text::new(""),
        font.clone(),
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        panel(true),
        HelpText,
    )).id();
    let st = commands.spawn((
        Text::new(""),
        font,
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
        panel(false),
        StatsText,
        Visibility::Hidden,
    )).id();
    if let Some(c) = ui_cam {
        commands.entity(h).insert(UiTargetCamera(c));
        commands.entity(st).insert(UiTargetCamera(c));
    }
}

/// green (0) -> yellow (0.5) -> red (1); magenta above 1 (overloaded).
fn util_color(u: f32) -> Color {
    if u > 1.0 {
        return Color::srgb(1.0, 0.0, 1.0);
    }
    let u = u.clamp(0.0, 1.0);
    if u < 0.5 {
        Color::srgb(u * 2.0, 1.0, 0.0)
    } else {
        Color::srgb(1.0, 2.0 - u * 2.0, 0.0)
    }
}

pub fn draw_overlays(
    mut gizmos: Gizmos,
    sim: Res<Sim>,
    ov: Res<Overlays>,
    mut fx: ResMut<Fx>,
    time: Res<Time<Real>>,
    mut coms: Local<Vec<Vec3>>,
) {
    let dt = time.delta_secs();
    // feedback effects
    fx.tracers.retain_mut(|t| {
        t.2 += dt;
        t.2 < 0.15
    });
    for (a, b, age, c) in &fx.tracers {
        gizmos.line(*a, *b, c.with_alpha(1.0 - age / 0.15));
    }
    fx.blasts.retain_mut(|b| {
        b.2 += dt;
        b.2 < 0.6
    });
    for (p, r, age) in &fx.blasts {
        if *age >= 0.0 {
            let k = age / 0.6;
            gizmos.sphere(Isometry3d::from_translation(*p), r * (0.3 + 0.7 * k), Color::srgba(1.0, 0.6, 0.1, 1.0 - k));
        }
    }

    let w = &sim.world;
    if !(ov.graph || ov.anchors || ov.sleep_tint) {
        return;
    }
    for (bi, bd) in w.buildings.iter().enumerate() {
        let n = bd.n_chunks();
        coms.clear();
        coms.reserve(n);
        for c in 0..n {
            let p = match bd.state[c] {
                ChunkState::Static | ChunkState::Detaching => bd.pose.transform_point(bd.bld.chunks[c].com.into()),
                ChunkState::Gone => rubble_core::Vec3::ZERO,
                _ => w.chunk_world_com(bi, c),
            };
            coms.push(to_bevy(p.to_array()));
        }
        if ov.graph {
            for (e, er) in bd.bld.edges.iter().enumerate() {
                if !bd.edge_alive[e] {
                    continue;
                }
                let (a, b) = (er.a as usize, er.b as usize);
                if bd.state[a] == ChunkState::Gone || bd.state[b] == ChunkState::Gone {
                    continue;
                }
                let u = bd.utilization.get(e).copied().unwrap_or(0.0);
                gizmos.line(coms[a], coms[b], util_color(u));
            }
        }
        if ov.anchors {
            let s = 0.25;
            for c in 0..n {
                if bd.anchor[c] && bd.state[c] != ChunkState::Gone {
                    let p = coms[c];
                    let col = Color::srgb(0.1, 0.4, 1.0);
                    gizmos.line(p - Vec3::X * s, p + Vec3::X * s, col);
                    gizmos.line(p - Vec3::Y * s, p + Vec3::Y * s, col);
                    gizmos.line(p - Vec3::Z * s, p + Vec3::Z * s, col);
                }
            }
        }
        if ov.sleep_tint {
            // chunks waiting out their collapse delay
            let s = 0.3;
            for c in 0..n {
                if bd.state[c] == ChunkState::Detaching {
                    let p = coms[c];
                    let col = Color::srgb(1.0, 1.0, 0.2);
                    gizmos.line(p - Vec3::X * s, p + Vec3::X * s, col);
                    gizmos.line(p - Vec3::Y * s, p + Vec3::Y * s, col);
                    gizmos.line(p - Vec3::Z * s, p + Vec3::Z * s, col);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn update_text(
    sim: Res<Sim>,
    spec: Res<crate::scene::WorldSpec>,
    catalog: Res<crate::scene::Catalog>,
    ov: Res<Overlays>,
    ctl: Res<Controls>,
    rs: Res<RenderState>,
    time: Res<Time<Real>>,
    cams: Query<&CamCtl>,
    mut help: Query<(&mut Text, &mut Visibility), (With<HelpText>, Without<StatsText>)>,
    mut stats: Query<(&mut Text, &mut Visibility), (With<StatsText>, Without<HelpText>)>,
    mut fps: Local<f32>,
    mut last: Local<f32>,
    mut cached: Local<Option<rubble_core::Stats>>,
) {
    let dt = time.delta_secs().max(1e-4);
    *fps = if *fps == 0.0 { 1.0 / dt } else { *fps + (1.0 / dt - *fps) * 0.05 };
    let mode = cams.single().map(|c| c.mode).unwrap_or(CamMode::Fly);
    if let Ok((mut t, mut v)) = help.single_mut() {
        *v = if ov.help { Visibility::Inherited } else { Visibility::Hidden };
        if ov.help {
            let on = |b: bool| if b { "on " } else { "off" };
            let wpn = match ctl.weapon {
                WeaponSel::Ar => "AR (hold)",
                WeaponSel::Sniper => "Sniper",
                WeaponSel::Launcher => "Launcher",
            };
            t.0 = format!(
                "rubble-viewer   [{}] {}   {}\n\
                 +/- next/prev building in assets/buildings\n\
                 weapon [1/2/3]: {wpn}   explosion radius [ ]: {:.1} m   camera [O]: {:?}\n\
                 LMB fire | E explode at cursor | G C4 | R reset | P pause | . step\n\
                 RMB+mouse look | WASD move, Space/C up/down, Shift fast, wheel speed/zoom\n\
                 F1 graph/stress {}  F2 anchors {}  F3 chunk colors {}  F4 dynamic tint {}  F5 stats {}  H help",
                catalog.label(),
                spec.title,
                if sim.paused { "[PAUSED]" } else { "" },
                ctl.radius,
                mode,
                on(ov.graph),
                on(ov.anchors),
                on(ov.random_colors),
                on(ov.sleep_tint),
                on(ov.stats),
            );
        }
    }
    let Ok((mut t, mut v)) = stats.single_mut() else { return };
    *v = if ov.stats { Visibility::Inherited } else { Visibility::Hidden };
    if !ov.stats {
        return;
    }
    let now = time.elapsed_secs();
    if cached.is_none() || now - *last > 0.25 {
        *cached = Some(sim.world.stats());
        *last = now;
    }
    let s = cached.unwrap();
    let tm = &sim.timings_avg;
    let ev: usize = sim.event_log.iter().map(|e| e.1).sum();
    let groups = rs.groups.len();
    t.0 = format!(
        "fps {:.0} ({:.1} ms)   tick {} t={:.2}s\n\
         sim tick {:.2} ms (last {:.2})\n\
         \x20 projectiles {:.2}  damage {:.2}\n\
         \x20 connectivity {:.2}  stress {:.2} ({} it)\n\
         \x20 promotion {:.2}  physics {:.2}\n\
         \x20 impacts {:.2}  settle {:.2}\n\
         chunks {}  static {}  in-flight {}\n\
         \x20 frozen {}  gone {}\n\
         clusters {}  pending collapses {}\n\
         rapier bodies {}  colliders {}\n\
         events/s {}\n\
         render groups {}  rebuilt {} ({:.2} ms)  diff {:.2} ms",
        *fps,
        1000.0 / *fps,
        s.tick,
        sim.world.time,
        tm.total_ms,
        sim.world.timings.total_ms,
        tm.projectiles_ms,
        tm.damage_ms,
        tm.connectivity_ms,
        tm.stress_ms,
        s.stress_iters,
        tm.promotion_ms,
        tm.physics_ms,
        tm.impacts_ms,
        tm.settle_ms,
        s.chunks_total,
        s.static_chunks,
        s.chunks_in_flight,
        s.frozen_chunks,
        s.gone_chunks,
        s.clusters,
        s.pending_collapses,
        s.rapier_bodies,
        s.rapier_colliders,
        ev,
        groups,
        rs.last_rebuilt,
        rs.last_rebuild_ms,
        rs.last_sync_ms,
    );
}
