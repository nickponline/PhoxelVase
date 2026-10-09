//! Debug overlays (gizmos) and the help / stats text.
use crate::coords::to_bevy;
use crate::render::RenderState;
use crate::{Fx, Overlays, Sim};
use bevy::prelude::*;
use rubble_core::ChunkState;

/// The help box (a column: rows up to "dynamic tint", the tint legend, the remaining rows).
#[derive(Component)]
pub struct HelpText;
#[derive(Component)]
pub struct StatsText;
/// A help-box text span.
#[derive(Component, Clone, Copy)]
pub enum HelpSpan {
    Title,
    Desc(usize),
    /// right-aligned on/off column
    State(usize),
}
/// The dynamic-tint legend under the "4 dynamic tint" row (shown only while the tint is on).
#[derive(Component)]
pub struct HelpLegend;

/// Which on/off state a help row shows.
#[derive(Clone, Copy)]
enum Toggle {
    Pause,
    Graph,
    Anchors,
    Colors,
    Tint,
    Stats,
    KeepDebris,
}

/// (key, description, state). Rows without a state are plain actions.
const HELP_ROWS: &[(&str, &str, Option<Toggle>)] = &[
    ("+/-", "next/prev building", None),
    ("LMB", "beam (hold)", None),
    ("RMB", "mouse look (hold)", None),
    ("WASD", "move", None),
    ("Q/E", "down/up", None),
    ("G", "demolish at cursor", None),
    ("X", "demolish", None),
    ("R", "reset", None),
    ("P", "pause", Some(Toggle::Pause)),
    (".", "step", None),
    ("1", "graph/stress", Some(Toggle::Graph)),
    ("2", "anchors", Some(Toggle::Anchors)),
    ("3", "chunk colors", Some(Toggle::Colors)),
    ("4", "dynamic tint", Some(Toggle::Tint)),
    ("5", "stats", Some(Toggle::Stats)),
    ("6", "keep debris", Some(Toggle::KeepDebris)),
    ("H", "help", None),
];
/// Help row after which the dynamic-tint legend is drawn.
const TINT_ROW: usize = 13;
const _: () = assert!(matches!(HELP_ROWS[TINT_ROW].2, Some(Toggle::Tint)));
/// Yellow cross drawn on chunks waiting out their collapse delay (part of the tint overlay).
const DETACHING_COLOR: Color = Color::srgb(1.0, 1.0, 0.2);
/// (swatch colour, description): every material `render::group_material` can return, plus the
/// detaching-chunk gizmo.
const TINT_LEGEND: &[(Color, &str)] = &[
    (crate::render::TINT_BASE, "static building (untinted)"),
    (crate::render::TINT_AWAKE, "falling / moving piece"),
    (crate::render::TINT_ASLEEP, "asleep piece (at rest)"),
    (crate::render::TINT_FROZEN, "frozen rubble (settled)"),
    (DETACHING_COLOR, "about to collapse (cross)"),
];
/// Legend lines start under the description column (6 chars in) and the swatch takes 2 chars.
const LEGEND_INDENT: usize = 6;
const LEGEND_SWATCH: usize = 2;
const FONT_PX: f32 = 13.0;
/// FiraMono advance width is 0.6 em.
const CHAR_W: f32 = FONT_PX * 0.6;
const STATE_ON: Color = Color::WHITE;
const STATE_OFF: Color = Color::srgb(0.5, 0.5, 0.5);

pub fn spawn_text(commands: &mut Commands, ui_cam: Option<Entity>) {
    let panel = |left: bool| Node {
        position_type: PositionType::Absolute,
        top: px(8.0),
        left: if left { px(8.0) } else { auto() },
        right: if left { auto() } else { px(8.0) },
        padding: UiRect::all(px(8.0)),
        ..default()
    };
    let font = TextFont::from_font_size(FontSize::Px(FONT_PX));
    let span = |hs: HelpSpan| (TextSpan::new(""), font.clone(), TextColor(Color::WHITE), hs);
    let h = commands
        .spawn((
            Node { flex_direction: FlexDirection::Column, ..panel(true) },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.55)),
            HelpText,
        ))
        .with_children(|p| {
            // title + rows up to and including "dynamic tint"
            p.spawn((Text::new(""), font.clone(), TextColor(Color::WHITE))).with_children(|t| {
                t.spawn(span(HelpSpan::Title));
                for r in 0..=TINT_ROW {
                    t.spawn(span(HelpSpan::Desc(r)));
                    t.spawn(span(HelpSpan::State(r)));
                }
            });
            // tint legend: a coloured square per state (the default font has no U+25A0 glyph)
            p.spawn((
                Node { flex_direction: FlexDirection::Column, display: Display::None, ..default() },
                HelpLegend,
            ))
            .with_children(|l| {
                for &(col, desc) in TINT_LEGEND {
                    l.spawn(Node {
                        flex_direction: FlexDirection::Row,
                        align_items: AlignItems::Center,
                        padding: UiRect::left(px(LEGEND_INDENT as f32 * CHAR_W)),
                        ..default()
                    })
                    .with_children(|row| {
                        let sq = 9.0;
                        row.spawn((
                            Node {
                                width: px(sq),
                                height: px(sq),
                                margin: UiRect::right(px(LEGEND_SWATCH as f32 * CHAR_W - sq)),
                                ..default()
                            },
                            BackgroundColor(col),
                        ));
                        row.spawn((Text::new(desc), font.clone(), TextColor(STATE_OFF.lighter(0.25))));
                    });
                }
            });
            // remaining rows
            p.spawn((Text::new(""), font.clone(), TextColor(Color::WHITE))).with_children(|t| {
                for r in TINT_ROW + 1..HELP_ROWS.len() {
                    t.spawn(span(HelpSpan::Desc(r)));
                    t.spawn(span(HelpSpan::State(r)));
                }
            });
        })
        .id();
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
                    let col = DETACHING_COLOR;
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
    rs: Res<RenderState>,
    time: Res<Time<Real>>,
    mut help: Query<&mut Visibility, (With<HelpText>, Without<StatsText>)>,
    mut legend: Query<&mut Node, With<HelpLegend>>,
    mut stats: Query<(&mut Text, &mut Visibility), (With<StatsText>, Without<HelpText>)>,
    mut help_spans: Query<(&HelpSpan, &mut TextSpan, &mut TextColor)>,
    mut help_cache: Local<Option<(String, Vec<Option<bool>>)>>,
    mut fps: Local<f32>,
    mut last: Local<f32>,
    mut cached: Local<Option<rubble_core::Stats>>,
) {
    let dt = time.delta_secs().max(1e-4);
    *fps = if *fps == 0.0 { 1.0 / dt } else { *fps + (1.0 / dt - *fps) * 0.05 };
    if let Ok(mut v) = help.single_mut() {
        *v = if ov.help { Visibility::Inherited } else { Visibility::Hidden };
        if ov.help {
            let title = format!("[{}] {}", catalog.label(), spec.title);
            let state = |tg: Toggle| match tg {
                Toggle::Pause => sim.paused,
                Toggle::Graph => ov.graph,
                Toggle::Anchors => ov.anchors,
                Toggle::Colors => ov.random_colors,
                Toggle::Tint => ov.sleep_tint,
                Toggle::Stats => ov.stats,
                Toggle::KeepDebris => ov.keep_debris,
            };
            let states: Vec<Option<bool>> = HELP_ROWS.iter().map(|r| r.2.map(state)).collect();
            // only touch the spans when something visible changed
            if help_cache.as_ref().is_none_or(|(ti, st)| *ti != title || *st != states) {
                // Monospace font: pad descriptions so the 3-wide state column sits at the
                // right edge of the box (the box is as wide as its widest line).
                let row_w = HELP_ROWS.iter().map(|r| 6 + r.1.len()).max().unwrap_or(0) + 2 + 3;
                let legend_w = TINT_LEGEND.iter().map(|l| LEGEND_INDENT + LEGEND_SWATCH + l.1.len()).max().unwrap_or(0) + 1;
                let w = row_w.max(legend_w).max(title.chars().count());
                let last = HELP_ROWS.len() - 1;
                for (hs, mut span, mut col) in &mut help_spans {
                    match *hs {
                        HelpSpan::Title => span.0 = format!("{title}\n"),
                        HelpSpan::Desc(row) => {
                            let (key, desc, _) = HELP_ROWS[row];
                            span.0 = format!("{:<w$}", format!("{key:<6}{desc}"), w = w - 3);
                        }
                        HelpSpan::State(row) => {
                            // the tint row ends its text block (the legend node follows it)
                            let nl = if row == last || row == TINT_ROW { "" } else { "\n" };
                            let (txt, c) = match states[row] {
                                Some(true) => ("on", STATE_ON),
                                Some(false) => ("off", STATE_OFF),
                                None => ("", Color::WHITE),
                            };
                            span.0 = format!("{txt:>3}{nl}");
                            col.0 = c;
                        }
                    }
                }
                if let Ok(mut n) = legend.single_mut() {
                    n.display = if ov.sleep_tint { Display::Flex } else { Display::None };
                }
                *help_cache = Some((title, states));
            }
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
    let rows: [(&str, f64); 29] = [
        ("fps", *fps as f64),
        ("frame (ms)", 1000.0 / *fps as f64),
        ("tick", s.tick as f64),
        ("time (s)", sim.world.time as f64),
        ("sim tick avg (ms)", tm.total_ms as f64),
        ("sim tick last (ms)", sim.world.timings.total_ms as f64),
        ("  projectiles (ms)", tm.projectiles_ms as f64),
        ("  damage (ms)", tm.damage_ms as f64),
        ("  connectivity (ms)", tm.connectivity_ms as f64),
        ("  stress (ms)", tm.stress_ms as f64),
        ("  stress iters", s.stress_iters as f64),
        ("  promotion (ms)", tm.promotion_ms as f64),
        ("  physics (ms)", tm.physics_ms as f64),
        ("  impacts (ms)", tm.impacts_ms as f64),
        ("  settle (ms)", tm.settle_ms as f64),
        ("chunks", s.chunks_total as f64),
        ("  static", s.static_chunks as f64),
        ("  in-flight", s.chunks_in_flight as f64),
        ("  frozen", s.frozen_chunks as f64),
        ("  gone", s.gone_chunks as f64),
        ("clusters", s.clusters as f64),
        ("pending collapses", s.pending_collapses as f64),
        ("rapier bodies", s.rapier_bodies as f64),
        ("rapier colliders", s.rapier_colliders as f64),
        ("events/s", ev as f64),
        ("render groups", groups as f64),
        ("  rebuilt", rs.last_rebuilt as f64),
        ("  rebuild (ms)", rs.last_rebuild_ms as f64),
        ("  diff (ms)", rs.last_sync_ms as f64),
    ];
    // Monospace default font (FiraMono): fixed-width label column, right-aligned values.
    let lw = rows.iter().map(|r| r.0.len()).max().unwrap_or(0);
    let vals: Vec<String> = rows.iter().map(|r| format!("{:.2}", r.1)).collect();
    // min width keeps the column steady as values grow
    let vw = vals.iter().map(|v| v.len()).max().unwrap_or(0).max(9);
    t.0 = rows
        .iter()
        .zip(&vals)
        .map(|((label, _), v)| format!("{label:<lw$}  {v:>vw$}"))
        .collect::<Vec<_>>()
        .join("\n");
}
