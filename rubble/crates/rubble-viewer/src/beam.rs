//! Visual of the cutting beam: a bright core cylinder from the muzzle out to the beam's range
//! (it cuts through everything in line, so it is never stopped by geometry).
use crate::Fx;
use bevy::light::NotShadowCaster;
use bevy::prelude::*;

/// Radius of the drawn beam core (the cutting width is `BEAM_RADIUS`, not drawn).
const CORE_RADIUS: f32 = 0.025;

#[derive(Component)]
pub struct BeamPart;

pub fn setup_beam(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>, mut mats: ResMut<Assets<StandardMaterial>>) {
    let material = mats.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.95, 0.8),
        emissive: LinearRgba::rgb(6.0, 5.0, 3.0),
        unlit: true,
        ..default()
    });
    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(1.0, 1.0))),
        MeshMaterial3d(material),
        Transform::default(),
        Visibility::Hidden,
        NotShadowCaster,
        BeamPart,
    ));
}

pub fn draw_beam(fx: Res<Fx>, time: Res<Time<Real>>, mut q: Query<(&mut Transform, &mut Visibility), With<BeamPart>>) {
    let flicker = 1.0 + 0.15 * (time.elapsed_secs() * 60.0).sin();
    for (mut xf, mut vis) in &mut q {
        let Some((a, b)) = fx.beam else {
            *vis = Visibility::Hidden;
            continue;
        };
        let d = b - a;
        let len = d.length();
        if len < 1e-3 {
            *vis = Visibility::Hidden;
            continue;
        }
        *vis = Visibility::Visible;
        let r = CORE_RADIUS * flicker;
        *xf = Transform {
            translation: (a + b) * 0.5,
            rotation: Quat::from_rotation_arc(Vec3::Y, d / len),
            scale: Vec3::new(r, len, r),
        };
    }
}
