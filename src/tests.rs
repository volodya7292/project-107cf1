use crate::game::overworld::block_model;
use crate::game::overworld::block_model::Quad;
use approx::{abs_diff_eq, assert_abs_diff_eq, AbsDiff};
use engine::vertex_impl;
use nalgebra as na;
use nalgebra_glm::{I32Vec3, Vec3};

#[derive(Default, Copy, Clone, PartialEq)]
struct Vertex {
    position: na::Vector3<f32>,
    tex_coord: na::Vector2<f32>,
}
vertex_impl!(Vertex, position, tex_coord);

#[test]
fn test_with_device() {
    let _vke = vk_wrapper::Entry::new().unwrap();
    // let instance = vke.create_instance("GOVNO!", &[]).unwrap();

    // let adapters = instance.enumerate_adapters(None).unwrap();
    // let adapter = adapters.first().unwrap();
    // let device = adapter.create_device().unwrap();

    // test_vertex_mesh(&device);
}

#[test]
fn cluster_facing() {
    use crate::game::overworld::block_component::Facing;

    assert!(Facing::from_direction(&I32Vec3::new(-1, 0, 1)).is_none());
    assert!(Facing::from_direction(&I32Vec3::new(1, 0, 1)).is_none());
    assert!(Facing::from_direction(&I32Vec3::new(0, 0, 0)).is_none());

    assert_eq!(
        Facing::from_direction(&I32Vec3::new(-1, 0, 0)).unwrap(),
        Facing::NegativeX
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(1, 0, 0)).unwrap(),
        Facing::PositiveX
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, -1, 0)).unwrap(),
        Facing::NegativeY
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, 1, 0)).unwrap(),
        Facing::PositiveY
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, 0, -1)).unwrap(),
        Facing::NegativeZ
    );
    assert_eq!(
        Facing::from_direction(&I32Vec3::new(0, 0, 1)).unwrap(),
        Facing::PositiveZ
    );

    assert_eq!(Facing::NegativeX.mirror(), Facing::PositiveX);
    assert_eq!(Facing::PositiveX.mirror(), Facing::NegativeX);
    assert_eq!(Facing::NegativeY.mirror(), Facing::PositiveY);
    assert_eq!(Facing::PositiveY.mirror(), Facing::NegativeY);
    assert_eq!(Facing::NegativeZ.mirror(), Facing::PositiveZ);
    assert_eq!(Facing::PositiveZ.mirror(), Facing::NegativeZ);
}

#[test]
fn calc_block_quad_area() {
    let quad = Quad::new([
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
    ]);

    assert_abs_diff_eq!(quad.area(), 1.0, epsilon = 1e-7);
}

// fn test_vertex_mesh(device: &Arc<vkw::Device>) {
//     let mut mesh = device.create_vertex_mesh::<Vertex>().unwrap();
//
//     let mut start_vertices = vec![];
//     for i in 0..4096 {
//         start_vertices.push(Vertex {
//             position: na::Vector3::from_element(i as f32),
//             tex_coord: na::Vector2::from_element(i as f32),
//         });
//     }
//
//     let mut start_indices = vec![];
//     for i in 0..16384 {
//         start_indices.push((i * 3 + 16) % 4096);
//     }
//
//     mesh.set_vertices(&start_vertices, Some(&start_indices));
//
//     let end_vertices = mesh.get_vertices(0, start_vertices.len() as u32);
//     let end_indices = mesh.get_indices(0, start_indices.len() as u32);
//
//     let match_vertices = start_vertices
//         .iter()
//         .zip(&end_vertices)
//         .filter(|&(a, b)| a == b)
//         .count();
//     let match_indices = start_indices
//         .iter()
//         .zip(&end_indices)
//         .filter(|&(a, b)| a == b)
//         .count();
//
//     assert_eq!(start_vertices.len(), match_vertices);
//     assert_eq!(start_indices.len(), match_indices);
//
//     // ------ Partial comparison ------
//
//     let end_vertices = mesh.get_vertices(1234, 2048);
//     let end_indices = mesh.get_indices(9252, 1024);
//
//     let match_vertices = start_vertices[1234..(1234 + 2048)]
//         .iter()
//         .zip(&end_vertices)
//         .filter(|&(a, b)| a == b)
//         .count();
//     let match_indices = start_indices[9252..(9252 + 1024)]
//         .iter()
//         .zip(&end_indices)
//         .filter(|&(a, b)| a == b)
//         .count();
//
//     assert_eq!(end_vertices.len(), match_vertices);
//     assert_eq!(end_indices.len(), match_indices);
// }
