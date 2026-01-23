use approx::{AbsDiff, abs_diff_eq, assert_abs_diff_eq};
use common::nalgebra as na;
use engine::module::main_renderer::vertex_mesh::{VAttributes, VertexMeshCreate};
use engine::{attributes_impl, vertex_impl_position, vkw, winit};
use std::sync::Arc;

#[derive(Default, Copy, Clone, PartialEq)]
struct Vertex {
    position: na::Vector3<f32>,
    tex_coord: na::Vector2<f32>,
}
attributes_impl!(Vertex, position, tex_coord);
vertex_impl_position!(Vertex);

#[test]
fn test_with_device() {
    let _vke = vkw::Entry::new().unwrap();
    let instance = _vke.create_instance("GOVNO!", None).unwrap();

    let adapters = instance.enumerate_adapters(None).unwrap();
    let adapter = adapters.first().unwrap();
    let device = adapter.create_device().unwrap();

    test_vertex_mesh(&device);
}

fn test_vertex_mesh(device: &Arc<vkw::Device>) {
    let mut start_vertices = vec![];
    for i in 0..4096 {
        start_vertices.push(Vertex {
            position: na::Vector3::from_element(i as f32),
            tex_coord: na::Vector2::from_element(i as f32),
        });
    }

    let mut start_indices = vec![];
    for i in 0..16384 {
        start_indices.push((i * 3 + 16) % 4096);
    }

    let mut mesh = device
        .create_vertex_mesh::<Vertex>(VAttributes::Slice(&start_vertices), Some(&start_indices))
        .unwrap();

    let end_vertices = mesh.get_vertices(0, start_vertices.len() as u32);
    let end_indices = mesh.get_indices(0, start_indices.len() as u32);

    let match_vertices = start_vertices
        .iter()
        .zip(&end_vertices)
        .filter(|&(a, b)| a == b)
        .count();
    let match_indices = start_indices
        .iter()
        .zip(&end_indices)
        .filter(|&(a, b)| a == b)
        .count();

    assert_eq!(start_vertices.len(), match_vertices);
    assert_eq!(start_indices.len(), match_indices);

    // ------ Partial comparison ------

    let end_vertices = mesh.get_vertices(1234, 2048);
    let end_indices = mesh.get_indices(9252, 1024);

    let match_vertices = start_vertices[1234..(1234 + 2048)]
        .iter()
        .zip(&end_vertices)
        .filter(|&(a, b)| a == b)
        .count();
    let match_indices = start_indices[9252..(9252 + 1024)]
        .iter()
        .zip(&end_indices)
        .filter(|&(a, b)| a == b)
        .count();

    assert_eq!(end_vertices.len(), match_vertices);
    assert_eq!(end_indices.len(), match_indices);
}
