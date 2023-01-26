use nalgebra_glm::Vec3;

use crate::renderer::vertex_mesh::{AttributesImpl, VertexNormalImpl, VertexPositionImpl};

/// Calculate interpolated normals using neighbour triangles.
pub fn calc_smooth_mesh_normals<T>(vertices: &mut [T], indices: &[u32])
where
    T: AttributesImpl + VertexPositionImpl + VertexNormalImpl,
{
    let mut vertex_triangle_counts = vec![0_u32; vertices.len()];
    let mut triangle_normals = Vec::<Vec3>::with_capacity(indices.len() / 3);

    for i in (0..indices.len()).step_by(3) {
        let ind = &indices[i..(i + 3)];
        let normal = base::utils::calc_triangle_normal(
            &vertices[ind[0] as usize].position(),
            &vertices[ind[1] as usize].position(),
            &vertices[ind[2] as usize].position(),
        );

        triangle_normals.push(normal);
    }

    for v in vertices.iter_mut() {
        v.set_normal(Vec3::from_element(0.0));
    }

    for (i, normal) in triangle_normals.iter().enumerate() {
        let indices_i = i * 3;
        let ind = &indices[indices_i..(indices_i + 3)];

        // Check for NaN
        if normal == normal {
            vertices[ind[0] as usize].set_normal(vertices[ind[0] as usize].normal() + *normal);
            vertices[ind[1] as usize].set_normal(vertices[ind[1] as usize].normal() + *normal);
            vertices[ind[2] as usize].set_normal(vertices[ind[2] as usize].normal() + *normal);

            vertex_triangle_counts[ind[0] as usize] += 1;
            vertex_triangle_counts[ind[1] as usize] += 1;
            vertex_triangle_counts[ind[2] as usize] += 1;
        }
    }

    for (i, v) in vertices.iter_mut().enumerate() {
        v.set_normal((v.normal() / vertex_triangle_counts[i] as f32).normalize());
    }
}

pub fn find_best_video_mode(monitor: &winit::monitor::MonitorHandle) -> winit::monitor::VideoMode {
    let curr_refresh_rate = monitor.refresh_rate_millihertz().unwrap();

    monitor
        .video_modes()
        .max_by(|a, b| {
            let a_width = a.size().width;
            let b_width = b.size().width;
            let a_fps_diff = a.refresh_rate_millihertz().abs_diff(curr_refresh_rate);
            let b_fsp_diff = b.refresh_rate_millihertz().abs_diff(curr_refresh_rate);

            a_width.cmp(&b_width).then(a_fps_diff.cmp(&b_fsp_diff).reverse())
        })
        .unwrap()
}
