pub mod wsi;

use crate::module::main_renderer::vertex_mesh::{AttributesImpl, VertexNormalImpl, VertexPositionImpl};
use nalgebra_glm::Vec3;
use std::mem;

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

pub trait U8SliceHelper {
    fn raw_copy_from<T: Copy>(&mut self, value: T);
}

impl U8SliceHelper for [u8] {
    fn raw_copy_from<T: Copy>(&mut self, value: T) {
        let size = mem::size_of_val(&value);
        let raw_data = unsafe { std::slice::from_raw_parts(&value as *const _ as *const u8, size) };
        self.copy_from_slice(raw_data);
    }
}
