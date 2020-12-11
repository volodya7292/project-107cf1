pub mod mesh_simplifier;
mod qef;

use crate::renderer::vertex_mesh;
use crate::renderer::vertex_mesh::{VertexImpl, VertexNormalImpl};
use nalgebra as na;
use std::mem;

pub fn is_pow_of_2(n: u64) -> bool {
    (n & (n - 1)) == 0
}

pub fn next_power_of_two(mut n: u32) -> u32 {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    n
}

pub fn make_mul_of(n: u32, m: u32) -> u32 {
    ((n + m - 1) / m) * m
}

pub fn log2(n: u32) -> u32 {
    (mem::size_of::<u32>() * 8) as u32 - n.leading_zeros() - 1
}

pub fn calc_triangle_normal(
    v0: &na::Vector3<f32>,
    v1: &na::Vector3<f32>,
    v2: &na::Vector3<f32>,
) -> na::Vector3<f32> {
    let side0 = v1 - v0;
    let side1 = v2 - v0;
    side0.cross(&side1).normalize()
}

/// Calculate interpolated normals using neighbour triangles.
pub fn calc_smooth_mesh_normals<T>(vertices: &mut [T], indices: &[u32])
where
    T: VertexImpl + VertexNormalImpl,
{
    let mut vertex_triangle_counts = vec![0_u32; vertices.len()];
    let mut triangle_normals = Vec::<na::Vector3<f32>>::with_capacity(indices.len() / 3);

    for i in (0..indices.len()).step_by(3) {
        let ind = &indices[i..(i + 3)];
        let normal = calc_triangle_normal(
            vertices[ind[0] as usize].position(),
            vertices[ind[1] as usize].position(),
            vertices[ind[2] as usize].position(),
        );

        triangle_normals.push(normal);
    }

    vertices
        .iter_mut()
        .for_each(|vertex| *vertex.normal_mut() = na::Vector3::from_element(0.0));

    for (i, normal) in triangle_normals.iter().enumerate() {
        let indices_i = i * 3;
        let ind = &indices[indices_i..(indices_i + 3)];

        // Check for NaN
        if normal == normal {
            *vertices[ind[0] as usize].normal_mut() += *normal;
            *vertices[ind[1] as usize].normal_mut() += *normal;
            *vertices[ind[2] as usize].normal_mut() += *normal;

            vertex_triangle_counts[ind[0] as usize] += 1;
            vertex_triangle_counts[ind[1] as usize] += 1;
            vertex_triangle_counts[ind[2] as usize] += 1;
        }
    }

    for (i, v) in vertices.iter_mut().enumerate() {
        *v.normal_mut() /= (vertex_triangle_counts[i] as f32);
    }
}
