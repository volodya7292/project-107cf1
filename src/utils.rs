pub mod index_alloc;
pub mod mesh_simplifier;
pub mod noise;
pub mod ordered_hashmap;
mod qef;
pub mod slice_split;
pub mod slot_vec;
pub mod value_noise;

use crate::render_engine::vertex_mesh::{VertexImpl, VertexNormalImpl, VertexPositionImpl};
use nalgebra as na;
pub use slice_split::SliceSplitImpl;

pub type HashSet<T> = ahash::AHashSet<T>;
pub type HashMap<K, V> = ahash::AHashMap<K, V>;

pub const fn is_pow_of_2(n: u64) -> bool {
    n != 0 && ((n & (n - 1)) == 0)
}

pub const fn prev_power_of_two(mut n: u32) -> u32 {
    n = n | (n >> 1);
    n = n | (n >> 2);
    n = n | (n >> 4);
    n = n | (n >> 8);
    n = n | (n >> 16);
    n - (n >> 1)
}

pub const fn make_mul_of(n: u32, m: u32) -> u32 {
    ((n + m - 1) / m) * m
}

/// log2(8) = 3  
/// log2(5) = 2
pub trait UInteger {
    fn log2(&self) -> Self;
}

pub trait Integer {
    fn div_floor(&self, other: Self) -> Self;
}

macro_rules! u_integer_impl {
    ($($t: ty)*) => ($(
        impl UInteger for $t {
            fn log2(&self) -> Self {
                <$t>::BITS as $t - self.leading_zeros() as $t - 1
            }
        }
    )*)
}
macro_rules! integer_impl {
    ($($t: ty)*) => ($(
        impl Integer for $t {
            #[inline]
            fn div_floor(&self, other: Self) -> Self {
                let d = self / other;
                let r = self % other;
                if (r > 0 && other < 0) || (r < 0 && other > 0) {
                    d - 1
                } else {
                    d
                }
            }
        }
    )*)
}

u_integer_impl! { u8 u16 u32 u64 }
integer_impl! { i8 i16 i32 i64 }

pub trait AllSame {
    fn all_same(&mut self) -> bool;
}

pub trait AllSameBy<I: Iterator> {
    fn all_same_by<F>(&mut self, f: F) -> bool
    where
        F: FnMut(&I::Item, &I::Item) -> bool;
}

impl<I: Iterator> AllSame for I
where
    I::Item: PartialEq,
{
    fn all_same(&mut self) -> bool {
        if let Some(first) = self.next() {
            self.all(|v| v == first)
        } else {
            true
        }
    }
}

impl<I: Iterator> AllSameBy<I> for I {
    fn all_same_by<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&I::Item, &I::Item) -> bool,
    {
        if let Some(first) = self.next() {
            self.all(|v| f(&first, &v))
        } else {
            true
        }
    }
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
    T: VertexImpl + VertexPositionImpl + VertexNormalImpl,
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
        *v.normal_mut() = (v.normal() / vertex_triangle_counts[i] as f32).normalize();
    }
}
