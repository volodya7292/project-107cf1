pub mod noise;
mod qef;
pub mod slice_split;
pub mod slot_vec;
pub mod thread_pool;
pub mod value_noise;

use crate::renderer::vertex_mesh::{VertexImpl, VertexNormalImpl, VertexPositionImpl};
use nalgebra_glm::{DVec3, Vec3};
pub use slice_split::SliceSplitImpl;
use std::sync::atomic;
use std::thread;
use std::time::{Duration, Instant};

pub type HashSet<T> = ahash::AHashSet<T>;
pub type HashMap<K, V> = ahash::AHashMap<K, V>;
pub type IndexSet<T> = indexmap::IndexSet<T, ahash::RandomState>;
pub type IndexMap<T> = indexmap::IndexMap<T, ahash::RandomState>;

pub const MO_RELAXED: atomic::Ordering = atomic::Ordering::Relaxed;
pub const MO_ACQUIRE: atomic::Ordering = atomic::Ordering::Acquire;
pub const MO_RELEASE: atomic::Ordering = atomic::Ordering::Release;

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
pub trait UInt {
    fn log2(&self) -> Self;
}

pub trait Int {}

macro_rules! uint_impl {
    ($($t: ty)*) => ($(
        impl UInt for $t {
            // TODO: remove when std log2 is stable
            fn log2(&self) -> Self {
                <$t>::BITS as $t - self.leading_zeros() as $t - 1
            }
        }
    )*)
}
macro_rules! int_impl {
    ($($t: ty)*) => ($(
        impl Int for $t {

        }
    )*)
}

uint_impl! { u8 u16 u32 u64 }
int_impl! { i8 i16 i32 i64 }

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

pub fn calc_triangle_area(v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Vec3 {
    let side0 = v1 - v0;
    let side1 = v2 - v0;
    side0.cross(&side1) / 2.0
}

pub fn calc_triangle_normal(v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Vec3 {
    let side0 = v1 - v0;
    let side1 = v2 - v0;
    side0.cross(&side1).normalize()
}

pub fn camera_move_xz(rotation: Vec3, front_back: f64, left_right: f64) -> DVec3 {
    let d = DVec3::new((-rotation.y).sin() as f64, 0.0, (-rotation.y).cos() as f64);
    let mut motion_delta = -d * (front_back as f64);
    motion_delta -= d.cross(&DVec3::new(0.0, 1.0, 0.0)).normalize() * (left_right) as f64;
    motion_delta
}

/// Calculate interpolated normals using neighbour triangles.
pub fn calc_smooth_mesh_normals<T>(vertices: &mut [T], indices: &[u32])
where
    T: VertexImpl + VertexPositionImpl + VertexNormalImpl,
{
    let mut vertex_triangle_counts = vec![0_u32; vertices.len()];
    let mut triangle_normals = Vec::<Vec3>::with_capacity(indices.len() / 3);

    for i in (0..indices.len()).step_by(3) {
        let ind = &indices[i..(i + 3)];
        let normal = calc_triangle_normal(
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

pub fn high_precision_sleep(duration: Duration, single_sleep_period: Duration) {
    let end_t = Instant::now() + duration;

    while Instant::now() < end_t {
        thread::sleep(single_sleep_period);
    }
}
