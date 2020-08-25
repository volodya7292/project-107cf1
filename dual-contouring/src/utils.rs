use nalgebra as na;
use std::mem;

pub fn is_pow_of_2(n: u64) -> bool {
    (n & (n - 1)) == 0
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
