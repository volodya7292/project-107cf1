use std::hash::Hash;
use std::slice;
use std::sync::atomic;
use std::time::{Duration, Instant};
use std::{mem, thread};

use nalgebra_glm::Vec3;

pub use slice_split::SliceSplitImpl;

pub mod noise;
mod qef;
pub mod resource_file;
pub mod slice_split;
pub mod threading;
pub mod timer;
pub mod unsafe_slice;
pub mod voronoi_noise;
pub mod white_noise;

pub type HashSet<T> = ahash::AHashSet<T>;
pub type HashMap<K, V> = ahash::AHashMap<K, V>;
pub type ConcurrentCache<K, V> = moka::sync::Cache<K, V, ahash::RandomState>;
pub type IndexSet<T> = indexmap::IndexSet<T, ahash::RandomState>;
pub type IndexMap<T, V> = indexmap::IndexMap<T, V, ahash::RandomState>;

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

pub trait Bool {
    /// Converts `false` into `0.0` and `true` into `1.0`.
    fn into_f32(self) -> f32;
    /// Converts `false` into `0.0` and `true` into `1.0`.
    fn into_f64(self) -> f64;
}

impl Bool for bool {
    fn into_f32(self) -> f32 {
        self as u32 as f32
    }

    fn into_f64(self) -> f64 {
        self as u64 as f64
    }
}

/// log2(8) = 3  
/// log2(5) = 2
pub trait UInt {
    // TODO: remove when std log2 is stable
    fn log2(self) -> Self;
    // TODO: remove when std log is stable
    fn log(self, base: Self) -> Self;
    // TODO: remove when std div_ceil is stable
    fn div_ceil(self, other: Self) -> Self;
    // TODO: remove when std next_multiple_of is stable
    fn next_multiple_of(self, m: Self) -> Self;
}

pub trait Int {}

macro_rules! uint_impl {
    ($($t: ty)*) => ($(
        impl UInt for $t {
            fn log2(self) -> Self {
                <$t>::BITS as $t - self.leading_zeros() as $t - 1
            }

            fn log(self, base: Self) -> Self {
                let mut n = 0;
                let mut r = self;

                // Optimization for 128 bit wide integers.
                if Self::BITS == 128 {
                    let b = Self::log2(self) / (Self::log2(base) + 1);
                    n += b;
                    r /= base.pow(b as u32);
                }

                while r >= base {
                    r /= base;
                    n += 1;
                }
                n
            }

            fn div_ceil(self, other: Self) -> Self {
                (self + other - 1) / other
            }

            fn next_multiple_of(self, m: Self) -> Self {
                ((self + m - 1) / m) * m
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

pub fn high_precision_sleep(duration: Duration, single_sleep_period: Duration) {
    let end_t = Instant::now() + duration;

    while Instant::now() < end_t {
        thread::sleep(single_sleep_period);
    }
}

pub unsafe fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    slice::from_raw_parts(slice.as_ptr() as *const u8, mem::size_of::<T>() * slice.len())
}

pub trait ConcurrentCacheImpl<K, V> {
    fn new(max_capacity: usize) -> ConcurrentCache<K, V>;
    fn with_weigher(
        max_capacity: usize,
        weigher: impl Fn(&K, &V) -> u32 + Send + Sync + 'static,
    ) -> ConcurrentCache<K, V>;
}

impl<K, V> ConcurrentCacheImpl<K, V> for ConcurrentCache<K, V>
where
    K: Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn new(max_capacity: usize) -> ConcurrentCache<K, V> {
        moka::sync::CacheBuilder::new(max_capacity as u64).build_with_hasher(ahash::RandomState::new())
    }

    fn with_weigher(
        max_capacity: usize,
        weigher: impl Fn(&K, &V) -> u32 + Send + Sync + 'static,
    ) -> ConcurrentCache<K, V> {
        moka::sync::CacheBuilder::new(max_capacity as u64)
            .weigher(weigher)
            .build_with_hasher(ahash::RandomState::new())
    }
}

#[macro_export]
macro_rules! unwrap_option {
    ($to_unwrap: expr, $on_else: expr) => {
        if let Some(v) = $to_unwrap {
            v
        } else {
            $on_else
        }
    };
}
