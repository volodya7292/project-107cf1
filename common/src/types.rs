use ahash::{AHashMap, AHashSet};
use std::hash::Hash;

pub type Hasher = ahash::RandomState;
pub type HashSet<T> = AHashSet<T>;
pub type HashMap<K, V> = AHashMap<K, V>;
pub type ConcurrentCache<K, V> = moka::sync::Cache<K, V, Hasher>;
pub type IndexSet<T> = indexmap::IndexSet<T, Hasher>;
pub type IndexMap<T, V> = indexmap::IndexMap<T, V, Hasher>;

pub trait ConcurrentCacheExt<K, V> {
    fn new(max_capacity: usize) -> ConcurrentCache<K, V>;
    fn with_weigher(
        max_capacity: usize,
        weigher: impl Fn(&K, &V) -> u32 + Send + Sync + 'static,
    ) -> ConcurrentCache<K, V>;
}

impl<K, V> ConcurrentCacheExt<K, V> for ConcurrentCache<K, V>
where
    K: Eq + Hash + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn new(max_capacity: usize) -> ConcurrentCache<K, V> {
        moka::sync::CacheBuilder::new(max_capacity as u64).build_with_hasher(Hasher::new())
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

pub trait UInt {
    // TODO: remove when std div_ceil is stable
    fn div_ceil(self, other: Self) -> Self;
    // TODO: remove when std next_multiple_of is stable
    fn next_multiple_of(self, m: Self) -> Self;
}

macro_rules! uint_impl {
    ($($t: ty)*) => ($(
        impl UInt for $t {
            fn div_ceil(self, other: Self) -> Self {
                (self + other - 1) / other
            }

            fn next_multiple_of(self, m: Self) -> Self {
                ((self + m - 1) / m) * m
            }
        }
    )*)
}

uint_impl! { u8 u16 u32 u64 }
