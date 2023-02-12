use std::hash::Hash;

pub type HashSet<T> = ahash::AHashSet<T>;
pub type HashMap<K, V> = ahash::AHashMap<K, V>;
pub type ConcurrentCache<K, V> = moka::sync::Cache<K, V, ahash::RandomState>;
pub type IndexSet<T> = indexmap::IndexSet<T, ahash::RandomState>;
pub type IndexMap<T, V> = indexmap::IndexMap<T, V, ahash::RandomState>;

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
    // TODO: remove when std div_ceil is stable
    fn div_ceil(self, other: Self) -> Self;
    // TODO: remove when std next_multiple_of is stable
    fn next_multiple_of(self, m: Self) -> Self;
}

pub trait Int {}

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
macro_rules! int_impl {
    ($($t: ty)*) => ($(
        impl Int for $t {

        }
    )*)
}

uint_impl! { u8 u16 u32 u64 }
int_impl! { i8 i16 i32 i64 }
