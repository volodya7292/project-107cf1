use crate::types::{ConcurrentCache, ConcurrentCacheExt};
use image::Rgba;
use std::any::{Any, TypeId};
use std::sync::Arc;

pub trait CachedResource: Send + Sync + 'static {
    fn footprint(&self) -> usize;
    fn as_any(&self) -> &dyn Any;
}

impl CachedResource for Vec<u8> {
    fn footprint(&self) -> usize {
        self.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CachedResource for image::ImageBuffer<Rgba<u8>, Vec<u8>> {
    fn footprint(&self) -> usize {
        self.as_raw().len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl<R: CachedResource> CachedResource for Arc<R> {
    fn footprint(&self) -> usize {
        R::footprint(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct ResourceCache {
    cache: ConcurrentCache<(String, TypeId), Arc<dyn CachedResource>>,
}

impl ResourceCache {
    const CAPACITY: usize = 100_000_000;

    fn cache_entry_weigher(_: &(String, TypeId), value: &Arc<dyn CachedResource>) -> u32 {
        value.footprint().min(u32::MAX as usize) as u32
    }

    pub fn new() -> Self {
        Self {
            cache: ConcurrentCache::with_weigher(Self::CAPACITY, Self::cache_entry_weigher),
        }
    }

    pub fn get<T: CachedResource, E: Sync + Send + 'static, F: FnOnce() -> Result<Arc<T>, E>>(
        &self,
        path: &str,
        init: F,
    ) -> Result<Arc<T>, E> {
        let entry = self
            .cache
            .entry((path.to_string(), TypeId::of::<T>()))
            .or_try_insert_with(|| init().map(|v: Arc<T>| Arc::new(v) as Arc<dyn CachedResource>))
            .map_err(|e| Arc::into_inner(e).unwrap())?;

        Ok(Arc::clone(
            entry.into_value().as_any().downcast_ref::<Arc<T>>().unwrap(),
        ))
    }
}

impl Default for ResourceCache {
    fn default() -> Self {
        Self::new()
    }
}