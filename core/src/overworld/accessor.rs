use std::collections::{hash_map, VecDeque};
use std::sync::Arc;

use entity_data::EntityStorage;
use lazy_static::lazy_static;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3};
use parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockWriteGuard};
use parking_lot::RawRwLock;

use crate::overworld::facing::Facing;
use crate::overworld::light_level::LightLevel;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::{BlockPos, ClusterPos};
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl, BlockDataMut};
use crate::overworld::{Cluster, ClusterState, LoadedClusters, OverworldCluster};
use crate::registry::Registry;
use crate::utils::{HashMap, MO_RELAXED};

lazy_static! {
    static ref EMPTY_BLOCK_STORAGE: EntityStorage = EntityStorage::new();
}

pub struct AccessGuard {
    o_cluster: Arc<OverworldCluster>,
    locked_state: ClusterState,
    lock: AccessGuardLock,
}

pub enum AccessGuardLock {
    Read(ArcRwLockReadGuard<RawRwLock, Option<Cluster>>),
    Write(ArcRwLockWriteGuard<RawRwLock, Option<Cluster>>),
}

impl AccessGuard {
    /// Returns cluster for the specified global block position.
    /// Returns `None` if the cluster is offloaded.
    #[inline]
    pub fn cluster(&self) -> Option<&Cluster> {
        match &self.lock {
            AccessGuardLock::Read(g) => g.as_ref(),
            AccessGuardLock::Write(g) => g.as_ref(),
        }
    }

    /// Returns cluster for the specified global block position.
    /// Returns `None` if the cluster is offloaded.
    #[inline]
    pub fn cluster_mut(&mut self) -> Option<&mut Cluster> {
        match &mut self.lock {
            AccessGuardLock::Write(g) => g.as_mut(),
            _ => unreachable!(),
        }
    }
}

pub struct ClustersAccessorCache {
    loaded_clusters: LoadedClusters,
    clusters_cache: HashMap<ClusterPos, AccessGuard>,
}

impl ClustersAccessorCache {
    pub fn new(loaded_clusters: LoadedClusters) -> Self {
        Self {
            loaded_clusters,
            clusters_cache: HashMap::with_capacity(32),
        }
    }

    /// Returns `None` if the cluster is not loaded yet.
    pub fn access_cluster(&mut self, cluster_pos: &ClusterPos) -> Option<&AccessGuard> {
        match self.clusters_cache.entry(*cluster_pos) {
            hash_map::Entry::Occupied(e) => Some(e.into_mut()),
            hash_map::Entry::Vacant(e) => {
                let clusters = self.loaded_clusters.read();
                let o_cluster = Arc::clone(clusters.get(cluster_pos)?);
                let state = o_cluster.state();

                if !state.is_readable() {
                    return None;
                }

                let lock = AccessGuardLock::Read(o_cluster.cluster.try_read_arc()?);

                Some(e.insert(AccessGuard {
                    o_cluster,
                    locked_state: state,
                    lock,
                }))
            }
        }
    }

    /// Returns `None` if the cluster is not loaded yet.
    pub fn access_cluster_mut(&mut self, cluster_pos: &ClusterPos) -> Option<&mut AccessGuard> {
        // Fast path
        if let hash_map::Entry::Occupied(e) = self.clusters_cache.entry(*cluster_pos) {
            if let AccessGuardLock::Write(_) = &e.get().lock {
                let mut_ref = e.into_mut();

                // TODO: use NLL when https://github.com/rust-lang/rust/issues/51545 is fixed
                // return Some(mut_ref);
                return Some(unsafe { &mut *(mut_ref as *mut _) });
            } else {
                e.remove();
            }
        }

        // Slow path: lock for writes
        let clusters = self.loaded_clusters.read();
        let o_cluster = Arc::clone(clusters.get(cluster_pos)?);
        let state = o_cluster.state();

        if state != ClusterState::Loaded {
            None
        } else {
            let lock = AccessGuardLock::Write(o_cluster.cluster.try_write_arc()?);

            o_cluster.dirty.store(true, MO_RELAXED);

            Some(self.clusters_cache.entry(*cluster_pos).or_insert(AccessGuard {
                o_cluster,
                locked_state: state,
                lock,
            }))
        }
    }
}

pub trait ReadOnlyOverworldAccessorImpl {
    /// Returns block data or `None` if the respective cluster is not loaded
    fn get_block(&mut self, pos: &BlockPos) -> Option<BlockData>;

    /// Returns intersection facing and position of the block that specified ray intersect.
    fn get_block_at_ray(
        &mut self,
        ray_origin: &DVec3,
        ray_dir: &DVec3,
        max_ray_length: f64,
    ) -> Option<(BlockPos, Facing)>;
}

pub struct OverworldAccessor {
    registry: Arc<Registry>,
    cache: ClustersAccessorCache,
}

impl ReadOnlyOverworldAccessorImpl for OverworldAccessor {
    /// Returns block data or `None` if the respective cluster is not loaded
    fn get_block(&mut self, pos: &BlockPos) -> Option<BlockData> {
        self.cache
            .access_cluster(&pos.cluster_pos())
            .map(|access| match access.cluster() {
                Some(cluster) => Some(cluster.raw.get(&pos.cluster_block_pos())),
                None if access.locked_state == ClusterState::OffloadedEmpty => Some(BlockData {
                    block_storage: &EMPTY_BLOCK_STORAGE,
                    info: self.registry.inner_block_state_empty(),
                }),
                _ => None,
            })
            .flatten()
    }

    /// Returns intersection facing and position of the block that specified ray intersect.
    fn get_block_at_ray(
        &mut self,
        ray_origin: &DVec3,
        ray_dir: &DVec3,
        max_ray_length: f64,
    ) -> Option<(BlockPos, Facing)> {
        assert!(ray_dir.magnitude_squared() > 0.0);

        let registry = Arc::clone(&self.registry);
        let step = ray_dir.map(|v| v.signum());
        let step_dt = step.component_div(ray_dir);

        let curr_origin = *ray_origin;
        let mut curr_block_pos = glm::floor(ray_origin);
        let mut dt = curr_block_pos.zip_zip_map(&curr_origin, ray_dir, |pos, origin, dir| {
            (pos + (dir > 0.0) as i64 as f64 - origin) / dir
        });
        let mut t = 0.0;

        loop {
            let curr_block_upos = BlockPos::from_f64(&curr_block_pos);
            let curr_block = self.get_block(&curr_block_upos);

            if let Some(data) = &curr_block {
                let block = registry.get_block(data.block_id()).unwrap();

                if !block.is_model_invisible() {
                    let model = registry.get_block_model(block.model_id()).unwrap();
                    let mut closest_inter = None;
                    let mut min_length = f64::MAX;

                    for aabb in model.aabbs() {
                        if let Some(inter) = aabb
                            .translate(curr_block_pos)
                            .ray_intersection(ray_origin, ray_dir)
                        {
                            let len = (inter.point() - ray_origin).magnitude();

                            if len < min_length {
                                min_length = len;
                                closest_inter = Some(inter);
                            }
                        }
                    }

                    if min_length > max_ray_length {
                        return None;
                    }
                    if let Some(inter) = closest_inter {
                        return Some((curr_block_upos, inter.facing()));
                    }
                }
            } else {
                return None;
            }

            let min_i = dt.imin();
            curr_block_pos[min_i] += step[min_i];
            t += dt[min_i];
            dt.add_scalar_mut(-dt[min_i]);
            dt[min_i] += step_dt[min_i];

            if t > max_ray_length {
                return None;
            }
        }
    }
}

impl OverworldAccessor {
    pub fn new(registry: Arc<Registry>, loaded_clusters: LoadedClusters) -> Self {
        Self {
            registry,
            cache: ClustersAccessorCache::new(loaded_clusters),
        }
    }

    pub fn into_read_only(self) -> ReadOnlyOverworldAccessor {
        ReadOnlyOverworldAccessor { inner: self }
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    pub fn cache_mut(&mut self) -> &mut ClustersAccessorCache {
        &mut self.cache
    }

    /// Returns block data or `None` if the respective cluster is not loaded
    pub fn update_block<F: FnOnce(&mut BlockDataMut)>(&mut self, pos: &BlockPos, update_fn: F) -> bool {
        if let Some(cluster) = self
            .cache
            .access_cluster_mut(&pos.cluster_pos())
            .map(|access| access.cluster_mut())
            .flatten()
        {
            let cluster_block_pos = pos.cluster_block_pos();
            let mut block_data = cluster.raw.get_mut(&cluster_block_pos);

            update_fn(&mut block_data);
            let active_after_update = block_data.active();

            cluster
                .active_blocks
                .set(cluster_block_pos.index(), active_after_update);
            cluster.dirty_parts.set_dirty(&cluster_block_pos);

            true
        } else {
            false
        }
    }

    /// Sets liquid state if the respective cluster is loaded. Activates the cell if the liquid is a source.
    pub fn set_liquid_state(&mut self, pos: &BlockPos, liquid: LiquidState) {
        if let Some(cluster) = self
            .cache
            .access_cluster_mut(&pos.cluster_pos())
            .map(|access| access.cluster_mut())
            .flatten()
        {
            let cluster_block_pos = pos.cluster_block_pos();
            let mut data = cluster.raw.get_mut(&cluster_block_pos);

            *data.liquid_state_mut() = liquid;

            // The liquid must spread or vanish
            *data.active_mut() = true;
            cluster
                .active_blocks
                .set(cluster_block_pos.index(), data.active());

            cluster.dirty_parts.set_dirty(&cluster_block_pos);
        }
    }

    /// Sets light level if the respective cluster is loaded
    fn set_light_level(&mut self, pos: &BlockPos, light_level: LightLevel) {
        if let Some(cluster) = self
            .cache
            .access_cluster_mut(&pos.cluster_pos())
            .map(|access| access.cluster_mut())
            .flatten()
        {
            let cluster_block_pos = pos.cluster_block_pos();
            let mut data = cluster.raw.get_mut(&cluster_block_pos);

            *data.light_level_mut() = light_level;

            cluster.dirty_parts.set_dirty(&cluster_block_pos);
        }
    }

    fn propagate_light_addition(&mut self, queue: &mut VecDeque<BlockPos>) {
        while let Some(curr_pos) = queue.pop_front() {
            let block = self.get_block(&curr_pos).unwrap();
            let curr_level = block.light_level();
            let curr_color = curr_level.components();

            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let rel_pos = curr_pos.offset(&dir);

                let block_id = self.get_block(&rel_pos).unwrap().block_id();
                let block = *self.registry.get_block(block_id).unwrap();
                let data = self.get_block(&rel_pos).unwrap();
                let level = data.light_level();
                let color = level.components();

                if !block.is_opaque() && glm::any(&color.add_scalar(2).zip_map(&curr_color, |a, b| a <= b)) {
                    let new_color = curr_color.map(|v| v.saturating_sub(1));
                    self.set_light_level(&rel_pos, LightLevel::from_vec(new_color));

                    queue.push_back(rel_pos);
                }
            }
        }
    }

    /// Use breadth-first search to set lighting across all lit area
    pub fn set_light(&mut self, pos: &BlockPos, light_level: LightLevel) {
        if let Some(cluster) = self
            .cache
            .access_cluster_mut(&pos.cluster_pos())
            .map(|accesss| accesss.cluster_mut())
            .flatten()
        {
            let cluster_block_pos = pos.cluster_block_pos();
            let mut data = cluster.raw.get_mut(&cluster_block_pos);
            *data.light_level_mut() = light_level;
            cluster.propagate_lighting(&cluster_block_pos);
        }
    }

    pub fn remove_light(&mut self, global_pos: &BlockPos) {
        let curr_data = self.get_block(&global_pos).unwrap();
        let curr_level = curr_data.light_level();
        self.set_light_level(global_pos, LightLevel::ZERO);

        let mut removal_queue = VecDeque::with_capacity((curr_level.components().max() as usize * 2).pow(3));
        let mut addition_queue = VecDeque::with_capacity(removal_queue.capacity());

        removal_queue.push_back((*global_pos, curr_level));

        while let Some((curr_pos, curr_level)) = removal_queue.pop_front() {
            let curr_color = curr_level.components();

            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let rel_pos = curr_pos.offset(&dir);

                let level = self.get_block(&rel_pos).unwrap().light_level();
                let color = level.components();

                if glm::any(&color.zip_map(&curr_color, |a, b| a < b)) {
                    if !level.is_zero() {
                        self.set_light_level(&rel_pos, LightLevel::ZERO);
                        removal_queue.push_back((rel_pos, level));
                    }
                }

                if glm::any(&color.zip_map(&curr_color, |a, b| a >= b)) {
                    addition_queue.push_back(rel_pos);
                }
            }
        }

        for pos in addition_queue {
            let cluster = self.cache.access_cluster_mut(&pos.cluster_pos()).unwrap();
            let cluster_block_pos = pos.cluster_block_pos();
            if let Some(cluster) = cluster.cluster_mut() {
                cluster.propagate_lighting(&cluster_block_pos);
            }
        }
    }

    /// Useful for restoring lighting from neighbours when removing block at `block_pos`.
    pub fn check_neighbour_lighting(&mut self, pos: &BlockPos) {
        for i in 0..6 {
            let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
            let rel_pos = pos.offset(&dir);

            let cluster = self.cache.access_cluster_mut(&pos.cluster_pos());

            if let Some(cluster) = cluster.map(|access| access.cluster_mut()).flatten() {
                let cluster_block_pos = rel_pos.cluster_block_pos();
                let level = cluster.raw.get(&cluster_block_pos).light_level();

                if !level.is_zero() {
                    cluster.propagate_lighting(&cluster_block_pos);
                }
            }
        }
    }
}

pub struct ReadOnlyOverworldAccessor {
    inner: OverworldAccessor,
}

impl ReadOnlyOverworldAccessorImpl for ReadOnlyOverworldAccessor {
    fn get_block(&mut self, pos: &BlockPos) -> Option<BlockData> {
        self.inner.get_block(pos)
    }

    fn get_block_at_ray(
        &mut self,
        ray_origin: &DVec3,
        ray_dir: &DVec3,
        max_ray_length: f64,
    ) -> Option<(BlockPos, Facing)> {
        self.inner.get_block_at_ray(ray_origin, ray_dir, max_ray_length)
    }
}
