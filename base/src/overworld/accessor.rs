use crate::overworld::facing::Facing;
use crate::overworld::light_state::LightLevel;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::{BlockPos, ClusterPos, RelativeBlockPos};
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl, BlockDataMut};
use crate::overworld::{is_cell_empty, ClusterState, LoadedClusters, TrackingCluster};
use crate::registry::Registry;
use common::glm::TVec3;
use common::nd_range::NDRange;
use common::parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockWriteGuard};
use common::parking_lot::RawRwLock;
use common::types::Bool;
use common::types::HashMap;
use common::{glm, MO_RELAXED};
use entity_data::EntityStorage;
use glm::{DVec3, I64Vec3};
use lazy_static::lazy_static;
use std::collections::hash_map;
use std::sync::Arc;

lazy_static! {
    static ref EMPTY_BLOCK_STORAGE: EntityStorage = EntityStorage::new();
}

pub struct AccessGuard {
    locked_state: ClusterState,
    lock: AccessGuardLock,
}

pub enum AccessGuardLock {
    Read(ArcRwLockReadGuard<RawRwLock, Option<TrackingCluster>>),
    Write(ArcRwLockWriteGuard<RawRwLock, Option<TrackingCluster>>),
}

impl AccessGuard {
    /// Returns cluster for the specified global block position.
    /// Returns `None` if the cluster is offloaded.
    #[inline]
    pub fn cluster(&self) -> Option<&TrackingCluster> {
        match &self.lock {
            AccessGuardLock::Read(g) => g.as_ref(),
            AccessGuardLock::Write(g) => g.as_ref(),
        }
    }

    /// Returns cluster for the specified global block position.
    /// Returns `None` if the cluster is offloaded.
    #[inline]
    pub fn cluster_mut(&mut self) -> Option<&mut TrackingCluster> {
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
                let o_cluster = clusters.get(cluster_pos)?;
                let state = o_cluster.state();

                if !state.is_loaded() {
                    return None;
                }

                let lock = AccessGuardLock::Read(o_cluster.cluster.read_arc());

                Some(e.insert(AccessGuard {
                    locked_state: state,
                    lock,
                }))
            }
        }
    }

    /// Returns `None` if the cluster is not loaded yet.
    pub fn access_cluster_mut(&mut self, cluster_pos: &ClusterPos) -> Option<&mut AccessGuard> {
        match self.clusters_cache.entry(*cluster_pos) {
            hash_map::Entry::Occupied(e) => {
                let guard = e.into_mut();

                if let AccessGuardLock::Read(_) = &guard.lock {
                    replace_with::replace_with_or_abort(&mut guard.lock, |v| {
                        let read_guard = if let AccessGuardLock::Read(v) = v {
                            v
                        } else {
                            unreachable!()
                        };

                        let rwlock = Arc::clone(&ArcRwLockReadGuard::rwlock(&read_guard));
                        drop(read_guard);

                        let clusters = self.loaded_clusters.read();
                        let ocluster = clusters.get(cluster_pos).unwrap();
                        ocluster.dirty.store(true, MO_RELAXED);

                        AccessGuardLock::Write(rwlock.write_arc())
                    });
                }

                Some(guard)
            }
            hash_map::Entry::Vacant(e) => {
                let clusters = self.loaded_clusters.read();
                let o_cluster = clusters.get(cluster_pos)?;
                let state = o_cluster.state();

                if state != ClusterState::Loaded {
                    None
                } else {
                    o_cluster.dirty.store(true, MO_RELAXED);

                    let lock = AccessGuardLock::Write(o_cluster.cluster.write_arc());

                    Some(e.insert(AccessGuard {
                        locked_state: state,
                        lock,
                    }))
                }
            }
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
            (pos + (dir > 0.0).into_f64() - origin) / dir
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
            let pos_idx = cluster_block_pos.index();
            let mut block_data = cluster.raw.get_mut(&cluster_block_pos);

            update_fn(&mut block_data);
            let active_after_update = block_data.active();
            let empty_after_update = is_cell_empty(&self.registry, block_data);

            cluster.active_cells.set(pos_idx, active_after_update);
            cluster.empty_cells.set(pos_idx, empty_after_update);
            cluster.dirty_parts.set_from_block_pos(&cluster_block_pos);

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
            cluster.active_cells.set(cluster_block_pos.index(), data.active());

            cluster.dirty_parts.set_from_block_pos(&cluster_block_pos);
        }
    }

    /// Sets light level if the respective cluster is loaded
    pub fn set_light_source(&mut self, pos: &BlockPos, light_level: LightLevel) {
        if let Some(cluster) = self
            .cache
            .access_cluster_mut(&pos.cluster_pos())
            .map(|access| access.cluster_mut())
            .flatten()
        {
            let cluster_block_pos = pos.cluster_block_pos();
            let mut data = cluster.raw.get_mut(&cluster_block_pos);

            *data.light_source_mut() = light_level;

            // The light must spread or vanish
            *data.active_mut() = true;
            cluster.active_cells.set(cluster_block_pos.index(), data.active());

            cluster.dirty_parts.set_from_block_pos(&cluster_block_pos);
        }
    }

    /// Sets light level if the respective cluster is loaded
    pub fn set_light_state(&mut self, pos: &BlockPos, light_level: LightLevel) {
        if let Some(cluster) = self
            .cache
            .access_cluster_mut(&pos.cluster_pos())
            .map(|access| access.cluster_mut())
            .flatten()
        {
            let cluster_block_pos = pos.cluster_block_pos();
            let mut data = cluster.raw.get_mut(&cluster_block_pos);

            *data.light_state_mut() = light_level;

            cluster.dirty_parts.set_from_block_pos(&cluster_block_pos);
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

pub struct ClusterNeighbourhoodAccessor {
    registry: Arc<Registry>,
    neighbours: [Option<AccessGuard>; 3 * 3 * 3],
}

impl ClusterNeighbourhoodAccessor {
    pub fn new(registry: Arc<Registry>, loaded_clusters: &LoadedClusters, pos: ClusterPos) -> Self {
        let o_clusters = loaded_clusters.read();

        let neighbours: Vec<_> = NDRange::of_size(TVec3::from_element(3))
            .into_iter()
            .map(|offset| {
                let offset: I64Vec3 = glm::convert(offset);
                let rel_pos = pos.offset(&offset).offset(&I64Vec3::from_element(-1));
                let o_cluster = o_clusters.get(&rel_pos)?;
                let state = o_cluster.state();

                if !state.is_loaded() {
                    return None;
                }

                let t_cluster = o_cluster.cluster.read_arc();

                if t_cluster.is_none() {
                    return None;
                }

                Some(AccessGuard {
                    locked_state: state,
                    lock: AccessGuardLock::Read(t_cluster),
                })
            })
            .collect();

        Self {
            registry,
            neighbours: neighbours.try_into().ok().unwrap(),
        }
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    pub fn is_center_available(&self) -> bool {
        let center = self.neighbours[1 * 9 + 1 * 3 + 1].as_ref();
        center.is_some_and(|v| v.cluster().is_some())
    }

    pub fn get_block(&self, pos: &RelativeBlockPos) -> Option<BlockData> {
        let neighbour_pos = pos.cluster_idx();

        let access = self.neighbours[neighbour_pos].as_ref()?;
        let t_cluster = access.cluster().unwrap();

        let cluster_block_pos = pos.cluster_block_pos();
        Some(t_cluster.raw.get(&cluster_block_pos))
    }
}
