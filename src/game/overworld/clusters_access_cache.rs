use std::collections::{hash_map, VecDeque};
use std::sync::Arc;

use entity_data::ArchetypeState;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, U32Vec3};
use parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockWriteGuard};
use parking_lot::{RawRwLock, RwLock};

use engine::utils::{HashMap, MO_RELAXED};

use crate::game::overworld;
use crate::game::overworld::block::{Block, BlockState};
use crate::game::overworld::facing::Facing;
use crate::game::overworld::light_level::LightLevel;
use crate::game::overworld::position::{BlockPos, ClusterBlockPos, ClusterPos};
use crate::game::overworld::raw_cluster::{BlockData, BlockDataImpl, BlockDataMut, RawCluster};
use crate::game::overworld::{
    block_component, raw_cluster, Cluster, LoadedClusters, OverworldCluster, CLUSTER_STATE_LOADED,
    CLUSTER_STATE_OFFLOADED_INVISIBLE,
};
use crate::game::registry::Registry;

pub struct AccessGuard {
    o_cluster: Arc<OverworldCluster>,
    lock: AccessGuardLock,
}

pub enum AccessGuardLock {
    Read(ArcRwLockReadGuard<RawRwLock, Option<Cluster>>),
    Write(ArcRwLockWriteGuard<RawRwLock, Option<Cluster>>),
}

impl AccessGuard {
    #[inline]
    pub fn cluster(&self) -> &Cluster {
        match &self.lock {
            AccessGuardLock::Read(g) => g.as_ref().unwrap(),
            AccessGuardLock::Write(g) => g.as_ref().unwrap(),
        }
    }

    #[inline]
    pub fn cluster_mut(&mut self) -> &mut Cluster {
        match &mut self.lock {
            AccessGuardLock::Write(g) => g.as_mut().unwrap(),
            _ => unreachable!(),
        }
    }
}

pub struct ClustersAccessor {
    pub(super) registry: Arc<Registry>,
    pub(super) loaded_clusters: LoadedClusters,
    pub(super) clusters_cache: HashMap<ClusterPos, AccessGuard>,
}

impl ClustersAccessor {
    /// Returns cluster for the specified global block position
    pub fn access_cluster(&mut self, cluster_pos: &ClusterPos) -> Option<&AccessGuard> {
        match self.clusters_cache.entry(*cluster_pos) {
            hash_map::Entry::Vacant(e) => {
                let clusters = self.loaded_clusters.read();
                let o_cluster = Arc::clone(clusters.get(cluster_pos)?);
                let state = o_cluster.state();

                if state != CLUSTER_STATE_LOADED && state != CLUSTER_STATE_OFFLOADED_INVISIBLE {
                    return None;
                }

                let lock = AccessGuardLock::Read(o_cluster.cluster.read_arc());

                Some(e.insert(AccessGuard { o_cluster, lock }))
            }
            hash_map::Entry::Occupied(e) => Some(e.into_mut()),
        }
    }

    /// Returns cluster for the specified global block position
    pub fn access_cluster_mut(&mut self, cluster_pos: &ClusterPos) -> Option<&mut AccessGuard> {
        match self.clusters_cache.entry(*cluster_pos) {
            hash_map::Entry::Vacant(e) => {
                let clusters = self.loaded_clusters.read();
                let o_cluster = Arc::clone(clusters.get(cluster_pos)?);

                if o_cluster.state() != CLUSTER_STATE_LOADED {
                    None
                } else {
                    o_cluster.dirty.store(true, MO_RELAXED);

                    let lock = AccessGuardLock::Write(o_cluster.cluster.write_arc());

                    Some(e.insert(AccessGuard { o_cluster, lock }))
                }
            }
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
        }
    }

    /// Returns block data or `None` if respective cluster is not loaded
    pub fn get_block(&mut self, pos: &BlockPos) -> Option<BlockData> {
        if let Some(access) = self.access_cluster(&pos.cluster_pos()) {
            Some(access.cluster().raw.get(&pos.cluster_block_pos()))
        } else {
            None
        }
    }

    /// Returns block data or `None` if respective cluster is not loaded
    pub fn update_block<F: FnOnce(&mut BlockDataMut)>(&mut self, pos: &BlockPos, update_fn: F) -> bool {
        if let Some(access) = self.access_cluster_mut(&pos.cluster_pos()) {
            let cluster = access.cluster_mut();
            let cluster_block_pos = pos.cluster_block_pos();
            let mut block_data = cluster.raw.get_mut(&cluster_block_pos);

            update_fn(&mut block_data);

            let active = block_data
                .get::<block_component::Activity>()
                .map_or(false, |v| v.active);

            cluster.active_blocks.set(cluster_block_pos.index(), active);
            cluster.may_have_active_blocks |= active;

            true
        } else {
            false
        }
    }

    /// Returns false if respective cluster is not loaded
    pub fn set_block<A: ArchetypeState>(&mut self, pos: &BlockPos, state: BlockState<A>) -> bool {
        if let Some(access) = self.access_cluster_mut(&pos.cluster_pos()) {
            let cluster_block_pos = pos.cluster_block_pos();
            let cluster = access.cluster_mut();
            let mut block_data = cluster.raw.set(&cluster_block_pos, state);

            // Determine activity
            let active = block_data
                .get::<block_component::Activity>()
                .map_or(false, |v| v.active);

            cluster.active_blocks.set(cluster_block_pos.index(), active);
            cluster.may_have_active_blocks |= active;
            cluster.dirty_parts.set_dirty(&cluster_block_pos);

            true
        } else {
            false
        }
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    fn get_light_level(&mut self, pos: &BlockPos) -> Option<LightLevel> {
        let cluster = self.access_cluster(&pos.cluster_pos())?;
        Some(cluster.cluster().raw.get_light_level(&pos.cluster_block_pos()))
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    fn set_light_level(&mut self, pos: &BlockPos, light_level: LightLevel) {
        if let Some(access) = self.access_cluster_mut(&pos.cluster_pos()) {
            let cluster_block_pos = pos.cluster_block_pos();
            let cluster = access.cluster_mut();
            cluster.raw.set_light_level(&cluster_block_pos, light_level);
            cluster.dirty_parts.set_dirty(&cluster_block_pos);
        }
    }

    /// Returns facing, intersection point, and block data of the block that intersects with specified ray.
    pub fn get_block_at_ray(
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

                if block.has_textured_model() {
                    let model = registry.get_textured_block_model(block.textured_model()).unwrap();
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

    fn propagate_light_addition(&mut self, queue: &mut VecDeque<BlockPos>) {
        while let Some(curr_pos) = queue.pop_front() {
            let curr_level = self.get_light_level(&curr_pos).unwrap();
            let curr_color = curr_level.components();

            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let rel_pos = curr_pos.offset(&dir);

                let block_id = self.get_block(&rel_pos).unwrap().block_id();
                let block = *self.registry.get_block(block_id).unwrap();
                let level = self.get_light_level(&rel_pos).unwrap();
                let color = level.components();

                if !self.registry.is_block_opaque(&block)
                    && glm::any(&color.add_scalar(2).zip_map(&curr_color, |a, b| a <= b))
                {
                    let new_color = curr_color.map(|v| v.saturating_sub(1));
                    self.set_light_level(&rel_pos, LightLevel::from_vec(new_color));

                    queue.push_back(rel_pos);
                }
            }
        }
    }

    /// Use breadth-first search to set lighting across all lit area
    pub fn set_light(&mut self, pos: &BlockPos, light_level: LightLevel) {
        if let Some(cluster) = self.access_cluster_mut(&pos.cluster_pos()) {
            let cluster = cluster.cluster_mut();
            let cluster_block_pos = pos.cluster_block_pos();

            cluster.raw.set_light_level(&cluster_block_pos, light_level);
            cluster.raw.propagate_lighting(&cluster_block_pos);
        }
    }

    pub fn remove_light(&mut self, global_pos: &BlockPos) {
        let curr_level = self.get_light_level(&global_pos).unwrap();
        self.set_light_level(global_pos, LightLevel::zero());

        let mut removal_queue = VecDeque::with_capacity((curr_level.components().max() as usize * 2).pow(3));
        let mut addition_queue = VecDeque::with_capacity(removal_queue.capacity());

        removal_queue.push_back((*global_pos, curr_level));

        while let Some((curr_pos, curr_level)) = removal_queue.pop_front() {
            let curr_color = curr_level.components();

            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let rel_pos = curr_pos.offset(&dir);

                let level = self.get_light_level(&rel_pos).unwrap();
                let color = level.components();

                if glm::any(&color.zip_map(&curr_color, |a, b| a < b)) {
                    if !level.is_zero() {
                        self.set_light_level(&rel_pos, LightLevel::zero());
                        removal_queue.push_back((rel_pos, level));
                    }
                }

                if glm::any(&color.zip_map(&curr_color, |a, b| a >= b)) {
                    addition_queue.push_back(rel_pos);
                }
            }
        }

        for pos in addition_queue {
            let cluster = self.access_cluster_mut(&pos.cluster_pos()).unwrap();
            let cluster_block_pos = pos.cluster_block_pos();
            cluster.cluster_mut().raw.propagate_lighting(&cluster_block_pos);
        }
    }

    /// Useful for restoring lighting from neighbours when removing block at `block_pos`.
    pub fn check_neighbour_lighting(&mut self, pos: &BlockPos) {
        for i in 0..6 {
            let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
            let rel_pos = pos.offset(&dir);

            let cluster = self.access_cluster_mut(&pos.cluster_pos());

            if let Some(cluster) = cluster {
                let cluster = cluster.cluster_mut();
                let cluster_block_pos = rel_pos.cluster_block_pos();
                let level = cluster.raw.get_light_level(&cluster_block_pos);

                if !level.is_zero() {
                    cluster.raw.propagate_lighting(&cluster_block_pos);
                }
            }
        }
    }
}
