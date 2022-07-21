use crate::game::overworld;
use crate::game::overworld::block::Block;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::{BlockData, BlockDataBuilder, BlockDataImpl, Cluster};
use crate::game::overworld::light_level::LightLevel;
use crate::game::overworld::{
    cluster, OverworldCluster, CLUSTER_STATE_LOADED, CLUSTER_STATE_OFFLOADED_INVISIBLE,
};
use crate::game::registry::Registry;
use crate::glm;
use engine::utils::{HashMap, MO_RELAXED};
use nalgebra_glm::{DVec3, I64Vec3, U32Vec3};
use parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockWriteGuard};
use parking_lot::{RawRwLock, RwLock};
use std::collections::{hash_map, VecDeque};
use std::sync::Arc;

pub struct AccessGuard {
    lock: AccessGuardLock,
}

pub enum AccessGuardLock {
    Read(ArcRwLockReadGuard<RawRwLock, Option<Cluster>>),
    Write(ArcRwLockWriteGuard<RawRwLock, Option<Cluster>>),
}

impl AccessGuard {
    pub fn get_cluster_mut(&mut self) -> &mut Cluster {
        match &mut self.lock {
            AccessGuardLock::Write(g) => g.as_mut().unwrap(),
            _ => unreachable!(),
        }
    }

    #[inline]
    pub fn get(&self, pos: &U32Vec3) -> BlockData {
        match &self.lock {
            AccessGuardLock::Read(g) => {
                if let Some(cluster) = g.as_ref() {
                    cluster.get(pos)
                } else {
                    BlockData::empty()
                }
            }
            AccessGuardLock::Write(g) => {
                if let Some(cluster) = g.as_ref() {
                    cluster.get(pos)
                } else {
                    BlockData::empty()
                }
            }
        }
    }

    #[inline]
    pub fn set(&mut self, pos: &U32Vec3, block: Block) -> BlockDataBuilder {
        match &mut self.lock {
            AccessGuardLock::Write(g) => g.as_mut().unwrap().set(pos, block),
            _ => unreachable!(),
        }
    }
}

pub struct ClustersAccessCache {
    pub(super) registry: Arc<Registry>,
    pub(super) loaded_clusters: Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>>,
    pub(super) clusters_cache: HashMap<I64Vec3, AccessGuard>,
}

impl ClustersAccessCache {
    /// Returns cluster for the specified global block position
    pub fn get_cluster_for_block_mut(&mut self, global_pos: &I64Vec3) -> Option<&mut Cluster> {
        let cluster_pos = global_pos.map(|v| v.div_euclid(cluster::SIZE as i64) * (cluster::SIZE as i64));

        match self.clusters_cache.entry(cluster_pos) {
            hash_map::Entry::Vacant(e) => {
                let clusters = self.loaded_clusters.read();
                let cluster = Arc::clone(clusters.get(&cluster_pos)?);

                if cluster.state() != CLUSTER_STATE_LOADED {
                    None
                } else {
                    cluster.changed.store(true, MO_RELAXED);
                    Some(
                        e.insert(AccessGuard {
                            lock: AccessGuardLock::Write(cluster.cluster.write_arc()),
                        })
                        .get_cluster_mut(),
                    )
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
                        let ocluster = clusters.get(&cluster_pos).unwrap();
                        ocluster.changed.store(true, MO_RELAXED);

                        AccessGuardLock::Write(rwlock.write_arc())
                    });
                }

                if let AccessGuardLock::Write(g) = &mut guard.lock {
                    Some(g.as_mut().unwrap())
                } else {
                    unreachable!()
                }
            }
        }
    }

    /// Returns block data or `None` if respective cluster is not loaded
    pub fn get_block(&mut self, pos: &I64Vec3) -> Option<BlockData> {
        let cluster_pos = pos.map(|v| v.div_euclid(cluster::SIZE as i64) * (cluster::SIZE as i64));
        let block_pos = overworld::cluster_block_pos_from_global(pos);

        match self.clusters_cache.entry(cluster_pos) {
            hash_map::Entry::Vacant(e) => {
                let clusters = self.loaded_clusters.read();
                let cluster = Arc::clone(clusters.get(&cluster_pos)?);
                let state = cluster.state();

                if state != CLUSTER_STATE_LOADED && state != CLUSTER_STATE_OFFLOADED_INVISIBLE {
                    return None;
                }

                Some(
                    e.insert(AccessGuard {
                        lock: AccessGuardLock::Read(cluster.cluster.read_arc()),
                    })
                    .get(&block_pos),
                )
            }
            hash_map::Entry::Occupied(e) => Some(e.into_mut().get(&block_pos)),
        }
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    pub fn set_block(&mut self, pos: &I64Vec3, block: Block) -> Option<BlockDataBuilder> {
        let cluster = self.get_cluster_for_block_mut(pos)?;
        let block_pos = overworld::cluster_block_pos_from_global(pos);

        Some(cluster.set(&block_pos, block))
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    fn get_light_level(&mut self, pos: &I64Vec3) -> Option<LightLevel> {
        let cluster = self.get_cluster_for_block_mut(pos)?;
        let block_pos = overworld::cluster_block_pos_from_global(pos);
        Some(cluster.get_light_level(&block_pos))
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    fn set_light_level(&mut self, pos: &I64Vec3, light_level: LightLevel) {
        if let Some(cluster) = self.get_cluster_for_block_mut(pos) {
            let block_pos = overworld::cluster_block_pos_from_global(pos);
            cluster.set_light_level(&block_pos, light_level);
        }
    }

    /// Returns facing, intersection point, and block data of the block that intersects with specified ray.
    pub fn get_block_at_ray(
        &mut self,
        ray_origin: &DVec3,
        ray_dir: &DVec3,
        max_ray_length: f64,
    ) -> Option<(I64Vec3, Facing)> {
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
            let curr_block_upos = glm::convert_unchecked(curr_block_pos);
            let curr_block = self.get_block(&curr_block_upos);

            if let Some(data) = &curr_block {
                let block = data.block();

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

    fn propagate_light_addition(&mut self, queue: &mut VecDeque<I64Vec3>) {
        while let Some(curr_pos) = queue.pop_front() {
            let curr_level = self.get_light_level(&curr_pos).unwrap();
            let curr_color = curr_level.components();

            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let rel_pos = curr_pos + dir;

                let block = self.get_block(&rel_pos).unwrap().block();
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
    pub fn set_light(&mut self, global_pos: &I64Vec3, light_level: LightLevel) {
        if let Some(cluster) = self.get_cluster_for_block_mut(&global_pos) {
            let block_pos = overworld::cluster_block_pos_from_global(&global_pos);
            cluster.set_light_level(&block_pos, light_level);
            cluster.propagate_lighting(&block_pos);
        }
    }

    pub fn remove_light(&mut self, global_pos: &I64Vec3) {
        let curr_level = self.get_light_level(&global_pos).unwrap();
        self.set_light_level(global_pos, LightLevel::zero());

        let mut removal_queue = VecDeque::with_capacity((curr_level.components().max() as usize * 2).pow(3));
        let mut addition_queue = VecDeque::with_capacity(removal_queue.capacity());

        removal_queue.push_back((*global_pos, curr_level));

        while let Some((curr_pos, curr_level)) = removal_queue.pop_front() {
            let curr_color = curr_level.components();

            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let rel_pos = curr_pos + dir;

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
            let cluster = self.get_cluster_for_block_mut(&pos).unwrap();
            let block_pos = overworld::cluster_block_pos_from_global(&pos);

            cluster.propagate_lighting(&block_pos);
        }
    }

    /// Useful for restoring lighting from neighbours when removing block at `block_pos`.
    pub fn check_neighbour_lighting(&mut self, block_pos: &I64Vec3) {
        for i in 0..6 {
            let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
            let rel_pos = block_pos + dir;
            let cluster = self.get_cluster_for_block_mut(block_pos);

            if let Some(cluster) = cluster {
                let block_pos = overworld::cluster_block_pos_from_global(&rel_pos);
                let level = cluster.get_light_level(&block_pos);

                if !level.is_zero() {
                    cluster.propagate_lighting(&block_pos);
                }
            }
        }
    }
}
