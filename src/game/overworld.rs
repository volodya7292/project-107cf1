pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster;
pub mod generator;
pub mod light_level;
pub mod occluder;
pub mod structure;
pub mod textured_block_model;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block::Block;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::{BlockData, BlockDataBuilder, BlockDataImpl, Cluster};
use crate::game::overworld::light_level::LightLevel;
use crate::game::overworld::structure::world::World;
use crate::game::overworld::structure::Structure;
use crate::game::overworld_streamer;
use crate::game::registry::Registry;
use crate::physics::aabb::AABBRayIntersection;
use engine::utils::value_noise::ValueNoise;
use engine::utils::{HashMap, Int, MO_RELAXED};
use nalgebra::{SimdBool, SimdPartialOrd};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, TVec3, U32Vec3, Vec3};
use parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockUpgradableReadGuard, ArcRwLockWriteGuard};
use parking_lot::{Mutex, MutexGuard, RawRwLock, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rand::Rng;
use std::cell::RefCell;
use std::collections::{hash_map, VecDeque};
use std::hash::Hash;
use std::ops::Add;
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;
use winit::event::VirtualKeyCode::P;

// TODO Main world - 'The Origin'

const MIN_WORLD_RADIUS: u64 = 2_048;
pub const MAX_WORLD_RADIUS: u64 = 32_000_000;
// const MIN_DIST_BETWEEN_WORLDS: u64 = 400_000_000;
// const MAX_DIST_BETWEEN_WORLDS: u64 = 40_000_000_000;

pub const LOD_LEVELS: usize = 24;

fn sample_world_size(rng: &mut impl rand::Rng) -> u64 {
    const AVG_R: u64 = (MIN_WORLD_RADIUS + MAX_WORLD_RADIUS) / 2;
    const R_HALF_DIST: f64 = ((MAX_WORLD_RADIUS - MIN_WORLD_RADIUS) / 2) as f64;

    let s: f64 = rng.sample(rand_distr::StandardNormal);
    AVG_R + (s / 3.0 * R_HALF_DIST).clamp(-R_HALF_DIST, R_HALF_DIST) as u64
}

/// Returns cluster-local block position from global position
#[inline]
fn cluster_block_pos_from_global(global_pos: &I64Vec3) -> U32Vec3 {
    global_pos.map(|v| v.rem_euclid(cluster::SIZE as i64) as u32)
}

pub const CLUSTER_STATE_INITIAL: u8 = 0;
pub const CLUSTER_STATE_LOADING: u8 = 1;
pub const CLUSTER_STATE_LOADED: u8 = 2;
/// The cluster is not needed anymore
pub const CLUSTER_STATE_DISCARDED: u8 = 3;
/// The cluster is invisible relative to the camera and is offloaded to reduce memory usage
pub const CLUSTER_STATE_OFFLOADED_INVISIBLE: u8 = 4;

// Cluster lifecycle
// ----------------------------------
// LOADING -> LOADED | DISCARDED
// LOADED  -> DISCARDED

pub struct OverworldCluster {
    pub cluster: Arc<RwLock<Option<Cluster>>>,
    pub state: AtomicU8,
    pub changed: AtomicBool,
}

impl OverworldCluster {
    pub fn state(&self) -> u8 {
        self.state.load(MO_RELAXED)
    }
}

pub struct Overworld {
    seed: u64,
    main_registry: Arc<MainRegistry>,
    value_noise: ValueNoise<u64>,
    loaded_clusters: Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>>,
}

pub struct BlockDataGuard {}

impl Overworld {
    pub fn new(registry: &Arc<MainRegistry>, seed: u64) -> Overworld {
        Overworld {
            seed,
            main_registry: Arc::clone(registry),
            value_noise: ValueNoise::new(seed),
            loaded_clusters: Default::default(),
        }
    }

    pub fn main_registry(&self) -> &Arc<MainRegistry> {
        &self.main_registry
    }

    pub fn loaded_clusters(&self) -> &Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>> {
        &self.loaded_clusters
    }

    fn get_world(&self, center_pos: I64Vec3) -> World {
        let center_pos = center_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));

        World::new(
            self.value_noise
                .state()
                .next(center_pos.x)
                .next(center_pos.y)
                .next(center_pos.z)
                .0,
        )
    }

    fn gen_spawn_point(&self) -> I64Vec3 {
        let reg = self.main_registry.registry();
        let st = reg.get_structure(self.main_registry.structure_world()).unwrap();

        let p = self.gen_structure_pos(st, I64Vec3::from_element(0)).0;
        let _world = self.get_world(p);

        todo!()
    }

    /// Returns position of the structure center (within gen-octant corresponding to the cluster position)
    /// and a `bool` indicating whether the structure is actually present there.  
    /// Gen-octant size = `structure.avg_spacing * cluster_size(structure.cluster_level)`.
    ///
    /// `cluster_pos` is a cluster position of level `self.cluster_level`.
    pub fn gen_structure_pos(&self, structure: &Structure, cluster_pos: I64Vec3) -> (I64Vec3, bool) {
        let structure_fit_size = cluster::size(structure.cluster_level());
        let octant_pos = cluster_pos.map(|v| v.div_euclid(structure.avg_spacing() as i64));
        let octant_pos_u64 = octant_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));
        let octant_size = structure.avg_spacing() * structure_fit_size;

        let mut rng = self
            .value_noise
            .state()
            .next(structure.cluster_level() as u64)
            .next(octant_pos_u64.x)
            .next(octant_pos_u64.y)
            .next(octant_pos_u64.z)
            .rng();
        let mut present = rng.gen::<bool>();

        let r = (structure_fit_size / 2)..(octant_size - structure_fit_size / 2);
        let dx = rng.gen_range(r.clone());
        let dy = rng.gen_range(r.clone());
        let dz = rng.gen_range(r.clone());
        let center_pos = octant_pos * (octant_size as i64) + I64Vec3::new(dx as i64, dy as i64, dz as i64);

        if present {
            present = structure.check_gen_pos(self, center_pos);
        }

        (center_pos, present)
    }

    /// Find the nearest structure position to the `starting_cluster_pos` position.
    ///
    /// `start_cluster_pos` is a starting cluster position of level `structure.cluster_level`.  
    /// `max_search_radius` is radius in gen-octants of search domain.
    pub fn find_structure_pos(
        &self,
        structure: &Structure,
        start_cluster_pos: I64Vec3,
        max_search_radius: u32,
    ) -> Option<I64Vec3> {
        use std::f32::consts::PI;

        let octant_size = (structure.avg_spacing() * cluster::size(structure.cluster_level())) as i64;
        let phi_units = max_search_radius * 4;
        let theta_units = phi_units * 2;

        for r_i in 0..max_search_radius {
            let r = r_i as f32;

            for phi_i in 0..phi_units {
                let phi = phi_i as f32 / phi_units as f32 * PI;

                for theta_i in 0..theta_units {
                    let theta = theta_i as f32 / theta_units as f32 * 2.0 * PI;

                    let x = r * phi.cos() * theta.sin();
                    let y = r * phi.sin() * theta.cos();
                    let z = r * theta.cos();

                    let v: I64Vec3 = glm::convert_unchecked(Vec3::new(x, y, z));
                    let p = start_cluster_pos + v * octant_size;

                    let result = self.gen_structure_pos(structure, p);
                    if result.1 {
                        return Some(result.0);
                    }
                }
            }
        }

        None
    }

    pub fn load_cluster(&self) {
        todo!()
    }

    pub fn access(&self) -> ClustersAccessCache {
        ClustersAccessCache {
            registry: Arc::clone(self.main_registry.registry()),
            block_empty: self.main_registry.block_empty(),
            loaded_clusters: Arc::clone(&self.loaded_clusters),
            clusters_cache: HashMap::with_capacity(32),
        }
    }
}

pub struct AccessGuard {
    block_empty: Block,
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
    registry: Arc<Registry>,
    block_empty: Block,
    loaded_clusters: Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>>,
    clusters_cache: HashMap<I64Vec3, AccessGuard>,
}

impl ClustersAccessCache {
    /// Returns cluster for the specified global block position
    pub fn get_cluster_for_pos_mut(&mut self, global_pos: &I64Vec3) -> Option<&mut Cluster> {
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
                            block_empty: self.block_empty,
                            lock: AccessGuardLock::Write(cluster.cluster.write_arc()),
                        })
                        .get_cluster_mut(),
                    )
                }
            }
            hash_map::Entry::Occupied(mut e) => {
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
        let block_pos = cluster_block_pos_from_global(pos);

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
                        block_empty: self.block_empty,
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
        let cluster = self.get_cluster_for_pos_mut(pos)?;
        let block_pos = cluster_block_pos_from_global(pos);

        Some(cluster.set(&block_pos, block))
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    fn get_light_level(&mut self, pos: &I64Vec3) -> Option<LightLevel> {
        let cluster = self.get_cluster_for_pos_mut(pos)?;
        let block_pos = cluster_block_pos_from_global(pos);
        Some(cluster.get_light_level(&block_pos))
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    fn set_light_level(&mut self, pos: &I64Vec3, light_level: LightLevel) {
        if let Some(cluster) = self.get_cluster_for_pos_mut(pos) {
            let block_pos = cluster_block_pos_from_global(pos);
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

        let mut curr_origin = *ray_origin;
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
                let dir: I64Vec3 = glm::convert(Facing::from_u8(i).direction());
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
        if let Some(cluster) = self.get_cluster_for_pos_mut(&global_pos) {
            let block_pos = cluster_block_pos_from_global(&global_pos);
            cluster.set_light_level(&block_pos, light_level);
            cluster.propagate_lighting(&block_pos);
        }

        // self.set_light_level(global_pos, light_level);
        //
        // let mut queue = VecDeque::with_capacity((light_level.components().max() as usize * 2).pow(2));
        // queue.push_back(*global_pos);
        //
        // self.propagate_light_addition(&mut queue);
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
                let dir: I64Vec3 = glm::convert(Facing::from_u8(i).direction());
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
            let cluster = self.get_cluster_for_pos_mut(&pos).unwrap();
            let block_pos = cluster_block_pos_from_global(&pos);

            cluster.propagate_lighting(&block_pos);
        }

        // self.propagate_light_addition(&mut addition_queue);
    }
}

// Overworld creation
// 1. Determine player position
//   1. Find a closest world to 0-coordinate. Choose a random reasonable position X in the world.
//   2. Find non-flooded-with-liquid cluster around X.
//   3. Choose a random reasonable player position in this cluster.
// 2. Start generating the overworld
