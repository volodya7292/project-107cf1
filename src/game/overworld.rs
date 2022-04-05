pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster;
pub mod generator;
pub mod structure;
pub mod textured_block_model;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block::Block;
use crate::game::overworld::cluster::{BlockData, BlockDataBuilder, Cluster};
use crate::game::overworld::structure::world::World;
use crate::game::overworld::structure::Structure;
use crate::game::overworld_streamer;
use engine::utils::value_noise::ValueNoise;
use engine::utils::{HashMap, Int, MO_RELAXED};
use nalgebra_glm as glm;
use nalgebra_glm::{I64Vec3, U32Vec3, Vec3};
use parking_lot::{Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};
use rand::Rng;
use std::cell::RefCell;
use std::collections::hash_map;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;

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

/// The cluster data is not yet loaded
pub const CLUSTER_STATE_INITIAL: u8 = 0;
/// The cluster is not needed anymore
pub const CLUSTER_STATE_DISCARDED: u8 = 1;
pub const CLUSTER_STATE_LOADING: u8 = 2;
pub const CLUSTER_STATE_LOADED: u8 = 3;

pub struct OverworldCluster {
    pub cluster: RwLock<Cluster>,
}

pub struct ClusterState {
    pub state: AtomicU8,
}

impl ClusterState {
    pub fn state(&self) -> u8 {
        self.state.load(MO_RELAXED)
    }
}

pub struct Clusters {
    pub loaded_clusters: HashMap<I64Vec3, Arc<OverworldCluster>>,
    pub clusters_states: HashMap<I64Vec3, Arc<ClusterState>>,
}

pub struct Overworld {
    seed: u64,
    main_registry: Arc<MainRegistry>,
    value_noise: ValueNoise<u64>,
    clusters: Arc<RwLock<Clusters>>,
}

pub struct BlockDataGuard {}

impl Overworld {
    pub fn new(registry: &Arc<MainRegistry>, seed: u64) -> Overworld {
        Overworld {
            seed,
            main_registry: Arc::clone(registry),
            value_noise: ValueNoise::new(seed),
            clusters: Arc::new(RwLock::new(Clusters {
                loaded_clusters: Default::default(),
                clusters_states: Default::default(),
            })),
        }
    }

    pub fn main_registry(&self) -> &Arc<MainRegistry> {
        &self.main_registry
    }

    pub fn clusters(&self) -> &Arc<RwLock<Clusters>> {
        &self.clusters
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

                    let v: I64Vec3 = glm::try_convert(Vec3::new(x, y, z)).unwrap();
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

    pub fn clusters_read(&self) -> ClustersRead {
        ClustersRead(self.clusters.read())
    }
}

pub struct ClustersRead<'a>(RwLockReadGuard<'a, Clusters>);

impl ClustersRead<'_> {
    pub fn access(&self) -> ClustersAccessCache {
        ClustersAccessCache {
            loaded_clusters: &self.0.loaded_clusters,
            cluster_states: &self.0.clusters_states,
            clusters_cache: HashMap::with_capacity(32),
        }
    }
}

pub enum AccessGuard<'a> {
    Read(RwLockReadGuard<'a, Cluster>),
    Write(RwLockWriteGuard<'a, Cluster>),
}

impl AccessGuard<'_> {
    #[inline]
    pub fn get(&self, pos: U32Vec3) -> BlockData {
        match self {
            AccessGuard::Read(g) => g.get(pos),
            AccessGuard::Write(g) => g.get(pos),
        }
    }

    #[inline]
    pub fn set(&mut self, pos: U32Vec3, block: Block) -> BlockDataBuilder {
        match self {
            AccessGuard::Write(g) => g.set(pos, block),
            _ => unreachable!(),
        }
    }
}

pub struct ClustersAccessCache<'a> {
    loaded_clusters: &'a HashMap<I64Vec3, Arc<OverworldCluster>>,
    cluster_states: &'a HashMap<I64Vec3, Arc<ClusterState>>,
    clusters_cache: HashMap<I64Vec3, AccessGuard<'a>>,
}

impl<'a> ClustersAccessCache<'a> {
    /// Returns block data or `None` if respective cluster is not loaded
    pub fn get_block(&mut self, pos: I64Vec3) -> Option<BlockData> {
        let cluster_pos = pos.map(|v| v.div_euclid(cluster::SIZE as i64) * (cluster::SIZE as i64));
        let block_pos = pos.map(|v| v.rem_euclid(cluster::SIZE as i64) as u32);

        match self.clusters_cache.entry(cluster_pos) {
            hash_map::Entry::Vacant(e) => {
                let cluster = self.loaded_clusters.get(&cluster_pos)?;

                if self.cluster_states[&cluster_pos].state() != CLUSTER_STATE_LOADED {
                    return None;
                }

                Some(e.insert(AccessGuard::Read(cluster.cluster.read())).get(block_pos))
            }
            hash_map::Entry::Occupied(e) => Some(e.into_mut().get(block_pos)),
        }
    }

    /// Returns block builder or `None` if respective cluster is not loaded
    pub fn set_block(&mut self, pos: I64Vec3, block: Block) -> Option<BlockDataBuilder> {
        let cluster_pos = pos.map(|v| v.div_euclid(cluster::SIZE as i64) * (cluster::SIZE as i64));
        let block_pos = pos.map(|v| v.rem_euclid(cluster::SIZE as i64) as u32);

        match self.clusters_cache.entry(cluster_pos) {
            hash_map::Entry::Vacant(e) => {
                let cluster = self.loaded_clusters.get(&cluster_pos)?;

                if self.cluster_states[&cluster_pos].state() != CLUSTER_STATE_LOADED {
                    return None;
                }

                Some(
                    e.insert(AccessGuard::Write(cluster.cluster.try_write()?))
                        .set(block_pos, block),
                )
            }
            hash_map::Entry::Occupied(e) => Some(e.into_mut().set(block_pos, block)),
        }
    }
}

// Overworld creation
// 1. Determine player position
//   1. Find a closest world to 0-coordinate. Choose a random reasonable position X in the world.
//   2. Find non-flooded-with-liquid cluster around X.
//   3. Choose a random reasonable player position in this cluster.
// 2. Start generating the overworld
