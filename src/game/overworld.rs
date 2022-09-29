/// World generation:
/// 1. Terrain
///   Temperature & humidity maps (perlin noise) - implicit biomes
///   Height map (perlin noise) - terrain shape (heights)
///   Ridge noise - rivers (various widths)
/// 2. Ores
/// 3. Structures
/// 4. Features
pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster;
pub mod clusters_access_cache;
pub mod facing;
pub mod generator;
pub mod light_level;
pub mod occluder;
pub mod structure;
pub mod textured_block_model;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::facing::Facing;
use crate::game::overworld::generator::OverworldGenerator;
use crate::game::overworld::structure::{Structure, StructuresIter};
use bit_vec::BitVec;
use clusters_access_cache::ClustersAccessCache;
use engine::utils::white_noise::WhiteNoise;
use engine::utils::{HashMap, UInt, MO_RELAXED};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, TVec3, U32Vec3, U64Vec3, Vec3};
use parking_lot::RwLock;
use rand::Rng;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;

// TODO Main world - 'The Origin'

// const MIN_DIST_BETWEEN_WORLDS: u64 = 400_000_000;
// const MAX_DIST_BETWEEN_WORLDS: u64 = 40_000_000_000;

pub const CLUSTER_STATE_INITIAL: u8 = 0;
pub const CLUSTER_STATE_LOADING: u8 = 1;
pub const CLUSTER_STATE_LOADED: u8 = 2;
/// The cluster is not needed anymore
pub const CLUSTER_STATE_DISCARDED: u8 = 3;
/// The cluster is invisible relative to the camera and is offloaded to reduce memory usage
pub const CLUSTER_STATE_OFFLOADED_INVISIBLE: u8 = 4;

/// Returns cluster-local block position from global position
#[inline]
pub fn cluster_block_pos_from_global(global_pos: &I64Vec3) -> U32Vec3 {
    global_pos.map(|v| v.rem_euclid(cluster::SIZE as i64) as u32)
}

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
    loaded_clusters: Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>>,
    generator: Arc<OverworldGenerator>,
}

impl Overworld {
    pub fn new(registry: &Arc<MainRegistry>, seed: u64) -> Overworld {
        Overworld {
            seed,
            main_registry: Arc::clone(registry),
            loaded_clusters: Default::default(),
            generator: Arc::new(OverworldGenerator::new(seed, registry)),
        }
    }

    pub fn main_registry(&self) -> &Arc<MainRegistry> {
        &self.main_registry
    }

    pub fn loaded_clusters(&self) -> &Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>> {
        &self.loaded_clusters
    }

    pub fn generator(&self) -> &Arc<OverworldGenerator> {
        &self.generator
    }

    pub fn load_cluster(&self) {
        todo!()
    }

    pub fn access(&self) -> ClustersAccessCache {
        ClustersAccessCache {
            registry: Arc::clone(self.main_registry.registry()),
            loaded_clusters: Arc::clone(&self.loaded_clusters),
            clusters_cache: HashMap::with_capacity(32),
        }
    }
}

// Overworld creation
// 1. Determine player position
//   1. Find a closest world to 0-coordinate. Choose a random reasonable position X in the world.
//   2. Find a cluster that is not flooded with liquid around X.
//   3. Choose a random reasonable player position in this cluster.
// 2. Start generating the overworld
