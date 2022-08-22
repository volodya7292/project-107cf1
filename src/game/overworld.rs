/// World generation:
/// 1. Terrain
///   Temperature & moisture maps (perlin noise) - implicit biomes
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
pub mod generator;
pub mod light_level;
pub mod occluder;
pub mod structure;
pub mod textured_block_model;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::structure::world::World;
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
    white_noise: WhiteNoise,
    loaded_clusters: Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>>,
}

impl Overworld {
    pub fn new(registry: &Arc<MainRegistry>, seed: u64) -> Overworld {
        Overworld {
            seed,
            main_registry: Arc::clone(registry),
            white_noise: WhiteNoise::new(seed),
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
            self.white_noise
                .state()
                .next(center_pos.x)
                .next(center_pos.y)
                .next(center_pos.z)
                .0,
            Arc::clone(self.main_registry.registry()),
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
    /// Gen-octant size = `structure.avg_spacing * cluster_size` blocks.
    /// Cluster position is in blocks.
    pub fn gen_structure_pos(&self, structure: &Structure, cluster_pos: I64Vec3) -> (I64Vec3, bool) {
        let structure_fit_size = UInt::next_multiple_of(structure.max_size().max(), cluster::SIZE as u64);
        let octant_size = structure.avg_spacing() * (cluster::SIZE as u64);
        let octant_pos = cluster_pos.map(|v| v.div_euclid(octant_size as i64));
        let octant_pos_u64 = octant_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));

        let mut rng = self
            .white_noise
            .state()
            .next(structure.uid() as u64)
            .next(octant_pos_u64.x)
            .next(octant_pos_u64.y)
            .next(octant_pos_u64.z)
            .rng();
        let mut present = rng.gen::<bool>();

        let low_margin = (structure_fit_size / 2).max(structure.min_spacing() / 2);
        let range = low_margin..(octant_size - low_margin);

        let dx = rng.gen_range(range.clone()) as i64;
        let dy = rng.gen_range(range.clone()) as i64;
        let dz = rng.gen_range(range.clone()) as i64;
        let center_pos = octant_pos * (octant_size as i64) + I64Vec3::new(dx, dy, dz);

        if present {
            present = structure.check_gen_pos(self, center_pos);
        }

        (center_pos, present)
    }

    /// Find the nearest structure positions to the `start_cluster_pos` position.  
    /// Returned positions are in blocks.
    ///
    /// `structure` is the structure to search for.
    /// `start_cluster_pos` is the starting cluster position.  
    /// `max_search_radius` is the radius in gen-octants of search domain.
    /// Gen-octant is of size `structure.avg_spacing` clusters.
    pub fn find_structure_pos<'a>(
        &'a self,
        structure: &'a Structure,
        start_cluster_pos: I64Vec3,
        max_search_radius: u32,
    ) -> StructuresIter<'a> {
        let diam = (max_search_radius * 2) as i64;
        let volume = diam.pow(3) as usize;
        let clusters_per_octant = structure.avg_spacing() as i64;
        let start_pos = start_cluster_pos.map(|v| v.div_euclid(clusters_per_octant));

        // Use breadth-first search
        let mut queue = VecDeque::with_capacity(volume);
        let mut traversed_nodes = BitVec::from_elem(volume, false);

        queue.push_back(start_pos);

        StructuresIter {
            overworld: self,
            structure,
            start_cluster_pos,
            max_search_radius,
            queue,
            traversed_nodes,
        }
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
