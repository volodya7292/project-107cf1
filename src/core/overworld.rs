use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8};

use bit_vec::BitVec;
use fixedbitset::FixedBitSet;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, TVec3, U32Vec3, U64Vec3, Vec3};
use parking_lot::RwLock;
use rand::Rng;

use accessor::OverworldAccessor;
use engine::utils::{HashMap, MO_RELAXED, UInt};
use engine::utils::white_noise::WhiteNoise;

use crate::core::main_registry::MainRegistry;
use crate::core::overworld::cluster_dirty_parts::ClusterDirtySides;
use crate::core::overworld::facing::Facing;
use crate::core::overworld::generator::OverworldGenerator;
use crate::core::overworld::position::{ClusterBlockPos, ClusterPos};
use crate::core::overworld::raw_cluster::{BlockData, BlockDataImpl, RawCluster};
use crate::core::overworld::structure::{Structure, StructuresIter};

pub mod accessor;
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
pub mod cluster_dirty_parts;
pub mod facing;
pub mod generator;
pub mod light_level;
pub mod material;
pub mod occluder;
pub mod position;
pub mod raw_cluster;
pub mod structure;
pub mod textured_block_model;

// TODO Main world - 'The Origin'

// const MIN_DIST_BETWEEN_WORLDS: u64 = 400_000_000;
// const MAX_DIST_BETWEEN_WORLDS: u64 = 40_000_000_000;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum ClusterState {
    Initial,
    Loading,
    /// The cluster is available in read/write mode
    Loaded,
    /// The cluster is not needed anymore
    Discarded,
    /// The cluster is invisible because it is empty and is offloaded to reduce memory usage
    OffloadedEmpty,
    /// The cluster is invisible relative to camera and is offloaded to reduce memory usage
    OffloadedOccluded,
}

impl ClusterState {
    /// Whether cluster can be accessed in read-only mode
    pub fn is_readable(&self) -> bool {
        matches!(*self, Self::Loaded | Self::OffloadedEmpty)
    }

    /// Whether cluster can be accessed in read-only mode
    pub fn is_empty_or_occluded(&self) -> bool {
        matches!(*self, Self::OffloadedEmpty | Self::OffloadedOccluded)
    }

    pub fn from_u32(v: u32) -> Self {
        assert!(v <= Self::OffloadedOccluded as u32);
        unsafe { std::mem::transmute(v) }
    }
}

/// Returns cluster-local block position from global position
#[inline]
pub fn cluster_block_pos_from_global(global_pos: &I64Vec3) -> U32Vec3 {
    global_pos.map(|v| v.rem_euclid(RawCluster::SIZE as i64) as u32)
}

// Cluster lifecycle
// ----------------------------------
// LOADING -> LOADED | DISCARDED
// LOADED  -> DISCARDED

pub struct Cluster {
    pub raw: RawCluster,
    pub dirty_parts: ClusterDirtySides,
    pub active_blocks: FixedBitSet,
    pub may_have_active_blocks: bool,
}

impl Cluster {
    pub fn new(raw: RawCluster) -> Cluster {
        let mut active_blocks = FixedBitSet::with_capacity(RawCluster::VOLUME);

        for x in 0..RawCluster::SIZE {
            for y in 0..RawCluster::SIZE {
                for z in 0..RawCluster::SIZE {
                    let pos = ClusterBlockPos::new(x, y, z);
                    let active = raw
                        .get(&pos)
                        .get::<block_component::Activity>()
                        .map_or(false, |v| v.active);

                    if active {
                        active_blocks.toggle(pos.index());
                    }
                }
            }
        }

        Self {
            raw,
            dirty_parts: ClusterDirtySides::all(),
            active_blocks,
            may_have_active_blocks: true,
        }
    }

    pub fn propagate_lighting(&mut self, pos: &ClusterBlockPos) {
        let dirty_parts = self.raw.propagate_lighting(pos);
        self.dirty_parts = dirty_parts;
    }

    pub fn active_blocks(&self) -> impl Iterator<Item = (ClusterBlockPos, BlockData)> + '_ {
        self.active_blocks.ones().map(|idx| {
            let block_pos = ClusterBlockPos::from_index(idx);
            (block_pos, self.raw.get(&block_pos))
        })
    }
}

pub struct OverworldCluster {
    pub(super) cluster: Arc<RwLock<Option<Cluster>>>,
    pub(super) state: AtomicU32,
    pub(super) dirty: AtomicBool,

    pub(super) active_blocks: FixedBitSet,
    pub(super) may_have_active_blocks: AtomicBool,
}

impl OverworldCluster {
    pub fn new() -> Self {
        Self {
            cluster: Default::default(),
            state: AtomicU32::new(ClusterState::Initial as u32),
            dirty: Default::default(),
            active_blocks: FixedBitSet::with_capacity(RawCluster::VOLUME),
            may_have_active_blocks: Default::default(),
        }
    }

    pub fn state(&self) -> ClusterState {
        ClusterState::from_u32(self.state.load(MO_RELAXED))
    }
}

pub type LoadedClusters = Arc<RwLock<HashMap<ClusterPos, Arc<OverworldCluster>>>>;

pub struct Overworld {
    seed: u64,
    main_registry: Arc<MainRegistry>,
    loaded_clusters: LoadedClusters,
    generator: Arc<OverworldGenerator>,
}

impl Overworld {
    pub fn new(registry: &Arc<MainRegistry>, seed: u64) -> Arc<Overworld> {
        Arc::new(Overworld {
            seed,
            main_registry: Arc::clone(registry),
            loaded_clusters: Default::default(),
            generator: Arc::new(OverworldGenerator::new(seed, registry)),
        })
    }

    pub fn main_registry(&self) -> &Arc<MainRegistry> {
        &self.main_registry
    }

    pub fn loaded_clusters(&self) -> &LoadedClusters {
        &self.loaded_clusters
    }

    pub fn generator(&self) -> &Arc<OverworldGenerator> {
        &self.generator
    }

    pub fn load_cluster(&self) {
        todo!()
    }

    pub fn access(&self) -> OverworldAccessor {
        OverworldAccessor::new(
            Arc::clone(self.main_registry.registry()),
            Arc::clone(&self.loaded_clusters),
        )
    }
}

// Overworld creation
// 1. Determine player position
//   1. Find a closest world to 0-coordinate. Choose a random reasonable position X in the world.
//   2. Find a cluster that is not flooded with liquid around X.
//   3. Choose a random reasonable player position in this cluster.
// 2. Start generating the overworld
