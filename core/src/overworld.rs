use std::sync::atomic::{AtomicBool, AtomicU32};
/// World generation:
/// 1. Terrain
///   Temperature & humidity maps (perlin noise) - implicit biomes
///   Height map (perlin noise) - terrain shape (heights)
///   Ridge noise - rivers (various widths)
/// 2. Ores
/// 3. Structures
/// 4. Features
use std::sync::Arc;

use fixedbitset::FixedBitSet;
use nalgebra_glm::{I64Vec3, U32Vec3};
use parking_lot::RwLock;

use accessor::OverworldAccessor;

use crate::main_registry::MainRegistry;
use crate::overworld::accessor::ReadOnlyOverworldAccessor;
use crate::overworld::cluster_part_set::ClusterPartSet;
use crate::overworld::generator::OverworldGenerator;
pub use crate::overworld::orchestrator::OverworldOrchestrator;
use crate::overworld::position::{ClusterBlockPos, ClusterPos};
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl, RawCluster};
use crate::registry::Registry;
use crate::utils::{HashMap, MO_RELAXED};

pub mod accessor;

pub mod actions_storage;
pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster_part_set;
pub mod facing;
pub mod generator;
pub mod light_level;
pub mod liquid_state;
pub mod occluder;
pub mod orchestrator;
pub mod position;
pub mod raw_cluster;
pub mod structure;

// TODO Main world - 'The Origin'

// const MIN_DIST_BETWEEN_WORLDS: u64 = 400_000_000;
// const MAX_DIST_BETWEEN_WORLDS: u64 = 40_000_000_000;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum ClusterState {
    /// The cluster is not ready to be read or written to.
    Initial,
    /// The cluster is available in read/write mode
    Loaded,
    /// The cluster is invisible because it is empty and is offloaded to reduce memory usage
    OffloadedEmpty,
    /// The cluster is invisible relative to camera and is offloaded to reduce memory usage
    OffloadedOccluded,
}

impl ClusterState {
    /// Whether cluster is loaded
    pub fn is_loaded(&self) -> bool {
        (*self == Self::Loaded) || self.is_offloaded()
    }

    /// Whether cluster is loaded
    pub fn is_writable(&self) -> bool {
        *self == Self::Loaded
    }

    /// Whether cluster can be accessed in read-only mode
    pub fn is_readable(&self) -> bool {
        matches!(*self, Self::Loaded | Self::OffloadedEmpty)
    }

    /// Whether cluster is empty or occluded by its neighbours
    pub fn is_empty_or_occluded(&self) -> bool {
        matches!(*self, Self::OffloadedEmpty | Self::OffloadedOccluded)
    }

    /// Whether cluster is offloaded due to emptiness or full occlusion by neighbours
    pub fn is_offloaded(&self) -> bool {
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

pub fn is_cell_empty(registry: &Registry, data: impl BlockDataImpl) -> bool {
    let block = registry.get_block(data.block_id()).unwrap();
    block.is_model_invisible() && data.liquid_state().level() == 0
}

// Cluster lifecycle
// ----------------------------------
// LOADING -> LOADED | DISCARDED
// LOADED  -> DISCARDED

pub struct TrackingCluster {
    pub raw: RawCluster,
    pub dirty_parts: ClusterPartSet,
    pub active_cells: FixedBitSet,
    pub empty_cells: FixedBitSet,
}

impl TrackingCluster {
    pub fn new(registry: &Registry, raw: RawCluster) -> TrackingCluster {
        let mut active_cells = FixedBitSet::with_capacity(RawCluster::VOLUME);
        let mut empty_cells = FixedBitSet::with_capacity(RawCluster::VOLUME);

        for x in 0..RawCluster::SIZE {
            for y in 0..RawCluster::SIZE {
                for z in 0..RawCluster::SIZE {
                    let pos = ClusterBlockPos::new(x, y, z);
                    let pos_idx = pos.index();
                    let data = raw.get(&pos);

                    let active = data.active();
                    let empty = is_cell_empty(registry, data);

                    active_cells.set(pos_idx, active);
                    empty_cells.set(pos_idx, empty);
                }
            }
        }

        Self {
            raw,
            dirty_parts: ClusterPartSet::ALL,
            active_cells,
            empty_cells,
        }
    }

    /// Complexity: O(N), where N is RawCluster::VOLUME.
    pub fn is_empty(&self) -> bool {
        self.empty_cells.as_slice().iter().all(|v| *v == !0)
    }

    /// Complexity: O(N), where N is RawCluster::VOLUME.
    pub fn has_active_blocks(&self) -> bool {
        self.active_cells.as_slice().iter().any(|v| *v != 0)
    }

    pub fn propagate_lighting(&mut self, pos: &ClusterBlockPos) {
        let dirty_parts = self.raw.propagate_lighting(pos);
        self.dirty_parts = dirty_parts;
    }

    pub fn active_blocks(&self) -> impl Iterator<Item = (ClusterBlockPos, BlockData)> + '_ {
        self.active_cells.ones().map(|idx| {
            let block_pos = ClusterBlockPos::from_index(idx);
            (block_pos, self.raw.get(&block_pos))
        })
    }
}

pub struct OverworldCluster {
    pub cluster: Arc<RwLock<Option<TrackingCluster>>>,
    pub state: Arc<AtomicU32>,
    pub dirty: AtomicBool,
    /// Managed by [OverworldOrchestrator].
    pub has_active_blocks: AtomicBool,
}

impl OverworldCluster {
    pub fn new() -> Self {
        Self {
            cluster: Default::default(),
            state: Arc::new(AtomicU32::new(ClusterState::Initial as u32)),
            dirty: AtomicBool::new(false),
            has_active_blocks: Default::default(),
        }
    }

    pub fn state(&self) -> ClusterState {
        ClusterState::from_u32(self.state.load(MO_RELAXED))
    }
}

pub type LoadedClusters = Arc<RwLock<HashMap<ClusterPos, OverworldCluster>>>;

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

#[derive(Copy, Clone)]
pub struct ReadOnlyOverworld<'a> {
    inner: &'a Overworld,
}

impl<'a> ReadOnlyOverworld<'a> {
    pub fn new(inner: &'a Overworld) -> Self {
        Self { inner }
    }

    pub fn main_registry(&self) -> &Arc<MainRegistry> {
        &self.inner.main_registry
    }

    pub fn access(&self) -> ReadOnlyOverworldAccessor {
        self.inner.access().into_read_only()
    }
}

// Overworld creation
// 1. Determine player position
//   1. Find a closest world to 0-coordinate. Choose a random reasonable position X in the world.
//   2. Find a cluster that is not flooded with liquid around X.
//   3. Choose a random reasonable player position in this cluster.
// 2. Start generating the overworld
