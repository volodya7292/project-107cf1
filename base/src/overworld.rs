/// World generation:
/// 1. Terrain
///   Temperature & humidity maps (perlin noise) - implicit biomes
///   Height map (perlin noise) - terrain shape (heights)
///   Ridge noise - rivers (various widths)
/// 2. Ores
/// 3. Structures
/// 4. Features
pub mod accessor;
pub mod actions_storage;
pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster_part_set;
pub mod facing;
pub mod generator;
pub mod light_state;
pub mod liquid_state;
pub mod occluder;
pub mod orchestrator;
pub mod position;
pub mod raw_cluster;
pub mod structure;

use crate::main_registry::MainRegistry;
use crate::overworld::accessor::ReadOnlyOverworldAccessor;
use crate::overworld::cluster_part_set::ClusterPartSet;
use crate::overworld::generator::OverworldGenerator;
pub use crate::overworld::orchestrator::OverworldOrchestrator;
use crate::overworld::position::{ClusterBlockPos, ClusterPos};
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl, CompressedCluster, RawCluster};
use crate::registry::Registry;
use accessor::OverworldAccessor;
use common::parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockUpgradableReadGuard, ArcRwLockWriteGuard};
use common::parking_lot::{Mutex, RawRwLock, RwLock};
use common::types::HashMap;
use common::{glm, MO_RELAXED};
use fixedbitset::FixedBitSet;
use glm::{I64Vec3, U32Vec3};
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::Arc;
use std::time::Instant;

// TODO Main world - 'The Origin'

// const MIN_DIST_BETWEEN_WORLDS: u64 = 400_000_000;
// const MAX_DIST_BETWEEN_WORLDS: u64 = 40_000_000_000;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u32)]
pub enum ClusterStateEnum {
    /// The cluster is not ready to be read or written to.
    Initial,
    /// The cluster is available in read/write mode
    Loaded,
}

impl ClusterStateEnum {
    /// Whether cluster is loaded
    pub fn is_loaded(&self) -> bool {
        *self == Self::Loaded
    }

    pub fn from_u32(v: u32) -> Self {
        assert!(v <= Self::Loaded as u32);
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
}

impl TrackingCluster {
    pub fn new(raw: RawCluster, dirty_parts: ClusterPartSet) -> TrackingCluster {
        let mut active_cells = FixedBitSet::with_capacity(RawCluster::VOLUME);

        for x in 0..RawCluster::SIZE {
            for y in 0..RawCluster::SIZE {
                for z in 0..RawCluster::SIZE {
                    let pos = ClusterBlockPos::new(x, y, z);
                    let pos_idx = pos.index();
                    let data = raw.get(&pos);

                    let active = data.active();
                    active_cells.set(pos_idx, active);
                }
            }
        }

        Self {
            raw,
            dirty_parts,
            active_cells,
        }
    }

    /// Complexity: O(N), where N is RawCluster::VOLUME.
    pub fn has_active_blocks(&self) -> bool {
        self.active_cells.as_slice().iter().any(|v| *v != 0)
    }

    pub fn active_blocks(&self) -> impl Iterator<Item = (ClusterBlockPos, BlockData)> + '_ {
        self.active_cells.ones().map(|idx| {
            let block_pos = ClusterBlockPos::from_index(idx);
            (block_pos, self.raw.get(&block_pos))
        })
    }
}

pub struct CompressedTrackingCluster {
    pub raw: CompressedCluster,
    pub dirty_parts: ClusterPartSet,
    pub has_active_blocks: bool,
}

pub enum ClusterState {
    Initial,
    Ready(TrackingCluster),
    Compressed(CompressedTrackingCluster),
}

impl ClusterState {
    pub fn is_initial(&self) -> bool {
        matches!(self, Self::Initial)
    }

    pub fn is_loaded(&self) -> bool {
        matches!(self, Self::Ready(_) | Self::Compressed(_))
    }

    pub fn is_compressed(&self) -> bool {
        matches!(self, Self::Compressed(_))
    }

    pub fn unwrap(&self) -> &TrackingCluster {
        if let Self::Ready(cluster) = self {
            cluster
        } else {
            panic!("Invalid state");
        }
    }

    pub fn unwrap_mut(&mut self) -> &mut TrackingCluster {
        if let Self::Ready(cluster) = self {
            cluster
        } else {
            panic!("Invalid state");
        }
    }

    pub fn take_dirty_parts(&mut self) -> ClusterPartSet {
        let dirty_parts = match self {
            ClusterState::Ready(cluster) => &mut cluster.dirty_parts,
            ClusterState::Compressed(cluster) => &mut cluster.dirty_parts,
            ClusterState::Initial => {
                panic!("Invalid state!")
            }
        };
        mem::replace(dirty_parts, ClusterPartSet::NONE)
    }

    pub fn has_active_blocks(&self) -> bool {
        match self {
            ClusterState::Ready(cluster) => cluster.has_active_blocks(),
            ClusterState::Compressed(cluster) => cluster.has_active_blocks,
            ClusterState::Initial => {
                panic!("Invalid state!")
            }
        }
    }

    pub fn compress(&mut self) {
        replace_with::replace_with_or_abort(self, |curr| {
            let Self::Ready(cluster) = curr else {
                panic!("Invalid state");
            };

            let has_active_blocks = cluster.has_active_blocks();
            let raw_compressed = cluster.raw.compress();

            Self::Compressed(CompressedTrackingCluster {
                raw: raw_compressed,
                dirty_parts: cluster.dirty_parts,
                has_active_blocks,
            })
        })
    }

    pub fn decompress(&mut self) {
        replace_with::replace_with_or_abort(self, |curr| {
            let Self::Compressed(compressed) = curr else {
                panic!("Invalid state");
            };

            let raw_cluster = RawCluster::from_compressed(compressed.raw);

            Self::Ready(TrackingCluster::new(raw_cluster, compressed.dirty_parts))
        })
    }
}

impl Default for ClusterState {
    fn default() -> Self {
        Self::Initial
    }
}

pub struct OverworldCluster {
    pub cluster: Arc<RwLock<ClusterState>>,
    pub state: AtomicU32,
    pub dirty: AtomicBool,
    /// Managed by [OverworldOrchestrator].
    pub has_active_blocks: AtomicBool,
}

impl OverworldCluster {
    pub fn new() -> Self {
        Self {
            cluster: Default::default(),
            state: AtomicU32::new(ClusterStateEnum::Initial as u32),
            dirty: AtomicBool::new(false),
            has_active_blocks: Default::default(),
        }
    }

    pub fn ready(&self) -> Option<ArcRwLockReadGuard<RawRwLock, ClusterState>> {
        let mut t_cluster_read = self.cluster.upgradable_read_arc();
        if t_cluster_read.is_initial() {
            return None;
        }
        if t_cluster_read.is_compressed() {
            let mut t_cluster = ArcRwLockUpgradableReadGuard::upgrade(t_cluster_read);
            t_cluster.decompress();
            t_cluster_read = ArcRwLockWriteGuard::downgrade_to_upgradable(t_cluster);
        }
        Some(ArcRwLockUpgradableReadGuard::downgrade(t_cluster_read))
    }

    pub fn state(&self) -> ClusterStateEnum {
        ClusterStateEnum::from_u32(self.state.load(MO_RELAXED))
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
