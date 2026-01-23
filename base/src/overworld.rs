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
pub mod interface;
pub mod item;
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
use crate::overworld::interface::OverworldInterface;
pub use crate::overworld::orchestrator::OverworldOrchestrator;
use crate::overworld::position::{ClusterBlockPos, ClusterPos};
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl, CompressedCluster, RawCluster};
use crate::registry::Registry;
use accessor::OverworldAccessor;
use common::glm::{DVec3, Vec3};
use common::parking_lot::lock_api::{ArcRwLockReadGuard, ArcRwLockUpgradableReadGuard, ArcRwLockWriteGuard};
use common::parking_lot::{Mutex, RawRwLock, RwLock};
use common::types::HashMap;
use common::{MO_RELAXED, glm};
use fixedbitset::FixedBitSet;
use glm::{I64Vec3, U32Vec3};
use serde::{Deserialize, Serialize};
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32};
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
    pub last_used_time: Mutex<Instant>,
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
            last_used_time: Mutex::new(Instant::now()),
        }
    }

    pub fn last_used_time(&self) -> Instant {
        *self.last_used_time.lock()
    }

    pub fn update_used_time(&self) {
        *self.last_used_time.lock() = Instant::now();
    }

    /// Complexity: O(N), where N is RawCluster::VOLUME.
    pub fn has_active_blocks(&self) -> bool {
        self.active_cells.as_slice().iter().any(|v| *v != 0)
    }

    pub fn active_blocks(&self) -> impl Iterator<Item = (ClusterBlockPos, BlockData<'_>)> + '_ {
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

    /// Decompressed the cluster if it's compressed and returns it.
    pub fn ready(cluster: &Arc<RwLock<ClusterState>>) -> Option<ArcRwLockReadGuard<RawRwLock, ClusterState>> {
        let mut t_cluster_read = cluster.upgradable_read_arc();
        if t_cluster_read.is_initial() {
            return None;
        }
        if t_cluster_read.is_compressed() {
            let mut t_cluster = ArcRwLockUpgradableReadGuard::upgrade(t_cluster_read);
            t_cluster.decompress();
            t_cluster_read = ArcRwLockWriteGuard::downgrade_to_upgradable(t_cluster);
        }
        t_cluster_read.unwrap().update_used_time();
        Some(ArcRwLockUpgradableReadGuard::downgrade(t_cluster_read))
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
    pub fn state(&self) -> ClusterStateEnum {
        ClusterStateEnum::from_u32(self.state.load(MO_RELAXED))
    }
}

impl Default for OverworldCluster {
    fn default() -> Self {
        Self::new()
    }
}

pub type LoadedClusters = Arc<RwLock<HashMap<ClusterPos, OverworldCluster>>>;

pub struct Overworld {
    main_registry: Arc<MainRegistry>,
    loaded_clusters: LoadedClusters,
    interface: Arc<dyn OverworldInterface>,
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct PlayerState {
    position: Option<[f64; 3]>,
    orientation: [f64; 3],
    velocity: [f32; 3],
    health: f64,
    satiety: f64,
}

impl PlayerState {
    pub fn new() -> Self {
        Self {
            position: None,
            orientation: Default::default(),
            velocity: Default::default(),
            health: 1.0,
            satiety: 1.0,
        }
    }

    pub fn position(&self) -> Option<DVec3> {
        self.position.map(|v| v.into())
    }

    pub fn set_position(&mut self, position: DVec3) {
        self.position = Some(position.into())
    }

    pub fn orientation(&self) -> DVec3 {
        self.orientation.into()
    }

    pub fn set_orientation(&mut self, orientation: DVec3) {
        self.orientation = orientation.into();
    }

    pub fn velocity(&self) -> Vec3 {
        self.velocity.into()
    }

    pub fn set_velocity(&mut self, velocity: Vec3) {
        self.velocity = velocity.into();
    }

    pub fn health(&self) -> f64 {
        self.health
    }

    pub fn set_health(&mut self, health: f64) {
        self.health = health.clamp(0.0, 1.0);
    }

    pub fn satiety(&self) -> f64 {
        self.satiety
    }

    pub fn set_satiety(&mut self, satiety: f64) {
        self.satiety = satiety.clamp(0.0, 1.0);
    }

    pub fn is_dead(&self) -> bool {
        self.health == 0.0
    }
}

impl Default for PlayerState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct OverworldState {
    pub seed: u64,
    pub tick_count: u64,
    pub player_state: PlayerState,
}

impl OverworldState {
    pub fn player_state(&self) -> PlayerState {
        self.player_state
    }

    pub fn update_player_state<F: FnOnce(&mut PlayerState)>(&mut self, f: F) {
        f(&mut self.player_state);
    }
}

impl Overworld {
    pub fn new(registry: &Arc<MainRegistry>, interface: Arc<dyn OverworldInterface>) -> Arc<Overworld> {
        Arc::new(Overworld {
            main_registry: Arc::clone(registry),
            loaded_clusters: Default::default(),
            interface,
        })
    }

    pub fn main_registry(&self) -> &Arc<MainRegistry> {
        &self.main_registry
    }

    pub fn loaded_clusters(&self) -> &LoadedClusters {
        &self.loaded_clusters
    }

    pub fn interface(&self) -> &Arc<dyn OverworldInterface> {
        &self.interface
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
