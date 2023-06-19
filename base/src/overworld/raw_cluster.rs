use crate::overworld::block::BlockState;
use crate::overworld::light_state::LightLevel;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::occluder::Occluder;
use crate::overworld::position::ClusterBlockPos;
use crate::registry::Registry;
use common::glm;
use entity_data::{ArchetypeState, Component, EntityId, EntityStorage};
use glm::TVec3;
use std::io::{Read, Write};
use std::sync::Arc;
use std::{mem, slice};

#[inline]
pub fn cell_index(pos: &TVec3<usize>) -> usize {
    const SIZE_SQR: usize = RawCluster::SIZE * RawCluster::SIZE;
    pos.x * SIZE_SQR + pos.y * RawCluster::SIZE + pos.z
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct CompactEntityId {
    arch_id: u16,
    /// Cluster size of 24^3 fits into 2^16 space.
    id: u16,
}

impl CompactEntityId {
    const NULL: Self = Self {
        arch_id: u16::MAX,
        id: u16::MAX,
    };

    fn new(entity_id: EntityId) -> Self {
        Self {
            arch_id: entity_id.archetype_id as u16,
            id: entity_id.id as u16,
        }
    }

    fn regular(&self) -> EntityId {
        EntityId::new(self.arch_id as u32, self.id as u32)
    }
}

impl Default for CompactEntityId {
    fn default() -> Self {
        Self::NULL
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CellInfo {
    pub entity_id: CompactEntityId,
    pub block_id: u16,
    pub occluder: Occluder,
    pub light_source: LightLevel,
    pub light_state: LightLevel,
    pub liquid_state: LiquidState,
    pub active: bool,
}

impl Default for CellInfo {
    fn default() -> Self {
        Self {
            entity_id: Default::default(),
            block_id: u16::MAX,
            occluder: Default::default(),
            light_source: Default::default(),
            light_state: Default::default(),
            liquid_state: LiquidState::NONE,
            active: false,
        }
    }
}

#[derive(Copy, Clone)]
pub struct BlockData<'a> {
    pub(super) block_storage: &'a EntityStorage,
    pub(super) info: &'a CellInfo,
}

pub trait BlockDataImpl {
    fn block_id(&self) -> u16;
    /// Returns the specified block component `C`
    fn get<C: Component>(&self) -> Option<&C>;
    fn light_source(&self) -> LightLevel;
    fn light_state(&self) -> LightLevel;
    fn liquid_state(&self) -> &LiquidState;
    fn occluder(&self) -> Occluder;
    fn active(&self) -> bool;
}

impl BlockDataImpl for BlockData<'_> {
    fn block_id(&self) -> u16 {
        self.info.block_id
    }

    fn get<C: Component>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.info.entity_id.regular())
    }

    fn light_source(&self) -> LightLevel {
        self.info.light_source
    }

    fn light_state(&self) -> LightLevel {
        self.info.light_state
    }

    fn liquid_state(&self) -> &LiquidState {
        &self.info.liquid_state
    }

    fn occluder(&self) -> Occluder {
        self.info.occluder
    }

    fn active(&self) -> bool {
        self.info.active
    }
}

pub struct BlockDataMut<'a> {
    registry: &'a Registry,
    block_storage: &'a mut EntityStorage,
    info: &'a mut CellInfo,
}

impl BlockDataImpl for BlockDataMut<'_> {
    fn block_id(&self) -> u16 {
        self.info.block_id
    }

    fn get<C: Component>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.info.entity_id.regular())
    }

    fn light_source(&self) -> LightLevel {
        self.info.light_source
    }

    fn light_state(&self) -> LightLevel {
        self.info.light_state
    }

    fn liquid_state(&self) -> &LiquidState {
        &self.info.liquid_state
    }

    fn occluder(&self) -> Occluder {
        self.info.occluder
    }

    fn active(&self) -> bool {
        self.info.active
    }
}

impl BlockDataMut<'_> {
    pub fn set(&mut self, state: BlockState<impl ArchetypeState>) {
        let block = self.registry.get_block(state.block_id).unwrap();

        // Remove previous block state if present
        if self.info.entity_id != CompactEntityId::NULL {
            self.block_storage.remove(&self.info.entity_id.regular());
        }

        // Add new state to the storage
        let entity_id = if state.components.num_components() > 0 {
            self.block_storage.add(state.components)
        } else {
            EntityId::NULL
        };

        self.info.entity_id = CompactEntityId::new(entity_id);
        self.info.block_id = state.block_id;
        self.info.occluder = block.occluder();
        self.info.active |= block.active_by_default();
    }

    /// Raw mutable access to light source.
    pub fn light_source_mut(&mut self) -> &mut LightLevel {
        &mut self.info.light_source
    }

    /// Raw mutable access to light state.
    pub fn light_state_mut(&mut self) -> &mut LightLevel {
        &mut self.info.light_state
    }

    /// Raw mutable access to liquid state.
    pub fn liquid_state_mut(&mut self) -> &mut LiquidState {
        &mut self.info.liquid_state
    }

    pub fn active_mut(&mut self) -> &mut bool {
        &mut self.info.active
    }

    pub fn get_mut<C: Component>(&mut self) -> Option<&mut C> {
        self.block_storage.get_mut::<C>(&self.info.entity_id.regular())
    }
}

pub struct RawCluster {
    registry: Arc<Registry>,
    block_state_storage: EntityStorage,
    cells: Vec<CellInfo>,
}

impl RawCluster {
    pub const SIZE: usize = 24;
    pub const VOLUME: usize = Self::SIZE * Self::SIZE * Self::SIZE;
    pub const APPROX_MEM_SIZE: usize = Self::VOLUME * mem::size_of::<CellInfo>();

    /// Creating a new cluster is expensive due to its big size of memory
    pub fn new(registry: &Arc<Registry>) -> Self {
        Self {
            registry: Arc::clone(registry),
            block_state_storage: Default::default(),
            cells: vec![Default::default(); Self::VOLUME],
        }
    }

    pub fn from_compressed(compressed: CompressedCluster) -> RawCluster {
        let mut decompressor = lz4_flex::frame::FrameDecoder::new(compressed.cells_data.as_slice());
        let cells = unsafe {
            let len = RawCluster::VOLUME;
            let mut vec = Vec::<CellInfo>::with_capacity(len);

            let u8_slice =
                slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, len * mem::size_of::<CellInfo>());
            decompressor.read_exact(u8_slice).unwrap();

            vec.set_len(len);
            vec
        };

        RawCluster {
            registry: compressed.registry,
            block_state_storage: compressed.block_state_storage,
            cells,
        }
    }

    pub fn compress(self) -> CompressedCluster {
        let mut cells_data = Vec::new();
        let mut compressor = lz4_flex::frame::FrameEncoder::new(&mut cells_data);
        unsafe {
            let cells_raw = slice::from_raw_parts(
                self.cells.as_ptr() as *const u8,
                Self::VOLUME * mem::size_of::<CellInfo>(),
            );
            compressor.write_all(cells_raw).unwrap();
        }
        compressor.finish().unwrap();

        CompressedCluster {
            registry: self.registry,
            cells_data,
            block_state_storage: self.block_state_storage,
        }
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    pub fn cells(&self) -> &[CellInfo] {
        &self.cells
    }

    /// Returns block data at `pos`.
    #[inline]
    pub fn get(&self, pos: &ClusterBlockPos) -> BlockData {
        BlockData {
            block_storage: &self.block_state_storage,
            info: &self.cells[cell_index(pos.get())],
        }
    }

    /// Returns mutable block data at `pos`.
    #[inline]
    pub fn get_mut(&mut self, pos: &ClusterBlockPos) -> BlockDataMut {
        BlockDataMut {
            registry: &self.registry,
            block_storage: &mut self.block_state_storage,
            info: &mut self.cells[cell_index(pos.get())],
        }
    }
}

pub struct CompressedCluster {
    registry: Arc<Registry>,
    cells_data: Vec<u8>,
    block_state_storage: EntityStorage,
}
