use crate::overworld::block::BlockState;
use crate::overworld::light_state::LightLevel;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::ClusterBlockPos;
use crate::persistence::block_states::{EntityStorageDeserializer, SerializableEntityStorage};
use crate::registry::Registry;
use common::glm;
use common::types::HashMap;
use entity_data::{ArchetypeState, Component, EntityId, EntityStorage};
use glm::TVec3;
use serde::de::SeqAccess;
use serde::ser::SerializeTuple;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt::Formatter;
use std::{mem, slice};

#[inline]
pub fn cell_index(pos: &TVec3<usize>) -> usize {
    const SIZE_SQR: usize = RawCluster::SIZE * RawCluster::SIZE;
    pos.x * SIZE_SQR + pos.y * RawCluster::SIZE + pos.z
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
#[repr(u8)]
pub enum LightType {
    Regular,
    Sky,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CellInfo {
    pub entity_id: CompactEntityId,
    pub block_id: u16,
    pub light_source: LightLevel,
    pub light_source_type: LightType,
    pub light_state: LightLevel,
    pub sky_light_state: LightLevel,
    pub liquid_state: LiquidState,
    pub active: bool,
}

impl Default for CellInfo {
    fn default() -> Self {
        Self {
            entity_id: Default::default(),
            block_id: u16::MAX,
            light_source: Default::default(),
            light_source_type: LightType::Regular,
            light_state: Default::default(),
            sky_light_state: Default::default(),
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
    fn raw_light_source(&self) -> LightLevel;
    fn light_source_type(&self) -> LightType;

    fn regular_light_source(&self) -> LightLevel {
        (self.light_source_type() == LightType::Regular)
            .then_some(self.raw_light_source())
            .unwrap_or(LightLevel::ZERO)
    }

    fn sky_light_source(&self) -> LightLevel {
        (self.light_source_type() == LightType::Sky)
            .then_some(self.raw_light_source())
            .unwrap_or(LightLevel::ZERO)
    }

    fn light_state(&self) -> LightLevel;
    fn sky_light_state(&self) -> LightLevel;

    fn light_state_by(&self, ty: LightType) -> LightLevel {
        match ty {
            LightType::Regular => self.light_state(),
            LightType::Sky => self.sky_light_state(),
        }
    }

    fn liquid_state(&self) -> &LiquidState;
    fn active(&self) -> bool;
}

impl BlockDataImpl for BlockData<'_> {
    fn block_id(&self) -> u16 {
        self.info.block_id
    }

    fn get<C: Component>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.info.entity_id.regular())
    }

    fn raw_light_source(&self) -> LightLevel {
        self.info.light_source
    }

    fn light_source_type(&self) -> LightType {
        self.info.light_source_type
    }

    fn light_state(&self) -> LightLevel {
        self.info.light_state
    }

    fn sky_light_state(&self) -> LightLevel {
        self.info.sky_light_state
    }

    fn liquid_state(&self) -> &LiquidState {
        &self.info.liquid_state
    }

    fn active(&self) -> bool {
        self.info.active
    }
}

pub struct BlockDataMut<'a> {
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

    fn raw_light_source(&self) -> LightLevel {
        self.info.light_source
    }

    fn light_source_type(&self) -> LightType {
        self.info.light_source_type
    }

    fn light_state(&self) -> LightLevel {
        self.info.light_state
    }

    fn sky_light_state(&self) -> LightLevel {
        self.info.sky_light_state
    }

    fn liquid_state(&self) -> &LiquidState {
        &self.info.liquid_state
    }

    fn active(&self) -> bool {
        self.info.active
    }
}

impl BlockDataMut<'_> {
    pub fn set(&mut self, state: BlockState<impl ArchetypeState>) {
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
    }

    pub fn raw_light_source_mut(&mut self) -> &mut LightLevel {
        &mut self.info.light_source
    }

    pub fn light_source_type_mut(&mut self) -> &mut LightType {
        &mut self.info.light_source_type
    }

    pub fn light_state_mut(&mut self) -> &mut LightLevel {
        &mut self.info.light_state
    }

    pub fn sky_light_state_mut(&mut self) -> &mut LightLevel {
        &mut self.info.sky_light_state
    }

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
    cells: Vec<CellInfo>,
    block_state_storage: EntityStorage,
}

impl RawCluster {
    pub const SIZE: usize = 24;
    pub const VOLUME: usize = Self::SIZE * Self::SIZE * Self::SIZE;
    pub const APPROX_MEM_SIZE: usize = Self::VOLUME * mem::size_of::<CellInfo>();

    /// Creating a new cluster is expensive due to its big size of memory
    pub fn new() -> Self {
        Self {
            cells: vec![Default::default(); Self::VOLUME],
            block_state_storage: Default::default(),
        }
    }

    pub fn from_compressed(compressed: CompressedCluster) -> RawCluster {
        let cells = unsafe {
            let len = RawCluster::VOLUME;
            let mut vec = Vec::<CellInfo>::with_capacity(len);
            let u8_slice =
                slice::from_raw_parts_mut(vec.as_mut_ptr() as *mut u8, len * mem::size_of::<CellInfo>());

            lz4_flex::decompress_into(&compressed.cells_data, u8_slice).unwrap();

            vec.set_len(len);
            vec
        };

        RawCluster {
            block_state_storage: compressed.block_state_storage,
            cells,
        }
    }

    pub fn compress(self) -> CompressedCluster {
        let cells_data = unsafe {
            let cells_raw = slice::from_raw_parts(
                self.cells.as_ptr() as *const u8,
                Self::VOLUME * mem::size_of::<CellInfo>(),
            );
            lz4_flex::compress(cells_raw)
        };

        CompressedCluster {
            cells_data,
            block_state_storage: self.block_state_storage,
        }
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
            block_storage: &mut self.block_state_storage,
            info: &mut self.cells[cell_index(pos.get())],
        }
    }
}

pub fn serialize_cluster<S>(
    cluster: &RawCluster,
    registry: &Registry,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let storage = SerializableEntityStorage {
        registry,
        storage: &cluster.block_state_storage,
    };
    let mut tup = serializer.serialize_tuple(2)?;
    tup.serialize_element(&cluster.cells)?;
    tup.serialize_element(&storage)?;
    tup.end()
}

pub fn deserialize_cluster<'de, D>(registry: &Registry, deserializer: D) -> Result<RawCluster, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Visitor;
    struct TupleVisitor<'a> {
        registry: &'a Registry,
    }

    impl<'de2> Visitor<'de2> for TupleVisitor<'_> {
        type Value = RawCluster;

        fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
            write!(formatter, "a tuple of (cells, storage)")
        }

        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: SeqAccess<'de2>,
        {
            let mut cells = seq.next_element::<Vec<CellInfo>>()?.unwrap();
            let mut storage_deser = EntityStorageDeserializer {
                registry: self.registry,
                storage: Default::default(),
                entity_mapping: HashMap::with_capacity(RawCluster::VOLUME),
            };
            seq.next_element_seed(&mut storage_deser)?.unwrap();

            // Renew old entity ids
            for cell in &mut cells {
                let old_entity = cell.entity_id.regular();
                if let Some(new_entity) = storage_deser.entity_mapping.get(&old_entity) {
                    cell.entity_id = CompactEntityId::new(*new_entity);
                }
            }

            Ok(RawCluster {
                cells,
                block_state_storage: storage_deser.storage,
            })
        }
    }

    deserializer.deserialize_tuple(2, TupleVisitor { registry })
}

pub struct CompressedCluster {
    cells_data: Vec<u8>,
    block_state_storage: EntityStorage,
}
