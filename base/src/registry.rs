use crate::overworld::block::{Block, BlockBuilder, BlockStateArchetype};
use crate::overworld::block_model::BlockModel;
use crate::overworld::interface::block_states::{StateDeserializeFn, StateSerializeInfo};
use crate::overworld::item::Item;
use crate::overworld::raw_cluster::{CellInfo, LightType};
use crate::overworld::structure::world::Biome;
use crate::overworld::structure::Structure;
use common::types::HashMap;
use entity_data::Archetype;
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use std::convert::TryInto;

#[derive(Copy, Clone, Archetype, Serialize, Deserialize)]
pub struct StatelessBlock;

impl BlockStateArchetype for StatelessBlock {
    fn canon_name() -> &'static str {
        "StatelessBlock"
    }
}

pub struct LiquidType {}

pub struct Registry {
    structures: Vec<Structure>,
    biomes: Vec<Biome>,
    // structures_by_level: [Vec<Structure>; overworld::LOD_LEVELS],
    models: Vec<BlockModel>,
    liquids: Vec<LiquidType>,
    blocks: Vec<Block>,
    items: Vec<Item>,
    block_state_serializers: HashMap<TypeId, StateSerializeInfo>,
    block_state_deserializers: HashMap<&'static str, StateDeserializeFn>,
    block_empty: u16,
    inner_block_state_empty: Option<CellInfo>,
}

impl Registry {
    pub const MODEL_ID_NULL: u16 = u16::MAX;
    pub const MAX_BLOCKS: u16 = u16::MAX;
    pub const MAX_LIQUIDS: u16 = u16::MAX;
    pub const MAX_MODELS: u16 = u16::MAX;
    pub const MAX_ITEMS: u16 = u16::MAX;

    pub fn new() -> Self {
        let mut registry = Registry {
            structures: vec![],
            // structures_by_level: Default::default(),
            biomes: vec![],
            models: vec![],
            liquids: vec![],
            blocks: vec![],
            items: vec![],
            block_state_serializers: Default::default(),
            block_state_deserializers: Default::default(),
            block_empty: u16::MAX,
            inner_block_state_empty: None,
        };

        let block_empty = registry.register_block::<StatelessBlock>(
            BlockBuilder::new(Self::MODEL_ID_NULL)
                .with_can_pass_light(true)
                .with_can_pass_liquid(true),
        );
        registry.block_empty = block_empty;
        registry.inner_block_state_empty = Some(CellInfo {
            entity_id: Default::default(),
            block_id: block_empty,
            light_source: Default::default(),
            light_source_type: LightType::Regular,
            regular_light_state: Default::default(),
            sky_light_state: Default::default(),
            liquid_state: Default::default(),
            active: false,
        });

        registry
    }

    pub fn register_block_model(&mut self, block_model: BlockModel) -> u16 {
        if self.models.len() >= Self::MAX_MODELS as usize {
            panic!("Maximum number of materials is reached!");
        }
        self.models.push(block_model);
        (self.models.len() - 1).try_into().unwrap()
    }

    pub fn block_empty(&self) -> u16 {
        self.block_empty
    }

    pub fn register_block<S: BlockStateArchetype>(&mut self, builder: BlockBuilder) -> u16 {
        if self.blocks.len() >= Self::MAX_BLOCKS as usize {
            panic!("Maximum number of blocks is reached!");
        }

        let block = builder.build(self);

        self.block_state_serializers.insert(
            TypeId::of::<S>(),
            StateSerializeInfo {
                canon_name: S::canon_name(),
                func: S::serialize_from,
            },
        );
        self.block_state_deserializers
            .insert(S::canon_name(), S::deserialize_into);

        self.blocks.push(block);
        (self.blocks.len() - 1) as u16
    }

    pub fn register_liquid(&mut self) -> u16 {
        if self.liquids.len() >= Self::MAX_LIQUIDS as usize {
            panic!("Maximum number of liquids is reached!");
        }
        self.liquids.push(LiquidType {});
        self.liquids.len() as u16 - 1
    }

    pub fn register_biome(&mut self, biome: Biome) -> u32 {
        self.biomes.push(biome);
        self.biomes.len() as u32 - 1
    }

    pub fn register_structure(&mut self, structure: Structure) -> u32 {
        self.structures.push(structure);
        self.structures.len() as u32 - 1
    }

    pub fn register_item(&mut self, item: Item) -> u32 {
        if self.items.len() >= Self::MAX_ITEMS as usize {
            panic!("Maximum number of items is reached!");
        }
        self.items.push(item);
        self.items.len() as u32 - 1
    }

    pub fn biomes(&self) -> &[Biome] {
        &self.biomes
    }

    pub fn get_biome(&self, id: u32) -> Option<&Biome> {
        self.biomes.get(id as usize)
    }

    pub fn get_block(&self, id: u16) -> Option<&Block> {
        self.blocks.get(id as usize)
    }

    pub fn get_block_model(&self, id: u16) -> Option<&BlockModel> {
        self.models.get(id as usize)
    }

    pub fn get_item(&self, id: u32) -> Option<&Item> {
        self.items.get(id as usize)
    }

    pub fn get_structure(&self, id: u32) -> Option<&Structure> {
        self.structures.get(id as usize)
    }

    pub fn get_block_state_serializer(&self, state_id: &TypeId) -> Option<&StateSerializeInfo> {
        self.block_state_serializers.get(state_id)
    }

    pub fn get_block_state_deserializer(&self, type_canon_name: &str) -> Option<&StateDeserializeFn> {
        self.block_state_deserializers.get(type_canon_name)
    }

    // pub fn get_structures_by_lod(&self, cluster_level: u32) -> &[Structure] {
    //     &self.structures_by_level[cluster_level as usize]
    // }

    pub fn inner_block_state_empty(&self) -> &CellInfo {
        self.inner_block_state_empty.as_ref().unwrap()
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::new()
    }
}
