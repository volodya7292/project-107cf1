use crate::game::overworld;
use crate::game::overworld::block::Block;
use crate::game::overworld::block_model::BlockModel;
use crate::game::overworld::structure::Structure;
use crate::game::overworld::textured_block_model::TexturedBlockModel;
use entity_data::EntityStorageLayout;

pub struct Registry {
    structures: Vec<Structure>,
    structured_by_level: [Vec<Structure>; overworld::LOD_LEVELS],
    models: Vec<BlockModel>,
    textured_models: Vec<TexturedBlockModel>,
    blocks: Vec<Block>,
    cluster_layout: EntityStorageLayout,
}

impl Registry {
    pub fn new() -> Self {
        Registry {
            structures: vec![],
            structured_by_level: Default::default(),
            models: vec![],
            textured_models: vec![],
            blocks: vec![],
            cluster_layout: Default::default(),
        }
    }

    pub fn cluster_layout(&self) -> &EntityStorageLayout {
        &self.cluster_layout
    }

    pub fn cluster_layout_mut(&mut self) -> &mut EntityStorageLayout {
        &mut self.cluster_layout
    }

    pub fn register_block_model(&mut self, block_model: BlockModel) -> u32 {
        self.models.push(block_model);
        (self.models.len() - 1) as u32
    }

    pub fn register_textured_block_model(&mut self, textured_block_model: TexturedBlockModel) -> u32 {
        self.textured_models.push(textured_block_model);
        (self.textured_models.len() - 1) as u32
    }

    pub fn register_block(&mut self, block: Block) -> u32 {
        self.blocks.push(block);
        (self.blocks.len() - 1) as u32
    }

    pub fn register_structure(&mut self, structure: Structure) -> u32 {
        self.structures.push(structure);
        (self.structures.len() - 1) as u32
    }

    pub fn get_block_model(&self, id: u32) -> Option<&BlockModel> {
        self.models.get(id as usize)
    }

    pub fn get_textured_block_model(&self, id: u32) -> Option<&TexturedBlockModel> {
        self.textured_models.get(id as usize)
    }

    pub fn get_structure(&self, id: u32) -> Option<&Structure> {
        self.structures.get(id as usize)
    }

    pub fn get_structures_by_lod(&self, cluster_level: u32) -> &[Structure] {
        &self.structured_by_level[cluster_level as usize]
    }
}
