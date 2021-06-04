use crate::game::world::block_component::Facing;
use crate::game::world::block_model;
use crate::game::world::block_model::BlockModel;
use crate::game::world::textured_block_model::{QuadMaterial, TexturedBlockModel};
use entity_data::EntityStorageLayout;
use glm::Vec3;
use nalgebra_glm as glm;

pub struct GameRegistry {
    cluster_layout: EntityStorageLayout,
    models: Vec<BlockModel>,
    textured_models: Vec<TexturedBlockModel>,
}

impl GameRegistry {
    pub fn new() -> Self {
        GameRegistry {
            cluster_layout: Default::default(),
            models: vec![],
            textured_models: vec![],
        }
    }

    pub fn cluster_layout(&self) -> &EntityStorageLayout {
        &self.cluster_layout
    }

    pub fn register_block_model(&mut self, block_model: BlockModel) -> u32 {
        self.models.push(block_model);
        (self.models.len() - 1) as u32
    }

    pub fn register_textured_block_model(&mut self, textured_block_model: TexturedBlockModel) -> u32 {
        self.textured_models.push(textured_block_model);
        (self.textured_models.len() - 1) as u32
    }

    pub fn get_block_model(&self, id: u32) -> Option<&BlockModel> {
        self.models.get(id as usize)
    }

    pub fn get_textured_block_model(&self, id: u32) -> Option<&TexturedBlockModel> {
        self.textured_models.get(id as usize)
    }

    pub fn predefined() -> Self {
        let mut reg = Self::new();

        reg.cluster_layout.add_archetype().build();

        // Block
        let block_model = reg.register_block_model(BlockModel::new(&block_model::cube_quads(
            Vec3::from_element(0.0),
            Vec3::from_element(1.0),
        )));

        reg.register_textured_block_model(TexturedBlockModel::new(
            reg.get_block_model(block_model).unwrap(),
            &[QuadMaterial::new(0, Default::default()); 6],
        ));

        reg
    }
}
