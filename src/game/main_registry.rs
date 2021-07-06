use crate::game::overworld;
use crate::game::overworld::block_model;
use crate::game::overworld::block_model::BlockModel;
use crate::game::overworld::structure::Structure;
use crate::game::overworld::textured_block_model::{QuadMaterial, TexturedBlockModel};
use crate::game::registry::Registry;
use nalgebra_glm::{U64Vec3, Vec3};
use std::sync::Arc;

pub struct MainRegistry {
    registry: Arc<Registry>,
    structure_world: u32,
}

impl MainRegistry {
    pub fn init() -> MainRegistry {
        let mut reg = Registry::new();
        reg.cluster_layout_mut().add_archetype().build();

        // Block
        let block_model = reg.register_block_model(BlockModel::new(&block_model::cube_quads(
            Vec3::from_element(0.0),
            Vec3::from_element(1.0),
        )));

        reg.register_textured_block_model(TexturedBlockModel::new(
            reg.get_block_model(block_model).unwrap(),
            &[QuadMaterial::new(0, Default::default()); 6],
        ));

        let structure_world = reg.register_structure(Structure::new(
            U64Vec3::from_element(overworld::MAX_WORLD_RADIUS as u64 * 2),
            12,
            625,
            |_, _, _| true,
        ));

        MainRegistry {
            registry: Arc::new(reg),
            structure_world,
        }
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    pub fn structure_world(&self) -> u32 {
        self.structure_world
    }
}
