use crate::game::overworld;
use crate::game::overworld::block::Block;
use crate::game::overworld::block_model;
use crate::game::overworld::block_model::BlockModel;
use crate::game::overworld::structure::Structure;
use crate::game::overworld::textured_block_model::{QuadMaterial, TexturedBlockModel};
use crate::game::registry::Registry;
use entity_data as ed;
use nalgebra_glm::{U64Vec3, Vec3};
use std::sync::Arc;

pub struct MainRegistry {
    registry: Arc<Registry>,
    structure_world: u32,
    block_empty: Block,
    block_default: Block,
}

macro_rules! add_getters {
    ($t: ty, $($name: ident)*) => ($(
        pub fn $name(&self) -> $t {
            self.$name
        }
    )*);
}

impl MainRegistry {
    pub fn init() -> Arc<MainRegistry> {
        let mut reg = Registry::new();

        // Textures
        let tex_default = 0;

        // Block models
        // ----------------------------------------------------------------------------------------------------
        let cube_model = reg.register_block_model(BlockModel::new(&block_model::cube_quads(
            Vec3::from_element(0.0),
            Vec3::from_element(1.0),
        )));

        let cluster_layout = reg.cluster_layout_mut();

        // Blocks
        // ----------------------------------------------------------------------------------------------------
        let block_empty = {
            let arch = cluster_layout.add_archetype().build();
            Block::new(arch as u16, u16::MAX)
        };
        let block_default = {
            let arch = cluster_layout.add_archetype().build();
            let tex_model = reg.register_textured_block_model(TexturedBlockModel::new(
                reg.get_block_model(cube_model).unwrap(),
                &[QuadMaterial::new(tex_default, Default::default()); 6],
            ));
            Block::new(arch as u16, tex_model)
        };

        // Structures
        // ----------------------------------------------------------------------------------------------------
        let structure_world = reg.register_structure(Structure::new(
            U64Vec3::from_element(overworld::MAX_WORLD_RADIUS as u64 * 2),
            12,
            625,
            |_, _, _| true,
        ));

        Arc::new(MainRegistry {
            registry: Arc::new(reg),
            structure_world,
            block_empty,
            block_default,
        })
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    add_getters! { u32, structure_world }
    add_getters! { Block, block_empty block_default }
}
