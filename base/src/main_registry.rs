use std::sync::Arc;

use common::glm;
use entity_data::Archetype;
use glm::{DVec3, U64Vec3, Vec3};

use crate::overworld::block::{BlockBuilder, BlockState};
use crate::overworld::block_model::BlockModel;
use crate::overworld::structure::world::biome::{MeanHumidity, MeanTemperature};
use crate::overworld::structure::world::Biome;
use crate::overworld::structure::{world, Structure};
use crate::overworld::{block, block_model};
use crate::physics::aabb::AABB;
use crate::registry::Registry;

#[derive(Copy, Clone, Archetype)]
pub struct StatelessBlock;

pub struct MainRegistry {
    registry: Arc<Registry>,
    structure_world: u32,
    model_cube: u16,
    pub block_empty: BlockState<StatelessBlock>,
    pub block_test: BlockState<StatelessBlock>,
    pub block_glow: BlockState<StatelessBlock>,
    // pub block_water: BlockState<StatelessBlock>,
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

        // Block models
        // ----------------------------------------------------------------------------------------------------
        let model_cube = reg.register_block_model(BlockModel::new(
            &block_model::cube_quads(Vec3::from_element(0.0), Vec3::from_element(1.0)),
            &[AABB::new(DVec3::from_element(0.0), DVec3::from_element(1.0))],
        ));

        // Blocks
        // ----------------------------------------------------------------------------------------------------
        let block_empty = {
            let id =
                reg.register_block(BlockBuilder::new(Registry::MODEL_ID_NULL).with_can_pass_liquid(true));
            BlockState::new(id, StatelessBlock)
        };
        let block_default = {
            let id = reg.register_block(BlockBuilder::new(model_cube));
            BlockState::new(id, StatelessBlock)
        };
        let block_glow = {
            let id = reg.register_block(
                BlockBuilder::new(model_cube)
                    .with_active_by_default(true)
                    .with_event_handlers(block::EventHandlers::new().with_on_tick(
                        |_, pos, _, _, _, mut after_actions| {
                            after_actions.set_activity(*pos, false);
                            println!("ON TICK!");
                        },
                    )),
            );
            BlockState::new(id, StatelessBlock)
        };

        // Biomes
        // ----------------------------------------------------------------------------------------------------
        let biome_tundra = reg.register_biome(Biome::new(
            MeanTemperature::TNeg30..=MeanTemperature::TPos7,
            MeanHumidity::H0..=MeanHumidity::H25,
            0.0..=1.0,
        ));
        let biome_taiga = reg.register_biome(Biome::new(
            MeanTemperature::TNeg15..=MeanTemperature::TPos15,
            MeanHumidity::H0..=MeanHumidity::H37,
            0.0..=1.0,
        ));
        let biome_temperate_deciduous_forest = reg.register_biome(Biome::new(
            MeanTemperature::TNeg7..=MeanTemperature::TPos22,
            MeanHumidity::H12..=MeanHumidity::H50,
            0.0..=1.0,
        ));
        let biome_temperate_rain_forest = reg.register_biome(Biome::new(
            MeanTemperature::TNeg7..=MeanTemperature::TPos22,
            MeanHumidity::H50..=MeanHumidity::H75,
            0.0..=1.0,
        ));
        let biome_tropical_rain_forest = reg.register_biome(Biome::new(
            MeanTemperature::TPos15..=MeanTemperature::TPos30,
            MeanHumidity::H62..=MeanHumidity::H100,
            0.0..=1.0,
        ));
        let biome_savanna = reg.register_biome(Biome::new(
            MeanTemperature::TPos15..=MeanTemperature::TPos30,
            MeanHumidity::H12..=MeanHumidity::H62,
            0.0..=1.0,
        ));
        let biome_chaparral = reg.register_biome(Biome::new(
            MeanTemperature::TPos7..=MeanTemperature::TPos22,
            MeanHumidity::H12..=MeanHumidity::H37,
            0.0..=1.0,
        ));
        let biome_grassland = reg.register_biome(Biome::new(
            MeanTemperature::TNeg7..=MeanTemperature::TPos22,
            MeanHumidity::H0..=MeanHumidity::H37,
            0.0..=1.0,
        ));
        let biome_desert = reg.register_biome(Biome::new(
            MeanTemperature::T0..=MeanTemperature::TPos22,
            MeanHumidity::H0..=MeanHumidity::H25,
            0.0..=1.0,
        ));
        let biome_subtropical_desert = reg.register_biome(Biome::new(
            MeanTemperature::TPos15..=MeanTemperature::TPos30,
            MeanHumidity::H0..=MeanHumidity::H25,
            0.0..=1.0,
        ));
        let biome_ocean = reg.register_biome(Biome::new(
            MeanTemperature::TNeg30..=MeanTemperature::TPos30,
            MeanHumidity::H0..=MeanHumidity::H100,
            -1.0..=0.0,
        ));
        let biome_coast = reg.register_biome(Biome::new(
            MeanTemperature::TNeg30..=MeanTemperature::TPos30,
            MeanHumidity::H0..=MeanHumidity::H100,
            0.0..=0.02,
        ));

        // Structures
        // ----------------------------------------------------------------------------------------------------
        let structure_world = reg.register_structure(Structure::new(
            0,
            U64Vec3::from_element(world::MAX_RADIUS as u64 * 2),
            9000,
            16000,
            |_, _, _| true,
            world::gen_fn,
            Some(world::spawn_point_fn),
        ));

        Arc::new(MainRegistry {
            registry: Arc::new(reg),
            structure_world,
            model_cube,
            block_empty,
            block_test: block_default,
            block_glow,
            // block_water,
            // water_states,
        })
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    add_getters! { u32, structure_world }

    add_getters! { u16, model_cube }
}
