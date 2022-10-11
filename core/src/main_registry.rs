use std::default::Default;
use std::sync::Arc;

use entity_data::Archetype;
use entity_data::ArchetypeState;
use nalgebra_glm::{DVec3, U64Vec3, Vec3, Vec4};

use engine::renderer::{MatComponent, MaterialInfo, TextureAtlasType, TEXTURE_ID_NONE};
use engine::resource_file::ResourceFile;

use crate::overworld;
use crate::overworld::block::event_handlers::AfterTickActionsStorage;
use crate::overworld::block::{Block, BlockState};
use crate::overworld::block_model::BlockModel;
use crate::overworld::material::Material;
use crate::overworld::structure::world::biome::{MeanHumidity, MeanTemperature};
use crate::overworld::structure::world::Biome;
use crate::overworld::structure::{world, Structure};
use crate::overworld::textured_block_model::{QuadMaterial, TexturedBlockModel};
use crate::overworld::{block, block_component, block_model, structure};
use crate::physics::aabb::AABB;
use crate::registry::Registry;

#[derive(Copy, Clone, Archetype)]
pub struct StatelessBlock;

#[derive(Copy, Clone, Archetype)]
pub struct GlowBlockState {
    activity: block_component::Activity,
}

pub struct MainRegistry {
    registry: Arc<Registry>,
    structure_world: u32,
    pub block_empty: BlockState<StatelessBlock>,
    pub block_default: BlockState<StatelessBlock>,
    pub block_glow: BlockState<GlowBlockState>,
    // pub block_water: BlockState<StatelessBlock>,
    pub water_states: Vec<BlockState<StatelessBlock>>,
}

macro_rules! add_getters {
    ($t: ty, $($name: ident)*) => ($(
        pub fn $name(&self) -> $t {
            self.$name
        }
    )*);
}

impl MainRegistry {
    pub fn init(resources: &Arc<ResourceFile>) -> Arc<MainRegistry> {
        let mut reg = Registry::new();

        // Textures
        let tex_default = reg.register_texture(
            TextureAtlasType::ALBEDO,
            resources.get("textures/test_texture.basis").unwrap(),
        );
        let tex_glow = reg.register_texture(
            TextureAtlasType::ALBEDO,
            resources.get("textures/glow_texture.basis").unwrap(),
        );
        let tex_water = reg.register_texture(
            TextureAtlasType::ALBEDO,
            resources.get("textures/water.basis").unwrap(),
        );

        // Materials
        let material_default = reg.register_material(Material::new(MatComponent::Texture(tex_default)));
        let material_glow = reg.register_material(Material::new(MatComponent::Texture(tex_glow)));
        let material_water = reg.register_material(
            Material::new(
                // MatComponent::Texture(tex_water),
                MatComponent::Color(Vec4::new(0.0, 0.0, 1.0, 0.5)),
            )
            .with_translucent(true),
        );

        // Block models
        // ----------------------------------------------------------------------------------------------------
        let cube_model = reg.register_block_model(BlockModel::new(
            &block_model::cube_quads(Vec3::from_element(0.0), Vec3::from_element(1.0)),
            &[AABB::new(DVec3::from_element(0.0), DVec3::from_element(1.0))],
        ));

        // Blocks
        // ----------------------------------------------------------------------------------------------------
        let block_empty = {
            let id = reg.register_block(Block::new_simple(&reg, u16::MAX));
            BlockState::new(id, StatelessBlock)
        };
        let block_default = {
            let tex_model = reg.register_textured_block_model(TexturedBlockModel::new(
                reg.get_block_model(cube_model).unwrap(),
                &[QuadMaterial::new(material_default); 6],
                &reg,
            ));
            let id = reg.register_block(Block::new_simple(&reg, tex_model));
            BlockState::new(id, StatelessBlock)
        };
        let block_glow = {
            let tex_model = reg.register_textured_block_model(TexturedBlockModel::new(
                reg.get_block_model(cube_model).unwrap(),
                &[QuadMaterial::new(material_glow); 6],
                &reg,
            ));
            let id = reg.register_block(Block::new(
                &reg,
                tex_model,
                block::EventHandlers::new().with_on_tick(|pos, _, _, mut after_actions| {
                    after_actions.set_component(*pos, block_component::Activity { active: false });
                    println!("ON TICK!");
                }),
            ));
            BlockState::new(
                id,
                GlowBlockState {
                    activity: block_component::Activity { active: true },
                },
            )
        };
        let water_states = {
            let models = block::water::gen_blocks(&reg, material_water, 7);

            let states: Vec<_> = models
                .into_iter()
                .map(|model| {
                    let tex_model = reg.register_textured_block_model(model);
                    let id = reg.register_block(Block::new_simple(&reg, tex_model));

                    BlockState::new(id, StatelessBlock)
                })
                .collect();

            states

            // let tex_model = reg.register_textured_block_model(TexturedBlockModel::new(
            //     reg.get_block_model(cube_model).unwrap(),
            //     &[QuadMaterial::new(material_water).with_transparency(true); 6],
            // ));
            // let id = reg.register_block(Block::new_simple(&reg, tex_model));
            // BlockState::new(id, StatelessBlock)
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
            block_empty,
            block_default,
            block_glow,
            // block_water,
            water_states,
        })
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    add_getters! { u32, structure_world }
}
