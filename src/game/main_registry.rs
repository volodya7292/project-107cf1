use crate::game::overworld;
use crate::game::overworld::block::{Block, BlockState};
use crate::game::overworld::block_model::BlockModel;
use crate::game::overworld::structure::world::biome::{MeanHumidity, MeanTemperature};
use crate::game::overworld::structure::world::Biome;
use crate::game::overworld::structure::{world, Structure};
use crate::game::overworld::textured_block_model::{QuadMaterial, TexturedBlockModel};
use crate::game::overworld::{block_model, structure};
use crate::game::registry::Registry;
use crate::physics::aabb::AABB;
use engine::renderer::{MatComponent, MaterialInfo, TextureAtlasType, TEXTURE_ID_NONE};
use engine::resource_file::ResourceFile;
use entity_data::Archetype;
use entity_data::ArchetypeState;
use nalgebra_glm::{DVec3, U64Vec3, Vec3};
use std::default::Default;
use std::sync::Arc;

#[derive(Copy, Clone, Archetype)]
pub struct StatelessBlock;

pub struct MainRegistry {
    registry: Arc<Registry>,
    structure_world: u32,
    pub block_empty: BlockState<StatelessBlock>,
    pub block_default: BlockState<StatelessBlock>,
    pub block_glow: BlockState<StatelessBlock>,
    pub block_water: BlockState<StatelessBlock>,
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
        let material_default = reg.register_material(MaterialInfo::new(
            MatComponent::Texture(tex_default),
            MatComponent::Color(Default::default()),
            TEXTURE_ID_NONE,
            Default::default(),
        ));
        let material_glow = reg.register_material(MaterialInfo::new(
            MatComponent::Texture(tex_glow),
            MatComponent::Color(Default::default()),
            TEXTURE_ID_NONE,
            Default::default(),
        ));
        let material_water = reg.register_material(MaterialInfo::new(
            MatComponent::Texture(tex_water),
            MatComponent::Color(Default::default()),
            TEXTURE_ID_NONE,
            Default::default(),
        ));

        // Block models
        // ----------------------------------------------------------------------------------------------------
        let cube_model = reg.register_block_model(BlockModel::new(
            &block_model::cube_quads(Vec3::from_element(0.0), Vec3::from_element(1.0)),
            &[AABB::new(DVec3::from_element(0.0), DVec3::from_element(1.0))],
        ));

        // Blocks
        // ----------------------------------------------------------------------------------------------------
        let block_empty = {
            let id = reg.register_block(Block::new(u16::MAX));
            BlockState::new(id, StatelessBlock)
        };
        let block_default = {
            let tex_model = reg.register_textured_block_model(TexturedBlockModel::new(
                reg.get_block_model(cube_model).unwrap(),
                &[QuadMaterial::new(material_default); 6],
            ));
            let id = reg.register_block(Block::new(tex_model));
            BlockState::new(id, StatelessBlock)
        };
        let block_glow = {
            let tex_model = reg.register_textured_block_model(TexturedBlockModel::new(
                reg.get_block_model(cube_model).unwrap(),
                &[QuadMaterial::new(material_glow); 6],
            ));
            let id = reg.register_block(Block::new(tex_model));
            BlockState::new(id, StatelessBlock)
        };
        let block_water = {
            let tex_model = reg.register_textured_block_model(TexturedBlockModel::new(
                reg.get_block_model(cube_model).unwrap(),
                &[QuadMaterial::new(material_water).with_transparency(true); 6],
            ));
            let id = reg.register_block(Block::new(tex_model));
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
            block_empty,
            block_default,
            block_glow,
            block_water,
        })
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    add_getters! { u32, structure_world }
}
