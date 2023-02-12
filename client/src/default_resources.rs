use crate::rendering::texture_material::TextureMaterial;
use crate::rendering::textured_block_model::{QuadMaterial, TexturedBlockModel};
use crate::resource_mapping::ResourceMapping;
use base::main_registry::MainRegistry;
use common::glm::Vec4;
use common::resource_file::ResourceFile;
use engine::module::main_renderer::material::MatComponent;
use engine::module::main_renderer::TextureAtlasType;
use std::sync::Arc;

const MAX_BLOCKS: usize = 16384;

macro_rules! add_getters {
    ($t: ty, $($name: ident)*) => ($(
        pub fn $name(&self) -> $t {
            self.$name
        }
    )*);
}

pub struct DefaultResourceMapping {
    mapping: Arc<ResourceMapping>,
    material_water: u16,
}

impl DefaultResourceMapping {
    pub fn init(main_reg: &MainRegistry, resources: &Arc<ResourceFile>) -> Arc<Self> {
        let reg = main_reg.registry();
        let null_tex_res = resources.get("textures/test_texture.basis").unwrap();

        let mut map = ResourceMapping::new(
            reg.get_block_model(main_reg.model_cube()).unwrap(),
            null_tex_res,
            MAX_BLOCKS,
        );

        // Textures
        let tex_default = map.register_texture(
            TextureAtlasType::ALBEDO,
            resources.get("textures/test_texture.basis").unwrap(),
        );

        let tex_glow = map.register_texture(
            TextureAtlasType::ALBEDO,
            resources.get("textures/glow_texture.basis").unwrap(),
        );
        let tex_water = map.register_texture(
            TextureAtlasType::ALBEDO,
            resources.get("textures/water.basis").unwrap(),
        );

        // Materials
        let material_default =
            map.register_material(TextureMaterial::new(MatComponent::Texture(tex_default)));
        let material_glow = map.register_material(TextureMaterial::new(MatComponent::Texture(tex_glow)));
        let material_water = map.register_material(
            TextureMaterial::new(
                // MatComponent::Texture(tex_water),
                MatComponent::Color(Vec4::new(0.0, 0.0, 1.0, 0.5)),
            )
            .with_translucent(true),
        );

        map.set_block_textured_model(
            main_reg.block_test.block_id,
            TexturedBlockModel::new(
                reg.get_block_model(main_reg.model_cube()).unwrap(),
                &[QuadMaterial::new(material_default); 6],
                &map,
            ),
        );
        map.set_block_textured_model(
            main_reg.block_glow.block_id,
            TexturedBlockModel::new(
                reg.get_block_model(main_reg.model_cube()).unwrap(),
                &[QuadMaterial::new(material_glow); 6],
                &map,
            ),
        );

        Arc::new(Self {
            mapping: Arc::new(map),
            material_water,
        })
    }

    pub fn storage(&self) -> &Arc<ResourceMapping> {
        &self.mapping
    }

    add_getters! { u16, material_water }
}
