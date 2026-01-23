use crate::rendering::item_visuals::ItemVisuals;
use crate::rendering::texture_material::TextureMaterial;
use crate::rendering::textured_block_model::{QuadMaterial, TexturedBlockModel};
use crate::rendering::ui::backgrounds;
use crate::rendering::ui::image::ImageSource;
use crate::resource_mapping::ResourceMapping;
use base::main_registry::MainRegistry;
use base::registry::Registry;
use common::resource_file::BufferedResourceReader;
use engine::module::main_renderer::TextureAtlasType;
use engine::module::main_renderer::material::MatComponent;
use engine::module::ui::color::Color;
use std::sync::Arc;

macro_rules! add_getters {
    ($t: ty, $($name: ident)*) => ($(
        pub fn $name(&self) -> $t {
            self.$name
        }
    )*);
}

pub struct DefaultResourceMapping {
    mapping: Arc<ResourceMapping>,
    material_lawn: u16,
    material_water: u16,
}

impl DefaultResourceMapping {
    pub fn init(main_reg: &MainRegistry, resources: &Arc<BufferedResourceReader>) -> Arc<Self> {
        let reg = main_reg.registry();
        let null_tex_res = resources.file().get("textures/test_texture.basis").unwrap();

        let mut map = ResourceMapping::new(
            reg.get_block_model(main_reg.model_cube()).unwrap(),
            null_tex_res,
            Registry::MAX_BLOCKS,
            Registry::MAX_LIQUIDS,
            Registry::MAX_ITEMS,
        );

        // Textures
        let tex_lawn = map.register_texture(
            TextureAtlasType::ALBEDO,
            resources.file().get("textures/lawn.basis").unwrap(),
        );

        let tex_glow = map.register_texture(
            TextureAtlasType::ALBEDO,
            resources.file().get("textures/glow_texture.basis").unwrap(),
        );
        let _tex_water = map.register_texture(
            TextureAtlasType::ALBEDO,
            resources.file().get("textures/water.basis").unwrap(),
        );

        // Materials
        let material_lawn = map.register_material(TextureMaterial::new(MatComponent::Texture(tex_lawn)));
        let material_glow = map.register_material(TextureMaterial::new(MatComponent::Texture(tex_glow)));
        let material_water = map.register_material(
            TextureMaterial::new(
                // MatComponent::Texture(tex_water),
                MatComponent::Color(Color::from_hex(0x2036c5ff).with_alpha(0.4)),
            )
            .with_translucent(true),
        );

        map.set_block_textured_model(
            main_reg.block_test.block_id,
            TexturedBlockModel::new(
                reg.get_block_model(main_reg.model_cube()).unwrap(),
                &[QuadMaterial::new(material_lawn); 6],
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

        map.set_liquid_material(main_reg.liquid_water, material_water);

        {
            let image_data = resources.get_image("/textures/lawn.png").unwrap();
            map.set_item_visuals(
                main_reg.item_block_default,
                ItemVisuals::new(backgrounds::material_item(ImageSource::Data(image_data))),
            );
        }

        Arc::new(Self {
            mapping: Arc::new(map),
            material_water,
            material_lawn,
        })
    }

    pub fn storage(&self) -> &Arc<ResourceMapping> {
        &self.mapping
    }

    add_getters! { u16, material_lawn material_water }
}
