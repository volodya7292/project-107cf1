use crate::rendering::item_visuals::ItemVisuals;
use crate::rendering::texture_material::TextureMaterial;
use crate::rendering::textured_block_model::{QuadMaterial, TexturedBlockModel};
use base::overworld::block_model::BlockModel;
use common::resource_file::ResourceRef;
use engine::module::main_renderer::material::MatComponent;
use engine::module::main_renderer::TextureAtlasType;

pub struct ResourceMapping {
    textures: Vec<(TextureAtlasType, ResourceRef)>,
    materials: Vec<TextureMaterial>,
    textured_block_models: Vec<Option<TexturedBlockModel>>,
    null_textured_block_model: Option<TexturedBlockModel>,
    null_material: u16,
    liquid_materials: Vec<Option<u16>>,
    item_visuals: Vec<Option<ItemVisuals>>,
}

impl ResourceMapping {
    pub const MAX_TEXTURES: u16 = 2048;
    pub const MAX_MATERIALS: u16 = u16::MAX - 1;

    pub fn new(
        null_model: &BlockModel,
        null_texture_res: ResourceRef,
        max_textured_models: u16,
        max_textured_liquid: u16,
        max_items: u16,
    ) -> Self {
        let mut this = Self {
            textures: vec![],
            materials: vec![],
            textured_block_models: (0..max_textured_models).map(|_| None).collect(),
            null_textured_block_model: None,
            null_material: u16::MAX,
            liquid_materials: (0..max_textured_liquid).map(|_| None).collect(),
            item_visuals: (0..max_items).map(|_| None).collect(),
        };

        let null_texture = this.register_texture(TextureAtlasType::ALBEDO, null_texture_res);
        let null_material = this.register_material(TextureMaterial::new(MatComponent::Texture(null_texture)));
        let null_tex_model =
            TexturedBlockModel::new(null_model, &[QuadMaterial::new(null_material); 6], &this);

        this.null_material = null_material;
        this.null_textured_block_model = Some(null_tex_model);

        this
    }

    pub fn register_texture(&mut self, ty: TextureAtlasType, res_ref: ResourceRef) -> u16 {
        if self.textures.len() == Self::MAX_TEXTURES as usize {
            panic!("Maximum number of textures is reached!");
        }
        self.textures.push((ty, res_ref));
        (self.textures.len() - 1).try_into().unwrap()
    }

    pub fn register_material(&mut self, material: TextureMaterial) -> u16 {
        if self.materials.len() == Self::MAX_MATERIALS as usize {
            panic!("Maximum number of materials is reached!");
        }
        self.materials.push(material);
        (self.materials.len() - 1).try_into().unwrap()
    }

    pub fn set_block_textured_model(&mut self, block_id: u16, textured_model: TexturedBlockModel) {
        self.textured_block_models[block_id as usize] = Some(textured_model);
    }

    pub fn set_liquid_material(&mut self, liquid_id: u16, material_id: u16) {
        self.liquid_materials[liquid_id as usize] = Some(material_id);
    }

    pub fn set_item_visuals(&mut self, item_id: u32, visuals: ItemVisuals) {
        self.item_visuals[item_id as usize] = Some(visuals);
    }

    pub fn get_material(&self, id: u16) -> Option<&TextureMaterial> {
        self.materials.get(id as usize)
    }

    pub fn textured_model_for_block(&self, block_id: u16) -> &TexturedBlockModel {
        self.textured_block_models
            .get(block_id as usize)
            .and_then(|v| v.as_ref())
            .unwrap_or(self.null_textured_block_model.as_ref().unwrap())
    }

    pub fn material_for_liquid(&self, liquid_id: u16) -> u16 {
        self.liquid_materials
            .get(liquid_id as usize)
            .cloned()
            .flatten()
            .unwrap_or(self.null_material)
    }

    pub fn get_item_visuals(&self, item_id: u32) -> Option<&ItemVisuals> {
        self.item_visuals[item_id as usize].as_ref()
    }

    pub fn textures(&self) -> &[(TextureAtlasType, ResourceRef)] {
        &self.textures
    }

    pub fn materials(&self) -> &[TextureMaterial] {
        &self.materials
    }
}
