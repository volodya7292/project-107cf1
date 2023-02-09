use std::convert::TryInto;

use crate::overworld;
use crate::overworld::block::{Block, BlockBuilder};
use crate::overworld::block_model::BlockModel;
use crate::overworld::occluder::Occluder;
use crate::overworld::raw_cluster::{BlockData, CellInfo};
use crate::overworld::structure::world::{biome, Biome};
use crate::overworld::structure::Structure;
use crate::utils::resource_file::ResourceRef;
use crate::utils::HashMap;

pub struct Registry {
    structures: Vec<Structure>,
    biomes: Vec<Biome>,
    // structures_by_level: [Vec<Structure>; overworld::LOD_LEVELS],
    models: Vec<BlockModel>,
    // textures: Vec<(TextureAtlasType, ResourceRef)>,
    // materials: Vec<Material>,
    material_count: u16,
    // textured_models: Vec<TexturedBlockModel>,
    blocks: Vec<Block>,
    block_empty: u16,
    inner_block_state_empty: Option<CellInfo>,
}

impl Registry {
    pub const MODEL_ID_NULL: u16 = u16::MAX;
    pub const MAX_BLOCKS: u16 = u16::MAX - 1;
    pub const MAX_MODELS: u16 = u16::MAX - 1;

    pub fn new() -> Self {
        let mut registry = Registry {
            structures: vec![],
            // structures_by_level: Default::default(),
            biomes: vec![],
            models: vec![],
            // textures: vec![],
            // materials: vec![],
            material_count: 0,
            // textured_models: vec![],
            blocks: vec![],
            block_empty: u16::MAX,
            inner_block_state_empty: None,
        };

        let block_empty = registry.register_block(BlockBuilder::new(Self::MODEL_ID_NULL));
        registry.block_empty = block_empty;

        registry.inner_block_state_empty = Some(CellInfo {
            entity_id: Default::default(),
            block_id: block_empty,
            occluder: Default::default(),
            light_level: Default::default(),
            liquid_state: Default::default(),
            active: false,
        });

        registry
    }

    pub fn register_block_model(&mut self, block_model: BlockModel) -> u16 {
        if self.models.len() == Self::MAX_MODELS as usize {
            panic!("Maximum number of materials is reached!");
        }
        self.models.push(block_model);
        (self.models.len() - 1).try_into().unwrap()
    }

    // pub fn register_texture(&mut self, ty: TextureAtlasType, res_ref: ResourceRef) -> u16 {
    //     if self.textures.len() == Self::MAX_TEXTURES as usize {
    //         panic!("Maximum number of textures is reached!");
    //     }
    //     self.textures.push((ty, res_ref));
    //     (self.textures.len() - 1).try_into().unwrap()
    // }

    // pub fn alloc_material(&mut self) -> u16 {
    //     if self.material_count == Self::MAX_MATERIALS {
    //         panic!("Maximum number of materials is reached!");
    //     }
    //     let id = self.material_count;
    //     self.material_count += 1;
    //     id
    // }

    // pub fn register_material(&mut self, material: Material) -> u16 {
    //     if self.materials.len() == Self::MAX_MATERIALS as usize {
    //         panic!("Maximum number of materials is reached!");
    //     }
    //     self.materials.push(material);
    //     (self.materials.len() - 1).try_into().unwrap()
    // }

    // pub fn alloc_textured_block_model(&mut self) -> u16 {
    //     if self.material_count == Self::MAX_MATERIALS {
    //         panic!("Maximum number of materials is reached!");
    //     }
    //     let id = self.material_count;
    //     self.material_count += 1;
    //     id
    // }

    // pub fn register_textured_block_model(&mut self, textured_block_model: TexturedBlockModel) -> u16 {
    //     if self.models.len() == Self::MAX_MODELS as usize {
    //         panic!("Maximum number of block models is reached!");
    //     }
    //     self.textured_models.push(textured_block_model);
    //     (self.textured_models.len() - 1).try_into().unwrap()
    // }

    pub fn register_block(&mut self, builder: BlockBuilder) -> u16 {
        let block = builder.build(self);

        if self.blocks.len() == Self::MAX_BLOCKS as usize {
            panic!("Maximum number of blocks is reached!");
        }
        self.blocks.push(block);
        (self.blocks.len() - 1) as u16
    }

    pub fn register_biome(&mut self, biome: Biome) -> u32 {
        self.biomes.push(biome);
        (self.biomes.len() - 1) as u32
    }

    pub fn register_structure(&mut self, structure: Structure) -> u32 {
        self.structures.push(structure);
        (self.structures.len() - 1) as u32
    }

    pub fn biomes(&self) -> &[Biome] {
        &self.biomes
    }

    pub fn get_biome(&self, id: u32) -> Option<&Biome> {
        self.biomes.get(id as usize)
    }

    pub fn get_block(&self, id: u16) -> Option<&Block> {
        self.blocks.get(id as usize)
    }

    pub fn get_block_model(&self, id: u16) -> Option<&BlockModel> {
        self.models.get(id as usize)
    }

    // pub fn get_textured_block_model(&self, id: u16) -> Option<&TexturedBlockModel> {
    //     self.textured_models.get(id as usize)
    // }

    pub fn get_structure(&self, id: u32) -> Option<&Structure> {
        self.structures.get(id as usize)
    }

    // pub fn get_structures_by_lod(&self, cluster_level: u32) -> &[Structure] {
    //     &self.structures_by_level[cluster_level as usize]
    // }

    pub fn inner_block_state_empty(&self) -> &CellInfo {
        self.inner_block_state_empty.as_ref().unwrap()
    }
}