use crate::render_engine;
use crate::render_engine::RenderEngine;
use crate::utils::HashMap;
use std::collections::hash_map;
use std::mem;
use std::sync::Arc;
use vk_wrapper as vkw;

pub trait UniformStruct {
    fn model_offset() -> u32;
}

macro_rules! uniform_struct_impl {
    ($uniform_struct: ty, $model_name: ident) => {
        impl $crate::material_pipeline::UniformStruct for $uniform_struct {
            fn model_offset() -> u32 {
                let dummy = <$uniform_struct>::default();
                let offset = ((&dummy.$model_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                offset as u32
            }
        }
    };
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PipelineMapping {
    pub render_pass: Arc<vkw::RenderPass>,
    pub subpass_index: u32,
    pub cull_back_faces: bool,
}

pub struct MaterialPipeline {
    pub(super) device: Arc<vkw::Device>,
    pub(super) signature: Arc<vkw::PipelineSignature>,
    pub(super) pipelines: HashMap<PipelineMapping, Arc<vkw::Pipeline>>,
    pub(super) uniform_buffer_size: u32,
    pub(super) uniform_buffer_model_offset: u32,
}

impl MaterialPipeline {
    pub fn prepare_pipeline(&mut self, mapping: &PipelineMapping) {
        match self.pipelines.entry(mapping.clone()) {
            hash_map::Entry::Vacant(entry) => {
                let pipeline = self
                    .device
                    .create_graphics_pipeline(
                        &mapping.render_pass,
                        mapping.subpass_index,
                        vkw::PrimitiveTopology::TRIANGLE_LIST,
                        vkw::PipelineDepthStencil::new()
                            .depth_test(true)
                            .depth_write(false),
                        vkw::PipelineRasterization::new().cull_back_faces(mapping.cull_back_faces),
                        &self.signature,
                    )
                    .unwrap();
                entry.insert(Arc::clone(&pipeline));
            }
            _ => {}
        }
    }

    pub fn signature(&self) -> &Arc<vkw::PipelineSignature> {
        &self.signature
    }

    pub fn get_pipeline(&self, mapping: &PipelineMapping) -> Option<&Arc<vkw::Pipeline>> {
        self.pipelines.get(mapping)
    }

    pub fn uniform_buffer_size(&self) -> u32 {
        self.uniform_buffer_size
    }

    /// Get model matrix (mat4) offset in uniform buffer struct
    pub fn uniform_buffer_offset_model(&self) -> u32 {
        self.uniform_buffer_model_offset
    }
}
