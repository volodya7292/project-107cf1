use crate::utils::HashMap;
use std::collections::hash_map;
use std::sync::Arc;
use vk_wrapper as vkw;

pub trait UniformStruct {
    fn model_offset() -> u32;
}

#[macro_export]
macro_rules! uniform_struct_impl {
    ($uniform_struct: ty, $model_name: ident) => {
        impl $crate::renderer::material_pipeline::UniformStruct for $uniform_struct {
            fn model_offset() -> u32 {
                let dummy = <$uniform_struct>::default();
                let offset = ((&dummy.$model_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                offset as u32
            }
        }
    };
}

pub struct PipelineConfig<'a> {
    pub render_pass: &'a Arc<vkw::RenderPass>,
    pub signature: &'a Arc<vkw::PipelineSignature>,
    pub subpass_index: u32,
    pub cull_back_faces: bool,
    pub depth_test: bool,
    pub depth_write: bool,
}

pub struct MaterialPipelineSet {
    pub(crate) device: Arc<vkw::Device>,
    pub(crate) main_signature: Arc<vkw::PipelineSignature>,
    pub(crate) pipelines: HashMap<u32, Arc<vkw::Pipeline>>,
    pub(crate) uniform_buffer_size: u32,
    pub(crate) uniform_buffer_model_offset: u32,
}

impl MaterialPipelineSet {
    pub fn prepare_pipeline(&mut self, id: u32, params: &PipelineConfig) {
        match self.pipelines.entry(id) {
            hash_map::Entry::Vacant(entry) => {
                let pipeline = self
                    .device
                    .create_graphics_pipeline(
                        params.render_pass,
                        params.subpass_index,
                        vkw::PrimitiveTopology::TRIANGLE_LIST,
                        vkw::PipelineDepthStencil::new()
                            .depth_test(params.depth_test)
                            .depth_write(params.depth_write),
                        vkw::PipelineRasterization::new().cull_back_faces(params.cull_back_faces),
                        params.signature,
                    )
                    .unwrap();
                entry.insert(Arc::clone(&pipeline));
            }
            _ => {}
        }
    }

    pub fn get_pipeline(&self, id: u32) -> Option<&Arc<vkw::Pipeline>> {
        self.pipelines.get(&id)
    }

    pub fn uniform_buffer_size(&self) -> u32 {
        self.uniform_buffer_size
    }

    /// Get model matrix (mat4) offset in uniform buffer struct
    pub fn uniform_buffer_offset_model(&self) -> u32 {
        self.uniform_buffer_model_offset
    }
}