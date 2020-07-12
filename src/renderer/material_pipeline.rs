use std::collections::{hash_map, HashMap};
use std::mem;
use std::sync::{Arc, Mutex};
use vk_wrapper as vkw;
use vk_wrapper::{Pipeline, RenderPass};

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

pub struct MaterialPipeline {
    device: Arc<vkw::Device>,
    signature: Arc<vkw::PipelineSignature>,
    pipelines: Mutex<HashMap<(Arc<vkw::RenderPass>, u32, bool), Arc<vkw::Pipeline>>>,
    uniform_buffer_size: u32,
    uniform_buffer_model_offset: u32,
}

impl MaterialPipeline {
    pub fn request_pipeline(
        &self,
        render_pass: &Arc<vkw::RenderPass>,
        subpass_index: u32,
        translucency: bool,
    ) -> Arc<vkw::Pipeline> {
        let mut pipelines = self.pipelines.lock().unwrap();
        match pipelines.entry((Arc::clone(render_pass), subpass_index, translucency)) {
            hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
            hash_map::Entry::Vacant(entry) => {
                let pipeline = self
                    .device
                    .create_graphics_pipeline(
                        render_pass,
                        subpass_index,
                        vkw::PrimitiveTopology::TRIANGLE_LIST,
                        vkw::PipelineDepthStencil::new()
                            .depth_test(true)
                            .depth_write(false),
                        vkw::PipelineRasterization::new().cull_back_faces(!translucency),
                        &self.signature,
                    )
                    .unwrap();
                entry.insert(Arc::clone(&pipeline));
                pipeline
            }
        }
    }

    pub fn uniform_buffer_size(&self) -> u32 {
        self.uniform_buffer_size
    }

    /// Get model matrix (mat4) offset in uniform buffer struct
    pub fn uniform_buffer_offset_model(&self) -> u32 {
        self.uniform_buffer_model_offset
    }
}

pub fn new<T: UniformStruct>(
    device: &Arc<vkw::Device>,
    vertex_shader: &Arc<vkw::Shader>,
    g_pixel_shader: &Arc<vkw::Shader>,
) -> Arc<MaterialPipeline> {
    let signature = device
        .create_pipeline_signature(&[Arc::clone(vertex_shader), Arc::clone(g_pixel_shader)])
        .unwrap();

    Arc::new(MaterialPipeline {
        device: Arc::clone(device),
        signature,
        pipelines: Default::default(),
        uniform_buffer_size: mem::size_of::<T>() as u32,
        uniform_buffer_model_offset: T::model_offset(),
    })
}
