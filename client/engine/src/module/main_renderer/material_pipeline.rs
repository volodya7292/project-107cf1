use common::types::HashMap;
use std::collections::hash_map;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::device::{SpecConstId, SpecConstValue};
use vk_wrapper::pipeline::{CompareOp, CullMode};
use vk_wrapper::{AttachmentColorBlend, PrimitiveTopology};

pub struct PipelineConfig<'a> {
    pub render_pass: &'a Arc<vkw::RenderPass>,
    pub signature: &'a Arc<vkw::PipelineSignature>,
    pub subpass_index: u32,
    pub cull: CullMode,
    pub blend_attachments: &'a [u32],
    pub depth_test: bool,
    pub depth_write: bool,
    pub spec_consts: &'a [(SpecConstId, SpecConstValue)],
}

pub struct MaterialPipelineSet {
    pub(crate) device: Arc<vkw::Device>,
    pub(crate) main_signature: Arc<vkw::PipelineSignature>,
    pub(crate) pipelines: HashMap<u32, Arc<vkw::Pipeline>>,
    pub(crate) topology: PrimitiveTopology,
    pub(crate) per_object_desc_pool: vkw::DescriptorPool,
    pub(crate) per_frame_desc_pool: vkw::DescriptorPool,
    pub(crate) per_frame_desc: vkw::DescriptorSet,
}

pub type PipelineKindId = u32;

impl MaterialPipelineSet {
    pub fn prepare_pipeline(&mut self, id: PipelineKindId, params: &PipelineConfig) {
        let hash_map::Entry::Vacant(entry) = self.pipelines.entry(id) else {
            return;
        };

        let pipeline = self
            .device
            .create_graphics_pipeline(
                vkw::PipelineOutputInfo::RenderPass {
                    pass: Arc::clone(params.render_pass),
                    subpass: params.subpass_index,
                },
                self.topology,
                vkw::PipelineDepthStencil::new()
                    .depth_test(params.depth_test)
                    .depth_write(params.depth_write)
                    // Reversed-Z
                    .depth_compare_op(CompareOp::GREATER_OR_EQUAL),
                vkw::PipelineRasterization::new().cull(params.cull),
                &params
                    .blend_attachments
                    .iter()
                    .map(|id| (*id, AttachmentColorBlend::default().enabled(true)))
                    .collect::<Vec<_>>(),
                params.signature,
                params.spec_consts,
            )
            .unwrap();

        entry.insert(Arc::clone(&pipeline));
    }

    pub fn get_pipeline(&self, id: PipelineKindId) -> Option<&Arc<vkw::Pipeline>> {
        self.pipelines.get(&id)
    }

    pub fn main_signature(&self) -> &Arc<vkw::PipelineSignature> {
        &self.main_signature
    }
}
