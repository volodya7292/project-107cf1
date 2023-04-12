use crate::module::main_renderer::resource_manager::ResourceManagementScope;
use crate::module::main_renderer::stage::post_process::PostProcessStage;
use crate::module::main_renderer::stage::{RenderStage, RenderStageId, StageContext, StageRunResult};
use std::sync::Arc;
use vk_wrapper::sampler::SamplerClamp;
use vk_wrapper::{
    AccessFlags, Attachment, AttachmentRef, BindingRes, CmdList, Device, Framebuffer, ImageLayout, ImageMod,
    LoadStore, PipelineSignature, PipelineStageFlags, PrimitiveTopology, QueueType, Sampler, SamplerFilter,
    SamplerMipmap, SignalSemaphore, Subpass, WaitSemaphore,
};

pub struct ComposeStage {
    device: Arc<Device>,
    compose_signature: Arc<PipelineSignature>,
    compose_sampler: Arc<Sampler>,
}

impl ComposeStage {
    pub fn new(device: &Arc<Device>) -> Self {
        let quad_vert_shader = device
            .create_vertex_shader(
                include_bytes!("../../../../shaders/build/quad.vert.spv"),
                &[],
                "quad.vert",
            )
            .unwrap();
        let compose_pixel_shader = device
            .create_pixel_shader(
                include_bytes!("../../../../shaders/build/compose.frag.spv"),
                "compose.frag",
            )
            .unwrap();
        let compose_signature = device
            .create_pipeline_signature(&[quad_vert_shader, compose_pixel_shader], &[])
            .unwrap();

        let compose_sampler = device
            .create_sampler(
                SamplerFilter::LINEAR,
                SamplerFilter::NEAREST,
                SamplerMipmap::NEAREST,
                SamplerClamp::REPEAT,
                1.0,
            )
            .unwrap();

        Self {
            device: Arc::clone(device),
            compose_signature,
            compose_sampler,
        }
    }
}

impl RenderStage for ComposeStage {
    fn name(&self) -> &'static str {
        "compose"
    }

    fn execution_dependencies(&self) -> Vec<RenderStageId> {
        vec![RenderStageId::of::<PostProcessStage>()]
    }

    fn run(
        &mut self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
    ) -> StageRunResult {
        let render_pass = resources.request(
            "compose-render_pass",
            (ctx.render_sw_image.get().format(),),
            |(format,), _| {
                self.device
                    .create_render_pass(
                        &[Attachment {
                            format: *format,
                            init_layout: ImageLayout::UNDEFINED,
                            final_layout: ImageLayout::PRESENT,
                            load_store: LoadStore::FinalSave,
                        }],
                        &[Subpass::new().with_color(vec![AttachmentRef {
                            index: 0,
                            layout: ImageLayout::COLOR_ATTACHMENT,
                        }])],
                        &[],
                    )
                    .unwrap()
            },
        );

        let compose_pipeline = resources.request(
            "compose-pipeline",
            (Arc::clone(&render_pass),),
            |(render_pass,), _| {
                self.device
                    .create_graphics_pipeline(
                        render_pass,
                        0,
                        PrimitiveTopology::TRIANGLE_LIST,
                        Default::default(),
                        Default::default(),
                        &[],
                        &self.compose_signature,
                        &[],
                    )
                    .unwrap()
            },
        );

        let sw_framebuffers = resources.request(
            "compose-framebuffer",
            (Arc::clone(ctx.swapchain),),
            |(swapchain,), _| {
                let framebuffers: Vec<_> = swapchain
                    .images()
                    .iter()
                    .map(|img| {
                        render_pass
                            .create_framebuffer(
                                img.size_2d(),
                                &[(0, ImageMod::OverrideImage(Arc::clone(img)))],
                            )
                            .unwrap()
                    })
                    .collect();
                Arc::new(framebuffers)
            },
        );

        let post_framebuffer: Arc<Framebuffer> = resources.get(PostProcessStage::RES_OUTPUT_FRAMEBUFFER);
        let source_image = post_framebuffer.get_image(0).unwrap();

        let compose_desc = resources.request_descriptors("compose-desc", compose_pipeline.signature(), 0, 1);

        // -----------------------------------------------------------------------------

        unsafe {
            self.device.update_descriptor_set(
                compose_desc.get(0),
                &[compose_desc.create_binding(
                    0,
                    0,
                    BindingRes::Image(
                        Arc::clone(source_image),
                        Some(Arc::clone(&self.compose_sampler)),
                        ImageLayout::SHADER_READ,
                    ),
                )],
            );
        }

        // -----------------------------------------------------------------------------

        let graphics_queue = self.device.get_queue(QueueType::Graphics);
        let present_queue = self.device.get_queue(QueueType::Present);

        cl.begin(true).unwrap();

        // Compose final swapchain image
        cl.begin_render_pass(
            &render_pass,
            &sw_framebuffers[ctx.render_sw_image.index() as usize],
            &[],
            false,
        );
        cl.bind_pipeline(&compose_pipeline);
        cl.bind_graphics_inputs(&self.compose_signature, 0, &[compose_desc.get(0)], &[]);
        cl.draw(3, 0);
        cl.end_render_pass();

        if graphics_queue != present_queue {
            cl.barrier_image(
                PipelineStageFlags::ALL_GRAPHICS,
                PipelineStageFlags::BOTTOM_OF_PIPE,
                &[ctx
                    .render_sw_image
                    .get()
                    .barrier()
                    .src_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(ImageLayout::PRESENT)
                    .new_layout(ImageLayout::PRESENT)
                    .src_queue(graphics_queue)
                    .dst_queue(present_queue)],
            );
        }

        cl.end().unwrap();

        let signal_semaphores = if graphics_queue == present_queue {
            vec![SignalSemaphore {
                semaphore: Arc::clone(&ctx.frame_completion_semaphore),
                signal_value: 0,
            }]
        } else {
            vec![]
        };

        StageRunResult::new()
            .with_wait_semaphores(vec![WaitSemaphore {
                semaphore: Arc::clone(ctx.swapchain.readiness_semaphore()),
                wait_dst_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                wait_value: 0,
            }])
            .with_signal_semaphores(signal_semaphores)
    }
}
