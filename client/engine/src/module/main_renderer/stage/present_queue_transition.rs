use crate::module::main_renderer::resource_manager::ResourceManagementScope;
use crate::module::main_renderer::stage::compose::ComposeStage;
use crate::module::main_renderer::stage::{RenderStage, RenderStageId, StageContext, StageRunResult};
use std::sync::Arc;
use vk_wrapper::{CmdList, Device, ImageLayout, PipelineStageFlags, QueueType, SignalSemaphore};

pub struct PresentQueueTransitionStage {
    device: Arc<Device>,
}

impl PresentQueueTransitionStage {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            device: Arc::clone(device),
        }
    }
}

impl RenderStage for PresentQueueTransitionStage {
    fn name(&self) -> &'static str {
        "present_queue_transition"
    }

    fn execution_dependencies(&self) -> Vec<RenderStageId> {
        vec![RenderStageId::of::<ComposeStage>()]
    }

    fn run(&mut self, cl: &mut CmdList, _: &ResourceManagementScope, ctx: &StageContext) -> StageRunResult {
        let graphics_queue = self.device.get_queue(QueueType::Graphics);
        let present_queue = self.device.get_queue(QueueType::Present);

        cl.begin(true).unwrap();
        cl.barrier_image(
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::BOTTOM_OF_PIPE,
            &[ctx
                .render_sw_image
                .get()
                .barrier()
                .old_layout(ImageLayout::PRESENT)
                .new_layout(ImageLayout::PRESENT)
                .src_queue(graphics_queue)
                .dst_queue(present_queue)],
        );
        cl.end().unwrap();

        StageRunResult::new().with_signal_semaphores(vec![SignalSemaphore {
            semaphore: Arc::clone(&ctx.frame_completion_semaphore),
            signal_value: 0,
        }])
    }
}
