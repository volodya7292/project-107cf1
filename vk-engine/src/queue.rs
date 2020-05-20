use crate::{pipeline::PipelineStageFlags, CmdList, Semaphore, Swapchain};
use ash::version::DeviceV1_0;
use ash::vk;
use std::slice;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueType(pub(crate) u32);

pub struct Queue {
    pub(crate) native_device: ash::Device,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
    pub(crate) native: vk::Queue,
    pub(crate) family_index: u32,
}

impl Queue {
    pub const TYPE_GRAPHICS: QueueType = QueueType(0);
    pub const TYPE_COMPUTE: QueueType = QueueType(1);
    pub const TYPE_TRANSFER: QueueType = QueueType(2);
    pub const TYPE_PRESENT: QueueType = QueueType(3);

    pub fn submit(&self, submit_infos: &[SubmitInfo]) -> Result<(), vk::Result> {
        let mut semaphore_count = 0;
        let mut cmd_list_count = 0;
        for info in submit_infos {
            semaphore_count += info.wait_semaphores.len() + info.signal_semaphores.len();
            cmd_list_count += info.cmd_lists.len();
        }

        let mut semaphores = Vec::<vk::Semaphore>::with_capacity(semaphore_count);
        let mut wait_masks = Vec::<vk::PipelineStageFlags>::with_capacity(semaphore_count);
        let mut command_buffers = Vec::<vk::CommandBuffer>::with_capacity(cmd_list_count);
        let mut native_submit_infos = Vec::<vk::SubmitInfo>::with_capacity(submit_infos.len());
        let mut sp_values = Vec::<u64>::with_capacity(semaphore_count);
        let mut native_sp_submit_infos =
            Vec::<vk::TimelineSemaphoreSubmitInfo>::with_capacity(submit_infos.len());

        for info in submit_infos {
            let wait_sp_index = semaphores.len();
            for sp in info.wait_semaphores {
                semaphores.push(sp.semaphore.native);
                wait_masks.push(sp.wait_dst_mask.0);
                sp_values.push(sp.wait_value);
            }
            let signal_sp_index = semaphores.len();
            for sp in info.signal_semaphores {
                semaphores.push(sp.semaphore.native);
                sp_values.push(sp.signal_value);
            }
            let cmd_buffer_index = command_buffers.len();
            for cmd_buffer in info.cmd_lists {
                command_buffers.push(cmd_buffer.native);
            }

            native_sp_submit_infos.push(
                vk::TimelineSemaphoreSubmitInfo::builder()
                    .wait_semaphore_values(&sp_values[wait_sp_index..info.wait_semaphores.len()])
                    .signal_semaphore_values(&sp_values[signal_sp_index..info.signal_semaphores.len()])
                    .build(),
            );

            native_submit_infos.push(
                vk::SubmitInfo::builder()
                    .wait_semaphores(&semaphores[wait_sp_index..info.wait_semaphores.len()])
                    .wait_dst_stage_mask(&wait_masks[wait_sp_index..info.wait_semaphores.len()])
                    .command_buffers(&command_buffers[cmd_buffer_index..info.cmd_lists.len()])
                    .signal_semaphores(&semaphores[signal_sp_index..info.signal_semaphores.len()])
                    .push_next(native_sp_submit_infos.last_mut().unwrap())
                    .build(),
            );
        }

        // TODO: RESOURCE (semaphores, cmd lists) HANDLING

        unsafe {
            self.native_device
                .queue_submit(self.native, native_submit_infos.as_slice(), vk::Fence::default())
        }
    }

    pub fn present(
        &self,
        swapchain: &Swapchain,
        image_index: u32,
        wait_semaphore: &Semaphore,
    ) -> Result<bool, vk::Result> {
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&wait_semaphore.native))
            .swapchains(slice::from_ref(&swapchain.native))
            .image_indices(slice::from_ref(&image_index));
        unsafe { self.swapchain_khr.queue_present(self.native, &present_info) }
    }
}

pub struct WaitSemaphore<'a> {
    pub semaphore: &'a Semaphore,
    pub wait_dst_mask: PipelineStageFlags,
    pub wait_value: u64,
}

pub struct SignalSemaphore<'a> {
    pub semaphore: &'a Semaphore,
    pub signal_value: u64,
}

pub struct SubmitInfo<'a> {
    wait_semaphores: &'a [WaitSemaphore<'a>],
    cmd_lists: &'a [&'a CmdList],
    signal_semaphores: &'a [SignalSemaphore<'a>],
}
