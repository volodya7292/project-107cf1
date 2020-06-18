use crate::device::DeviceWrapper;
use crate::SwapchainImage;
use crate::{pipeline::PipelineStageFlags, swapchain, CmdList, Fence, Semaphore};
use ash::version::DeviceV1_0;
use ash::vk;
use std::cell::RefCell;
use std::rc::Rc;
use std::slice;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueType(pub(crate) u32);

pub struct Queue {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
    pub(crate) native: vk::Queue,
    pub(crate) semaphore: Semaphore,
    pub(crate) timeline_sp: Rc<Semaphore>,
    pub(crate) fence: Fence,
    pub(crate) family_index: u32,
}

impl Queue {
    pub const TYPE_GRAPHICS: QueueType = QueueType(0);
    pub const TYPE_COMPUTE: QueueType = QueueType(1);
    pub const TYPE_TRANSFER: QueueType = QueueType(2);
    pub const TYPE_PRESENT: QueueType = QueueType(3);

    fn create_cmd_list(&self, level: vk::CommandBufferLevel) -> Result<Rc<CmdList>, vk::Result> {
        let create_info = vk::CommandPoolCreateInfo::builder().queue_family_index(self.family_index);
        let native_pool = unsafe { self.device_wrapper.0.create_command_pool(&create_info, None)? };

        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(native_pool)
            .level(level)
            .command_buffer_count(1);

        Ok(Rc::new(CmdList {
            device_wrapper: Arc::clone(&self.device_wrapper),
            pool: native_pool,
            native: unsafe { self.device_wrapper.0.allocate_command_buffers(&alloc_info)?[0] },
            render_passes: RefCell::new(vec![]),
            framebuffers: RefCell::new(vec![]),
            secondary_cmd_lists: RefCell::new(vec![]),
        }))
    }

    pub fn create_primary_cmd_list(&self) -> Result<Rc<CmdList>, vk::Result> {
        self.create_cmd_list(vk::CommandBufferLevel::PRIMARY)
    }

    pub fn create_secondary_cmd_list(&self) -> Result<Rc<CmdList>, vk::Result> {
        self.create_cmd_list(vk::CommandBufferLevel::SECONDARY)
    }

    fn submit_infos(&self, submit_infos: &[SubmitInfo], fence: &Fence) -> Result<(), vk::Result> {
        let mut semaphore_count = 0;
        let mut cmd_list_count = 0;
        for info in submit_infos.iter() {
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
            for sp in &info.wait_semaphores {
                semaphores.push(sp.semaphore.native);
                wait_masks.push(sp.wait_dst_mask.0);
                sp_values.push(sp.wait_value);
            }
            let signal_sp_index = semaphores.len();
            for sp in &info.signal_semaphores {
                semaphores.push(sp.semaphore.native);
                sp_values.push(sp.signal_value);
            }
            let cmd_buffer_index = command_buffers.len();
            for cmd_buffer in &info.cmd_lists {
                command_buffers.push(cmd_buffer.native);
            }

            native_sp_submit_infos.push(
                vk::TimelineSemaphoreSubmitInfo::builder()
                    .wait_semaphore_values(
                        &sp_values[wait_sp_index..wait_sp_index + info.wait_semaphores.len()],
                    )
                    .signal_semaphore_values(
                        &sp_values[signal_sp_index..signal_sp_index + info.signal_semaphores.len()],
                    )
                    .build(),
            );
            native_submit_infos.push(
                vk::SubmitInfo::builder()
                    .wait_semaphores(&semaphores[wait_sp_index..wait_sp_index + info.wait_semaphores.len()])
                    .wait_dst_stage_mask(
                        &wait_masks[wait_sp_index..wait_sp_index + info.wait_semaphores.len()],
                    )
                    .command_buffers(
                        &command_buffers[cmd_buffer_index..cmd_buffer_index + info.cmd_lists.len()],
                    )
                    .signal_semaphores(
                        &semaphores[signal_sp_index..signal_sp_index + info.signal_semaphores.len()],
                    )
                    .push_next(native_sp_submit_infos.last_mut().unwrap())
                    .build(),
            );
        }

        fence.reset()?;
        unsafe {
            self.device_wrapper
                .0
                .queue_submit(self.native, native_submit_infos.as_slice(), fence.native)?
        }
        Ok(())
    }

    pub fn submit(&self, packet: &mut SubmitPacket) -> Result<(), vk::Result> {
        let mut last_signal_value = self.timeline_sp.last_signal_value.get();

        for info in &mut packet.infos {
            info.wait_semaphores.push(WaitSemaphore {
                semaphore: Rc::clone(&self.timeline_sp),
                wait_dst_mask: PipelineStageFlags::TOP_OF_PIPE,
                wait_value: last_signal_value,
            });

            last_signal_value += 1;
            info.signal_semaphores.push(SignalSemaphore {
                semaphore: Rc::clone(&self.timeline_sp),
                signal_value: last_signal_value,
            });
        }

        self.submit_infos(packet.infos.as_slice(), &packet.fence)?;
        self.timeline_sp.last_signal_value.set(last_signal_value);
        Ok(())
    }

    pub fn present(
        &self,
        sw_image: SwapchainImage,
        wait_semaphore: &Semaphore,
        wait_value: u64,
    ) -> Result<bool, swapchain::Error> {
        let wait_dst_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
        let signal_value = 0u64;
        let mut submit_sp_info = vk::TimelineSemaphoreSubmitInfo::builder()
            .wait_semaphore_values(slice::from_ref(&wait_value))
            .signal_semaphore_values(slice::from_ref(&signal_value));
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(slice::from_ref(&wait_semaphore.native))
            .wait_dst_stage_mask(slice::from_ref(&wait_dst_stage))
            //.command_buffers(slice::from_ref(&cmd_list.native))
            .signal_semaphores(slice::from_ref(&self.semaphore.native))
            .push_next(&mut submit_sp_info);

        self.fence.reset()?;
        unsafe {
            self.device_wrapper.0.queue_submit(
                self.native,
                slice::from_ref(&submit_info),
                self.fence.native,
            )?
        };

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&self.semaphore.native))
            .swapchains(slice::from_ref(&sw_image.swapchain.wrapper.native))
            .image_indices(slice::from_ref(&sw_image.index));
        let result = unsafe { self.swapchain_khr.queue_present(self.native, &present_info) };
        self.fence.wait()?;

        let optimal = match result {
            Ok(a) => !a,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Err(swapchain::Error::IncompatibleSurface);
            }
            Err(e) => {
                return Err(swapchain::Error::VkError(e));
            }
        };

        sw_image.swapchain.curr_image.set(None);
        Ok(optimal)
    }

    pub fn get_semaphore(&self) -> &Semaphore {
        &self.timeline_sp
    }
}

impl PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        self.device_wrapper.0.handle() == other.device_wrapper.0.handle()
            && self.native == other.native
            && self.family_index == other.family_index
    }
}

#[derive(Clone)]
pub struct WaitSemaphore {
    pub semaphore: Rc<Semaphore>,
    pub wait_dst_mask: PipelineStageFlags,
    pub wait_value: u64,
}

#[derive(Clone)]
pub struct SignalSemaphore {
    pub semaphore: Rc<Semaphore>,
    pub signal_value: u64,
}

#[derive(Clone)]
pub struct SubmitInfo {
    wait_semaphores: Vec<WaitSemaphore>,
    cmd_lists: Vec<Rc<CmdList>>,
    signal_semaphores: Vec<SignalSemaphore>,
}

impl SubmitInfo {
    pub fn new(wait_semaphores: &[WaitSemaphore], cmd_lists: &[Rc<CmdList>]) -> SubmitInfo {
        SubmitInfo {
            wait_semaphores: wait_semaphores.to_vec(),
            cmd_lists: cmd_lists.to_vec(),
            signal_semaphores: vec![],
        }
    }
}

pub struct SubmitPacket {
    pub(crate) infos: Vec<SubmitInfo>,
    pub(crate) fence: Fence,
}

impl SubmitPacket {
    pub fn set(&mut self, infos: &[SubmitInfo]) -> Result<(), vk::Result> {
        self.wait()?;
        self.infos = infos.to_vec();
        Ok(())
    }

    pub fn get_signal_value(&self, submit_index: u32) -> u64 {
        self.infos[submit_index as usize].signal_semaphores[0].signal_value
    }

    pub fn wait(&self) -> Result<(), vk::Result> {
        self.fence.wait()
    }
}