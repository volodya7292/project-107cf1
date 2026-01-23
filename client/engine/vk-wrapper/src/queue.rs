use crate::device::DeviceWrapper;
use crate::{CmdList, Fence, Semaphore, pipeline::PipelineStageFlags, swapchain};
use crate::{DeviceError, SwapchainImage};
use ash::vk;
use ash::vk::Handle;
use common::parking_lot::RwLock;
use smallvec::SmallVec;
use std::sync::Arc;
use std::{ptr, slice};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(usize)]
pub enum QueueType {
    Graphics = 0,
    Compute = 1,
    Transfer = 2,
    Present = 3,
}

impl QueueType {
    pub(crate) const TOTAL_QUEUES: usize = 4;

    pub fn from_idx(idx: usize) -> Self {
        assert!(idx <= Self::Present as usize);
        unsafe { std::mem::transmute(idx) }
    }
}

pub struct Queue {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
    pub(crate) native: RwLock<vk::Queue>,
    pub(crate) family_index: u32,
    pub(crate) ty: QueueType,
}

impl Queue {
    pub fn family_index(&self) -> u32 {
        self.family_index
    }

    pub fn ty(&self) -> QueueType {
        self.ty
    }

    fn create_cmd_list(&self, name: &str, level: vk::CommandBufferLevel) -> Result<CmdList, DeviceError> {
        let create_info = vk::CommandPoolCreateInfo::builder().queue_family_index(self.family_index);
        let native_pool = unsafe {
            self.device_wrapper
                .native
                .create_command_pool(&create_info, None)?
        };

        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(native_pool)
            .level(level)
            .command_buffer_count(1);
        let native = unsafe { self.device_wrapper.native.allocate_command_buffers(&alloc_info)?[0] };

        unsafe {
            self.device_wrapper.debug_set_object_name(
                vk::ObjectType::COMMAND_POOL,
                native_pool.as_raw(),
                name,
            )?;
            self.device_wrapper.debug_set_object_name(
                vk::ObjectType::COMMAND_BUFFER,
                native.as_raw(),
                name,
            )?;
        }

        Ok(CmdList {
            device_wrapper: Arc::clone(&self.device_wrapper),
            pool: native_pool,
            native,
            one_time_exec: false,
            last_pipeline: ptr::null(),
            curr_framebuffer_size: (0, 0),
        })
    }

    pub fn create_primary_cmd_list(&self, name: &str) -> Result<CmdList, DeviceError> {
        self.create_cmd_list(name, vk::CommandBufferLevel::PRIMARY)
    }

    pub fn create_secondary_cmd_list(&self, name: &str) -> Result<CmdList, DeviceError> {
        self.create_cmd_list(name, vk::CommandBufferLevel::SECONDARY)
    }

    /// # Safety
    /// All resources used in command lists must be valid until all pending operations are complete.
    pub fn submit_infos(
        &self,
        submit_infos: &[SubmitInfo],
        fence: Option<&mut Fence>,
    ) -> Result<(), DeviceError> {
        if submit_infos.is_empty() {
            return Ok(());
        }

        let mut semaphore_count = 0;
        let mut cmd_list_count = 0;
        for info in submit_infos.iter() {
            semaphore_count += info.wait_semaphores.len() + info.signal_semaphores.len();
            cmd_list_count += info.cmd_lists.len();
        }

        let mut semaphores = SmallVec::<[vk::Semaphore; 4]>::with_capacity(semaphore_count);
        let mut wait_masks = SmallVec::<[vk::PipelineStageFlags; 4]>::with_capacity(semaphore_count);
        let mut command_buffers = SmallVec::<[vk::CommandBuffer; 4]>::with_capacity(cmd_list_count);
        let mut native_submit_infos = SmallVec::<[vk::SubmitInfo; 4]>::with_capacity(submit_infos.len());
        let mut sp_values = SmallVec::<[u64; 4]>::with_capacity(semaphore_count);
        let mut native_sp_submit_infos =
            SmallVec::<[vk::TimelineSemaphoreSubmitInfo; 4]>::with_capacity(submit_infos.len());

        for info in submit_infos {
            let wait_sp_index = semaphores.len();
            let wait_masks_index = wait_masks.len();
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
                        &wait_masks[wait_masks_index..wait_masks_index + info.wait_semaphores.len()],
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

        let queue = self.native.write();

        let fence_native = if let Some(fence) = fence {
            fence.reset()?;
            fence.native
        } else {
            vk::Fence::null()
        };

        unsafe {
            self.device_wrapper
                .native
                .queue_submit(*queue, &native_submit_infos, fence_native)?
        }

        Ok(())
    }

    /// Returns whether the swapchain is in suboptimal state.
    pub fn present(
        &self,
        wait_semaphore: &Semaphore,
        sw_image: SwapchainImage,
    ) -> Result<bool, swapchain::Error> {
        let queue = self.native.write();
        let swapchain = sw_image
            .image
            .wrapper
            ._swapchain_wrapper
            .as_ref()
            .unwrap()
            .native
            .lock();

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(slice::from_ref(&wait_semaphore.native))
            .swapchains(slice::from_ref(&*swapchain))
            .image_indices(slice::from_ref(&sw_image.index));
        let result = unsafe { self.swapchain_khr.queue_present(*queue, &present_info) };

        match result {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(swapchain::Error::IncompatibleSurface),
            Err(e) => Err(swapchain::Error::VkError(e)),
        }
    }

    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        let queue = self.native.write();
        unsafe { self.device_wrapper.native.queue_wait_idle(*queue) }
    }
}

impl PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        *self.native.read() == *other.native.read()
    }
}

#[derive(Clone)]
pub struct WaitSemaphore {
    pub semaphore: Arc<Semaphore>,
    pub wait_dst_mask: PipelineStageFlags,
    pub wait_value: u64,
}

#[derive(Clone)]
pub struct SignalSemaphore {
    pub semaphore: Arc<Semaphore>,
    pub signal_value: u64,
}

#[derive(Clone, Default)]
pub struct SubmitInfo<'a> {
    pub wait_semaphores: Vec<WaitSemaphore>,
    pub cmd_lists: Vec<&'a CmdList>,
    pub signal_semaphores: Vec<SignalSemaphore>,
}
