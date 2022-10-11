use std::{ptr, slice};
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use parking_lot::{Mutex, RwLock};
use smallvec::SmallVec;

use crate::{CmdList, Fence, pipeline::PipelineStageFlags, Semaphore, swapchain};
use crate::{DeviceError, SwapchainImage};
use crate::device::DeviceWrapper;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct QueueType(pub(crate) u32);

pub struct Queue {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
    pub(crate) native: RwLock<vk::Queue>,
    pub(crate) semaphore: Arc<Semaphore>,
    pub(crate) timeline_sp: Arc<Semaphore>,
    pub(crate) family_index: u32,
}

impl Queue {
    pub const TYPE_GRAPHICS: QueueType = QueueType(0);
    pub const TYPE_COMPUTE: QueueType = QueueType(1);
    pub const TYPE_TRANSFER: QueueType = QueueType(2);
    pub const TYPE_PRESENT: QueueType = QueueType(3);

    fn create_cmd_list(
        &self,
        name: &str,
        level: vk::CommandBufferLevel,
    ) -> Result<Arc<Mutex<CmdList>>, DeviceError> {
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

        Ok(Arc::new(Mutex::new(CmdList {
            device_wrapper: Arc::clone(&self.device_wrapper),
            pool: native_pool,
            native,
            one_time_exec: false,
            last_pipeline: ptr::null(),
            curr_framebuffer_size: (0, 0),
        })))
    }

    pub fn create_primary_cmd_list(&self, name: &str) -> Result<Arc<Mutex<CmdList>>, DeviceError> {
        self.create_cmd_list(name, vk::CommandBufferLevel::PRIMARY)
    }

    pub fn create_secondary_cmd_list(&self, name: &str) -> Result<Arc<Mutex<CmdList>>, DeviceError> {
        self.create_cmd_list(name, vk::CommandBufferLevel::SECONDARY)
    }

    fn submit_infos(&self, submit_infos: &[SubmitInfo], fence: &mut Fence) -> Result<(), vk::Result> {
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
                command_buffers.push(cmd_buffer.lock().native);
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

        let queue = self.native.write();

        fence.reset()?;

        unsafe {
            self.device_wrapper
                .native
                .queue_submit(*queue, &native_submit_infos, fence.native)?
        }

        Ok(())
    }

    /// # Safety
    /// All resources used in command lists must be valid
    pub unsafe fn submit(&self, packet: &mut SubmitPacket) -> Result<(), vk::Result> {
        if packet.infos.is_empty() {
            return Ok(());
        }

        let mut sp_last_signal_value = self.timeline_sp.last_signal_value.lock();
        let mut new_last_signal_value = *sp_last_signal_value;

        for info in &mut packet.infos {
            info.wait_semaphores.push(WaitSemaphore {
                semaphore: Arc::clone(&self.timeline_sp),
                wait_dst_mask: PipelineStageFlags::ALL_COMMANDS,
                wait_value: new_last_signal_value,
            });

            new_last_signal_value += 1;
            let signal_sp = SignalSemaphore {
                semaphore: Arc::clone(&self.timeline_sp),
                signal_value: new_last_signal_value,
            };
            info.signal_semaphores.push(signal_sp.clone());
            info.completion_signal_sp = Some(signal_sp);
        }

        self.submit_infos(&packet.infos, &mut packet.fence)?;
        *sp_last_signal_value = new_last_signal_value;

        // Remove implicitly added semaphores
        for info in &mut packet.infos {
            info.wait_semaphores.pop();
            info.signal_semaphores.pop();
        }

        Ok(())
    }

    pub fn present(&self, sw_image: SwapchainImage) -> Result<bool, swapchain::Error> {
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
            .wait_semaphores(slice::from_ref(&self.semaphore.native))
            .swapchains(slice::from_ref(&*swapchain))
            .image_indices(slice::from_ref(&sw_image.index));
        let result = unsafe { self.swapchain_khr.queue_present(*queue, &present_info) };

        match result {
            Ok(suboptimal) => Ok(suboptimal),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(swapchain::Error::IncompatibleSurface),
            Err(e) => Err(swapchain::Error::VkError(e)),
        }
    }

    pub fn end_of_frame_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
    }

    pub fn timeline_semaphore(&self) -> &Arc<Semaphore> {
        &self.timeline_sp
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

#[derive(Clone)]
pub struct SubmitInfo {
    wait_semaphores: SmallVec<[WaitSemaphore; 4]>,
    cmd_lists: SmallVec<[Arc<Mutex<CmdList>>; 4]>,
    signal_semaphores: SmallVec<[SignalSemaphore; 4]>,
    completion_signal_sp: Option<SignalSemaphore>,
}

impl SubmitInfo {
    pub fn new(
        wait_semaphores: &[WaitSemaphore],
        cmd_lists: &[Arc<Mutex<CmdList>>],
        signal_semaphores: &[SignalSemaphore],
    ) -> SubmitInfo {
        SubmitInfo {
            wait_semaphores: wait_semaphores.iter().cloned().collect(),
            cmd_lists: cmd_lists.iter().cloned().collect(),
            signal_semaphores: signal_semaphores.iter().cloned().collect(),
            completion_signal_sp: None,
        }
    }

    pub fn set_cmd_lists(&mut self, cmd_lists: Vec<Arc<Mutex<CmdList>>>) {
        self.cmd_lists = cmd_lists.into();
    }

    pub fn get_wait_semaphores(&self) -> &[WaitSemaphore] {
        &self.wait_semaphores
    }

    pub fn get_wait_semaphores_mut(&mut self) -> &mut [WaitSemaphore] {
        &mut self.wait_semaphores
    }
}

pub struct SubmitPacket {
    pub(crate) infos: SmallVec<[SubmitInfo; 4]>,
    pub(crate) fence: Fence,
}

impl SubmitPacket {
    pub fn get(&self) -> &[SubmitInfo] {
        &self.infos
    }

    pub fn get_mut(&mut self) -> Result<&mut [SubmitInfo], vk::Result> {
        self.wait()?;
        Ok(&mut self.infos)
    }

    pub fn set(&mut self, infos: &[SubmitInfo]) -> Result<(), vk::Result> {
        self.wait()?;
        self.infos = infos.into();
        Ok(())
    }

    pub fn get_signal_value(&self, submit_index: u32) -> Option<u64> {
        let sp = self.infos[submit_index as usize].completion_signal_sp.as_ref()?;
        Some(sp.signal_value)
    }

    pub fn wait(&mut self) -> Result<(), vk::Result> {
        self.fence.wait()?;

        for info in &self.infos {
            for cmd_list in &info.cmd_lists {
                let mut cmd_list = cmd_list.lock();
                if cmd_list.one_time_exec {
                    cmd_list.clear_resources();
                }
            }
        }

        Ok(())
    }
}

impl Drop for SubmitPacket {
    fn drop(&mut self) {
        self.wait().unwrap();
    }
}
