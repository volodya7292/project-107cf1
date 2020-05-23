use crate::{Framebuffer, RenderPass};
use ash::{version::DeviceV1_0, vk};
use std::rc::Rc;

pub struct PrimaryCmdList {
    pub(crate) native_device: Rc<ash::Device>,
    pub(crate) pool: vk::CommandPool,
    pub(crate) native: vk::CommandBuffer,
}

impl PrimaryCmdList {
    pub fn begin(&self, one_time_execution: bool) -> Result<(), vk::Result> {
        let mut begin_info = vk::CommandBufferBeginInfo::builder();
        if one_time_execution {
            begin_info.flags |= vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;
        }

        unsafe {
            self.native_device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.native_device
                .begin_command_buffer(self.native, &begin_info)?;
        }

        Ok(())
    }

    pub fn end(&self) -> Result<(), vk::Result> {
        unsafe { self.native_device.end_command_buffer(self.native) }
    }
}

impl Drop for PrimaryCmdList {
    fn drop(&mut self) {
        unsafe { self.native_device.destroy_command_pool(self.pool, None) };
    }
}

pub struct SecondaryCmdList {
    pub(crate) native_device: Rc<ash::Device>,
    pub(crate) pool: vk::CommandPool,
    pub(crate) native: vk::CommandBuffer,
    pub(crate) renderpass: Option<Rc<RenderPass>>,
    pub(crate) framebuffer: Option<Rc<Framebuffer>>,
}

impl SecondaryCmdList {
    pub fn begin(
        &self,
        renderpass: &Rc<RenderPass>,
        subpass: u32,
        framebuffer: Option<&Rc<Framebuffer>>,
    ) -> Result<(), vk::Result> {
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(renderpass.native)
            .subpass(subpass)
            .framebuffer(framebuffer.map_or(vk::Framebuffer::default(), |fb| fb.native));

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        unsafe {
            self.native_device
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.native_device
                .begin_command_buffer(self.native, &begin_info)?;
        }

        // if (Framebuffer) {
        //     SetViewport(Framebuffer->size);
        //     SetScissor(uint2(0), Framebuffer->size);
        // }

        Ok(())
    }
}
