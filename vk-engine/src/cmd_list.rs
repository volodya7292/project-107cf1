use crate::{BufferBarrier, Framebuffer, ImageBarrier, PipelineStageFlags, RenderPass};
use ash::{version::DeviceV1_0, vk};
use std::{rc::Rc, slice};

pub struct CmdList {
    pub(crate) native_device: Rc<ash::Device>,
    pub(crate) pool: vk::CommandPool,
    pub(crate) native: vk::CommandBuffer,
    pub(crate) renderpass: Option<Rc<RenderPass>>,
    pub(crate) framebuffer: Option<Rc<Framebuffer>>,
}

impl CmdList {
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

    pub fn begin_secondary_graphics(
        &mut self,
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

        self.renderpass = Some(Rc::clone(renderpass));

        if let Some(framebuffer) = framebuffer {
            self.framebuffer = Some(Rc::clone(framebuffer));
            self.set_viewport(framebuffer.size);
            self.set_scissor(framebuffer.size);
        }

        Ok(())
    }

    pub fn end(&self) -> Result<(), vk::Result> {
        unsafe { self.native_device.end_command_buffer(self.native) }
    }

    pub fn set_viewport(&self, size: (u32, u32)) {
        unsafe {
            self.native_device.cmd_set_viewport(
                self.native,
                0,
                slice::from_ref(&vk::Viewport {
                    x: 0f32,
                    y: size.1 as f32 - 1f32,
                    width: size.0 as f32,
                    height: -(size.1 as f32),
                    min_depth: 0f32,
                    max_depth: 1f32,
                }),
            )
        };
    }

    pub fn set_scissor(&self, size: (u32, u32)) {
        unsafe {
            self.native_device.cmd_set_scissor(
                self.native,
                0,
                slice::from_ref(&vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: size.0,
                        height: size.1,
                    },
                }),
            )
        };
    }

    pub fn barrier_buffer_image(
        &self,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        buffer_barriers: &[BufferBarrier],
        image_barriers: &[ImageBarrier],
    ) {
        let native_buffer_barriers: Vec<vk::BufferMemoryBarrier> =
            buffer_barriers.to_vec().iter().map(|v| v.0).collect();
        let native_image_barriers: Vec<vk::ImageMemoryBarrier> =
            image_barriers.to_vec().iter().map(|v| v.0).collect();

        unsafe {
            self.native_device.cmd_pipeline_barrier(
                self.native,
                src_stage_mask.0,
                dst_stage_mask.0,
                vk::DependencyFlags::default(),
                &[],
                native_buffer_barriers.as_slice(),
                native_image_barriers.as_slice(),
            );
        }
    }

    pub fn barrier_image(
        &self,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        image_barriers: &[ImageBarrier],
    ) {
        self.barrier_buffer_image(src_stage_mask, dst_stage_mask, &[], image_barriers)
    }
}

impl Drop for CmdList {
    fn drop(&mut self) {
        unsafe { self.native_device.destroy_command_pool(self.pool, None) };
    }
}
