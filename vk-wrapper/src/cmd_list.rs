use crate::device::DeviceWrapper;
use crate::{BufferBarrier, ClearValue, Framebuffer, ImageBarrier, PipelineStageFlags, RenderPass};
use ash::{version::DeviceV1_0, vk};
use std::cell::RefCell;
use std::sync::Arc;
use std::{rc::Rc, slice};

pub struct CmdList {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) pool: vk::CommandPool,
    pub(crate) native: vk::CommandBuffer,
    pub(crate) render_passes: RefCell<Vec<Rc<RenderPass>>>,
    pub(crate) framebuffers: RefCell<Vec<Rc<Framebuffer>>>,
    pub(crate) secondary_cmd_lists: RefCell<Vec<Rc<CmdList>>>,
}

impl CmdList {
    pub fn begin(&self, one_time_execution: bool) -> Result<(), vk::Result> {
        self.framebuffers.borrow_mut().clear();
        self.render_passes.borrow_mut().clear();

        let mut begin_info = vk::CommandBufferBeginInfo::builder();
        if one_time_execution {
            begin_info.flags |= vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT;
        }

        unsafe {
            self.device_wrapper
                .0
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.device_wrapper
                .0
                .begin_command_buffer(self.native, &begin_info)?;
        }

        Ok(())
    }

    pub fn begin_secondary_graphics(
        &self,
        render_pass: &Rc<RenderPass>,
        subpass: u32,
        framebuffer: Option<&Rc<Framebuffer>>,
    ) -> Result<(), vk::Result> {
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(render_pass.native)
            .subpass(subpass)
            .framebuffer(framebuffer.map_or(vk::Framebuffer::default(), |fb| fb.native));

        let begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info);

        unsafe {
            self.device_wrapper
                .0
                .reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
            self.device_wrapper
                .0
                .begin_command_buffer(self.native, &begin_info)?;
        }

        self.render_passes.borrow_mut().push(Rc::clone(render_pass));

        if let Some(framebuffer) = framebuffer {
            self.framebuffers.borrow_mut().push(Rc::clone(framebuffer));
            self.set_viewport(framebuffer.size);
            self.set_scissor(framebuffer.size);
        }

        Ok(())
    }

    pub fn end(&self) -> Result<(), vk::Result> {
        unsafe { self.device_wrapper.0.end_command_buffer(self.native) }
    }

    pub fn begin_render_pass(
        &self,
        render_pass: &Rc<RenderPass>,
        framebuffer: &Rc<Framebuffer>,
        clear_values: &[ClearValue],
        secondary_cmd_lists: bool,
    ) {
        let clear_values: Vec<vk::ClearValue> = clear_values
            .iter()
            .map(|val| match val {
                ClearValue::ColorF32(c) => vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: c.to_owned(),
                    },
                },
                ClearValue::Depth(d) => vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: *d,
                        stencil: 0,
                    },
                },
            })
            .collect();

        let begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(render_pass.native)
            .framebuffer(framebuffer.native)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: framebuffer.size.0,
                    height: framebuffer.size.1,
                },
            })
            .clear_values(&clear_values);

        unsafe {
            self.device_wrapper.0.cmd_begin_render_pass(
                self.native,
                &begin_info,
                if secondary_cmd_lists {
                    vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
                } else {
                    vk::SubpassContents::INLINE
                },
            )
        }

        self.render_passes.borrow_mut().push(Rc::clone(render_pass));
        self.framebuffers.borrow_mut().push(Rc::clone(framebuffer));
    }

    pub fn next_subpass(&self, secondary_cmd_lists: bool) {
        unsafe {
            self.device_wrapper.0.cmd_next_subpass(
                self.native,
                if secondary_cmd_lists {
                    vk::SubpassContents::SECONDARY_COMMAND_BUFFERS
                } else {
                    vk::SubpassContents::INLINE
                },
            )
        };
    }

    pub fn end_render_pass(&self) {
        unsafe { self.device_wrapper.0.cmd_end_render_pass(self.native) };
    }

    pub fn set_viewport(&self, size: (u32, u32)) {
        unsafe {
            self.device_wrapper.0.cmd_set_viewport(
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
            self.device_wrapper.0.cmd_set_scissor(
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
            self.device_wrapper.0.cmd_pipeline_barrier(
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

    pub fn execute_secondary(&self, cmd_lists: &[Rc<CmdList>]) {
        let native_cmd_lists: Vec<vk::CommandBuffer> =
            cmd_lists.iter().map(|cmd_list| cmd_list.native).collect();

        unsafe {
            self.device_wrapper
                .0
                .cmd_execute_commands(self.native, &native_cmd_lists)
        };

        self.secondary_cmd_lists.borrow_mut().extend_from_slice(cmd_lists);
    }
}

impl Drop for CmdList {
    fn drop(&mut self) {
        unsafe { self.device_wrapper.0.destroy_command_pool(self.pool, None) };
    }
}
