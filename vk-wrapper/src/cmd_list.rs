use crate::buffer::Buffer;
use crate::device::DeviceWrapper;
use crate::{
    BufferBarrier, ClearValue, DeviceBuffer, Framebuffer, HostBuffer, Image, ImageBarrier, ImageLayout,
    Pipeline, PipelineInput, PipelineSignature, PipelineStageFlags, QueryPool, RenderPass,
};
use ash::{version::DeviceV1_0, vk};
use std::sync::{Arc, Mutex};

pub struct CmdList {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) pool: vk::CommandPool,
    pub(crate) native: vk::CommandBuffer,
    pub(crate) one_time_exec: bool,
    pub(crate) render_passes: Vec<Arc<RenderPass>>,
    pub(crate) framebuffers: Vec<Arc<Framebuffer>>,
    pub(crate) secondary_cmd_lists: Vec<Arc<Mutex<CmdList>>>,
    pub(crate) pipelines: Vec<Arc<Pipeline>>,
    pub(crate) pipeline_signatures: Vec<Arc<PipelineSignature>>,
    pub(crate) pipeline_inputs: Vec<Arc<PipelineInput>>,
    pub(crate) buffers: Vec<Arc<Buffer>>,
    pub(crate) images: Vec<Arc<Image>>,
    pub(crate) query_pools: Vec<Arc<QueryPool>>,
}

impl CmdList {
    pub(crate) fn clear_resources(&mut self) {
        self.render_passes.clear();
        self.framebuffers.clear();
        self.secondary_cmd_lists.clear();
        self.pipelines.clear();
        self.pipeline_signatures.clear();
        self.pipeline_inputs.clear();
        self.buffers.clear();
        self.images.clear();
        self.query_pools.clear();
    }

    pub fn begin(&mut self, one_time_execution: bool) -> Result<(), vk::Result> {
        self.clear_resources();

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

        self.one_time_exec = one_time_execution;
        Ok(())
    }

    pub fn begin_secondary_graphics(
        &mut self,
        one_time_execution: bool,
        render_pass: &Arc<RenderPass>,
        subpass: u32,
        framebuffer: Option<&Arc<Framebuffer>>,
    ) -> Result<(), vk::Result> {
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(render_pass.native)
            .subpass(subpass)
            .framebuffer(framebuffer.map_or(vk::Framebuffer::default(), |fb| fb.native));

        let mut begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE)
            .inheritance_info(&inheritance_info)
            .build();
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

        self.render_passes.push(Arc::clone(render_pass));

        if let Some(framebuffer) = framebuffer {
            self.framebuffers.push(Arc::clone(framebuffer));
            self.set_viewport(framebuffer.size);
            self.set_scissor(framebuffer.size);
        }

        Ok(())
    }

    pub fn end(&mut self) -> Result<(), vk::Result> {
        unsafe { self.device_wrapper.0.end_command_buffer(self.native) }
    }

    pub fn begin_render_pass(
        &mut self,
        render_pass: &Arc<RenderPass>,
        framebuffer: &Arc<Framebuffer>,
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

        self.render_passes.push(Arc::clone(render_pass));
        self.framebuffers.push(Arc::clone(framebuffer));

        if !secondary_cmd_lists {
            self.set_viewport(framebuffer.size);
            self.set_scissor(framebuffer.size);
        }
    }

    pub fn next_subpass(&mut self, secondary_cmd_lists: bool) {
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

    pub fn end_render_pass(&mut self) {
        unsafe { self.device_wrapper.0.cmd_end_render_pass(self.native) };
    }

    pub fn begin_query(&mut self, query_pool: &Arc<QueryPool>, query: u32) {
        unsafe {
            self.device_wrapper.0.cmd_begin_query(
                self.native,
                query_pool.native,
                query,
                vk::QueryControlFlags::default(),
            );
        };
        self.query_pools.push(Arc::clone(query_pool));
    }

    pub fn end_query(&mut self, query: u32) {
        unsafe {
            self.device_wrapper
                .0
                .cmd_end_query(self.native, self.query_pools.last().unwrap().native, query)
        };
    }

    pub fn reset_query_pool(&mut self, query_pool: &Arc<QueryPool>, first_query: u32, query_count: u32) {
        unsafe {
            self.device_wrapper.0.cmd_reset_query_pool(
                self.native,
                query_pool.native,
                first_query,
                query_count,
            )
        };
        self.query_pools.push(Arc::clone(query_pool));
    }

    pub fn set_viewport(&mut self, size: (u32, u32)) {
        unsafe {
            self.device_wrapper.0.cmd_set_viewport(
                self.native,
                0,
                &[vk::Viewport {
                    x: 0f32,
                    y: size.1 as f32 - 1f32,
                    width: size.0 as f32,
                    height: -(size.1 as f32),
                    min_depth: 0f32,
                    max_depth: 1f32,
                }],
            )
        };
    }

    pub fn set_scissor(&mut self, size: (u32, u32)) {
        unsafe {
            self.device_wrapper.0.cmd_set_scissor(
                self.native,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: vk::Extent2D {
                        width: size.0,
                        height: size.1,
                    },
                }],
            )
        };
    }

    /// Returns whether pipeline is already bound
    pub fn bind_pipeline(&mut self, pipeline: &Arc<Pipeline>) -> bool {
        if !self.pipelines.is_empty() && pipeline == self.pipelines.last().unwrap() {
            return true;
        }
        unsafe {
            self.device_wrapper
                .0
                .cmd_bind_pipeline(self.native, pipeline.bind_point, pipeline.native)
        };
        self.pipelines.push(Arc::clone(pipeline));
        false
    }

    fn bind_pipeline_input(
        &mut self,
        signature: &Arc<PipelineSignature>,
        bind_point: vk::PipelineBindPoint,
        set_id: u32,
        pipeline_input: &Arc<PipelineInput>,
    ) {
        unsafe {
            self.device_wrapper.0.cmd_bind_descriptor_sets(
                self.native,
                bind_point,
                signature.pipeline_layout,
                set_id,
                &[pipeline_input.native],
                &[],
            )
        };
        self.pipeline_inputs.push(Arc::clone(pipeline_input));
    }

    pub fn bind_graphics_input(
        &mut self,
        signature: &Arc<PipelineSignature>,
        set_id: u32,
        pipeline_input: &Arc<PipelineInput>,
    ) {
        self.bind_pipeline_input(signature, vk::PipelineBindPoint::GRAPHICS, set_id, pipeline_input);
    }

    /// buffers (max: 16): [buffer, offset]
    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers: &[(Arc<DeviceBuffer>, u64)]) {
        let mut native_buffers = [vk::Buffer::default(); 16];
        let mut offsets = [0u64; 16];

        self.buffers.reserve(buffers.len());

        for (i, (buffer, offset)) in buffers.iter().enumerate() {
            native_buffers[i] = buffer.buffer.native;
            offsets[i] = *offset;
            self.buffers.push(Arc::clone(&buffer.buffer));
        }

        unsafe {
            self.device_wrapper.0.cmd_bind_vertex_buffers(
                self.native,
                first_binding,
                &native_buffers[0..buffers.len()],
                &offsets[0..buffers.len()],
            )
        };
    }

    pub fn bind_index_buffer(&mut self, buffer: &Arc<DeviceBuffer>, offset: u64) {
        unsafe {
            self.device_wrapper.0.cmd_bind_index_buffer(
                self.native,
                buffer.buffer.native,
                offset,
                vk::IndexType::UINT16,
            )
        };
        self.buffers.push(Arc::clone(&buffer.buffer));
    }

    pub fn draw(&mut self, vertex_count: u32, first_vertex: u32) {
        unsafe {
            self.device_wrapper
                .0
                .cmd_draw(self.native, vertex_count, 1, first_vertex, 0)
        };
    }

    pub fn draw_indexed(&mut self, index_count: u32, first_index: u32, vertex_offset: i32) {
        unsafe {
            self.device_wrapper
                .0
                .cmd_draw_indexed(self.native, index_count, 1, first_index, vertex_offset, 0)
        };
    }

    pub fn copy_buffer_to_device<T>(
        &mut self,
        src_buffer: &HostBuffer<T>,
        src_element_index: u64,
        dst_buffer: &Arc<DeviceBuffer>,
        dst_element_index: u64,
        size: u64,
    ) {
        if size == 0 {
            return;
        }

        let region = vk::BufferCopy {
            src_offset: src_element_index * src_buffer.buffer.aligned_elem_size,
            dst_offset: dst_element_index * src_buffer.buffer.aligned_elem_size,
            size: size * src_buffer.buffer.aligned_elem_size,
        };
        unsafe {
            self.device_wrapper.0.cmd_copy_buffer(
                self.native,
                src_buffer.buffer.native,
                dst_buffer.buffer.native,
                &[region],
            )
        };
        self.buffers
            .extend_from_slice(&[Arc::clone(&src_buffer.buffer), Arc::clone(&dst_buffer.buffer)]);
    }

    pub fn copy_host_buffer_to_image(
        &mut self,
        src_buffer: &HostBuffer<u8>,
        src_offset: u64,
        dst_image: &Arc<Image>,
        dst_image_layout: ImageLayout,
    ) {
        let region = vk::BufferImageCopy {
            buffer_offset: src_offset,
            buffer_row_length: dst_image.size.0,
            buffer_image_height: dst_image.size.1,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst_image.aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: dst_image.size.0,
                height: dst_image.size.1,
                depth: 1,
            },
        };

        unsafe {
            self.device_wrapper.0.cmd_copy_buffer_to_image(
                self.native,
                src_buffer.buffer.native,
                dst_image.native,
                dst_image_layout.0,
                &[region],
            )
        };

        self.buffers.push(Arc::clone(&src_buffer.buffer));
        self.images.push(Arc::clone(dst_image));
    }

    pub fn copy_image_2d(
        &mut self,
        src_image: &Arc<Image>,
        src_image_layout: ImageLayout,
        src_offset: (u32, u32),
        src_mip_level: u32,
        dst_image: &Arc<Image>,
        dst_image_layout: ImageLayout,
        dst_offset: (u32, u32),
        dst_mip_level: u32,
        size: (u32, u32),
    ) {
        let region = vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src_image.aspect,
                mip_level: src_mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: vk::Offset3D {
                x: src_offset.0 as i32,
                y: src_offset.1 as i32,
                z: 0,
            },
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst_image.aspect,
                mip_level: dst_mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: vk::Offset3D {
                x: dst_offset.0 as i32,
                y: dst_offset.1 as i32,
                z: 0,
            },
            extent: vk::Extent3D {
                width: size.0,
                height: size.1,
                depth: 1,
            },
        };

        unsafe {
            self.device_wrapper.0.cmd_copy_image(
                self.native,
                src_image.native,
                src_image_layout.0,
                dst_image.native,
                dst_image_layout.0,
                &[region],
            );
        };

        self.images
            .extend_from_slice(&[Arc::clone(src_image), Arc::clone(dst_image)]);
    }

    pub fn blit_image_2d(
        &mut self,
        src_image: &Arc<Image>,
        src_image_layout: ImageLayout,
        src_offset: (u32, u32),
        src_size: (u32, u32),
        src_mip_level: u32,
        dst_image: &Arc<Image>,
        dst_image_layout: ImageLayout,
        dst_offset: (u32, u32),
        dst_size: (u32, u32),
        dst_mip_level: u32,
    ) {
        let region = vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src_image.aspect,
                mip_level: src_mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offsets: [
                vk::Offset3D {
                    x: src_offset.0 as i32,
                    y: src_offset.1 as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: (src_offset.0 + src_size.0) as i32,
                    y: (src_offset.1 + src_size.1) as i32,
                    z: 1,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst_image.aspect,
                mip_level: dst_mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offsets: [
                vk::Offset3D {
                    x: dst_offset.0 as i32,
                    y: dst_offset.1 as i32,
                    z: 0,
                },
                vk::Offset3D {
                    x: (dst_offset.0 + dst_size.0) as i32,
                    y: (dst_offset.1 + dst_size.1) as i32,
                    z: 1,
                },
            ],
        };

        unsafe {
            self.device_wrapper.0.cmd_blit_image(
                self.native,
                src_image.native,
                src_image_layout.0,
                dst_image.native,
                dst_image_layout.0,
                &[region],
                vk::Filter::NEAREST,
            )
        };

        self.images
            .extend_from_slice(&[Arc::clone(src_image), Arc::clone(dst_image)]);
    }

    pub fn copy_query_pool_results_to_host(
        &mut self,
        query_pool: &Arc<QueryPool>,
        first_query: u32,
        query_count: u32,
        dst_buffer: &HostBuffer<u8>,
        dst_offset: u64,
    ) {
        unsafe {
            self.device_wrapper.0.cmd_copy_query_pool_results(
                self.native,
                query_pool.native,
                first_query,
                query_count,
                dst_buffer.buffer.native,
                dst_offset,
                0,
                vk::QueryResultFlags::WAIT,
            )
        };
        self.query_pools.push(Arc::clone(query_pool));
        self.buffers.push(Arc::clone(&dst_buffer.buffer));
    }

    pub fn barrier_buffer_image(
        &mut self,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        buffer_barriers: &[BufferBarrier],
        image_barriers: &[ImageBarrier],
    ) {
        self.buffers.reserve(buffer_barriers.len());
        self.images.reserve(image_barriers.len());

        let native_buffer_barriers: Vec<vk::BufferMemoryBarrier> = buffer_barriers
            .to_vec()
            .iter()
            .map(|v| {
                self.buffers.push(Arc::clone(&v.buffer));
                v.native
            })
            .collect();

        let native_image_barriers: Vec<vk::ImageMemoryBarrier> = image_barriers
            .to_vec()
            .iter()
            .map(|v| {
                self.images.push(Arc::clone(&v.image));
                v.native
            })
            .collect();

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
        &mut self,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        image_barriers: &[ImageBarrier],
    ) {
        self.barrier_buffer_image(src_stage_mask, dst_stage_mask, &[], image_barriers)
    }

    pub fn execute_secondary(&mut self, cmd_lists: &[Arc<Mutex<CmdList>>]) {
        let native_cmd_lists: Vec<vk::CommandBuffer> = cmd_lists
            .iter()
            .map(|cmd_list| cmd_list.lock().unwrap().native)
            .collect();

        unsafe {
            self.device_wrapper
                .0
                .cmd_execute_commands(self.native, &native_cmd_lists)
        };

        self.secondary_cmd_lists.extend_from_slice(cmd_lists);
    }
}

impl Drop for CmdList {
    fn drop(&mut self) {
        unsafe { self.device_wrapper.0.destroy_command_pool(self.pool, None) };
    }
}
