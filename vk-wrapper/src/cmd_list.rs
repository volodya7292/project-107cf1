use crate::buffer::Buffer;
use crate::device::DeviceWrapper;
use crate::render_pass::vk_clear_value;
use crate::{
    BufferBarrier, ClearValue, DescriptorPool, DeviceBuffer, Framebuffer, HostBuffer, Image, ImageBarrier,
    ImageLayout, ImageView, Pipeline, PipelineSignature, PipelineStageFlags, QueryPool, RenderPass, Sampler,
    ShaderStage,
};
use ahash::AHashMap;
use ash::{version::DeviceV1_0, vk};
use std::sync::{Arc, Mutex};
use std::{mem, ptr, slice};

pub(crate) struct UsedDescriptorPool {
    _pool: Arc<DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

pub struct CmdList {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) pool: vk::CommandPool,
    pub(crate) native: vk::CommandBuffer,
    pub(crate) one_time_exec: bool,
    pub(crate) render_passes: Vec<Arc<RenderPass>>,
    pub(crate) framebuffers: Vec<Arc<Framebuffer>>,
    pub(crate) secondary_cmd_lists: Vec<Arc<Mutex<CmdList>>>,
    pub(crate) pipelines: AHashMap<Arc<Pipeline>, bool>,
    pub(crate) pipeline_signatures: Vec<Arc<PipelineSignature>>,
    pub(crate) descriptor_pools: Vec<UsedDescriptorPool>,
    pub(crate) buffers: AHashMap<Arc<Buffer>, bool>,
    pub(crate) images: Vec<Arc<Image>>,
    pub(crate) image_views: AHashMap<Arc<ImageView>, bool>,
    pub(crate) samplers: AHashMap<Arc<Sampler>, bool>,
    pub(crate) query_pools: Vec<Arc<QueryPool>>,
    pub(crate) last_pipeline: *const Pipeline,
}

unsafe impl Send for CmdList {}

impl CmdList {
    pub(crate) fn clear_resources(&mut self) {
        self.render_passes.clear();
        self.framebuffers.clear();
        self.secondary_cmd_lists.clear();
        self.pipeline_signatures.clear();
        self.descriptor_pools.clear();
        self.images.clear();
        self.query_pools.clear();
        self.last_pipeline = ptr::null();

        self.pipelines.retain(|_, v| {
            let save = *v;
            *v = false;
            save
        });
        self.buffers.retain(|_, v| {
            let save = *v;
            *v = false;
            save
        });
        // TODO: images
        self.image_views.retain(|_, v| {
            let save = *v;
            *v = false;
            save
        });
        self.samplers.retain(|_, v| {
            let save = *v;
            *v = false;
            save
        });
    }

    fn use_pipeline(&mut self, pipeline: &Arc<Pipeline>) {
        if let Some(a) = self.pipelines.get_mut(pipeline) {
            *a = true;
        } else {
            self.pipelines.insert(Arc::clone(pipeline), true);
        }
    }

    fn use_buffer(&mut self, buffer: &Arc<Buffer>) {
        if let Some(a) = self.buffers.get_mut(buffer) {
            *a = true;
        } else {
            self.buffers.insert(Arc::clone(buffer), true);
        }
    }

    fn use_image_view(&mut self, image_view: &Arc<ImageView>) {
        if let Some(a) = self.image_views.get_mut(image_view) {
            *a = true;
        } else {
            self.image_views.insert(Arc::clone(image_view), true);
        }
    }

    fn use_sampler(&mut self, sampler: &Arc<Sampler>) {
        if let Some(a) = self.samplers.get_mut(sampler) {
            *a = true;
        } else {
            self.samplers.insert(Arc::clone(sampler), true);
        }
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
        self.clear_resources();

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
        let clear_values: Vec<vk::ClearValue> = clear_values.iter().map(|val| vk_clear_value(val)).collect();

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
        if Arc::as_ptr(pipeline) == self.last_pipeline {
            return true;
        }

        unsafe {
            self.device_wrapper
                .0
                .cmd_bind_pipeline(self.native, pipeline.bind_point, pipeline.native)
        };

        self.use_pipeline(pipeline);
        self.last_pipeline = Arc::as_ptr(pipeline);

        false
    }

    pub fn use_descriptor_pool(&mut self, descriptor_pool: Arc<DescriptorPool>) -> u32 {
        let descriptor_sets = {
            let pool = descriptor_pool.lock().unwrap();

            pool.used_buffers
                .iter()
                .for_each(|(_, b)| self.use_buffer(&b.buffer));
            pool.used_image_views
                .iter()
                .for_each(|(_, v)| self.use_image_view(v));
            pool.used_samplers.iter().for_each(|(_, s)| self.use_sampler(s));

            pool.allocated.clone()
        };

        self.descriptor_pools.push(UsedDescriptorPool {
            _pool: descriptor_pool,
            descriptor_sets,
        });

        (self.descriptor_pools.len() - 1) as u32
    }

    fn bind_pipeline_input(
        &mut self,
        signature: &Arc<PipelineSignature>,
        bind_point: vk::PipelineBindPoint,
        set_id: u32,
        used_descriptor_pool: u32,
        descriptor_set_id: u32,
    ) {
        let native_set = self
            .descriptor_pools
            .get(used_descriptor_pool as usize)
            .unwrap()
            .descriptor_sets[descriptor_set_id as usize];

        unsafe {
            self.device_wrapper.0.cmd_bind_descriptor_sets(
                self.native,
                bind_point,
                signature.pipeline_layout,
                set_id,
                &[native_set],
                &[],
            );
        };
    }

    pub fn bind_graphics_input(
        &mut self,
        signature: &Arc<PipelineSignature>,
        set_id: u32,
        used_descriptor_pool: u32,
        descriptor_set_id: u32,
    ) {
        self.bind_pipeline_input(
            signature,
            vk::PipelineBindPoint::GRAPHICS,
            set_id,
            used_descriptor_pool,
            descriptor_set_id,
        );
    }

    pub fn bind_compute_input(
        &mut self,
        signature: &Arc<PipelineSignature>,
        set_id: u32,
        used_descriptor_pool: u32,
        descriptor_set_id: u32,
    ) {
        self.bind_pipeline_input(
            signature,
            vk::PipelineBindPoint::COMPUTE,
            set_id,
            used_descriptor_pool,
            descriptor_set_id,
        );
    }

    /// buffers (max: 16): [buffer, offset]
    pub fn bind_vertex_buffers(&mut self, first_binding: u32, buffers: &[(Arc<DeviceBuffer>, u64)]) {
        let mut native_buffers = [vk::Buffer::default(); 16];
        let mut offsets = [0u64; 16];

        for (i, (buffer, offset)) in buffers.iter().enumerate() {
            native_buffers[i] = buffer.buffer.native;
            offsets[i] = *offset;
            self.use_buffer(&buffer.buffer);
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
                vk::IndexType::UINT32,
            )
        };

        self.use_buffer(&buffer.buffer);
    }

    pub fn push_constants(
        &mut self,
        signature: &Arc<PipelineSignature>,
        stage: ShaderStage,
        base_index: u32,
        constants: &[u32],
    ) {
        unsafe {
            let data = slice::from_raw_parts(constants.as_ptr() as *const u8, constants.len() * 4);
            assert!(data.len() <= 128);

            self.device_wrapper.0.cmd_push_constants(
                self.native,
                signature.pipeline_layout,
                stage.0,
                base_index * 4,
                data,
            )
        };
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

    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.device_wrapper
                .0
                .cmd_dispatch(self.native, group_count_x, group_count_y, group_count_z)
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

        self.use_buffer(&src_buffer.buffer);
        self.use_buffer(&dst_buffer.buffer);
    }

    pub fn copy_buffer_to_host<T>(
        &mut self,
        src_buffer: &Arc<DeviceBuffer>,
        src_element_index: u64,
        dst_buffer: &HostBuffer<T>,
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

        self.use_buffer(&src_buffer.buffer);
        self.use_buffer(&dst_buffer.buffer);
    }

    pub fn copy_host_buffer_to_image_2d(
        &mut self,
        src_buffer: &HostBuffer<u8>,
        src_offset: u64,
        dst_image: &Arc<Image>,
        dst_image_layout: ImageLayout,
        dst_offset: (u32, u32),
        dst_mip_level: u32,
        size: (u32, u32),
    ) {
        let region = vk::BufferImageCopy {
            buffer_offset: src_offset,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst_image.wrapper.aspect,
                mip_level: dst_mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D {
                x: dst_offset.0 as i32,
                y: dst_offset.1 as i32,
                z: 0,
            },
            image_extent: vk::Extent3D {
                width: size.0,
                height: size.1,
                depth: 1,
            },
        };

        unsafe {
            self.device_wrapper.0.cmd_copy_buffer_to_image(
                self.native,
                src_buffer.buffer.native,
                dst_image.wrapper.native,
                dst_image_layout.0,
                &[region],
            )
        };

        self.use_buffer(&src_buffer.buffer);
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
                aspect_mask: src_image.wrapper.aspect,
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
                aspect_mask: dst_image.wrapper.aspect,
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
                src_image.wrapper.native,
                src_image_layout.0,
                dst_image.wrapper.native,
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
                aspect_mask: src_image.wrapper.aspect,
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
                aspect_mask: dst_image.wrapper.aspect,
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
                src_image.wrapper.native,
                src_image_layout.0,
                dst_image.wrapper.native,
                dst_image_layout.0,
                &[region],
                vk::Filter::NEAREST,
            )
        };

        self.images
            .extend_from_slice(&[Arc::clone(src_image), Arc::clone(dst_image)]);
    }

    pub fn copy_query_pool_results_to_host<T>(
        &mut self,
        query_pool: &Arc<QueryPool>,
        first_query: u32,
        query_count: u32,
        dst_buffer: &HostBuffer<T>,
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
                mem::size_of::<u32>() as u64,
                vk::QueryResultFlags::WAIT,
            )
        };
        self.query_pools.push(Arc::clone(query_pool));
        self.use_buffer(&dst_buffer.buffer);
    }

    pub fn clear_image(&mut self, image: &Arc<Image>, layout: ImageLayout, color: ClearValue) {
        let range = vk::ImageSubresourceRange {
            aspect_mask: image.wrapper.aspect,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let clear_value = vk_clear_value(&color);

        unsafe {
            if image.wrapper.aspect == vk::ImageAspectFlags::COLOR {
                self.device_wrapper.0.cmd_clear_color_image(
                    self.native,
                    image.wrapper.native,
                    layout.0,
                    &clear_value.color,
                    &[range],
                );
            } else if image.wrapper.aspect == vk::ImageAspectFlags::DEPTH {
                self.device_wrapper.0.cmd_clear_depth_stencil_image(
                    self.native,
                    image.wrapper.native,
                    layout.0,
                    &clear_value.depth_stencil,
                    &[range],
                );
            }
        }
        self.images.push(Arc::clone(image));
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
                self.use_buffer(&v.buffer);
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

    pub fn barrier_buffer(
        &mut self,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        buffer_barriers: &[BufferBarrier],
    ) {
        self.barrier_buffer_image(src_stage_mask, dst_stage_mask, buffer_barriers, &[]);
    }

    pub fn barrier_image(
        &mut self,
        src_stage_mask: PipelineStageFlags,
        dst_stage_mask: PipelineStageFlags,
        image_barriers: &[ImageBarrier],
    ) {
        self.barrier_buffer_image(src_stage_mask, dst_stage_mask, &[], image_barriers);
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

    pub fn debug_full_memory_barrier(&mut self) {
        let access_mask = vk::AccessFlags::INDIRECT_COMMAND_READ
            | vk::AccessFlags::INDEX_READ
            | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
            | vk::AccessFlags::UNIFORM_READ
            | vk::AccessFlags::INPUT_ATTACHMENT_READ
            | vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::TRANSFER_WRITE
            | vk::AccessFlags::HOST_READ
            | vk::AccessFlags::HOST_WRITE;
        let memory_barrier = vk::MemoryBarrier::builder()
            .src_access_mask(access_mask)
            .dst_access_mask(access_mask)
            .build();

        unsafe {
            self.device_wrapper.0.cmd_pipeline_barrier(
                self.native,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::default(),
                &[memory_barrier],
                &[],
                &[],
            )
        };
    }
}

impl Drop for CmdList {
    fn drop(&mut self) {
        unsafe { self.device_wrapper.0.destroy_command_pool(self.pool, None) };
    }
}
