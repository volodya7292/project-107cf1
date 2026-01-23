pub mod context;

use self::context::{
    CPUBufferId, CPUBufferInfo, GPUBufferId, GPUContext, GPUImageId, GPUPipelineId, GPUResourceId,
};
use crate::{
    AccessFlags, Binding, BindingRes, BufferHandle, ClearValue, DeviceError, ImageLayout, LoadStore,
    PipelineStageFlags, Semaphore, SignalSemaphore, SubmitInfo, WaitSemaphore, buffer::BufferHandleImpl,
};
use common::types::HashMap;
use smallvec::SmallVec;
use std::sync::Arc;

pub struct DrawCall {
    pub data_buffer: GPUBufferId,
    /// Buffer offset for index buffer.
    pub indices_offset: u64,
    /// Provides a buffer offset for each vertex attribute.
    pub vertices_offsets: SmallVec<[u64; 4]>,
    /// If `None`, the draw call won't use indices.
    pub num_indices: Option<u32>,
    pub num_vertices: u32,
    pub num_instances: u32,
    pub pipeline: GPUPipelineId,
    pub set1_bindings: SmallVec<[GPUBinding; 2]>,
}

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl MemAccess {
    pub fn from_flags(readable: bool, writable: bool) -> Self {
        match (readable, writable) {
            (true, true) => Self::ReadWrite,
            (true, false) => Self::ReadOnly,
            (false, true) => Self::WriteOnly,
            (false, false) => panic!("should be at least readable or writable"),
        }
    }
}

pub struct GPUBinding {
    pub id: u32,
    pub resource: GPUResourceId,
    pub mem_access: MemAccess,
}

pub struct PushConstants([u8; 128]);

pub struct GraphicsTask {
    pub set0_bindings: Vec<GPUBinding>,
    pub draws: Vec<DrawCall>,
    pub attachments: Vec<(GPUImageId, LoadStore)>,
    pub clear_values: Vec<ClearValue>,
    pub depth_attachment: Option<(GPUImageId, LoadStore)>,
    pub depth_clear_value: ClearValue,
}

pub struct ComputeTask {
    pub set0_bindings: Vec<GPUBinding>,
    pub push_constants: PushConstants,
    pub num_groups_x: u32,
    pub num_groups_y: u32,
    pub num_groups_z: u32,
    pub pipeline: GPUPipelineId,
}

pub struct CopyTask {
    pub src_buffer: GPUBufferId,
    pub dst_buffer: GPUBufferId,
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

pub struct CopyToDeviceBufferTask {
    pub src_data: Vec<u8>,
    pub dst_buffer: GPUBufferId,
    pub dst_offset: u64,
}

pub struct CopyToDeviceImageTask {
    pub src_data: Vec<u8>,
    pub dst_image: GPUImageId,
    pub dst_offset: (u32, u32),
    pub dst_mip_level: u32,
    pub size: (u32, u32),
}

pub struct CopyFromDeviceTask {
    pub src_buffer: GPUBufferId,
    pub dst_buffer: CPUBufferId,
    pub src_offset: u64,
    pub dst_offset: u64,
    pub size: u64,
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub struct GPUImageBarrier {
    image_id: GPUImageId,
    old_layout: ImageLayout,
    new_layout: ImageLayout,
    src_access: AccessFlags,
    dst_access: AccessFlags,
}

#[derive(Clone, Default, Eq, PartialEq)]
pub struct ExecMemBarrierTask {
    src_stage: PipelineStageFlags,
    dst_stage: PipelineStageFlags,
    src_access: AccessFlags,
    dst_access: AccessFlags,
    image_barriers: Vec<GPUImageBarrier>,
}

pub struct BufferClearTask {
    buffer_id: GPUBufferId,
    value: u32,
}

pub struct ImageClearTask {
    image_id: GPUImageId,
    value: ClearValue,
}

pub enum GPUTask {
    Graphics(GraphicsTask),
    Compute(ComputeTask),
    Copy(CopyTask),
    CopyToDeviceBuffer(CopyToDeviceBufferTask),
    CopyToDeviceImage(CopyToDeviceImageTask),
    CopyFromDevice(CopyFromDeviceTask),
    FillBuffer(BufferClearTask),
    ClearImage(ImageClearTask),
    ExecMemBarrier(ExecMemBarrierTask),
}

impl From<GraphicsTask> for GPUTask {
    fn from(value: GraphicsTask) -> Self {
        Self::Graphics(value)
    }
}

impl From<ComputeTask> for GPUTask {
    fn from(value: ComputeTask) -> Self {
        Self::Compute(value)
    }
}

impl From<CopyTask> for GPUTask {
    fn from(value: CopyTask) -> Self {
        Self::Copy(value)
    }
}

impl From<CopyToDeviceBufferTask> for GPUTask {
    fn from(value: CopyToDeviceBufferTask) -> Self {
        Self::CopyToDeviceBuffer(value)
    }
}

impl From<CopyFromDeviceTask> for GPUTask {
    fn from(value: CopyFromDeviceTask) -> Self {
        Self::CopyFromDevice(value)
    }
}

pub struct GPUJoinHandle {
    owned_context: Option<GPUContext>,
}

impl GPUJoinHandle {
    fn join_ref(&mut self) -> GPUContext {
        let ctx = self.owned_context.take().unwrap();
        ctx.finish_fence.wait().unwrap();
        ctx
    }

    pub fn finish_semaphore(&self) -> &Semaphore {
        &self.owned_context.as_ref().unwrap().finish_semaphore
    }

    pub fn join(mut self) -> GPUContext {
        self.join_ref()
    }
}

impl Drop for GPUJoinHandle {
    fn drop(&mut self) {
        self.join_ref();
    }
}

pub struct GPUTaskGraph {
    syncronized_tasks: Vec<GPUTask>,
}

struct ImageIntermediateInfo {
    accum_access: AccessFlags,
    last_layout: ImageLayout,
}

impl GPUTaskGraph {
    /// The order of execution is determined (1) by the order of tasks in the `Vec`
    /// and (2) by the resource usage by tasks.
    pub fn build(tasks: Vec<GPUTask>, ctx: &GPUContext) -> Self {
        let mut syncronized_tasks = Vec::with_capacity(tasks.len() * 2);
        let mut last_stage = PipelineStageFlags::default();
        let mut last_memory_access = AccessFlags::default();
        let mut last_images_infos: HashMap<_, _> = ctx
            .iter_images()
            .map(|(key, img)| {
                (
                    key,
                    ImageIntermediateInfo {
                        accum_access: Default::default(),
                        last_layout: img.last_layout,
                    },
                )
            })
            .collect();

        for (idx, task) in tasks.into_iter().enumerate() {
            let barrier = Self::construct_next_barrier(
                &task,
                &mut last_stage,
                &mut last_memory_access,
                &mut last_images_infos,
            );
            // Barrier is meaningless at the start because happens-before must be already synchronized
            if idx > 0 {
                syncronized_tasks.push(GPUTask::ExecMemBarrier(barrier));
            }
            syncronized_tasks.push(task);
        }

        Self { syncronized_tasks }
    }

    fn build_cmd_list(&self, ctx: &mut GPUContext) {
        ctx.clear_temp_buffers();

        let mut cl = ctx.cmd_list.lock_arc();
        cl.begin(true).unwrap();

        for task in &self.syncronized_tasks {
            match task {
                GPUTask::Graphics(task) => {
                    {
                        let attachments: Vec<_> = task
                            .attachments
                            .iter()
                            .map(|(image_id, load_store)| {
                                let image = ctx.get_image(*image_id).unwrap();
                                (image.inner.clone(), *load_store)
                            })
                            .collect();
                        let depth_attachment = task.depth_attachment.map(|(image_id, load_store)| {
                            let image = ctx.get_image(image_id).unwrap();
                            (image.inner.clone(), load_store)
                        });

                        cl.begin_rendering(
                            &attachments,
                            depth_attachment,
                            &task.clear_values,
                            task.depth_clear_value,
                        );
                    }

                    for draw in &task.draws {
                        let pipeline = ctx.get_pipeline_mut(draw.pipeline).unwrap();
                        pipeline.set0_descriptor_pool.reset();
                        pipeline.set1_descriptor_pool.reset();
                    }

                    for draw in &task.draws {
                        let pipeline = ctx.get_pipeline_mut(draw.pipeline).unwrap();
                        cl.bind_pipeline(&pipeline.inner);

                        // TODO: implement caching
                        {
                            let set0 = pipeline.set0_descriptor_pool.alloc().unwrap();
                            let set1 = pipeline.set1_descriptor_pool.alloc().unwrap();
                            let pipeline = ctx.get_pipeline(draw.pipeline).unwrap();

                            let bindings0: SmallVec<[Binding; 8]> = task
                                .set0_bindings
                                .iter()
                                .map(|b| {
                                    pipeline.set0_descriptor_pool.create_binding(b.id, 0, {
                                        match b.resource {
                                            GPUResourceId::Buffer(id) => {
                                                BindingRes::Buffer(ctx.get_buffer(id).unwrap().inner.handle())
                                            }
                                            GPUResourceId::Image(id) => BindingRes::Image(
                                                ctx.get_image(id).unwrap().inner.clone(),
                                                None,
                                                match b.mem_access {
                                                    MemAccess::ReadOnly => ImageLayout::SHADER_READ,
                                                    MemAccess::WriteOnly | MemAccess::ReadWrite => {
                                                        ImageLayout::GENERAL
                                                    }
                                                },
                                            ),
                                        }
                                    })
                                })
                                .collect();

                            let bindings1: SmallVec<[Binding; 8]> = draw
                                .set1_bindings
                                .iter()
                                .map(|b| {
                                    pipeline.set0_descriptor_pool.create_binding(b.id, 0, {
                                        match b.resource {
                                            GPUResourceId::Buffer(id) => {
                                                BindingRes::Buffer(ctx.get_buffer(id).unwrap().inner.handle())
                                            }
                                            GPUResourceId::Image(id) => BindingRes::Image(
                                                ctx.get_image(id).unwrap().inner.clone(),
                                                None,
                                                match b.mem_access {
                                                    MemAccess::ReadOnly => ImageLayout::SHADER_READ,
                                                    MemAccess::WriteOnly | MemAccess::ReadWrite => {
                                                        ImageLayout::GENERAL
                                                    }
                                                },
                                            ),
                                        }
                                    })
                                })
                                .collect();

                            unsafe {
                                ctx.device.update_descriptor_set(set0, &bindings0);
                                ctx.device.update_descriptor_set(set1, &bindings1);
                            }

                            cl.bind_graphics_inputs(&pipeline.inner.signature, 0, &[set0, set1], &[]);
                        }

                        let data_buffer = ctx.get_buffer(draw.data_buffer).unwrap();

                        let vertex_attributes: SmallVec<[(BufferHandle, u64); 8]> = draw
                            .vertices_offsets
                            .iter()
                            .map(|offset| (data_buffer.inner.handle(), *offset))
                            .collect();

                        cl.bind_vertex_buffers(0, &vertex_attributes);

                        if let Some(num_indices) = draw.num_indices {
                            cl.bind_index_buffer(&data_buffer.inner, draw.indices_offset);
                            cl.draw_indexed_instanced(num_indices, 0, 0, 0, draw.num_instances);
                        } else {
                            cl.draw_instanced(draw.num_vertices, 0, 0, draw.num_instances);
                        }
                    }

                    cl.end_rendering();
                }
                GPUTask::Compute(task) => {
                    let pipeline = ctx.get_pipeline_mut(task.pipeline).unwrap();
                    cl.bind_pipeline(&pipeline.inner);

                    {
                        let set0 = pipeline.set0_descriptor_pool.alloc().unwrap();
                        let pipeline = ctx.get_pipeline(task.pipeline).unwrap();

                        let bindings0: SmallVec<[Binding; 8]> = task
                            .set0_bindings
                            .iter()
                            .map(|b| {
                                pipeline.set0_descriptor_pool.create_binding(b.id, 0, {
                                    match b.resource {
                                        GPUResourceId::Buffer(id) => {
                                            BindingRes::Buffer(ctx.get_buffer(id).unwrap().inner.handle())
                                        }
                                        GPUResourceId::Image(id) => BindingRes::Image(
                                            ctx.get_image(id).unwrap().inner.clone(),
                                            None,
                                            match b.mem_access {
                                                MemAccess::ReadOnly => ImageLayout::SHADER_READ,
                                                MemAccess::WriteOnly | MemAccess::ReadWrite => {
                                                    ImageLayout::GENERAL
                                                }
                                            },
                                        ),
                                    }
                                })
                            })
                            .collect();

                        unsafe {
                            ctx.device.update_descriptor_set(set0, &bindings0);
                        }

                        cl.bind_compute_inputs(&pipeline.inner.signature, 0, &[set0], &[]);
                    }

                    let pipeline = ctx.get_pipeline(task.pipeline).unwrap();

                    cl.push_constants(&pipeline.inner.signature, &task.push_constants.0);
                    cl.dispatch(task.num_groups_x, task.num_groups_y, task.num_groups_z);
                }
                GPUTask::Copy(task) => {
                    let src_buffer = ctx.get_buffer(task.src_buffer).unwrap();
                    let dst_buffer = ctx.get_buffer(task.dst_buffer).unwrap();
                    cl.copy_buffer(
                        &src_buffer.inner,
                        task.src_offset,
                        &dst_buffer.inner,
                        task.dst_offset,
                        task.size,
                    );
                }
                GPUTask::CopyToDeviceBuffer(task) => {
                    let src_buffer = ctx
                        .create_temp_host_buffer(CPUBufferInfo::Bytes(&task.src_data))
                        .unwrap();
                    let dst_buffer = ctx.get_buffer(task.dst_buffer).unwrap();
                    cl.copy_buffer(
                        &src_buffer,
                        0,
                        &dst_buffer.inner,
                        task.dst_offset,
                        task.src_data.len() as u64,
                    );
                }
                GPUTask::CopyToDeviceImage(task) => {
                    let src_buffer = ctx
                        .create_temp_host_buffer(CPUBufferInfo::Bytes(&task.src_data))
                        .unwrap();
                    let dst_image = ctx.get_image(task.dst_image).unwrap();

                    cl.copy_host_buffer_to_image_2d(
                        src_buffer,
                        0,
                        &dst_image.inner,
                        ImageLayout::TRANSFER_DST,
                        task.dst_offset,
                        task.dst_mip_level,
                        task.size,
                    );
                }
                GPUTask::CopyFromDevice(task) => {
                    let src_buffer = ctx.get_buffer(task.src_buffer).unwrap();
                    let dst_buffer = ctx.get_cpu_buffer(task.dst_buffer).unwrap();

                    cl.copy_buffer(
                        &src_buffer.inner,
                        task.src_offset,
                        &dst_buffer.inner,
                        task.dst_offset,
                        task.size,
                    );
                }
                GPUTask::FillBuffer(task) => {
                    let buffer = ctx.get_buffer(task.buffer_id).unwrap();
                    cl.fill_buffer(&buffer.inner, task.value);
                }
                GPUTask::ClearImage(task) => {
                    let image = ctx.get_image(task.image_id).unwrap();
                    cl.clear_image(&image.inner, ImageLayout::TRANSFER_DST, task.value);
                }
                GPUTask::ExecMemBarrier(task) => {
                    let image_barriers: Vec<_> = task
                        .image_barriers
                        .iter()
                        .map(|img_barrier| {
                            ctx.get_image(img_barrier.image_id)
                                .unwrap()
                                .inner
                                .barrier()
                                .old_layout(img_barrier.old_layout)
                                .new_layout(img_barrier.new_layout)
                                .src_access_mask(img_barrier.src_access)
                                .dst_access_mask(img_barrier.dst_access)
                        })
                        .collect();

                    cl.barrier_all(
                        task.src_stage,
                        task.dst_stage,
                        task.src_access,
                        task.dst_access,
                        &[],
                        &image_barriers,
                    );
                }
            }
        }

        cl.end().unwrap()
    }

    pub fn run(
        &self,
        mut ctx: GPUContext,
        swapchain_wait_semaphore: Arc<Semaphore>,
    ) -> Result<GPUJoinHandle, DeviceError> {
        self.build_cmd_list(&mut ctx);
        let cmd_list = ctx.cmd_list.lock();

        let queue = ctx.device.get_queue(crate::QueueType::Graphics);
        queue.submit_infos(
            &[SubmitInfo {
                wait_semaphores: vec![WaitSemaphore {
                    semaphore: swapchain_wait_semaphore,
                    wait_dst_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    wait_value: 0,
                }],
                cmd_lists: vec![&cmd_list],
                signal_semaphores: vec![SignalSemaphore {
                    semaphore: ctx.finish_semaphore.clone(),
                    signal_value: 0,
                }],
            }],
            Some(&mut ctx.finish_fence),
        )?;

        drop(cmd_list);
        Ok(GPUJoinHandle {
            owned_context: Some(ctx),
        })
    }

    fn request_image_barrier(
        image_id: GPUImageId,
        last_image_infos: &mut HashMap<GPUImageId, ImageIntermediateInfo>,
        new_layout: ImageLayout,
        new_access: AccessFlags,
    ) -> Option<GPUImageBarrier> {
        let inter_info = last_image_infos.get_mut(&image_id).unwrap();

        if inter_info.last_layout == new_layout {
            return None;
        }

        let img_barrier = GPUImageBarrier {
            image_id,
            old_layout: inter_info.last_layout,
            new_layout,
            src_access: inter_info.accum_access,
            dst_access: new_access,
        };

        inter_info.accum_access = img_barrier.dst_access;
        inter_info.last_layout = img_barrier.new_layout;

        Some(img_barrier)
    }

    fn construct_next_barrier(
        task: &GPUTask,
        last_stage: &mut PipelineStageFlags,
        last_memory_access: &mut AccessFlags,
        last_image_infos: &mut HashMap<GPUImageId, ImageIntermediateInfo>,
    ) -> ExecMemBarrierTask {
        let mut barrier = ExecMemBarrierTask {
            src_stage: *last_stage,
            src_access: *last_memory_access,
            ..Default::default()
        };
        barrier.image_barriers.reserve(last_image_infos.len());

        #[inline]
        fn process_bindings<'a>(
            bindings: impl Iterator<Item = &'a GPUBinding>,
            barrier: &mut ExecMemBarrierTask,
            last_image_infos: &mut HashMap<GPUImageId, ImageIntermediateInfo>,
        ) {
            for binding in bindings {
                if let GPUResourceId::Buffer(_) = binding.resource {
                    let new_access = match binding.mem_access {
                        MemAccess::ReadOnly => AccessFlags::SHADER_READ,
                        MemAccess::WriteOnly => AccessFlags::SHADER_WRITE,
                        MemAccess::ReadWrite => AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
                    };
                    barrier.dst_access |= new_access;
                }
                if let GPUResourceId::Image(id) = binding.resource {
                    let new_layout = match binding.mem_access {
                        MemAccess::ReadOnly => ImageLayout::SHADER_READ,
                        MemAccess::WriteOnly | MemAccess::ReadWrite => ImageLayout::GENERAL,
                    };
                    if let Some(img_barrier) = GPUTaskGraph::request_image_barrier(
                        id,
                        last_image_infos,
                        new_layout,
                        AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE,
                    ) {
                        barrier.image_barriers.push(img_barrier);
                    }
                }
            }
        }

        match task {
            GPUTask::Graphics(task) => {
                barrier.dst_stage = PipelineStageFlags::ALL_GRAPHICS;

                for (attachment, _) in &task.attachments {
                    if let Some(img_barrier) = Self::request_image_barrier(
                        *attachment,
                        last_image_infos,
                        ImageLayout::COLOR_ATTACHMENT,
                        AccessFlags::COLOR_ATTACHMENT_READ | AccessFlags::COLOR_ATTACHMENT_WRITE,
                    ) {
                        barrier.image_barriers.push(img_barrier);
                    }
                }

                if let Some((attachment, _)) = &task.depth_attachment {
                    if let Some(img_barrier) = Self::request_image_barrier(
                        *attachment,
                        last_image_infos,
                        ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                        AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                            | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ) {
                        barrier.image_barriers.push(img_barrier);
                    }
                }

                barrier.dst_access |= AccessFlags::COLOR_ATTACHMENT_READ
                    | AccessFlags::COLOR_ATTACHMENT_WRITE
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE;

                let all_resource_accesses = task
                    .set0_bindings
                    .iter()
                    .chain(task.draws.iter().flat_map(|v| v.set1_bindings.iter()));

                process_bindings(all_resource_accesses, &mut barrier, last_image_infos);
            }
            GPUTask::Compute(task) => {
                barrier.dst_stage = PipelineStageFlags::COMPUTE;
                process_bindings(task.set0_bindings.iter(), &mut barrier, last_image_infos);
            }
            GPUTask::Copy(_) | GPUTask::CopyToDeviceBuffer(_) => {
                barrier.dst_stage = PipelineStageFlags::TRANSFER;
                barrier.dst_access |= AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE;
            }
            GPUTask::CopyToDeviceImage(task) => {
                barrier.dst_stage = PipelineStageFlags::TRANSFER;
                barrier.dst_access |= AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE;

                if let Some(img_barrier) = Self::request_image_barrier(
                    task.dst_image,
                    last_image_infos,
                    ImageLayout::TRANSFER_DST,
                    AccessFlags::TRANSFER_READ | AccessFlags::TRANSFER_WRITE,
                ) {
                    barrier.image_barriers.push(img_barrier);
                }
            }
            GPUTask::FillBuffer(_) => {
                barrier.dst_stage = PipelineStageFlags::TRANSFER;
                barrier.dst_access |= AccessFlags::TRANSFER_WRITE;
            }
            GPUTask::ClearImage(task) => {
                barrier.dst_stage = PipelineStageFlags::TRANSFER;
                barrier.dst_access |= AccessFlags::TRANSFER_WRITE;

                if let Some(img_barrier) = Self::request_image_barrier(
                    task.image_id,
                    last_image_infos,
                    ImageLayout::TRANSFER_DST,
                    AccessFlags::TRANSFER_WRITE,
                ) {
                    barrier.image_barriers.push(img_barrier);
                }
            }
            _ => {}
        }

        *last_stage = barrier.dst_stage;
        *last_memory_access = barrier.dst_access;

        barrier
    }
}
