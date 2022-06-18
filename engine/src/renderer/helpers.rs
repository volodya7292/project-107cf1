use crate::renderer::material_pipeline::{MaterialPipelineSet, PipelineConfig, UniformStruct};
use crate::renderer::{
    BufferUpdate, GBVertexMesh, MaterialInfo, TextureAtlasType, ADDITIONAL_PIPELINE_BINDINGS,
    MAX_BASIC_UNIFORM_BLOCK_SIZE, MAX_MATERIAL_COUNT, PIPELINE_COLOR_SOLID, PIPELINE_COLOR_TRANSLUCENT,
    PIPELINE_DEPTH_READ, PIPELINE_DEPTH_READ_WRITE,
};
use crate::resource_file::ResourceRef;
use crate::utils::UInt;
use crate::{utils, Renderer};
use basis_universal::TranscodeParameters;
use range_alloc::{RangeAllocationError, RangeAllocator};
use smallvec::SmallVec;
use std::collections::hash_map;
use std::fmt::{Display, Formatter};
use std::mem;
use std::ops::{Deref, Range};
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{
    BufferHandle, BufferUsageFlags, CmdList, CopyRegion, Device, DeviceBuffer, DeviceError, Queue, Shader,
    ShaderStage,
};

pub struct LargeBuffer {
    buffer: DeviceBuffer,
    allocator: RangeAllocator<u32>,
    alignment: u32,
}

impl BufferHandleImpl for LargeBuffer {
    fn handle(&self) -> BufferHandle {
        self.buffer.handle()
    }
}

#[derive(Clone)]
pub struct LargeBufferAllocation {
    range: Range<u32>,
    aligned_offset: u32,
    len: u32,
}

impl LargeBufferAllocation {
    pub fn start(&self) -> u32 {
        self.aligned_offset
    }

    pub fn len(&self) -> u32 {
        self.len
    }
}

impl LargeBuffer {
    pub fn new(
        device: &Arc<Device>,
        usage: BufferUsageFlags,
        size: u32,
        name: &str,
    ) -> Result<Self, DeviceError> {
        let limits = device.adapter().props().limits;
        // Minimum alignment for RWByteAddressBuffer addressing is 4 bytes
        let mut alignment = 4;

        if usage.contains(BufferUsageFlags::UNIFORM) {
            alignment = alignment.make_mul_of(limits.min_uniform_buffer_offset_alignment as u32);
        }
        if usage.contains(BufferUsageFlags::STORAGE) {
            alignment = alignment.make_mul_of(limits.min_storage_buffer_offset_alignment as u32);
        }

        let buffer = device.create_device_buffer_named(usage, 1, size as u64, name)?;
        let allocator = RangeAllocator::new(0..size);

        Ok(Self {
            buffer,
            allocator,
            alignment: alignment as u32,
        })
    }

    pub fn allocate(&mut self, size: u32) -> Result<LargeBufferAllocation, RendererError> {
        let size_with_align = size + self.alignment - 1;
        let range = self.allocator.allocate_range(size_with_align)?;
        let aligned_offset = range.start.make_mul_of(self.alignment);

        Ok(LargeBufferAllocation {
            range,
            aligned_offset,
            len: size,
        })
    }

    pub fn allocate_multiple(&mut self, sizes: &[u32]) -> Result<Vec<LargeBufferAllocation>, RendererError> {
        let mut allocs = Vec::with_capacity(sizes.len());

        for size in sizes {
            match self.allocate(*size) {
                Ok(alloc) => allocs.push(alloc),
                Err(err) => {
                    for alloc in allocs {
                        self.free(alloc);
                    }
                    return Err(err);
                }
            }
        }

        Ok(allocs)
    }

    pub fn free(&mut self, alloc: LargeBufferAllocation) {
        self.allocator.free_range(alloc.range);
    }
}

impl Deref for LargeBuffer {
    type Target = DeviceBuffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

#[derive(Debug)]
pub enum RendererError {
    // The global buffer is not large enough to contain necessary data
    NotEnoughGBMemory(RangeAllocationError<u32>),
}

impl From<RangeAllocationError<u32>> for RendererError {
    fn from(e: RangeAllocationError<u32>) -> Self {
        Self::NotEnoughGBMemory(e)
    }
}

impl Display for RendererError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Not enough memory inside global buffer")
    }
}

impl Renderer {
    pub fn load_texture_into_atlas(
        &mut self,
        texture_index: u32,
        atlas_type: TextureAtlasType,
        res_ref: ResourceRef,
    ) {
        let res_data = res_ref.read().unwrap();

        let mut t = basis_universal::Transcoder::new();
        t.prepare_transcoding(&res_data).unwrap();

        let img_info = t.image_info(&res_data, 0).unwrap();
        let width = img_info.m_width;
        let height = img_info.m_height;

        if !utils::is_pow_of_2(width as u64)
            || width != height
            || width < (self.settings.texture_quality as u32)
        {
            return;
        }

        let mipmaps: Vec<_> = (0..img_info.m_total_levels)
            .map(|i| {
                t.transcode_image_level(
                    &res_data,
                    atlas_type.basis_decode_type(),
                    TranscodeParameters {
                        image_index: 0,
                        level_index: i,
                        decode_flags: None,
                        output_row_pitch_in_blocks_or_pixels: None,
                        output_rows_in_pixels: None,
                    },
                )
                .unwrap()
            })
            .collect();

        t.end_transcoding();

        let first_level = UInt::log2(width / (self.settings.texture_quality as u32));
        let last_level = UInt::log2(width / 4); // BC block size = 4x4

        self.texture_atlases[atlas_type as usize]
            .set_texture(
                texture_index,
                &mipmaps[(first_level as usize)..(last_level as usize + 1)],
            )
            .unwrap();
    }

    /// Returns id of registered material pipeline.
    pub fn register_material_pipeline<T: UniformStruct>(&mut self, shaders: &[Arc<Shader>]) -> u32 {
        assert!(mem::size_of::<T>() <= MAX_BASIC_UNIFORM_BLOCK_SIZE as usize);

        let main_signature = self
            .device
            .create_pipeline_signature(shaders, &*ADDITIONAL_PIPELINE_BINDINGS)
            .unwrap();

        let vertex_shader = Arc::clone(shaders.iter().find(|v| v.stage() == ShaderStage::VERTEX).unwrap());
        let depth_signature = self
            .device
            .create_pipeline_signature(&[vertex_shader], &*ADDITIONAL_PIPELINE_BINDINGS)
            .unwrap();

        let mut pipeline_set = MaterialPipelineSet {
            device: Arc::clone(&self.device),
            main_signature: Arc::clone(&main_signature),
            pipelines: Default::default(),
            uniform_buffer_size: mem::size_of::<T>() as u32,
            uniform_buffer_model_offset: T::model_offset(),
        };

        pipeline_set.prepare_pipeline(
            PIPELINE_DEPTH_READ,
            &PipelineConfig {
                render_pass: &self.depth_render_pass,
                signature: &depth_signature,
                subpass_index: 0,
                cull_back_faces: true,
                depth_test: true,
                depth_write: false,
            },
        );
        pipeline_set.prepare_pipeline(
            PIPELINE_DEPTH_READ_WRITE,
            &PipelineConfig {
                render_pass: &self.depth_render_pass,
                signature: &depth_signature,
                subpass_index: 0,
                cull_back_faces: true,
                depth_test: true,
                depth_write: true,
            },
        );
        pipeline_set.prepare_pipeline(
            PIPELINE_COLOR_SOLID,
            &PipelineConfig {
                render_pass: &self.g_render_pass,
                signature: &main_signature,
                subpass_index: 0,
                cull_back_faces: true,
                depth_test: true,
                depth_write: false,
            },
        );
        pipeline_set.prepare_pipeline(
            PIPELINE_COLOR_TRANSLUCENT,
            &PipelineConfig {
                render_pass: &self.g_render_pass,
                signature: &main_signature,
                subpass_index: 0,
                cull_back_faces: false,
                depth_test: true,
                depth_write: false,
            },
        );

        if let hash_map::Entry::Vacant(e) = self.g_per_pipeline_pools.entry(main_signature) {
            let pool = e.key().create_pool(1, 16).unwrap();
            e.insert(pool);
        }

        let mat_pipelines = &mut self.material_pipelines;
        mat_pipelines.push(pipeline_set);
        (mat_pipelines.len() - 1) as u32
    }

    pub fn set_material(&mut self, id: u32, info: MaterialInfo) {
        assert!(id < MAX_MATERIAL_COUNT);
        self.material_updates.insert(id, info);
    }

    /// Returns true if vertex mesh of `entity` is being updated (i.e. uploaded to the GPU).
    // pub fn is_vertex_mesh_updating(&self, entity: Entity) -> bool {
    //     self.vertex_mesh_updates.contains_key(&entity)
    //         || self
    //             .vertex_mesh_pending_updates
    //             .iter()
    //             .any(|v| v.entity == entity)
    // }

    /// Copy each [u8] slice to appropriate DeviceBuffer with offset u64
    pub(crate) unsafe fn update_device_buffers(&mut self, updates: &[BufferUpdate]) {
        if updates.is_empty() {
            return;
        }

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);

        let update_count = updates.len();
        let staging_size = self.staging_buffer.size();
        let mut used_size = 0;
        let mut i = 0;

        while i < update_count {
            let mut cl = self.staging_cl.lock();
            cl.begin(true).unwrap();

            while i < update_count {
                let update = &updates[i];

                let (copy_size, new_used_size) = match update {
                    BufferUpdate::Type1(update) => {
                        let copy_size = update.data.len() as u64;
                        assert!(copy_size <= staging_size);
                        (copy_size, used_size + copy_size)
                    }
                    BufferUpdate::Type2(update) => {
                        let copy_size = update.data.len() as u64;
                        assert!(copy_size <= staging_size);
                        (copy_size, used_size + copy_size)
                    }
                };

                if new_used_size > staging_size {
                    used_size = 0;
                    break;
                }

                match update {
                    BufferUpdate::Type1(update) => {
                        self.staging_buffer.write(used_size as u64, &update.data);
                        cl.copy_buffer_to_device(
                            &self.staging_buffer,
                            used_size,
                            &update.buffer,
                            update.offset,
                            copy_size,
                        );
                    }
                    BufferUpdate::Type2(update) => {
                        self.staging_buffer.write(used_size as u64, &update.data);

                        let regions: SmallVec<[CopyRegion; 64]> = update
                            .regions
                            .iter()
                            .map(|region| {
                                CopyRegion::new(
                                    used_size + region.src_offset(),
                                    region.dst_offset(),
                                    (region.size() as u64).try_into().unwrap(),
                                )
                            })
                            .collect();

                        cl.copy_buffer_regions_to_device_bytes(
                            &self.staging_buffer,
                            &update.buffer,
                            &regions,
                        );
                    }
                }

                used_size = new_used_size;
                i += 1;
            }

            cl.end().unwrap();
            drop(cl);

            let submit = &mut self.staging_submit;
            graphics_queue.submit(submit).unwrap();
            submit.wait().unwrap();
        }
    }

    pub(crate) fn bind_and_draw_vertex_mesh(&self, cl: &mut CmdList, vertex_mesh: &GBVertexMesh) {
        let gb_handle = self.global_buffer.handle();
        let binding_offsets: SmallVec<[_; 6]> = vertex_mesh
            .gb_binding_offsets
            .iter()
            .map(|v| (gb_handle, *v as u64))
            .collect();

        if vertex_mesh.raw.indexed && vertex_mesh.raw.index_count > 0 {
            cl.bind_vertex_buffers(0, &binding_offsets);
            cl.bind_index_buffer(&self.global_buffer, vertex_mesh.gb_indices_offset as u64);
            cl.draw_indexed(vertex_mesh.raw.index_count, 0, 0);
        } else if vertex_mesh.raw.vertex_count > 0 {
            cl.bind_vertex_buffers(0, &*binding_offsets);
            cl.draw(vertex_mesh.raw.vertex_count, 0);
        }
    }
}
