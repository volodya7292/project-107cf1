use std::mem;
use std::sync::Arc;

use basis_universal::TranscodeParameters;
use smallvec::SmallVec;

use core::utils::resource_file::ResourceRef;
use core::utils::UInt;
use vk_wrapper::{CopyRegion, PrimitiveTopology, Queue, Shader, ShaderStageFlags};

use crate::renderer::material_pipeline::{MaterialPipelineSet, PipelineConfig, UniformStruct};
use crate::renderer::{
    BufferUpdate, MaterialInfo, TextureAtlasType, ADDITIONAL_PIPELINE_BINDINGS, DESC_SET_CUSTOM_PER_OBJECT,
    MAX_BASIC_UNIFORM_BLOCK_SIZE, N_MAX_MATERIALS, PIPELINE_COLOR, PIPELINE_COLOR_WITH_BLENDING,
    PIPELINE_DEPTH_WRITE, PIPELINE_TRANSLUCENCY_DEPTHS,
};
use crate::Renderer;

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

        if !core::utils::is_pow_of_2(width as u64)
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
    pub fn register_material_pipeline<T: UniformStruct>(
        &mut self,
        shaders: &[Arc<Shader>],
        topology: PrimitiveTopology,
        cull_back_faces: bool,
    ) -> u32 {
        assert!(mem::size_of::<T>() <= MAX_BASIC_UNIFORM_BLOCK_SIZE as usize);

        let main_signature = self
            .device
            .create_pipeline_signature(shaders, &*ADDITIONAL_PIPELINE_BINDINGS)
            .unwrap();
        let combined_bindings: Vec<_> = main_signature.bindings().clone().into_iter().collect();

        let vertex_shader = Arc::clone(
            shaders
                .iter()
                .find(|v| v.stage() == ShaderStageFlags::VERTEX)
                .unwrap(),
        );

        let depth_signature = self
            .device
            .create_pipeline_signature(&[Arc::clone(&vertex_shader)], &combined_bindings)
            .unwrap();

        let translucency_depth_signature = self
            .device
            .create_pipeline_signature(
                &[
                    Arc::clone(&vertex_shader),
                    Arc::clone(&self.translucency_depths_pixel_shader),
                ],
                &combined_bindings,
            )
            .unwrap();

        let per_object_desc_pool = main_signature
            .create_pool(DESC_SET_CUSTOM_PER_OBJECT, 16)
            .unwrap();

        let mut pipeline_set = MaterialPipelineSet {
            device: Arc::clone(&self.device),
            main_signature: Arc::clone(&main_signature),
            pipelines: Default::default(),
            topology,
            uniform_buffer_size: mem::size_of::<T>() as u32,
            uniform_buffer_model_offset: T::model_offset(),
            per_object_desc_pool,
            custom_per_frame_uniform_desc: None,
        };

        let albedo_attachment_id = 0;

        pipeline_set.prepare_pipeline(
            PIPELINE_DEPTH_WRITE,
            &PipelineConfig {
                render_pass: &self.depth_render_pass,
                signature: &depth_signature,
                subpass_index: 0,
                cull_back_faces,
                blend_attachments: &[],
                depth_test: true,
                depth_write: true,
            },
        );
        pipeline_set.prepare_pipeline(
            PIPELINE_TRANSLUCENCY_DEPTHS,
            &PipelineConfig {
                render_pass: &self.depth_render_pass,
                signature: &translucency_depth_signature,
                subpass_index: 1,
                cull_back_faces,
                blend_attachments: &[],
                depth_test: true,
                depth_write: false,
            },
        );
        pipeline_set.prepare_pipeline(
            PIPELINE_COLOR,
            &PipelineConfig {
                render_pass: &self.g_render_pass,
                signature: &main_signature,
                subpass_index: 0,
                cull_back_faces,
                blend_attachments: &[],
                depth_test: true,
                depth_write: false,
            },
        );
        pipeline_set.prepare_pipeline(
            PIPELINE_COLOR_WITH_BLENDING,
            &PipelineConfig {
                render_pass: &self.g_render_pass,
                signature: &main_signature,
                subpass_index: 0,
                cull_back_faces,
                blend_attachments: &[albedo_attachment_id],
                depth_test: true,
                depth_write: false,
            },
        );

        self.material_pipelines.push(pipeline_set);
        (self.material_pipelines.len() - 1) as u32
    }

    pub fn get_material_pipeline(&self, id: u32) -> Option<&MaterialPipelineSet> {
        self.material_pipelines.get(id as usize)
    }

    pub fn get_material_pipeline_mut(&mut self, id: u32) -> Option<&mut MaterialPipelineSet> {
        self.material_pipelines.get_mut(id as usize)
    }

    pub fn set_material(&mut self, id: u32, info: MaterialInfo) {
        assert!(id < N_MAX_MATERIALS);
        self.material_updates.insert(id, info);
    }

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
                    BufferUpdate::WithOffset(update) => {
                        let copy_size = update.data.len() as u64;
                        assert!(copy_size <= staging_size);
                        (copy_size, used_size + copy_size)
                    }
                    BufferUpdate::Regions(update) => {
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
                    BufferUpdate::WithOffset(update) => {
                        self.staging_buffer.write(used_size as u64, &update.data);
                        cl.copy_buffer_to_device(
                            &self.staging_buffer,
                            used_size,
                            &update.buffer,
                            update.offset,
                            copy_size,
                        );
                    }
                    BufferUpdate::Regions(update) => {
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
}
