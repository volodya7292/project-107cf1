use crate::utils::{HashMap, LruCache, UInt};
use index_pool::IndexPool;
use std::str::Chars;
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    Format, Image, ImageLayout, ImageUsageFlags, Pipeline, PipelineStageFlags,
};

const GLYPH_SIZE: u32 = 64;
const PREFERRED_MAX_GLYPHS: u32 = 1024;

pub struct TextRendererModule {
    pipeline: Arc<Pipeline>,
    _pool: DescriptorPool,
    descriptor: DescriptorSet,
    glyph_array: Arc<Image>,
    char_cache: LruCache<char, u16>,
    allocated_chars: u16,
}

pub struct MBSPayload {
    pub morton_codes_offset: u32,
    pub n_codes: u32,
}

// #[repr(C)]
// struct PushConstants {
//     h: u32,
//     algorithm: MBSAlgorithm,
//     n_values: u32,
//     data_offset: u32,
// }

impl TextRendererModule {
    pub fn new(device: &Arc<Device>) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/rt_morton_bitonic_sort.comp.hlsl.spv"),
                &[],
                &[],
            )
            .unwrap();
        let signature = device.create_pipeline_signature(&[shader], &[]).unwrap();
        let pipeline = device.create_compute_pipeline(&signature).unwrap();

        let glyph_array = device
            .create_image_2d_array_named(
                Format::RGBA8_UNORM,
                1,
                1.0,
                ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                (GLYPH_SIZE, GLYPH_SIZE, PREFERRED_MAX_GLYPHS),
                "glyph_array",
            )
            .unwrap();
        let char_cache = LruCache::new(glyph_array.size().2 as usize);

        let mut pool = signature.create_pool(0, 1).unwrap();
        let descriptor = pool.alloc().unwrap();
        unsafe {
            device.update_descriptor_set(
                descriptor,
                &[pool.create_binding(
                    0,
                    0,
                    BindingRes::Image(Arc::clone(&glyph_array), ImageLayout::SHADER_READ),
                )],
            );
        }

        Self {
            pipeline,
            _pool: pool,
            descriptor,
            glyph_array,
            char_cache,
            allocated_chars: 0,
        }
    }

    pub fn prepare_characters(&mut self, chars: &[char]) {
        for c in chars {
            if !self.char_cache.contains(c) {
                if self.char_cache.len() == self.char_cache.cap() {
                    // let index = self.char_cache.pop_lru().unwrap().1;
                    // self.free_indices.return_id(index as usize)
                }

                // let new_id = self.free_indices.new_id();
            }
        }

        // TODO: Create glyphs in parallel
    }

    pub fn dispatch(&self, cl: &mut CmdList, payload: &MBSPayload) {
        // cl.bind_pipeline(&self.pipeline);
        // cl.bind_compute_input(self.pipeline.signature(), 0, self.descriptor, &[]);
        //
        // let aligned_n_elements = payload.n_codes.next_power_of_two();
        // let work_group_count = UInt::div_ceil(aligned_n_elements / 2, WORK_GROUP_SIZE);
        //
        // let buf_offset = payload.morton_codes_offset as u64;
        // let buf_size = (payload.n_codes * 8) as u64; // sizeof(uvec2) = 8
        // let buf_barrier = self
        //     .gb_barrier
        //     .clone()
        //     .offset(buf_offset)
        //     .size(buf_size)
        //     .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
        //     .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);
        //
        // let mut execute = |h: u32, alg: MBSAlgorithm| {
        //     let consts = PushConstants {
        //         h,
        //         algorithm: alg,
        //         n_values: payload.n_codes,
        //         data_offset: payload.morton_codes_offset,
        //     };
        //     cl.push_constants(self.pipeline.signature(), &consts);
        //     cl.dispatch(work_group_count, 1, 1);
        //     cl.barrier_buffer(
        //         PipelineStageFlags::COMPUTE,
        //         PipelineStageFlags::COMPUTE,
        //         &[buf_barrier],
        //     );
        // };
        //
        // let mut h = WORK_GROUP_SIZE * 2;
        // execute(h, MBSAlgorithm::LocalBitonicMergeSort);
        // h *= 2;
        //
        // while h <= aligned_n_elements {
        //     execute(h, MBSAlgorithm::BigFlip);
        //
        //     let mut hh = h / 2;
        //     while hh > 1 {
        //         if hh <= WORK_GROUP_SIZE * 2 {
        //             execute(hh, MBSAlgorithm::LocalDisperse);
        //             break;
        //         } else {
        //             execute(hh, MBSAlgorithm::BigDisperse);
        //         }
        //         hh /= 2;
        //     }
        //
        //     h *= 2;
        // }
    }
}
