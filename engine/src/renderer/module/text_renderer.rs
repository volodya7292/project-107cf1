use crate::utils::unsafe_slice::UnsafeSlice;
use crate::utils::HashMap;
use crate::HashSet;
use bit_set::BitSet;
use font_kit::font::Font;
use index_pool::IndexPool;
use rayon::prelude::*;
use std::collections::hash_map;
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, BufferUsageFlags, CmdList, DescriptorPool, DescriptorSet, Device,
    DeviceBuffer, Format, HostBuffer, Image, ImageLayout, ImageUsageFlags, Pipeline, PipelineDepthStencil,
    PipelineRasterization, PipelineStageFlags, PrimitiveTopology, RenderPass,
};

const GLYPH_SIZE: u32 = 64;
// RGBA8
const GLYPH_BYTE_SIZE: usize = (GLYPH_SIZE * GLYPH_SIZE * 4) as usize;
const PREFERRED_MAX_GLYPHS: u32 = 1024;
const MSDF_PX_RANGE: f32 = 4.0;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum FontStyle {
    Normal = 0,
    Italic = 1,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GlyphUID(u64);

impl GlyphUID {
    const fn new(glyph_id: u32, font_id: u16, style: FontStyle) -> Self {
        Self(((glyph_id as u64) << 24) | ((font_id as u64) << 8) | (style as u64))
    }

    const fn glyph_id(&self) -> u32 {
        (self.0 >> 24) as u32
    }

    const fn font_id(&self) -> u16 {
        ((self.0 >> 8) & 0xffff) as u16
    }

    const fn style(&self) -> FontStyle {
        let v = (self.0 & 0xff) as u8;
        match v {
            0 => FontStyle::Normal,
            1 => FontStyle::Italic,
            _ => unreachable!(),
        }
    }
}

struct GlyphLocation {
    index: u32,
    ref_count: u32,
}

#[derive(Clone)]
pub struct FontSet {
    normal: Font,
    italic: Option<Font>,
}

impl FontSet {
    /// Takes TTF or OTF encoded data
    fn new(normal: Font, italic: Option<Font>) -> Self {
        Self { normal, italic }
    }

    /// Takes TTF or OTF encoded data
    pub fn from_bytes(normal: Vec<u8>, italic: Option<Vec<u8>>) -> Option<Self> {
        let normal = Font::from_bytes(Arc::new(normal), 0).ok()?;
        let italic = if let Some(data) = italic {
            Some(Font::from_bytes(Arc::new(data), 0).ok()?)
        } else {
            None
        };
        Some(Self { normal, italic })
    }
}

pub struct TextRendererModule {
    pipeline: Arc<Pipeline>,
    _pool: DescriptorPool,
    descriptor: DescriptorSet,
    fonts: Vec<FontSet>,
    glyph_array: Arc<Image>,
    staging_buffer: HostBuffer<u8>,
    char_locations: HashMap<GlyphUID, GlyphLocation>,
    free_locations: BitSet,
    chars_to_load: HashSet<GlyphUID>,
    atlas_overflow_glyph: GlyphUID,
}

#[repr(C)]
struct PushConstants {
    px_range: f32,
}

impl TextRendererModule {
    pub fn new(device: &Arc<Device>, render_pass: &Arc<RenderPass>, subpass_index: u32) -> Self {
        let vertex_shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/text_char.vert.spv"),
                &[
                    ("inGlyphIndex", Format::R32_UINT),
                    ("inColor", Format::RGBA8_UNORM),
                    ("inTransformCol0", Format::RGB32_FLOAT),
                    ("inTransformCol1", Format::RGB32_FLOAT),
                    ("inTransformCol2", Format::RGB32_FLOAT),
                ],
                &[],
            )
            .unwrap();
        let pixel_shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/text_char.frag.spv"),
                &[],
                &[],
            )
            .unwrap();
        let signature = device
            .create_pipeline_signature(&[vertex_shader, pixel_shader], &[])
            .unwrap();
        let pipeline = device
            .create_graphics_pipeline(
                render_pass,
                subpass_index,
                PrimitiveTopology::TRIANGLE_LIST,
                PipelineDepthStencil::default(),
                PipelineRasterization::new().cull_back_faces(true),
                &signature,
            )
            .unwrap();

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

        let size3d = glyph_array.size();
        let staging_buffer = device
            .create_host_buffer_named(
                BufferUsageFlags::TRANSFER_SRC,
                (size3d.0 * size3d.1 * size3d.2 * 4) as u64,
                "glyph_array-staging",
            )
            .unwrap();

        let mut pool = signature.create_pool(1, 1).unwrap();
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

        let fallback_families = [
            font_kit::family_name::FamilyName::Serif,
            font_kit::family_name::FamilyName::SansSerif,
        ];
        let fallback_normal = font_kit::source::SystemSource::new()
            .select_best_match(
                &fallback_families,
                font_kit::properties::Properties::new().style(font_kit::properties::Style::Normal),
            )
            .unwrap();
        let fallback_italic = font_kit::source::SystemSource::new()
            .select_best_match(
                &fallback_families,
                font_kit::properties::Properties::new().style(font_kit::properties::Style::Italic),
            )
            .unwrap();
        let fallback_font = FontSet::new(
            fallback_normal.load().unwrap(),
            Some(fallback_italic.load().unwrap()),
        );

        let mut char_locations = HashMap::with_capacity(size3d.2 as usize);
        let mut free_locations: BitSet = (0..char_locations.capacity()).collect();
        let mut chars_to_load = HashSet::with_capacity(size3d.2 as usize);
        let fonts = vec![fallback_font];
        println!("fallback {}", fonts[0].normal.full_name());
        let atlas_overflow_glyph = {
            let glyph = GlyphUID::new(0, 0, FontStyle::Normal);
            char_locations.insert(
                glyph,
                GlyphLocation {
                    index: 0,
                    ref_count: 1,
                },
            );
            free_locations.remove(0);
            chars_to_load.insert(glyph);
            glyph
        };

        Self {
            pipeline,
            _pool: pool,
            descriptor,
            fonts,
            glyph_array,
            staging_buffer,
            char_locations,
            free_locations,
            chars_to_load,
            atlas_overflow_glyph,
        }
    }

    fn glyph_array_capacity(&mut self) -> u32 {
        self.glyph_array.size().2
    }

    pub fn register_font(&mut self, font_set: FontSet) -> u16 {
        self.fonts.push(font_set);
        (self.fonts.len() - 1) as u16
    }

    pub fn allocate_glyphs(
        &mut self,
        font_id: u16,
        style: FontStyle,
        chars: impl Iterator<Item = char>,
    ) -> Vec<GlyphUID> {
        let size_hint = chars.size_hint();
        let mut glyphs = Vec::with_capacity(size_hint.1.unwrap_or(size_hint.0));

        for c in chars {
            let font_set = &self.fonts[font_id as usize];

            let font = if style == FontStyle::Italic && font_set.italic.is_some() {
                font_set.italic.as_ref().unwrap()
            } else {
                &font_set.normal
            };

            let g_uid = font
                .glyph_for_char(c)
                .map(|id| GlyphUID::new(id, font_id, style))
                .or_else(|| {
                    // Try fallback font
                    let font_set = &self.fonts[0];
                    let font = if style == FontStyle::Italic && font_set.italic.is_some() {
                        font_set.italic.as_ref().unwrap()
                    } else {
                        &font_set.normal
                    };
                    font.glyph_for_char(c).map(|id| GlyphUID::new(id, 0, style))
                })
                .unwrap_or(self.atlas_overflow_glyph);

            if self.char_locations.contains_key(&g_uid) {
                glyphs.push(g_uid);
                continue;
            }

            if let Some(index) = self.free_locations.iter().next() {
                self.free_locations.remove(index);
                self.char_locations.insert(
                    g_uid,
                    GlyphLocation {
                        index: index as u32,
                        ref_count: 1,
                    },
                );
                self.chars_to_load.insert(g_uid);
            } else {
                glyphs.push(self.atlas_overflow_glyph);
            }
        }

        glyphs
    }

    pub fn free_glyphs(&mut self, chars: &[GlyphUID]) {
        for c in chars {
            if let hash_map::Entry::Occupied(mut v) = self.char_locations.entry(*c) {
                let e = v.get_mut();
                if e.ref_count >= 2 {
                    e.ref_count -= 1;
                } else {
                    self.free_locations.insert(e.index as usize);
                    v.remove();
                    self.chars_to_load.remove(c);
                }
            } else {
                unreachable!()
            }
        }
    }

    pub fn load_glyphs(&mut self, cl: &mut CmdList) {
        let staging_slice = UnsafeSlice::new(self.staging_buffer.as_mut_slice());

        self.chars_to_load.par_iter().for_each(|g| {
            let font_set = &self.fonts[g.font_id() as usize];

            let font = if g.style() == FontStyle::Italic && font_set.italic.is_some() {
                font_set.italic.as_ref().unwrap()
            } else {
                &font_set.normal
            };

            let location = &self.char_locations[g];
            let byte_offset = location.index as usize * GLYPH_BYTE_SIZE;

            let output = unsafe { staging_slice.slice_mut(byte_offset..(byte_offset + GLYPH_BYTE_SIZE)) };
            let _ = msdfgen::generate_msdf(font, g.glyph_id(), GLYPH_SIZE, MSDF_PX_RANGE, output);
        });

        cl.barrier_image(
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::TRANSFER,
            &[self
                .glyph_array
                .barrier()
                .old_layout(ImageLayout::SHADER_READ)
                .new_layout(ImageLayout::TRANSFER_DST)
                .dst_access_mask(AccessFlags::TRANSFER_WRITE)],
        );

        for g in self.chars_to_load.drain() {
            let location = &self.char_locations[&g];
            let byte_offset = location.index as usize * GLYPH_BYTE_SIZE;

            cl.copy_host_buffer_to_image_2d_array(
                &self.staging_buffer,
                byte_offset as u64,
                &self.glyph_array,
                ImageLayout::TRANSFER_DST,
                (0, 0, location.index as u32),
                0,
                (GLYPH_SIZE, GLYPH_SIZE),
            );
        }

        cl.barrier_image(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::BOTTOM_OF_PIPE,
            &[self
                .glyph_array
                .barrier()
                .old_layout(ImageLayout::TRANSFER_DST)
                .new_layout(ImageLayout::SHADER_READ)
                .src_access_mask(AccessFlags::TRANSFER_WRITE)],
        );
    }

    pub fn render(&self, cl: &mut CmdList) {
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
