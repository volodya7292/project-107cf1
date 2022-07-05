use crate::ecs::component;
use crate::ecs::component::internal::GlobalTransform;
use crate::ecs::component::simple_text::{FontStyle, TextHAlign, TextStyle};
use crate::ecs::scene::{Entity, Scene};
use crate::ecs::scene_storage::{ComponentStorageImpl, ComponentStorageMut, Event};
use crate::renderer::vertex_mesh::{RawVertexMesh, VertexMeshCreate};
use crate::utils::unsafe_slice::UnsafeSlice;
use crate::utils::HashMap;
use crate::HashSet;
use bit_set::BitSet;
use nalgebra_glm::{Mat4, U8Vec4, Vec2};
use rayon::prelude::*;
use rusttype::Font;
use std::collections::hash_map;
use std::mem;
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::shader::VInputRate;
use vk_wrapper::{
    AccessFlags, AttachmentColorBlend, BindingRes, BufferUsageFlags, CmdList, DescriptorPool, DescriptorSet,
    Device, DeviceBuffer, Format, HostBuffer, Image, ImageLayout, ImageUsageFlags, Pipeline,
    PipelineDepthStencil, PipelineRasterization, PipelineStageFlags, PrimitiveTopology, RenderPass, Sampler,
    SamplerFilter, SamplerMipmap,
};

const GLYPH_SIZE: u32 = 64;
const GLYPH_BYTE_SIZE: usize = (GLYPH_SIZE * GLYPH_SIZE * 4) as usize; // RGBA8
const PREFERRED_MAX_GLYPHS: u32 = 1024;
const MSDF_PX_RANGE: f32 = 4.0;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GlyphUID(u64);

impl GlyphUID {
    fn new(glyph_id: rusttype::GlyphId, font_id: u16, style: FontStyle) -> Self {
        assert!(mem::size_of_val(&glyph_id.0) <= mem::size_of::<u32>());
        Self(((glyph_id.0 as u64) << 24) | ((font_id as u64) << 8) | (style as u64))
    }

    fn glyph_id(&self) -> rusttype::GlyphId {
        rusttype::GlyphId((self.0 >> 24) as u16)
    }

    fn font_id(&self) -> u16 {
        ((self.0 >> 8) & 0xffff) as u16
    }

    fn style(&self) -> FontStyle {
        let v = (self.0 & 0xff) as u8;
        FontStyle::from_u8(v)
    }
}

struct GlyphLocation {
    index: u32,
    ref_count: u32,
}

#[derive(Clone)]
pub struct FontSet {
    normal: Font<'static>,
    italic: Option<Font<'static>>,
}

impl FontSet {
    /// Takes TTF or OTF encoded data
    pub fn from_bytes(normal: Vec<u8>, italic: Option<Vec<u8>>) -> Option<Self> {
        let normal = Font::try_from_vec(normal)?;
        let italic = if let Some(data) = italic {
            Some(Font::try_from_vec(data)?)
        } else {
            None
        };
        Some(Self { normal, italic })
    }

    pub fn best_for_style(&self, style: FontStyle) -> &Font {
        if style == FontStyle::Italic && self.italic.is_some() {
            self.italic.as_ref().unwrap()
        } else {
            &self.normal
        }
    }
}

struct StyledGlyphSequence {
    glyphs: Vec<GlyphUID>,
    style: TextStyle,
}

pub struct TextRendererModule {
    device: Arc<Device>,
    pipeline: Arc<Pipeline>,
    _pool: DescriptorPool,
    descriptor: DescriptorSet,

    glyph_sampler: Arc<Sampler>,
    glyph_array: Arc<Image>,
    glyph_array_initialized: bool,
    uniform_buffer: DeviceBuffer,
    staging_buffer: HostBuffer<u8>,
    staging_uniform_buffer: HostBuffer<UniformData>,

    fonts: Vec<FontSet>,
    char_locations: HashMap<GlyphUID, GlyphLocation>,
    free_locations: BitSet,
    chars_to_load: HashSet<GlyphUID>,
    atlas_overflow_glyph: GlyphUID,

    allocated_glyphs: HashMap<Entity, StyledGlyphSequence>,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct GlyphInstance {
    glyph_index: u32,
    color: U8Vec4,
    offset: Vec2,
}
attributes_impl!(GlyphInstance, glyph_index, color, offset);

#[repr(C)]
struct UniformData {
    px_range: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    proj_view: Mat4,
}

#[repr(C)]
struct PushConstants {
    transform: Mat4,
}

impl TextRendererModule {
    pub fn new(device: &Arc<Device>, render_pass: &Arc<RenderPass>, subpass_index: u32) -> Self {
        let vertex_shader = device
            .create_vertex_shader(
                include_bytes!("../../../shaders/build/text_char.vert.spv"),
                &[
                    ("inGlyphIndex", Format::R32_UINT, VInputRate::INSTANCE),
                    ("inColor", Format::RGBA8_UNORM, VInputRate::INSTANCE),
                    ("inOffset", Format::RG32_FLOAT, VInputRate::INSTANCE),
                ],
                &[],
            )
            .unwrap();
        let pixel_shader = device
            .create_pixel_shader(include_bytes!("../../../shaders/build/text_char.frag.spv"), &[])
            .unwrap();
        let signature = device
            .create_pipeline_signature(&[vertex_shader, pixel_shader], &[])
            .unwrap();
        let pipeline = device
            .create_graphics_pipeline(
                render_pass,
                subpass_index,
                PrimitiveTopology::TRIANGLE_STRIP,
                PipelineDepthStencil::default(),
                PipelineRasterization::new().cull_back_faces(true),
                &[(0, AttachmentColorBlend::default().enabled(true))],
                &signature,
            )
            .unwrap();

        let glyph_array = device
            .create_image_2d_array_named(
                Format::RGBA8_UNORM,
                1,
                ImageUsageFlags::SAMPLED | ImageUsageFlags::TRANSFER_DST,
                (GLYPH_SIZE, GLYPH_SIZE, PREFERRED_MAX_GLYPHS),
                "glyph_array",
            )
            .unwrap();
        // let sampler = device.create_s
        let size3d = glyph_array.size();
        let staging_buffer = device
            .create_host_buffer_named(
                BufferUsageFlags::TRANSFER_SRC,
                (size3d.0 * size3d.1 * size3d.2 * 4) as u64,
                "glyph_array-staging",
            )
            .unwrap();

        let uniform_buffer = device
            .create_device_buffer_named(
                BufferUsageFlags::UNIFORM | BufferUsageFlags::TRANSFER_DST,
                mem::size_of::<UniformData>() as u64,
                1,
                "text_renderer_uniform",
            )
            .unwrap();
        let staging_uniform_buffer = device
            .create_host_buffer_named::<UniformData>(
                BufferUsageFlags::TRANSFER_SRC,
                1,
                "text_renderer_uniform-staging",
            )
            .unwrap();

        let glyph_sampler = device
            .create_sampler(
                SamplerFilter::LINEAR,
                SamplerFilter::LINEAR,
                SamplerMipmap::NEAREST,
                1.0,
            )
            .unwrap();

        let mut pool = signature.create_pool(0, 1).unwrap();
        let descriptor = pool.alloc().unwrap();
        unsafe {
            device.update_descriptor_set(
                descriptor,
                &[
                    pool.create_binding(0, 0, BindingRes::Buffer(uniform_buffer.handle())),
                    pool.create_binding(
                        1,
                        0,
                        BindingRes::Image(
                            Arc::clone(&glyph_array),
                            Some(Arc::clone(&glyph_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                ],
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
        let fallback_font = FontSet::from_bytes(
            fallback_normal.load().unwrap().copy_font_data().unwrap().to_vec(),
            Some(fallback_italic.load().unwrap().copy_font_data().unwrap().to_vec()),
        )
        .unwrap();

        let mut char_locations = HashMap::with_capacity(size3d.2 as usize);
        let mut free_locations: BitSet = (0..char_locations.capacity()).collect();
        let mut chars_to_load = HashSet::with_capacity(size3d.2 as usize);
        let fonts = vec![fallback_font];

        let atlas_overflow_glyph = {
            let glyph = GlyphUID::new(rusttype::GlyphId(0), 0, FontStyle::Normal);
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
            device: Arc::clone(device),
            pipeline,
            _pool: pool,
            descriptor,
            fonts,
            glyph_array,
            glyph_array_initialized: false,
            uniform_buffer,
            staging_buffer,
            char_locations,
            free_locations,
            chars_to_load,
            atlas_overflow_glyph,
            allocated_glyphs: Default::default(),
            staging_uniform_buffer,
            glyph_sampler,
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
        let chars = chars.nfc();
        let size_hint = chars.size_hint();
        let mut glyphs = Vec::with_capacity(size_hint.1.unwrap_or(size_hint.0));

        for c in chars {
            let font = self.fonts[font_id as usize].best_for_style(style);
            let mut g_uid = GlyphUID::new(font.glyph(c).id(), font_id, style);

            if g_uid.glyph_id().0 == 0 {
                // Try fallback font
                let fallback_font = self.fonts[0].best_for_style(style);
                g_uid = GlyphUID::new(fallback_font.glyph(c).id(), 0, style);

                if g_uid.glyph_id().0 == 0 {
                    g_uid = self.atlas_overflow_glyph;
                }
            }

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
                glyphs.push(g_uid);
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
            let font = font_set.best_for_style(g.style());
            let glyph = font.glyph(g.glyph_id());

            let location = &self.char_locations[g];
            let byte_offset = location.index as usize * GLYPH_BYTE_SIZE;

            let output = unsafe { staging_slice.slice_mut(byte_offset..(byte_offset + GLYPH_BYTE_SIZE)) };
            let _ = msdfgen::generate_msdf(glyph, GLYPH_SIZE, MSDF_PX_RANGE, output);
        });

        cl.barrier_image(
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::TRANSFER,
            &[self
                .glyph_array
                .barrier()
                .old_layout(if self.glyph_array_initialized {
                    ImageLayout::SHADER_READ
                } else {
                    ImageLayout::UNDEFINED
                })
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
            PipelineStageFlags::PIXEL_SHADER,
            &[self
                .glyph_array
                .barrier()
                .old_layout(ImageLayout::TRANSFER_DST)
                .new_layout(ImageLayout::SHADER_READ)
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_READ)],
        );

        self.glyph_array_initialized = true;
    }

    fn lay_out_glyphs(
        &self,
        seq: &StyledGlyphSequence,
        h_align: TextHAlign,
        max_width: f32,
        max_height: f32,
    ) -> Vec<GlyphInstance> {
        let mut glyph_instances = Vec::with_capacity(seq.glyphs.len());
        let mut line_widths = Vec::with_capacity(32);

        let scale = rusttype::Scale::uniform(1.0);
        let style = &seq.style;
        let font = self.fonts[style.font_id() as usize].best_for_style(style.font_style());
        let space_id = font.glyph(' ').id();
        let v_metrics = font.v_metrics(scale);
        let line_height = v_metrics.ascent - v_metrics.descent;

        let mut prev_glyph = None::<GlyphUID>;
        // The origin (0,0) is at top-left corner
        let mut curr_offset = Vec2::new(0.0, line_height);

        let mut curr_word = Vec::<GlyphInstance>::with_capacity(32);
        let mut curr_word_glyph_widths = Vec::<f32>::with_capacity(32);
        let mut curr_word_width = 0.0;
        let mut curr_word_appended = false;

        for g in &seq.glyphs {
            let g_id = g.glyph_id();
            let glyph = font.glyph(g_id).scaled(scale);
            let h_metrics = glyph.h_metrics();
            let kerning = prev_glyph
                .map(|prev| font.pair_kerning(scale, prev.glyph_id(), g_id))
                .unwrap_or(0.0);

            curr_offset.x += kerning;

            if g_id != space_id {
                let glyph_width = if curr_word.len() >= 1 { kerning } else { 0.0 } + h_metrics.advance_width;
                curr_word.push(GlyphInstance {
                    glyph_index: self.char_locations[g].index,
                    color: style.color(),
                    offset: curr_offset,
                });
                curr_word_glyph_widths.push(glyph_width);
                curr_word_width += glyph_width;
                curr_word_appended = false;
            } else if !curr_word.is_empty() {
                let start_x = curr_word[0].offset.x;

                if start_x + curr_word_width >= max_width {
                    // Width of the word exceeds the maximum line width => break it into two parts.

                    if curr_word_width >= max_width {
                        if curr_offset.x > 0.0 {
                            line_widths.push(curr_offset.x - curr_word_width - kerning); // FIXME: -kerning ?
                            curr_offset.x = 0.0;
                            curr_offset.y += v_metrics.line_gap + line_height;
                        }
                    } else {
                        curr_offset.x = start_x;
                    }

                    for (i, (glyph_width, mut instance)) in curr_word_glyph_widths
                        .drain(..)
                        .zip(curr_word.drain(..))
                        .enumerate()
                    {
                        if i > 0 && (curr_offset.x + glyph_width) > max_width {
                            line_widths.push(curr_offset.x); // FIXME: -kerning ?
                            curr_offset.x = 0.0;
                            curr_offset.y += v_metrics.line_gap + line_height;
                        }

                        instance.offset = curr_offset;
                        glyph_instances.push(instance);

                        curr_offset.x += glyph_width;
                    }
                } else {
                    glyph_instances.extend(curr_word.drain(..));
                }
                curr_word_appended = true;
            }

            curr_offset.x += h_metrics.advance_width;
            prev_glyph = Some(*g);
        }

        if !curr_word_appended {
            glyph_instances.extend(curr_word.drain(..));
        }

        line_widths.push(curr_offset.x);

        if !glyph_instances.is_empty() {
            let mut curr_y = f32::NAN;
            let mut curr_line = 0;
            let mut curr_diff = 0.0;

            for inst in &mut glyph_instances {
                if curr_y != inst.offset.y {
                    match h_align {
                        TextHAlign::LEFT => { /* The text is already in this state */ }
                        TextHAlign::CENTER => {
                            curr_diff = (max_width - line_widths[curr_line]) / 2.0;
                        }
                        TextHAlign::RIGHT => {
                            curr_diff = max_width - line_widths[curr_line];
                        }
                    }

                    curr_line += 1;
                    curr_y = inst.offset.y;
                }

                inst.offset.x += curr_diff;
            }
        }

        glyph_instances
    }

    // fn allocate_text_glyphs_for_entity(&mut self, entity: Entity, ) {
    //
    // }

    fn alloc_text_glyphs_on_entity(
        &mut self,
        entity: Entity,
        texts: &mut ComponentStorageMut<component::SimpleText>,
        meshes: &mut ComponentStorageMut<component::VertexMesh>,
    ) {
        let simple_text = texts.get(entity).unwrap();

        let style = *simple_text.string().style();
        let glyphs = self.allocate_glyphs(
            style.font_id(),
            style.font_style(),
            simple_text.string().data().chars(),
        );

        let seq = StyledGlyphSequence { glyphs, style };

        let instances = self.lay_out_glyphs(
            &seq,
            simple_text.h_align(),
            simple_text.max_width(),
            simple_text.max_height(),
        );
        self.allocated_glyphs.insert(entity, seq);

        let mesh = self
            .device
            .create_instanced_vertex_mesh::<(), GlyphInstance>(&[], &instances, None)
            .unwrap();
        meshes.set(entity, component::VertexMesh::new(&mesh.raw()));
    }

    fn free_text_glyphs_from_entity(&mut self, entity: Entity) {
        let seq = self.allocated_glyphs.remove(&entity).unwrap();
        self.free_glyphs(&seq.glyphs);
    }

    pub fn update(&mut self, scene: &mut Scene) {
        let mut simple_texts = scene.storage_write::<component::SimpleText>();
        let mut vertex_meshes = scene.storage_write::<component::VertexMesh>();
        let events = simple_texts.events();

        for e in events {
            match e {
                Event::Created(e) => {
                    self.alloc_text_glyphs_on_entity(e, &mut simple_texts, &mut vertex_meshes);
                }
                Event::Modified(e) => {
                    self.free_text_glyphs_from_entity(e);
                    self.alloc_text_glyphs_on_entity(e, &mut simple_texts, &mut vertex_meshes);
                }
                Event::Removed(e) => {
                    self.free_text_glyphs_from_entity(e);
                }
            }
        }
    }

    pub fn pre_render(&mut self, cl: &mut CmdList, proj_view: Mat4) {
        let uniform_data = UniformData {
            px_range: MSDF_PX_RANGE,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            proj_view,
        };
        self.staging_uniform_buffer.write(0, &[uniform_data]);

        cl.copy_raw_host_buffer_to_device(&self.staging_uniform_buffer.raw(), 0, &self.uniform_buffer, 0, 1);

        cl.barrier_buffer(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::VERTEX_SHADER,
            &[self
                .uniform_buffer
                .barrier()
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::UNIFORM_READ)],
        );

        self.load_glyphs(cl);
    }

    pub fn render(
        &mut self,
        scene: &mut Scene,
        vertex_meshes: &HashMap<Entity, Arc<RawVertexMesh>>,
        cl: &mut CmdList,
    ) {
        cl.bind_pipeline(&self.pipeline);
        cl.bind_graphics_input(self.pipeline.signature(), 0, self.descriptor, &[]);

        let texts = scene.storage_read::<component::SimpleText>();
        let transforms = scene.storage_read::<GlobalTransform>();

        for e in texts.entries().intersection(&transforms).iter() {
            let mesh = vertex_meshes.get(&e);
            if mesh.is_none() {
                continue;
            }
            let mesh = mesh.unwrap();

            if mesh.instance_count == 0 {
                continue;
            }

            let transform = transforms.get(e).unwrap();
            let text = &self.allocated_glyphs[&e];
            let n_glyphs = text.glyphs.len();

            let consts = PushConstants {
                transform: transform.matrix_f32(),
            };
            cl.push_constants(self.pipeline.signature(), &consts);

            cl.bind_vertex_buffers(0, &mesh.bindings());
            cl.draw_instanced(4, 0, 0, n_glyphs as u32);
        }

        // self.allocated_glyphs

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
