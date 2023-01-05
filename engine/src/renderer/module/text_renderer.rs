use std::any::Any;
use std::collections::hash_map;
use std::mem;
use std::sync::Arc;

use entity_data::{Archetype, EntityId};
use fixedbitset::FixedBitSet;
use nalgebra_glm::{Mat4, U8Vec4, Vec2};
use parking_lot::Mutex;
use rayon::prelude::*;
use rusttype::{Font, GlyphId};
use unicode_normalization::UnicodeNormalization;

use core::utils::unsafe_slice::UnsafeSlice;
use core::utils::HashMap;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::sampler::SamplerClamp;
use vk_wrapper::shader::VInputRate;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferUsageFlags, CmdList, Device, DeviceBuffer, Format, HostBuffer, Image,
    ImageLayout, ImageUsageFlags, PipelineStageFlags, PrimitiveTopology, Queue, Sampler, SamplerFilter,
    SamplerMipmap,
};

use crate::ecs::component;
use crate::ecs::component::simple_text::{FontStyle, TextHAlign, TextStyle};
use crate::ecs::component::SimpleText;
use crate::renderer::module::RendererModule;
use crate::renderer::vertex_mesh::VertexMeshCreate;
use crate::renderer::{Internals, SceneObject};
use crate::{HashSet, Renderer};

const GLYPH_SIZE: u32 = 64;
const GLYPH_BYTE_SIZE: usize = (GLYPH_SIZE * GLYPH_SIZE * 4) as usize; // RGBA8
const PREFERRED_MAX_GLYPHS: u32 = 1024;
const MSDF_PX_RANGE: u32 = 4;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct GlyphUID(u64);

impl GlyphUID {
    fn new(glyph_id: GlyphId, font_id: u16, style: FontStyle) -> Self {
        assert!(mem::size_of_val(&glyph_id.0) <= mem::size_of::<u32>());
        Self(((glyph_id.0 as u64) << 24) | ((font_id as u64) << 8) | (style as u64))
    }

    fn glyph_id(&self) -> GlyphId {
        GlyphId((self.0 >> 24) as u16)
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

pub struct WordSplitter<'a> {
    glyphs: &'a [GlyphUID],
    space_id: GlyphId,
}

impl<'a> Iterator for WordSplitter<'a> {
    type Item = &'a [GlyphUID];

    fn next(&mut self) -> Option<Self::Item> {
        if self.glyphs.is_empty() {
            return None;
        }

        // let mut len = 0;
        // let is_word = self.glyphs[0].glyph_id() != self.space_id;
        //
        // for g in self.glyphs {
        //     let g_id = g.glyph_id();
        //     if (is_word && g_id == self.space_id) || (!is_word && g_id != self.space_id) {
        //         break;
        //     }
        //     len += 1;
        // }

        let mut len = 0;
        let mut word_found = false;

        for g in self.glyphs {
            if g.glyph_id() != self.space_id {
                word_found = true;
            } else if word_found {
                break;
            }
            len += 1;
        }

        let slice = &self.glyphs[..len];
        self.glyphs = &self.glyphs[len..];
        Some(slice)
    }
}

pub struct GlyphAllocator {
    fonts: Vec<FontSet>,
    atlas_overflow_glyph: GlyphUID,

    char_locations: HashMap<GlyphUID, GlyphLocation>,
    char_locations_rev: HashMap<u32, GlyphUID>,
    free_locations: FixedBitSet,
    chars_to_load: HashSet<GlyphUID>,
}

impl GlyphAllocator {
    pub fn alloc(
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

            // Check if glyph is already allocated
            if self.char_locations.contains_key(&g_uid) {
                glyphs.push(g_uid);
                continue;
            }

            // Check if glyph cannot be allocated due to insufficient space in glyph array
            if self.free_locations.is_empty() {
                glyphs.push(self.atlas_overflow_glyph);
                continue;
            }

            // Check if glyph outline is not empty
            let glyph = font.glyph(g_uid.glyph_id());
            let index = if glyph
                .scaled(rusttype::Scale::uniform(1.0))
                .exact_bounding_box()
                .is_none()
            {
                u32::MAX
            } else {
                let index = self.free_locations.ones().next().unwrap();
                self.free_locations.toggle(index);
                index as u32
            };

            // Allocate a new glyph
            if index != u32::MAX {
                self.char_locations_rev.insert(index, g_uid);
                self.chars_to_load.insert(g_uid);
            }
            self.char_locations
                .insert(g_uid, GlyphLocation { index, ref_count: 1 });
            glyphs.push(g_uid);
        }

        glyphs
    }
    fn free(&mut self, chars: &[GlyphUID]) {
        for c in chars {
            if let hash_map::Entry::Occupied(mut v) = self.char_locations.entry(*c) {
                let e = v.get_mut();
                if e.ref_count >= 2 {
                    e.ref_count -= 1;
                } else {
                    self.free_locations.insert(e.index as usize);
                    self.char_locations_rev.remove(&e.index);
                    self.chars_to_load.remove(c);
                    v.remove();
                }
            } else {
                unreachable!()
            }
        }
    }
}

pub struct TextRenderer {
    device: Arc<Device>,
    mat_pipeline: u32,
    staging_cl: Arc<Mutex<CmdList>>,

    glyph_sampler: Arc<Sampler>,
    glyph_array: Arc<Image>,
    glyph_array_initialized: bool,
    uniform_buffer: DeviceBuffer,
    staging_buffer: HostBuffer<u8>,
    staging_uniform_buffer: HostBuffer<UniformData>,

    allocator: GlyphAllocator,

    allocated_sequences: HashMap<EntityId, StyledGlyphSequence>,
    sequences_to_destroy: Vec<StyledGlyphSequence>,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct GlyphInstance {
    glyph_index: u32,
    glyph_size: Vec2,
    color: U8Vec4,
    offset: Vec2,
    scale: f32,
}
attributes_impl!(GlyphInstance, glyph_index, glyph_size, color, offset, scale);

#[repr(C)]
struct UniformData {
    px_range: f32,
}

#[derive(Default)]
pub struct ObjectUniformInfo {
    model: Mat4,
}

uniform_struct_impl!(ObjectUniformInfo, model);

#[derive(Archetype)]
pub struct TextObject {
    relation: component::internal::Relation,
    global_transform: component::internal::GlobalTransform,
    transform: component::Transform,
    renderer: component::MeshRenderConfig,
    mesh: component::VertexMesh,
    text: SimpleText,
}

impl TextObject {
    pub fn new(transform: component::Transform, text: SimpleText) -> Self {
        Self {
            relation: Default::default(),
            global_transform: Default::default(),
            transform,
            renderer: Default::default(),
            mesh: Default::default(),
            text,
        }
    }
}

impl SceneObject for TextObject {}

impl TextRenderer {
    pub(crate) fn new(renderer: &mut Renderer) -> Self {
        let device = Arc::clone(renderer.device());
        let vertex_shader = device
            .create_vertex_shader(
                include_bytes!("../../../shaders/build/text_char.vert.spv"),
                &[
                    ("inGlyphIndex", Format::R32_UINT, VInputRate::INSTANCE),
                    ("inGlyphSize", Format::RG32_FLOAT, VInputRate::INSTANCE),
                    ("inColor", Format::RGBA8_UNORM, VInputRate::INSTANCE),
                    ("inOffset", Format::RG32_FLOAT, VInputRate::INSTANCE),
                    ("inScale", Format::R32_FLOAT, VInputRate::INSTANCE),
                ],
            )
            .unwrap();
        let pixel_shader = device
            .create_pixel_shader(include_bytes!("../../../shaders/build/text_char.frag.spv"))
            .unwrap();
        // let signature = device
        //     .create_pipeline_signature(&[vertex_shader, pixel_shader], &[])
        //     .unwrap();
        // let pipeline = device
        //     .create_graphics_pipeline(
        //         render_pass,
        //         subpass_index,
        //         PrimitiveTopology::TRIANGLE_STRIP,
        //         PipelineDepthStencil::default().depth_test(true).depth_write(true),
        //         PipelineRasterization::new().cull_back_faces(true),
        //         &[(0, AttachmentColorBlend::default().enabled(true))],
        //         &signature,
        //     )
        //     .unwrap();

        let mat_pipeline = renderer.register_material_pipeline::<ObjectUniformInfo>(
            &[vertex_shader, pixel_shader],
            PrimitiveTopology::TRIANGLE_STRIP,
            true,
        );
        let pipe = renderer.get_material_pipeline_mut(mat_pipeline).unwrap();

        let glyph_array = device
            .create_image_2d_array_named(
                Format::RGBA8_UNORM,
                1,
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
                SamplerClamp::CLAMP_TO_EDGE,
                1.0,
            )
            .unwrap();

        let (pool, descriptor) = pipe.create_custom_frame_uniform_descriptor().unwrap();

        unsafe {
            device.update_descriptor_set(
                *descriptor,
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

        let staging_cl = device
            .get_queue(Queue::TYPE_GRAPHICS)
            .create_primary_cmd_list("text-staging-cl")
            .unwrap();

        let fallback_font =
            FontSet::from_bytes(include_bytes!("../../../fallback-font.ttf").to_vec(), None).unwrap();

        let mut char_locations = HashMap::with_capacity(size3d.2 as usize);
        let mut char_locations_rev = HashMap::with_capacity(size3d.2 as usize);
        let mut free_locations: FixedBitSet = (0..char_locations.capacity()).collect();
        let mut chars_to_load = HashSet::with_capacity(size3d.2 as usize);
        let fonts = vec![fallback_font];

        let atlas_overflow_glyph = {
            let g_uid = GlyphUID::new(GlyphId(0), 0, FontStyle::Normal);
            char_locations.insert(
                g_uid,
                GlyphLocation {
                    index: 0,
                    ref_count: 1,
                },
            );
            char_locations_rev.insert(0, g_uid);
            free_locations.set(0, false);
            chars_to_load.insert(g_uid);
            g_uid
        };

        Self {
            device,
            mat_pipeline,
            glyph_array,
            glyph_array_initialized: false,
            uniform_buffer,
            staging_buffer,
            allocator: GlyphAllocator {
                fonts,
                atlas_overflow_glyph,
                char_locations,
                char_locations_rev,
                free_locations,
                chars_to_load,
            },
            allocated_sequences: Default::default(),
            staging_uniform_buffer,
            glyph_sampler,
            staging_cl,
            sequences_to_destroy: vec![],
        }
    }

    fn glyph_array_capacity(&mut self) -> u32 {
        self.glyph_array.size().2
    }

    pub fn register_font(&mut self, font_set: FontSet) -> u16 {
        self.allocator.fonts.push(font_set);
        (self.allocator.fonts.len() - 1) as u16
    }

    pub fn load_glyphs(&mut self, cl: &mut CmdList) {
        let staging_slice = UnsafeSlice::new(self.staging_buffer.as_mut_slice());

        self.allocator.chars_to_load.par_iter().for_each(|g| {
            let font_set = &self.allocator.fonts[g.font_id() as usize];
            let font = font_set.best_for_style(g.style());
            let glyph = font.glyph(g.glyph_id());

            let location = &self.allocator.char_locations[g];
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

        for g in self.allocator.chars_to_load.drain() {
            let location = &self.allocator.char_locations[&g];
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

    fn layout_glyphs(
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
        let font = self.allocator.fonts[style.font_id() as usize].best_for_style(style.font_style());
        let space_id = font.glyph(' ').id();
        let v_metrics = font.v_metrics(scale);

        // The origin (0,0) is at top-left corner
        let mut curr_offset = Vec2::new(0.0, v_metrics.ascent);
        let mut prev_glyph = None::<GlyphUID>;

        let mut curr_word = Vec::<GlyphInstance>::with_capacity(32);
        let mut curr_word_glyph_widths = Vec::<f32>::with_capacity(32);

        // A segment is a word or blank space(s)
        let segments = WordSplitter {
            glyphs: &seq.glyphs,
            space_id,
        };

        for segment in segments {
            let mut curr_word_width = 0.0;
            curr_word.clear();
            curr_word_glyph_widths.clear();

            // Layout single word
            for g in segment {
                let g_id = g.glyph_id();
                let glyph_index = self.allocator.char_locations[g].index;
                let glyph = font.glyph(g_id).scaled(scale);
                let h_metrics = glyph.h_metrics();
                let kerning = prev_glyph
                    .map(|prev| font.pair_kerning(scale, prev.glyph_id(), g_id))
                    .unwrap_or(0.0);

                let glyph_width = kerning + h_metrics.advance_width;
                let glyph_size = if let Some(size) = msdfgen::glyph_size(&glyph, GLYPH_SIZE, MSDF_PX_RANGE) {
                    Vec2::new(size.0, size.1)
                } else {
                    Vec2::default()
                };

                curr_offset.x += kerning;

                curr_word.push(GlyphInstance {
                    glyph_index,
                    glyph_size,
                    color: style.color(),
                    offset: curr_offset,
                    scale: 1.0,
                });
                curr_word_glyph_widths.push(glyph_width);
                curr_word_width += glyph_width;

                curr_offset.x += h_metrics.advance_width;
                prev_glyph = Some(*g);
            }

            if curr_word.is_empty() {
                continue;
            }
            let start_x = curr_word[0].offset.x;

            // Put the word inside the global layout
            if start_x + curr_word_width >= max_width {
                // Width of the word exceeds the maximum line width => break it into two parts.

                if curr_offset.x > 0.0 {
                    line_widths.push(curr_offset.x - curr_word_width);
                    curr_offset.x = 0.0;
                    curr_offset.y += v_metrics.line_gap + (v_metrics.ascent - v_metrics.descent);
                }

                for (i, (glyph_width, mut instance)) in curr_word_glyph_widths
                    .drain(..)
                    .zip(curr_word.drain(..))
                    .enumerate()
                {
                    if self
                        .allocator
                        .char_locations_rev
                        .get(&instance.glyph_index)
                        .map_or(true, |v| v.glyph_id() == space_id)
                    {
                        continue;
                    }
                    if i > 0 && (curr_offset.x + glyph_width) > max_width {
                        line_widths.push(curr_offset.x);
                        curr_offset.x = 0.0;
                        curr_offset.y += v_metrics.line_gap + (v_metrics.ascent - v_metrics.descent);
                    }

                    instance.offset = curr_offset;
                    glyph_instances.push(instance);

                    curr_offset.x += glyph_width;
                }
            } else {
                glyph_instances.extend(curr_word.drain(..));
            }
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

            glyph_instances = glyph_instances
                .into_iter()
                .filter(|v| v.glyph_index != u32::MAX)
                .collect();

            for inst in &mut glyph_instances {
                let g_uid = self.allocator.char_locations_rev[&inst.glyph_index];
                let glyph = font.glyph(g_uid.glyph_id());

                if let Some(trans) = msdfgen::glyph_reverse_transform(glyph, GLYPH_SIZE, MSDF_PX_RANGE) {
                    inst.scale *= trans.scale;
                    inst.offset.x += trans.offset_x;
                    inst.offset.y -= trans.offset_y;
                }
            }
        }

        // TODO: use max_height too if necessary

        glyph_instances
    }
}

impl RendererModule for TextRenderer {
    fn on_object_remove(&mut self, id: &EntityId, _scene: Internals) {
        if let Some(seq) = self.allocated_sequences.remove(id) {
            self.sequences_to_destroy.push(seq);
        }
    }

    fn on_update(&mut self, internals: Internals) -> Option<Arc<Mutex<CmdList>>> {
        let mut dirty_texts = internals.dirty_comps.take_changes::<SimpleText>();

        for seq in self.sequences_to_destroy.drain(..) {
            self.allocator.free(&seq.glyphs);
        }

        for entity in &dirty_texts {
            if let Some(seq) = self.allocated_sequences.remove(entity) {
                self.allocator.free(&seq.glyphs);
            }

            let mut entry = internals.storage.entry_mut(entity).unwrap();
            let simple_text = entry.get::<SimpleText>().unwrap();
            let style = *simple_text.string().style();

            let glyphs = self.allocator.alloc(
                style.font_id(),
                style.font_style(),
                simple_text.string().data().chars(),
            );

            let seq = StyledGlyphSequence { glyphs, style };

            let instances = self.layout_glyphs(
                &seq,
                simple_text.h_align(),
                simple_text.max_width(),
                simple_text.max_height(),
            );
            self.allocated_sequences.insert(*entity, seq);

            let mesh = self
                .device
                .create_instanced_vertex_mesh::<(), GlyphInstance>(&[], &instances, None)
                .unwrap();

            *entry.get_mut::<component::VertexMesh>().unwrap() = component::VertexMesh::new(&mesh.raw());
            *entry.get_mut::<component::MeshRenderConfig>().unwrap() =
                component::MeshRenderConfig::new(self.mat_pipeline, true).with_fake_vertex_count(4);
        }

        let staging_cl = Arc::clone(&self.staging_cl);
        let mut cl = staging_cl.lock();
        cl.begin(true).unwrap();

        // Copy uniforms to the GPU
        let uniform_data = UniformData {
            px_range: MSDF_PX_RANGE as f32,
        };
        self.staging_uniform_buffer.write(0, &[uniform_data]);
        cl.copy_raw_host_buffer_to_device(&self.staging_uniform_buffer.raw(), 0, &self.uniform_buffer, 0, 1);

        // Copy new glyphs to the GPU
        self.load_glyphs(&mut cl);

        cl.end().unwrap();
        drop(cl);

        Some(staging_cl)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
