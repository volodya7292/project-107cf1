use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::render_config::RenderLayer;
use crate::ecs::component::simple_text::{FontStyle, StyledString, TextHAlign, TextStyle};
use crate::ecs::component::{MeshRenderConfigC, SimpleTextC, TransformC, UniformDataC, VertexMeshC};
use crate::module::main_renderer::gpu_executor::{GPUJob, GPUJobDeviceExt};
use crate::module::main_renderer::vertex_mesh::{VAttributes, VertexMeshCreate};
use crate::module::main_renderer::{MainRenderer, MaterialPipelineId};
use crate::module::scene::change_manager::{ChangeType, ComponentChangesHandle};
use crate::module::scene::Scene;
use crate::module::scene::SceneObject;
use crate::module::EngineModule;
use crate::{attributes_impl, EngineContext};
use common::glm::{U8Vec4, Vec2};
use common::lrc::{Lrc, LrcExt, LrcExtSized};
use common::rayon::prelude::*;
use common::scene::relation::Relation;
use common::types::HashMap;
use common::types::HashSet;
use common::unsafe_slice::UnsafeSlice;
use entity_data::{Archetype, EntityId};
use fixedbitset::FixedBitSet;
use rusttype::{Font, GlyphId};
use std::collections::hash_map;
use std::mem;
use std::rc::Rc;
use std::sync::Arc;
use unicode_normalization::UnicodeNormalization;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::sampler::SamplerClamp;
use vk_wrapper::shader::VInputRate;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferUsageFlags, CmdList, Device, DeviceBuffer, Format, HostBuffer, Image,
    ImageLayout, ImageUsageFlags, PipelineStageFlags, PrimitiveTopology, QueueType, Sampler, SamplerFilter,
    SamplerMipmap, Shader,
};

const GLYPH_SIZE: u32 = 64;
const GLYPH_BYTE_SIZE: usize = (GLYPH_SIZE * GLYPH_SIZE * 4) as usize; // RGBA8
const PREFERRED_MAX_GLYPHS: u32 = 1024;
const MSDF_PX_RANGE: u32 = 4;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
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

        let mut len = 0;
        let mut word_found = false;

        for g in self.glyphs {
            if g.glyph_id() != self.space_id {
                // `g` is a word-character
                word_found = true;
            } else {
                // `g` is a space
                if word_found {
                    break;
                }
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
    fn chart_to_glyph_uid(&self, ch: char, font_id: u16, style: FontStyle) -> GlyphUID {
        let font = self.fonts[font_id as usize].best_for_style(style).clone();

        let mut g_uid = GlyphUID::new(font.glyph(ch).id(), font_id, style);

        if g_uid.glyph_id().0 == 0 {
            // Try fallback font
            let fallback_font = self.fonts[0].best_for_style(style);
            g_uid = GlyphUID::new(fallback_font.glyph(ch).id(), 0, style);

            if g_uid.glyph_id().0 == 0 {
                g_uid = self.atlas_overflow_glyph;
            }
        }

        g_uid
    }

    fn alloc(&mut self, font_id: u16, style: FontStyle, chars: impl Iterator<Item = char>) -> Vec<GlyphUID> {
        let chars = chars.nfc();
        let size_hint = chars.size_hint();
        let mut glyphs = Vec::with_capacity(size_hint.1.unwrap_or(size_hint.0));
        let font = self.fonts[font_id as usize].best_for_style(style);

        for c in chars {
            let g_uid = self.chart_to_glyph_uid(c, font_id, style);

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

    fn alloc_for(&mut self, string: &StyledString) -> StyledGlyphSequence {
        let style = string.style();
        let glyphs = self.alloc(style.font_id(), style.font_style(), string.data().chars());

        StyledGlyphSequence {
            glyphs,
            style: *style,
        }
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

fn calculate_string_width(string: &str, font: &Font) -> f32 {
    let scale = rusttype::Scale::uniform(1.0);
    let mut prev_glyph = None::<GlyphId>;
    let mut curr_word_width = 0.0;

    // Layout single word
    for ch in string.chars() {
        let glyph = font.glyph(ch).scaled(scale);
        let h_metrics = glyph.h_metrics();
        let g_id = glyph.id();

        let kerning = prev_glyph
            .map(|prev| font.pair_kerning(scale, prev, g_id))
            .unwrap_or(0.0);

        let glyph_width = kerning + h_metrics.advance_width;

        curr_word_width += glyph_width;
        prev_glyph = Some(g_id);
    }
    curr_word_width
}

pub type BlockSize = Vec2;

#[derive(Debug)]
struct PositioningInfo {
    uid: GlyphUID,
    offset: Vec2,
    scale: Vec2,
}

/// Returns glyph instances and resulting block size.
fn layout_glyphs(
    allocator: &GlyphAllocator,
    seq: &StyledString,
    h_align: TextHAlign,
    long_word_breaking: bool,
    max_width: f32,
    max_height: f32,
    normalize_transforms: bool,
) -> (Vec<PositioningInfo>, BlockSize) {
    let mut glyph_positions = Vec::with_capacity(seq.data().chars().count());
    let mut line_widths = Vec::with_capacity(32);
    let mut final_size = Vec2::new(0.0, 0.0);

    let scale = rusttype::Scale::uniform(1.0);
    let style = &seq.style();
    let font = allocator.fonts[style.font_id() as usize].best_for_style(style.font_style());
    let space_id = font.glyph(' ').id();
    let v_metrics = font.v_metrics(scale);
    let line_height = v_metrics.line_gap + (v_metrics.ascent - v_metrics.descent);

    // The origin (0,0) is at top-left corner
    let mut curr_offset = Vec2::new(0.0, v_metrics.ascent);
    let mut prev_glyph = None::<GlyphUID>;

    let mut curr_word = Vec::<PositioningInfo>::with_capacity(32);
    let mut curr_word_glyph_widths = Vec::<f32>::with_capacity(32);

    let glyph_uids: Vec<_> = seq
        .data()
        .chars()
        .map(|ch| allocator.chart_to_glyph_uid(ch, style.font_id(), style.font_style()))
        .collect();

    // A segment is a word or blank space(s)
    let segments = WordSplitter {
        glyphs: &glyph_uids,
        space_id,
    };

    for segment in segments {
        let curr_word_offset_x = curr_offset.x;
        let mut curr_word_width = 0.0;
        curr_word.clear();
        curr_word_glyph_widths.clear();

        // Layout single word
        for g in segment {
            let g_id = g.glyph_id();
            let glyph = font.glyph(g_id).scaled(scale);
            let h_metrics = glyph.h_metrics();
            let kerning = prev_glyph
                .map(|prev| font.pair_kerning(scale, prev.glyph_id(), g_id))
                .unwrap_or(0.0);

            let glyph_width = kerning + h_metrics.advance_width;

            curr_word.push(PositioningInfo {
                uid: *g,
                offset: curr_offset,
                scale: Vec2::from_element(1.0),
            });

            curr_word_glyph_widths.push(glyph_width);
            curr_word_width += glyph_width;

            prev_glyph = Some(*g);
        }
        if curr_word.is_empty() {
            continue;
        }

        let word_start_x = curr_word[0].offset.x;

        // Put the word inside the global layout

        if (word_start_x + curr_word_width >= max_width) && curr_word_offset_x > 0.0 {
            // Put the word onto a new line
            // TODO: remove space before the word (or after)
            line_widths.push(curr_offset.x - curr_word_width);
            curr_offset.x = 0.0;
            curr_offset.y += line_height;
        }

        for (i, (glyph_width, mut pos_info)) in curr_word_glyph_widths
            .drain(..)
            .zip(curr_word.drain(..))
            .enumerate()
        {
            if long_word_breaking && i > 0 && (curr_offset.x + glyph_width) > max_width {
                // The width of the word exceeds the maximum line width => break it into two parts.
                line_widths.push(curr_offset.x);
                curr_offset.x = 0.0;
                curr_offset.y += line_height;
            } else {
                final_size.x = final_size.x.max(curr_offset.x + glyph_width);
            }

            pos_info.offset = curr_offset;
            glyph_positions.push(pos_info);

            curr_offset.x += glyph_width;
        }
    }

    line_widths.push(curr_offset.x);

    if glyph_positions.is_empty() {
        return (glyph_positions, final_size);
    }

    final_size.y = curr_offset.y - v_metrics.descent;

    let mut curr_y = f32::NAN;
    let mut curr_line = 0;
    let mut curr_diff = 0.0;

    for inst in &mut glyph_positions {
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

    let final_size_inv = Vec2::from_element(1.0).component_div(&final_size);

    for inst in &mut glyph_positions {
        let g_uid = inst.uid;
        let glyph = font.glyph(g_uid.glyph_id());

        if let Some(trans) = msdfgen::glyph_reverse_transform(glyph, GLYPH_SIZE, MSDF_PX_RANGE) {
            inst.scale *= trans.scale;
            inst.offset.x += trans.offset_x;
            inst.offset.y -= trans.offset_y;
        }

        if normalize_transforms {
            inst.offset.component_mul_assign(&final_size_inv);
            inst.scale.component_mul_assign(&final_size_inv);
        }
    }

    // TODO: use max_height too if necessary

    (glyph_positions, final_size)
}

pub struct TextRenderer {
    device: Arc<Device>,
    registered_mat_pipelines: Vec<MaterialPipelineId>,
    staging_job: Lrc<GPUJob>,

    glyph_sampler: Arc<Sampler>,
    glyph_array: Arc<Image>,
    glyph_array_initialized: bool,
    uniform_buffer: DeviceBuffer,
    staging_buffer: HostBuffer<u8>,
    staging_uniform_buffer: HostBuffer<FrameUniformData>,

    allocator: GlyphAllocator,

    allocated_sequences: HashMap<EntityId, StyledGlyphSequence>,
    simple_text_changes: ComponentChangesHandle,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct GlyphInstance {
    glyph_index: u32,
    glyph_size: Vec2,
    color: U8Vec4,
    offset: Vec2,
    scale: Vec2,
}
attributes_impl!(GlyphInstance, glyph_index, glyph_size, color, offset, scale);

#[derive(Copy, Clone)]
#[repr(C)]
struct FrameUniformData {
    px_range: f32,
}

#[derive(Archetype)]
pub struct RawTextObject {
    relation: Relation,
    global_transform: GlobalTransformC,
    transform: TransformC,
    uniforms: UniformDataC,
    renderer: MeshRenderConfigC,
    mesh: VertexMeshC,
    text: SimpleTextC,
}

impl RawTextObject {
    pub fn new(transform: TransformC, text: SimpleTextC) -> Self {
        Self {
            relation: Default::default(),
            global_transform: Default::default(),
            transform,
            uniforms: Default::default(),
            renderer: Default::default(),
            mesh: Default::default(),
            text,
        }
    }
}

impl SceneObject for RawTextObject {}

impl TextRenderer {
    pub fn new(ctx: &EngineContext) -> Self {
        let scene = ctx.module_mut::<Scene>();
        let simple_text_changes = scene
            .change_manager_mut()
            .register_component_flow::<SimpleTextC>();

        let mut renderer = ctx.module_mut::<MainRenderer>();
        let device = Arc::clone(renderer.device());

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
                mem::size_of::<FrameUniformData>() as u64,
                1,
                "text_renderer_uniform",
            )
            .unwrap();
        let staging_uniform_buffer = device
            .create_host_buffer_named::<FrameUniformData>(
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

        let staging_job = Lrc::wrap(device.create_job("text-staging", QueueType::Graphics).unwrap());

        let fallback_font =
            FontSet::from_bytes(include_bytes!("../../fallback-font.ttf").to_vec(), None).unwrap();

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

        {
            renderer.register_update_handler(Box::new(move |ctx| {
                let mut text_renderer = ctx.module_mut::<Self>();
                Self::on_render_update(&mut text_renderer, ctx)
            }));
        }

        Self {
            device,
            registered_mat_pipelines: vec![],
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
            staging_job,
            simple_text_changes,
        }
    }

    pub fn register_text_pipeline(&mut self, renderer: &mut MainRenderer, pixel_shader: Arc<Shader>) -> u32 {
        let vertex_shader = renderer
            .device()
            .create_vertex_shader(
                include_bytes!("../../shaders/build/text_char.vert.spv"),
                &[
                    ("inGlyphIndex", Format::R32_UINT, VInputRate::INSTANCE),
                    ("inGlyphSize", Format::RG32_FLOAT, VInputRate::INSTANCE),
                    ("inColor", Format::RGBA8_UNORM, VInputRate::INSTANCE),
                    ("inOffset", Format::RG32_FLOAT, VInputRate::INSTANCE),
                    ("inScale", Format::RG32_FLOAT, VInputRate::INSTANCE),
                ],
                "text_char.vert",
            )
            .unwrap();

        let mat_pipe_id = renderer.register_material_pipeline(
            &[vertex_shader, pixel_shader],
            PrimitiveTopology::TRIANGLE_STRIP,
            true,
        );

        let pipe = renderer.get_material_pipeline(mat_pipe_id).unwrap();
        let per_frame_pool = &pipe.per_frame_desc_pool;

        unsafe {
            renderer.device().update_descriptor_set(
                pipe.per_frame_desc,
                &[
                    per_frame_pool.create_binding(0, 0, BindingRes::Buffer(self.uniform_buffer.handle())),
                    per_frame_pool.create_binding(
                        1,
                        0,
                        BindingRes::Image(
                            Arc::clone(&self.glyph_array),
                            Some(Arc::clone(&self.glyph_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                ],
            );
        }

        self.registered_mat_pipelines.push(mat_pipe_id);
        mat_pipe_id
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

    pub fn calculate_minimum_text_size(&self, seq: &StyledString) -> BlockSize {
        let (_, size) = layout_glyphs(&self.allocator, seq, TextHAlign::LEFT, false, 0.0, 0.0, false);
        size * seq.style().font_size()
    }

    fn on_render_update(&mut self, ctx: &EngineContext) -> Option<Lrc<GPUJob>> {
        let mut scene = ctx.module_mut::<Scene>();

        let unit_scale = rusttype::Scale::uniform(1.0);
        let changes = scene.change_manager_mut().take(self.simple_text_changes);

        for change in &changes {
            let entity = change.entity();

            if change.ty() == ChangeType::Removed {
                let seq = self.allocated_sequences.remove(entity).unwrap();
                self.allocator.free(&seq.glyphs);
                continue;
            }

            let mut entry = scene.entry(entity);
            let simple_text = entry.get::<SimpleTextC>();
            let stage = simple_text.render_type;
            let mat_pipeline = simple_text.mat_pipeline;
            let normalize_transforms = stage == RenderLayer::Overlay;

            let seq = self.allocator.alloc_for(&simple_text.text);

            let (positions, _) = layout_glyphs(
                &self.allocator,
                &simple_text.text,
                simple_text.h_align,
                simple_text.long_word_breaking,
                simple_text.max_width,
                simple_text.max_height,
                normalize_transforms,
            );
            let instances: Vec<_> = positions
                .into_iter()
                .filter_map(|info| {
                    let glyph_loc = self.allocator.char_locations.get(&info.uid).unwrap();
                    let font = self.allocator.fonts[info.uid.font_id() as usize]
                        .best_for_style(seq.style.font_style());
                    let space_id = font.glyph(' ').id();

                    if self
                        .allocator
                        .char_locations_rev
                        .get(&glyph_loc.index)
                        .map_or(true, |v| v.glyph_id() == space_id)
                    {
                        return None;
                    }

                    let glyph = font.glyph(info.uid.glyph_id()).scaled(unit_scale);
                    let glyph_size =
                        if let Some(size) = msdfgen::glyph_size(&glyph, GLYPH_SIZE, MSDF_PX_RANGE) {
                            Vec2::new(size.0, size.1)
                        } else {
                            Vec2::default()
                        };

                    Some(GlyphInstance {
                        glyph_index: glyph_loc.index,
                        glyph_size,
                        color: seq.style.color(),
                        offset: info.offset,
                        scale: info.scale,
                    })
                })
                .collect();
            self.allocated_sequences.insert(*entity, seq);

            let mesh = self
                .device
                .create_instanced_vertex_mesh::<(), GlyphInstance>(
                    VAttributes::WithoutData(4),
                    VAttributes::Slice(&instances),
                    None,
                )
                .unwrap();

            *entry.get_mut::<VertexMeshC>() = VertexMeshC::new(&mesh.raw());
            *entry.get_mut::<MeshRenderConfigC>() =
                MeshRenderConfigC::new(mat_pipeline, true).with_render_layer(stage)
        }

        let mut staging_job = self.staging_job.borrow_mut_owned();
        let cl = staging_job.get_cmd_list_for_recording();
        cl.begin(true).unwrap();

        // Copy uniforms to the GPU
        let uniform_data = FrameUniformData {
            px_range: MSDF_PX_RANGE as f32,
        };
        self.staging_uniform_buffer.write(0, &[uniform_data]);
        cl.copy_buffer(
            &self.staging_uniform_buffer,
            0,
            &self.uniform_buffer,
            0,
            self.staging_uniform_buffer.element_size(),
        );

        // Copy new glyphs to the GPU
        self.load_glyphs(cl);

        cl.end().unwrap();

        Some(Rc::clone(&self.staging_job))
    }
}

impl EngineModule for TextRenderer {}
