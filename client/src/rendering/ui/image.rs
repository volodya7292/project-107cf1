use crate::rendering::ui::UIContext;
use common::glm::Vec2;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::{RenderLayer, Resource};
use engine::ecs::component::ui::{Factor, RectUniformData, Sizing, UILayoutC, UILayoutCacheC};
use engine::ecs::component::{MeshRenderConfigC, SceneEventHandler, UniformDataC, VertexMeshC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, Scene};
use engine::module::ui::management::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::vkw::image::ImageParams;
use engine::vkw::pipeline::CullMode;
use engine::vkw::{Format, ImageUsageFlags, PrimitiveTopology};
use engine::EngineContext;
use entity_data::EntityId;
use smallvec::smallvec;

struct ImageImplContext {
    mat_pipe_id: MaterialPipelineId,
}

pub enum ImageSource {
    Data(image::RgbaImage),
}

impl ImageSource {
    pub fn size(&self) -> (u32, u32) {
        match self {
            ImageSource::Data(img) => (img.width(), img.height()),
        }
    }

    pub fn into_raw(self) -> Vec<u8> {
        match self {
            ImageSource::Data(img) => img.into_raw(),
        }
    }
}

#[derive(Copy, Clone)]
pub enum ImageAspect {
    Original,
    Cover,
}

pub struct ImageState {
    aspect: ImageAspect,
    original_aspect_ratio: f32,
    source: Option<ImageSource>,
}

impl UIState for ImageState {}

pub type UIImage = UIObject<ImageState>;

pub trait ImageImpl {
    fn register(ctx: &EngineContext) {
        let mut renderer = ctx.module_mut::<MainRenderer>();
        let scene = ctx.module_mut::<Scene>();

        let vertex = renderer
            .device()
            .create_vertex_shader(
                include_bytes!("../../../res/shaders/ui_image.vert.spv"),
                &[],
                "ui_image.vert",
            )
            .unwrap();
        let pixel = renderer
            .device()
            .create_pixel_shader(
                include_bytes!("../../../res/shaders/ui_image.frag.spv"),
                "ui_image.frag",
            )
            .unwrap();

        let mat_pipe_id = renderer.register_material_pipeline(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_STRIP,
            CullMode::BACK,
        );

        scene.add_resource(ImageImplContext { mat_pipe_id });
    }

    fn new(ui_ctx: &UIContext, aspect: ImageAspect) -> UIImage {
        let impl_ctx = ui_ctx.scene.resource::<ImageImplContext>();
        let renderer = ui_ctx.ctx.module::<MainRenderer>();
        let initial_image = Resource::image(
            renderer.device(),
            ImageParams::d2(Format::RGBA8_UNORM, ImageUsageFlags::SAMPLED, (1, 1)),
            vec![0_u8; 4],
        )
        .unwrap();

        UIImage::new_raw(
            UILayoutC::new()
                .with_width(Sizing::Grow(1.0))
                .with_height(Sizing::Grow(1.0)),
            ImageState {
                aspect,
                original_aspect_ratio: 1.0,
                source: None,
            },
        )
        .with_renderer(
            MeshRenderConfigC::new(impl_ctx.mat_pipe_id, true)
                .with_render_layer(RenderLayer::Overlay)
                .with_shader_resources(smallvec![initial_image]),
        )
        .with_mesh(VertexMeshC::without_data(4, 1))
        .with_scene_event_handler(
            SceneEventHandler::new()
                .with_on_update(on_update)
                .with_on_component_update::<UILayoutCacheC>(on_layout_cache_update),
        )
    }
}

impl ImageImpl for UIImage {}

pub trait ImageAccess {
    fn set_image(&mut self, source: ImageSource);
}

impl ImageAccess for EntityAccess<'_, UIImage> {
    fn set_image(&mut self, source: ImageSource) {
        self.state_mut().source = Some(source);
        self.request_update();
    }
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct ObjectUniformData {
    clip_rect: RectUniformData,
    img_offset: Vec2,
    img_scale: Vec2,
}

fn on_layout_cache_update(entity: &EntityId, scene: &mut Scene, _: &EngineContext) {
    let mut obj = scene.object::<UIImage>(entity);
    let cache = obj.get::<UILayoutCacheC>();
    let rect_data = *cache.calculated_clip_rect();
    let final_size = *cache.final_size();

    let ui_element_aspect_ratio = final_size.x / final_size.y;
    let shader_aspect_ratio = obj.state().original_aspect_ratio / ui_element_aspect_ratio;

    // Calculate offset and scale for 'cover' image fitness.
    let (img_offset, img_scale) = if shader_aspect_ratio > 1.0 {
        let width = shader_aspect_ratio;
        (Vec2::new(-0.5 * (width - 1.0), 0.0), Vec2::new(width, 1.0))
    } else {
        let height = 1.0 / shader_aspect_ratio;
        (Vec2::new(0.0, -0.5 * (height - 1.0)), Vec2::new(1.0, height))
    };

    let uniform_data = obj.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(offset_of!(ObjectUniformData, clip_rect), rect_data);
    uniform_data.copy_from_with_offset(offset_of!(ObjectUniformData, img_offset), img_offset);
    uniform_data.copy_from_with_offset(offset_of!(ObjectUniformData, img_scale), img_scale);
}

fn on_update(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext, _dt: f64) {
    let renderer = ctx.module::<MainRenderer>();

    let mut obj = scene.object::<UIImage>(entity);
    let img_aspect = obj.state().aspect;
    let Some(source) = obj.state_mut().source.take() else {
        return;
    };
    let size = source.size();

    let original_aspect_ratio = size.0 as f32 / size.1 as f32;
    obj.state_mut().original_aspect_ratio = original_aspect_ratio;

    let layout = obj.get_mut::<UILayoutC>();
    match img_aspect {
        ImageAspect::Original => {
            layout.aspect = Some(original_aspect_ratio);
        }
        ImageAspect::Cover => {
            layout.aspect = None;
        }
    }

    let res = Resource::image(
        renderer.device(),
        ImageParams::d2(Format::RGBA8_SRGB, ImageUsageFlags::SAMPLED, source.size())
            .with_preferred_mip_levels(1),
        source.into_raw(),
    )
    .unwrap();

    let mesh_cfg = obj.get_mut::<MeshRenderConfigC>();
    mesh_cfg.set_shader_resource(0, res);
}
