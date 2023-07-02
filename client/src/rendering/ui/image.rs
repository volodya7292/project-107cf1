use crate::rendering::ui::UIContext;
use engine::ecs::component::render_config::{RenderLayer, Resource};
use engine::ecs::component::ui::{Factor, Sizing, UILayoutC};
use engine::ecs::component::{MeshRenderConfigC, SceneEventHandler, VertexMeshC};
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

pub struct ImageState {
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
            .create_pixel_shader(
                include_bytes!("../../../res/shaders/ui_rect.vert.spv"),
                "ui_rect.vert",
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

    fn new(ui_ctx: &UIContext, layout: UILayoutC) -> UIImage {
        let impl_ctx = ui_ctx.scene.resource::<ImageImplContext>();
        let renderer = ui_ctx.ctx.module::<MainRenderer>();
        let initial_image = Resource::image(
            renderer.device(),
            ImageParams::d2(Format::RGBA8_UNORM, ImageUsageFlags::SAMPLED, (1, 1)),
            vec![0_u8; 4],
        )
        .unwrap();

        UIImage::new_raw(layout, ImageState { source: None })
            .with_renderer(
                MeshRenderConfigC::new(impl_ctx.mat_pipe_id, true)
                    .with_render_layer(RenderLayer::Overlay)
                    .with_shader_resources(smallvec![initial_image]),
            )
            .with_mesh(VertexMeshC::without_data(4, 1))
            .with_scene_event_handler(SceneEventHandler::new().with_on_update(on_update))
    }
}

impl ImageImpl for UIImage {}

pub trait ImageAccess {
    fn set_image(&mut self, image: ImageSource);
}

impl ImageAccess for EntityAccess<'_, UIImage> {
    fn set_image(&mut self, source: ImageSource) {
        self.state_mut().source = Some(source);
        self.request_update();
    }
}

fn on_update(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext, _dt: f64) {
    let renderer = ctx.module::<MainRenderer>();

    let mut obj = scene.object::<UIImage>(entity);
    let Some(source) = obj.state_mut().source.take() else {
        return;
    };

    let mesh_cfg = obj.get_mut::<MeshRenderConfigC>();

    let res = Resource::image(
        renderer.device(),
        ImageParams::d2(Format::RGBA8_SRGB, ImageUsageFlags::SAMPLED, source.size())
            .with_preferred_mip_levels(1),
        source.into_raw(),
    )
    .unwrap();

    mesh_cfg.set_shader_resource(0, res);
}
