use crate::rendering::ui::UIContext;
use common::glm::Vec2;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::{RenderLayer, Resource};
use engine::ecs::component::ui::{RectUniformData, UIEventHandlerC, UILayoutC, UILayoutCacheC};
use engine::ecs::component::{MeshRenderConfigC, UniformDataC, VertexMeshC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, ObjectEntityId, Scene};
use engine::module::ui::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::vkw::image::ImageParams;
use engine::vkw::pipeline::CullMode;
use engine::vkw::{Format, ImageUsageFlags, PrimitiveTopology};
use engine::EngineContext;
use entity_data::EntityId;
use smallvec::smallvec;
use std::sync::Arc;

struct ImageImplContext {
    mat_pipe_id: MaterialPipelineId,
}

#[derive(Clone)]
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
pub enum ImageFitness {
    Contain,
    Cover,
}

pub struct ImageState {
    fitness: ImageFitness,
    image_aspect_ratio: f32,
    source: Option<ImageSource>,
}

impl UIState for ImageState {
    fn on_update(entity: &EntityId, ctx: &EngineContext, _dt: f64) {
        let mut scene = ctx.module_mut::<Scene>();
        let renderer = ctx.module::<MainRenderer>();

        let mut obj = scene.object::<UIImage>(&entity.into());
        let Some(source) = obj.state_mut().source.take() else {
            return;
        };
        let size = source.size();

        let original_aspect_ratio = size.0 as f32 / size.1 as f32;
        obj.state_mut().image_aspect_ratio = original_aspect_ratio;

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
}

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

        scene.register_resource(ImageImplContext { mat_pipe_id });
    }

    fn new(
        ui_ctx: &mut UIContext,
        parent: EntityId,
        layout: UILayoutC,
        source: Option<ImageSource>,
        fitness: ImageFitness,
    ) -> ObjectEntityId<UIImage> {
        let impl_ctx = ui_ctx.scene.resource::<ImageImplContext>();
        let renderer = ui_ctx.ctx.module::<MainRenderer>();
        let initial_image = Resource::image(
            renderer.device(),
            ImageParams::d2(Format::RGBA8_UNORM, ImageUsageFlags::SAMPLED, (1, 1)),
            vec![0_u8; 4],
        )
        .unwrap();

        let ui_image = UIImage::new_raw(
            layout,
            ImageState {
                fitness,
                image_aspect_ratio: 1.0,
                source,
            },
        )
        .with_renderer(
            MeshRenderConfigC::new(impl_ctx.mat_pipe_id, true)
                .with_render_layer(RenderLayer::Overlay)
                .with_shader_resources(smallvec![initial_image]),
        )
        .with_mesh(VertexMeshC::without_data(4, 1));

        let entity = ui_ctx.scene().add_object(Some(parent), ui_image).unwrap();

        {
            let mut obj = ui_ctx.scene().entry(&entity);
            obj.get_mut::<UIEventHandlerC>().on_size_update = Some(Arc::new(on_size_update));
        }

        entity
    }

    fn with_source(self, source: ImageSource) -> Self;
}

impl ImageImpl for UIImage {
    fn with_source(mut self, source: ImageSource) -> Self {
        self.state.source = Some(source);
        self
    }
}

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
struct UniformData {
    clip_rect: RectUniformData,
    img_offset: Vec2,
    img_scale: Vec2,
    opacity: f32,
}

fn on_size_update(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut obj = scene.object::<UIImage>(&entity.into());
    let cache = obj.get::<UILayoutCacheC>();
    let rect_data = *cache.normalized_clip_rect();
    let final_size = *cache.final_size();
    let ui_element_aspect_ratio = final_size.x / final_size.y;

    let state = obj.state();
    let fitness = state.fitness;
    let shader_aspect_ratio = state.image_aspect_ratio / ui_element_aspect_ratio;

    // Calculate offset and scale for 'cover' image fitness.
    let (img_offset, img_scale) = match fitness {
        ImageFitness::Contain => {
            if shader_aspect_ratio > 1.0 {
                let height = 1.0 / shader_aspect_ratio;
                (Vec2::new(0.0, 0.5 * (1.0 - height)), Vec2::new(1.0, height))
            } else {
                let width = shader_aspect_ratio;
                (Vec2::new(0.5 * (1.0 - width), 0.0), Vec2::new(width, 1.0))
            }
        }
        ImageFitness::Cover => {
            if shader_aspect_ratio > 1.0 {
                let width = shader_aspect_ratio;
                (Vec2::new(-0.5 * (width - 1.0), 0.0), Vec2::new(width, 1.0))
            } else {
                let height = 1.0 / shader_aspect_ratio;
                (Vec2::new(0.0, -0.5 * (height - 1.0)), Vec2::new(1.0, height))
            }
        }
    };

    let uniform_data = obj.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(offset_of!(UniformData, clip_rect), rect_data);
    uniform_data.copy_from_with_offset(offset_of!(UniformData, img_offset), img_offset);
    uniform_data.copy_from_with_offset(offset_of!(UniformData, img_scale), img_scale);
}

pub mod reactive {
    use super::UniformData;
    use crate::rendering::ui::image::{ImageAccess, ImageFitness, ImageImpl, ImageSource, UIImage};
    use crate::rendering::ui::{UIContext, LOCAL_VAR_OPACITY, STATE_ENTITY_ID};
    use common::memoffset::offset_of;
    use common::scene::relation::Relation;
    use engine::ecs::component::ui::UILayoutC;
    use engine::ecs::component::UniformDataC;
    use engine::module::scene::Scene;
    use engine::module::ui::reactive::UIScopeContext;
    use entity_data::EntityId;

    pub fn ui_image<F: Fn(&mut UIScopeContext) + 'static>(
        local_id: &str,
        ctx: &mut UIScopeContext,
        layout: UILayoutC,
        source: Option<ImageSource>,
        fitness: ImageFitness,
        children: F,
    ) {
        let parent = ctx.scope_id().clone();
        let parent_entity = *ctx
            .reactor()
            .get_state::<EntityId>(parent.clone(), STATE_ENTITY_ID.to_string())
            .unwrap()
            .value();
        let parent_opacity = ctx.reactor().local_var::<f32>(&parent, LOCAL_VAR_OPACITY, 1.0);
        let child_num = ctx.num_children();

        ctx.descend(
            local_id,
            move |ctx| {
                {
                    let mut ui_ctx = UIContext::new(*ctx.ctx());
                    let entity_state = ctx.request_state(STATE_ENTITY_ID, || {
                        *UIImage::new(&mut ui_ctx, parent_entity, layout, None, fitness)
                    });

                    // Set consecutive order
                    {
                        let mut parent = ui_ctx.scene().entry(&parent_entity);
                        parent
                            .get_mut::<Relation>()
                            .set_child_order(entity_state.value(), Some(child_num as u32));
                    }

                    let mut obj = ui_ctx.scene().object::<UIImage>(&entity_state.value().into());

                    let opacity = *parent_opacity;
                    ctx.set_local_var(LOCAL_VAR_OPACITY, opacity);

                    let raw_uniform_data = obj.get_mut::<UniformDataC>();
                    UniformDataC::copy_from_with_offset(
                        raw_uniform_data,
                        offset_of!(UniformData, opacity),
                        opacity,
                    );

                    if let Some(source) = source.clone() {
                        obj.set_image(source);
                    }
                }
                children(ctx);
            },
            move |ctx, scope| {
                let entity = scope.state::<EntityId>(STATE_ENTITY_ID).unwrap();
                let mut scene = ctx.module_mut::<Scene>();
                scene.remove_object(&*entity);
            },
            true,
        );
    }
}