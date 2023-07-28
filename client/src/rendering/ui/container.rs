use crate::rendering::ui::{UICallbacks, UIContext, STATE_ENTITY_ID};
use common::glm::Vec4;
use common::memoffset::offset_of;
use common::scene::relation::Relation;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::ui::{RectUniformData, Sizing, UILayoutC, UILayoutCacheC};
use engine::ecs::component::{MeshRenderConfigC, SceneEventHandler, UniformDataC, VertexMeshC};
use engine::module::main_renderer::MaterialPipelineId;
use engine::module::scene::Scene;
use engine::module::ui::reactive::UIScopeContext;
use engine::module::ui::UIObject;
use engine::module::ui::UIObjectEntityImpl;
use engine::module::ui::UIState;
use engine::utils::U8SliceHelper;
use engine::EngineContext;
use entity_data::EntityId;
use smallvec::{smallvec, SmallVec};
use std::mem;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SolidColorUniformData {
    color: Vec4,
}

#[derive(Default)]
pub struct ContainerBackground {
    mat_pipe_res_name: &'static str,
    uniform_data: SmallVec<[u8; 128]>,
}

impl ContainerBackground {
    pub fn new_raw<U: Copy>(mat_pipe_res_name: &'static str, data: U) -> Self {
        let mut uniform_data: SmallVec<[u8; 128]> = smallvec![0; mem::size_of::<U>()];
        uniform_data.raw_copy_from(data);

        Self {
            mat_pipe_res_name,
            uniform_data,
        }
    }
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct UniformData {
    clip_rect: RectUniformData,
    opacity: f32,
    // ...custom properties
}

#[derive(Default)]
pub struct ContainerProps {
    pub layout: UILayoutC,
    pub callbacks: UICallbacks,
    pub background: Option<ContainerBackground>,
}

pub mod background {
    use crate::rendering::ui::container::{ContainerBackground, SolidColorUniformData};
    use common::make_static_id;
    use engine::module::main_renderer::MainRenderer;
    use engine::module::scene::Scene;
    use engine::module::ui::color::Color;
    use engine::vkw::pipeline::CullMode;
    use engine::vkw::PrimitiveTopology;
    use engine::EngineContext;

    const SOLID_COLOR_MATERIAL_PIPE_RES_NAME: &str = make_static_id!();

    pub fn register_backgrounds(ctx: &EngineContext) {
        let mut renderer = ctx.module_mut::<MainRenderer>();
        let scene = ctx.module_mut::<Scene>();

        let vertex = renderer
            .device()
            .create_vertex_shader(
                include_bytes!("../../../res/shaders/ui_rect.vert.spv"),
                &[],
                "ui_rect.vert",
            )
            .unwrap();
        let pixel = renderer
            .device()
            .create_pixel_shader(
                include_bytes!("../../../res/shaders/ui_container.frag.spv"),
                "ui_container.frag",
            )
            .unwrap();

        let mat_pipe_id = renderer.register_material_pipeline(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_STRIP,
            CullMode::BACK,
        );

        scene.register_named_resource(SOLID_COLOR_MATERIAL_PIPE_RES_NAME, mat_pipe_id);
    }

    pub fn solid_color(color: Color) -> ContainerBackground {
        ContainerBackground::new_raw(
            SOLID_COLOR_MATERIAL_PIPE_RES_NAME,
            SolidColorUniformData {
                color: color.into_raw(),
            },
        )
    }
}

fn on_layout_cache_update(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut entry = scene.object::<UIObject<()>>(&entity.into());

    let cache = entry.layout_cache();
    let clip_rect = *cache.normalized_clip_rect();
    let final_opacity = cache.final_opacity();

    let uniform_data = entry.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(offset_of!(UniformData, clip_rect), clip_rect);
    uniform_data.copy_from_with_offset(offset_of!(UniformData, opacity), final_opacity);
}

pub fn container<F: Fn(&mut UIScopeContext) + 'static>(
    local_name: &str,
    ctx: &mut UIScopeContext,
    props: ContainerProps,
    children_fn: F,
) {
    let parent = ctx.scope_id().clone();
    let child_num = ctx.num_children();
    let parent_entity = *ctx
        .reactor()
        .get_state::<EntityId>(parent, STATE_ENTITY_ID.to_string())
        .unwrap()
        .value();

    ctx.descend(
        local_name,
        move |ctx| {
            {
                let mut ui_ctx = UIContext::new(*ctx.ctx());
                let entity_state = ctx.request_state(STATE_ENTITY_ID, || {
                    let obj = UIObject::new_raw(props.layout, ())
                        .with_mesh(VertexMeshC::without_data(4, 1))
                        .with_scene_event_handler(
                            SceneEventHandler::new()
                                .with_on_component_update::<UILayoutCacheC>(on_layout_cache_update),
                        );
                    *ui_ctx.scene().add_object(Some(parent_entity), obj).unwrap()
                });

                let new_render_config = if let Some(background) = &props.background {
                    let mat_pipe_id = ui_ctx
                        .scene()
                        .named_resource::<MaterialPipelineId>(background.mat_pipe_res_name);
                    MeshRenderConfigC::new(*mat_pipe_id, true).with_render_layer(RenderLayer::Overlay)
                } else {
                    Default::default()
                };

                // Set consecutive order
                {
                    let mut parent = ui_ctx.scene().entry(&parent_entity);
                    parent
                        .get_mut::<Relation>()
                        .set_child_order(entity_state.value(), Some(child_num as u32));
                }

                let mut obj = ui_ctx
                    .scene()
                    .object::<UIObject<()>>(&entity_state.value().into());
                *obj.get_mut::<MeshRenderConfigC>() = new_render_config;
                *obj.layout_mut() = props.layout;

                if let Some(background) = &props.background {
                    let raw_uniform_data = obj.get_mut::<UniformDataC>();
                    raw_uniform_data.copy_from_slice(mem::size_of::<UniformData>(), &background.uniform_data);
                }

                let event_handler = obj.event_handler_mut();
                props.callbacks.apply_to_event_handler(event_handler);
            }
            children_fn(ctx);
        },
        move |ctx, scope| {
            let entity = scope.state::<EntityId>(STATE_ENTITY_ID).unwrap();
            let mut scene = ctx.module_mut::<Scene>();
            scene.remove_object(&*entity);
        },
        true,
    );
}

pub fn expander(local_id: &str, ctx: &mut UIScopeContext, fraction: f32) {
    container(
        local_id,
        ctx,
        ContainerProps {
            layout: UILayoutC::new()
                .with_width(Sizing::Grow(fraction))
                .with_height(Sizing::Grow(fraction)),
            ..Default::default()
        },
        |_| {},
    );
}

pub fn width_spacer(local_id: &str, ctx: &mut UIScopeContext, width: f32) {
    container(
        local_id,
        ctx,
        ContainerProps {
            layout: UILayoutC::new().with_min_width(width),
            ..Default::default()
        },
        |_| {},
    );
}

pub fn height_spacer(local_id: &str, ctx: &mut UIScopeContext, height: f32) {
    container(
        local_id,
        ctx,
        ContainerProps {
            layout: UILayoutC::new().with_min_height(height),
            ..Default::default()
        },
        |_| {},
    );
}
