use crate::game::EngineCtxGameExt;
use crate::rendering::ui::{UICallbacks, LOCAL_VAR_OPACITY, STATE_ENTITY_ID};
use common::glm::{Vec2, Vec4};
use common::memoffset::offset_of;
use common::scene::relation::Relation;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::ui::{RectUniformData, Sizing, UILayoutC};
use engine::ecs::component::{MeshRenderConfigC, UniformDataC, VertexMeshC};
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
use std::sync::Arc;

#[derive(Copy, Clone)]
#[repr(C)]
pub struct SolidColorUniformData {
    color: Vec4,
}

#[derive(Default, PartialEq, Clone)]
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
    corner_radius: f32,
    // ...custom properties
}

#[derive(PartialEq, Clone)]
pub struct ContainerProps<P> {
    pub layout: UILayoutC,
    pub callbacks: UICallbacks,
    pub background: Option<ContainerBackground>,
    pub opacity: f32,
    pub corner_radius: f32,
    pub children_props: P,
}

impl<P> ContainerProps<P> {
    pub fn layout(mut self, layout: UILayoutC) -> Self {
        self.layout = layout;
        self
    }

    pub fn callbacks(mut self, callbacks: UICallbacks) -> Self {
        self.callbacks = callbacks;
        self
    }

    pub fn background(mut self, background: Option<ContainerBackground>) -> Self {
        self.background = background;
        self
    }

    pub fn corner_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }

    pub fn children_props(mut self, props: P) -> Self {
        self.children_props = props;
        self
    }
}

impl<P: Default> Default for ContainerProps<P> {
    fn default() -> Self {
        Self {
            layout: Default::default(),
            callbacks: Default::default(),
            background: None,
            opacity: 1.0,
            corner_radius: 0.0,
            children_props: Default::default(),
        }
    }
}

pub fn container_props<P: Default>() -> ContainerProps<P> {
    Default::default()
}

pub fn container_props_init<P>(children_props: P) -> ContainerProps<P> {
    ContainerProps {
        layout: Default::default(),
        callbacks: Default::default(),
        background: None,
        opacity: 0.0,
        corner_radius: 0.0,
        children_props,
    }
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
                include_bytes!("../../../res/shaders/ui_background_solid.frag.spv"),
                "ui_background_solid.frag",
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
                color: color.into_raw_linear(),
            },
        )
    }
}

fn on_size_update(entity: &EntityId, ctx: &EngineContext, _new_size: Vec2) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut entry = scene.object::<UIObject<()>>(&entity.into());

    let cache = entry.layout_cache();
    let clip_rect = *cache.normalized_clip_rect();

    let uniform_data = entry.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(offset_of!(UniformData, clip_rect), clip_rect);
}

pub fn container<P, F>(local_name: &str, ctx: &mut UIScopeContext, props: ContainerProps<P>, children_fn: F)
where
    P: Clone + PartialEq + 'static,
    F: Fn(&mut UIScopeContext, P) + 'static,
{
    let child_num = ctx.num_children();
    let parent_entity = *ctx.state::<EntityId>(STATE_ENTITY_ID).value();
    let parent_opacity = *ctx.local_var::<f32>(LOCAL_VAR_OPACITY, 1.0);

    ctx.descend(
        local_name,
        (props, parent_opacity, child_num),
        move |scope_ctx, (props, parent_opacity, child_num)| {
            {
                let ctx = *scope_ctx.ctx();
                let entity_state = scope_ctx.request_state(STATE_ENTITY_ID, || {
                    let obj = UIObject::new_raw(props.layout, ()).with_mesh(VertexMeshC::without_data(4, 1));
                    *ctx.scene().add_object(Some(parent_entity), obj).unwrap()
                });

                let new_render_config = if let Some(background) = &props.background {
                    let mat_pipe_id = ctx
                        .scene()
                        .named_resource::<MaterialPipelineId>(background.mat_pipe_res_name);
                    MeshRenderConfigC::new(*mat_pipe_id, true).with_render_layer(RenderLayer::Overlay)
                } else {
                    Default::default()
                };

                // Set consecutive order
                {
                    let mut scene = ctx.scene();
                    let mut parent = scene.entry(&parent_entity);
                    parent
                        .get_mut::<Relation>()
                        .set_child_order(&entity_state.value(), Some(child_num as u32));
                }

                let opacity = parent_opacity * props.layout.visibility.opacity();
                scope_ctx.set_local_var(LOCAL_VAR_OPACITY, opacity);

                let mut scene = ctx.scene();
                let mut obj = scene.object::<UIObject<()>>(&(*entity_state.value()).into());
                *obj.get_mut::<MeshRenderConfigC>() = new_render_config;
                *obj.layout_mut() = props.layout;

                let raw_uniform_data = obj.get_mut::<UniformDataC>();
                raw_uniform_data.copy_from_with_offset(offset_of!(UniformData, opacity), opacity);
                raw_uniform_data
                    .copy_from_with_offset(offset_of!(UniformData, corner_radius), props.corner_radius);
                if let Some(background) = &props.background {
                    raw_uniform_data.copy_from_slice(mem::size_of::<UniformData>(), &background.uniform_data);
                }

                let event_handler = obj.event_handler_mut();
                *event_handler = props.callbacks.clone().into();
                event_handler.add_on_size_update(Arc::new(on_size_update));

                drop(obj);
            }
            children_fn(scope_ctx, props.children_props);
        },
        move |ctx, scope| {
            let entity = scope.state::<EntityId>(STATE_ENTITY_ID).unwrap();
            let mut scene = ctx.module_mut::<Scene>();
            scene.remove_object(&entity);
        },
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
            callbacks: UICallbacks::new().with_interaction(false),
            ..Default::default()
        },
        |_, ()| {},
    );
}

pub fn width_spacer(local_id: &str, ctx: &mut UIScopeContext, width: f32) {
    container(
        local_id,
        ctx,
        ContainerProps {
            layout: UILayoutC::new().with_min_width(width),
            callbacks: UICallbacks::new().with_interaction(false),
            ..Default::default()
        },
        |_, ()| {},
    );
}

pub fn height_spacer(local_id: &str, ctx: &mut UIScopeContext, height: f32) {
    container(
        local_id,
        ctx,
        ContainerProps {
            layout: UILayoutC::new().with_min_height(height),
            callbacks: UICallbacks::new().with_interaction(false),
            ..Default::default()
        },
        |_, ()| {},
    );
}
