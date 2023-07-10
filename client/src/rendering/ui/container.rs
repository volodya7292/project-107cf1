use crate::rendering::ui::UIContext;
use common::glm::Vec4;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::transition::{AnimatedValue, TransitionTarget};
use engine::ecs::component::ui::{Factor, Sizing, UILayoutC};
use engine::ecs::component::{MeshRenderConfigC, SceneEventHandler, UniformDataC, VertexMeshC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, ObjectEntityId, Scene};
use engine::module::ui::color::Color;
use engine::module::ui::management::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::vkw::pipeline::CullMode;
use engine::vkw::PrimitiveTopology;
use engine::EngineContext;
use entity_data::EntityId;

#[derive(Copy, Clone)]
struct ContainerImplContext {
    mat_pipe_id: MaterialPipelineId,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct UniformData {
    background_color: Vec4,
}

#[derive(Clone)]
pub struct ContainerState {
    background_color: AnimatedValue<Color>,
}

impl UIState for ContainerState {}

pub type Container = UIObject<ContainerState>;

pub trait ContainerImpl {
    fn register(ctx: &EngineContext) {
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

        scene.add_resource(ContainerImplContext { mat_pipe_id });
    }

    fn new(ui_ctx: &mut UIContext, parent: EntityId, layout: UILayoutC) -> ObjectEntityId<Container> {
        let impl_ctx = ui_ctx.scene.resource::<ContainerImplContext>();

        let container = Container::new_raw(
            layout,
            ContainerState {
                background_color: Default::default(),
            },
        )
        .with_renderer(
            MeshRenderConfigC::new(impl_ctx.mat_pipe_id, true).with_render_layer(RenderLayer::Overlay),
        )
        .with_mesh(VertexMeshC::without_data(4, 1))
        .with_scene_event_handler(SceneEventHandler::new().with_on_update(on_update));

        ui_ctx.scene().add_object(Some(parent), container).unwrap()
    }

    fn expander(ui_ctx: &mut UIContext, parent: EntityId, fraction: Factor) -> ObjectEntityId<Container> {
        Self::new(
            ui_ctx,
            parent,
            UILayoutC::new()
                .with_width(Sizing::Grow(fraction))
                .with_height(Sizing::Grow(fraction)),
        )
    }

    fn spacer(ui_ctx: &mut UIContext, parent: EntityId, size: f32) -> ObjectEntityId<Container> {
        Container::new(
            ui_ctx,
            parent,
            UILayoutC::new()
                .with_width(Sizing::Preferred(size))
                .with_height(Sizing::Preferred(size)),
        )
    }
}

impl ContainerImpl for Container {}

pub trait ContainerAccess {
    fn get_background_color(&self) -> &Color;
    fn set_background_color(&mut self, color: TransitionTarget<Color>);
}

impl<'a> ContainerAccess for EntityAccess<'a, Container> {
    fn get_background_color(&self) -> &Color {
        self.state().background_color.current()
    }

    fn set_background_color(&mut self, color: TransitionTarget<Color>) {
        self.state_mut().background_color.retarget(color);
        self.request_update();
    }
}

fn on_update(entity: &EntityId, scene: &mut Scene, _: &EngineContext, dt: f64) {
    let mut entry = scene.object::<Container>(&entity.into());
    let mut state = entry.state_mut().clone();

    if !state.background_color.advance(dt) {
        entry.request_update();
    }

    let uniform_data = entry.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(
        offset_of!(UniformData, background_color),
        state.background_color.current(),
    );
}
