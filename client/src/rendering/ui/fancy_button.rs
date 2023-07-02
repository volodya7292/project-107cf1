use crate::rendering::ui::text::{TextAccess, UIText, UITextImpl};
use crate::rendering::ui::UIContext;
use common::glm::{U8Vec4, Vec4};
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::simple_text::{StyledString, TextStyle};
use engine::ecs::component::transition::Transition;
use engine::ecs::component::ui::{UIEventHandlerC, UILayoutC};
use engine::ecs::component::{transition, MeshRenderConfigC, SceneEventHandler, UniformDataC, VertexMeshC};
use engine::module::main_renderer::MaterialPipelineId;
use engine::module::scene::{EntityAccess, Scene};
use engine::module::ui::management::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::EngineContext;
use entity_data::EntityId;

#[derive(Clone)]
pub struct FancyButtonState {
    text_entity: EntityId,
    text: String,
    background_color: Vec4,
    transition: Transition<Vec4>,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct UniformData {
    background_color: Vec4,
}

impl UIState for FancyButtonState {}

pub type FancyButton = UIObject<FancyButtonState>;

pub trait FancyButtonImpl {
    fn new(ctx: &mut UIContext, parent: EntityId, mat_pipeline: MaterialPipelineId) -> EntityId {
        let surface_obj = FancyButton::new_raw(
            UILayoutC::new(),
            FancyButtonState {
                text_entity: Default::default(),
                text: "".to_owned(),
                background_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
                transition: Transition::none(Vec4::new(0.5, 0.5, 0.5, 1.0)),
            },
        )
        .with_scene_event_handler(SceneEventHandler::new())
        .with_renderer(MeshRenderConfigC::new(mat_pipeline, true).with_render_layer(RenderLayer::Overlay))
        .with_mesh(VertexMeshC::without_data(4, 1))
        .with_scene_event_handler(SceneEventHandler::new().with_on_update(on_update))
        .add_event_handler(
            UIEventHandlerC::new()
                .add_on_cursor_enter(on_cursor_enter)
                .add_on_cursor_leave(on_cursor_leave),
        );

        let btn_entity = ctx.scene.add_object(Some(parent), surface_obj).unwrap();
        let text_entity = UIText::new(ctx, btn_entity);

        let mut surface_obj = ctx.scene.object::<FancyButton>(&btn_entity);
        surface_obj.get_mut::<FancyButtonState>().text_entity = text_entity;

        btn_entity
    }
}

impl FancyButtonImpl for FancyButton {}

pub trait FancyButtonAccess {
    fn set_text(&mut self, text: &str);
}

impl FancyButtonAccess for EntityAccess<'_, FancyButton> {
    fn set_text(&mut self, text: &str) {
        let state = self.get_mut::<FancyButtonState>();
        state.text = text.to_owned();
        self.request_update();
    }
}

fn on_update(entity: &EntityId, scene: &mut Scene, _: &EngineContext, dt: f64) {
    let mut entry = scene.object::<FancyButton>(entity);
    let state = entry.state_mut();
    let mut update_needed = false;

    if !state.transition.advance(&mut state.background_color, dt) {
        update_needed = true;
    }

    let state = state.clone();

    let raw_uniform_data = entry.get_mut::<UniformDataC>();
    raw_uniform_data.copy_from(UniformData {
        background_color: state.background_color,
    });

    if update_needed {
        entry.request_update();
    }

    drop(entry);

    let mut text_obj = scene.object::<UIText>(&state.text_entity);
    text_obj.set_text(StyledString::new(
        state.text,
        TextStyle::new()
            .with_font_size(50.0)
            .with_color(U8Vec4::new(0, 0, 0, 255)),
    ));
}

fn on_cursor_enter(entity: &EntityId, scene: &mut Scene, _: &EngineContext) {
    let mut entry = scene.object::<FancyButton>(entity);
    let state = entry.state_mut();
    state.transition = Transition::new(
        state.background_color,
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        0.2,
        transition::FN_EASE_IN_OUT,
    );
    entry.request_update();
}

fn on_cursor_leave(entity: &EntityId, scene: &mut Scene, _: &EngineContext) {
    let mut entry = scene.object::<FancyButton>(entity);
    let state = entry.state_mut();
    state.transition = Transition::new(
        state.background_color,
        Vec4::new(0.5, 0.5, 0.5, 1.0),
        0.2,
        transition::FN_EASE_IN_OUT,
    );
    entry.request_update();
}
