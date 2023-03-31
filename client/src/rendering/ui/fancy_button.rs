use common::glm::{Mat4, U8Vec4, Vec4};
use common::scene::relation::Relation;
use engine::ecs::component::render_config::RenderType;
use engine::ecs::component::simple_text::{StyledString, TextStyle};
use engine::ecs::component::transition::{TransValue, Transition};
use engine::ecs::component::ui::{UIEventHandlerC, UILayoutC};
use engine::ecs::component::{transition, MeshRenderConfigC, SceneEventHandler, UniformDataC, VertexMeshC};
use engine::module::main_renderer::MainRenderer;
use engine::module::scene::{EntityAccess, Scene};
use engine::module::ui::element::{TextState, UIText};
use engine::module::ui::management::UIState;
use engine::module::ui::{UIObject, UIRenderer};
use engine::vkw::PrimitiveTopology;
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

pub fn load_pipeline(renderer: &mut MainRenderer) -> u32 {
    let device = renderer.device();

    let vertex = device
        .create_vertex_shader(
            include_bytes!("../../../res/shaders/ui_rect.vert.spv"),
            &[],
            "ui_rect.vert",
        )
        .unwrap();
    let pixel = device
        .create_pixel_shader(
            include_bytes!("../../../res/shaders/fancy_button.frag.spv"),
            "fancy_button.frag",
        )
        .unwrap();

    renderer.register_material_pipeline(&[vertex, pixel], PrimitiveTopology::TRIANGLE_STRIP, true)
}

impl UIState for FancyButtonState {}

pub type FancyButton = UIObject<FancyButtonState>;

pub fn new(scene: &mut Scene, parent: EntityId, mat_pipeline: u32) -> EntityId {
    let mut surface_obj = FancyButton::new_raw(
        UILayoutC::new(),
        FancyButtonState {
            text_entity: Default::default(),
            text: "".to_owned(),
            background_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            transition: Transition::none(Vec4::new(0.5, 0.5, 0.5, 1.0)),
        },
    )
    .with_scene_event_handler(SceneEventHandler::new())
    .with_renderer(MeshRenderConfigC::new(mat_pipeline, true).with_stage(RenderType::OVERLAY))
    .with_mesh(VertexMeshC::without_data(4, 1))
    .with_scene_event_handler(SceneEventHandler::new().with_on_update(on_update))
    .add_event_handler(
        UIEventHandlerC::new()
            .add_on_cursor_enter(on_cursor_enter)
            .add_on_cursor_leave(on_cursor_leave),
    );

    let btn_entity = scene.add_object(Some(parent), surface_obj).unwrap();
    let text_entity = scene.add_object(Some(btn_entity), UIText::new()).unwrap();

    let mut surface_obj = scene.object::<FancyButton>(&btn_entity);
    surface_obj.get_mut::<FancyButtonState>().text_entity = text_entity;

    btn_entity
}

pub trait FancyButtonImpl {
    fn set_text(&mut self, text: &str);
}

impl FancyButtonImpl for EntityAccess<'_, FancyButton> {
    fn set_text(&mut self, text: &str) {
        let state = self.get_mut::<FancyButtonState>();
        state.text = text.to_owned();
        self.request_update();
    }
}

fn on_update(entity: &EntityId, scene: &mut Scene, _: &EngineContext, dt: f64) {
    let mut entry = scene.entry(entity);
    let state = entry.get_mut::<FancyButtonState>();
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
    let mut entry = scene.entry(entity);
    let state = entry.get_mut::<FancyButtonState>();
    state.transition = Transition::new(
        state.background_color,
        Vec4::new(1.0, 1.0, 1.0, 1.0),
        0.2,
        transition::FN_EASE_IN_OUT,
    );
    entry.request_update();
}

fn on_cursor_leave(entity: &EntityId, scene: &mut Scene, _: &EngineContext) {
    let mut entry = scene.entry(entity);
    let state = entry.get_mut::<FancyButtonState>();
    state.transition = Transition::new(
        state.background_color,
        Vec4::new(0.5, 0.5, 0.5, 1.0),
        0.2,
        transition::FN_EASE_IN_OUT,
    );
    entry.request_update();
}
