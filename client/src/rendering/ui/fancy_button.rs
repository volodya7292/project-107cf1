use crate::rendering::ui::text::{TextAccess, UIText, UITextImpl};
use crate::rendering::ui::UIContext;
use common::glm::Vec4;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{
    BasicEventCallback, RectUniformData, UIEventHandlerC, UILayoutC, UILayoutCacheC,
};
use engine::ecs::component::{MeshRenderConfigC, SceneEventHandler, UniformDataC, VertexMeshC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, ObjectEntityId, Scene};
use engine::module::ui::color::Color;
use engine::module::ui::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::vkw::pipeline::CullMode;
use engine::vkw::PrimitiveTopology;
use engine::EngineContext;
use entity_data::EntityId;

struct FancyButtonImplContext {
    mat_pipe_id: MaterialPipelineId,
}

#[derive(Clone)]
pub struct FancyButtonState {
    text_entity: ObjectEntityId<UIText>,
    text: StyledString,
    text_color: AnimatedValue<Color>,
    background_color: Color,
    active_color: Color,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct UniformData {
    background_color: Vec4,
    clip_rect: RectUniformData,
    opacity: f32,
}

impl UIState for FancyButtonState {
    fn on_update(entity: &EntityId, ctx: &EngineContext, dt: f64) {
        let mut scene = ctx.module_mut::<Scene>();
        let mut entry = scene.object::<FancyButton>(&(*entity).into());
        let state = entry.state_mut();
        let mut update_needed = false;

        if !state.text_color.advance(dt) {
            update_needed = true;
        }

        let state = state.clone();

        let raw_uniform_data = entry.get_mut::<UniformDataC>();
        raw_uniform_data
            .copy_from_with_offset(offset_of!(UniformData, background_color), state.background_color);

        if update_needed {
            entry.request_update();
        }

        drop(entry);

        let mut text_obj = scene.object::<UIText>(&state.text_entity);
        text_obj.set_text(
            state
                .text
                .clone()
                .with_style(state.text.style().clone().with_color(*state.text_color.current())),
        );
    }
}

pub type FancyButton = UIObject<FancyButtonState>;

pub trait FancyButtonImpl {
    const DEFAULT_NORMAL_COLOR: Color = Color::grayscale(0.2);
    const DEFAULT_ACTIVE_COLOR: Color = Color::grayscale(0.8);
    const ACTIVE_TEXT_COLOR: Color = Color::grayscale(0.8);

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
                include_bytes!("../../../res/shaders/fancy_button.frag.spv"),
                "fancy_button.frag",
            )
            .unwrap();

        let mat_pipe_id = renderer.register_material_pipeline(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_STRIP,
            CullMode::BACK,
        );

        scene.add_resource(FancyButtonImplContext { mat_pipe_id });
    }

    fn new(
        ui_ctx: &mut UIContext,
        parent: EntityId,
        layout: UILayoutC,
        text: StyledString,
        on_click: BasicEventCallback,
    ) -> ObjectEntityId<FancyButton> {
        let impl_ctx = ui_ctx.scene.resource::<FancyButtonImplContext>();

        let surface_obj = FancyButton::new_raw(
            layout,
            FancyButtonState {
                text_entity: Default::default(),
                text: text.clone(),
                text_color: (*text.style().color()).into(),
                background_color: Self::DEFAULT_NORMAL_COLOR.into(),
                active_color: Self::DEFAULT_ACTIVE_COLOR,
            },
        )
        .with_renderer(
            MeshRenderConfigC::new(impl_ctx.mat_pipe_id, true).with_render_layer(RenderLayer::Overlay),
        )
        .with_mesh(VertexMeshC::without_data(4, 1))
        .with_scene_event_handler(
            SceneEventHandler::new().with_on_component_update::<UILayoutCacheC>(on_layout_cache_update),
        )
        .add_event_handler(
            UIEventHandlerC::new()
                .add_on_cursor_enter(on_cursor_enter)
                .add_on_cursor_leave(on_cursor_leave)
                .add_on_click(on_click),
        );

        let btn_entity = ui_ctx.scene.add_object(Some(parent), surface_obj).unwrap();
        let text_entity = UIText::new(ui_ctx, *btn_entity, text);

        let mut surface_obj = ui_ctx.scene.object::<FancyButton>(&btn_entity);
        surface_obj.get_mut::<FancyButtonState>().text_entity = text_entity;
        drop(surface_obj);

        btn_entity
    }
}

impl FancyButtonImpl for FancyButton {}

pub trait FancyButtonAccess {
    fn set_text(&mut self, text: StyledString);
}

impl FancyButtonAccess for EntityAccess<'_, FancyButton> {
    fn set_text(&mut self, mut text: StyledString) {
        let curr_col = *self.state().text_color.current();
        *text.style_mut().color_mut() = curr_col;
        self.get_mut::<FancyButtonState>().text = text;
        self.request_update();
    }
}

fn on_layout_cache_update(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut entry = scene.entry(entity);
    let clip_rect = *entry.get::<UILayoutCacheC>().calculated_clip_rect();
    let final_opacity = entry.get::<UILayoutCacheC>().final_opacity();
    let raw_uniform_data = entry.get_mut::<UniformDataC>();
    raw_uniform_data.copy_from_with_offset(offset_of!(UniformData, clip_rect), clip_rect);
    raw_uniform_data.copy_from_with_offset(offset_of!(UniformData, opacity), final_opacity);
}

fn on_cursor_enter(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut entry = scene.object::<FancyButton>(&entity.into());
    let state = entry.state_mut();

    let mut active_color = state.text.style().color().into_raw();
    active_color.x *= 1.6;
    active_color.y *= 1.6;
    active_color.z *= 1.6;
    state
        .text_color
        .retarget(TransitionTarget::new(active_color.into(), 0.2));

    entry.request_update();
}

fn on_cursor_leave(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut entry = scene.object::<FancyButton>(&entity.into());
    let state = entry.state_mut();
    state
        .text_color
        .retarget(TransitionTarget::new(*state.text.style().color(), 0.2));
    entry.request_update();
}
