use crate::rendering::ui::UIContext;
use common::glm::Vec4;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{Constraint, RectUniformData, UILayoutC, UILayoutCacheC};
use engine::ecs::component::{SceneEventHandler, SimpleTextC, TransformC, UniformDataC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, Scene};
use engine::module::text_renderer::{RawTextObject, TextRenderer};
use engine::module::ui::management::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::EngineContext;
use entity_data::EntityId;

#[derive(Copy, Clone)]
struct TextImplContext {
    mat_pipe_id: MaterialPipelineId,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct UniformData {
    background_color: Vec4,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
pub struct ObjectUniformData {
    clip_rect: RectUniformData,
    inner_shadow_intensity: f32,
}

#[derive(Clone)]
pub struct TextState {
    raw_text_entity: EntityId,
    text: StyledString,
    inner_shadow_intensity: f32,
}

impl UIState for TextState {}

pub type UIText = UIObject<TextState>;

pub trait UITextImpl {
    fn register(ctx: &EngineContext) {
        let mut renderer = ctx.module_mut::<MainRenderer>();
        let mut text_renderer = ctx.module_mut::<TextRenderer>();
        let scene = ctx.module_mut::<Scene>();

        let pixel = renderer
            .device()
            .create_pixel_shader(
                include_bytes!("../../../res/shaders/text_char_ui.frag.spv"),
                "text_ui.frag",
            )
            .unwrap();

        let mat_pipe_id = text_renderer.register_text_pipeline(&mut renderer, pixel);

        scene.add_resource(TextImplContext { mat_pipe_id });
    }

    fn new(ctx: &mut UIContext, parent: EntityId) -> EntityId {
        let impl_ctx = *ctx.scene.resource::<TextImplContext>();

        let main_obj = UIText::new_raw(
            UILayoutC::new().with_shader_inverted_y(true),
            TextState {
                raw_text_entity: Default::default(),
                text: Default::default(),
                inner_shadow_intensity: 0.0,
            },
        )
        .with_scene_event_handler(
            SceneEventHandler::new()
                .with_on_update(on_update)
                .with_on_component_update::<UILayoutCacheC>(on_layout_cache_update),
        )
        .disable_pointer_events();

        let raw_text_obj = RawTextObject::new(
            TransformC::new(),
            SimpleTextC::new(impl_ctx.mat_pipe_id)
                .with_max_width(0.0)
                .with_render_type(RenderLayer::Overlay),
        );

        let text_entity = ctx.scene.add_object(Some(parent), main_obj).unwrap();
        let raw_text_entity = ctx.scene.add_object(Some(text_entity), raw_text_obj).unwrap();

        let mut main_obj = ctx.scene.object::<UIText>(&text_entity);
        main_obj.get_mut::<TextState>().raw_text_entity = raw_text_entity;

        text_entity
    }
}

impl UITextImpl for UIText {}

fn on_layout_cache_update(entity: &EntityId, scene: &mut Scene, _: &EngineContext) {
    let entry = scene.entry(entity);
    let cache = entry.get::<UILayoutCacheC>();
    let rect_data = *cache.calculated_clip_rect();

    let raw_text_entity = entry.get::<TextState>().raw_text_entity;
    drop(entry);
    let mut raw_text = scene.entry(&raw_text_entity);

    let uniform_data = raw_text.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(offset_of!(ObjectUniformData, clip_rect), rect_data);
}

fn on_update(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext, _: f64) {
    let mut entry = scene.entry(entity);
    let state = entry.get_mut::<TextState>().clone();

    let text_renderer = ctx.module_mut::<TextRenderer>();
    let size = text_renderer.calculate_minimum_text_size(&state.text);

    let layout = entry.get_mut::<UILayoutC>();
    layout.constraints[0].min = size.x;
    layout.constraints[1].min = size.y;
    drop(entry);

    let mut raw_text_entry = scene.entry(&state.raw_text_entity);
    let simple_text = raw_text_entry.get_mut::<SimpleTextC>();
    simple_text.text = state.text;
}

pub trait TextAccess {
    fn get_text(&self) -> &StyledString;
    fn set_text(&mut self, text: StyledString);
}

impl<'a> TextAccess for EntityAccess<'a, UIText> {
    fn get_text(&self) -> &StyledString {
        &self.state().text
    }

    fn set_text(&mut self, text: StyledString) {
        let simple_text = self.state_mut();
        simple_text.text = text;
        self.request_update();
    }
}
