use common::glm::Vec4;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::RenderType;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{RectUniformData, UILayoutC, UILayoutCacheC};
use engine::ecs::component::{SceneEventHandler, SimpleTextC, TransformC, UniformDataC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, Scene};
use engine::module::text_renderer::{RawTextObject, TextRenderer};
use engine::module::ui::management::UIState;
use engine::module::ui::UIObject;
use engine::EngineContext;
use entity_data::EntityId;

pub fn load_pipeline(renderer: &mut MainRenderer, text_renderer: &mut TextRenderer) -> MaterialPipelineId {
    let pixel = renderer
        .device()
        .create_pixel_shader(
            include_bytes!("../../../res/shaders/text_char_ui.frag.spv"),
            "text_ui.frag",
        )
        .unwrap();

    text_renderer.register_text_pipeline(renderer, pixel)
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

pub fn new(scene: &mut Scene, parent: EntityId, mat_pipeline: MaterialPipelineId) -> EntityId {
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
        SimpleTextC::new(mat_pipeline)
            .with_max_width(0.0)
            .with_render_type(RenderType::OVERLAY),
    );

    let text_entity = scene.add_object(Some(parent), main_obj).unwrap();
    let raw_text_entity = scene.add_object(Some(text_entity), raw_text_obj).unwrap();

    let mut main_obj = scene.object::<UIText>(&text_entity);
    main_obj.get_mut::<TextState>().raw_text_entity = raw_text_entity;

    text_entity
}

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

pub trait TextImpl {
    fn get_text(&self) -> &StyledString;
    fn set_text(&mut self, text: StyledString);
}

impl<'a> TextImpl for EntityAccess<'a, UIText> {
    fn get_text(&self) -> &StyledString {
        &self.get::<TextState>().text
    }

    fn set_text(&mut self, text: StyledString) {
        let simple_text = self.get_mut_checked::<TextState>().unwrap();
        simple_text.text = text;
        self.request_update();
    }
}
