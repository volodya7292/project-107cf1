use crate::rendering::ui::container::{Container, ContainerImpl};
use crate::rendering::ui::UIContext;
use common::glm::Vec2;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{
    Position, RectUniformData, Sizing, UIEventHandlerC, UILayoutC, UILayoutCacheC,
};
use engine::ecs::component::{SceneEventHandler, SimpleTextC, TransformC, UniformDataC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, ObjectEntityId, Scene, SceneObject};
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
pub struct ObjectUniformData {
    clip_rect: RectUniformData,
    opacity: f32,
    inner_shadow_intensity: f32,
}

#[derive(Clone)]
pub struct TextState {
    wrapper_entity: ObjectEntityId<Container>,
    raw_text_entity: ObjectEntityId<RawTextObject>,
    text: StyledString,
    wrap: bool,
    inner_shadow_intensity: f32,
}

impl UIState for TextState {
    fn on_update(entity: &EntityId, ctx: &EngineContext, _: f64) {
        let mut scene = ctx.module_mut::<Scene>();
        let mut entry = scene.entry(entity);
        let state = entry.get_mut::<TextState>().clone();

        let cache = entry.get::<UILayoutCacheC>();
        let rect_data = *cache.calculated_clip_rect();
        let final_opacity = cache.final_opacity();
        let final_size = *cache.final_size();

        let text_block_size = {
            let text_renderer = ctx.module_mut::<TextRenderer>();
            text_renderer.calculate_minimum_text_size(
                &state.text,
                if state.wrap { final_size.x } else { f32::INFINITY },
            )
        };

        drop(entry);

        let mut wrapper = scene.entry(&state.wrapper_entity);
        let layout = wrapper.get_mut::<UILayoutC>();
        layout.constraints[0].min = text_block_size.x;
        layout.constraints[1].min = text_block_size.y;
        drop(wrapper);

        let mut entry = scene.entry(entity);
        let layout = entry.get_mut::<UILayoutC>();
        if !state.wrap {
            layout.constraints[0].min = text_block_size.x;
        }
        layout.constraints[1].min = text_block_size.y;
        drop(entry);

        let mut raw_text_entry = scene.entry(&state.raw_text_entity);
        let simple_text = raw_text_entry.get_mut::<SimpleTextC>();
        simple_text.max_width = final_size.x;
        simple_text.text = state.text;

        let uniform_data = raw_text_entry.get_mut::<UniformDataC>();
        uniform_data.copy_from_with_offset(offset_of!(ObjectUniformData, clip_rect), rect_data);
        uniform_data.copy_from_with_offset(offset_of!(ObjectUniformData, opacity), final_opacity);
        uniform_data.copy_from_with_offset(
            offset_of!(ObjectUniformData, inner_shadow_intensity),
            state.inner_shadow_intensity,
        );
        drop(raw_text_entry);
    }
}

pub type UIText = UIObject<TextState>;

pub trait UITextImpl {
    fn register(ctx: &EngineContext) {
        let mut renderer = ctx.module_mut::<MainRenderer>();
        let mut text_renderer = ctx.module_mut::<TextRenderer>();
        let scene = ctx.module_mut::<Scene>();

        let pixel = renderer
            .device()
            .create_pixel_shader(
                include_bytes!("../../../res/shaders/ui_text_char.frag.spv"),
                "ui_text_char.frag",
            )
            .unwrap();

        let mat_pipe_id = text_renderer.register_text_pipeline(&mut renderer, pixel);

        scene.add_resource(TextImplContext { mat_pipe_id });
    }

    fn new(ui_ctx: &mut UIContext, parent: EntityId, text: StyledString) -> ObjectEntityId<UIText> {
        let impl_ctx = *ui_ctx.scene.resource::<TextImplContext>();

        let main_obj = UIText::new_raw(
            UILayoutC::new()
                .with_width(Sizing::Grow(1.0))
                .with_height(Sizing::FitContent),
            TextState {
                wrapper_entity: Default::default(),
                raw_text_entity: Default::default(),
                text,
                wrap: false,
                inner_shadow_intensity: 0.0,
            },
        )
        .with_scene_event_handler(
            SceneEventHandler::new().with_on_component_update::<UILayoutCacheC>(on_layout_cache_update),
        )
        .disable_pointer_events();

        let raw_text_obj = RawTextObject::new(
            TransformC::new(),
            SimpleTextC::new(impl_ctx.mat_pipe_id)
                .with_max_width(f32::INFINITY)
                .with_render_type(RenderLayer::Overlay),
        );

        let main_entity = ui_ctx.scene.add_object(Some(parent), main_obj).unwrap();
        let wrapper_entity = Container::new(
            ui_ctx,
            *main_entity,
            UILayoutC::new()
                .with_position(Position::Relative(Vec2::new(0.0, 0.0)))
                .with_width(Sizing::FitContent)
                .with_height(Sizing::FitContent)
                .with_shader_inverted_y(true),
        );
        {
            let mut wrapper = ui_ctx.scene().object(&wrapper_entity);
            wrapper.get_mut::<UIEventHandlerC>().enabled = false;
        }

        let raw_text_entity = ui_ctx
            .scene
            .add_object(Some(*wrapper_entity), raw_text_obj)
            .unwrap();

        let mut main_obj = ui_ctx.scene.object::<UIText>(&main_entity);
        main_obj.get_mut::<TextState>().wrapper_entity = wrapper_entity;
        main_obj.get_mut::<TextState>().raw_text_entity = raw_text_entity;
        drop(main_obj);

        main_entity
    }
}

impl UITextImpl for UIText {}

fn on_layout_cache_update(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    scene.object::<UIText>(&entity.into()).request_update();
}

pub trait TextAccess {
    fn get_text(&self) -> &StyledString;
    fn set_text(&mut self, text: StyledString);
    fn get_wrap(&self) -> bool;
    fn set_wrap(&mut self, wrap: bool);
}

impl<'a> TextAccess for EntityAccess<'a, UIText> {
    fn get_text(&self) -> &StyledString {
        &self.state().text
    }

    fn set_text(&mut self, text: StyledString) {
        self.state_mut().text = text;
        self.request_update();
    }

    fn get_wrap(&self) -> bool {
        self.state().wrap
    }

    fn set_wrap(&mut self, wrap: bool) {
        self.state_mut().wrap = wrap;
        self.request_update();
    }
}
