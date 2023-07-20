use crate::rendering::ui::UIContext;
use common::glm::Vec2;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{
    Position, RectUniformData, Sizing, UIEventHandlerC, UILayoutC, UILayoutCacheC,
};
use engine::ecs::component::{SimpleTextC, TransformC, UniformDataC};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, ObjectEntityId, Scene};
use engine::module::text_renderer::{RawTextObject, TextRenderer};
use engine::module::ui::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::EngineContext;
use entity_data::EntityId;

#[derive(Copy, Clone)]
struct TextImplContext {
    mat_pipe_id: MaterialPipelineId,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
pub struct UniformData {
    clip_rect: RectUniformData,
    opacity: f32,
    inner_shadow_intensity: f32,
}

#[derive(Clone)]
pub struct TextState {
    wrapper_entity: ObjectEntityId<UIObject<WrapperState>>,
    raw_text_entity: ObjectEntityId<RawTextObject>,
    text: StyledString,
    wrap: bool,
    inner_shadow_intensity: f32,
}

#[derive(Clone)]
struct WrapperState {
    text: StyledString,
    wrap: bool,
}

impl UIState for WrapperState {}

impl UIState for TextState {
    fn on_update(entity: &EntityId, ctx: &EngineContext, _: f64) {
        let mut scene = ctx.module_mut::<Scene>();

        let mut entry = scene.entry(entity);
        let state = entry.get_mut::<TextState>().clone();
        drop(entry);

        let mut wrapper = scene.object(&state.wrapper_entity);
        wrapper.state_mut().text = state.text.clone();
        wrapper.state_mut().wrap = state.wrap;
        drop(wrapper);

        let mut raw_text_entry = scene.entry(&state.raw_text_entity);
        let simple_text = raw_text_entry.get_mut::<SimpleTextC>();
        simple_text.text = state.text;

        let uniform_data = raw_text_entry.get_mut::<UniformDataC>();
        uniform_data.copy_from_with_offset(
            offset_of!(UniformData, inner_shadow_intensity),
            state.inner_shadow_intensity,
        );
        drop(raw_text_entry);
    }
}

fn on_size_update(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut entry = scene.entry(entity);
    let state = entry.get_mut::<TextState>().clone();

    let cache = entry.get::<UILayoutCacheC>();
    let rect_data = *cache.calculated_clip_rect();
    let final_opacity = cache.final_opacity();
    let final_size = *cache.final_size();

    let text_block_size = {
        let text_renderer = ctx.module_mut::<TextRenderer>();
        text_renderer
            .calculate_minimum_text_size(&state.text, if state.wrap { final_size.x } else { f32::INFINITY })
    };

    let layout = entry.get_mut::<UILayoutC>();
    if !state.wrap {
        layout.constraints[0].min = text_block_size.x;
    }
    layout.constraints[1].min = text_block_size.y;
    drop(entry);

    let mut raw_text_entry = scene.entry(&state.raw_text_entity);
    let simple_text = raw_text_entry.get_mut::<SimpleTextC>();
    simple_text.max_width = final_size.x;

    let uniform_data = raw_text_entry.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(offset_of!(UniformData, clip_rect), rect_data);
    uniform_data.copy_from_with_offset(offset_of!(UniformData, opacity), final_opacity);

    drop(raw_text_entry);
}

fn wrapper_width_calc(entry: &EntityAccess<()>, ctx: &EngineContext, parent_size: &Vec2) -> f32 {
    let state = entry.get::<WrapperState>();
    let text_renderer = ctx.module_mut::<TextRenderer>();
    let block_size = text_renderer.calculate_minimum_text_size(
        &state.text,
        if state.wrap { parent_size.x } else { f32::INFINITY },
    );
    block_size.x
}

fn wrapper_height_calc(entry: &EntityAccess<()>, ctx: &EngineContext, parent_size: &Vec2) -> f32 {
    let state = entry.get::<WrapperState>();
    let text_renderer = ctx.module_mut::<TextRenderer>();
    let block_size = text_renderer.calculate_minimum_text_size(
        &state.text,
        if state.wrap { parent_size.x } else { f32::INFINITY },
    );
    block_size.y
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

        scene.register_resource(TextImplContext { mat_pipe_id });
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
                text: text.clone(),
                wrap: false,
                inner_shadow_intensity: 0.0,
            },
        )
        .disable_pointer_events();

        let raw_text_obj = RawTextObject::new(
            TransformC::new(),
            SimpleTextC::new(impl_ctx.mat_pipe_id)
                .with_text(text.clone())
                .with_max_width(f32::INFINITY)
                .with_render_type(RenderLayer::Overlay),
        );

        let main_entity = ui_ctx.scene.add_object(Some(parent), main_obj).unwrap();
        let wrapper_obj = UIObject::new_raw(
            UILayoutC::new()
                .with_position(Position::Relative(Vec2::new(0.0, 0.0)))
                .with_width(Sizing::ParentBased(wrapper_width_calc))
                .with_height(Sizing::ParentBased(wrapper_height_calc))
                .with_shader_inverted_y(true),
            WrapperState { text, wrap: false },
        );
        let wrapper_entity = ui_ctx.scene.add_object(Some(*main_entity), wrapper_obj).unwrap();
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
        main_obj.get_mut::<UIEventHandlerC>().on_size_update = Some(on_size_update);
        drop(main_obj);

        ui_ctx.ctx.dispatch_callback(move |ctx, _| {
            on_size_update(&main_entity, ctx);
        });

        main_entity
    }
}

impl UITextImpl for UIText {}

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

pub mod reactive {
    use crate::rendering::ui::text::{TextAccess, UIText, UITextImpl};
    use crate::rendering::ui::{UIContext, STATE_ENTITY_ID};
    use engine::ecs::component::simple_text::StyledString;
    use engine::module::scene::Scene;
    use engine::module::ui::reactive::UIScopeContext;
    use entity_data::EntityId;

    pub fn ui_text(local_name: &str, ctx: &mut UIScopeContext, text: StyledString) {
        let parent = ctx.scope_id().clone();
        let parent_entity = *ctx
            .reactor()
            .get_state::<EntityId>(parent, STATE_ENTITY_ID.to_string())
            .unwrap()
            .value();

        ctx.descend(
            local_name,
            move |ctx| {
                let mut ui_ctx = UIContext::new(*ctx.ctx());
                let entity_state = ctx.request_state(STATE_ENTITY_ID, || {
                    *UIText::new(&mut ui_ctx, parent_entity, text.clone())
                });
                let mut obj = ui_ctx.scene().object::<UIText>(&entity_state.value().into());

                obj.set_text(text.clone());
            },
            move |ctx, scope| {
                let entity = scope.state::<EntityId>(STATE_ENTITY_ID).unwrap();
                let mut scene = ctx.module_mut::<Scene>();
                scene.remove_object(&entity);
            },
            true,
        );
    }
}
