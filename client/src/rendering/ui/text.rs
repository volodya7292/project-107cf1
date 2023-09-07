use crate::game::EngineCtxGameExt;
use common::glm::Vec2;
use common::memoffset::offset_of;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::simple_text::{StyledString, TextHAlign};
use engine::ecs::component::ui::{
    Position, RectUniformData, Sizing, UIEventHandlerC, UILayoutC, UILayoutCacheC,
};
use engine::ecs::component::{SimpleTextC, TransformC, UniformDataC};
use engine::module::main_renderer::shader::VkwShaderBundleDeviceExt;
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::scene::{EntityAccess, ObjectEntityId, Scene, SceneObject};
use engine::module::text_renderer::{self, RawTextObject, TextRenderer};
use engine::module::ui::UIState;
use engine::module::ui::{UIObject, UIObjectEntityImpl};
use engine::EngineContext;
use entity_data::EntityId;
use std::sync::Arc;

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
    main_obj_size: Vec2,
    wrapper_entity: ObjectEntityId<UIObject<WrapperState>>,
    raw_text_entity: ObjectEntityId<RawTextObject>,
    text: StyledString,
    wrap: bool,
    h_align: TextHAlign,
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
        simple_text.h_align = state.h_align;

        let uniform_data = raw_text_entry.get_mut::<UniformDataC>();
        uniform_data.copy_from_with_offset(
            offset_of!(UniformData, inner_shadow_intensity),
            state.inner_shadow_intensity,
        );
        drop(raw_text_entry);
    }
}

fn on_size_update(entity: &EntityId, ctx: &EngineContext, new_size: Option<Vec2>) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut entry = scene.entry(entity);
    let state = {
        let state = entry.get_mut::<TextState>();
        if let Some(new_size) = new_size {
            state.main_obj_size = new_size;
        }
        state.clone()
    };

    let cache = entry.get::<UILayoutCacheC>();
    let rect_data = *cache.normalized_clip_rect();

    let text_block_size = {
        let text_renderer = ctx.module_mut::<TextRenderer>();
        text_renderer.calculate_minimum_text_size(
            state.text.data(),
            state.text.style(),
            if state.wrap {
                state.main_obj_size.x
            } else {
                f32::INFINITY
            },
        )
    };

    let layout = entry.get_mut::<UILayoutC>();
    if state.wrap {
        layout.constraints[0].min = 0.0;
    } else {
        layout.constraints[0].min = text_block_size.x;
    }
    layout.constraints[1].min = text_block_size.y;
    drop(entry);

    let mut raw_text_entry = scene.entry(&state.raw_text_entity);
    let simple_text = raw_text_entry.get_mut::<SimpleTextC>();

    simple_text.max_width = if state.wrap {
        state.main_obj_size.x
    } else {
        f32::INFINITY
    };

    let uniform_data = raw_text_entry.get_mut::<UniformDataC>();
    uniform_data.copy_from_with_offset(offset_of!(UniformData, clip_rect), rect_data);

    drop(raw_text_entry);
}

fn wrapper_width_calc(entry: &EntityAccess<()>, ctx: &EngineContext, parent_size: &Vec2) -> f32 {
    let state = entry.get::<WrapperState>();
    let text_renderer = ctx.module_mut::<TextRenderer>();
    let block_size = text_renderer.calculate_minimum_text_size(
        state.text.data(),
        state.text.style(),
        if state.wrap { parent_size.x } else { f32::INFINITY },
    );
    block_size.x
}

fn wrapper_height_calc(entry: &EntityAccess<()>, ctx: &EngineContext, parent_size: &Vec2) -> f32 {
    let state = entry.get::<WrapperState>();
    let text_renderer = ctx.module_mut::<TextRenderer>();
    let block_size = text_renderer.calculate_minimum_text_size(
        state.text.data(),
        state.text.style(),
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

        let vertex = renderer
            .device()
            .load_vertex_shader_bundle(
                include_bytes!("../../../res/shaders/text_char.vert.b"),
                &text_renderer::VERTEX_INPUTS,
                "ui_text_char.frag",
            )
            .unwrap();
        let pixel = renderer
            .device()
            .load_pixel_shader_bundle(
                include_bytes!("../../../res/shaders/ui_text_char.frag.b"),
                "ui_text_char.frag",
            )
            .unwrap();

        let mat_pipe_id = text_renderer.register_text_pipeline(&mut renderer, &[vertex, pixel]);

        scene.register_resource(TextImplContext { mat_pipe_id });
    }

    fn create(
        ctx: &EngineContext,
        parent: EntityId,
        text: StyledString,
        wrap: bool,
        h_align: TextHAlign,
    ) -> ObjectEntityId<UIText> {
        let impl_ctx = ctx.scene().resource::<TextImplContext>();

        let main_obj = UIText::new_raw(
            UILayoutC::new()
                .with_width(Sizing::Grow(1.0))
                .with_height(Sizing::FitContent),
            TextState {
                main_obj_size: Vec2::new(f32::INFINITY, 0.0),
                wrapper_entity: Default::default(),
                raw_text_entity: Default::default(),
                text: text.clone(),
                wrap,
                h_align,
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

        let main_entity = ctx.scene().add_object(Some(parent), main_obj).unwrap();
        let wrapper_obj = UIObject::new_raw(
            UILayoutC::new()
                .with_position(Position::Relative(Vec2::new(0.0, 0.0)))
                .with_width(Sizing::ParentBased(wrapper_width_calc))
                .with_height(Sizing::ParentBased(wrapper_height_calc))
                .with_shader_inverted_y(true),
            WrapperState { text, wrap },
        )
        .disable_pointer_events();
        let wrapper_entity = ctx.scene().add_object(Some(*main_entity), wrapper_obj).unwrap();

        let raw_text_entity = ctx
            .scene()
            .add_object(Some(*wrapper_entity), raw_text_obj)
            .unwrap();

        let mut scene = ctx.scene();
        let mut main_obj = scene.object::<UIText>(&main_entity);
        main_obj.get_mut::<TextState>().wrapper_entity = wrapper_entity;
        main_obj.get_mut::<TextState>().raw_text_entity = raw_text_entity;
        main_obj.get_mut::<UIEventHandlerC>().on_size_update = Some(Arc::new(|entity, ctx, new_size| {
            on_size_update(entity, ctx, Some(new_size))
        }));
        drop(main_obj);
        drop(scene);

        on_size_update(&main_entity, ctx, None);

        main_entity
    }

    fn modify<F: FnOnce(&mut TextState)>(entity: &ObjectEntityId<UIText>, ctx: &EngineContext, f: F) {
        let mut scene = ctx.module_mut::<Scene>();
        let mut obj = scene.object(entity);
        f(obj.state_mut());
        drop(obj);
        drop(scene);
        UIText::on_update(entity, ctx, 0.0);
    }
}

impl UITextImpl for UIText {}

pub mod reactive {
    use super::UniformData;
    use crate::game::EngineCtxGameExt;
    use crate::rendering::ui::container::{container, container_props_init};
    use crate::rendering::ui::text::{on_size_update, UIText, UITextImpl};
    use crate::rendering::ui::{UICallbacks, LOCAL_VAR_OPACITY, STATE_ENTITY_ID};
    use common::make_static_id;
    use common::memoffset::offset_of;
    use engine::ecs::component::simple_text::{StyledString, TextHAlign, TextStyle};
    use engine::ecs::component::ui::UILayoutC;
    use engine::ecs::component::UniformDataC;
    use engine::module::scene::Scene;
    use engine::module::ui::reactive::UIScopeContext;
    use engine::module::ui::UIObjectEntityImpl;
    use entity_data::EntityId;

    #[derive(Default, Clone, PartialEq)]
    pub struct UITextProps {
        pub layout: UILayoutC,
        pub callbacks: UICallbacks,
        pub text: String,
        pub align: TextHAlign,
        pub style: TextStyle,
        pub wrap: bool,
    }

    impl UITextProps {
        pub fn layout(mut self, layout: UILayoutC) -> Self {
            self.layout = layout;
            self
        }

        pub fn callbacks(mut self, callbacks: UICallbacks) -> Self {
            self.callbacks = callbacks;
            self
        }

        pub fn style(mut self, style: TextStyle) -> Self {
            self.style = style;
            self
        }

        pub fn align(mut self, align: TextHAlign) -> Self {
            self.align = align;
            self
        }

        pub fn wrap(mut self, wrap: bool) -> Self {
            self.wrap = wrap;
            self
        }
    }

    pub fn ui_text_props(text: impl Into<String>) -> UITextProps {
        UITextProps {
            text: text.into(),
            ..Default::default()
        }
    }

    pub fn ui_text(local_name: &str, ctx: &mut UIScopeContext, props: UITextProps) {
        container(
            local_name,
            ctx,
            container_props_init((props.text, props.style, props.wrap, props.align))
                .layout(props.layout)
                .callbacks(props.callbacks.clone()),
            move |ctx, props| {
                let parent_entity = *ctx.state(STATE_ENTITY_ID).value();
                let parent_opacity = *ctx.local_var::<f32>(LOCAL_VAR_OPACITY, 1.0);

                ctx.descend(
                    make_static_id!(),
                    (props, parent_opacity),
                    move |scope_ctx, ((text, style, wrap, align), parent_opacity)| {
                        let styled_string = StyledString::new(text.clone(), style);

                        let ctx = *scope_ctx.ctx();
                        let entity_state = scope_ctx.request_state(STATE_ENTITY_ID, || {
                            *UIText::create(&ctx, parent_entity, styled_string.clone(), wrap, align)
                        });

                        let opacity = parent_opacity;
                        scope_ctx.set_local_var(LOCAL_VAR_OPACITY, opacity);

                        UIText::modify(&(*entity_state.value()).into(), scope_ctx.ctx(), |state| {
                            state.text = styled_string;
                            state.wrap = wrap;
                            state.h_align = align;
                        });

                        {
                            let mut scene = ctx.scene();

                            let obj = scene.object::<UIText>(&(*entity_state.value()).into());
                            let raw_text_entity = obj.state().raw_text_entity;
                            drop(obj);

                            let mut raw_text = scene.entry(&raw_text_entity);
                            let raw_uniform_data = raw_text.get_mut::<UniformDataC>();
                            UniformDataC::copy_from_with_offset(
                                raw_uniform_data,
                                offset_of!(UniformData, opacity),
                                opacity,
                            );
                        }

                        on_size_update(&entity_state.value(), scope_ctx.ctx(), None);
                    },
                    move |ctx, scope| {
                        let entity = scope.state::<EntityId>(STATE_ENTITY_ID).unwrap();
                        let mut scene = ctx.module_mut::<Scene>();
                        scene.remove_object(&entity);
                    },
                );
            },
        )
    }
}
