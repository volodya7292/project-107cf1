use crate::ecs::component::render_config::RenderStage;
use crate::ecs::component::simple_text::StyledString;
use crate::ecs::component::ui::{UIEventHandlerC, UIEventHandlerI, UILayoutC};
use crate::ecs::component::SimpleTextC;
use crate::ecs::{EntityAccess, SceneAccess};
use crate::module::main_renderer::ui::management::UIState;
use crate::module::text_renderer;
use crate::module::text_renderer::TextRenderer;
use crate::module::ui_renderer::UIObject;
use crate::EngineContext;
use entity_data::EntityId;

pub type UIText = UIObject<SimpleTextC>;

impl UIText {
    pub fn new() -> Self {
        Self::new_raw(
            UILayoutC::new()
                .with_shader_inverted_y(true)
                .with_uniform_crop_rect_offset(text_renderer::ObjectUniformData::clip_rect_offset()),
            SimpleTextC::new()
                .with_max_width(0.0)
                .with_stage(RenderStage::OVERLAY),
        )
        .with_event_handler(UIEventHandlerC::new::<Self>())
    }
}

impl UIEventHandlerI for UIText {
    fn on_hover_enter(_: &EntityId, _: &mut SceneAccess) {
        println!("HOVER ENTER");
    }

    fn on_hover_exit(_: &EntityId, _: &mut SceneAccess) {
        println!("HOVER EXIT");
    }
}

impl UIState for SimpleTextC {}

pub trait TextState {
    fn get_text(&self) -> &StyledString;
    fn set_text(&mut self, text: StyledString);
}

impl<'a> TextState for EntityAccess<'a, UIText> {
    fn get_text(&self) -> &StyledString {
        &self.get::<SimpleTextC>().text
    }

    fn set_text(&mut self, text: StyledString) {
        let ctx = self.context();
        let text_renderer = ctx.module_mut::<TextRenderer>();
        let size = text_renderer.calculate_minimum_text_size(&text);

        self.modify(move |mut access| {
            access.get_mut::<SimpleTextC>().unwrap().text = text;

            let layout = access.get_mut::<UILayoutC>().unwrap();
            layout.constraints[0].min = size.x;
            layout.constraints[1].min = size.y;
        });
    }
}
