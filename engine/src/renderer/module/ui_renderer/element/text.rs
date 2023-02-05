use crate::ecs::component::event_handler::EventHandlerI;
use crate::ecs::component::render_config::RenderStage;
use crate::ecs::component::simple_text::StyledString;
use crate::ecs::component::ui::UILayoutC;
use crate::ecs::component::{EventHandlerC, SimpleTextC};
use crate::ecs::EntityAccess;
use crate::renderer::module::text_renderer::TextRenderer;
use crate::renderer::module::ui_renderer::{UIObject, UIState};
use crate::renderer::{RendererContext, RendererContextI};

pub type UIText = UIObject<SimpleTextC>;

impl UIText {
    pub fn new() -> Self {
        Self::new_raw(
            UILayoutC::new().with_shader_inverted_y(true),
            SimpleTextC::new()
                .with_max_width(0.0)
                .with_stage(RenderStage::OVERLAY),
        )
        .with_event_handler(EventHandlerC::new::<Self>())
    }
}

impl EventHandlerI for UIText {}

impl UIState for SimpleTextC {}

pub trait TextState {
    fn get_text(&self) -> &StyledString;
    fn set_text(&mut self, text: StyledString);
}

impl<'a> TextState for EntityAccess<'a, RendererContext<'_>, UIText> {
    fn get_text(&self) -> &StyledString {
        &self.get::<SimpleTextC>().text
    }

    fn set_text(&mut self, text: StyledString) {
        let mut ctx = self.context_mut();
        let text_renderer = ctx.module_mut::<TextRenderer>().unwrap();
        let size = text_renderer.calculate_minimum_text_size(&text);

        self.modify(move |mut access| {
            access.get_mut::<SimpleTextC>().text = text;

            let layout = access.get_mut::<UILayoutC>();
            layout.constraints[0].min = size.x;
            layout.constraints[1].min = size.y;
        });
    }
}
