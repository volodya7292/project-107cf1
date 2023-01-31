use crate::ecs::component::event_handler::EventHandlerI;
use crate::ecs::component::simple_text::StyledString;
use crate::ecs::component::ui::UILayoutC;
use crate::ecs::component::{ui, EventHandlerC, SimpleTextC};
use crate::renderer::module::text_renderer::TextRenderer;
use crate::renderer::module::ui_renderer::{UIElement, UIObject};
use crate::renderer::Renderer;
use entity_data::EntityId;
use nalgebra_glm::Vec2;

pub type UIText = UIObject<SimpleTextC>;

impl UIText {
    pub fn new(text: StyledString) -> Self {
        Self::new_raw(UILayoutC::new(), SimpleTextC::new(text))
            .with_event_handler(EventHandlerC::new::<Self>())
    }
}

impl EventHandlerI for UIText {
    fn on_added(entity: &EntityId, renderer: &mut Renderer) {
        let simple_text = renderer.storage.get::<SimpleTextC>(entity).unwrap();
        let text_renderer = renderer.module::<TextRenderer>().unwrap();
        let min_width = text_renderer.calculate_minimum_text_width(simple_text.string());

        let mut object = renderer.access_object(entity);

        let layout = object.get_mut::<UILayoutC>().unwrap();
        layout.constraints[0].min = min_width;
    }
}

impl UIElement for SimpleTextC {}
