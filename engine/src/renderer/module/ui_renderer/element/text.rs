use crate::ecs::component::event_handler::EventHandlerI;
use crate::ecs::component::render_config::RenderStage;
use crate::ecs::component::simple_text::StyledString;
use crate::ecs::component::ui::{Sizing, UILayoutC};
use crate::ecs::component::{ui, EventHandlerC, SimpleTextC};
use crate::renderer::module::text_renderer::TextRenderer;
use crate::renderer::module::ui_renderer::{UIObject, UIState};
use crate::renderer::Renderer;
use entity_data::EntityId;
use nalgebra_glm::Vec2;

pub type UIText = UIObject<SimpleTextC>;

impl UIText {
    pub fn new(text: StyledString) -> Self {
        Self::new_raw(
            UILayoutC::new().with_shader_inverted_y(true),
            SimpleTextC::new(text)
                .with_max_width(0.0)
                .with_stage(RenderStage::OVERLAY),
        )
        .with_event_handler(EventHandlerC::new::<Self>())
    }
}

impl EventHandlerI for UIText {
    fn on_added(entity: &EntityId, renderer: &mut Renderer) {
        let simple_text = renderer.storage.get::<SimpleTextC>(entity).unwrap();
        let text_renderer = renderer.module::<TextRenderer>().unwrap();
        let size = text_renderer.calculate_minimum_text_size(simple_text.string());

        // TODO: set simple_text.max_width according to constraints and parent size.
        //  update it when final size of the element is updated .

        let mut object = renderer.access_object(entity);
        let layout = object.get_mut::<UILayoutC>().unwrap();

        layout.constraints[0].min = size.x;
        layout.constraints[1].min = size.y;
    }
}

impl UIState for SimpleTextC {}
