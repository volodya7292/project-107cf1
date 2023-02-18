use crate::ecs::component::render_config::RenderStage;
use crate::ecs::component::simple_text::StyledString;
use crate::ecs::component::ui::{UIEventHandlerC, UIEventHandlerI, UILayoutC};
use crate::ecs::component::{SceneEventHandler, SimpleTextC};
use crate::module::scene::{EntityAccess, Scene};
use crate::module::text_renderer;
use crate::module::text_renderer::TextRenderer;
use crate::module::ui::management::UIState;
use crate::module::ui::UIObject;
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
        .with_scene_event_handler(SceneEventHandler::new().with_on_update(Self::on_update))
        .add_event_handler(UIEventHandlerC::from_impl::<Self>())
    }

    fn on_update(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext) {
        let mut entry = scene.entry(entity);

        let simple_text = entry.get::<SimpleTextC>();
        let text_renderer = ctx.module_mut::<TextRenderer>();
        let size = text_renderer.calculate_minimum_text_size(&simple_text.text);

        let layout = entry.get_mut::<UILayoutC>();
        layout.constraints[0].min = size.x;
        layout.constraints[1].min = size.y;

        let scene_handler = entry.get_mut::<SceneEventHandler>();
        scene_handler.set_on_update_active(false);
    }
}

impl UIEventHandlerI for UIText {
    fn on_cursor_enter(_: &EntityId, _: &mut Scene, _: &EngineContext) {
        println!("HOVER ENTER");
    }

    fn on_cursor_leave(_: &EntityId, _: &mut Scene, _: &EngineContext) {
        println!("HOVER EXIT");
    }

    fn on_click(_: &EntityId, _: &mut Scene, _: &EngineContext) {
        println!("ON CLICK");
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
        let simple_text = self.get_mut_checked::<SimpleTextC>().unwrap();
        simple_text.text = text;

        let scene_handler = self.get_mut_checked::<SceneEventHandler>().unwrap();
        scene_handler.set_on_update_active(true);
    }
}
