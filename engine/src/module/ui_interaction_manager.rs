use crate::ecs::component::ui::UIEventHandlerC;
use crate::event::{Event, WindowEvent};
use crate::module::ui_renderer::UIRenderer;
use crate::module::EngineModule;
use crate::EngineContext;
use entity_data::EntityId;
use std::any::Any;

pub struct UIInteractionManager {
    ctx: EngineContext,
    global_event_handler: UIEventHandlerC,
    curr_hover_entity: EntityId,
}

impl UIInteractionManager {
    pub fn new(ctx: EngineContext) -> Self {
        Self {
            ctx,
            global_event_handler: Default::default(),
            curr_hover_entity: EntityId::NULL,
        }
    }
}

impl EngineModule for UIInteractionManager {
    fn on_object_remove(&mut self, id: &EntityId) {
        if id == &self.curr_hover_entity {
            self.curr_hover_entity = EntityId::NULL;
        }
    }

    fn on_event(&mut self, event: &Event) {
        let Event::WindowEvent { event: win_event, .. } = event else {
            return;
        };

        let mut scene = self.ctx.scene();
        let ui_renderer = self.ctx.module_mut::<UIRenderer>();

        match win_event {
            WindowEvent::CursorMoved { position, .. } => {
                let new_hover_entity = ui_renderer
                    .find_element_at_point(&position.logical(), &mut scene)
                    .unwrap_or(EntityId::NULL);

                if self.curr_hover_entity != new_hover_entity {
                    if let Some(entry) = scene.object_raw(&self.curr_hover_entity) {
                        let on_hover_exit = entry.get::<UIEventHandlerC>().on_hover_exit;
                        on_hover_exit(&self.curr_hover_entity, &mut scene);
                    }

                    if let Some(entry) = scene.object_raw(&new_hover_entity) {
                        let on_hover_enter = entry.get::<UIEventHandlerC>().on_hover_enter;
                        on_hover_enter(&new_hover_entity, &mut scene);
                    }

                    self.curr_hover_entity = new_hover_entity;
                }
            }
            _ => {}
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
