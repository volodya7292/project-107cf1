use crate::ecs::component::ui::UIEventHandlerC;
use crate::event::WSIEvent;
use crate::module::ui::UIRenderer;
use crate::module::EngineModule;
use crate::EngineContext;
use entity_data::EntityId;
use std::any::Any;
use winit::event::{ElementState, MouseButton};

pub struct UIInteractionManager {
    ctx: EngineContext,
    global_event_handler: UIEventHandlerC,
    curr_hover_entity: EntityId,
    mouse_button_press_entity: EntityId,
}

impl UIInteractionManager {
    pub fn new(ctx: EngineContext) -> Self {
        Self {
            ctx,
            global_event_handler: Default::default(),
            curr_hover_entity: EntityId::NULL,
            mouse_button_press_entity: EntityId::NULL,
        }
    }
}

impl EngineModule for UIInteractionManager {
    fn on_object_remove(&mut self, id: &EntityId) {
        if id == &self.curr_hover_entity {
            self.curr_hover_entity = EntityId::NULL;
        }
    }

    fn on_wsi_event(&mut self, event: &WSIEvent) {
        let mut scene = self.ctx.scene();
        let ui_renderer = self.ctx.module_mut::<UIRenderer>();

        match *event {
            WSIEvent::CursorMoved { position, .. } => {
                let new_hover_entity = ui_renderer
                    .find_element_at_point(&position.logical(), &mut scene)
                    .unwrap_or(EntityId::NULL);

                if self.curr_hover_entity != new_hover_entity {
                    if let Some(entry) = scene.object_raw(&self.curr_hover_entity) {
                        let on_hover_exit = entry.get::<UIEventHandlerC>().on_cursor_leave;
                        on_hover_exit(&self.curr_hover_entity, &mut scene);
                    }

                    if let Some(entry) = scene.object_raw(&new_hover_entity) {
                        let on_hover_enter = entry.get::<UIEventHandlerC>().on_cursor_enter;
                        on_hover_enter(&new_hover_entity, &mut scene);
                    }

                    self.curr_hover_entity = new_hover_entity;
                }
            }
            WSIEvent::MouseInput { state, button } => {
                if button != MouseButton::Left {
                    return;
                }
                if state == ElementState::Pressed {
                    self.mouse_button_press_entity = self.curr_hover_entity;

                    if let Some(entry) = scene.object_raw(&self.curr_hover_entity) {
                        let on_mouse_press = entry.get::<UIEventHandlerC>().on_mouse_press;
                        on_mouse_press(&self.curr_hover_entity, &mut scene);
                    }
                } else if state == ElementState::Released {
                    let on_release_entity = self.curr_hover_entity;

                    if self.mouse_button_press_entity == on_release_entity {
                        if let Some(entry) = scene.object_raw(&on_release_entity) {
                            let on_click = entry.get::<UIEventHandlerC>().on_click;
                            on_click(&on_release_entity, &mut scene);
                        }
                    }

                    if let Some(entry) = scene.object_raw(&on_release_entity) {
                        let on_mouse_release = entry.get::<UIEventHandlerC>().on_mouse_release;
                        on_mouse_release(&on_release_entity, &mut scene);
                    }

                    self.mouse_button_press_entity = EntityId::NULL;
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
