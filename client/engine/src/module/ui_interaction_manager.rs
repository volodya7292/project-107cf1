use crate::ecs::component::ui::{UIEventHandlerC, UILayoutC};
use crate::event::WSIEvent;
use crate::module::scene::change_manager::{ChangeType, ComponentChangesHandle};
use crate::module::scene::Scene;
use crate::module::ui::UIRenderer;
use crate::module::EngineModule;
use crate::EngineContext;
use entity_data::EntityId;
use winit::event::{ElementState, MouseButton};
use winit::window::Window;

pub struct UIInteractionManager {
    global_event_handler: UIEventHandlerC,
    curr_hover_entity: EntityId,
    mouse_button_press_entity: EntityId,
    active: bool,
    ui_layout_changes: ComponentChangesHandle,
}

impl UIInteractionManager {
    pub fn new(ctx: &EngineContext) -> Self {
        let scene = ctx.module_mut::<Scene>();
        let ui_layout_changes = scene.change_manager_mut().register_component_flow::<UILayoutC>();

        Self {
            global_event_handler: Default::default(),
            curr_hover_entity: Default::default(),
            mouse_button_press_entity: Default::default(),
            active: true,
            ui_layout_changes,
        }
    }

    pub fn set_active(&mut self, active: bool) {
        self.active = active;
    }

    fn on_object_remove(&mut self, id: &EntityId) {
        if id == &self.curr_hover_entity {
            self.curr_hover_entity = EntityId::NULL;
        }
    }
}

impl EngineModule for UIInteractionManager {
    fn on_update(&mut self, _: f64, ctx: &EngineContext) {
        let scene = ctx.module_mut::<Scene>();
        let changes = scene.change_manager_mut().take(self.ui_layout_changes);

        for change in changes.iter().filter(|v| v.ty() == ChangeType::Removed) {
            self.on_object_remove(change.entity());
        }
    }

    fn on_wsi_event(&mut self, _: &Window, event: &WSIEvent, ctx: &EngineContext) {
        if !self.active {
            return;
        }

        let mut scene = ctx.module_mut::<Scene>();
        let ui_renderer = ctx.module_mut::<UIRenderer>();
        let global_handler = &self.global_event_handler;

        match *event {
            WSIEvent::CursorMoved { position, .. } => {
                let mut new_hover_entity = EntityId::NULL;

                ui_renderer.traverse_at_point(&position.logical(), &mut scene, |entry| {
                    let handler = entry.get::<UIEventHandlerC>();
                    if handler.enabled {
                        new_hover_entity = *entry.entity();
                    }
                    handler.enabled
                });

                if self.curr_hover_entity != new_hover_entity {
                    if let Some(handler) = scene
                        .entry_checked(&self.curr_hover_entity)
                        .map(|e| e.get::<UIEventHandlerC>().on_cursor_leave.clone())
                    {
                        ui_invoke_callback_set!(
                            global_handler.on_cursor_leave,
                            &self.curr_hover_entity,
                            &mut scene,
                            ctx,
                        );
                        ui_invoke_callback_set!(handler, &self.curr_hover_entity, &mut scene, ctx);
                    }

                    self.curr_hover_entity = new_hover_entity;

                    if let Some(handler) = scene
                        .entry_checked(&self.curr_hover_entity)
                        .map(|e| e.get::<UIEventHandlerC>().on_cursor_enter.clone())
                    {
                        ui_invoke_callback_set!(
                            global_handler.on_cursor_enter,
                            &self.curr_hover_entity,
                            &mut scene,
                            ctx,
                        );
                        ui_invoke_callback_set!(handler, &self.curr_hover_entity, &mut scene, ctx);
                    }
                }
            }
            WSIEvent::MouseInput { state, button } => {
                if button != MouseButton::Left {
                    return;
                }

                if state == ElementState::Pressed {
                    self.mouse_button_press_entity = self.curr_hover_entity;

                    if let Some(handler) = scene
                        .entry_checked(&self.curr_hover_entity)
                        .map(|e| e.get::<UIEventHandlerC>().on_mouse_press.clone())
                    {
                        ui_invoke_callback_set!(
                            global_handler.on_mouse_press,
                            &self.curr_hover_entity,
                            &mut scene,
                            ctx,
                        );
                        ui_invoke_callback_set!(handler, &self.curr_hover_entity, &mut scene, ctx);
                    }
                } else if state == ElementState::Released {
                    let on_release_entity = self.curr_hover_entity;

                    if self.mouse_button_press_entity == on_release_entity {
                        if let Some(handler) = scene
                            .entry_checked(&self.curr_hover_entity)
                            .map(|e| e.get::<UIEventHandlerC>().on_click.clone())
                        {
                            ui_invoke_callback_set!(
                                global_handler.on_click,
                                &self.curr_hover_entity,
                                &mut scene,
                                ctx,
                            );
                            ui_invoke_callback_set!(handler, &self.curr_hover_entity, &mut scene, ctx);
                        }
                    }

                    if let Some(handler) = scene
                        .entry_checked(&self.curr_hover_entity)
                        .map(|e| e.get::<UIEventHandlerC>().on_mouse_release.clone())
                    {
                        ui_invoke_callback_set!(
                            global_handler.on_mouse_release,
                            &self.curr_hover_entity,
                            &mut scene,
                            ctx,
                        );
                        ui_invoke_callback_set!(handler, &self.curr_hover_entity, &mut scene, ctx);
                    }

                    self.mouse_button_press_entity = EntityId::NULL;
                }
            }
            _ => {}
        }
    }
}