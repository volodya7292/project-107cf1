use crate::ecs::component::ui::{UIEventHandlerC, UILayoutC};
use crate::event::{WSIEvent, WSIKeyboardInput};
use crate::module::scene::change_manager::{ChangeType, ComponentChangesHandle};
use crate::module::scene::Scene;
use crate::module::ui::UIRenderer;
use crate::module::EngineModule;
use crate::utils::wsi::WSIPosition;
use crate::EngineContext;
use common::glm::Vec2;
use common::types::HashSet;
use entity_data::EntityId;
use winit::event::{ElementState, MouseButton};
use winit::window::Window;

pub struct UIInteractionManager {
    curr_hover_entity: EntityId,
    curr_focused_entity: EntityId,
    autofocusable_entities: HashSet<EntityId>,
    mouse_button_press_entity: EntityId,
    curr_mouse_position: Vec2,
    ui_layout_changes: ComponentChangesHandle,
    event_handler_changes: ComponentChangesHandle,
}

fn global_on_cursor_move(ctx: &EngineContext, position: WSIPosition<f32>) {
    let mut scene = ctx.module_mut::<Scene>();
    let ui_renderer = ctx.module_mut::<UIRenderer>();
    let mut new_hover_entity = EntityId::NULL;
    ui_renderer.traverse_at_point(&position.logical(), &mut scene, |entry| {
        let Some(handler) = entry.get_checked::<UIEventHandlerC>() else {
            return false;
        };
        if handler.enabled {
            new_hover_entity = *entry.entity();
        }
        handler.enabled
    });
    drop(ui_renderer);
    drop(scene);

    let mut interaction_manager = ctx.module_mut::<UIInteractionManager>();
    interaction_manager.curr_mouse_position = position.logical();
    interaction_manager.mouse_button_press_entity = new_hover_entity;

    let last_hover_entity = interaction_manager.curr_hover_entity;
    interaction_manager.curr_hover_entity = new_hover_entity;
    drop(interaction_manager);

    if last_hover_entity == new_hover_entity {
        return;
    }

    if last_hover_entity != EntityId::NULL {
        global_on_cursor_leave(&last_hover_entity, ctx);
    }

    if new_hover_entity == EntityId::NULL {
        return;
    }

    let mut scene = ctx.module_mut::<Scene>();
    let entry = scene.entry(&new_hover_entity);
    let ui_handler = entry.get::<UIEventHandlerC>();

    if let Some(handler) = ui_handler.on_cursor_enter.clone() {
        drop(entry);
        drop(scene);
        handler(&new_hover_entity, ctx);
    }
}

fn global_on_cursor_leave(entity: &EntityId, ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let entry = scene.entry(entity);
    let ui_handler = entry.get::<UIEventHandlerC>();

    if let Some(handler) = ui_handler.on_cursor_leave.clone() {
        drop(entry);
        drop(scene);
        handler(entity, ctx);
    }
}

fn global_on_mouse_press(ctx: &EngineContext) {
    let mut scene = ctx.module_mut::<Scene>();
    let mut interaction_manager = ctx.module_mut::<UIInteractionManager>();

    let curr_hover_entity = interaction_manager.curr_hover_entity;
    interaction_manager.mouse_button_press_entity = curr_hover_entity;
    drop(interaction_manager);

    let Some(entry) = scene.entry_checked(&curr_hover_entity) else {
        return;
    };
    let ui_handler = entry.get::<UIEventHandlerC>();

    if let Some(handler) = ui_handler.on_mouse_press.clone() {
        drop(entry);
        drop(scene);
        handler(&curr_hover_entity, ctx);
    }
}

fn global_on_mouse_release(ctx: &EngineContext) {
    let mut interaction_manager = ctx.module_mut::<UIInteractionManager>();

    let on_release_entity = interaction_manager.curr_hover_entity;
    let mouse_button_press_entity = interaction_manager.mouse_button_press_entity;
    let curr_mouse_position = interaction_manager.curr_mouse_position;
    interaction_manager.mouse_button_press_entity = EntityId::NULL;
    drop(interaction_manager);

    if on_release_entity == EntityId::NULL {
        return;
    }

    if mouse_button_press_entity == on_release_entity {
        global_on_click(&on_release_entity, ctx, curr_mouse_position);
    }

    let mut scene = ctx.module_mut::<Scene>();
    let entry = scene.entry(&on_release_entity);
    let ui_handler = entry.get::<UIEventHandlerC>();
    if let Some(handler) = ui_handler.on_mouse_release.clone() {
        drop(entry);
        drop(scene);
        handler(&on_release_entity, ctx);
    }
}

fn global_on_click(entity: &EntityId, ctx: &EngineContext, pos: Vec2) {
    let interaction_manager = ctx.module_mut::<UIInteractionManager>();
    let last_focus_entity = interaction_manager.curr_focused_entity;
    drop(interaction_manager);

    let mut scene = ctx.module_mut::<Scene>();
    let entry = scene.entry(entity);
    let (on_click, on_focus_in, focusable) = {
        let ui_handler = entry.get::<UIEventHandlerC>();
        (
            ui_handler.on_click.clone(),
            ui_handler.on_focus_in.clone(),
            ui_handler.focusable,
        )
    };
    drop(entry);
    drop(scene);

    if let Some(on_click) = &on_click {
        on_click(entity, ctx, pos);
    }

    if *entity != last_focus_entity {
        let mut interaction_manager = ctx.module_mut::<UIInteractionManager>();
        interaction_manager.curr_focused_entity = if focusable { *entity } else { EntityId::NULL };
        drop(interaction_manager);

        if focusable {
            if let Some(on_focus_in) = on_focus_in {
                on_focus_in(entity, ctx);
            }
        }

        if last_focus_entity != EntityId::NULL {
            let mut scene = ctx.module_mut::<Scene>();
            let entry = scene.entry(&last_focus_entity);
            let ui_handler = entry.get::<UIEventHandlerC>();

            if let Some(on_focus_out) = ui_handler.on_focus_out.clone() {
                drop(entry);
                drop(scene);
                on_focus_out(entity, ctx);
            }
        }
    }
}

fn global_on_scroll(ctx: &EngineContext, delta: f64) {
    let curr_mouse_pos = ctx.module_mut::<UIInteractionManager>().curr_mouse_position;

    let scroll_entity = {
        let mut scene = ctx.module_mut::<Scene>();
        let ui_renderer = ctx.module_mut::<UIRenderer>();
        let mut scroll_entity = EntityId::NULL;
        ui_renderer.traverse_at_point(&curr_mouse_pos, &mut scene, |entry| {
            let Some(handler) = entry.get_checked::<UIEventHandlerC>() else {
                return false;
            };
            if handler.enabled && handler.on_scroll.is_some() {
                scroll_entity = *entry.entity();
            }
            handler.enabled
        });
        scroll_entity
    };

    if scroll_entity == EntityId::NULL {
        return;
    }

    let mut scene = ctx.module_mut::<Scene>();
    let entry = scene.entry(&scroll_entity);
    let ui_handler = entry.get::<UIEventHandlerC>();

    if let Some(handler) = ui_handler.on_scroll.clone() {
        drop(entry);
        drop(scene);
        handler(&scroll_entity, ctx, delta);
    }
}

fn manage_focused_entity(ctx: &EngineContext) -> EntityId {
    let last_focused_entity = ctx.module_mut::<UIInteractionManager>().curr_focused_entity;

    if last_focused_entity == EntityId::NULL {
        let new_focused_entity = ctx
            .module_mut::<UIInteractionManager>()
            .autofocusable_entities
            .iter()
            .cloned()
            .next()
            .unwrap_or_default();

        if new_focused_entity != last_focused_entity {
            let mut scene = ctx.module_mut::<Scene>();
            let entry = scene.entry(&new_focused_entity);
            let ui_handler = entry.get::<UIEventHandlerC>();

            if let Some(on_focus_in) = ui_handler.on_focus_in.clone() {
                drop(entry);
                drop(scene);
                on_focus_in(&new_focused_entity, ctx);
            }

            ctx.module_mut::<UIInteractionManager>().curr_focused_entity = new_focused_entity;
        }

        new_focused_entity
    } else {
        last_focused_entity
    }
}

fn global_on_keyboard_input(ctx: &EngineContext, input: WSIKeyboardInput) {
    let focused_entity = manage_focused_entity(ctx);
    if focused_entity == EntityId::NULL {
        return;
    }

    let mut scene = ctx.module_mut::<Scene>();
    let entry = scene.entry(&focused_entity);
    let ui_handler = entry.get::<UIEventHandlerC>();

    if let Some(handler) = ui_handler.on_keyboard.clone() {
        drop(entry);
        drop(scene);
        handler(&focused_entity, ctx, input);
    }
}

impl UIInteractionManager {
    pub fn new(ctx: &EngineContext) -> Self {
        let scene = ctx.module_mut::<Scene>();
        let ui_layout_changes = scene.change_manager_mut().register_component_flow::<UILayoutC>();
        let event_handler_changes = scene
            .change_manager_mut()
            .register_component_flow::<UIEventHandlerC>();

        Self {
            curr_hover_entity: Default::default(),
            curr_focused_entity: Default::default(),
            mouse_button_press_entity: Default::default(),
            curr_mouse_position: Default::default(),
            ui_layout_changes,
            event_handler_changes,
            autofocusable_entities: Default::default(),
        }
    }

    fn on_object_remove(&mut self, id: &EntityId) {
        if *id == self.curr_hover_entity {
            self.curr_hover_entity = EntityId::NULL;
        }
        if *id == self.curr_focused_entity {
            self.curr_focused_entity = EntityId::NULL;
        }
        self.autofocusable_entities.remove(id);
    }
}

impl EngineModule for UIInteractionManager {
    fn on_update(&mut self, _: f64, ctx: &EngineContext) {
        let mut scene = ctx.module_mut::<Scene>();
        let event_handler_changes: Vec<_> = scene.change_manager_mut().take_new(self.event_handler_changes);
        let ui_layout_changes = scene.change_manager_mut().take(self.ui_layout_changes);

        for entity in event_handler_changes {
            let entry = scene.entry(&entity);
            let event_handler = entry.get::<UIEventHandlerC>();

            if event_handler.focusable && event_handler.autofocus {
                self.autofocusable_entities.insert(entity);
            } else {
                self.autofocusable_entities.remove(&entity);
            }
        }

        for change in ui_layout_changes.iter().filter(|v| v.ty() == ChangeType::Removed) {
            self.on_object_remove(change.entity());
        }
    }

    fn on_wsi_event(&mut self, _: &Window, event: &WSIEvent, ctx: &EngineContext) {
        match *event {
            WSIEvent::CursorMoved { position, .. } => {
                ctx.dispatch_callback(move |ctx, _| {
                    global_on_cursor_move(ctx, position);
                });
            }
            WSIEvent::MouseInput { state, button } => {
                if button != MouseButton::Left {
                    return;
                }

                if state == ElementState::Pressed {
                    ctx.dispatch_callback(move |ctx, _| {
                        global_on_mouse_press(ctx);
                    });
                } else if state == ElementState::Released {
                    ctx.dispatch_callback(move |ctx, _| {
                        global_on_mouse_release(ctx);
                    });
                }
            }
            WSIEvent::MouseWheel { delta } => {
                ctx.dispatch_callback(move |ctx, _| {
                    global_on_scroll(ctx, delta);
                });
            }
            WSIEvent::KeyboardInput { input } => {
                ctx.dispatch_callback(move |ctx, _| {
                    global_on_keyboard_input(ctx, input);
                });
            }
            _ => {}
        }
    }
}
