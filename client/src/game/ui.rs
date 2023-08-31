pub mod common;
mod elements;
mod hud;
mod inventory;
mod main_menu;
mod actions;

use crate::game::ui::{hud::hud_overlay, main_menu::main_menu_overlay};
use common::*;

pub fn overlay_root(ctx: &mut UIScopeContext, root_entity: EntityId) {
    ctx.request_state(STATE_ENTITY_ID, || root_entity);

    ctx.request_state(&*ui_root_states::MENU_VISIBLE, || true);
    ctx.request_state(&*ui_root_states::CURR_MENU_TAB, || "");
    ctx.request_state(&*ui_root_states::WORLD_NAME_LIST, Vec::<String>::new);
    ctx.request_state(&*ui_root_states::ACTIVE_MODAL_VIEWS, Vec::<ModalFn>::new);
    ctx.request_state(&*ui_root_states::PLAYER_HEALTH, || 0.0_f64);
    ctx.request_state(&*ui_root_states::PLAYER_SATIETY, || 0.0_f64);
    ctx.request_state(&*ui_root_states::VISION_OBSTRUCTED, || false);
    ctx.request_state(&*ui_root_states::INVENTORY_VISIBLE, || false);
    ctx.request_state(&*ui_root_states::INVENTORY_ITEMS, Vec::<ItemStack>::new);
    ctx.request_state(&*ui_root_states::SELECTED_INVENTORY_ITEM_IDX, || 0);

    let in_game_process_state = ctx.request_state(&*ui_root_states::IN_GAME_PROCESS, || false);
    let in_game_process = ctx.subscribe(&in_game_process_state);

    ctx.descend(
        make_static_id!(),
        (),
        |ui_ctx, ()| {
            let is_in_game = ui_ctx.subscribe(&ui_ctx.root_state(&ui_root_states::IN_GAME_PROCESS));
            let menu_visible = ui_ctx.subscribe(&ui_ctx.root_state(&ui_root_states::MENU_VISIBLE));
            let player_health = ui_ctx.subscribe(&ui_ctx.root_state(&ui_root_states::PLAYER_HEALTH));
            let inventory_visible = ui_ctx.subscribe(&ui_ctx.root_state(&ui_root_states::INVENTORY_VISIBLE));

            let player_dead = *is_in_game && *player_health == 0.0;
            let cursor_grabbed = !*menu_visible && !player_dead && !*inventory_visible;

            let mut app = ui_ctx.ctx().app();
            app.grab_cursor(&ui_ctx.ctx().window(), cursor_grabbed, ui_ctx.ctx());
        },
        |_, _| {},
    );

    fn on_keyboard(_: &EntityId, ctx: &EngineContext, input: WSIKeyboardInput) {
        let WSIKeyboardInput::Virtual(code, state) = input else {
            return;
        };

        let app = ctx.app();
        let ui_reactor = app.ui_reactor();

        if code == VirtualKeyCode::Tab {
            let inventory_state = ui_reactor.root_state(&ui_root_states::INVENTORY_VISIBLE).unwrap();
            inventory_state.update(state == ElementState::Pressed);
        }
        if code == VirtualKeyCode::Q {
            let selected_item_idx = ui_reactor
                .root_state(&ui_root_states::SELECTED_INVENTORY_ITEM_IDX)
                .unwrap();
            selected_item_idx
                .update_with(|v| (*v as i32 - 1).rem_euclid(NUM_INTENTORY_ITEM_SLOTS as i32) as u32);
        }
        if code == VirtualKeyCode::E {
            let selected_item_idx = ui_reactor
                .root_state(&ui_root_states::SELECTED_INVENTORY_ITEM_IDX)
                .unwrap();
            selected_item_idx
                .update_with(|v| (*v as i32 + 1).rem_euclid(NUM_INTENTORY_ITEM_SLOTS as i32) as u32);
        }
    }

    container(
        make_static_id!(),
        ctx,
        container_props()
            .layout(UILayoutC::new().with_grow())
            .children_props(*in_game_process)
            .callbacks(
                ui_callbacks()
                    .focusable(true)
                    .autofocus(true)
                    .on_keyboard(Arc::new(on_keyboard)),
            ),
        move |ctx, in_game| {
            if in_game {
                hud_overlay(ctx);
                game_inventory_overlay(ctx);
            }
            main_menu_overlay(ctx);
        },
    );
}
