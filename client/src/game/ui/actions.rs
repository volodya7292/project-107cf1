use super::common::*;

pub fn push_modal_view<F>(ctx: &mut UIReactor, view_fn: F)
where
    F: Fn(&mut UIScopeContext) + Send + Sync + 'static,
{
    let views_state = ctx.root_state(&ui_root_states::ACTIVE_MODAL_VIEWS).unwrap();
    views_state.update_with(move |prev| {
        let mut new = prev.clone();
        new.push(CmpArc(Arc::new(view_fn)));
        new
    });
}

pub fn pop_modal_view(ctx: &mut UIReactor) {
    let views_state = ctx.root_state(&ui_root_states::ACTIVE_MODAL_VIEWS).unwrap();
    views_state.update_with(move |prev| {
        let mut new = prev.clone();
        new.pop().unwrap();
        // let idx = new
        //     .iter()
        //     .position(|v| *v as *const ModalFn == view_fn as *const ModalFn)
        //     .unwrap();
        // new.remove(idx);
        new
    });
}

pub fn update_overworlds_list(ctx: &EngineContext) {
    let app = ctx.app();
    let reactor = app.ui_reactor();
    let world_names = app.get_world_name_list();

    let names_state = reactor.root_state(&ui_root_states::WORLD_NAME_LIST).unwrap();
    names_state.update(world_names);
}

pub fn load_overworld(ctx: &EngineContext, overworld_name: &str) {
    let mut app = ctx.app();
    app.enter_overworld(ctx, overworld_name);
    app.show_main_menu(false);

    let reactor = app.ui_reactor();
    reactor
        .root_state(&ui_root_states::CURR_MENU_TAB)
        .unwrap()
        .update("");
    reactor
        .root_state(&ui_root_states::IN_GAME_PROCESS)
        .unwrap()
        .update(true);
    // This disables death screen when health = 0
    reactor
        .root_state(&ui_root_states::PLAYER_HEALTH)
        .unwrap()
        .update(1.0_f64);
    reactor
        .root_state(&ui_root_states::PLAYER_SATIETY)
        .unwrap()
        .update(1.0_f64);
}

pub fn close_overworld(ctx: &EngineContext) {
    let mut app = ctx.app();
    app.exit_overworld(ctx);

    let reactor = app.ui_reactor();
    reactor
        .root_state(&ui_root_states::IN_GAME_PROCESS)
        .unwrap()
        .update(false);
}
