use super::super::common::*;
use crate::game::ui::actions::{load_overworld, pop_modal_view, push_modal_view, update_overworlds_list};

fn world_control_button(local_id: &str, ctx: &mut UIScopeContext, text: &str, on_click: ClickedCallback) {
    fancy_button(
        local_id,
        ctx,
        UILayoutC::new()
            .with_min_height(24.0)
            .with_padding(Padding::hv(8.0, 6.0)),
        StyledString::new(
            text,
            TextStyle::new()
                .with_color(BUTTON_TEXT_COLOR)
                .with_font_size(24.0),
        ),
        on_click,
    );
}

fn confirm_overworld_delete_modal(
    ctx: &mut UIScopeContext,
    overworld_name: String,
    on_dispose: ClickedCallback,
) {
    container(
        make_static_id!(),
        ctx,
        container_props()
            .layout(
                UILayoutC::column()
                    .with_preferred_width(300.0)
                    .with_preferred_height(200.0)
                    .with_padding(Padding::equal(20.0)),
            )
            .background(Some(background::solid_color(Color::BLACK)))
            .corner_radius(10.0)
            .children_props(overworld_name),
        move |ctx, overworld_name| {
            let on_dispose2 = on_dispose.clone();
            let name = overworld_name.clone();
            let on_confirm = Arc::new(move |entity: &EntityId, ctx: &EngineContext, pos: Vec2| {
                let mut app = ctx.app();
                app.delete_overworld(&name);
                println!("Deleted overworld '{}'.", &name);
                drop(app);
                update_overworlds_list(ctx);
                on_dispose2(entity, ctx, pos);
            });

            let on_dispose2 = on_dispose.clone();
            ui_text(
                make_static_id!(),
                ctx,
                ui_text_props(format!(
                    "Do you want to irreversibly delete '{}'?",
                    overworld_name
                ))
                .style(TextStyle::new().with_font_size(24.0)),
            );
            height_spacer(make_static_id!(), ctx, 20.0);
            container(
                make_static_id!(),
                ctx,
                container_props().layout(UILayoutC::row().with_width_grow()),
                move |ctx, ()| {
                    world_control_button(make_static_id!(), ctx, "DELETE", on_confirm.clone());
                    expander(make_static_id!(), ctx, 1.0);
                    world_control_button(make_static_id!(), ctx, "CANCEL", on_dispose2.clone());
                },
            );
        },
    );
}

fn world_item(local_id: &str, ctx: &mut UIScopeContext, overworld_name: String) {
    container(
        local_id,
        ctx,
        container_props()
            .layout(
                UILayoutC::column()
                    .with_width(Sizing::Grow(1.0))
                    .with_padding(Padding::equal(10.0)),
            )
            .background(Some(background::solid_color(Color::WHITE.with_alpha(0.02)))),
        move |ctx, ()| {
            let overworld_name = overworld_name.clone();
            ui_text(
                make_static_id!(),
                ctx,
                ui_text_props(overworld_name.clone()).style(TextStyle::new().with_font_size(24.0)),
            );
            height_spacer(make_static_id!(), ctx, 4.0);
            container(
                make_static_id!(),
                ctx,
                container_props().layout(UILayoutC::row().with_width(Sizing::Grow(1.0))),
                move |ctx, ()| {
                    let name = overworld_name.clone();
                    let on_continue = move |_entity: &EntityId, ctx: &EngineContext, _: Vec2| {
                        load_overworld(ctx, &name);
                    };
                    let name = overworld_name.clone();
                    let on_delete = move |_entity: &EntityId, ctx: &EngineContext, _: Vec2| {
                        let app = ctx.app();
                        let mut reactor = app.ui_reactor();
                        let name = name.clone();

                        push_modal_view(&mut reactor, move |ctx| {
                            confirm_overworld_delete_modal(
                                ctx,
                                name.clone(),
                                Arc::new(|_, ctx, _| {
                                    let app = ctx.app();
                                    let mut reactor = app.ui_reactor();
                                    pop_modal_view(&mut reactor);
                                }),
                            )
                        });
                    };

                    world_control_button(make_static_id!(), ctx, "Continue", Arc::new(on_continue));
                    width_spacer(make_static_id!(), ctx, 20.0);
                    world_control_button(make_static_id!(), ctx, "Delete", Arc::new(on_delete));
                },
            )
        },
    );
}

pub fn world_selection_list(local_id: &str, ctx: &mut UIScopeContext) {
    scrollable_container(
        local_id,
        ctx,
        container_props().layout(
            UILayoutC::column()
                .with_width(Sizing::Grow(1.0))
                .with_height(Sizing::Grow(2.0)),
        ),
        |ctx, ()| {
            let world_names = ctx.subscribe(&ctx.root_state::<Vec<String>>(&ui_root_states::WORLD_NAME_LIST));

            for (i, name) in world_names.iter().enumerate() {
                world_item(&make_static_id!(format!("{}_{}", i, name)), ctx, name.clone());
                height_spacer(&make_static_id!(i), ctx, 10.0);
            }
        },
    );
}
