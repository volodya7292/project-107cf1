use crate::game::ui::common::*;

pub fn inventory_slot(
    local_name: &str,
    ctx: &mut UIScopeContext,
    item: ItemVisuals,
    on_click: ClickedCallback,
) {
    container(
        local_name,
        ctx,
        container_props_init(item).layout(UILayoutC::new().with_fixed_size(100.0)),
        move |ctx, item| {
            remember_state!(ctx, hovered, false);

            let on_cursor_enter = {
                let hovered = hovered.state();
                move |_: &EntityId, _: &EngineContext| {
                    hovered.update(true);
                }
            };
            let on_cursor_leave = {
                let hovered = hovered.state();
                move |_: &EntityId, _: &EngineContext| {
                    hovered.update(false);
                }
            };

            container(
                make_static_id!(),
                ctx,
                container_props_init(item)
                    .layout(UILayoutC::new().with_grow().with_padding(Padding::equal(10.0)))
                    .callbacks(
                        ui_callbacks()
                            .on_click(on_click.clone())
                            .on_cursor_enter(Arc::new(on_cursor_enter))
                            .on_cursor_leave(Arc::new(on_cursor_leave)),
                    )
                    .background(Some(backgrounds::item_slot(if *hovered {
                        Color::WHITE
                    } else {
                        Color::WHITE.with_alpha(0.5)
                    }))),
                |ctx, item| {
                    container(
                        make_static_id!(),
                        ctx,
                        container_props()
                            .layout(UILayoutC::new().with_grow())
                            .callbacks(ui_callbacks().interaction(false))
                            .background(Some(item.ui_background().clone())),
                        |_, ()| {},
                    );
                },
            );
        },
    );
}

pub fn inventory_slots(local_name: &str, ctx: &mut UIScopeContext) {
    const ROWS: u32 = 3;
    const COLS: u32 = 5;
    assert_eq!(ROWS * COLS, player::NUM_INTENTORY_ITEM_SLOTS);

    container(
        local_name,
        ctx,
        container_props().layout(UILayoutC::column()),
        |ctx, ()| {
            for i in 0..3 {
                container(
                    &make_static_id!(i),
                    ctx,
                    container_props().layout(UILayoutC::row()),
                    move |ctx, ()| {
                        for j in 0..5 {
                            let item_idx = COLS * i + j;
                            let visuals = {
                                let app = ctx.ctx().app();
                                let item_id = app.main_registry.item_block_default;
                                app.res_map.storage().get_item_visuals(item_id).unwrap().clone()
                            };

                            inventory_slot(
                                &make_static_id!(j),
                                ctx,
                                visuals,
                                Arc::new(move |_, ctx, _| {
                                    let app = ctx.app();
                                    let ui_reactor = app.ui_reactor();
                                    let selected_item_idx = ui_reactor
                                        .root_state(&ui_root_states::SELECTED_INVENTORY_ITEM_IDX)
                                        .unwrap();
                                    selected_item_idx.update(item_idx);
                                }),
                            );

                            if j < 5 - 1 {
                                width_spacer(&make_static_id!(j), ctx, 12.0);
                            }
                        }
                    },
                );
                if i < 4 - 1 {
                    height_spacer(&make_static_id!(i), ctx, 10.0);
                }
            }
        },
    );
}

pub fn game_inventory_overlay(ctx: &mut UIScopeContext) {
    let visible = ctx.subscribe(&ctx.root_state(&ui_root_states::INVENTORY_VISIBLE));
    let opacity = if *visible { 1.0 } else { 0.0 };

    container(
        make_static_id!(),
        ctx,
        container_props()
            .layout(
                UILayoutC::new()
                    .with_grow()
                    .with_position(Position::Relative(Vec2::zeros())),
            )
            .callbacks(ui_callbacks().interaction(*visible))
            .opacity(opacity),
        |ctx, ()| {
            expander(make_static_id!(), ctx, 1.0);
            container(
                make_static_id!(),
                ctx,
                container_props()
                    .layout(
                        UILayoutC::column()
                            .with_align(CrossAlign::Center)
                            .with_padding(Padding::equal(75.0)),
                    )
                    .background(Some(backgrounds::hud_popup(Color::BLACK.with_alpha(0.9))))
                    .corner_radius(75.0),
                |ctx, ()| {
                    ui_text(
                        make_static_id!(),
                        ctx,
                        ui_text_props("My stuff").style(TextStyle::new().with_font_size(36.0)),
                    );
                    height_spacer(make_static_id!(), ctx, 20.0);
                    inventory_slots(make_static_id!(), ctx);
                },
            );
            expander(make_static_id!(), ctx, 1.0);
        },
    );
}
