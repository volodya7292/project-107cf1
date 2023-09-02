use super::common::*;

pub fn debug_overlay(ctx: &mut UIScopeContext) {
    let visible = ctx.subscribe(&ctx.root_state(&ui_root_states::DEBUG_INFO_VISIBLE));
    let opacity = if *visible { 1.0 } else { 0.0 };

    container(
        make_static_id!(),
        ctx,
        container_props()
            .layout(
                UILayoutC::column()
                    .with_grow()
                    .with_position(Position::Relative(Vec2::zeros())),
            )
            .callbacks(ui_callbacks().interaction(false))
            .opacity(opacity),
        |ctx, ()| {
            let debug_info = ctx.subscribe(&ctx.root_state(&ui_root_states::DEBUG_INFO));

            for (idx, str_info) in debug_info.iter().enumerate() {
                ui_text(
                    &make_static_id!(idx),
                    ctx,
                    ui_text_props(str_info.clone())
                        .layout(UILayoutC::new().with_width_grow())
                        .style(TextStyle::new().with_font_size(20.0))
                        .wrap(true),
                );
            }
        },
    );
}
