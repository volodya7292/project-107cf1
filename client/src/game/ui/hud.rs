use super::common::*;
use crate::game::ui::elements::action_button;

pub fn hud_overlay(ctx: &mut UIScopeContext) {
    let is_in_game = ctx.subscribe(&ctx.root_state(&ui_root_states::IN_GAME_PROCESS));
    let vision_obstructed = ctx.subscribe(&ctx.root_state(&ui_root_states::VISION_OBSTRUCTED));
    let player_health = ctx.subscribe(&ctx.root_state(&ui_root_states::PLAYER_HEALTH));

    let player_dead = *is_in_game && *player_health == 0.0;

    let color_filter = if player_dead {
        // death
        Color::new(0.0, 0.0, 0.0, 0.9)
    } else {
        Color::TRANSPARENT
    };

    fn on_respawn(_: &EntityId, ctx: &EngineContext, _: Vec2) {
        let app = ctx.app();
        app.respawn_player();
    }

    container(
        make_static_id!(),
        ctx,
        container_props()
            .layout(
                UILayoutC::column()
                    .with_grow()
                    .with_position(Position::Relative(Vec2::zeros())),
            )
            .background(Some(game_effects(
                color_filter,
                1.0 - *player_health as f32,
                *vision_obstructed,
            )))
            .children_props((*vision_obstructed, player_dead)),
        move |ctx, (vision_obstructed, player_dead)| {
            expander(make_static_id!(), ctx, 1.0);

            if player_dead {
                container(
                    make_static_id!(),
                    ctx,
                    container_props().layout(UILayoutC::column().with_grow()),
                    |ctx, ()| {
                        expander(make_static_id!(), ctx, 1.0);
                        ui_text(
                            make_static_id!(),
                            ctx,
                            ui_text_props("You have come to the dead")
                                .layout(UILayoutC::row().with_align(CrossAlign::Center))
                                .style(TextStyle::new().with_font_size(30.0))
                                .wrap(false),
                        );
                        height_spacer(make_static_id!(), ctx, 100.0);
                        action_button(
                            make_static_id!(),
                            ctx,
                            CrossAlign::Center,
                            "RESPAWN",
                            Arc::new(on_respawn),
                        );
                        expander(make_static_id!(), ctx, 1.0);
                    },
                );
            }

            expander(make_static_id!(), ctx, 0.5);

            container(
                make_static_id!(),
                ctx,
                container_props()
                    .layout(UILayoutC::row().with_width_grow())
                    .children_props(vision_obstructed),
                |ctx, vision_obstructed| {
                    container(
                        make_static_id!(),
                        ctx,
                        container_props().layout(UILayoutC::row().with_width_grow()),
                        |ctx, ()| {
                            let player_health =
                                ctx.subscribe(&ctx.root_state(&ui_root_states::PLAYER_HEALTH));
                            let player_satiety =
                                ctx.subscribe(&ctx.root_state(&ui_root_states::PLAYER_SATIETY));

                            container(
                                make_static_id!(),
                                ctx,
                                container_props()
                                    .layout(UILayoutC::new().with_fixed_size(140.0))
                                    .background(Some(backgrounds::health_indicators(
                                        *player_health as f32,
                                        *player_satiety as f32,
                                    ))),
                                |_, ()| {},
                            )
                        },
                    );

                    if vision_obstructed {
                        ui_text(
                            make_static_id!(),
                            ctx,
                            ui_text_props("Your vision is obstructed!")
                                .layout(
                                    UILayoutC::new()
                                        .with_width_grow()
                                        .with_padding(Padding::hv(0.0, 30.0)),
                                )
                                .style(TextStyle::new().with_font_size(30.0))
                                .align(TextHAlign::Center)
                                .wrap(true),
                        );
                    }

                    let mesh = {
                        let app = ctx.ctx().app();
                        app.test_mesh.clone()
                    };

                    container(
                        make_static_id!(),
                        ctx,
                        container_props()
                            .layout(UILayoutC::new().with_fixed_size(200.0))
                            .background(Some(backgrounds::model_3d(mesh))),
                        |_, ()| {},
                    );

                    expander(make_static_id!(), ctx, 1.0);
                },
            );
        },
    );
}
