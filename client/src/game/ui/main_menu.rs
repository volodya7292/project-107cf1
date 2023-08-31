mod world_list;

use super::common::*;
use crate::game::ui::{
    actions::{close_overworld, load_overworld, update_overworlds_list},
    elements::action_button,
    main_menu::world_list::world_selection_list,
};

fn menu_button(local_id: &str, ctx: &mut UIScopeContext, text: &str, on_click: ClickedCallback) {
    fancy_button(
        local_id,
        ctx,
        UILayoutC::new()
            .with_min_width(240.0)
            .with_padding(Padding::hv(12.0, 6.0)),
        StyledString::new(
            text,
            TextStyle::new()
                .with_color(BUTTON_TEXT_COLOR)
                .with_font_size(38.0),
        ),
        on_click,
    );
}

fn main_menu_controls(local_id: &str, ctx: &mut UIScopeContext, curr_tab_state: ReactiveState<&'static str>) {
    let image_source =
        EngineContext::resource_image(&ctx.ctx().scene(), "/textures/main_menu_background.jpg").unwrap();

    fn settings_on_click(entity: &EntityId, ctx: &EngineContext) {}

    fn exit_on_click(_: &EntityId, ctx: &EngineContext) {
        ctx.request_stop();
    }

    ui_image(
        local_id,
        ctx,
        UIImageProps {
            layout: UILayoutC::column()
                .with_width(Sizing::Preferred(400.0))
                .with_height(Sizing::Grow(1.0))
                .with_padding(Padding::equal(30.0)),
            source: Some(ImageSource::Data(image_source)),
            fitness: ImageFitness::Cover,
            ..Default::default()
        },
        move |ctx, ()| {
            let is_in_game = ctx.subscribe(&ctx.root_state(&ui_root_states::IN_GAME_PROCESS));

            expander(make_static_id!(), ctx, 1.0);

            if !*is_in_game {
                world_selection_list(make_static_id!(), ctx);
            }

            height_spacer(make_static_id!(), ctx, 30.0);

            if *is_in_game {
                menu_button(
                    make_static_id!(),
                    ctx,
                    "CONTINUE",
                    Arc::new(move |_: &EntityId, ctx: &EngineContext, _| {
                        ctx.app().show_main_menu(false);
                    }),
                );
            } else {
                let curr_tab_state2 = curr_tab_state.clone();
                menu_button(
                    make_static_id!(),
                    ctx,
                    "NEW OVERWORLD",
                    Arc::new(move |_: &EntityId, ctx: &EngineContext, _| {
                        curr_tab_state2.update(TAB_WORLD_CREATION);
                    }),
                );
            }
            height_spacer(make_static_id!(), ctx, 30.0);

            let curr_tab_state2 = curr_tab_state.clone();
            menu_button(
                make_static_id!(),
                ctx,
                "SETTINGS",
                Arc::new(move |_: &EntityId, ctx: &EngineContext, _| {
                    curr_tab_state2.update(TAB_SETTINGS);
                }),
            );
            height_spacer(make_static_id!(), ctx, 30.0);

            if *is_in_game {
                menu_button(
                    make_static_id!(),
                    ctx,
                    "EXIT OVERWORLD",
                    Arc::new(|_: &EntityId, ctx: &EngineContext, _| {
                        close_overworld(ctx);
                    }),
                );
            } else {
                menu_button(
                    make_static_id!(),
                    ctx,
                    "EXIT",
                    Arc::new(|_: &EntityId, ctx: &EngineContext, _| ctx.request_stop()),
                );
            }

            height_spacer(make_static_id!(), ctx, 100.0);
        },
    );
}

const TAB_WORLD_CREATION: &str = "world_creation";
const TAB_SETTINGS: &str = "settings";
const TABS: [&str; 2] = [TAB_WORLD_CREATION, TAB_SETTINGS];

fn tab_name(tab_id: &str) -> &'static str {
    match tab_id {
        TAB_WORLD_CREATION => "New Overworld",
        TAB_SETTINGS => "Settings",
        _ => "",
    }
}

fn world_creation_view(local_id: &str, ctx: &mut UIScopeContext) {
    container(
        local_id,
        ctx,
        container_props().layout(UILayoutC::column().with_grow()),
        |ctx, ()| {
            remember_state!(ctx, name, "world".to_string());
            remember_state!(ctx, seed, rand::random::<u64>().to_string());
            remember_state!(ctx, error, "".to_string());

            let name_state = name.state();
            let seed_state = seed.state();
            let error_state = error.state();
            let on_proceed = move |_: &EntityId, ctx: &EngineContext, _: Vec2| {
                let mut app = ctx.app();
                let overworld_name = name_state.value();

                if MainApp::make_world_path(&overworld_name).exists() {
                    error_state.update(format!(
                        "Overworld with name \"{}\" already exists!",
                        overworld_name
                    ));
                } else {
                    app.create_overworld(&overworld_name, &seed_state.value());
                    drop(app);
                    update_overworlds_list(ctx);
                    load_overworld(ctx, &overworld_name);
                }
            };

            fancy_text_input(
                make_static_id!(),
                ctx,
                FancyTextInputProps {
                    label: "Name".to_string(),
                    layout: UILayoutC::new().with_width_grow().with_max_width(300.0),
                    multiline: false,
                    text_state: name.state(),
                    style: TextStyle::new().with_font_size(20.0),
                },
            );
            height_spacer(make_static_id!(), ctx, 30.0);
            fancy_text_input(
                make_static_id!(),
                ctx,
                FancyTextInputProps {
                    label: "Seed".to_string(),
                    layout: UILayoutC::new().with_width_grow().with_max_width(300.0),
                    multiline: false,
                    text_state: seed.state(),
                    style: TextStyle::new().with_font_size(20.0),
                },
            );
            expander(make_static_id!(), ctx, 1.0);
            ui_text(
                make_static_id!(),
                ctx,
                ui_text_props(error.clone())
                    .layout(UILayoutC::new().with_width_grow().with_align(CrossAlign::End))
                    .align(TextHAlign::Right)
                    .style(TextStyle::new().with_color(Color::DARK_RED).with_font_size(20.0))
                    .wrap(true),
            );
            height_spacer(make_static_id!(), ctx, 10.0);
            action_button(
                make_static_id!(),
                ctx,
                CrossAlign::End,
                "> PROCEED",
                Arc::new(on_proceed),
            );
        },
    );
}

fn settings_view(local_id: &str, ctx: &mut UIScopeContext) {
    container(
        local_id,
        ctx,
        container_props().layout(UILayoutC::column()),
        |ctx, ()| {
            // ui_text(
            //     make_static_id!(),
            //     ctx,
            //     UITextProps {
            //         text: "SETT sdf sdjf djsa".to_string(),
            //         style: TextStyle::new().with_color(TEXT_COLOR).with_font_size(30.0),
            //         ..Default::default()
            //     },
            // );
        },
    );
}

fn navigation_view(local_id: &str, ctx: &mut UIScopeContext, tab_id: &'static str) {
    let image_source =
        EngineContext::resource_image(&ctx.ctx().scene(), "/textures/main_menu_background.jpg").unwrap();

    if !TABS.into_iter().any(|v| v == tab_id) {
        return;
    }

    ui_image(
        local_id,
        ctx,
        ui_image_props()
            .layout(UILayoutC::column().with_grow().with_padding(Padding::equal(30.0)))
            .source(ImageSource::Data(image_source))
            .fitness(ImageFitness::Cover)
            .children_props((tab_id,)),
        move |ctx, &(tab_id,)| {
            ui_text(
                make_static_id!(),
                ctx,
                ui_text_props(tab_name(tab_id))
                    .style(TextStyle::new().with_color(TAB_TITLE_COLOR).with_font_size(30.0)),
            );
            height_spacer(make_static_id!(), ctx, 30.0);
            match tab_id {
                TAB_WORLD_CREATION => {
                    world_creation_view(make_static_id!(), ctx);
                }
                TAB_SETTINGS => {
                    settings_view(make_static_id!(), ctx);
                }
                _ => {}
            }
            height_spacer(make_static_id!(), ctx, 100.0);
        },
    );
}

pub fn main_menu_overlay(ctx: &mut UIScopeContext) {
    let active_modal_views_state = ctx.root_state(&ui_root_states::ACTIVE_MODAL_VIEWS);

    ctx.once(make_static_id!(), |ctx| {
        ctx.ctx().dispatch_callback(|ctx, _| {
            update_overworlds_list(ctx);
        });
    });

    let player_health = ctx.subscribe(&ctx.root_state(&ui_root_states::PLAYER_HEALTH));
    let is_in_game = ctx.subscribe(&ctx.root_state(&ui_root_states::IN_GAME_PROCESS));

    let menu_visible = {
        let menu_visible = ctx.subscribe(&ctx.root_state(&ui_root_states::MENU_VISIBLE));
        let death_screen_visible = *is_in_game && *player_health == 0.0;
        *menu_visible && !death_screen_visible
    };

    remember_state!(ctx, menu_opacity, AnimatedValue::immediate(0.0));

    let menu_opacity_state = menu_opacity.state();
    ctx.descend(
        make_static_id!(),
        menu_visible,
        move |_, menu_visible| {
            menu_opacity_state.update_with(move |prev| {
                let mut d = *prev;
                let opacity = if menu_visible { 1.0 } else { 0.0 };
                d.retarget(TransitionTarget::new(opacity, 0.07));
                d
            });
        },
        |_, _| {},
    );
    ctx.drive_transition(&menu_opacity);

    // TODO: add background
    container(
        make_static_id!(),
        ctx,
        container_props()
            .layout(
                UILayoutC::row()
                    .with_position(Position::Relative(Vec2::zeros()))
                    .with_grow(),
            )
            .opacity(*menu_opacity.current())
            .callbacks(ui_callbacks().interaction(menu_visible)),
        move |ctx, ()| {
            let curr_nav_view = ctx.subscribe(&ctx.root_state(&ui_root_states::CURR_MENU_TAB));

            expander(make_static_id!(), ctx, 0.2);

            main_menu_controls(make_static_id!(), ctx, curr_nav_view.state());
            width_spacer(make_static_id!(), ctx, 50.0);

            container(
                make_static_id!(),
                ctx,
                container_props_init((*curr_nav_view,)).layout(
                    UILayoutC::new()
                        .with_width(Sizing::Grow(1.5))
                        .with_height(Sizing::Grow(1.0)),
                ),
                move |ctx, (tab_id,)| {
                    navigation_view(make_static_id!(), ctx, tab_id);
                },
            );

            expander(make_static_id!(), ctx, 1.0);
        },
    );

    let active_modal_views = ctx.subscribe(&active_modal_views_state);

    for modal_fn in active_modal_views.iter().cloned() {
        container(
            "modal_wrapper",
            ctx,
            container_props()
                .layout(
                    UILayoutC::column()
                        .with_position(Position::Relative(Vec2::new(0.0, 0.0)))
                        .with_grow(),
                )
                .background(Some(background::solid_color(Color::BLACK.with_alpha(0.4)))),
            move |ctx, ()| {
                let modal_fn = modal_fn.clone();
                expander(make_static_id!(), ctx, 1.0);
                container(
                    make_static_id!(),
                    ctx,
                    container_props().layout(
                        UILayoutC::new()
                            .with_width(Sizing::FitContent)
                            .with_height(Sizing::FitContent)
                            .with_align(CrossAlign::Center),
                    ),
                    move |ctx, ()| {
                        modal_fn(ctx);
                    },
                );
                expander(make_static_id!(), ctx, 1.0);
            },
        );
    }
}
