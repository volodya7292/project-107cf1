use crate::game::{EngineCtxGameExt, MainApp};
use crate::rendering::item_visuals::ItemVisuals;
use crate::rendering::ui::backgrounds::game_effects;
use crate::rendering::ui::container::{
    background, container, container_props, container_props_init, expander, height_spacer, width_spacer,
    ContainerProps,
};
use crate::rendering::ui::fancy_button::fancy_button;
use crate::rendering::ui::fancy_text_input::{fancy_text_input, FancyTextInputProps};
use crate::rendering::ui::image::reactive::{ui_image, ui_image_props, UIImageProps};
use crate::rendering::ui::image::{ImageFitness, ImageSource};
use crate::rendering::ui::scrollable_container::scrollable_container;
use crate::rendering::ui::text::reactive::{ui_text, ui_text_props, UITextProps};
use crate::rendering::ui::{backgrounds, UICallbacks, STATE_ENTITY_ID};
use common::glm::Vec2;
use common::make_static_id;
use common::types::CmpArc;
use engine::ecs::component::simple_text::{StyledString, TextHAlign, TextStyle};
use engine::ecs::component::ui::{ClickedCallback, CrossAlign, Padding, Position, Sizing, UILayoutC};
use engine::event::WSIKeyboardInput;
use engine::module::ui::color::Color;
use engine::module::ui::reactive::{ReactiveState, UIReactor, UIScopeContext};
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::winit::event::{ElementState, VirtualKeyCode};
use engine::{remember_state, EngineContext};
use entity_data::EntityId;
use std::sync::Arc;

const TAB_TITLE_COLOR: Color = Color::rgb(0.5, 1.8, 0.5);
// const BUTTON_TEXT_COLOR: Color = Color::rgb(3.0, 6.0, 3.0);
const BUTTON_TEXT_COLOR: Color = Color::rgb(0.9, 2.0, 0.9);
const TEXT_COLOR: Color = Color::grayscale(0.9);

pub mod ui_root_states {
    use super::ModalFn;
    use engine::module::ui::reactive::StateId;
    use lazy_static::lazy_static;

    lazy_static! {
        pub static ref MENU_VISIBLE: StateId<bool> = "menu_visible".into();
        pub static ref ACTIVE_MODAL_VIEWS: StateId<Vec<ModalFn>> = "curr_modal_view".into();
        pub static ref CURR_MENU_TAB: StateId<&'static str> = "curr_menu_tab".into();
        pub static ref IN_GAME_PROCESS: StateId<bool> = "in_game_process".into();
        pub static ref VISION_OBSTRUCTED: StateId<bool> = "vision_obstructed".into();
        pub static ref PLAYER_HEALTH: StateId<f64> = "player_health".into();
        pub static ref PLAYER_SATIETY: StateId<f64> = "player_satiety".into();
        pub static ref WORLD_NAME_LIST: StateId<Vec<String>> = "world_name_list".into();
        pub static ref INVENTORY_VISIBLE: StateId<bool> = "inventory_visible".into();
    }
}

type ModalFn = CmpArc<dyn Fn(&mut UIScopeContext) + Send + Sync + 'static>;

fn push_modal_view<F>(ctx: &mut UIReactor, view_fn: F)
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

fn pop_modal_view(ctx: &mut UIReactor) {
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

fn action_button(
    local_id: &str,
    ctx: &mut UIScopeContext,
    align: CrossAlign,
    text: &str,
    on_click: ClickedCallback,
) {
    fancy_button(
        local_id,
        ctx,
        UILayoutC::new()
            .with_min_height(24.0)
            .with_padding(Padding::hv(14.0, 10.0))
            .with_align(align),
        StyledString::new(
            text,
            TextStyle::new()
                .with_color(BUTTON_TEXT_COLOR)
                .with_font_size(26.0),
        ),
        on_click,
    );
}

fn update_overworlds_list(ctx: &EngineContext) {
    let app = ctx.app();
    let reactor = app.ui_reactor();
    let world_names = app.get_world_name_list();

    let names_state = reactor.root_state(&ui_root_states::WORLD_NAME_LIST).unwrap();
    names_state.update(world_names);
}

fn load_overworld(ctx: &EngineContext, overworld_name: &str) {
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

fn close_overworld(ctx: &EngineContext) {
    let mut app = ctx.app();
    app.exit_overworld(ctx);

    let reactor = app.ui_reactor();
    reactor
        .root_state(&ui_root_states::IN_GAME_PROCESS)
        .unwrap()
        .update(false);
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
        ContainerProps {
            layout: UILayoutC::column()
                .with_width(Sizing::Grow(1.0))
                .with_padding(Padding::equal(10.0)),
            background: Some(background::solid_color(Color::WHITE.with_alpha(0.02))),
            ..Default::default()
        },
        move |ctx, ()| {
            let overworld_name = overworld_name.clone();
            ui_text(
                make_static_id!(),
                ctx,
                UITextProps {
                    text: overworld_name.clone(),
                    style: TextStyle::new().with_font_size(24.0),
                    ..Default::default()
                },
            );
            height_spacer(make_static_id!(), ctx, 4.0);
            container(
                make_static_id!(),
                ctx,
                ContainerProps {
                    layout: UILayoutC::row().with_width(Sizing::Grow(1.0)),
                    ..Default::default()
                },
                move |ctx, ()| {
                    let name = overworld_name.clone();
                    let on_continue = move |entity: &EntityId, ctx: &EngineContext, _: Vec2| {
                        load_overworld(ctx, &name);
                    };
                    let name = overworld_name.clone();
                    let on_delete = move |entity: &EntityId, ctx: &EngineContext, _: Vec2| {
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

fn world_selection_list(local_id: &str, ctx: &mut UIScopeContext) {
    scrollable_container(
        local_id,
        ctx,
        ContainerProps {
            layout: UILayoutC::column()
                .with_width(Sizing::Grow(1.0))
                .with_height(Sizing::Grow(2.0)),
            ..Default::default()
        },
        |ctx, ()| {
            let world_names = ctx.subscribe(&ctx.root_state::<Vec<String>>(&ui_root_states::WORLD_NAME_LIST));

            for (i, name) in world_names.iter().enumerate() {
                world_item(&make_static_id!(format!("{}_{}", i, name)), ctx, name.clone());
                height_spacer(&make_static_id!(i), ctx, 10.0);
            }
        },
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
const TABS: [&'static str; 2] = [TAB_WORLD_CREATION, TAB_SETTINGS];

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
        ContainerProps {
            layout: UILayoutC::column().with_grow(),
            ..Default::default()
        },
        |ctx, ()| {
            remember_state!(ctx, name, "world".to_string());
            remember_state!(ctx, seed, rand::random::<u64>().to_string());
            remember_state!(ctx, error, "".to_string());

            let name_state = name.state();
            let seed_state = seed.state();
            let error_state = error.state();
            let on_proceed = move |entity: &EntityId, ctx: &EngineContext, _: Vec2| {
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
                UITextProps {
                    layout: UILayoutC::new().with_width_grow().with_align(CrossAlign::End),
                    text: error.clone(),
                    align: TextHAlign::Right,
                    style: TextStyle::new().with_color(Color::DARK_RED).with_font_size(20.0),
                    wrap: true,
                    ..Default::default()
                },
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
        ContainerProps {
            layout: UILayoutC::column(),
            ..Default::default()
        },
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

    if TABS.into_iter().find(|v| *v == tab_id).is_none() {
        return;
    }

    ui_image(
        local_id,
        ctx,
        UIImageProps {
            layout: UILayoutC::column().with_grow().with_padding(Padding::equal(30.0)),
            source: Some(ImageSource::Data(image_source)),
            fitness: ImageFitness::Cover,
            children_props: (tab_id,),
            ..Default::default()
        },
        move |ctx, &(tab_id,)| {
            ui_text(
                make_static_id!(),
                ctx,
                UITextProps {
                    text: tab_name(tab_id).to_string(),
                    style: TextStyle::new().with_color(TAB_TITLE_COLOR).with_font_size(30.0),
                    ..Default::default()
                },
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

fn game_menu(ctx: &mut UIScopeContext) {
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
            .callbacks(UICallbacks::new().with_interaction(menu_visible)),
        move |ctx, ()| {
            let curr_nav_view = ctx.subscribe(&ctx.root_state(&ui_root_states::CURR_MENU_TAB));

            expander(make_static_id!(), ctx, 0.2);

            main_menu_controls(make_static_id!(), ctx, curr_nav_view.state());
            width_spacer(make_static_id!(), ctx, 50.0);

            container(
                make_static_id!(),
                ctx,
                ContainerProps {
                    layout: UILayoutC::new()
                        .with_width(Sizing::Grow(1.5))
                        .with_height(Sizing::Grow(1.0)),
                    children_props: (*curr_nav_view,),
                    ..Default::default()
                },
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
            ContainerProps {
                layout: UILayoutC::column()
                    .with_position(Position::Relative(Vec2::new(0.0, 0.0)))
                    .with_grow(),
                background: Some(background::solid_color(Color::BLACK.with_alpha(0.4))),
                ..Default::default()
            },
            move |ctx, ()| {
                let modal_fn = modal_fn.clone();
                expander(make_static_id!(), ctx, 1.0);
                container(
                    make_static_id!(),
                    ctx,
                    ContainerProps {
                        layout: UILayoutC::new()
                            .with_width(Sizing::FitContent)
                            .with_height(Sizing::FitContent)
                            .with_align(CrossAlign::Center),
                        ..Default::default()
                    },
                    move |ctx, ()| {
                        modal_fn(ctx);
                    },
                );
                expander(make_static_id!(), ctx, 1.0);
            },
        );
    }
}

pub fn game_overlay(ctx: &mut UIScopeContext) {
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

                    expander(make_static_id!(), ctx, 1.0);
                },
            );
        },
    );
}

pub fn inventory_slot(local_name: &str, ctx: &mut UIScopeContext, item: ItemVisuals) {
    container(
        local_name,
        ctx,
        container_props_init(item)
            .layout(
                UILayoutC::new()
                    .with_fixed_size(60.0)
                    .with_padding(Padding::equal(10.0)),
            )
            .background(Some(backgrounds::item_slot(Color::WHITE.with_alpha(0.5)))),
        |ctx, item| {
            container(
                make_static_id!(),
                ctx,
                container_props()
                    .layout(UILayoutC::new().with_grow())
                    .background(Some(item.ui_background().clone())),
                |_, ()| {},
            );
        },
    )
}

pub fn inventory_slots(local_name: &str, ctx: &mut UIScopeContext) {
    container(
        local_name,
        ctx,
        container_props().layout(UILayoutC::column()),
        |ctx, ()| {
            for i in 0..4 {
                container(
                    &make_static_id!(i),
                    ctx,
                    container_props().layout(UILayoutC::row()),
                    |ctx, ()| {
                        for j in 0..5 {
                            let visuals = {
                                let app = ctx.ctx().app();
                                let item_id = app.main_registry.item_block_default;
                                app.res_map.storage().get_item_visuals(item_id).unwrap().clone()
                            };

                            inventory_slot(&make_static_id!(j), ctx, visuals);
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
            .layout(UILayoutC::new().with_grow())
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

    let in_game_process_state = ctx.request_state(&*ui_root_states::IN_GAME_PROCESS, || false);
    let in_game_process = ctx.subscribe(&in_game_process_state);

    ctx.descend(
        make_static_id!(),
        (),
        |ui_ctx, ()| {
            let is_in_game = ui_ctx.subscribe(&ui_ctx.root_state(&ui_root_states::IN_GAME_PROCESS));
            let menu_visible = ui_ctx.subscribe(&ui_ctx.root_state(&ui_root_states::MENU_VISIBLE));
            let player_health = ui_ctx.subscribe(&ui_ctx.root_state(&ui_root_states::PLAYER_HEALTH));

            let player_dead = *is_in_game && *player_health == 0.0;
            let cursor_grabbed = !*menu_visible && !player_dead;

            let mut app = ui_ctx.ctx().app();
            app.grab_cursor(&ui_ctx.ctx().window(), cursor_grabbed, ui_ctx.ctx());
        },
        |_, _| {},
    );

    fn on_keyboard(_: &EntityId, ctx: &EngineContext, input: WSIKeyboardInput) {
        let WSIKeyboardInput::Virtual(code, state) = input else {
            return;
        };
        if code == VirtualKeyCode::Tab {
            let app = ctx.app();
            let ui_reactor = app.ui_reactor();
            let inventory_state = ui_reactor.root_state(&ui_root_states::INVENTORY_VISIBLE).unwrap();
            inventory_state.update(state == ElementState::Pressed);
        }
    }

    container(
        make_static_id!(),
        ctx,
        container_props()
            .layout(UILayoutC::new().with_grow())
            .children_props(*in_game_process)
            .callbacks(
                UICallbacks::new()
                    .with_focusable(true)
                    .with_autofocus(true)
                    .with_on_keyboard(Arc::new(on_keyboard)),
            ),
        move |ctx, in_game| {
            if in_game {
                game_overlay(ctx);
                game_inventory_overlay(ctx);
            }
            game_menu(ctx);
        },
    );
}
