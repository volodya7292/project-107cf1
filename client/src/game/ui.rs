use crate::game::{EngineCtxGameExt, MainApp};
use crate::rendering::ui::container::{
    background, container, container_props, expander, height_spacer, width_spacer, ContainerProps,
};
use crate::rendering::ui::fancy_button::fancy_button;
use crate::rendering::ui::fancy_text_input::{fancy_text_input, FancyTextInputProps};
use crate::rendering::ui::image::reactive::{ui_image, UIImageProps};
use crate::rendering::ui::image::{ImageFitness, ImageSource};
use crate::rendering::ui::text::reactive::{ui_text, ui_text_props, UITextProps};
use crate::rendering::ui::STATE_ENTITY_ID;
use common::glm::Vec2;
use common::make_static_id;
use engine::ecs::component::simple_text::{StyledString, TextHAlign, TextStyle};
use engine::ecs::component::ui::{
    ClickedCallback, CrossAlign, Padding, Position, Sizing, UILayoutC, Visibility,
};
use engine::module::ui::color::Color;
use engine::module::ui::reactive::{ReactiveState, UIReactor, UIScopeContext};
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::{remember_state, EngineContext};
use entity_data::EntityId;
use std::sync::Arc;

const TAB_TITLE_COLOR: Color = Color::rgb(0.5, 1.8, 0.5);
// const BUTTON_TEXT_COLOR: Color = Color::rgb(3.0, 6.0, 3.0);
const BUTTON_TEXT_COLOR: Color = Color::rgb(0.8, 2.0, 0.8);
const TEXT_COLOR: Color = Color::grayscale(0.9);

pub mod ui_root_states {
    pub const MENU_VISIBLE: &'static str = "menu_visible";
    pub const ACTIVE_MODAL_VIEWS: &'static str = "curr_modal_view";

    pub const CURR_MENU_TAB: &'static str = "curr_menu_tab";
    pub const IN_GAME_PROCESS: &'static str = "in_game_process";
    pub const WORLD_NAME_LIST: &'static str = "world_name_list";
}

type ModalFn = Arc<dyn Fn(&mut UIScopeContext) + Send + Sync + 'static>;

fn push_modal_view<F>(ctx: &mut UIReactor, view_fn: F)
where
    F: Fn(&mut UIScopeContext) + Send + Sync + 'static,
{
    let views_state = ctx
        .root_state::<Vec<ModalFn>>(ui_root_states::ACTIVE_MODAL_VIEWS)
        .unwrap();
    views_state.update_with(move |prev| {
        let mut new = prev.clone();
        new.push(Arc::new(view_fn));
        new
    });
}

fn pop_modal_view(ctx: &mut UIReactor) {
    let views_state = ctx
        .root_state::<Vec<ModalFn>>(ui_root_states::ACTIVE_MODAL_VIEWS)
        .unwrap();
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

fn action_button(local_id: &str, ctx: &mut UIScopeContext, text: &str, on_click: ClickedCallback) {
    fancy_button(
        local_id,
        ctx,
        UILayoutC::new()
            .with_min_height(24.0)
            .with_padding(Padding::hv(14.0, 10.0))
            .with_align(CrossAlign::End),
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

    let names_state = reactor
        .root_state::<Vec<String>>(ui_root_states::WORLD_NAME_LIST)
        .unwrap();
    names_state.update(world_names);
}

fn load_overworld(ctx: &EngineContext, overworld_name: &str) {
    let mut app = ctx.app();
    app.start_game_process(ctx, overworld_name);

    let reactor = app.ui_reactor();
    reactor
        .root_state(ui_root_states::MENU_VISIBLE)
        .unwrap()
        .update(false);
    reactor
        .root_state(ui_root_states::CURR_MENU_TAB)
        .unwrap()
        .update("");
    reactor
        .root_state(ui_root_states::IN_GAME_PROCESS)
        .unwrap()
        .update(true);
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
                .with_style(TextStyle::new().with_font_size(24.0)),
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
    container(
        local_id,
        ctx,
        ContainerProps {
            layout: UILayoutC::column()
                .with_width(Sizing::Grow(1.0))
                .with_height(Sizing::FitContent),
            ..Default::default()
        },
        |ctx, ()| {
            let world_names = ctx.subscribe(&ctx.root_state::<Vec<String>>(ui_root_states::WORLD_NAME_LIST));

            for (i, name) in world_names.iter().enumerate() {
                world_item(&make_static_id!(i), ctx, name.clone());
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
            let is_in_game = ctx.subscribe(&ctx.root_state::<bool>(ui_root_states::IN_GAME_PROCESS));

            expander(make_static_id!(), ctx, 1.0);

            if !*is_in_game {
                world_selection_list(make_static_id!(), ctx);
            }

            height_spacer(make_static_id!(), ctx, 30.0);

            if !*is_in_game {
                let curr_tab_state2 = curr_tab_state.clone();
                menu_button(
                    make_static_id!(),
                    ctx,
                    "START",
                    Arc::new(move |_: &EntityId, ctx: &EngineContext, _| {
                        curr_tab_state2.update(TAB_WORLD_CREATION);
                    }),
                );
                height_spacer(make_static_id!(), ctx, 30.0);
            }

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

            menu_button(
                make_static_id!(),
                ctx,
                "EXIT",
                Arc::new(move |_: &EntityId, ctx: &EngineContext, _| ctx.request_stop()),
            );

            expander(make_static_id!(), ctx, 0.5);
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

                if MainApp::make_world_path(overworld_name).exists() {
                    error_state.update(format!(
                        "Overworld with name \"{}\" already exists!",
                        overworld_name
                    ));
                } else {
                    app.create_overworld(overworld_name, seed_state.value());
                    drop(app);
                    load_overworld(ctx, overworld_name);
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
            action_button(make_static_id!(), ctx, "> PROCEED", Arc::new(on_proceed));
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
            ui_text(
                make_static_id!(),
                ctx,
                UITextProps {
                    text: "SETT sdf sdjf djsa".to_string(),
                    style: TextStyle::new().with_color(TEXT_COLOR).with_font_size(30.0),
                    ..Default::default()
                },
            )
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

pub fn ui_root(ctx: &mut UIScopeContext, root_entity: EntityId) {
    ctx.request_state(STATE_ENTITY_ID, || root_entity);

    ctx.request_state(ui_root_states::CURR_MENU_TAB, || "");
    ctx.request_state(ui_root_states::IN_GAME_PROCESS, || false);
    let active_modal_views_state =
        ctx.request_state(ui_root_states::ACTIVE_MODAL_VIEWS, || Vec::<ModalFn>::new());
    ctx.request_state(ui_root_states::WORLD_NAME_LIST, || Vec::<String>::new());

    ctx.once(make_static_id!(), |ctx| {
        ctx.ctx().dispatch_callback(|ctx, _| {
            update_overworlds_list(ctx);
        });
    });

    let menu_visible = ctx.request_state(ui_root_states::MENU_VISIBLE, || true);
    let menu_visible = ctx.subscribe(&menu_visible);
    remember_state!(ctx, menu_opacity, AnimatedValue::immediate(0.0));

    let menu_opacity_state = menu_opacity.state();
    ctx.descend(
        make_static_id!(),
        *menu_visible,
        move |ctx, menu_visible| {
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

    container(
        make_static_id!(),
        ctx,
        ContainerProps {
            layout: UILayoutC::row()
                .with_grow()
                .with_visibility(Visibility::Opacity(*menu_opacity)),
            ..Default::default()
        },
        move |ctx, ()| {
            let curr_nav_view = ctx.subscribe(&ctx.root_state::<&str>(ui_root_states::CURR_MENU_TAB));

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
