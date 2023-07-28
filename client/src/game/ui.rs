use crate::game::EngineCtxGameExt;
use crate::rendering::ui::container::{
    background, container, expander, height_spacer, width_spacer, ContainerProps,
};
use crate::rendering::ui::fancy_button::fancy_button;
use crate::rendering::ui::fancy_text_input::{fancy_text_input, FancyTextInputProps};
use crate::rendering::ui::image::reactive::ui_image;
use crate::rendering::ui::image::{ImageFitness, ImageSource};
use crate::rendering::ui::text::reactive::{ui_text, UITextProps};
use crate::rendering::ui::{UIContext, STATE_ENTITY_ID};
use common::glm::Vec2;
use common::make_static_id;
use engine::ecs::component::simple_text::{StyledString, TextStyle};
use engine::ecs::component::ui::{
    BasicEventCallback, ClickedCallback, CrossAlign, Padding, Sizing, UILayoutC, Visibility,
};
use engine::module::ui::color::Color;
use engine::module::ui::reactive::{ReactiveState, UIScopeContext};
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::{remember_state, EngineContext};
use entity_data::EntityId;

const TAB_TITLE_COLOR: Color = Color::rgb(0.5, 1.8, 0.5);
// const BUTTON_TEXT_COLOR: Color = Color::rgb(3.0, 6.0, 3.0);
const BUTTON_TEXT_COLOR: Color = Color::rgb(0.8, 2.0, 0.8);
const TEXT_COLOR: Color = Color::grayscale(0.9);

fn menu_button(local_id: &str, ctx: &mut UIScopeContext, text: &str, on_click: impl ClickedCallback) {
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

fn world_control_button(
    local_id: &str,
    ctx: &mut UIScopeContext,
    text: &str,
    on_click: impl ClickedCallback,
) {
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

fn action_button(local_id: &str, ctx: &mut UIScopeContext, text: &str, on_click: impl ClickedCallback) {
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

fn world_item(local_id: &str, ctx: &mut UIScopeContext, name: String) {
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
        move |ctx| {
            ui_text(
                make_static_id!(),
                ctx,
                UITextProps {
                    text: name.clone(),
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
                |ctx| {
                    world_control_button(make_static_id!(), ctx, "Continue", |entity, ctx, _| {});
                    width_spacer(make_static_id!(), ctx, 20.0);
                    world_control_button(make_static_id!(), ctx, "Delete", |entity, ctx, _| {});
                },
            )
        },
    );
}

fn world_selection_list(ctx: &mut UIScopeContext) {
    container(
        make_static_id!(),
        ctx,
        ContainerProps {
            layout: UILayoutC::column()
                .with_width(Sizing::Grow(1.0))
                .with_height(Sizing::FitContent),
            ..Default::default()
        },
        |ctx| {
            let mut ui_ctx = UIContext::new(*ctx.ctx());
            let world_names = ui_ctx.app().get_world_name_list();
            drop(ui_ctx);

            for (i, name) in world_names.iter().enumerate() {
                world_item(&make_static_id!(i), ctx, name.clone());
                height_spacer(&make_static_id!(i), ctx, 10.0);
            }
        },
    );
}

fn main_menu_controls(local_id: &str, ctx: &mut UIScopeContext, curr_tab_state: ReactiveState<&'static str>) {
    let resources = ctx.ctx().resources();

    let image_source = UIContext::resource_image(&resources, "/textures/main_menu_background.jpg")
        .unwrap()
        .unwrap();

    fn settings_on_click(entity: &EntityId, ctx: &EngineContext) {}

    fn exit_on_click(_: &EntityId, ctx: &EngineContext) {
        ctx.request_stop();
    }

    ui_image(
        local_id,
        ctx,
        UILayoutC::column()
            .with_width(Sizing::Preferred(400.0))
            .with_height(Sizing::Grow(1.0))
            .with_padding(Padding::equal(30.0)),
        Some(ImageSource::Data(image_source)),
        ImageFitness::Cover,
        move |ctx| {
            expander(make_static_id!(), ctx, 1.0);

            world_selection_list(ctx);

            height_spacer(make_static_id!(), ctx, 30.0);

            let curr_tab_state2 = curr_tab_state.clone();
            menu_button(
                make_static_id!(),
                ctx,
                "START",
                move |_: &EntityId, ctx: &EngineContext, _| {
                    curr_tab_state2.update(TAB_WORLD_CREATION);
                },
            );
            height_spacer(make_static_id!(), ctx, 30.0);

            let curr_tab_state2 = curr_tab_state.clone();
            menu_button(
                make_static_id!(),
                ctx,
                "SETTINGS",
                move |_: &EntityId, ctx: &EngineContext, _| {
                    curr_tab_state2.update(TAB_SETTINGS);
                },
            );
            height_spacer(make_static_id!(), ctx, 30.0);

            menu_button(
                make_static_id!(),
                ctx,
                "EXIT",
                move |_: &EntityId, ctx: &EngineContext, _| ctx.request_stop(),
            );

            expander(make_static_id!(), ctx, 0.5);
        },
    );
}

const TAB_WORLD_CREATION: &str = "world_creation";
const TAB_SETTINGS: &str = "settings";

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
        |ctx| {
            remember_state!(ctx, text, "world".to_string());
            remember_state!(ctx, seed, rand::random::<u64>().to_string());

            fn on_proceed(entity: &EntityId, ctx: &EngineContext, _: Vec2) {}

            fancy_text_input(
                make_static_id!(),
                ctx,
                FancyTextInputProps {
                    label: "Name".to_string(),
                    layout: UILayoutC::new().with_width_grow().with_max_width(300.0),
                    multiline: false,
                    text_state: text.state().clone(),
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
                    text_state: seed.state().clone(),
                    style: TextStyle::new().with_font_size(20.0),
                },
            );
            expander(make_static_id!(), ctx, 1.0);
            action_button(make_static_id!(), ctx, "> PROCEED", on_proceed);
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
        |ctx| {
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
    let resources = ctx.ctx().resources();
    let image_source = UIContext::resource_image(&resources, "/textures/main_menu_background.jpg")
        .unwrap()
        .unwrap();

    ui_image(
        local_id,
        ctx,
        UILayoutC::column().with_grow().with_padding(Padding::equal(30.0)),
        Some(ImageSource::Data(image_source)),
        ImageFitness::Cover,
        move |ctx| {
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

pub mod ui_root_states {
    pub const MENU_VISIBLE: &'static str = "menu_visible";
}

pub fn ui_root(ctx: &mut UIScopeContext, root_entity: EntityId) {
    let menu_visible = ctx.request_state(ui_root_states::MENU_VISIBLE, || true);
    let menu_visible = ctx.subscribe(&menu_visible);

    ctx.request_state(STATE_ENTITY_ID, || root_entity);

    remember_state!(ctx, menu_opacity, AnimatedValue::immediate(0.0));

    let menu_opacity2 = menu_opacity.state().clone();
    ctx.descend(
        make_static_id!(),
        move |ctx| {
            let menu_visible = ctx.subscribe(menu_visible.state()).state().clone();
            menu_opacity2.update_with(move |prev| {
                let mut d = *prev;
                let opacity = if *menu_visible.value() { 1.0 } else { 0.0 };
                d.retarget(TransitionTarget::new(opacity, 0.07));
                d
            });
        },
        |_, _| {},
        false,
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
        move |ctx| {
            remember_state!(ctx, curr_nav_view, "");

            expander(make_static_id!(), ctx, 0.2);

            main_menu_controls(make_static_id!(), ctx, curr_nav_view.state().clone());
            width_spacer(make_static_id!(), ctx, 50.0);

            container(
                make_static_id!(),
                ctx,
                ContainerProps {
                    layout: UILayoutC::new()
                        .with_width(Sizing::Grow(1.5))
                        .with_height(Sizing::Grow(1.0)),
                    ..Default::default()
                },
                move |ctx| {
                    navigation_view(make_static_id!(), ctx, *curr_nav_view);
                },
            );

            expander(make_static_id!(), ctx, 1.0);
        },
    );
}
