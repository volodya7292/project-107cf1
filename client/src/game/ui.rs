use crate::game::EngineCtxGameExt;
use crate::game::MainApp;
use crate::rendering::ui::container::{
    background, container, expander, height_spacer, width_spacer, ContainerProps,
};
use crate::rendering::ui::fancy_button::fancy_button;
use crate::rendering::ui::image::reactive::ui_image;
use crate::rendering::ui::image::{ImageFitness, ImageSource};
use crate::rendering::ui::text::reactive::ui_text;
use crate::rendering::ui::{UIContext, STATE_ENTITY_ID};
use common::make_static_id;
use engine::ecs::component::simple_text::{StyledString, TextStyle};
use engine::ecs::component::ui::{BasicEventCallback2, Padding, Position, Sizing, UILayoutC, Visibility};
use engine::module::ui::color::Color;
use engine::module::ui::reactive::{ScopeId, UIScopeContext};
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::{remember_state, EngineContext};
use entity_data::EntityId;

fn menu_button(local_id: &str, ctx: &mut UIScopeContext, text: &str, on_click: impl BasicEventCallback2) {
    fancy_button(
        local_id,
        ctx,
        UILayoutC::new()
            .with_min_width(240.0)
            .with_padding(Padding::hv(12.0, 6.0)),
        StyledString::new(
            text,
            TextStyle::new()
                .with_color(Color::rgb(3.0, 5.0, 3.0))
                .with_font_size(38.0),
        ),
        on_click,
    );
}

fn world_control_button(
    local_id: &str,
    ctx: &mut UIScopeContext,
    text: &str,
    on_click: impl BasicEventCallback2,
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
                .with_color(Color::grayscale(0.8))
                .with_font_size(24.0),
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
                StyledString::new(name.clone(), TextStyle::new().with_font_size(24.0)),
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
                    world_control_button(make_static_id!(), ctx, "Continue", |entity, ctx| {});
                    width_spacer(make_static_id!(), ctx, 20.0);
                    world_control_button(make_static_id!(), ctx, "Delete", |entity, ctx| {});
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

fn main_menu_controls(ctx: &mut UIScopeContext) {
    let resources = ctx.ctx().resources();

    // TODO: implement resource caching

    let image_source = UIContext::resource_image(&resources, "/textures/main_menu_background.jpg")
        .unwrap()
        .unwrap();

    fn start_on_click(_: &EntityId, ctx: &EngineContext) {
        let mut game = ctx.module_mut::<MainApp>();
        game.start_game_process(ctx);
    }

    fn settings_on_click(entity: &EntityId, ctx: &EngineContext) {}

    fn exit_on_click(_: &EntityId, ctx: &EngineContext) {
        ctx.request_stop();
    }

    ui_image(
        make_static_id!(),
        ctx,
        UILayoutC::column()
            .with_position(Position::Relative(Default::default()))
            .with_width(Sizing::Preferred(400.0))
            .with_height(Sizing::Grow(1.0))
            .with_padding(Padding::equal(30.0)),
        Some(ImageSource::Data(image_source)),
        ImageFitness::Cover,
        |ctx| {
            expander(make_static_id!(), ctx, 1.0);

            world_selection_list(ctx);

            height_spacer(make_static_id!(), ctx, 30.0);

            menu_button(make_static_id!(), ctx, "START", start_on_click);
            height_spacer(make_static_id!(), ctx, 30.0);
            menu_button(make_static_id!(), ctx, "SETTINGS", settings_on_click);
            height_spacer(make_static_id!(), ctx, 30.0);
            menu_button(make_static_id!(), ctx, "EXIT", exit_on_click);

            expander(make_static_id!(), ctx, 0.5);
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
            let menu_visible = ctx.subscribe(menu_visible.state());

            ctx.set_state(&menu_opacity2, |prev| {
                let mut d = *prev;
                let opacity = if *menu_visible { 1.0 } else { 0.0 };
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
            expander(make_static_id!(), ctx, 0.2);

            main_menu_controls(ctx);

            expander(make_static_id!(), ctx, 1.0);
        },
    );
}
