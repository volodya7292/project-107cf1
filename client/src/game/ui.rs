use crate::game::MainApp;
use crate::rendering::ui::container::{Container, ContainerAccess, ContainerImpl};
use crate::rendering::ui::fancy_button::{FancyButton, FancyButtonImpl};
use crate::rendering::ui::image::{ImageFitness, ImageImpl, ImageSource, UIImage};
use crate::rendering::ui::text::{UIText, UITextImpl};
use crate::rendering::ui::UIContext;
use engine::ecs::component::simple_text::{StyledString, TextStyle};
use engine::ecs::component::ui::{BasicEventCallback, Padding, Sizing, UILayoutC};
use engine::module::scene::ObjectEntityId;
use engine::module::ui::color::Color;
use engine::EngineContext;
use entity_data::EntityId;

pub fn make_main_menu_screen(ui_ctx: &mut UIContext, root: &EntityId) -> ObjectEntityId<Container> {
    let container = Container::new(
        ui_ctx,
        *root,
        UILayoutC::row()
            .with_width(Sizing::Grow(1.0))
            .with_height(Sizing::Grow(1.0)),
    );

    Container::expander(ui_ctx, *container, 0.2);

    make_main_menu_controls(ui_ctx, &container);

    Container::expander(ui_ctx, *container, 1.0);

    container
}

fn make_menu_button(
    ui_ctx: &mut UIContext,
    parent: EntityId,
    text: &str,
    on_click: BasicEventCallback,
) -> ObjectEntityId<FancyButton> {
    FancyButton::new(
        ui_ctx,
        parent,
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
    )
}

fn make_world_control_button(
    ui_ctx: &mut UIContext,
    parent: EntityId,
    text: &str,
    on_click: BasicEventCallback,
) -> ObjectEntityId<FancyButton> {
    let btn = FancyButton::new(
        ui_ctx,
        parent,
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
    btn
}

fn start_on_click(_: &EntityId, ctx: &EngineContext) {
    let mut game = ctx.module_mut::<MainApp>();
    game.start_game_process(ctx);
}

fn settings_on_click(entity: &EntityId, ctx: &EngineContext) {}

fn exit_on_click(_: &EntityId, ctx: &EngineContext) {
    ctx.request_stop();
}

fn make_main_menu_controls(ui_ctx: &mut UIContext, parent: &EntityId) -> EntityId {
    let image_source = ui_ctx
        .resource_image("/textures/main_menu_background.jpg")
        .unwrap()
        .unwrap();

    let container = UIImage::new(
        ui_ctx,
        *parent,
        UILayoutC::new()
            .with_width(Sizing::Preferred(400.0))
            .with_height(Sizing::Grow(1.0))
            .with_padding(Padding::equal(30.0)),
        Some(ImageSource::Data(image_source)),
        ImageFitness::Cover,
    );

    Container::expander(ui_ctx, *container, 1.0);

    make_world_list(ui_ctx, *container);

    Container::height_spacer(ui_ctx, *container, 30.0);

    make_menu_button(ui_ctx, *container, "START", start_on_click);
    Container::height_spacer(ui_ctx, *container, 30.0);
    make_menu_button(ui_ctx, *container, "SETTINGS", settings_on_click);
    Container::height_spacer(ui_ctx, *container, 30.0);
    make_menu_button(ui_ctx, *container, "EXIT", exit_on_click);

    Container::expander(ui_ctx, *container, 0.5);

    *container
}

fn make_world_list(ui_ctx: &mut UIContext, parent: EntityId) {
    let container = Container::new(
        ui_ctx,
        parent,
        UILayoutC::column()
            .with_width(Sizing::Grow(1.0))
            .with_height(Sizing::FitContent),
    );

    ui_ctx.ctx().dispatch_callback(move |ctx, _| {
        let mut ui_ctx = UIContext::new(ctx, ctx.module::<MainApp>().resources());
        let world_names = ui_ctx.app().get_world_name_list();

        ui_ctx.scene().clear_children(&*container);

        for name in world_names {
            let item = Container::new(
                &mut ui_ctx,
                *container,
                UILayoutC::column()
                    .with_width(Sizing::Grow(1.0))
                    .with_padding(Padding::equal(10.0)),
            );
            ui_ctx
                .scene()
                .object(&item)
                .set_background_color(Color::WHITE.with_alpha(0.02).into());

            UIText::new(
                &mut ui_ctx,
                *item,
                StyledString::new(name, TextStyle::new().with_font_size(24.0)),
            );
            Container::height_spacer(&mut ui_ctx, *item, 4.0);

            let buttons = Container::new(&mut ui_ctx, *item, UILayoutC::row().with_width(Sizing::Grow(1.0)));
            make_world_control_button(&mut ui_ctx, *buttons, "Continue", |entity, ctx| {});
            Container::width_spacer(&mut ui_ctx, *buttons, 20.0);
            make_world_control_button(&mut ui_ctx, *buttons, "Delete", |entity, ctx| {});

            Container::height_spacer(&mut ui_ctx, *container, 10.0);
        }
    });

    // worlds

    // let resources = ui_ctx.
}
