use crate::game::Game;
use crate::rendering::ui::container::{Container, ContainerImpl};
use crate::rendering::ui::fancy_button::{FancyButton, FancyButtonAccess, FancyButtonImpl};
use crate::rendering::ui::image::{ImageFitness, ImageImpl, ImageSource, UIImage};
use crate::rendering::ui::UIContext;
use common::glm::Vec4;
use common::resource_file::ResourceFile;
use engine::ecs::component::simple_text::{StyledString, TextStyle};
use engine::ecs::component::ui::{BasicEventCallback, Padding, Sizing, UILayoutC};
use engine::module::scene::{ObjectEntityId, Scene};
use engine::module::ui::color::Color;
use engine::EngineContext;
use entity_data::EntityId;

pub fn make_main_menu_screen(ui_ctx: &mut UIContext, root: &EntityId) -> EntityId {
    let container = ui_ctx
        .scene()
        .add_object(
            Some(*root),
            Container::new(
                UILayoutC::row()
                    .with_width(Sizing::Grow(1.0))
                    .with_height(Sizing::Grow(1.0)),
            ),
        )
        .unwrap();

    ui_ctx
        .scene()
        .add_object(Some(*container), Container::expander(0.2));

    make_main_menu_controls(ui_ctx, &container);

    ui_ctx
        .scene()
        .add_object(Some(*container), Container::expander(1.0));

    *container
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
                .with_color(Color::rgb(5.0, 10.0, 5.0))
                .with_font_size(38.0),
        ),
        on_click,
    )
}

fn start_on_click(_: &EntityId, ctx: &EngineContext) {
    let mut game = ctx.module_mut::<Game>();
    game.start_game_process(ctx);
}

fn settings_on_click(entity: &EntityId, ctx: &EngineContext) {}

fn exit_on_click(_: &EntityId, ctx: &EngineContext) {
    ctx.request_stop();
}

fn make_main_menu_controls(ui_ctx: &mut UIContext, root: &EntityId) -> EntityId {
    let image_source = ui_ctx
        .resource_image("/textures/main_menu_background.jpg")
        .unwrap()
        .unwrap();

    let back_ui_img = UIImage::new(
        ui_ctx,
        UILayoutC::new()
            .with_width(Sizing::Preferred(400.0))
            .with_height(Sizing::Grow(1.0))
            .with_padding(Padding::equal(30.0)),
        ImageFitness::Cover,
    )
    .with_source(ImageSource::Data(image_source));

    let container = ui_ctx.scene().add_object(Some(*root), back_ui_img).unwrap();

    ui_ctx
        .scene()
        .add_object(Some(*container), Container::expander(1.0));

    make_menu_button(ui_ctx, *container, "START", start_on_click);
    ui_ctx
        .scene()
        .add_object(Some(*container), Container::spacer(30.0));
    make_menu_button(ui_ctx, *container, "SETTINGS", settings_on_click);
    ui_ctx
        .scene()
        .add_object(Some(*container), Container::spacer(30.0));
    make_menu_button(ui_ctx, *container, "EXIT", exit_on_click);

    ui_ctx
        .scene()
        .add_object(Some(*container), Container::expander(2.0));

    *container
}
