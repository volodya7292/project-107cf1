use crate::rendering::ui::backgrounds;
use crate::rendering::ui::backgrounds::DEFAULT_FANCY_COLOR;
use crate::rendering::ui::container::container;
use crate::rendering::ui::text::reactive::ui_text;
use crate::rendering::ui::text_input::{TextInputProps, ui_text_input};
use common::make_static_id;
use engine::ecs::component::simple_text::TextStyle;
use engine::ecs::component::ui::{Padding, UILayoutC};
use engine::module::ui::reactive::{ReactiveState, UIScopeContext};

use super::container::container_props;
use super::text::reactive::ui_text_props;

pub struct FancyTextInputProps {
    pub label: String,
    pub layout: UILayoutC,
    pub multiline: bool,
    pub text_state: ReactiveState<String>,
    pub style: TextStyle,
}

pub fn fancy_text_input(local_name: &str, ctx: &mut UIScopeContext, props: FancyTextInputProps) {
    container(
        local_name,
        ctx,
        container_props()
            .layout(props.layout.with_padding(Padding::equal(2.0)))
            .background(Some(backgrounds::fancy(DEFAULT_FANCY_COLOR))),
        move |ctx, ()| {
            let label = props.label.clone();
            let label_style = props
                .style
                .with_font_size(props.style.font_size())
                .with_color(DEFAULT_FANCY_COLOR.with_brightness(1.2));

            container(
                make_static_id!(),
                ctx,
                container_props().layout(UILayoutC::new().with_padding(Padding::hv(10.0, 0.0))),
                move |ctx, ()| {
                    ui_text(
                        make_static_id!(),
                        ctx,
                        ui_text_props(label.clone()).style(label_style).wrap(false),
                    );
                },
            );
            ui_text_input(
                make_static_id!(),
                ctx,
                TextInputProps {
                    layout: UILayoutC::new().with_width_grow().with_padding(Padding {
                        left: 10.0,
                        right: 10.0,
                        top: 5.0,
                        bottom: 10.0,
                    }),
                    multiline: props.multiline,
                    text_state: props.text_state.clone(),
                    style: props.style,
                },
            );
        },
    );
}
