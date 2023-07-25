use crate::rendering::ui::backgrounds;
use crate::rendering::ui::backgrounds::DEFAULT_FANCY_COLOR;
use crate::rendering::ui::container::{container, ContainerProps};
use crate::rendering::ui::text::reactive::{ui_text, UITextProps};
use crate::rendering::ui::text_input::{ui_text_input, TextChangeCallback, TextInputProps};
use common::make_static_id;
use engine::ecs::component::simple_text::TextStyle;
use engine::ecs::component::ui::{Padding, UILayoutC};
use engine::module::ui::reactive::UIScopeContext;
use std::sync::Arc;

pub struct FancyTextInputProps {
    pub label: String,
    pub layout: UILayoutC,
    pub multiline: bool,
    pub initial_text: String,
    pub style: TextStyle,
    pub on_change: Option<Arc<dyn TextChangeCallback<Output = ()>>>,
}

pub fn fancy_text_input(local_name: &str, ctx: &mut UIScopeContext, props: FancyTextInputProps) {
    container(
        local_name,
        ctx,
        ContainerProps {
            layout: props.layout.with_padding(Padding::equal(2.0)),
            background: Some(backgrounds::fancy(DEFAULT_FANCY_COLOR)),
            ..Default::default()
        },
        move |ctx| {
            let label = props.label.clone();
            let label_style = props
                .style
                .with_font_size(props.style.font_size() * 0.9)
                .with_color(props.style.color().with_brightness(0.7));

            container(
                make_static_id!(),
                ctx,
                ContainerProps {
                    layout: UILayoutC::new().with_padding(Padding::hv(10.0, 0.0)),
                    ..Default::default()
                },
                move |ctx| {
                    ui_text(
                        make_static_id!(),
                        ctx,
                        UITextProps {
                            text: label.clone(),
                            style: label_style,
                            wrap: false,
                        },
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
                    initial_text: props.initial_text.clone(),
                    style: props.style,
                    on_change: props.on_change.clone(),
                },
            );
        },
    );
}
