use crate::rendering::ui::backgrounds;
use crate::rendering::ui::backgrounds::DEFAULT_FANCY_COLOR;
use crate::rendering::ui::container::{container, ContainerProps};
use crate::rendering::ui::text_input::{ui_text_input, TextChangeCallback, TextInputProps};
use common::make_static_id;
use engine::ecs::component::simple_text::TextStyle;
use engine::ecs::component::ui::{Padding, UILayoutC};
use engine::module::ui::reactive::UIScopeContext;
use std::sync::Arc;

pub struct FancyTextInputProps {
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
            ui_text_input(
                make_static_id!(),
                ctx,
                TextInputProps {
                    layout: UILayoutC::new()
                        .with_width_grow()
                        .with_padding(Padding::equal(10.0)),
                    multiline: props.multiline,
                    initial_text: props.initial_text.clone(),
                    style: props.style,
                    on_change: props.on_change.clone(),
                },
            );
        },
    );
}
