use super::common::*;

pub fn action_button(
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
