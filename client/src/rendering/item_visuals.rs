use super::ui::container::ContainerBackground;

#[derive(Clone, PartialEq)]
pub struct ItemVisuals {
    ui_background: ContainerBackground,
}

impl ItemVisuals {
    pub fn new(ui_background: ContainerBackground) -> Self {
        Self { ui_background }
    }

    pub fn ui_background(&self) -> &ContainerBackground {
        &self.ui_background
    }
}
