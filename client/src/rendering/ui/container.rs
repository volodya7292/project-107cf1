use engine::ecs::component::ui::{Factor, Sizing, UILayoutC};
use engine::module::ui::management::UIState;
use engine::module::ui::UIObject;

#[derive(Default)]
pub struct ContainerState {}

impl UIState for ContainerState {}

pub type Container = UIObject<ContainerState>;

pub trait ContainerImpl {
    fn new(layout: UILayoutC) -> Container {
        Container::new_raw(layout, Default::default())
    }

    fn width_spacer(fraction: Factor) -> Container {
        Container::new(UILayoutC::new().with_width(Sizing::Grow(fraction)))
    }
}

impl ContainerImpl for Container {}
