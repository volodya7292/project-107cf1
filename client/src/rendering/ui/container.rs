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

    fn expander(fraction: Factor) -> Container {
        Container::new(
            UILayoutC::new()
                .with_width(Sizing::Grow(fraction))
                .with_height(Sizing::Grow(fraction)),
        )
    }

    fn spacer(size: f32) -> Container {
        Container::new(
            UILayoutC::new()
                .with_width(Sizing::Preferred(size))
                .with_height(Sizing::Preferred(size)),
        )
    }
}

impl ContainerImpl for Container {}
