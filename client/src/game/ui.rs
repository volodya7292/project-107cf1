use crate::rendering::ui::container::{Container, ContainerImpl};
use engine::ecs::component::ui::UILayoutC;
use engine::module::scene::Scene;
use entity_data::EntityId;

fn make_menu_screen(scene: &mut Scene, root: &EntityId) -> EntityId {
    let container = scene
        .add_object(Some(*root), Container::new(UILayoutC::row()))
        .unwrap();

    scene.add_object(Some(container), Container::width_spacer(0.2));

    make_menu_controls(scene, &container);

    container
}

fn make_menu_controls(scene: &mut Scene, root: &EntityId) -> EntityId {
    let container = scene
        .add_object(Some(*root), Container::new(UILayoutC::column()))
        .unwrap();

    // scene.add_object(Some(container));
    // Start, Settings, Exit
    todo!();

    container
}
