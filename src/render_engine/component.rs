mod camera;
mod relation;
pub mod renderer;
mod transform;
mod vertex_mesh;
mod world_transform;

use crate::render_engine::scene::{ComponentStorage, ComponentStorageImpl, ComponentStorageMut, Entity};
use crate::render_engine::Scene;
pub use camera::Camera;
pub use relation::Children;
pub use relation::Parent;
pub use renderer::Renderer;
pub use transform::Transform;
pub use vertex_mesh::VertexMesh;
pub use world_transform::WorldTransform;

pub(super) fn collect_children_recursively(
    children: &mut Vec<Entity>,
    entity: Entity,
    children_comps: &ComponentStorage<Children>,
) {
    let mut stack = Vec::<Entity>::with_capacity(children_comps.len());
    stack.push(entity);

    while let Some(entity) = stack.pop() {
        if let Some(children_comp) = children_comps.get(entity) {
            children.extend(children_comp.get());
            stack.extend(children_comp.get());
        }
    }
}

/// Sets children and accordingly adds `Parent` components
pub fn set_children(
    parent: Entity,
    children: &[Entity],
    parent_comps: &mut ComponentStorageMut<Parent>,
    children_comps: &mut ComponentStorageMut<Children>,
) {
    children_comps.set(parent, Children::new(children.to_vec(), false));

    for &child in children {
        parent_comps.set(child, Parent(parent));
    }
}
