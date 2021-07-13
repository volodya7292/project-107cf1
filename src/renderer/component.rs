mod camera;
mod model_transform;
mod relation;
pub mod renderer;
mod transform;
mod vertex_mesh;
mod world_transform;

use crate::renderer::scene::{ComponentStorage, ComponentStorageMut};
use crate::renderer::Scene;
pub use camera::Camera;
pub use model_transform::ModelTransform;
pub use relation::Children;
pub use relation::Parent;
pub use renderer::Renderer;
pub use transform::Transform;
pub use vertex_mesh::VertexMesh;
pub use world_transform::WorldTransform;

fn collect_children(children: &mut Vec<u32>, child_comps: &ComponentStorage<Children>, entity: u32) {
    if let Some(childred_comp) = child_comps.get(entity) {
        for &child in &childred_comp.0 {
            collect_children(children, child_comps, child);
        }
        children.extend(&childred_comp.0);
    }
}

/// Remove entities and their children from the scene recursively.
pub fn remove_entities(scene: &Scene, entities: &[u32]) {
    let child_comps = scene.storage::<Children>();
    let child_comps = child_comps.read().unwrap();
    let mut total_entites = entities.to_vec();

    for &entity in entities {
        collect_children(&mut total_entites, &child_comps, entity);
    }

    drop(child_comps);
    scene.remove_entities(&total_entites);
}

pub fn set_children(
    parent: u32,
    children: &[u32],
    parent_comps: &mut ComponentStorageMut<Parent>,
    children_comps: &mut ComponentStorageMut<Children>,
) {
    children_comps.set(parent, Children(children.to_vec()));

    for &child in children {
        parent_comps.set(child, Parent(parent));
    }
}
