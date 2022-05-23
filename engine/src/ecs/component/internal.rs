use crate::ecs::component::Transform;
use crate::ecs::scene_storage::Entity;
use crate::utils::IndexSet;
use nalgebra_glm::Mat4;

pub struct Parent(pub Entity);

#[derive(Default)]
pub struct Children {
    pub children: IndexSet<Entity>,
}

#[derive(Copy, Clone, Default)]
pub struct GlobalTransform {
    pub transform: Transform,
    pub matrix: Mat4,
    pub matrix_inverse: Mat4,
}

impl GlobalTransform {
    pub fn new(transform: Transform) -> Self {
        let matrix = transform.matrix_f32();
        let matrix_inverse = matrix.try_inverse().unwrap_or(Default::default());

        Self {
            transform,
            matrix,
            matrix_inverse,
        }
    }
}
