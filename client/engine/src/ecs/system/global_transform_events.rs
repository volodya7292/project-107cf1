use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::uniform_data::{MODEL_MATRIX_OFFSET, MODEL_MATRIX_SIZE};
use crate::ecs::component::UniformDataC;
use common::glm::Mat4;
use common::types::HashSet;
use entity_data::{EntityId, SystemAccess, SystemHandler};
use std::time::Instant;
use std::{mem, slice};

// Updates global transform uniform buffers
pub(crate) struct GlobalTransformEvents {
    pub dirty_components: Vec<EntityId>,
    pub changed_uniforms: HashSet<EntityId>,
    pub run_time: f64,
}

impl SystemHandler for GlobalTransformEvents {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let global_transform_comps = data.component::<GlobalTransformC>();
        let mut uniform_data_comps = data.component_mut::<UniformDataC>();

        self.changed_uniforms.reserve(self.dirty_components.len());

        for entity in &self.dirty_components {
            let global_transform = global_transform_comps.get(entity).unwrap();
            let Some(uniform_data) = uniform_data_comps.get_mut(entity) else {
                continue;
            };

            let matrix = global_transform.matrix_f32();
            let matrix_bytes =
                unsafe { slice::from_raw_parts(matrix.as_ptr() as *const u8, mem::size_of::<Mat4>()) };

            uniform_data.0[MODEL_MATRIX_OFFSET..(MODEL_MATRIX_OFFSET + MODEL_MATRIX_SIZE)]
                .copy_from_slice(matrix_bytes);

            self.changed_uniforms.insert(*entity);
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
