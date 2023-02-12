use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::{MeshRenderConfigC, UniformDataC};
use crate::module::main_renderer::material_pipeline::MaterialPipelineSet;
use base::utils::HashSet;
use entity_data::{EntityId, SystemAccess, SystemHandler};
use nalgebra_glm::Mat4;
use std::time::Instant;
use std::{mem, slice};

// Updates global transform uniform buffers
pub(crate) struct GlobalTransformEvents<'a> {
    pub dirty_components: HashSet<EntityId>,
    pub changed_uniforms: HashSet<EntityId>,
    pub material_pipelines: &'a [MaterialPipelineSet],
    pub run_time: f64,
}

impl SystemHandler for GlobalTransformEvents<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let global_transform_comps = data.component::<GlobalTransformC>();
        let renderer_comps = data.component::<MeshRenderConfigC>();
        let mut uniform_data_comps = data.component_mut::<UniformDataC>();

        self.changed_uniforms.reserve(self.dirty_components.len());

        for entity in &self.dirty_components {
            let global_transform = global_transform_comps.get(entity).unwrap();
            let (Some(render_config), Some(uniform_data)) =
                    (renderer_comps.get(entity), uniform_data_comps.get_mut(entity)) else {
                continue;
            };
            let Some(pipe) = self.material_pipelines.get(render_config.mat_pipeline as usize) else {
                continue;
            };

            let matrix = global_transform.matrix_f32();
            let matrix_bytes =
                unsafe { slice::from_raw_parts(matrix.as_ptr() as *const u8, mem::size_of::<Mat4>()) };

            let offset = pipe.uniform_buffer_offset_model() as usize;
            uniform_data.0[offset..(offset + matrix_bytes.len())].copy_from_slice(matrix_bytes);

            self.changed_uniforms.insert(*entity);
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
