use std::time::Instant;
use std::{mem, slice};

use entity_data::{EntityId, SystemAccess, SystemHandler};
use nalgebra_glm::Mat4;

use base::unwrap_option;
use base::utils::{HashMap, HashSet};
use vk_wrapper as vkw;

use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::MeshRenderConfigC;
use crate::renderer;
use crate::renderer::material_pipeline::MaterialPipelineSet;
use crate::renderer::resources::Renderable;
use crate::renderer::BufferUpdate2;

// Updates global transform uniform buffers
pub(crate) struct GlobalTransformEvents<'a> {
    pub uniform_buffer_updates: &'a mut BufferUpdate2,
    pub dirty_components: HashSet<EntityId>,
    pub material_pipelines: &'a [MaterialPipelineSet],
    pub renderables: &'a HashMap<EntityId, Renderable>,
    pub run_time: f64,
}

impl SystemHandler for GlobalTransformEvents<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let global_transform_comps = data.component::<GlobalTransformC>();
        let renderer_comps = data.component::<MeshRenderConfigC>();

        for entity in &self.dirty_components {
            let global_transform = unwrap_option!(global_transform_comps.get(entity), continue);
            let render_config = unwrap_option!(renderer_comps.get(entity), continue);

            let Some(pipe) = self.material_pipelines.get(render_config.mat_pipeline as usize) else {
                continue;
            };

            let matrix = global_transform.matrix_f32();
            let matrix_bytes =
                unsafe { slice::from_raw_parts(matrix.as_ptr() as *const u8, mem::size_of::<Mat4>()) };
            let renderable = &self.renderables[&entity];
            let src_offset = self.uniform_buffer_updates.data.len();

            self.uniform_buffer_updates.data.extend_from_slice(matrix_bytes);

            self.uniform_buffer_updates.regions.push(vkw::CopyRegion::new(
                src_offset as u64,
                renderable.uniform_buf_index as u64 * renderer::MAX_BASIC_UNIFORM_BLOCK_SIZE
                    + pipe.uniform_buffer_offset_model() as u64,
                (matrix_bytes.len() as u64).try_into().unwrap(),
            ));
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
