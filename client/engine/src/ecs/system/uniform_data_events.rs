use crate::ecs::component::UniformDataC;
use crate::ecs::component::uniform_data::BASIC_UNIFORM_BLOCK_MAX_SIZE;
use crate::module::main_renderer::BufferUpdate2;
use crate::module::main_renderer::resources::Renderable;
use common::types::HashMap;
use entity_data::{EntityId, SystemAccess, SystemHandler};
use std::time::Instant;
use vk_wrapper as vkw;

// Updates global transform uniform buffers
pub(crate) struct UniformDataCompEvents<'a> {
    pub uniform_buffer_updates: &'a mut BufferUpdate2,
    pub dirty_components: Vec<EntityId>,
    pub renderables: &'a HashMap<EntityId, Renderable>,
    pub run_time: f64,
}

impl SystemHandler for UniformDataCompEvents<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let uniform_data_comps = data.component::<UniformDataC>();

        for entity in &self.dirty_components {
            let uniform_data = uniform_data_comps.get(entity).unwrap();
            let Some(renderable) = self.renderables.get(entity) else {
                continue;
            };
            let src_offset = self.uniform_buffer_updates.data.len();

            self.uniform_buffer_updates
                .data
                .extend_from_slice(&uniform_data.0);

            self.uniform_buffer_updates.regions.push(vkw::CopyRegion::new(
                src_offset as u64,
                (renderable.uniform_buf_index * BASIC_UNIFORM_BLOCK_MAX_SIZE) as u64,
                (uniform_data.0.len() as u64).try_into().unwrap(),
            ));
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
