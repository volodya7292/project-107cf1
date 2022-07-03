use crate::ecs::component::internal::GlobalTransform;
use crate::ecs::scene_storage::{ComponentStorageImpl, Entity, Event};
use crate::ecs::{component, scene_storage};
use crate::renderer;
use crate::renderer::{BufferUpdate2, Renderable};
use crate::utils::HashMap;
use nalgebra_glm::Mat4;
use std::{mem, slice};
use vk_wrapper as vkw;

// Updates global transform uniform buffers
pub(crate) struct GlobalTransformEvents<'a> {
    pub uniform_buffer_updates: &'a mut BufferUpdate2,
    pub global_transform_comps: scene_storage::LockedStorage<'a, GlobalTransform>,
    pub renderer_comps: scene_storage::LockedStorage<'a, component::MeshRenderConfig>,
    pub renderables: &'a HashMap<Entity, Renderable>,
}

impl GlobalTransformEvents<'_> {
    fn global_transform_modified(
        entity: Entity,
        global_transform: &GlobalTransform,
        render_config: Option<&component::MeshRenderConfig>,
        buffer_updates: &mut BufferUpdate2,
        renderables: &HashMap<Entity, Renderable>,
    ) {
        if let Some(config) = render_config {
            let matrix = global_transform.matrix_f32();
            let matrix_bytes =
                unsafe { slice::from_raw_parts(matrix.as_ptr() as *const u8, mem::size_of::<Mat4>()) };
            let renderable = &renderables[&entity];
            let src_offset = buffer_updates.data.len();

            buffer_updates.data.extend_from_slice(matrix_bytes);

            buffer_updates.regions.push(vkw::CopyRegion::new(
                src_offset as u64,
                renderable.uniform_buf_index as u64 * renderer::MAX_BASIC_UNIFORM_BLOCK_SIZE
                    + config.uniform_buffer_offset_model as u64,
                (matrix_bytes.len() as u64).try_into().unwrap(),
            ));
        }
    }

    pub fn run(&mut self) {
        let events = self.global_transform_comps.write().events();
        let global_transform_comps = self.global_transform_comps.read();
        let renderer_comps = self.renderer_comps.read();

        for event in events {
            match event {
                Event::Created(entity) | Event::Modified(entity) => {
                    Self::global_transform_modified(
                        entity,
                        global_transform_comps.get(entity).unwrap(),
                        renderer_comps.get(entity),
                        self.uniform_buffer_updates,
                        self.renderables,
                    );
                }
                _ => {}
            }
        }
    }
}
