use crate::ecs::component::internal::WorldTransform;
use crate::ecs::scene_storage::{ComponentStorageImpl, Entity, Event};
use crate::ecs::{component, scene_storage};
use crate::renderer;
use crate::renderer::{BufferUpdate, Renderable};
use crate::utils::HashMap;
use nalgebra_glm::Mat4;
use std::{mem, slice};
use vk_wrapper as vkw;

// Updates world transform uniform buffers
pub(crate) struct WorldTransformEvents<'a> {
    pub uniform_buffer_updates: &'a mut [BufferUpdate],
    pub world_transform_comps: scene_storage::LockedStorage<'a, WorldTransform>,
    pub renderer_comps: scene_storage::LockedStorage<'a, component::RenderConfig>,
    pub renderables: &'a HashMap<Entity, Renderable>,
}

impl WorldTransformEvents<'_> {
    fn world_transform_modified(
        entity: Entity,
        world_transform: &WorldTransform,
        renderer: Option<&component::RenderConfig>,
        buffer_updates: &mut [BufferUpdate],
        renderables: &HashMap<Entity, Renderable>,
    ) {
        if let Some(renderer) = renderer {
            let matrix = world_transform.matrix_f32();
            let matrix_bytes =
                unsafe { slice::from_raw_parts(matrix.as_ptr() as *const u8, mem::size_of::<Mat4>()) };
            let renderable = &renderables[&entity];

            if let crate::renderer::BufferUpdate::Type2(upd) = &mut buffer_updates[0] {
                let src_offset = upd.data.len();
                upd.data.extend(matrix_bytes);
                upd.regions.push(vkw::CopyRegion::new(
                    src_offset as u64,
                    renderable.uniform_buf_index as u64 * renderer::MAX_BASIC_UNIFORM_BLOCK_SIZE
                        + renderer.uniform_buffer_offset_model as u64,
                    matrix_bytes.len() as u64,
                ));
            } else {
                unreachable!()
            }
        }
    }

    pub fn run(&mut self) {
        let events = self.world_transform_comps.write().events();
        let world_transform_comps = self.world_transform_comps.read();
        let renderer_comps = self.renderer_comps.read();

        for event in events {
            match event {
                Event::Created(entity) | Event::Modified(entity) => {
                    Self::world_transform_modified(
                        entity,
                        world_transform_comps.get(entity).unwrap(),
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
