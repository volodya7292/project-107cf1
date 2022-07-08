use crate::ecs::component;
use crate::ecs::scene_storage;
use crate::ecs::scene_storage::{Entity, Event};
use crate::renderer::material_pipeline::MaterialPipelineSet;
use crate::renderer::{BufferUpdate, BufferUpdate1, Renderable};
use crate::utils::HashMap;
use index_pool::IndexPool;
use smallvec::{smallvec, SmallVec};
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::DescriptorSet;

pub(crate) struct RendererComponentEvents<'a> {
    pub device: &'a Arc<vkw::Device>,
    pub renderer_comps: scene_storage::LockedStorage<'a, component::MeshRenderConfig>,
    pub renderables: &'a mut HashMap<Entity, Renderable>,
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub material_pipelines: &'a mut [MaterialPipelineSet],
    pub uniform_buffer_offsets: &'a mut IndexPool,
}

impl RendererComponentEvents<'_> {
    fn renderer_comp_created(renderable: &mut Renderable, object_desc_pool: &mut vkw::DescriptorPool) {
        renderable.descriptor_sets = smallvec![object_desc_pool.alloc().unwrap()];
    }

    fn renderer_comp_modified(
        config: &mut component::MeshRenderConfig,
        renderable: &mut Renderable,
        buffer_updates: &mut Vec<BufferUpdate>,
        binding_updates: &mut Vec<vkw::Binding>,
        object_desc_pool: &mut vkw::DescriptorPool,
        desc_updates: &mut Vec<(DescriptorSet, Range<usize>)>,
    ) {
        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let inputs = &mut renderable.descriptor_sets;

        let mut updates: SmallVec<[vkw::Binding; 4]> = smallvec![];

        for (binding_id, res) in &mut config.resources {
            if let component::render_config::Resource::Buffer(buf_res) = res {
                if buf_res.changed {
                    let data = mem::take(&mut buf_res.buffer);

                    buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                        buffer: buf_res.device_buffer.handle(),
                        offset: 0,
                        data: data.into(),
                    }));
                    buf_res.changed = false;

                    updates.push(object_desc_pool.create_binding(
                        *binding_id,
                        0,
                        vkw::BindingRes::Buffer(buf_res.device_buffer.handle()),
                    ));
                }
            }
        }

        let s0 = binding_updates.len();
        binding_updates.extend(updates);
        let s = s0..binding_updates.len();
        if !s.is_empty() {
            desc_updates.push((inputs[0], s));
        }
    }

    fn renderer_comp_removed(renderable: &Renderable, object_desc_pool: &mut vkw::DescriptorPool) {
        object_desc_pool.free(renderable.descriptor_sets[0]);
    }

    pub fn run(&mut self) {
        let mut renderer_comps = self.renderer_comps.write();
        let events = renderer_comps.events();
        let mut binding_updates = Vec::<vkw::Binding>::with_capacity(events.len());
        let mut desc_updates = Vec::<(vkw::DescriptorSet, Range<usize>)>::with_capacity(events.len());

        for event in events {
            match event {
                Event::Created(entity) => {
                    let config = renderer_comps.get_mut_unmarked(entity).unwrap();
                    let object_desc_pool =
                        &mut self.material_pipelines[config.mat_pipeline as usize].per_object_desc_pool;

                    let uniform_buf_index = self.uniform_buffer_offsets.new_id();
                    let mut renderable = Renderable {
                        buffers: smallvec![],
                        mat_pipeline: config.mat_pipeline,
                        uniform_buf_index,
                        descriptor_sets: Default::default(),
                    };

                    Self::renderer_comp_created(&mut renderable, object_desc_pool);
                    Self::renderer_comp_modified(
                        config,
                        &mut renderable,
                        self.buffer_updates,
                        &mut binding_updates,
                        object_desc_pool,
                        &mut desc_updates,
                    );
                    self.renderables.insert(entity, renderable);
                }
                Event::Modified(entity) => {
                    let config = renderer_comps.get_mut_unmarked(entity).unwrap();
                    let object_desc_pool =
                        &mut self.material_pipelines[config.mat_pipeline as usize].per_object_desc_pool;

                    let mut renderable = self.renderables.remove(&entity).unwrap();
                    Self::renderer_comp_removed(&renderable, object_desc_pool);

                    renderable.buffers = smallvec![];
                    renderable.mat_pipeline = config.mat_pipeline;
                    renderable.descriptor_sets = Default::default();

                    Self::renderer_comp_created(&mut renderable, object_desc_pool);
                    Self::renderer_comp_modified(
                        config,
                        &mut renderable,
                        self.buffer_updates,
                        &mut binding_updates,
                        object_desc_pool,
                        &mut desc_updates,
                    );
                    self.renderables.insert(entity, renderable);
                }
                Event::Removed(entity) => {
                    let renderable = &self.renderables[&entity];
                    let object_desc_pool =
                        &mut self.material_pipelines[renderable.mat_pipeline as usize].per_object_desc_pool;

                    Self::renderer_comp_removed(&renderable, object_desc_pool);

                    self.uniform_buffer_offsets
                        .return_id(renderable.uniform_buf_index)
                        .unwrap();
                    self.renderables.remove(&entity);
                }
            }
        }

        unsafe {
            self.device
                .update_descriptor_sets(&binding_updates, &desc_updates)
        };
    }
}
