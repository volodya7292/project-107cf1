use crate::ecs::component;
use crate::ecs::component::render_config::Resource;
use crate::ecs::component::uniform_data::BASIC_UNIFORM_BLOCK_MAX_SIZE;
use crate::ecs::component::MeshRenderConfigC;
use crate::module::main_renderer::material_pipeline::MaterialPipelineSet;
use crate::module::main_renderer::resources::{Renderable, GENERAL_OBJECT_DESCRIPTOR_IDX};
use crate::module::main_renderer::{BufferUpdate, BufferUpdate1, ImageUpdate};
use crate::module::scene::change_manager::{ChangeType, ComponentChange};
use common::types::HashMap;
use entity_data::{EntityId, SystemAccess, SystemHandler};
use index_pool::IndexPool;
use smallvec::{smallvec, SmallVec};
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{BindingRes, DescriptorSet, DeviceBuffer, ImageLayout};

pub(crate) struct RenderConfigComponentEvents<'a> {
    pub device: &'a Arc<vkw::Device>,
    pub renderables: &'a mut HashMap<EntityId, Renderable>,
    pub component_changes: Vec<ComponentChange>,
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub material_pipelines: &'a mut [MaterialPipelineSet],
    pub uniform_buffer_basic: &'a DeviceBuffer,
    pub uniform_buffer_offsets: &'a mut IndexPool,
    pub run_time: f64,
}

impl RenderConfigComponentEvents<'_> {
    fn renderer_comp_created(renderable: &mut Renderable, mat_pipelines: &mut [MaterialPipelineSet]) {
        let mat_pipe = &mut mat_pipelines[renderable.mat_pipeline as usize];

        renderable.descriptor_sets[GENERAL_OBJECT_DESCRIPTOR_IDX] =
            mat_pipe.per_object_desc_pool.alloc().unwrap();
    }

    fn renderer_comp_modified(
        config: &mut MeshRenderConfigC,
        renderable: &mut Renderable,
        buffer_updates: &mut Vec<BufferUpdate>,
        binding_updates: &mut Vec<vkw::Binding>,
        material_pipelines: &mut [MaterialPipelineSet],
        desc_updates: &mut Vec<(DescriptorSet, Range<usize>)>,
        uniform_buffer_basic: &DeviceBuffer,
    ) {
        let mat_pipe = &mut material_pipelines[renderable.mat_pipeline as usize];
        let object_desc_pool = &mat_pipe.per_object_desc_pool;

        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let inputs = &mut renderable.descriptor_sets;
        let mut new_binding_updates: SmallVec<[vkw::Binding; 4]> = smallvec![];

        // Add default uniform buffer binding
        new_binding_updates.push(object_desc_pool.create_binding(
            shader_ids::BINDING_OBJECT_INFO,
            0,
            BindingRes::BufferRange(
                uniform_buffer_basic.handle(),
                0..BASIC_UNIFORM_BLOCK_MAX_SIZE as u64,
            ),
        ));

        for (binding_id, res) in &mut config.resources {
            match res {
                Resource::Buffer {
                    new_source_data,
                    buffer,
                } => {
                    let Some(data) = new_source_data.take() else {
                        continue;
                    };

                    buffer_updates.push(BufferUpdate::WithOffset(BufferUpdate1 {
                        buffer: buffer.handle(),
                        dst_offset: 0,
                        data: data.into(),
                    }));
                    new_binding_updates.push(object_desc_pool.create_binding(
                        *binding_id,
                        0,
                        BindingRes::Buffer(buffer.handle()),
                    ));
                }
                Resource::Image {
                    new_source_data,
                    image,
                } => {
                    let Some(data) = new_source_data.take() else {
                        continue;
                    };

                    buffer_updates.push(BufferUpdate::Image(ImageUpdate {
                        image: Arc::clone(image),
                        data,
                    }));
                    new_binding_updates.push(object_desc_pool.create_binding(
                        *binding_id,
                        0,
                        BindingRes::Image(Arc::clone(image), None, ImageLayout::SHADER_READ),
                    ));
                }
                Resource::None => {}
            }
        }

        if !new_binding_updates.is_empty() {
            let s0 = binding_updates.len();
            binding_updates.extend(new_binding_updates);
            let bindings_range = s0..binding_updates.len();

            desc_updates.push((inputs[GENERAL_OBJECT_DESCRIPTOR_IDX], bindings_range));
        }
    }

    pub fn renderer_comp_removed(
        renderable: &Renderable,
        mat_pipelines: &mut [MaterialPipelineSet],
        uniform_buffer_offsets: &mut IndexPool,
    ) {
        let mat_pipe = &mut mat_pipelines[renderable.mat_pipeline as usize];

        mat_pipe
            .per_object_desc_pool
            .free(renderable.descriptor_sets[GENERAL_OBJECT_DESCRIPTOR_IDX]);

        let _ = uniform_buffer_offsets.return_id(renderable.uniform_buf_index);
    }
}

impl SystemHandler for RenderConfigComponentEvents<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();

        let mut renderer_comps = data.component_mut::<MeshRenderConfigC>();

        let mut binding_updates = Vec::<vkw::Binding>::with_capacity(self.component_changes.len());
        let mut desc_updates =
            Vec::<(DescriptorSet, Range<usize>)>::with_capacity(self.component_changes.len());

        for change in &self.component_changes {
            let entity = change.entity();

            if change.ty() == ChangeType::Removed {
                Self::renderer_comp_removed(
                    &self.renderables.remove(entity).unwrap(),
                    self.material_pipelines,
                    self.uniform_buffer_offsets,
                );
                continue;
            }

            let config = renderer_comps.get_mut(entity).unwrap();
            if config.mat_pipeline == u32::MAX {
                continue;
            }

            if change.ty() == ChangeType::Modified {
                Self::renderer_comp_removed(
                    &self.renderables.remove(&entity).unwrap(),
                    self.material_pipelines,
                    self.uniform_buffer_offsets,
                );
            }

            let mut renderable = Renderable {
                buffers: smallvec![],
                mat_pipeline: config.mat_pipeline,
                uniform_buf_index: self.uniform_buffer_offsets.new_id(),
                descriptor_sets: Default::default(),
            };
            Self::renderer_comp_created(&mut renderable, self.material_pipelines);
            Self::renderer_comp_modified(
                config,
                &mut renderable,
                self.buffer_updates,
                &mut binding_updates,
                self.material_pipelines,
                &mut desc_updates,
                self.uniform_buffer_basic,
            );
            self.renderables.insert(*entity, renderable);
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();

        unsafe {
            self.device
                .update_descriptor_sets(&binding_updates, &desc_updates)
        };
    }
}
