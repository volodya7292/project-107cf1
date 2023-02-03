use crate::ecs::component;
use crate::ecs::component::MeshRenderConfigC;
use crate::renderer::material_pipeline::MaterialPipelineSet;
use crate::renderer::{
    BufferUpdate, BufferUpdate1, Renderable, CUSTOM_OBJECT_DESCRIPTOR_IDX, MAX_BASIC_UNIFORM_BLOCK_SIZE,
};
use base::unwrap_option;
use base::utils::{HashMap, HashSet};
use entity_data::{EntityId, SystemAccess, SystemHandler};
use index_pool::IndexPool;
use smallvec::{smallvec, SmallVec};
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{BindingRes, DescriptorSet, DeviceBuffer};

pub(crate) struct RendererComponentEvents<'a> {
    pub device: &'a Arc<vkw::Device>,
    pub renderables: &'a mut HashMap<EntityId, Renderable>,
    pub dirty_components: HashSet<EntityId>,
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub material_pipelines: &'a mut [MaterialPipelineSet],
    pub uniform_buffer_basic: &'a DeviceBuffer,
    pub uniform_buffer_offsets: &'a mut IndexPool,
    pub run_time: f64,
}

impl RendererComponentEvents<'_> {
    fn renderer_comp_created(renderable: &mut Renderable, object_desc_pool: &mut vkw::DescriptorPool) {
        renderable.descriptor_sets[CUSTOM_OBJECT_DESCRIPTOR_IDX] = object_desc_pool.alloc().unwrap();
    }

    fn renderer_comp_modified(
        config: &mut MeshRenderConfigC,
        renderable: &mut Renderable,
        buffer_updates: &mut Vec<BufferUpdate>,
        binding_updates: &mut Vec<vkw::Binding>,
        object_desc_pool: &mut vkw::DescriptorPool,
        desc_updates: &mut Vec<(DescriptorSet, Range<usize>)>,
        uniform_buffer_basic: &DeviceBuffer,
    ) {
        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let inputs = &mut renderable.descriptor_sets;
        let mut new_binding_updates: SmallVec<[vkw::Binding; 4]> = smallvec![];

        // Add default uniform buffer binding
        new_binding_updates.push(object_desc_pool.create_binding(
            0,
            0,
            BindingRes::BufferRange(uniform_buffer_basic.handle(), 0..MAX_BASIC_UNIFORM_BLOCK_SIZE),
        ));

        for (binding_id, res) in &mut config.resources {
            if let component::render_config::Resource::Buffer(buf_res) = res {
                if buf_res.changed {
                    let data = mem::take(&mut buf_res.buffer);

                    buffer_updates.push(BufferUpdate::WithOffset(BufferUpdate1 {
                        buffer: buf_res.device_buffer.handle(),
                        dst_offset: 0,
                        data: data.into(),
                    }));
                    buf_res.changed = false;

                    new_binding_updates.push(object_desc_pool.create_binding(
                        *binding_id,
                        0,
                        vkw::BindingRes::Buffer(buf_res.device_buffer.handle()),
                    ));
                }
            }
        }

        if !new_binding_updates.is_empty() {
            let s0 = binding_updates.len();
            binding_updates.extend(new_binding_updates);
            let bindings_range = s0..binding_updates.len();

            desc_updates.push((inputs[CUSTOM_OBJECT_DESCRIPTOR_IDX], bindings_range));
        }
    }

    pub fn renderer_comp_removed(
        renderable: &Renderable,
        object_desc_pool: &mut vkw::DescriptorPool,
        uniform_buffer_offsets: &mut IndexPool,
    ) {
        object_desc_pool.free(renderable.descriptor_sets[0]);
        let _ = uniform_buffer_offsets.return_id(renderable.uniform_buf_index);
    }
}

impl SystemHandler for RendererComponentEvents<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();

        let mut renderer_comps = data.component_mut::<MeshRenderConfigC>();

        let mut binding_updates = Vec::<vkw::Binding>::with_capacity(self.dirty_components.len());
        let mut desc_updates =
            Vec::<(DescriptorSet, Range<usize>)>::with_capacity(self.dirty_components.len());

        for entity in &self.dirty_components {
            let config = unwrap_option!(renderer_comps.get_mut(entity), continue);

            if config.mat_pipeline == u32::MAX {
                continue;
            }

            let object_desc_pool =
                &mut self.material_pipelines[config.mat_pipeline as usize].per_object_desc_pool;

            let mut renderable = if let Some(mut renderable) = self.renderables.remove(&entity) {
                Self::renderer_comp_removed(&renderable, object_desc_pool, self.uniform_buffer_offsets);

                renderable.buffers = smallvec![];
                renderable.mat_pipeline = config.mat_pipeline;
                renderable.descriptor_sets = Default::default();

                renderable
            } else {
                Renderable {
                    buffers: smallvec![],
                    mat_pipeline: config.mat_pipeline,
                    uniform_buf_index: self.uniform_buffer_offsets.new_id(),
                    descriptor_sets: Default::default(),
                }
            };

            Self::renderer_comp_created(&mut renderable, object_desc_pool);
            Self::renderer_comp_modified(
                config,
                &mut renderable,
                self.buffer_updates,
                &mut binding_updates,
                object_desc_pool,
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
