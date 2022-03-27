use crate::component;
use crate::material_pipeline::MaterialPipelineSet;
use crate::render_engine::internal::{BufferUpdate, BufferUpdate1, Renderable};
use crate::render_engine::scene;
use crate::render_engine::scene::{Entity, Event};
use crate::utils::HashMap;
use index_pool::IndexPool;
use smallvec::{smallvec, SmallVec};
use std::mem;
use std::ops::Range;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::DescriptorSet;

pub struct RendererComponentEvents<'a> {
    pub device: &'a Arc<vkw::Device>,
    pub renderer_comps: scene::LockedStorage<'a, component::Renderer>,
    pub g_per_pipeline_pools: &'a mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
    pub renderables: &'a mut HashMap<Entity, Renderable>,
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub material_pipelines: &'a [MaterialPipelineSet],
    pub uniform_buffer_offsets: &'a mut IndexPool,
}

impl RendererComponentEvents<'_> {
    fn renderer_comp_created(
        renderable: &mut Renderable,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        signature: &vkw::PipelineSignature,
    ) {
        renderable.descriptor_sets =
            smallvec![g_per_pipeline_pools.get_mut(signature).unwrap().alloc().unwrap(),];
    }

    fn renderer_comp_modified(
        renderer: &mut component::Renderer,
        renderable: &mut Renderable,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        buffer_updates: &mut Vec<BufferUpdate>,
        signature: &vkw::PipelineSignature,
        binding_updates: &mut Vec<vkw::Binding>,
        desc_updates: &mut Vec<(DescriptorSet, Range<usize>)>,
    ) {
        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let inputs = &mut renderable.descriptor_sets;

        let g_pool = g_per_pipeline_pools.get_mut(signature).unwrap();
        let mut updates: SmallVec<[vkw::Binding; 4]> = smallvec![];

        for (binding_id, res) in &mut renderer.resources {
            if let component::renderer::Resource::Buffer(buf_res) = res {
                if buf_res.changed {
                    let data = mem::take(&mut buf_res.buffer);

                    buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                        buffer: buf_res.device_buffer.handle(),
                        offset: 0,
                        data,
                    }));
                    buf_res.changed = false;

                    updates.push(g_pool.create_binding(
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

    fn renderer_comp_removed(
        renderable: &Renderable,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        mat_pipes: &[MaterialPipelineSet],
    ) {
        g_per_pipeline_pools
            .get_mut(&mat_pipes[renderable.material_pipe as usize].main_signature)
            .unwrap()
            .free(renderable.descriptor_sets[0]);
    }

    pub fn run(&mut self) {
        let mut renderer_comps = self.renderer_comps.write();
        let events = renderer_comps.events();
        let mut binding_updates = Vec::<vkw::Binding>::with_capacity(events.len());
        let mut desc_updates = Vec::<(vkw::DescriptorSet, Range<usize>)>::with_capacity(events.len());

        for event in events {
            match event {
                scene::Event::Created(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unmarked(entity).unwrap();
                    let signature =
                        &self.material_pipelines[renderer_comp.mat_pipeline as usize].main_signature;

                    let uniform_buf_index = self.uniform_buffer_offsets.new_id();
                    let mut renderable = Renderable {
                        buffers: smallvec![],
                        material_pipe: renderer_comp.mat_pipeline,
                        uniform_buf_index,
                        descriptor_sets: Default::default(),
                    };

                    Self::renderer_comp_created(&mut renderable, self.g_per_pipeline_pools, signature);
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut renderable,
                        self.g_per_pipeline_pools,
                        self.buffer_updates,
                        signature,
                        &mut binding_updates,
                        &mut desc_updates,
                    );
                    self.renderables.insert(entity, renderable);
                }
                scene::Event::Modified(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unmarked(entity).unwrap();
                    let signature =
                        &self.material_pipelines[renderer_comp.mat_pipeline as usize].main_signature;

                    let mut renderable = self.renderables.remove(&entity).unwrap();
                    Self::renderer_comp_removed(
                        &renderable,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );

                    renderable.buffers = smallvec![];
                    renderable.material_pipe = renderer_comp.mat_pipeline;
                    renderable.descriptor_sets = Default::default();

                    Self::renderer_comp_created(&mut renderable, self.g_per_pipeline_pools, signature);
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut renderable,
                        self.g_per_pipeline_pools,
                        self.buffer_updates,
                        signature,
                        &mut binding_updates,
                        &mut desc_updates,
                    );
                    self.renderables.insert(entity, renderable);
                }
                Event::Removed(entity) => {
                    let renderable = &self.renderables[&entity];
                    Self::renderer_comp_removed(
                        &renderable,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );
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
