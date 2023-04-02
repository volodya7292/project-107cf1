use crate::ecs::component::render_config::RenderType;
use crate::ecs::component::uniform_data::BASIC_UNIFORM_BLOCK_MAX_SIZE;
use crate::ecs::component::MeshRenderConfigC;
use crate::module::main_renderer::compose_descriptor_sets;
use crate::module::main_renderer::material_pipeline::{MaterialPipelineSet, PipelineConfig, PipelineKindId};
use crate::module::main_renderer::resource_manager::{CmdListParams, ImageParams, ResourceManagementScope};
use crate::module::main_renderer::resources::{MaterialPipelineParams, GENERAL_OBJECT_DESCRIPTOR_IDX};
use crate::module::main_renderer::stage::depth::{DepthStage, VisibilityType};
use crate::module::main_renderer::stage::{RenderStage, RenderStageId, StageContext, StageRunResult};
use crate::module::main_renderer::vertex_mesh::VertexMeshCmdList;
use common::parking_lot::Mutex;
use common::rayon;
use common::rayon::prelude::*;
use common::types::HashMap;
use std::any::{Any, TypeId};
use std::iter;
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, Attachment, AttachmentRef, ClearValue, CmdList, Device, Format, Framebuffer, HostBuffer,
    ImageLayout, ImageMod, ImageUsageFlags, LoadStore, PipelineStageFlags, QueueType, RenderPass, Subpass,
    SubpassDependency,
};

const ALBEDO_ATTACHMENT_ID: u32 = 0;

#[repr(C)]
struct GPassConsts {
    is_translucent_pass: u32,
}

pub struct GBufferStage {
    color_pipe: PipelineKindId,
    color_with_blending_pipe: PipelineKindId,
    overlay_pipe: PipelineKindId,
    render_pass: Arc<RenderPass>,
}

impl GBufferStage {
    pub const RES_FRAMEBUFFER: &'static str = "g-framebuffer";

    pub fn new(device: &Arc<Device>) -> Self {
        // Create G-Buffer pass resources
        // -----------------------------------------------------------------------------------------------------------------
        let render_pass = device
            .create_render_pass(
                &[
                    // Albedo
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::InitClearFinalSave,
                    },
                    // Specular
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::FinalSave,
                    },
                    // Emission
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::FinalSave,
                    },
                    // Normal
                    Attachment {
                        format: Format::RG16_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::FinalSave,
                    },
                    // Depth (read)
                    Attachment {
                        format: Format::D32_FLOAT,
                        init_layout: ImageLayout::DEPTH_STENCIL_READ,
                        final_layout: ImageLayout::DEPTH_STENCIL_READ,
                        load_store: LoadStore::InitLoad,
                    },
                    // Overlay depth (read/write)
                    Attachment {
                        format: Format::D32_FLOAT,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                        load_store: LoadStore::InitClear,
                    },
                ],
                &[
                    // Main pass
                    Subpass::new()
                        .with_color(vec![
                            AttachmentRef {
                                index: 0,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 1,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 2,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 3,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                        ])
                        .with_depth(AttachmentRef {
                            index: 4,
                            layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                        }),
                    // Overlay pass
                    Subpass::new()
                        .with_color(vec![
                            AttachmentRef {
                                index: 0,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 1,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 2,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 3,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                        ])
                        .with_depth(AttachmentRef {
                            index: 5,
                            layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                        }),
                ],
                &[SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: 1,
                    src_stage_mask: PipelineStageFlags::PIXEL_SHADER,
                    dst_stage_mask: PipelineStageFlags::PIXEL_SHADER,
                    src_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    dst_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                }],
            )
            .unwrap();

        Self {
            color_pipe: u32::MAX,
            color_with_blending_pipe: u32::MAX,
            overlay_pipe: u32::MAX,
            render_pass,
        }
    }

    fn record_g_cmd_lists(
        &self,
        solid_objects_cls: &mut [CmdList],
        translucent_objects_cls: &mut [CmdList],
        framebuffer: &Framebuffer,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
    ) {
        let mat_pipelines = ctx.material_pipelines;
        let object_count = ctx.ordered_entities.len();
        let draw_count_step = object_count / solid_objects_cls.len() + 1;
        let ordered_entities = ctx.ordered_entities;

        let visibility_host_buffer = resources
            .get_host_buffer::<VisibilityType>(DepthStage::RES_VISIBILITY_HOST_BUFFER)
            .lock_arc();

        solid_objects_cls
            .par_iter_mut()
            .zip(translucent_objects_cls)
            .enumerate()
            .for_each(|(i, (cmd_list_solid, cmd_list_translucent))| {
                let cl_sol = cmd_list_solid;
                let cl_trans = cmd_list_translucent;

                cl_sol
                    .begin_secondary_graphics(true, &self.render_pass, 0, Some(framebuffer))
                    .unwrap();
                cl_trans
                    .begin_secondary_graphics(true, &self.render_pass, 0, Some(framebuffer))
                    .unwrap();

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let renderable_id = ordered_entities[entity_index];
                    let Some(render_config) = ctx.storage.get::<MeshRenderConfigC>(&renderable_id) else {
                        continue;
                    };
                    if render_config.render_ty != RenderType::MAIN {
                        continue;
                    }

                    let Some(vertex_mesh) = ctx.curr_vertex_meshes.get(&renderable_id) else {
                        continue;
                    };

                    if visibility_host_buffer[entity_index] == 0 && vertex_mesh.is_empty() {
                        continue;
                    }

                    let renderable = ctx.renderables.get(&renderable_id).unwrap();
                    let mut consts = GPassConsts {
                        is_translucent_pass: 0,
                    };

                    let mat_pipeline = &mat_pipelines[render_config.mat_pipeline as usize];
                    let (cl, pipeline_id) = if render_config.translucent {
                        consts.is_translucent_pass = 1;
                        (&mut *cl_trans, self.color_with_blending_pipe)
                    } else {
                        (&mut *cl_sol, self.color_pipe)
                    };
                    let pipeline = mat_pipeline.get_pipeline(pipeline_id).unwrap();
                    let signature = pipeline.signature();

                    cl.bind_pipeline(pipeline);

                    let descriptors = compose_descriptor_sets(
                        ctx.g_per_frame_in,
                        mat_pipeline.per_frame_desc,
                        renderable.descriptor_sets[GENERAL_OBJECT_DESCRIPTOR_IDX],
                    );

                    cl.bind_graphics_inputs(
                        signature,
                        0,
                        &descriptors,
                        &[
                            render_config.render_ty as u32 * ctx.per_frame_ub.aligned_element_size() as u32,
                            renderable.uniform_buf_index as u32 * BASIC_UNIFORM_BLOCK_MAX_SIZE as u32,
                        ],
                    );
                    cl.push_constants(signature, &consts);

                    cl.bind_and_draw_vertex_mesh(vertex_mesh);
                }

                cl_sol.end().unwrap();
                cl_trans.end().unwrap();
            });
    }

    fn record_overlay_cmd_list(&self, cl: &mut CmdList, framebuffer: &Framebuffer, ctx: &StageContext) {
        let mat_pipelines = ctx.material_pipelines;

        cl.begin_secondary_graphics(true, &self.render_pass, 1, Some(framebuffer))
            .unwrap();

        for renderable_id in ctx.ordered_entities {
            let Some(render_config) = ctx.storage.get::<MeshRenderConfigC>(renderable_id) else {
                continue;
            };
            if render_config.render_ty != RenderType::OVERLAY {
                continue;
            }

            let Some(vertex_mesh) = ctx.curr_vertex_meshes.get(renderable_id) else {
                continue;
            };
            let renderable = ctx.renderables.get(renderable_id).unwrap();

            let mat_pipeline = &mat_pipelines[render_config.mat_pipeline as usize];
            let pipeline = mat_pipeline.get_pipeline(self.overlay_pipe).unwrap();
            let signature = pipeline.signature();
            cl.bind_pipeline(pipeline);

            let descriptors = compose_descriptor_sets(
                ctx.g_per_frame_in,
                mat_pipeline.per_frame_desc,
                renderable.descriptor_sets[GENERAL_OBJECT_DESCRIPTOR_IDX],
            );

            cl.bind_graphics_inputs(
                signature,
                0,
                &descriptors,
                &[
                    render_config.render_ty as u32 * ctx.per_frame_ub.aligned_element_size() as u32,
                    renderable.uniform_buf_index as u32 * BASIC_UNIFORM_BLOCK_MAX_SIZE as u32,
                ],
            );
            let consts = GPassConsts {
                // Overlay is by default translucent and doesn't have translucency pass.
                // Set this to 0 to prevent shaders writing to translucency texture.
                is_translucent_pass: 0,
            };

            // TODO: FIXME on release target (push constants blocks are stripped when optimizations are on):
            // 2023-01-31T11:44:18.418Z ERROR [vulkan] [VAL] "Validation Error: [ VUID-vkCmdPushConstants-offset-01795 ] Object 0: handle = 0x142658008, name = overlay_secondary, type = VK_OBJECT_TYPE_COMMAND_BUFFER; | MessageID = 0x27bc88c6 | vkCmdPushConstants(): VK_SHADER_STAGE_ALL, VkPushConstantRange in VkPipelineLayout 0x3307610000000114[] overlapping offset = 0 and size = 4, do not contain VK_SHADER_STAGE_ALL. The Vulkan spec states: For each byte in the range specified by offset and size and for each shader gpu_executor in stageFlags, there must be a push constant range in layout that includes that byte and that gpu_executor (https://vulkan.lunarg.com/doc/view/1.3.239.0/mac/1.3-extensions/vkspec.html#VUID-vkCmdPushConstants-offset-01795)"
            cl.push_constants(signature, &consts);

            cl.bind_and_draw_vertex_mesh(vertex_mesh);
        }

        cl.end().unwrap();
    }
}

impl RenderStage for GBufferStage {
    fn name(&self) -> &'static str {
        "g-buffer-pass"
    }

    fn num_pipeline_kinds(&self) -> u32 {
        3
    }

    fn record_dependencies(&self) -> Vec<RenderStageId> {
        vec![RenderStageId::of::<DepthStage>()]
    }

    fn setup(&mut self, pipeline_kinds: &[PipelineKindId]) {
        self.color_pipe = pipeline_kinds[0];
        self.color_with_blending_pipe = pipeline_kinds[1];
        self.overlay_pipe = pipeline_kinds[2];
    }

    fn register_pipeline_kind(&self, params: MaterialPipelineParams, pipeline_set: &mut MaterialPipelineSet) {
        pipeline_set.prepare_pipeline(
            self.color_pipe,
            &PipelineConfig {
                render_pass: &self.render_pass,
                signature: params.main_signature,
                subpass_index: 0,
                cull_back_faces: params.cull_back_faces,
                blend_attachments: &[],
                depth_test: true,
                depth_write: false,
            },
        );
        pipeline_set.prepare_pipeline(
            self.color_with_blending_pipe,
            &PipelineConfig {
                render_pass: &self.render_pass,
                signature: &params.main_signature,
                subpass_index: 0,
                cull_back_faces: params.cull_back_faces,
                blend_attachments: &[ALBEDO_ATTACHMENT_ID],
                depth_test: true,
                depth_write: false,
            },
        );
        pipeline_set.prepare_pipeline(
            self.overlay_pipe,
            &PipelineConfig {
                render_pass: &self.render_pass,
                signature: &params.main_signature,
                subpass_index: 1,
                cull_back_faces: params.cull_back_faces,
                blend_attachments: &[ALBEDO_ATTACHMENT_ID],
                depth_test: true,
                depth_write: true,
            },
        );
    }

    fn run(
        &mut self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &dyn Any,
    ) -> StageRunResult {
        let ctx = ctx.downcast_ref::<StageContext>().unwrap();
        let available_threads = rayon::current_num_threads();

        let solid_objects_cmd_lists = resources.request_cmd_lists(
            "solid-objects-secondary",
            CmdListParams::secondary(QueueType::Graphics).with_count(available_threads),
        );
        let translucent_objects_cmd_lists = resources.request_cmd_lists(
            "translucent-objects-secondary",
            CmdListParams::secondary(QueueType::Graphics).with_count(available_threads),
        );
        let overlay_cmd_list = resources.request_cmd_list(
            "translucent-objects-secondary",
            CmdListParams::secondary(QueueType::Graphics),
        );

        let depth_image = resources.get_image(DepthStage::RES_DEPTH_IMAGE);

        let framebuffer = resources.request(
            Self::RES_FRAMEBUFFER,
            (
                ctx.render_size,
                Arc::clone(&self.render_pass),
                Arc::clone(&depth_image),
            ),
            |(render_size, render_pass, depth_image), _| {
                render_pass
                    .create_framebuffer(
                        *render_size,
                        &[
                            (
                                0,
                                ImageMod::AdditionalUsage(
                                    ImageUsageFlags::INPUT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                                ),
                            ),
                            (1, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                            (2, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                            (3, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                            (4, ImageMod::OverrideImage(Arc::clone(&depth_image))),
                        ],
                    )
                    .unwrap()
            },
        );

        let translucency_colors_image = resources.request_image(
            "translucency_colors",
            ImageParams::d2(
                Format::RGBA8_UNORM,
                ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST,
                ctx.render_size,
            ),
        );

        // ------------------------------------------------------------------------------------------

        self.record_g_cmd_lists(
            &mut solid_objects_cmd_lists.lock(),
            &mut translucent_objects_cmd_lists.lock(),
            &framebuffer,
            resources,
            ctx,
        );
        self.record_overlay_cmd_list(&mut overlay_cmd_list.lock(), &framebuffer, ctx);

        // ------------------------------------------------------------------------------------------

        cl.begin(true).unwrap();

        cl.barrier_image(
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::TRANSFER,
            &[translucency_colors_image
                .barrier()
                .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                .old_layout(ImageLayout::UNDEFINED)
                .new_layout(ImageLayout::GENERAL)],
        );
        cl.clear_image(
            &translucency_colors_image,
            ImageLayout::GENERAL,
            ClearValue::ColorF32([0.0; 4]),
        );
        cl.barrier_image(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::PIXEL_SHADER,
            &[translucency_colors_image
                .barrier()
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_WRITE)
                .layout(ImageLayout::GENERAL)],
        );

        // Main g-buffer pass
        cl.begin_render_pass(
            &self.render_pass,
            &framebuffer,
            &[
                ClearValue::Undefined,
                ClearValue::Undefined,
                ClearValue::Undefined,
                ClearValue::Undefined,
                ClearValue::Undefined,
                ClearValue::Depth(1.0),
            ],
            true,
        );
        cl.execute_secondary(
            solid_objects_cmd_lists
                .lock()
                .iter()
                .chain(translucent_objects_cmd_lists.lock().iter()),
        );

        // Overlay subpass
        cl.next_subpass(true);
        cl.execute_secondary(iter::once(&*overlay_cmd_list.lock()));

        cl.end_render_pass();

        StageRunResult::new()
    }
}
