use crate::ecs::component::render_config::RenderLayer;
use crate::ecs::component::uniform_data::BASIC_UNIFORM_BLOCK_MAX_SIZE;
use crate::ecs::component::MeshRenderConfigC;
use crate::module::main_renderer::material_pipeline::{MaterialPipelineSet, PipelineConfig, PipelineKindId};
use crate::module::main_renderer::resource_manager::{CmdListParams, ImageParams, ResourceManagementScope};
use crate::module::main_renderer::resources::{MaterialPipelineParams, GENERAL_OBJECT_DESCRIPTOR_IDX};
use crate::module::main_renderer::stage::depth::{DepthStage, VisibilityType};
use crate::module::main_renderer::stage::{
    FrameContext, RenderStage, RenderStageId, StageContext, StageRunResult,
};
use crate::module::main_renderer::vertex_mesh::VertexMeshCmdList;
use crate::module::main_renderer::{camera, compose_descriptor_sets, CameraInfo, FrameInfo};
use common::glm::Vec4;
use common::rayon::prelude::*;
use common::{glm, rayon};
use std::iter;
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, Attachment, AttachmentRef, BindingRes, ClearValue, CmdList, Device, Format, Framebuffer,
    ImageLayout, ImageMod, ImageUsageFlags, LoadStore, PipelineStageFlags, QueueType, RenderPass, Subpass,
    SubpassDependency,
};

const TRANSLUCENCY_N_DEPTH_LAYERS: u32 = shader_ids::OIT_N_CLOSEST_LAYERS;

pub struct GBufferStage {
    device: Arc<Device>,
    color_pipe: PipelineKindId,
    color_with_blending_pipe: PipelineKindId,
    overlay_pipe: PipelineKindId,
    render_pass: Arc<RenderPass>,
    frame_infos_indices: Vec<usize>,
}

impl GBufferStage {
    pub const POSITION_ATTACHMENT_ID: u32 = 0;
    pub const ALBEDO_ATTACHMENT_ID: u32 = 1;
    pub const SPECULAR_ATTACHMENT_ID: u32 = 2;
    pub const EMISSIVE_ATTACHMENT_ID: u32 = 3;
    pub const NORMAL_ATTACHMENT_ID: u32 = 4;
    pub const DEPTH_ATTACHMENT_ID: u32 = 5;
    pub const OVERLAY_DEPTH_ATTACHMENT_ID: u32 = 6;
    pub const RES_FRAMEBUFFER: &'static str = "g-framebuffer";
    pub const RES_TRANSLUCENCY_COLORS_IMAGE: &'static str = "translucency_colors_image";
    pub const RES_OVERLAY_DEPTH_IMAGE: &'static str = "g-overlay-depth";

    pub fn new(device: &Arc<Device>) -> Self {
        // Create G-Buffer pass resources
        // -----------------------------------------------------------------------------------------------------------------
        let render_pass = device
            .create_render_pass(
                &[
                    // Position
                    Attachment {
                        format: Format::RGBA32_FLOAT,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::InitClearFinalStore,
                    },
                    // Albedo
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::InitClearFinalStore,
                    },
                    // Specular
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::InitClearFinalStore,
                    },
                    // Emission
                    Attachment {
                        format: Format::RGBA16_FLOAT,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::InitClearFinalStore,
                    },
                    // Normal
                    Attachment {
                        format: Format::RG16_FLOAT,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::InitClearFinalStore,
                    },
                    // Depth (read)
                    Attachment {
                        format: Format::D32_FLOAT,
                        init_layout: ImageLayout::DEPTH_STENCIL_READ,
                        final_layout: ImageLayout::DEPTH_STENCIL_READ,
                        load_store: LoadStore::InitLoadFinalStore,
                    },
                    // Overlay depth (read/write)
                    Attachment {
                        format: Format::D32_FLOAT,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::DEPTH_STENCIL_READ,
                        load_store: LoadStore::InitClearFinalStore,
                    },
                ],
                &[
                    // Solid-colors pass
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
                            AttachmentRef {
                                index: 4,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                        ])
                        .with_depth(AttachmentRef {
                            index: 5,
                            layout: ImageLayout::DEPTH_STENCIL_READ,
                        }),
                    // Transparent pass
                    Subpass::new()
                        .with_color(vec![AttachmentRef {
                            index: 0,
                            layout: ImageLayout::COLOR_ATTACHMENT,
                        }])
                        .with_depth(AttachmentRef {
                            index: 5,
                            layout: ImageLayout::DEPTH_STENCIL_READ,
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
                            AttachmentRef {
                                index: 4,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                        ])
                        .with_depth(AttachmentRef {
                            index: 6,
                            layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                        }),
                ],
                &[
                    SubpassDependency {
                        src_subpass: 0,
                        dst_subpass: 1,
                        src_stage_mask: PipelineStageFlags::PIXEL_SHADER,
                        dst_stage_mask: PipelineStageFlags::PIXEL_SHADER,
                        src_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        dst_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    },
                    SubpassDependency {
                        src_subpass: 1,
                        dst_subpass: 2,
                        src_stage_mask: PipelineStageFlags::PIXEL_SHADER,
                        dst_stage_mask: PipelineStageFlags::PIXEL_SHADER,
                        src_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                        dst_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    },
                ],
            )
            .unwrap();

        Self {
            device: Arc::clone(device),
            color_pipe: u32::MAX,
            color_with_blending_pipe: u32::MAX,
            overlay_pipe: u32::MAX,
            render_pass,
            frame_infos_indices: vec![],
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
                    .begin_secondary_graphics(true, &self.render_pass, 1, Some(framebuffer))
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
                    if render_config.render_layer != RenderLayer::Main {
                        continue;
                    }
                    let Some(vertex_mesh) = ctx.curr_vertex_meshes.get(&renderable_id) else {
                        continue;
                    };
                    if visibility_host_buffer[entity_index] == 0 || vertex_mesh.is_empty() {
                        continue;
                    }

                    let renderable = ctx.renderables.get(&renderable_id).unwrap();

                    let mat_pipeline = &mat_pipelines[render_config.mat_pipeline as usize];
                    let (cl, pipeline_id) = if render_config.translucent {
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
                            self.frame_infos_indices[0] as u32 * ctx.per_frame_ub.element_size() as u32,
                            renderable.uniform_buf_index as u32 * BASIC_UNIFORM_BLOCK_MAX_SIZE as u32,
                        ],
                    );

                    cl.bind_and_draw_vertex_mesh(vertex_mesh);
                }

                cl_sol.end().unwrap();
                cl_trans.end().unwrap();
            });
    }

    fn record_overlay_cmd_list(&self, cl: &mut CmdList, framebuffer: &Framebuffer, ctx: &StageContext) {
        let mat_pipelines = ctx.material_pipelines;

        cl.begin_secondary_graphics(true, &self.render_pass, 2, Some(framebuffer))
            .unwrap();

        for renderable_id in ctx.ordered_entities {
            let Some(render_config) = ctx.storage.get::<MeshRenderConfigC>(renderable_id) else {
                continue;
            };
            if render_config.render_layer != RenderLayer::Overlay {
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
                    self.frame_infos_indices[1] as u32 * ctx.per_frame_ub.element_size() as u32,
                    renderable.uniform_buf_index as u32 * BASIC_UNIFORM_BLOCK_MAX_SIZE as u32,
                ],
            );

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

    fn num_per_frame_infos(&self) -> u32 {
        2
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
                cull: params.cull,
                blend_attachments: &[],
                depth_test: true,
                depth_write: false,
                spec_consts: &[(shader_ids::CONST_ID_PASS_TYPE, shader_ids::PASS_TYPE_G_BUFFER)],
            },
        );
        pipeline_set.prepare_pipeline(
            self.color_with_blending_pipe,
            &PipelineConfig {
                render_pass: &self.render_pass,
                signature: &params.main_signature,
                subpass_index: 1,
                cull: params.cull,
                blend_attachments: &[Self::ALBEDO_ATTACHMENT_ID],
                depth_test: true,
                depth_write: false,
                spec_consts: &[(
                    shader_ids::CONST_ID_PASS_TYPE,
                    shader_ids::PASS_TYPE_G_BUFFER_TRANSLUCENCY,
                )],
            },
        );
        pipeline_set.prepare_pipeline(
            self.overlay_pipe,
            &PipelineConfig {
                render_pass: &self.render_pass,
                signature: &params.main_signature,
                subpass_index: 2,
                cull: params.cull,
                blend_attachments: &[Self::ALBEDO_ATTACHMENT_ID],
                depth_test: true,
                depth_write: true,
                spec_consts: &[(
                    shader_ids::CONST_ID_PASS_TYPE,
                    shader_ids::PASS_TYPE_G_BUFFER_OVERLAY,
                )],
            },
        );
    }

    fn update_frame_infos(
        &mut self,
        infos: &mut [FrameInfo],
        frame_infos_indices: &[usize],
        ctx: &FrameContext,
    ) {
        self.frame_infos_indices = frame_infos_indices.to_vec();
        let overlay = &mut infos[frame_infos_indices[1]];

        let overlay_cam_dir = ctx.overlay_camera.direction();
        let overlay_proj = ctx.overlay_camera.projection();
        let overlay_view = camera::create_view_matrix(
            &glm::convert(*ctx.overlay_camera.position()),
            ctx.overlay_camera.rotation(),
        );

        overlay.camera = CameraInfo {
            pos: Vec4::from_element(0.0),
            dir: overlay_cam_dir.push(0.0),
            proj: overlay_proj,
            view: overlay_view,
            view_inverse: glm::inverse(&overlay_view),
            proj_view: overlay_proj * overlay_view,
            z_near: ctx.overlay_camera.z_near(),
            fovy: 0.0,
        };
    }

    fn run(
        &mut self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
    ) -> StageRunResult {
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
            "overlay-objects-secondary",
            CmdListParams::secondary(QueueType::Graphics),
        );

        let depth_image = resources.get_image(DepthStage::RES_DEPTH_IMAGE);
        let overlay_depth_image = resources.request_image(
            Self::RES_OVERLAY_DEPTH_IMAGE,
            ImageParams::d2(
                Format::D32_FLOAT,
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsageFlags::INPUT_ATTACHMENT
                    | ImageUsageFlags::SAMPLED,
                ctx.render_size,
            ),
        );

        let framebuffer = resources.request(
            Self::RES_FRAMEBUFFER,
            (
                ctx.render_size,
                Arc::clone(&self.render_pass),
                Arc::clone(&depth_image),
                Arc::clone(&overlay_depth_image),
            ),
            |(render_size, render_pass, depth_image, overlay_depth_image), _| {
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
                            (
                                1,
                                ImageMod::AdditionalUsage(
                                    ImageUsageFlags::INPUT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                                ),
                            ),
                            (
                                2,
                                ImageMod::AdditionalUsage(
                                    ImageUsageFlags::INPUT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                                ),
                            ),
                            (
                                3,
                                ImageMod::AdditionalUsage(
                                    ImageUsageFlags::INPUT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                                ),
                            ),
                            (
                                4,
                                ImageMod::AdditionalUsage(
                                    ImageUsageFlags::INPUT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                                ),
                            ),
                            (5, ImageMod::OverrideImage(Arc::clone(&depth_image))),
                            (6, ImageMod::OverrideImage(Arc::clone(&overlay_depth_image))),
                        ],
                    )
                    .unwrap()
            },
        );

        let translucency_colors_image = resources.request_image(
            Self::RES_TRANSLUCENCY_COLORS_IMAGE,
            ImageParams::d2_array(
                Format::RGBA8_UNORM,
                ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST,
                (ctx.render_size.0, ctx.render_size.1, TRANSLUCENCY_N_DEPTH_LAYERS),
            ),
        );

        // ------------------------------------------------------------------------------------------

        unsafe {
            self.device.update_descriptor_set(
                ctx.g_per_frame_in,
                &[ctx.g_per_frame_pool.create_binding(
                    shader_ids::BINDING_TRANSPARENCY_COLORS,
                    0,
                    BindingRes::Image(Arc::clone(&translucency_colors_image), None, ImageLayout::GENERAL),
                )],
            );
        }

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
                ClearValue::ColorU32([0; 4]),
                ClearValue::ColorU32([0; 4]),
                ClearValue::ColorU32([0; 4]),
                ClearValue::ColorU32([0; 4]),
                ClearValue::ColorU32([0; 4]),
                ClearValue::Undefined,
                ClearValue::Depth(0.0),
            ],
            true,
        );
        cl.execute_secondary(solid_objects_cmd_lists.lock().iter());

        // Transparency subpass
        cl.next_subpass(true);
        cl.execute_secondary(translucent_objects_cmd_lists.lock().iter());

        // Overlay subpass
        cl.next_subpass(true);
        cl.execute_secondary(iter::once(&*overlay_cmd_list.lock()));

        cl.end_render_pass();
        cl.end().unwrap();

        StageRunResult::new()
    }
}
