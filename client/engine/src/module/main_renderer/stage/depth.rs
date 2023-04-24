use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::render_config::RenderLayer;
use crate::ecs::component::MeshRenderConfigC;
use crate::module::main_renderer::camera::Frustum;
use crate::module::main_renderer::material_pipeline::{MaterialPipelineSet, PipelineConfig, PipelineKindId};
use crate::module::main_renderer::resource_manager::{
    CmdListParams, DeviceBufferParams, HostBufferParams, ImageParams, ResourceManagementScope,
};
use crate::module::main_renderer::resources::{MaterialPipelineParams, GENERAL_OBJECT_DESCRIPTOR_IDX};
use crate::module::main_renderer::stage::{FrameContext, RenderStage, StageContext, StageRunResult};
use crate::module::main_renderer::vertex_mesh::VertexMeshCmdList;
use crate::module::main_renderer::{
    calc_group_count, camera, compose_descriptor_sets, CameraInfo, FrameInfo,
};
use crate::module::scene::N_MAX_OBJECTS;
use common::glm::{Mat4, Vec2, Vec3, Vec4};
use common::parking_lot::Mutex;
use common::rayon::prelude::*;
use common::utils::prev_power_of_two;
use common::{glm, rayon};
use std::sync::Arc;
use std::{iter, mem};
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{
    AccessFlags, Attachment, AttachmentRef, BindingRes, BufferUsageFlags, ClearValue, CmdList, Device,
    Format, Framebuffer, ImageLayout, ImageMod, ImageUsageFlags, LoadStore, Pipeline, PipelineStageFlags,
    QueueType, RenderPass, Shader, ShaderStageFlags, Subpass, SubpassDependency,
};
use winit::event::VirtualKeyCode::M;

const TRANSLUCENCY_N_DEPTH_LAYERS: u32 = 4;

pub type VisibilityType = u32;

pub struct DepthStage {
    device: Arc<Device>,
    depth_render_pass: Arc<RenderPass>,
    translucency_depths_pixel_shader: Arc<Shader>,

    depth_pyramid_pipeline: Arc<Pipeline>,
    cull_pipeline: Arc<Pipeline>,

    shadow_map_render_pass: Arc<RenderPass>,

    depth_write_pipe: PipelineKindId,
    translucency_depths_pipe: PipelineKindId,
    shadow_map_pipe: PipelineKindId,
    frame_infos_indices: Vec<usize>,
    last_light_proj_view: Mat4,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullObject {
    pub sphere: Vec4,
    pub id: u32,
}

#[repr(C)]
struct DepthPyramidConstants {
    out_size: Vec2,
}

#[repr(C)]
struct CullConstants {
    pyramid_size: Vec2,
    max_pyramid_levels: u32,
    object_count: u32,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
pub struct ShadowCascade {
    pub split_depth: f32,
    pub proj_view: Mat4,
}

fn calc_direct_light_cascade_shadow_projections(
    cam_proj_view: &Mat4,
    z_near: f32,
    z_far: f32,
    n_cascades: u32,
    split_lambda: f32,
    light_dir: &Vec3,
) -> Vec<ShadowCascade> {
    let z_range = z_far - z_near;
    let z_ratio = z_far / z_near;

    let mut cascades = vec![ShadowCascade::default(); n_cascades as usize];
    let mut last_split_dist = 0.0;

    for (i, cascade) in cascades.iter_mut().enumerate() {
        let p = (i + 1) as f32 / n_cascades as f32;
        let log = z_near * z_ratio.powf(p);
        let uniform = z_near + z_range * p;
        let split_depth = split_lambda * log + (1.0 - split_lambda) * uniform;
        let split_dist = (split_depth - z_near) / z_range;

        let mut frustum_corners = [
            Vec3::new(-1.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(1.0, -1.0, 0.0),
            Vec3::new(-1.0, -1.0, 0.0),
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
        ];

        let inv_cam = glm::inverse(&cam_proj_view);

        // Project frustum corners into world space
        for corner in &mut frustum_corners {
            let inv_corner = inv_cam * corner.push(1.0);
            *corner = inv_corner.xyz() / inv_corner.w;
        }

        for i in 0..4 {
            let dist = frustum_corners[i + 4] - frustum_corners[i];
            frustum_corners[i + 4] = frustum_corners[i] + dist * split_dist;
            frustum_corners[i] = frustum_corners[i] + dist * last_split_dist;
        }

        let frustum_center = frustum_corners
            .iter()
            .cloned()
            .reduce(|acc, corner| acc + corner)
            .unwrap()
            / 8.0;

        let radius = frustum_corners
            .iter()
            .map(|corner| (corner - frustum_center).magnitude())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();
        let radius_rounded = (radius * 16.0).ceil() / 16.0;

        let max_extents = Vec3::from_element(radius_rounded);
        let min_extents = -max_extents;

        let light_proj = glm::ortho_rh_zo(
            min_extents.x,
            max_extents.x,
            min_extents.y,
            max_extents.y,
            max_extents.z - min_extents.z,
            0.0,
        );
        let light_view = glm::look_at_rh(
            &(frustum_center - light_dir * -min_extents.z),
            &frustum_center,
            &Vec3::new(0.0002324, 0.999224523453, 0.0006574),
        );

        *cascade = ShadowCascade {
            split_depth: -split_depth,
            proj_view: light_proj * light_view,
        };

        last_split_dist = split_dist;
    }

    cascades
}

impl DepthStage {
    pub const MAIN_LIGHT_DIR: Vec3 = Vec3::new(0.0, -1.0, 0.0);
    pub const RES_DEPTH_IMAGE: &'static str = "depth_image";
    pub const RES_VISIBILITY_HOST_BUFFER: &'static str = "visibility_host_buffer";
    pub const RES_TRANSLUCENCY_DEPTHS_IMAGE: &'static str = "translucency_depths_image";
    pub const RES_MAIN_SHADOW_MAP: &'static str = "main_shadow_map";
    pub const RES_LIGHT_PROJ_VIEW: &'static str = "depth-main_light_proj_view";

    pub fn new(device: &Arc<Device>) -> Self {
        let depth_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::D32_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::InitClearFinalStore,
                }],
                &[
                    Subpass::new().with_depth(AttachmentRef {
                        index: 0,
                        layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                    }),
                    Subpass::new()
                        .with_input(vec![AttachmentRef {
                            index: 0,
                            layout: ImageLayout::DEPTH_STENCIL_READ,
                        }])
                        .with_depth(AttachmentRef {
                            index: 0,
                            layout: ImageLayout::DEPTH_STENCIL_READ,
                        }),
                ],
                &[SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: 1,
                    src_stage_mask: PipelineStageFlags::LATE_TESTS_AND_DS_STORE,
                    dst_stage_mask: PipelineStageFlags::DS_LOAD_AND_EARLY_TESTS,
                    src_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    dst_access_mask: AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                    // src_stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                    // dst_stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                    // src_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    // dst_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                }],
            )
            .unwrap();

        let shadow_map_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::D32_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::InitClearFinalStore,
                }],
                &[Subpass::new().with_depth(AttachmentRef {
                    index: 0,
                    layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                })],
                &[],
            )
            .unwrap();

        // Depth pyramid pipeline
        // -----------------------------------------------------------------------------------------------------------------
        let depth_pyramid_compute = device
            .create_compute_shader(
                include_bytes!("../../../../shaders/build/depth_pyramid.comp.spv"),
                "depth_pyramid",
            )
            .unwrap();
        let depth_pyramid_signature = device
            .create_pipeline_signature(&[depth_pyramid_compute], &[])
            .unwrap();
        let depth_pyramid_pipeline = device.create_compute_pipeline(&depth_pyramid_signature).unwrap();

        // Cull pipeline
        // -----------------------------------------------------------------------------------------------------------------
        let cull_compute = device
            .create_compute_shader(include_bytes!("../../../../shaders/build/cull.comp.spv"), "cull")
            .unwrap();
        let cull_signature = device.create_pipeline_signature(&[cull_compute], &[]).unwrap();
        let cull_pipeline = device.create_compute_pipeline(&cull_signature).unwrap();

        // Translucency depth pipeline
        // -----------------------------------------------------------------------------------------------------------------
        let translucency_depths_pixel_shader = device
            .create_pixel_shader(
                include_bytes!("../../../../shaders/build/translucency_closest_depths.frag.spv"),
                "translucency_closest_depths",
            )
            .unwrap();

        Self {
            device: Arc::clone(device),
            depth_render_pass,
            translucency_depths_pixel_shader,
            depth_pyramid_pipeline,
            cull_pipeline,
            shadow_map_render_pass,
            depth_write_pipe: u32::MAX,
            translucency_depths_pipe: u32::MAX,
            shadow_map_pipe: u32::MAX,
            frame_infos_indices: vec![],
            last_light_proj_view: Default::default(),
        }
    }

    fn record_depth_cmd_lists(
        &mut self,
        depth_cls: &mut [CmdList],
        depth_translucency_cls: &mut [CmdList],
        framebuffer: &Framebuffer,
        ctx: &StageContext,
    ) -> Vec<CullObject> {
        let mat_pipelines = ctx.material_pipelines;
        let object_count = ctx.ordered_entities.len();
        let draw_count_step = object_count / depth_cls.len() + 1;
        let cull_objects = Mutex::new(Vec::<CullObject>::with_capacity(object_count));

        let proj_mat = ctx.active_camera.projection();
        let view_mat = camera::create_view_matrix(&ctx.relative_camera_pos, &ctx.active_camera.rotation());
        let frustum = Frustum::new(&(proj_mat * view_mat));

        depth_cls.par_iter_mut()
            .zip(depth_translucency_cls)
            .enumerate()
            .for_each(|(i, (cl_sol, cl_trans))| {
                let mut curr_cull_objects = Vec::with_capacity(draw_count_step);

                cl_sol
                    .begin_secondary_graphics(
                        false,
                        &self.depth_render_pass,
                        0,
                        Some(framebuffer),
                    )
                    .unwrap();
                cl_trans
                    .begin_secondary_graphics(
                        false,
                        &self.depth_render_pass,
                        1,
                        Some(framebuffer),
                    )
                    .unwrap();

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let renderable_id = ctx.ordered_entities[entity_index];
                    let entry = ctx.storage.entry(&renderable_id).unwrap();

                    let (Some(global_transform), Some(render_config), Some(vertex_mesh)) =
                        (entry.get::<GlobalTransformC>(), entry.get::<MeshRenderConfigC>(), ctx.curr_vertex_meshes.get(&renderable_id)) else {
                        continue;
                    };

                    if render_config.render_layer != RenderLayer::Main || !render_config.visible || vertex_mesh.is_empty() {
                        continue;
                    }

                    if let Some(sphere) = vertex_mesh.sphere() {
                        let center = sphere.center() + global_transform.position_f32();
                        let radius = sphere.radius() * global_transform.scale.max();

                        if !frustum.is_sphere_visible(&center, radius) {
                            continue;
                        }

                        curr_cull_objects.push(CullObject {
                            sphere: Vec4::new(center.x, center.y, center.z, radius),
                            id: entity_index as u32,
                        });
                    } else {
                        curr_cull_objects.push(CullObject {
                            sphere: Default::default(),
                            id: entity_index as u32,
                        });
                    }

                    let mat_pipeline = &mat_pipelines[render_config.mat_pipeline as usize];
                    let renderable = &ctx.renderables[&renderable_id];

                    let (cl, pipeline_id) = if render_config.translucent {
                        (&mut *cl_trans, self.translucency_depths_pipe)
                    } else {
                        (&mut *cl_sol, self.depth_write_pipe)
                    };
                    let pipeline = mat_pipeline.get_pipeline(pipeline_id).unwrap();
                    let signature = pipeline.signature();
                    cl.bind_pipeline(pipeline);

                    let descriptors = compose_descriptor_sets(
                        ctx.g_per_frame_in,
                        mat_pipeline.per_frame_desc,
                        renderable.descriptor_sets[GENERAL_OBJECT_DESCRIPTOR_IDX],
                    );

                    cl.bind_graphics_inputs(signature, 0, &descriptors, &[
                        self.frame_infos_indices[0] as u32 * ctx.per_frame_ub.element_size() as u32,
                        renderable.uniform_buf_index as u32 * ctx.uniform_buffer_basic.element_size() as u32
                    ]);

                    cl.bind_and_draw_vertex_mesh(vertex_mesh);
                }

                cl_sol.end().unwrap();
                cl_trans.end().unwrap();

                cull_objects.lock().extend(curr_cull_objects);
            });

        cull_objects.into_inner()
    }

    fn record_main_shadow_map_cmd_lists(
        &self,
        cls: &mut [CmdList],
        framebuffer: &Framebuffer,
        ctx: &StageContext,
        proj_view: &Mat4,
    ) {
        let mat_pipelines = ctx.material_pipelines;
        let object_count = ctx.ordered_entities.len();
        let draw_count_step = object_count / cls.len() + 1;
        let frustum = Frustum::new(proj_view);

        cls.par_iter_mut()
            .enumerate()
            .for_each(|(i, cl)| {
                cl.begin_secondary_graphics(false, framebuffer.render_pass(), 0, Some(framebuffer))
                    .unwrap();

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let renderable_id = ctx.ordered_entities[entity_index];
                    let entry = ctx.storage.entry(&renderable_id).unwrap();

                    let (Some(global_transform), Some(render_config), Some(vertex_mesh)) =
                        (entry.get::<GlobalTransformC>(), entry.get::<MeshRenderConfigC>(), ctx.curr_vertex_meshes.get(&renderable_id)) else {
                        continue;
                    };

                    if render_config.render_layer != RenderLayer::Main
                        || !render_config.visible
                        || render_config.translucent
                        || vertex_mesh.is_empty()
                    {
                        continue;
                    }

                    if let Some(sphere) = vertex_mesh.sphere() {
                        let center = sphere.center() + global_transform.position_f32();
                        let radius = sphere.radius() * global_transform.scale.max();

                        if !frustum.is_sphere_visible(&center, radius) {
                            continue;
                        }
                    }

                    let mat_pipeline = &mat_pipelines[render_config.mat_pipeline as usize];
                    let renderable = &ctx.renderables[&renderable_id];

                    let pipeline = mat_pipeline.get_pipeline(self.shadow_map_pipe).unwrap();
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
                            renderable.uniform_buf_index as u32 * ctx.uniform_buffer_basic.element_size() as u32,
                        ],
                    );

                    cl.bind_and_draw_vertex_mesh(vertex_mesh);
                }

                cl.end().unwrap();
            });
    }

    fn gen_shadow_maps(&mut self, cl: &mut CmdList, resources: &ResourceManagementScope, ctx: &StageContext) {
        let available_threads = rayon::current_num_threads();
        let cmd_lists = resources.request_cmd_lists(
            "depth-main_shadow_map_cl",
            CmdListParams::secondary(QueueType::Graphics).with_count(available_threads),
        );

        // TODO: may be extract this to renderer settings
        let shadow_map_size = (1024, 1024);

        let main_shadow_map = resources.request_image(
            Self::RES_MAIN_SHADOW_MAP,
            ImageParams::d2(
                Format::D32_FLOAT,
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
                shadow_map_size,
            ),
        );

        let main_shadow_framebuffer = resources.request(
            "depth-main_shadow_fb",
            (main_shadow_map,),
            |(main_shadow_map,), _| {
                self.shadow_map_render_pass
                    .create_framebuffer(
                        shadow_map_size,
                        &[(0, ImageMod::OverrideImage(Arc::clone(main_shadow_map)))],
                    )
                    .unwrap()
            },
        );

        self.record_main_shadow_map_cmd_lists(
            &mut cmd_lists.lock(),
            &main_shadow_framebuffer,
            ctx,
            &self.last_light_proj_view,
        );

        // Render solid objects for each cascade
        cl.begin_render_pass(
            &self.shadow_map_render_pass,
            &main_shadow_framebuffer,
            &[ClearValue::Depth(0.0)],
            true,
        );
        cl.execute_secondary(cmd_lists.lock().iter());
        cl.end_render_pass();

        resources.create(Self::RES_LIGHT_PROJ_VIEW, Arc::new(self.last_light_proj_view));
    }
}

impl RenderStage for DepthStage {
    fn name(&self) -> &'static str {
        "depth_pass"
    }

    fn num_pipeline_kinds(&self) -> u32 {
        3
    }

    fn num_per_frame_infos(&self) -> u32 {
        1 + 1 // main depth map + main shadow depth map
    }

    fn setup(&mut self, pipeline_kinds: &[PipelineKindId]) {
        self.depth_write_pipe = pipeline_kinds[0];
        self.translucency_depths_pipe = pipeline_kinds[1];
        self.shadow_map_pipe = pipeline_kinds[2];
    }

    fn update_frame_infos(
        &mut self,
        infos: &mut [FrameInfo],
        frame_infos_indices: &[usize],
        ctx: &FrameContext,
    ) {
        self.frame_infos_indices = frame_infos_indices.to_vec();

        let light_proj = glm::ortho_rh_zo(-256.0, 256.0, -256.0, 256.0, 512.0, 0.0);
        let light_view = glm::look_at_rh(
            &(ctx.relative_camera_pos + glm::vec3(0.0, 256.0, 0.0).component_mul(&(-ctx.main_light_dir))),
            &ctx.relative_camera_pos,
            &Vec3::new(0.0002324, 0.999224523453, 0.0006574),
        );
        let light_proj_view = light_proj * light_view;

        let shadow_info = &mut infos[frame_infos_indices[1]];
        shadow_info.camera = CameraInfo {
            pos: Default::default(), // TODO
            dir: Default::default(), // TODO
            proj: light_proj,
            view: light_view,
            view_inverse: Default::default(), // TODO
            proj_view: light_proj_view,
            z_near: 0.0,
            fovy: 0.0,
        };
        shadow_info.frame_size = glm::vec2(1024, 1024);

        self.last_light_proj_view = light_proj_view;
    }

    fn register_pipeline_kind(&self, params: MaterialPipelineParams, pipeline_set: &mut MaterialPipelineSet) {
        let combined_bindings: Vec<_> = params.main_signature.bindings().clone().into_iter().collect();

        let vertex_shader = Arc::clone(
            params
                .shaders
                .iter()
                .find(|v| v.stage() == ShaderStageFlags::VERTEX)
                .unwrap(),
        );

        let depth_signature = self
            .device
            .create_pipeline_signature(&[Arc::clone(&vertex_shader)], &combined_bindings)
            .unwrap();

        let translucency_depth_signature = self
            .device
            .create_pipeline_signature(
                &[
                    Arc::clone(&vertex_shader),
                    Arc::clone(&self.translucency_depths_pixel_shader),
                ],
                &combined_bindings,
            )
            .unwrap();

        pipeline_set.prepare_pipeline(
            self.depth_write_pipe,
            &PipelineConfig {
                render_pass: &self.depth_render_pass,
                signature: &depth_signature,
                subpass_index: 0,
                cull_back_faces: params.cull_back_faces,
                blend_attachments: &[],
                depth_test: true,
                depth_write: true,
                spec_consts: &[(shader_ids::CONST_ID_PASS_TYPE, shader_ids::PASS_TYPE_DEPTH)],
            },
        );
        pipeline_set.prepare_pipeline(
            self.translucency_depths_pipe,
            &PipelineConfig {
                render_pass: &self.depth_render_pass,
                signature: &translucency_depth_signature,
                subpass_index: 1,
                cull_back_faces: params.cull_back_faces,
                blend_attachments: &[],
                depth_test: true,
                depth_write: false,
                spec_consts: &[(
                    shader_ids::CONST_ID_PASS_TYPE,
                    shader_ids::PASS_TYPE_DEPTH_TRANSLUCENCY,
                )],
            },
        );
        pipeline_set.prepare_pipeline(
            self.shadow_map_pipe,
            &PipelineConfig {
                render_pass: &self.shadow_map_render_pass,
                signature: &depth_signature,
                subpass_index: 0,
                cull_back_faces: params.cull_back_faces,
                blend_attachments: &[],
                depth_test: true,
                depth_write: true,
                spec_consts: &[(
                    shader_ids::CONST_ID_PASS_TYPE,
                    shader_ids::PASS_TYPE_DEPTH_MAIN_SHADOW_MAP,
                )],
            },
        );
    }

    fn run(
        &mut self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
    ) -> StageRunResult {
        let available_threads = rayon::current_num_threads();

        let depth_cmd_lists = resources.request_cmd_lists(
            "depth-secondary",
            CmdListParams::secondary(QueueType::Graphics).with_count(available_threads),
        );
        let translucency_depths_cmd_lists = resources.request_cmd_lists(
            "depth-secondary-translucency_depths",
            CmdListParams::secondary(QueueType::Graphics).with_count(available_threads),
        );

        let depth_image = resources.request_image(
            Self::RES_DEPTH_IMAGE,
            ImageParams::d2(
                Format::D32_FLOAT,
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | ImageUsageFlags::INPUT_ATTACHMENT
                    | ImageUsageFlags::SAMPLED,
                ctx.render_size,
            ),
        );
        let depth_pyramid_image = resources.request_image(
            "depth_pyramid",
            ImageParams::d2(
                Format::R32_FLOAT,
                ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE,
                // Note: prev_power_of_two makes sure all reductions are at most by 2x2
                // which makes sure they are conservative
                (
                    prev_power_of_two(ctx.render_size.0),
                    prev_power_of_two(ctx.render_size.1),
                ),
            )
            .with_preferred_mip_levels(0),
        );

        let depth_pyramid_views = resources.request(
            "depth_pyramid_views",
            (Arc::clone(&depth_pyramid_image),),
            |(depth_pyramid_image,), _| {
                let views = (0..depth_pyramid_image.mip_levels())
                    .map(|i| {
                        depth_pyramid_image
                            .create_view_named(&format!("view-mip{}", i))
                            .base_mip_level(i)
                            .mip_level_count(1)
                            .build()
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                Arc::new(views)
            },
        );
        let depth_pyramid_levels = depth_pyramid_image.mip_levels();

        let depth_pyramid_descs = resources.request_descriptors(
            "depth-pyramid-descs",
            self.depth_pyramid_pipeline.signature(),
            0,
            depth_pyramid_levels as usize,
        );

        let translucency_depths_image = resources.request_device_buffer(
            Self::RES_TRANSLUCENCY_DEPTHS_IMAGE,
            DeviceBufferParams::new(
                BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_DST,
                mem::size_of::<u32>() as u64,
                (ctx.render_size.0 * ctx.render_size.1 * TRANSLUCENCY_N_DEPTH_LAYERS) as u64,
            ),
        );

        let framebuffer = resources.request(
            "depth-framebuffer",
            (
                ctx.render_size,
                Arc::clone(&self.depth_render_pass),
                Arc::clone(&depth_image),
            ),
            |(render_size, render_pass, depth_image), _| {
                render_pass
                    .create_framebuffer(
                        *render_size,
                        &[(0, ImageMod::OverrideImage(Arc::clone(depth_image)))],
                    )
                    .unwrap()
            },
        );

        let cull_host_buffer = resources.request_host_buffer::<CullObject>(
            "cull_host_buffer",
            HostBufferParams::new(BufferUsageFlags::TRANSFER_SRC, N_MAX_OBJECTS as u64),
        );

        let cull_buffer = resources.request_device_buffer(
            "cull_buffer",
            DeviceBufferParams::new(
                BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_DST,
                mem::size_of::<CullObject>() as u64,
                N_MAX_OBJECTS as u64,
            ),
        );

        let visibility_buffer = resources.request_device_buffer(
            "visibility_buffer",
            DeviceBufferParams::new(
                BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST,
                mem::size_of::<u32>() as u64,
                N_MAX_OBJECTS as u64,
            ),
        );

        let visibility_host_buffer = resources.request_host_buffer::<VisibilityType>(
            Self::RES_VISIBILITY_HOST_BUFFER,
            HostBufferParams::new(BufferUsageFlags::TRANSFER_DST, N_MAX_OBJECTS as u64),
        );

        let cull_descriptor =
            resources.request_descriptors("depth-cull-desc", self.cull_pipeline.signature(), 0, 1);

        // ----------------------------------------------------------------------------------

        unsafe {
            self.device.update_descriptor_set(
                ctx.g_per_frame_in,
                &[
                    ctx.g_per_frame_pool.create_binding(
                        shader_ids::BINDING_TRANSPARENCY_DEPTHS,
                        0,
                        BindingRes::Buffer(translucency_depths_image.handle()),
                    ),
                    ctx.g_per_frame_pool.create_binding(
                        shader_ids::BINDING_SOLID_DEPTHS,
                        0,
                        BindingRes::Image(Arc::clone(&depth_image), None, ImageLayout::DEPTH_STENCIL_READ),
                    ),
                ],
            );

            for (i, set) in depth_pyramid_descs.sets().iter().enumerate() {
                self.device.update_descriptor_set(
                    *set,
                    &[
                        depth_pyramid_descs.create_binding(
                            0,
                            0,
                            if i == 0 {
                                BindingRes::ImageView(
                                    Arc::clone(depth_image.view()),
                                    None,
                                    ImageLayout::SHADER_READ,
                                )
                            } else {
                                BindingRes::ImageView(
                                    Arc::clone(&depth_pyramid_views[i - 1]),
                                    None,
                                    ImageLayout::GENERAL,
                                )
                            },
                        ),
                        depth_pyramid_descs.create_binding(
                            1,
                            0,
                            BindingRes::ImageView(
                                Arc::clone(&depth_pyramid_views[i]),
                                None,
                                ImageLayout::GENERAL,
                            ),
                        ),
                    ],
                );
            }

            self.device.update_descriptor_set(
                cull_descriptor.get(0),
                &[
                    cull_descriptor.create_binding(
                        0,
                        0,
                        BindingRes::Image(Arc::clone(&depth_pyramid_image), None, ImageLayout::GENERAL),
                    ),
                    cull_descriptor.create_binding(1, 0, BindingRes::Buffer(ctx.per_frame_ub.handle())),
                    cull_descriptor.create_binding(2, 0, BindingRes::Buffer(cull_buffer.handle())),
                    cull_descriptor.create_binding(3, 0, BindingRes::Buffer(visibility_buffer.handle())),
                ],
            )
        };

        let cull_objects = self.record_depth_cmd_lists(
            &mut depth_cmd_lists.lock(),
            &mut translucency_depths_cmd_lists.lock(),
            &framebuffer,
            ctx,
        );
        let n_frustum_visible_objects = cull_objects.len();

        cull_host_buffer.lock().write(0, &cull_objects);

        // ----------------------------------------------------------------------------------

        cl.begin(true).unwrap();

        cl.fill_buffer(&*translucency_depths_image, u32::MAX);

        cl.barrier_buffer(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::PIXEL_SHADER,
            &[translucency_depths_image
                .barrier()
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)],
        );

        // Render solid objects
        cl.begin_render_pass(
            &self.depth_render_pass,
            &framebuffer,
            // 0.0 due to reversed-z
            &[ClearValue::Depth(0.0)],
            true,
        );
        cl.execute_secondary(depth_cmd_lists.lock().iter());

        // Find closest depths of translucent objects
        cl.next_subpass(true);
        cl.execute_secondary(translucency_depths_cmd_lists.lock().iter());
        cl.end_render_pass();

        // Render main shadow map
        // ----------------------------------------------------------------------------------------

        self.gen_shadow_maps(cl, resources, ctx);

        // Build depth pyramid
        // ----------------------------------------------------------------------------------------

        cl.barrier_image(
            PipelineStageFlags::ALL_GRAPHICS,
            PipelineStageFlags::COMPUTE,
            &[
                depth_image
                    .barrier()
                    .src_access_mask(AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)
                    .layout(ImageLayout::SHADER_READ),
                depth_pyramid_image
                    .barrier()
                    .dst_access_mask(AccessFlags::SHADER_WRITE)
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::GENERAL),
            ],
        );

        cl.bind_pipeline(&self.depth_pyramid_pipeline);

        let mut out_size = depth_pyramid_image.size_2d();

        for i in 0..(depth_pyramid_image.mip_levels() as usize) {
            cl.bind_compute_inputs(
                self.depth_pyramid_pipeline.signature(),
                0,
                &[depth_pyramid_descs.get(i)],
                &[],
            );

            let constants = DepthPyramidConstants {
                out_size: Vec2::new(out_size.0 as f32, out_size.1 as f32),
            };
            cl.push_constants(self.depth_pyramid_pipeline.signature(), &constants);

            cl.dispatch(calc_group_count(out_size.0), calc_group_count(out_size.1), 1);

            cl.barrier_image(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[depth_pyramid_image
                    .barrier()
                    .src_access_mask(AccessFlags::SHADER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)
                    .old_layout(ImageLayout::GENERAL)
                    .new_layout(ImageLayout::GENERAL)
                    .mip_levels(i as u32, 1)],
            );

            out_size = ((out_size.0 >> 1).max(1), (out_size.1 >> 1).max(1));
        }

        cl.barrier_image(
            PipelineStageFlags::COMPUTE,
            PipelineStageFlags::BOTTOM_OF_PIPE,
            &[depth_image
                .barrier()
                .src_access_mask(AccessFlags::SHADER_READ)
                .dst_access_mask(Default::default())
                .old_layout(ImageLayout::SHADER_READ)
                .new_layout(ImageLayout::DEPTH_STENCIL_READ)],
        );

        // Compute visibilities
        // ----------------------------------------------------------------------------------------

        cl.copy_buffer_to_device(
            &*cull_host_buffer.lock(),
            0,
            &*cull_buffer,
            0,
            n_frustum_visible_objects as u64,
        );
        cl.fill_buffer(&*visibility_buffer, 0);

        cl.barrier_buffer(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::COMPUTE,
            &[
                cull_buffer
                    .barrier()
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ),
                visibility_buffer
                    .barrier()
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_WRITE),
            ],
        );

        cl.bind_pipeline(&self.cull_pipeline);
        cl.bind_compute_inputs(self.cull_pipeline.signature(), 0, &[cull_descriptor.get(0)], &[]);

        let pyramid_size = depth_pyramid_image.size_2d();
        let constants = CullConstants {
            pyramid_size: Vec2::new(pyramid_size.0 as f32, pyramid_size.1 as f32),
            max_pyramid_levels: depth_pyramid_image.mip_levels(),
            object_count: n_frustum_visible_objects as u32,
        };
        cl.push_constants(self.cull_pipeline.signature(), &constants);

        cl.dispatch(calc_group_count(n_frustum_visible_objects as u32), 1, 1);

        cl.barrier_buffer(
            PipelineStageFlags::COMPUTE,
            PipelineStageFlags::TRANSFER,
            &[visibility_buffer
                .barrier()
                .src_access_mask(AccessFlags::SHADER_WRITE)
                .dst_access_mask(AccessFlags::TRANSFER_READ)],
        );

        let object_count = ctx.ordered_entities.len() as u32;
        cl.copy_buffer_to_host(
            &*visibility_buffer,
            0,
            &*visibility_host_buffer.lock(),
            0,
            object_count as u64,
        );

        cl.end().unwrap();

        StageRunResult::new()
    }
}
