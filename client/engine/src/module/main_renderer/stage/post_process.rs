use crate::module::main_renderer::resource_manager::ResourceManagementScope;
use crate::module::main_renderer::stage::depth::DepthStage;
use crate::module::main_renderer::stage::g_buffer::GBufferStage;
use crate::module::main_renderer::stage::{RenderStage, RenderStageId, StageContext, StageRunResult};
use common::glm;
use common::glm::{Mat4, Vec2, Vec4};
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::image::ImageParams;
use vk_wrapper::sampler::SamplerClamp;
use vk_wrapper::{
    AccessFlags, Attachment, AttachmentColorBlend, AttachmentRef, BindingRes, CmdList, Device, DeviceBuffer,
    Format, Framebuffer, Image, ImageLayout, ImageMod, ImageUsageFlags, ImageView, LoadStore, Pipeline,
    PipelineStageFlags, PrimitiveTopology, RenderPass, Sampler, SamplerFilter, SamplerMipmap, Shader,
    Subpass, SubpassDependency,
};

pub struct PostProcessStage {
    device: Arc<Device>,
    quad_shader: Arc<Shader>,
    custom_process_render_pass: Arc<RenderPass>,

    merge_render_pass: Arc<RenderPass>,
    merge_pipeline: Arc<Pipeline>,

    combine_main_overlay_render_pass: Arc<RenderPass>,
    combine_main_overlay_pipeline: Arc<Pipeline>,

    bloom_downscale_render_pass: Arc<RenderPass>,
    bloom_upscale_render_pass: Arc<RenderPass>,
    bloom_downscale_pipeline: Arc<Pipeline>,
    bloom_upscale_pipeline: Arc<Pipeline>,
    bloom_sampler: Arc<Sampler>,

    post_processes: Vec<(String, Arc<Pipeline>)>,

    tonemap_render_pass: Arc<RenderPass>,
    tonemap_pipeline: Arc<Pipeline>,

    shadow_map_sampler: Arc<Sampler>,
}

#[repr(C)]
struct DownscalePushConstants {
    src_resolution: Vec2,
}

#[repr(C)]
struct UpscalePushConstants {
    filter_radius: f32,
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct MainShadowInfo {
    light_view: Mat4,
    light_proj_view: Mat4,
    light_dir: Vec4,
}

impl PostProcessStage {
    pub const RES_OUTPUT_FRAMEBUFFER: &'static str = "post-output-fb";
    const RES_MERGE_FRAMEBUFFER: &'static str = "post-merge_framebuffer";

    pub fn new(device: &Arc<Device>) -> Self {
        let merge_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::RGBA16_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::FinalSave,
                }],
                &[Subpass::new().with_color(vec![AttachmentRef {
                    index: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT,
                }])],
                &[SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: Subpass::EXTERNAL,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    dst_stage_mask: PipelineStageFlags::PIXEL_SHADER,
                    src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
                    dst_access_mask: AccessFlags::SHADER_READ,
                }],
            )
            .unwrap();

        let quad_shader = device
            .create_vertex_shader(
                include_bytes!("../../../../shaders/build/quad.vert.spv"),
                &[],
                "quad.vert",
            )
            .unwrap();
        let merge_pixel_shader = device
            .create_pixel_shader(
                include_bytes!("../../../../shaders/build/g_buffer_merge.frag.spv"),
                "g_buffer_merge.frag",
            )
            .unwrap();
        let merge_signature = device
            .create_pipeline_signature(&[Arc::clone(&quad_shader), merge_pixel_shader], &[])
            .unwrap();
        let merge_pipeline = device
            .create_graphics_pipeline(
                &merge_render_pass,
                0,
                PrimitiveTopology::TRIANGLE_LIST,
                Default::default(),
                Default::default(),
                &[],
                &merge_signature,
                &[],
            )
            .unwrap();

        let combine_main_overlay_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::RGBA16_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::FinalSave,
                }],
                &[Subpass::new().with_color(vec![AttachmentRef {
                    index: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT,
                }])],
                &[],
            )
            .unwrap();

        let combine_main_overlay_pixel_shader = device
            .create_pixel_shader(
                include_bytes!("../../../../shaders/build/combine_main_and_overlay.frag.spv"),
                "combine_main_and_overlay.frag",
            )
            .unwrap();
        let combine_main_overlay_signature = device
            .create_pipeline_signature(
                &[Arc::clone(&quad_shader), combine_main_overlay_pixel_shader],
                &[],
            )
            .unwrap();
        let combine_main_overlay_pipeline = device
            .create_graphics_pipeline(
                &combine_main_overlay_render_pass,
                0,
                PrimitiveTopology::TRIANGLE_LIST,
                Default::default(),
                Default::default(),
                &[],
                &combine_main_overlay_signature,
                &[],
            )
            .unwrap();

        let bloom_downscale_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::RGBA16_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::FinalSave,
                }],
                &[Subpass::new().with_color(vec![AttachmentRef {
                    index: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT,
                }])],
                &[SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: Subpass::EXTERNAL,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | PipelineStageFlags::PIXEL_SHADER,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | PipelineStageFlags::PIXEL_SHADER,
                    src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::SHADER_READ,
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::SHADER_READ,
                    // src_stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                    // dst_stage_mask: PipelineStageFlags::ALL_GRAPHICS,
                    // src_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                    // dst_access_mask: AccessFlags::MEMORY_READ | AccessFlags::MEMORY_WRITE,
                }],
            )
            .unwrap();
        let bloom_upscale_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::RGBA16_FLOAT,
                    init_layout: ImageLayout::SHADER_READ,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::InitLoadFinalStore,
                }],
                &[Subpass::new().with_color(vec![AttachmentRef {
                    index: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT,
                }])],
                &[SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: Subpass::EXTERNAL,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | PipelineStageFlags::PIXEL_SHADER,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | PipelineStageFlags::PIXEL_SHADER,
                    src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::SHADER_READ,
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::SHADER_READ,
                }],
            )
            .unwrap();

        let bloom_downscale_pixel_shader = device
            .create_pixel_shader(
                include_bytes!("../../../../shaders/build/bloom_downscale.frag.spv"),
                "bloom_downscale.frag",
            )
            .unwrap();
        let bloom_upscale_pixel_shader = device
            .create_pixel_shader(
                include_bytes!("../../../../shaders/build/bloom_upscale.frag.spv"),
                "bloom_upscale.frag",
            )
            .unwrap();
        let bloom_downscale_signature = device
            .create_pipeline_signature(&[Arc::clone(&quad_shader), bloom_downscale_pixel_shader], &[])
            .unwrap();
        let bloom_upscale_signature = device
            .create_pipeline_signature(&[Arc::clone(&quad_shader), bloom_upscale_pixel_shader], &[])
            .unwrap();

        let bloom_downscale_pipeline = device
            .create_graphics_pipeline(
                &bloom_downscale_render_pass,
                0,
                PrimitiveTopology::TRIANGLE_LIST,
                Default::default(),
                Default::default(),
                &[],
                &bloom_downscale_signature,
                &[],
            )
            .unwrap();

        let bloom_upscale_pipeline = device
            .create_graphics_pipeline(
                &bloom_upscale_render_pass,
                0,
                PrimitiveTopology::TRIANGLE_LIST,
                Default::default(),
                Default::default(),
                &[(0, AttachmentColorBlend::additive())],
                &bloom_upscale_signature,
                &[],
            )
            .unwrap();

        let custom_process_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::RGBA16_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::FinalSave,
                }],
                &[Subpass::new().with_color(vec![AttachmentRef {
                    index: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT,
                }])],
                &[SubpassDependency {
                    src_subpass: 0,
                    dst_subpass: Subpass::EXTERNAL,
                    src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | PipelineStageFlags::PIXEL_SHADER,
                    dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | PipelineStageFlags::PIXEL_SHADER,
                    src_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::SHADER_READ,
                    dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE | AccessFlags::SHADER_READ,
                }],
            )
            .unwrap();

        let tonemap_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::RGBA16_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::FinalSave,
                }],
                &[Subpass::new().with_color(vec![AttachmentRef {
                    index: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT,
                }])],
                &[],
            )
            .unwrap();

        let tonemap_pixel_shader = device
            .create_pixel_shader(
                include_bytes!("../../../../shaders/build/tonemap.frag.spv"),
                "tonemap.frag",
            )
            .unwrap();
        let tonemap_signature = device
            .create_pipeline_signature(&[Arc::clone(&quad_shader), tonemap_pixel_shader], &[])
            .unwrap();
        let tonemap_pipeline = device
            .create_graphics_pipeline(
                &tonemap_render_pass,
                0,
                PrimitiveTopology::TRIANGLE_LIST,
                Default::default(),
                Default::default(),
                &[],
                &tonemap_signature,
                &[],
            )
            .unwrap();

        let bloom_sampler = device
            .create_sampler(
                SamplerFilter::LINEAR,
                SamplerFilter::LINEAR,
                SamplerMipmap::NEAREST,
                SamplerClamp::CLAMP_TO_EDGE,
                1.0,
            )
            .unwrap();

        let shadow_map_sampler = device
            .create_sampler(
                SamplerFilter::NEAREST,
                SamplerFilter::NEAREST,
                SamplerMipmap::NEAREST,
                SamplerClamp::CLAMP_TO_EDGE,
                1.0,
            )
            .unwrap();

        Self {
            device: Arc::clone(device),
            quad_shader,
            custom_process_render_pass,
            merge_render_pass,
            merge_pipeline,
            combine_main_overlay_render_pass,
            combine_main_overlay_pipeline,
            bloom_downscale_render_pass,
            bloom_upscale_render_pass,
            bloom_downscale_pipeline,
            bloom_upscale_pipeline,
            bloom_sampler,
            post_processes: vec![],
            tonemap_render_pass,
            tonemap_pipeline,
            shadow_map_sampler,
        }
    }

    pub fn add_custom_post_process(&mut self, name: &str, pixel_shader: Arc<Shader>) {
        let signature = self
            .device
            .create_pipeline_signature(&[Arc::clone(&self.quad_shader), pixel_shader], &[])
            .unwrap();

        let pipeline = self
            .device
            .create_graphics_pipeline(
                &self.custom_process_render_pass,
                0,
                PrimitiveTopology::TRIANGLE_LIST,
                Default::default(),
                Default::default(),
                &[],
                &signature,
                &[],
            )
            .unwrap();

        self.post_processes.push((name.to_string(), pipeline));
    }

    fn g_buffer_merge_stage(
        &self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
    ) -> Arc<Image> {
        let merge_framebuffer = resources.request(
            Self::RES_MERGE_FRAMEBUFFER,
            (ctx.render_size,),
            |(render_size,), _| {
                self.merge_render_pass
                    .create_framebuffer(
                        *render_size,
                        &[(0, ImageMod::AdditionalUsage(ImageUsageFlags::SAMPLED))],
                    )
                    .unwrap()
            },
        );
        let merge_descriptor =
            resources.request_descriptors("post-merge-desc", self.merge_pipeline.signature(), 0, 1);

        let g_framebuffer: Arc<Framebuffer> = resources.get(GBufferStage::RES_FRAMEBUFFER);

        let translucency_depths_image: Arc<DeviceBuffer> =
            resources.get(DepthStage::RES_TRANSLUCENCY_DEPTHS_IMAGE);
        let translucency_colors_image: Arc<Image> =
            resources.get(GBufferStage::RES_TRANSLUCENCY_COLORS_IMAGE);

        let g_position = g_framebuffer
            .get_image(GBufferStage::POSITION_ATTACHMENT_ID)
            .unwrap();
        let g_albedo = g_framebuffer
            .get_image(GBufferStage::ALBEDO_ATTACHMENT_ID)
            .unwrap();
        let g_specular = g_framebuffer
            .get_image(GBufferStage::SPECULAR_ATTACHMENT_ID)
            .unwrap();
        let g_emissive = g_framebuffer
            .get_image(GBufferStage::EMISSIVE_ATTACHMENT_ID)
            .unwrap();
        let g_normal = g_framebuffer
            .get_image(GBufferStage::NORMAL_ATTACHMENT_ID)
            .unwrap();
        let g_depth = g_framebuffer
            .get_image(GBufferStage::DEPTH_ATTACHMENT_ID)
            .unwrap();

        let main_shadow_map = resources.get_image(DepthStage::RES_MAIN_SHADOW_MAP);
        let main_light_proj: Arc<Mat4> = resources.get(DepthStage::RES_LIGHT_PROJ);
        let main_light_view: Arc<Mat4> = resources.get(DepthStage::RES_LIGHT_VIEW);

        let main_shadow_info = MainShadowInfo {
            light_view: *main_light_view,
            light_proj_view: *main_light_proj * *main_light_view,
            light_dir: DepthStage::MAIN_LIGHT_DIR.push(0.0),
        };

        let main_shadow_info_ub =
            resources.request_uniform_buffer("post-main_shadow_ub", main_shadow_info, cl);

        unsafe {
            self.device.update_descriptor_set(
                merge_descriptor.get(0),
                &[
                    merge_descriptor.create_binding(
                        0,
                        0,
                        BindingRes::Image(Arc::clone(g_position), None, ImageLayout::SHADER_READ),
                    ),
                    merge_descriptor.create_binding(
                        1,
                        0,
                        BindingRes::Image(Arc::clone(g_albedo), None, ImageLayout::SHADER_READ),
                    ),
                    merge_descriptor.create_binding(
                        2,
                        0,
                        BindingRes::Image(Arc::clone(g_specular), None, ImageLayout::SHADER_READ),
                    ),
                    merge_descriptor.create_binding(
                        3,
                        0,
                        BindingRes::Image(Arc::clone(g_emissive), None, ImageLayout::SHADER_READ),
                    ),
                    merge_descriptor.create_binding(
                        4,
                        0,
                        BindingRes::Image(Arc::clone(g_normal), None, ImageLayout::SHADER_READ),
                    ),
                    merge_descriptor.create_binding(
                        5,
                        0,
                        BindingRes::Image(Arc::clone(g_depth), None, ImageLayout::DEPTH_STENCIL_READ),
                    ),
                    merge_descriptor.create_binding(6, 0, BindingRes::Buffer(ctx.per_frame_ub.handle())),
                    merge_descriptor.create_binding(
                        7,
                        0,
                        BindingRes::Buffer(translucency_depths_image.handle()),
                    ),
                    merge_descriptor.create_binding(
                        8,
                        0,
                        BindingRes::Image(Arc::clone(&translucency_colors_image), None, ImageLayout::GENERAL),
                    ),
                    merge_descriptor.create_binding(
                        9,
                        0,
                        BindingRes::Image(
                            Arc::clone(&main_shadow_map),
                            Some(Arc::clone(&self.shadow_map_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                    merge_descriptor.create_binding(10, 0, BindingRes::Buffer(main_shadow_info_ub.handle())),
                ],
            )
        }

        cl.begin_render_pass(&self.merge_render_pass, &merge_framebuffer, &[], false);
        cl.bind_pipeline(&self.merge_pipeline);
        cl.bind_graphics_inputs(
            self.merge_pipeline.signature(),
            0,
            &[merge_descriptor.get(0)],
            &[],
        );
        cl.draw(3, 0);
        cl.end_render_pass();

        Arc::clone(merge_framebuffer.get_image(0).unwrap())
    }

    fn bloom_stage(
        &self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
        combined_source_image: &Arc<Image>,
    ) -> Arc<ImageView> {
        // Algorithm overview (example with four mipmaps)
        // 1. Downscaling
        // I1 = downscale(SOURCE);
        // I2 = downscale(I1);
        // I3 = downscale(I2);
        // I4 = downscale(I3);
        //
        // 2. Upscaling
        // I3 += blur(I4, radius3)
        // I2 += blur(I3, radius2)
        // I1 += blur(I2, radius1)
        //
        // FINAL = mix(SOURCE, I1, factor)

        const BLURRED_MAX_SIZE: u32 = 2_u32.pow(9);
        const BLURRED_MAX_MIPS: u32 = 6;

        let render_size_min = ctx.render_size.0.min(ctx.render_size.1);
        let blur_img_scale_factor = (BLURRED_MAX_SIZE as f32 / render_size_min as f32).min(1.0);
        let blurred_image_size = (
            (ctx.render_size.0 as f32 * blur_img_scale_factor) as u32,
            (ctx.render_size.1 as f32 * blur_img_scale_factor) as u32,
        );

        let bloom_image = resources.request_image(
            "post-blurred-output",
            ImageParams::d2(
                Format::RGBA16_FLOAT,
                ImageUsageFlags::COLOR_ATTACHMENT | ImageUsageFlags::SAMPLED,
                blurred_image_size,
            )
            .with_preferred_mip_levels(BLURRED_MAX_MIPS),
        );

        let bloom_image_views = resources.request(
            "post-blurred-image-view",
            (Arc::clone(&bloom_image),),
            |(blurred_image,), _| {
                let views: Vec<_> = (0..blurred_image.mip_levels())
                    .map(|i| {
                        blurred_image
                            .create_view()
                            .base_mip_level(i)
                            .mip_level_count(1)
                            .build()
                            .unwrap()
                    })
                    .collect();
                Arc::new(views)
            },
        );

        let bloom_framebuffers = resources.request(
            "post-bloom-downscale-fb",
            (Arc::clone(&bloom_image_views),),
            |(blurred_image_views,), _| {
                let fbs: Vec<_> = blurred_image_views
                    .iter()
                    .enumerate()
                    .map(|(i, view)| {
                        self.bloom_downscale_render_pass
                            .create_framebuffer(
                                bloom_image.mip_size(i as u32),
                                &[(0, ImageMod::OverrideImageView(Arc::clone(&view)))],
                            )
                            .unwrap()
                    })
                    .collect();
                Arc::new(fbs)
            },
        );

        let downscale_descs = resources.request_descriptors(
            "post-downscale-desc",
            self.bloom_downscale_pipeline.signature(),
            0,
            bloom_image.mip_levels() as usize,
        );
        let upscale_descs = resources.request_descriptors(
            "post-upscale-desc",
            self.bloom_upscale_pipeline.signature(),
            0,
            bloom_image.mip_levels() as usize - 1,
        );

        unsafe {
            for (i, set) in downscale_descs.sets().iter().enumerate() {
                self.device.update_descriptor_set(
                    *set,
                    &[downscale_descs.create_binding(
                        0,
                        0,
                        BindingRes::ImageView(
                            if i == 0 {
                                Arc::clone(combined_source_image.view())
                            } else {
                                Arc::clone(&bloom_image_views[i - 1])
                            },
                            Some(Arc::clone(&self.bloom_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    )],
                );
            }
            for (i, set) in upscale_descs.sets().iter().enumerate() {
                self.device.update_descriptor_set(
                    *set,
                    &[upscale_descs.create_binding(
                        0,
                        0,
                        BindingRes::ImageView(
                            Arc::clone(&bloom_image_views[bloom_image_views.len() - 1 - i]),
                            Some(Arc::clone(&self.bloom_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    )],
                );
            }
        }

        // ------------------------------------------------------------------------------------

        // Do downscaling
        cl.bind_pipeline(&self.bloom_downscale_pipeline);

        for (i, (fb, desc)) in bloom_framebuffers.iter().zip(downscale_descs.sets()).enumerate() {
            cl.begin_render_pass(&self.bloom_downscale_render_pass, fb, &[], false);
            cl.bind_graphics_inputs(self.bloom_downscale_pipeline.signature(), 0, &[*desc], &[]);

            let src_size = if i == 0 {
                combined_source_image.size_2d()
            } else {
                bloom_image.mip_size((i - 1) as u32)
            };

            cl.push_constants(
                self.bloom_downscale_pipeline.signature(),
                &DownscalePushConstants {
                    src_resolution: glm::vec2(src_size.0 as f32, src_size.1 as f32),
                },
            );
            cl.draw(3, 0);
            cl.end_render_pass();
        }

        // Do upscaling
        cl.bind_pipeline(&self.bloom_upscale_pipeline);
        cl.push_constants(
            self.bloom_upscale_pipeline.signature(),
            &UpscalePushConstants { filter_radius: 0.005 },
        );

        for (fb, desc) in bloom_framebuffers[0..bloom_framebuffers.len() - 1]
            .iter()
            .rev()
            .zip(upscale_descs.sets())
        {
            cl.begin_render_pass(&self.bloom_upscale_render_pass, fb, &[], false);
            cl.bind_graphics_inputs(self.bloom_upscale_pipeline.signature(), 0, &[*desc], &[]);
            cl.draw(3, 0);
            cl.end_render_pass();
        }

        Arc::clone(&bloom_image_views[0])
    }

    fn custom_post_processing(
        &self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
        input_image: Arc<Image>,
        input_depth: &Arc<Image>,
    ) -> Arc<Image> {
        let mut curr_back_image = input_image;

        for (name, pipeline) in &self.post_processes {
            let fb = resources.request(
                &format!("post_{name}_fb"),
                (ctx.render_size,),
                |(render_size,), _| {
                    self.custom_process_render_pass
                        .create_framebuffer(
                            *render_size,
                            &[(0, ImageMod::AdditionalUsage(ImageUsageFlags::SAMPLED))],
                        )
                        .unwrap()
                },
            );

            let resources_desc =
                resources.request_descriptors(&format!("post_{name}_desc"), pipeline.signature(), 0, 1);

            unsafe {
                self.device.update_descriptor_set(
                    resources_desc.get(0),
                    &[
                        resources_desc.create_binding(
                            0,
                            0,
                            BindingRes::Image(Arc::clone(&curr_back_image), None, ImageLayout::SHADER_READ),
                        ),
                        resources_desc.create_binding(
                            1,
                            0,
                            BindingRes::Image(
                                Arc::clone(&input_depth),
                                None,
                                ImageLayout::DEPTH_STENCIL_READ,
                            ),
                        ),
                        resources_desc.create_binding(2, 0, BindingRes::Buffer(ctx.per_frame_ub.handle())),
                    ],
                );
            }

            cl.bind_pipeline(pipeline);
            cl.begin_render_pass(&self.custom_process_render_pass, &fb, &[], false);

            cl.bind_graphics_inputs(pipeline.signature(), 0, &[resources_desc.get(0)], &[]);
            cl.draw(3, 0);

            cl.end_render_pass();

            curr_back_image = Arc::clone(fb.get_image(0).unwrap());
        }

        curr_back_image
    }

    fn combine_main_and_overlay_stage(
        &self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
        main_source_image: &Arc<Image>,
        g_overlay_albedo: &Arc<Image>,
    ) -> Arc<Image> {
        let desc = resources.request_descriptors(
            "post_combine-main-overlay_desc",
            self.combine_main_overlay_pipeline.signature(),
            0,
            1,
        );

        unsafe {
            self.device.update_descriptor_set(
                desc.get(0),
                &[
                    desc.create_binding(
                        0,
                        0,
                        BindingRes::Image(Arc::clone(&main_source_image), None, ImageLayout::SHADER_READ),
                    ),
                    desc.create_binding(
                        1,
                        0,
                        BindingRes::Image(Arc::clone(&g_overlay_albedo), None, ImageLayout::SHADER_READ),
                    ),
                ],
            );
        }

        let framebuffer = resources.request(
            "post_combine-main-overlay",
            (ctx.render_size,),
            |(render_size,), _| {
                self.combine_main_overlay_render_pass
                    .create_framebuffer(
                        *render_size,
                        &[(0, ImageMod::AdditionalUsage(ImageUsageFlags::SAMPLED))],
                    )
                    .unwrap()
            },
        );

        // Mix source and bloom
        cl.bind_pipeline(&self.combine_main_overlay_pipeline);
        cl.bind_graphics_inputs(
            self.combine_main_overlay_pipeline.signature(),
            0,
            &[desc.get(0)],
            &[],
        );
        cl.begin_render_pass(&self.combine_main_overlay_render_pass, &framebuffer, &[], false);
        cl.draw(3, 0);
        cl.end_render_pass();

        Arc::clone(framebuffer.get_image(0).unwrap())
    }

    fn tonemap_stage(
        &self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
        g_overlay_albedo: &Arc<Image>,
        main_source_image: &Arc<Image>,
        bloom: &Arc<ImageView>,
    ) {
        let tonemap_desc =
            resources.request_descriptors("post-tonemap-desc", self.tonemap_pipeline.signature(), 0, 1);

        unsafe {
            self.device.update_descriptor_set(
                tonemap_desc.get(0),
                &[
                    tonemap_desc.create_binding(
                        0,
                        0,
                        BindingRes::Image(Arc::clone(&main_source_image), None, ImageLayout::SHADER_READ),
                    ),
                    tonemap_desc.create_binding(
                        1,
                        0,
                        BindingRes::Image(Arc::clone(&g_overlay_albedo), None, ImageLayout::SHADER_READ),
                    ),
                    tonemap_desc.create_binding(
                        2,
                        0,
                        BindingRes::ImageView(Arc::clone(bloom), None, ImageLayout::SHADER_READ),
                    ),
                ],
            );
        }

        let tonemap_fb = resources.request(
            Self::RES_OUTPUT_FRAMEBUFFER,
            (ctx.render_size,),
            |(render_size,), _| {
                self.tonemap_render_pass
                    .create_framebuffer(
                        *render_size,
                        &[(0, ImageMod::AdditionalUsage(ImageUsageFlags::SAMPLED))],
                    )
                    .unwrap()
            },
        );

        // Mix source and bloom
        cl.bind_pipeline(&self.tonemap_pipeline);
        cl.bind_graphics_inputs(self.tonemap_pipeline.signature(), 0, &[tonemap_desc.get(0)], &[]);
        cl.begin_render_pass(&self.tonemap_render_pass, &tonemap_fb, &[], false);
        cl.draw(3, 0);
        cl.end_render_pass();
    }
}

impl RenderStage for PostProcessStage {
    fn name(&self) -> &'static str {
        "post"
    }

    fn execution_dependencies(&self) -> Vec<RenderStageId> {
        vec![RenderStageId::of::<GBufferStage>()]
    }

    fn run(
        &mut self,
        cl: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
    ) -> StageRunResult {
        cl.begin(true).unwrap();

        let g_framebuffer: Arc<Framebuffer> = resources.get(GBufferStage::RES_FRAMEBUFFER);
        let g_depth = g_framebuffer
            .get_image(GBufferStage::DEPTH_ATTACHMENT_ID)
            .unwrap();
        let g_overlay_framebuffer: Arc<Framebuffer> = resources.get(GBufferStage::RES_OVERLAY_FRAMEBUFFER);
        let g_overlay_albedo = g_overlay_framebuffer
            .get_image(GBufferStage::OVERLAY_ALBEDO_ATTACHMENT_ID)
            .unwrap();

        let main_source_image = self.g_buffer_merge_stage(cl, resources, ctx);

        let main_source_image = self.custom_post_processing(cl, resources, ctx, main_source_image, g_depth);

        let combined_source_image =
            self.combine_main_and_overlay_stage(cl, resources, ctx, &main_source_image, g_overlay_albedo);

        let bloom = self.bloom_stage(cl, resources, ctx, &combined_source_image);

        self.tonemap_stage(cl, resources, ctx, g_overlay_albedo, &main_source_image, &bloom);

        cl.end().unwrap();

        StageRunResult::new()
    }
}
