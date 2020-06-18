use crate::PipelineSignature;
use crate::Subpass;
use crate::SwapchainWrapper;
use crate::{format, LoadStore, RenderPass, Shader, SubmitInfo, SubmitPacket};
use crate::{
    utils, Fence, Format, Image, Queue, QueueType, Semaphore, Surface, Swapchain,
    {Buffer, BufferUsageFlags, DeviceBuffer, HostBuffer}, {ImageType, ImageUsageFlags},
};
use crate::{Adapter, PipelineDepthStencil, SubpassDependency};
use crate::{Attachment, Pipeline, PipelineRasterization, PrimitiveTopology, ShaderBinding, ShaderStage};
use ash::version::DeviceV1_0;
use ash::vk;
use spirv_cross::glsl;
use spirv_cross::spirv;
use std::cell::Cell;
use std::collections::{hash_map, HashMap};
use std::sync::Arc;
use std::{cmp, marker::PhantomData, mem, ptr, rc::Rc, slice};

#[derive(Debug)]
pub enum DeviceError {
    VmaError(vk_mem::Error),
    VkError(vk::Result),
    SwapchainError(String),
    SpirvError(spirv_cross::ErrorCode),
    InvalidShader(String),
}

impl From<vk_mem::Error> for DeviceError {
    fn from(err: vk_mem::Error) -> Self {
        DeviceError::VmaError(err)
    }
}

impl From<vk::Result> for DeviceError {
    fn from(err: vk::Result) -> Self {
        DeviceError::VkError(err)
    }
}

impl From<spirv_cross::ErrorCode> for DeviceError {
    fn from(err: spirv_cross::ErrorCode) -> Self {
        DeviceError::SpirvError(err)
    }
}

pub(crate) struct DeviceWrapper(pub(crate) ash::Device);

impl Drop for DeviceWrapper {
    fn drop(&mut self) {
        unsafe {
            self.0.device_wait_idle().unwrap();
            self.0.destroy_device(None);
        }
    }
}

pub struct Device {
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) wrapper: Arc<DeviceWrapper>,
    pub(crate) allocator: vk_mem::Allocator,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
    pub(crate) queues: Vec<Rc<Queue>>,
    pub(crate) pipeline_cache: Cell<vk::PipelineCache>,
}

pub(crate) fn create_semaphore(
    device_wrapper: &Arc<DeviceWrapper>,
    sp_type: vk::SemaphoreType,
) -> Result<Semaphore, vk::Result> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
        .initial_value(0)
        .semaphore_type(sp_type);
    let create_info = vk::SemaphoreCreateInfo::builder().push_next(&mut type_info);
    Ok(Semaphore {
        device_wrapper: Arc::clone(device_wrapper),
        native: unsafe { device_wrapper.0.create_semaphore(&create_info, None)? },
        semaphore_type: sp_type,
        last_signal_value: Cell::new(0),
    })
}

pub(crate) fn create_binary_semaphore(device_wrapper: &Arc<DeviceWrapper>) -> Result<Semaphore, vk::Result> {
    create_semaphore(device_wrapper, vk::SemaphoreType::BINARY)
}

pub(crate) fn create_timeline_semaphore(
    device_wrapper: &Arc<DeviceWrapper>,
) -> Result<Semaphore, vk::Result> {
    create_semaphore(device_wrapper, vk::SemaphoreType::TIMELINE)
}

pub(crate) fn create_fence(device_wrapper: &Arc<DeviceWrapper>) -> Result<Fence, vk::Result> {
    let mut create_info = vk::FenceCreateInfo::builder().build();
    create_info.flags = vk::FenceCreateFlags::SIGNALED;
    Ok(Fence {
        device_wrapper: Arc::clone(device_wrapper),
        native: unsafe { device_wrapper.0.create_fence(&create_info, None)? },
    })
}

impl Device {
    pub fn get_adapter(&self) -> &Arc<Adapter> {
        &self.adapter
    }

    pub fn is_extension_supported(&self, name: &str) -> bool {
        self.adapter.is_extension_enabled(name)
    }

    pub fn get_queue(&self, queue_type: QueueType) -> &Queue {
        &self.queues[queue_type.0 as usize]
    }

    fn create_buffer<T>(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        size: u64,
        mem_usage: vk_mem::MemoryUsage,
    ) -> Result<(Buffer, vk_mem::AllocationInfo), DeviceError> {
        let mut elem_align = 1;

        if usage.intersects(BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST)
            && (elem_align % self.adapter.props.limits.optimal_buffer_copy_offset_alignment != 0)
        {
            elem_align *= self.adapter.props.limits.optimal_buffer_copy_offset_alignment;
        }
        if usage.intersects(BufferUsageFlags::UNIFORM)
            && (elem_align % self.adapter.props.limits.min_uniform_buffer_offset_alignment != 0)
        {
            elem_align *= self.adapter.props.limits.min_uniform_buffer_offset_alignment;
        }
        if usage.intersects(BufferUsageFlags::STORAGE)
            && (elem_align % self.adapter.props.limits.min_storage_buffer_offset_alignment != 0)
        {
            elem_align *= self.adapter.props.limits.min_storage_buffer_offset_alignment;
        }

        let aligned_elem_size = utils::make_mul_of(mem::size_of::<T>(), elem_align as usize);
        let bytesize = aligned_elem_size as u64 * size;

        let buffer_info = vk::BufferCreateInfo::builder()
            .usage(usage.0)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(bytesize as vk::DeviceSize);

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: mem_usage,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: Default::default(),
            preferred_flags: Default::default(),
            memory_type_bits: 0,
            pool: None,
            user_data: None,
        };

        let (buffer, alloc, alloc_info) = self
            .allocator
            .create_buffer(&buffer_info, &allocation_create_info)?;

        Ok((
            Buffer {
                device: Arc::clone(self),
                native: buffer,
                allocation: alloc,
                aligned_elem_size: aligned_elem_size as u64,
                size,
                _bytesize: bytesize as u64,
            },
            alloc_info,
        ))
    }

    pub fn create_host_buffer<T>(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        size: u64,
    ) -> Result<HostBuffer<T>, DeviceError> {
        let (buffer, alloc_info) = self.create_buffer::<T>(usage, size, vk_mem::MemoryUsage::CpuOnly)?;

        let p_data = alloc_info.get_mapped_data();
        if p_data == ptr::null_mut() {}

        Ok(HostBuffer {
            _type_marker: PhantomData,
            buffer: Arc::new(buffer),
            p_data: alloc_info.get_mapped_data(),
        })
    }

    pub fn create_device_buffer<T>(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        size: u64,
    ) -> Result<DeviceBuffer, DeviceError> {
        let (buffer, _) = self.create_buffer::<T>(usage, size, vk_mem::MemoryUsage::GpuOnly)?;
        Ok(DeviceBuffer { buffer })
    }

    fn create_image(
        self: &Arc<Self>,
        image_type: ImageType,
        view_type: vk::ImageViewType,
        format: Format,
        mipmaps: bool,
        usage: ImageUsageFlags,
        mut size: (u32, u32, u32),
    ) -> Result<Rc<Image>, DeviceError> {
        let format_props = self.adapter.get_image_format_properties(
            format.0,
            image_type.0,
            vk::ImageTiling::OPTIMAL,
            usage.0,
        )?;

        let (mip_levels, usage) = if mipmaps {
            (
                ((cmp::min(size.0, size.1) as f64).log2().floor() as u32).min(format_props.max_mip_levels),
                usage.0 | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
            )
        } else {
            (1u32, usage.0)
        };

        size.0 = size.0.min(format_props.max_extent.width);
        size.1 = size.1.min(format_props.max_extent.height);

        let (extent, array_layers) = if image_type == Image::TYPE_2D {
            size.2 = size.2.min(format_props.max_array_layers);
            (
                vk::Extent3D {
                    width: size.0,
                    height: size.1,
                    depth: 1,
                },
                size.2,
            )
        } else {
            size.2 = size.2.min(format_props.max_extent.height);
            (
                vk::Extent3D {
                    width: size.0,
                    height: size.1,
                    depth: size.2,
                },
                1,
            )
        };

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(image_type.0)
            .format(format.0)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::GpuOnly,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: Default::default(),
            preferred_flags: Default::default(),
            memory_type_bits: 0,
            pool: None,
            user_data: None,
        };

        let (image, alloc, _alloc_info) = self
            .allocator
            .create_image(&image_info, &allocation_create_info)?;

        let aspect = if format == format::DEPTH_FORMAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(view_type)
            .format(format.0)
            .components(vk::ComponentMapping {
                r: vk::ComponentSwizzle::IDENTITY,
                g: vk::ComponentSwizzle::IDENTITY,
                b: vk::ComponentSwizzle::IDENTITY,
                a: vk::ComponentSwizzle::IDENTITY,
            })
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: array_layers,
            });
        let view = unsafe { self.wrapper.0.create_image_view(&view_info, None)? };

        Ok(Rc::new(Image {
            device: Arc::clone(self),
            _swapchain_wrapper: None,
            native: image,
            allocation: alloc,
            view,
            aspect,
            owned_handle: true,
            format,
            size,
        }))
    }

    pub fn create_image_2d(
        self: &Arc<Self>,
        format: Format,
        mipmaps: bool,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32),
    ) -> Result<Rc<Image>, DeviceError> {
        self.create_image(
            Image::TYPE_2D,
            vk::ImageViewType::TYPE_2D,
            format,
            mipmaps,
            usage,
            (preferred_size.0, preferred_size.1, 1),
        )
    }

    pub fn create_image_3d(
        self: &Arc<Self>,
        format: Format,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32, u32),
    ) -> Result<Rc<Image>, DeviceError> {
        self.create_image(
            Image::TYPE_2D,
            vk::ImageViewType::TYPE_3D,
            format,
            false,
            usage,
            preferred_size,
        )
    }

    pub fn create_swapchain(
        self: &Arc<Self>,
        surface: &Rc<Surface>,
        size: (u32, u32),
        vsync: bool,
    ) -> Result<Rc<Swapchain>, DeviceError> {
        let surface_capabs = self.adapter.get_surface_capabilities(&surface)?;
        let surface_formats = self.adapter.get_surface_formats(&surface)?;
        let surface_present_modes = self.adapter.get_surface_present_modes(&surface)?;

        let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST;
        if !surface_capabs.supported_usage_flags.contains(image_usage) {
            return Err(DeviceError::SwapchainError(
                "Image usage flags are not supported!".to_string(),
            ));
        }

        let composite_alpha = vk::CompositeAlphaFlagsKHR::OPAQUE;
        if !surface_capabs.supported_composite_alpha.contains(composite_alpha) {
            return Err(DeviceError::SwapchainError(
                "Composite alpha not supported!".to_string(),
            ));
        }

        // TODO: HDR metadata

        let size = (
            size.0
                .min(surface_capabs.max_image_extent.width)
                .max(surface_capabs.min_image_extent.width),
            size.1
                .min(surface_capabs.max_image_extent.height)
                .max(surface_capabs.min_image_extent.height),
        );

        let mut s_format = surface_formats.iter().find(|&s_format| {
            s_format.format == vk::Format::R8G8B8A8_UNORM
                && s_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        });
        if s_format.is_none() {
            s_format = surface_formats.first();
        }

        let s_format = match s_format {
            Some(a) => a,
            None => {
                return Err(DeviceError::SwapchainError(
                    "Swapchain format not found!".to_string(),
                ))
            }
        };

        let present_mode = if vsync {
            vk::PresentModeKHR::FIFO
        } else {
            let mailbox_mode = surface_present_modes
                .iter()
                .find(|&mode| *mode == vk::PresentModeKHR::MAILBOX);
            match mailbox_mode {
                Some(a) => *a,
                None => vk::PresentModeKHR::IMMEDIATE,
            }
        };

        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.native)
            .min_image_count(cmp::min(
                surface_capabs.max_image_count,
                cmp::max(3, surface_capabs.min_image_count),
            ))
            .image_format(s_format.format)
            .image_color_space(s_format.color_space)
            .image_extent(vk::Extent2D {
                width: size.0,
                height: size.1,
            })
            .image_array_layers(1)
            .image_usage(image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabs.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let swapchain_wrapper = Rc::new(SwapchainWrapper {
            device: Arc::clone(self),
            native: unsafe { self.swapchain_khr.create_swapchain(&create_info, None)? },
        });

        let images: Result<Vec<Rc<Image>>, vk::Result> = unsafe {
            self.swapchain_khr
                .get_swapchain_images(swapchain_wrapper.native)?
        }
        .iter()
        .map(|&native_image| {
            let view_info = vk::ImageViewCreateInfo::builder()
                .image(native_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(s_format.format)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            Ok(Rc::new(Image {
                device: Arc::clone(self),
                _swapchain_wrapper: Some(Rc::clone(&swapchain_wrapper)),
                native: native_image,
                allocation: vk_mem::Allocation::null(),
                view: unsafe { self.wrapper.0.create_image_view(&view_info, None)? },
                aspect: view_info.subresource_range.aspect_mask,
                owned_handle: false,
                format: Format(s_format.format),
                size: (size.0, size.1, 1),
            }))
        })
        .collect();

        Ok(Rc::new(Swapchain {
            wrapper: swapchain_wrapper,
            _surface: Rc::clone(surface),
            semaphore: Rc::new(create_binary_semaphore(&self.wrapper)?),
            images: images?,
            curr_image: Cell::new(None),
        }))
    }

    pub fn create_shader(self: &Arc<Self>, code: &[u8]) -> Result<Rc<Shader>, DeviceError> {
        #[allow(clippy::cast_ptr_alignment)]
        let code_words = unsafe {
            std::slice::from_raw_parts(
                code.as_ptr() as *const u32,
                code.len() / std::mem::size_of::<u32>(),
            )
        };

        let ast = spirv::Ast::<glsl::Target>::parse(&spirv::Module::from_words(code_words))?;
        let entry_points = ast.get_entry_points()?;
        let entry_point = entry_points
            .first()
            .ok_or_else(|| DeviceError::InvalidShader("Entry point not found!".to_string()))?;
        let resources = ast.get_shader_resources()?;

        macro_rules! binding_image_array_size {
            ($res: ident, $img_type: ident, $desc_type: ident) => {{
                let var_type = ast.get_type($res.type_id)?;
                ShaderBinding {
                    binding_type: vk::DescriptorType::$desc_type,
                    id: ast.get_decoration($res.id, spirv::Decoration::Binding)?,
                    count: match var_type {
                        spirv::Type::$img_type { array } => {
                            if array.is_empty() {
                                1
                            } else {
                                65536
                            }
                        }
                        _ => unreachable!(),
                    },
                }
            }};
        }

        macro_rules! binding_buffer_array_size {
            ($res: ident, $desc_type: ident) => {{
                let var_type = ast.get_type($res.type_id)?;
                ShaderBinding {
                    binding_type: vk::DescriptorType::$desc_type,
                    id: ast.get_decoration($res.id, spirv::Decoration::Binding)?,
                    count: match var_type {
                        spirv::Type::Struct {
                            member_types: _,
                            array,
                        } => {
                            if array.is_empty() {
                                1
                            } else {
                                65536
                            }
                        }
                        _ => unreachable!(),
                    },
                }
            }};
        }

        let stage = match entry_point.execution_model {
            spirv::ExecutionModel::Vertex => ShaderStage::VERTEX,
            spirv::ExecutionModel::Fragment => ShaderStage::PIXEL,
            spirv::ExecutionModel::Geometry => ShaderStage::GEOMETRY,
            spirv::ExecutionModel::GlCompute => ShaderStage::COMPUTE,
            m => {
                return Err(DeviceError::InvalidShader(format!(
                    "Unsupported execution model {:?}",
                    m
                )))
            }
        };

        let mut bindings = HashMap::new();

        for res in &resources.sampled_images {
            let binding = binding_image_array_size!(res, SampledImage, COMBINED_IMAGE_SAMPLER);
            bindings.insert(res.name.clone(), binding);
        }
        for res in &resources.storage_images {
            let binding = binding_image_array_size!(res, Image, STORAGE_IMAGE);
            bindings.insert(res.name.clone(), binding);
        }
        for res in &resources.uniform_buffers {
            let binding = binding_buffer_array_size!(res, UNIFORM_BUFFER);
            bindings.insert(res.name.clone(), binding);
        }
        for res in &resources.storage_buffers {
            let binding = binding_buffer_array_size!(res, STORAGE_BUFFER);
            bindings.insert(res.name.clone(), binding);
        }

        let mut push_constants = HashMap::new();
        let mut push_constants_size = 0;
        if !resources.push_constant_buffers.is_empty() {
            let id = resources.push_constant_buffers[0].id;
            let type_id = resources.push_constant_buffers[0].base_type_id;

            push_constants_size = ast.get_declared_struct_size(type_id)?;
            let ranges = ast.get_active_buffer_ranges(id)?;

            for (i, range) in ranges.iter().enumerate() {
                push_constants.insert(ast.get_member_name(type_id, i as u32)?, range.clone());
            }
        }

        let create_info = vk::ShaderModuleCreateInfo::builder().code(code_words);

        Ok(Rc::new(Shader {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.0.create_shader_module(&create_info, None)? },
            stage,
            bindings,
            push_constants,
            push_constants_size,
        }))
    }

    pub fn create_render_pass(
        self: &Arc<Self>,
        attachments: &[Attachment],
        subpasses: &[Subpass],
        dependencies: &[SubpassDependency],
    ) -> Result<Rc<RenderPass>, vk::Result> {
        let native_attachments: Vec<vk::AttachmentDescription> = attachments
            .iter()
            .map(|info| {
                let mut native_info = vk::AttachmentDescription::builder()
                    .format(info.format.0)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .initial_layout(info.init_layout.0)
                    .final_layout(info.final_layout.0)
                    .build();
                match info.load_store {
                    LoadStore::None => {
                        native_info.load_op = vk::AttachmentLoadOp::DONT_CARE;
                        native_info.store_op = vk::AttachmentStoreOp::DONT_CARE;
                    }
                    LoadStore::InitSave => {
                        native_info.load_op = vk::AttachmentLoadOp::LOAD;
                        native_info.store_op = vk::AttachmentStoreOp::DONT_CARE;
                    }
                    LoadStore::InitClear => {
                        native_info.load_op = vk::AttachmentLoadOp::CLEAR;
                        native_info.store_op = vk::AttachmentStoreOp::DONT_CARE;
                    }
                    LoadStore::FinalSave => {
                        native_info.load_op = vk::AttachmentLoadOp::DONT_CARE;
                        native_info.store_op = vk::AttachmentStoreOp::STORE;
                    }
                    LoadStore::InitClearFinalSave => {
                        native_info.load_op = vk::AttachmentLoadOp::CLEAR;
                        native_info.store_op = vk::AttachmentStoreOp::STORE;
                    }
                    LoadStore::InitSaveFinalSave => {
                        native_info.load_op = vk::AttachmentLoadOp::LOAD;
                        native_info.store_op = vk::AttachmentStoreOp::STORE;
                    }
                }
                native_info
            })
            .collect();

        let mut attachment_ref_count = 0;
        for subpass in subpasses {
            attachment_ref_count += subpass.color.len();
            if subpass.depth.is_some() {
                attachment_ref_count += 1;
            }
        }

        let mut color_attachments = Vec::with_capacity(attachments.len());
        let mut depth_attachments = Vec::with_capacity(attachments.len());

        let mut native_attachment_refs = Vec::with_capacity(attachment_ref_count);
        let mut native_subpass_descs = Vec::with_capacity(subpasses.len());

        for subpass in subpasses {
            // Color attachments
            let color_att_ref_index = native_attachment_refs.len();
            for color_attachment in subpass.color {
                native_attachment_refs.push(
                    vk::AttachmentReference::builder()
                        .attachment(color_attachment.index)
                        .layout(color_attachment.layout.0)
                        .build(),
                );

                if !color_attachments.contains(&color_attachment.index) {
                    color_attachments.push(color_attachment.index);
                }
            }

            // Depth attachment
            let depth_att_ref_index;
            if let Some(depth_attachment) = &subpass.depth {
                depth_att_ref_index = native_attachment_refs.len() as u32;
                native_attachment_refs.push(
                    vk::AttachmentReference::builder()
                        .attachment(depth_attachment.index)
                        .layout(depth_attachment.layout.0)
                        .build(),
                );

                if !depth_attachments.contains(&depth_attachment.index) {
                    depth_attachments.push(depth_attachment.index);
                }
            } else {
                depth_att_ref_index = u32::MAX;
            }

            // Build description
            let mut subpass_desc = vk::SubpassDescription::builder()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(
                    &native_attachment_refs[color_att_ref_index..color_att_ref_index + subpass.color.len()],
                );
            if subpass.depth.is_some() {
                subpass_desc = subpass_desc
                    .depth_stencil_attachment(&native_attachment_refs[depth_att_ref_index as usize]);
            }

            native_subpass_descs.push(subpass_desc.build());
        }

        let native_dependencies: Vec<vk::SubpassDependency> = dependencies
            .iter()
            .map(|dep| {
                vk::SubpassDependency::builder()
                    .src_subpass(dep.src_subpass)
                    .dst_subpass(dep.dst_subpass)
                    .src_stage_mask(dep.src_stage_mask.0)
                    .dst_stage_mask(dep.dst_stage_mask.0)
                    .src_access_mask(dep.src_access_mask.0)
                    .dst_access_mask(dep.dst_access_mask.0)
                    .build()
            })
            .collect();

        let create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&native_attachments)
            .subpasses(&native_subpass_descs)
            .dependencies(&native_dependencies);

        Ok(Rc::new(RenderPass {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.0.create_render_pass(&create_info, None)? },
            attachments: attachments.to_vec(),
            color_attachments,
            depth_attachments,
        }))
    }

    pub fn create_pipeline_signature(
        self: &Arc<Self>,
        shaders: &[Rc<Shader>],
    ) -> Result<Arc<PipelineSignature>, vk::Result> {
        let mut native_bindings = Vec::<vk::DescriptorSetLayoutBinding>::new();
        let mut binding_flags = Vec::<vk::DescriptorBindingFlagsEXT>::new();

        let mut descriptor_sizes = Vec::<vk::DescriptorPoolSize>::new();
        let mut descriptor_sizes_indices = HashMap::<vk::DescriptorType, u32>::new();
        let mut binding_types = HashMap::<u32, vk::DescriptorType>::new();
        let mut push_constant_ranges = HashMap::<ShaderStage, (u32, u32)>::new();
        let mut push_constants_size = 0u32;

        for shader in shaders {
            for (_name, binding) in &shader.bindings {
                if let hash_map::Entry::Vacant(entry) = descriptor_sizes_indices.entry(binding.binding_type) {
                    entry.insert(descriptor_sizes.len() as u32);
                    descriptor_sizes.push(
                        vk::DescriptorPoolSize::builder()
                            .ty(binding.binding_type)
                            .descriptor_count(0)
                            .build(),
                    );
                }

                descriptor_sizes[descriptor_sizes_indices[&binding.binding_type] as usize]
                    .descriptor_count += binding.count;
                binding_types.insert(binding.id, binding.binding_type);

                native_bindings.push(
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(binding.id)
                        .descriptor_type(binding.binding_type)
                        .descriptor_count(binding.count)
                        .stage_flags(shader.stage.0)
                        .build(),
                );

                binding_flags.push(if binding.count > 1 {
                    vk::DescriptorBindingFlags::PARTIALLY_BOUND
                } else {
                    vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                });
            }

            if shader.push_constants_size > 0 {
                push_constant_ranges.insert(shader.stage, (push_constants_size, shader.push_constants_size));
                push_constants_size += shader.push_constants_size;
            }
        }

        let mut binding_flags_info =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&binding_flags);

        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
            .bindings(&native_bindings)
            .push_next(&mut binding_flags_info);

        Ok(Arc::new(PipelineSignature {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.0.create_descriptor_set_layout(&create_info, None)? },
            descriptor_sizes,
            descriptor_sizes_indices,
            binding_types,
            push_constant_ranges,
            push_constants_size,
        }))
    }

    pub fn load_pipeline_cache(&self, data: &[u8]) -> Result<(), vk::Result> {
        unsafe {
            self.wrapper
                .0
                .destroy_pipeline_cache(self.pipeline_cache.get(), None)
        };
        let create_info = vk::PipelineCacheCreateInfo::builder().initial_data(data);
        self.pipeline_cache
            .set(unsafe { self.wrapper.0.create_pipeline_cache(&create_info, None)? });
        Ok(())
    }

    pub fn get_pipeline_cache(&self) -> Result<Vec<u8>, vk::Result> {
        unsafe { self.wrapper.0.get_pipeline_cache_data(self.pipeline_cache.get()) }
    }

    pub fn create_graphics_pipeline(
        self: &Arc<Self>,
        render_pass: &Arc<RenderPass>,
        subpass_index: u32,
        primitive_topology: PrimitiveTopology,
        depth_stencil: PipelineDepthStencil,
        rasterization: PipelineRasterization,
        pipeline_signature: &Arc<PipelineSignature>,
    ) -> Result<Arc<Pipeline>, vk::Result> {
        // Push constants
        /*

        std::vector<VkPushConstantRange> push_constant_ranges;
        push_constant_ranges.reserve(signature->push_constants_ranges.size());

        for (auto const& [shader_type, range] : signature->push_constants_ranges) {
            push_constant_ranges.push_back(VkPushConstantRange {
                .stageFlags = shader_type,
                .offset     = range[0],
                .size       = range[1],
            });
        }

        */

        let layout_create_info = vk::PipelineLayoutCreateInfo::builder();
        let layout = unsafe { self.wrapper.0.create_pipeline_layout(&layout_create_info, None)? };

        let create_info = vk::GraphicsPipelineCreateInfo::builder();
        let native_pipeline = unsafe {
            self.wrapper.0.create_graphics_pipelines(
                self.pipeline_cache.get(),
                slice::from_ref(&create_info),
                None,
            )
        };
        if let Err((_, err)) = native_pipeline {
            return Err(err);
        }
        let pipeline = native_pipeline.unwrap()[0];

        Ok(Arc::new(Pipeline {
            device: Arc::clone(self),
            layout,
            native: pipeline,
        }))
    }

    pub fn create_submit_packet(
        self: &Arc<Self>,
        submit_infos: &[SubmitInfo],
    ) -> Result<SubmitPacket, vk::Result> {
        Ok(SubmitPacket {
            infos: submit_infos.to_vec(),
            fence: create_fence(&self.wrapper)?,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.wrapper
                .0
                .destroy_pipeline_cache(self.pipeline_cache.get(), None)
        };
        self.allocator.destroy();
    }
}