use crate::FORMAT_SIZES;
use crate::IMAGE_FORMATS;
use crate::{format, LoadStore, RenderPass, Shader, SubmitInfo, SubmitPacket};
use crate::{
    utils, BindingType, Fence, Format, Image, Queue, QueueType, Semaphore, Surface, Swapchain,
    {Buffer, BufferUsageFlags, DeviceBuffer, HostBuffer}, {ImageType, ImageUsageFlags},
};
use crate::{Adapter, PipelineDepthStencil, SubpassDependency};
use crate::{Attachment, Pipeline, PipelineRasterization, PrimitiveTopology, ShaderBinding, ShaderStage};
use crate::{ImageWrapper, SamplerReduction};
use crate::{PipelineSignature, ShaderBindingMod};
use crate::{QueryPool, Subpass};
use crate::{Sampler, DEPTH_FORMAT};
use crate::{SwapchainWrapper, BC_IMAGE_FORMATS};
use ash::version::DeviceV1_0;
use ash::vk;
use spirv_cross::glsl;
use spirv_cross::spirv;
use std::collections::{hash_map, HashMap};
use std::sync::{Arc, Mutex};
use std::{cmp, ffi::CStr, marker::PhantomData, mem, slice};

#[derive(Debug)]
pub enum DeviceError {
    VmaError(vk_mem::Error),
    VkError(vk::Result),
    ZeroBufferElementSize,
    ZeroBufferSize,
    SwapchainError(String),
    SpirvError(spirv_cross::ErrorCode),
    InvalidShader(String),
    InvalidSignature(&'static str),
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
    pub(crate) queues: Vec<Arc<Queue>>,
    pub(crate) pipeline_cache: Mutex<vk::PipelineCache>,
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
        last_signal_value: Mutex::new(0),
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

    fn create_buffer(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        elem_size: u64,
        size: u64,
        mem_usage: vk_mem::MemoryUsage,
    ) -> Result<(Arc<Buffer>, vk_mem::AllocationInfo), DeviceError> {
        if elem_size == 0 {
            return Err(DeviceError::ZeroBufferElementSize);
        }
        if size == 0 {
            return Err(DeviceError::ZeroBufferSize);
        }

        let mut elem_align = 1;

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

        let aligned_elem_size = utils::make_mul_of_u64(elem_size, elem_align as u64);
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
            Arc::new(Buffer {
                device: Arc::clone(self),
                native: buffer,
                allocation: alloc,
                _elem_size: elem_size,
                aligned_elem_size: aligned_elem_size as u64,
                size,
                _bytesize: bytesize as u64,
            }),
            alloc_info,
        ))
    }

    pub fn create_host_buffer<T>(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        size: u64,
    ) -> Result<HostBuffer<T>, DeviceError> {
        let (buffer, alloc_info) = self.create_buffer(
            usage,
            mem::size_of::<T>() as u64,
            size,
            vk_mem::MemoryUsage::CpuOnly,
        )?;

        Ok(HostBuffer {
            _type_marker: PhantomData,
            buffer,
            p_data: alloc_info.get_mapped_data(),
        })
    }

    pub fn create_device_buffer(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        element_size: u64,
        size: u64,
    ) -> Result<Arc<DeviceBuffer>, DeviceError> {
        let (buffer, _) = self.create_buffer(usage, element_size, size, vk_mem::MemoryUsage::GpuOnly)?;
        Ok(Arc::new(DeviceBuffer { buffer }))
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    fn create_image(
        self: &Arc<Self>,
        image_type: ImageType,
        format: Format,
        max_mip_levels: u32,
        max_anisotropy: f32,
        usage: ImageUsageFlags,
        mut size: (u32, u32, u32),
    ) -> Result<Arc<Image>, DeviceError> {
        if !IMAGE_FORMATS.contains(&format) && !BC_IMAGE_FORMATS.contains(&format) && format != DEPTH_FORMAT {
            panic!("Image format {:?} is not supported!", format);
        }

        let format_props = self.adapter.get_image_format_properties(
            format.0,
            image_type.0,
            vk::ImageTiling::OPTIMAL,
            usage.0,
        )?;

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
        let mip_levels = if max_mip_levels == 0 {
            utils::log2(size.0.max(size.1).max(size.2)) + 1
        } else {
            max_mip_levels.min(format_props.max_mip_levels)
        };

        let image_info = vk::ImageCreateInfo::builder()
            .image_type(image_type.0)
            .format(format.0)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage.0)
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

        let sampler = if usage.intersects(
            ImageUsageFlags::COLOR_ATTACHMENT
                | ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                | ImageUsageFlags::INPUT_ATTACHMENT
                | ImageUsageFlags::SAMPLED
                | ImageUsageFlags::STORAGE,
        ) {
            let linear_filter_supported = self
                .adapter
                .is_linear_filter_supported(format.0, vk::ImageTiling::OPTIMAL);

            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(if linear_filter_supported {
                    vk::Filter::LINEAR
                } else {
                    vk::Filter::NEAREST
                })
                .mipmap_mode(if linear_filter_supported {
                    vk::SamplerMipmapMode::LINEAR
                } else {
                    vk::SamplerMipmapMode::NEAREST
                })
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(max_anisotropy != 1.0)
                .max_anisotropy(max_anisotropy)
                .compare_enable(false)
                .min_lod(0.0)
                .max_lod(mip_levels as f32 - 1.0)
                .unnormalized_coordinates(false);

            unsafe { self.wrapper.0.create_sampler(&sampler_info, None)? }
        } else {
            vk::Sampler::default()
        };

        let image_wrapper = Arc::new(ImageWrapper {
            device: Arc::clone(self),
            _swapchain_wrapper: None,
            native: image,
            allocation: alloc,
            owned_handle: true,
            ty: image_type,
            format,
            aspect,
        });
        let view = image_wrapper.create_view().build()?;

        Ok(Arc::new(Image {
            wrapper: image_wrapper,
            view,
            sampler: Arc::new(Sampler {
                device: Arc::clone(self),
                native: sampler,
            }),
            size,
            mip_levels,
        }))
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    pub fn create_image_2d(
        self: &Arc<Self>,
        format: Format,
        max_mip_levels: u32,
        max_anisotropy: f32,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32),
    ) -> Result<Arc<Image>, DeviceError> {
        self.create_image(
            Image::TYPE_2D,
            format,
            max_mip_levels,
            max_anisotropy,
            usage,
            (preferred_size.0, preferred_size.1, 1),
        )
    }

    pub fn create_image_3d(
        self: &Arc<Self>,
        format: Format,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32, u32),
    ) -> Result<Arc<Image>, DeviceError> {
        self.create_image(Image::TYPE_3D, format, 1, 1.0, usage, preferred_size)
    }

    pub fn create_reduction_sampler(
        self: &Arc<Self>,
        mode: SamplerReduction,
    ) -> Result<Arc<Sampler>, DeviceError> {
        let mut reduction_info = vk::SamplerReductionModeCreateInfo::builder()
            .reduction_mode(mode.0)
            .build();

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .anisotropy_enable(false)
            .compare_enable(false)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .unnormalized_coordinates(false)
            .push_next(&mut reduction_info);

        Ok(Arc::new(Sampler {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.0.create_sampler(&sampler_info, None)? },
        }))
    }

    pub fn create_query_pool(self: &Arc<Self>, query_count: u32) -> Result<Arc<QueryPool>, vk::Result> {
        let create_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(query_count);

        Ok(Arc::new(QueryPool {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.0.create_query_pool(&create_info, None)? },
        }))
    }

    pub fn create_swapchain(
        self: &Arc<Self>,
        surface: &Arc<Surface>,
        size: (u32, u32),
        vsync: bool,
    ) -> Result<Arc<Swapchain>, DeviceError> {
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
            (s_format.format == vk::Format::R8G8B8A8_UNORM || s_format.format == vk::Format::B8G8R8A8_UNORM)
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
                ));
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

        let swapchain_wrapper = Arc::new(SwapchainWrapper {
            device: Arc::clone(self),
            native: Mutex::new(unsafe { self.swapchain_khr.create_swapchain(&create_info, None)? }),
        });

        let images: Result<Vec<Arc<Image>>, DeviceError> = unsafe {
            let native_swapchain = swapchain_wrapper.native.lock().unwrap();
            self.swapchain_khr.get_swapchain_images(*native_swapchain)?
        }
        .iter()
        .map(|&native_image| {
            let sampler_info = vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::NEAREST)
                .min_filter(vk::Filter::NEAREST)
                .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
                .address_mode_u(vk::SamplerAddressMode::REPEAT)
                .address_mode_v(vk::SamplerAddressMode::REPEAT)
                .address_mode_w(vk::SamplerAddressMode::REPEAT)
                .anisotropy_enable(true)
                .max_anisotropy(1f32)
                .compare_enable(false)
                .max_lod(0f32)
                .unnormalized_coordinates(false);

            let image_wrapper = Arc::new(ImageWrapper {
                device: Arc::clone(self),
                _swapchain_wrapper: Some(Arc::clone(&swapchain_wrapper)),
                native: native_image,
                allocation: vk_mem::Allocation::null(),
                owned_handle: false,
                ty: Image::TYPE_2D,
                format: Format(s_format.format),
                aspect: vk::ImageAspectFlags::COLOR,
            });
            let view = image_wrapper.create_view().build()?;

            Ok(Arc::new(Image {
                wrapper: image_wrapper,
                view,
                sampler: Arc::new(Sampler {
                    device: Arc::clone(self),
                    native: unsafe { self.wrapper.0.create_sampler(&sampler_info, None)? },
                }),
                size: (size.0, size.1, 1),
                mip_levels: 1,
            }))
        })
        .collect();

        Ok(Arc::new(Swapchain {
            wrapper: swapchain_wrapper,
            _surface: Arc::clone(surface),
            semaphore: Arc::new(create_binary_semaphore(&self.wrapper)?),
            images: images?,
        }))
    }

    pub fn create_shader(
        self: &Arc<Self>,
        code: &[u8],
        input_formats: &[(&str, Format)],
        binding_types: &[(&str, ShaderBindingMod)],
    ) -> Result<Arc<Shader>, DeviceError> {
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

        let binding_types_map: HashMap<&str, ShaderBindingMod> = binding_types.iter().cloned().collect();

        macro_rules! binding_image_array_size {
            ($res: ident, $img_type: ident, $desc_type: ident) => {{
                let var_type = ast.get_type($res.type_id)?;
                let shader_binding_mod = binding_types_map
                    .get($res.name.as_str())
                    .or_else(|| Some(&ShaderBindingMod::DEFAULT))
                    .unwrap();
                ShaderBinding {
                    binding_type: BindingType(vk::DescriptorType::$desc_type),
                    binding_mod: *shader_binding_mod,
                    descriptor_set: ast
                        .get_decoration($res.id, spirv::Decoration::DescriptorSet)
                        .unwrap(),
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
            ($res: ident, $desc_type0: ident, $desc_type1: ident) => {{
                let var_type = ast.get_type($res.type_id)?;
                let shader_binding_mod = binding_types_map
                    .get($res.name.as_str())
                    .or_else(|| Some(&ShaderBindingMod::DEFAULT))
                    .unwrap();
                ShaderBinding {
                    binding_type: if shader_binding_mod == &ShaderBindingMod::DEFAULT
                        || shader_binding_mod == &ShaderBindingMod::DYNAMIC_UPDATE
                    {
                        BindingType(vk::DescriptorType::$desc_type0)
                    } else {
                        BindingType(vk::DescriptorType::$desc_type1)
                    },
                    binding_mod: *shader_binding_mod,
                    descriptor_set: ast
                        .get_decoration($res.id, spirv::Decoration::DescriptorSet)
                        .unwrap(),
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
                )));
            }
        };

        let input_formats: HashMap<&str, Format> = input_formats.iter().cloned().collect();
        let mut input_locations = HashMap::<u32, Format>::with_capacity(resources.stage_inputs.len());

        if stage == ShaderStage::VERTEX {
            for res in resources.stage_inputs {
                let format = *input_formats
                    .get(res.name.as_str())
                    .ok_or(DeviceError::InvalidShader(format!(
                        "Input format for {} not provided!",
                        res.name
                    )))?;
                let location = ast.get_decoration(res.id, spirv::Decoration::Location).unwrap();

                input_locations.insert(location, format);
            }
        }

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
            let binding = binding_buffer_array_size!(res, UNIFORM_BUFFER, UNIFORM_BUFFER_DYNAMIC);
            bindings.insert(res.name.clone(), binding);
        }
        for res in &resources.storage_buffers {
            let binding = binding_buffer_array_size!(res, STORAGE_BUFFER, STORAGE_BUFFER_DYNAMIC);
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

        Ok(Arc::new(Shader {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.0.create_shader_module(&create_info, None)? },
            stage,
            input_locations,
            bindings,
            _push_constants: push_constants,
            push_constants_size,
        }))
    }

    pub fn create_render_pass(
        self: &Arc<Self>,
        attachments: &[Attachment],
        subpasses: &[Subpass],
        dependencies: &[SubpassDependency],
    ) -> Result<Arc<RenderPass>, vk::Result> {
        let native_attachments: Vec<vk::AttachmentDescription> = attachments
            .iter()
            .map(|info| {
                let mut native_info = vk::AttachmentDescription::builder()
                    .format(info.format.0)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
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
            for color_attachment in &subpass.color {
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

        Ok(Arc::new(RenderPass {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.0.create_render_pass(&create_info, None)? },
            subpasses: subpasses.into(),
            attachments: attachments.to_vec(),
            _color_attachments: color_attachments,
            _depth_attachments: depth_attachments,
        }))
    }

    pub fn create_pipeline_signature(
        self: &Arc<Self>,
        shaders: &[Arc<Shader>],
    ) -> Result<Arc<PipelineSignature>, vk::Result> {
        // MAX 4 descriptor sets
        // [descriptor_set, bindings]
        let mut native_bindings: [Vec<vk::DescriptorSetLayoutBinding>; 4] = Default::default();
        let mut binding_flags: [Vec<vk::DescriptorBindingFlagsEXT>; 4] = Default::default();
        let mut descriptor_sizes: [Vec<vk::DescriptorPoolSize>; 4] = Default::default();
        let mut descriptor_sizes_indices: [HashMap<vk::DescriptorType, u32>; 4] = Default::default();

        let mut binding_types = HashMap::<u32, vk::DescriptorType>::new();
        let mut push_constants_size = 0u32;

        for shader in shaders {
            for (_name, binding) in &shader.bindings {
                let set = binding.descriptor_set as usize;
                let descriptor_sizes = &mut descriptor_sizes[set];
                let descriptor_sizes_indices = &mut descriptor_sizes_indices[set];

                if let hash_map::Entry::Vacant(entry) = descriptor_sizes_indices.entry(binding.binding_type.0)
                {
                    entry.insert(descriptor_sizes.len() as u32);
                    descriptor_sizes.push(
                        vk::DescriptorPoolSize::builder()
                            .ty(binding.binding_type.0)
                            .descriptor_count(0)
                            .build(),
                    );
                }

                descriptor_sizes[descriptor_sizes_indices[&binding.binding_type.0] as usize]
                    .descriptor_count += binding.count;
                binding_types.insert(binding.id, binding.binding_type.0);

                let native_binding = native_bindings[set].iter_mut().find(|b| b.binding == binding.id);

                if let Some(native_binding) = native_binding {
                    native_binding.stage_flags |= shader.stage.0;
                } else {
                    native_bindings[set].push(
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(binding.id)
                            .descriptor_type(binding.binding_type.0)
                            .descriptor_count(binding.count)
                            .stage_flags(shader.stage.0)
                            .build(),
                    );

                    let mut flags = vk::DescriptorBindingFlags::default();
                    if binding.count > 1 {
                        flags |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
                    }

                    binding_flags[set].push(flags);
                }
            }

            push_constants_size += shader.push_constants_size;
        }

        let mut native_descriptor_sets = [vk::DescriptorSetLayout::default(); 4];

        for (i, bindings) in native_bindings.iter().enumerate() {
            if !bindings.is_empty() {
                let mut binding_flags_info =
                    vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&binding_flags[i]);

                let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(bindings)
                    .push_next(&mut binding_flags_info);

                native_descriptor_sets[i] =
                    unsafe { self.wrapper.0.create_descriptor_set_layout(&create_info, None)? };
            }
        }

        let shaders: HashMap<ShaderStage, Arc<Shader>> = shaders
            .iter()
            .cloned()
            .map(|shader| (shader.stage, shader))
            .collect();

        let pipeline_layout = {
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL,
                offset: 0,
                size: push_constants_size,
            };

            // Layout
            let set_layouts: Vec<vk::DescriptorSetLayout> = native_descriptor_sets
                .iter()
                .cloned()
                .filter(|&set_layout| set_layout != vk::DescriptorSetLayout::default())
                .collect();

            let mut layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

            if push_constants_size > 0 {
                layout_create_info =
                    layout_create_info.push_constant_ranges(slice::from_ref(&push_constant_range));
            }

            unsafe { self.wrapper.0.create_pipeline_layout(&layout_create_info, None)? }
        };

        Ok(Arc::new(PipelineSignature {
            device: Arc::clone(self),
            native: native_descriptor_sets,
            pipeline_layout,
            descriptor_sizes,
            binding_types,
            _push_constants_size: push_constants_size,
            shaders,
        }))
    }

    pub fn create_custom_pipeline_signature(
        self: &Arc<Self>,
        bindings: &[(ShaderStage, &[ShaderBinding])],
    ) -> Result<Arc<PipelineSignature>, vk::Result> {
        let bindings: HashMap<ShaderStage, &[ShaderBinding]> = bindings.iter().cloned().collect();

        // MAX 4 descriptor sets
        // [descriptor_set, bindings]
        let mut native_bindings: [Vec<vk::DescriptorSetLayoutBinding>; 4] = Default::default();
        let mut binding_flags: [Vec<vk::DescriptorBindingFlagsEXT>; 4] = Default::default();
        let mut descriptor_sizes: [Vec<vk::DescriptorPoolSize>; 4] = Default::default();
        let mut descriptor_sizes_indices: [HashMap<vk::DescriptorType, u32>; 4] = Default::default();

        let mut binding_types = HashMap::<u32, vk::DescriptorType>::new();

        for (stage, &bindings) in &bindings {
            for binding in bindings {
                let set = binding.descriptor_set as usize;
                let descriptor_sizes = &mut descriptor_sizes[set];
                let descriptor_sizes_indices = &mut descriptor_sizes_indices[set];

                if let hash_map::Entry::Vacant(entry) = descriptor_sizes_indices.entry(binding.binding_type.0)
                {
                    entry.insert(descriptor_sizes.len() as u32);
                    descriptor_sizes.push(
                        vk::DescriptorPoolSize::builder()
                            .ty(binding.binding_type.0)
                            .descriptor_count(0)
                            .build(),
                    );
                }

                descriptor_sizes[descriptor_sizes_indices[&binding.binding_type.0] as usize]
                    .descriptor_count += binding.count;
                binding_types.insert(binding.id, binding.binding_type.0);

                native_bindings[set].push(
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(binding.id)
                        .descriptor_type(binding.binding_type.0)
                        .descriptor_count(binding.count)
                        .stage_flags(stage.0)
                        .build(),
                );

                let mut flags = vk::DescriptorBindingFlags::default();
                if binding.count > 1 {
                    flags |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
                }

                binding_flags[set].push(flags);
            }
        }

        let mut native_descriptor_sets = [vk::DescriptorSetLayout::default(); 4];

        for (i, bindings) in native_bindings.iter().enumerate() {
            if !bindings.is_empty() {
                let mut binding_flags_info =
                    vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&binding_flags[i]);

                let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(bindings)
                    .push_next(&mut binding_flags_info);

                native_descriptor_sets[i] =
                    unsafe { self.wrapper.0.create_descriptor_set_layout(&create_info, None)? };
            }
        }

        let pipeline_layout = {
            let set_layouts: Vec<vk::DescriptorSetLayout> = native_descriptor_sets
                .iter()
                .cloned()
                .filter(|&set_layout| set_layout != vk::DescriptorSetLayout::default())
                .collect();

            let layout_create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
            unsafe { self.wrapper.0.create_pipeline_layout(&layout_create_info, None)? }
        };

        Ok(Arc::new(PipelineSignature {
            device: Arc::clone(self),
            native: native_descriptor_sets,
            pipeline_layout,
            descriptor_sizes,
            binding_types,
            _push_constants_size: 0,
            shaders: Default::default(),
        }))
    }

    pub fn load_pipeline_cache(&self, data: &[u8]) -> Result<(), vk::Result> {
        let mut pipeline_cache = self.pipeline_cache.lock().unwrap();

        unsafe { self.wrapper.0.destroy_pipeline_cache(*pipeline_cache, None) };
        let create_info = vk::PipelineCacheCreateInfo::builder().initial_data(data);
        *pipeline_cache = unsafe { self.wrapper.0.create_pipeline_cache(&create_info, None)? };

        Ok(())
    }

    pub fn get_pipeline_cache(&self) -> Result<Vec<u8>, vk::Result> {
        let pipeline_cache = self.pipeline_cache.lock().unwrap();
        unsafe { self.wrapper.0.get_pipeline_cache_data(*pipeline_cache) }
    }

    pub fn create_graphics_pipeline(
        self: &Arc<Self>,
        render_pass: &Arc<RenderPass>,
        subpass_index: u32,
        primitive_topology: PrimitiveTopology,
        depth_stencil: PipelineDepthStencil,
        rasterization: PipelineRasterization,
        signature: &Arc<PipelineSignature>,
    ) -> Result<Arc<Pipeline>, DeviceError> {
        // Input assembly
        let input_assembly_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(primitive_topology.0)
            .primitive_restart_enable(false);

        // Find vertex shader
        let vertex_shader = signature
            .shaders
            .get(&ShaderStage::VERTEX)
            .ok_or(DeviceError::InvalidSignature("Vertex shader not provided!"))?;

        // Vertex input
        let vertex_binding_count = vertex_shader.input_locations.len();
        let mut vertex_binding_descs =
            vec![vk::VertexInputBindingDescription::default(); vertex_binding_count];
        let mut vertex_attrib_descs =
            vec![vk::VertexInputAttributeDescription::default(); vertex_binding_count];

        for (location, format) in &vertex_shader.input_locations {
            let buffer_index = *location;

            vertex_binding_descs[buffer_index as usize] = vk::VertexInputBindingDescription {
                binding: buffer_index,
                stride: FORMAT_SIZES[&format] as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            };
            vertex_attrib_descs[buffer_index as usize] = vk::VertexInputAttributeDescription {
                location: *location,
                binding: buffer_index,
                format: format.0,
                offset: 0,
            };
        }

        let vertex_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&vertex_binding_descs)
            .vertex_attribute_descriptions(&vertex_attrib_descs);

        // Rasterization
        let rasterization_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(if rasterization.cull_back_faces {
                vk::CullModeFlags::BACK
            } else {
                vk::CullModeFlags::NONE
            })
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        // Multisample
        let multisample_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1)
            .sample_shading_enable(false)
            .alpha_to_coverage_enable(false)
            .alpha_to_one_enable(false);

        // DepthStencil
        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(depth_stencil.depth_test)
            .depth_write_enable(depth_stencil.depth_write)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(depth_stencil.stencil_test);

        // Viewport
        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: 1280.0,
            height: 720.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 1280,
                height: 720,
            },
        };
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .viewports(slice::from_ref(&viewport))
            .scissors(slice::from_ref(&scissor));

        // Color blend
        let def_attachment = vk::PipelineColorBlendAttachmentState::builder()
            .blend_enable(false)
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .build();
        let blend_attachments =
            vec![def_attachment; render_pass.subpasses[subpass_index as usize].color.len()];
        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&blend_attachments);

        // Dynamic
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_info = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        // Shader stages
        let stages: Vec<vk::PipelineShaderStageCreateInfo> = signature
            .shaders
            .iter()
            .map(|(stage, shader)| {
                vk::PipelineShaderStageCreateInfo::builder()
                    .stage(stage.0)
                    .module(shader.native)
                    .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
                    .build()
            })
            .collect();

        let create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&stages)
            .vertex_input_state(&vertex_info)
            .input_assembly_state(&input_assembly_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisample_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_info)
            .dynamic_state(&dynamic_info)
            .layout(signature.pipeline_layout)
            .render_pass(render_pass.native)
            .subpass(subpass_index);

        let pipeline_cache = self.pipeline_cache.lock().unwrap();
        let native_pipeline = unsafe {
            self.wrapper
                .0
                .create_graphics_pipelines(*pipeline_cache, slice::from_ref(&create_info), None)
        };
        if let Err((_, err)) = native_pipeline {
            return Err(err.into());
        }
        let pipeline = native_pipeline.unwrap()[0];

        Ok(Arc::new(Pipeline {
            device: Arc::clone(self),
            _render_pass: Some(Arc::clone(render_pass)),
            signature: Arc::clone(signature),
            native: pipeline,
            bind_point: vk::PipelineBindPoint::GRAPHICS,
        }))
    }

    pub fn create_compute_pipeline(
        self: &Arc<Self>,
        signature: &Arc<PipelineSignature>,
    ) -> Result<Arc<Pipeline>, vk::Result> {
        // Stage
        let stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(signature.shaders[&ShaderStage::COMPUTE].native)
            .name(CStr::from_bytes_with_nul(b"main\0").unwrap())
            .build();

        let create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(stage_info)
            .layout(signature.pipeline_layout);

        let pipeline_cache = self.pipeline_cache.lock().unwrap();
        let native_pipeline = unsafe {
            self.wrapper
                .0
                .create_compute_pipelines(*pipeline_cache, slice::from_ref(&create_info), None)
        };
        if let Err((_, err)) = native_pipeline {
            return Err(err.into());
        }
        let pipeline = native_pipeline.unwrap()[0];

        Ok(Arc::new(Pipeline {
            device: Arc::clone(self),
            _render_pass: None,
            signature: Arc::clone(signature),
            native: pipeline,
            bind_point: vk::PipelineBindPoint::COMPUTE,
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

    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        for queue in &self.queues {
            queue.wait_idle()?;
        }
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let pipeline_cache = self.pipeline_cache.lock().unwrap();
        unsafe { self.wrapper.0.destroy_pipeline_cache(*pipeline_cache, None) };

        self.allocator.destroy();
    }
}
