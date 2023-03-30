use crate::format::{FormatFeatureFlags, BUFFER_FORMATS};
use crate::sampler::{SamplerClamp, SamplerFilter, SamplerMipmap};
use crate::shader::{BindingLoc, ShaderStage, VInputRate};
use crate::PipelineSignature;
use crate::FORMAT_SIZES;
use crate::IMAGE_FORMATS;
use crate::{
    utils, BindingType, Format, Image, Queue, QueueType, Semaphore, Surface, Swapchain,
    {Buffer, BufferUsageFlags, DeviceBuffer, HostBuffer}, {ImageType, ImageUsageFlags},
};
use crate::{Adapter, PipelineDepthStencil, SubpassDependency};
use crate::{Attachment, Pipeline, PipelineRasterization, PrimitiveTopology, ShaderBinding};
use crate::{AttachmentColorBlend, ImageWrapper};
use crate::{LoadStore, RenderPass, Shader};
use crate::{QueryPool, Subpass};
use crate::{Sampler, DEPTH_FORMAT};
use crate::{SwapchainWrapper, BC_IMAGE_FORMATS};
use ash::vk;
use ash::vk::Handle;
use common::parking_lot::Mutex;
use common::types::HashMap;
use spirv_cross::glsl;
use spirv_cross::spirv;
use std::collections::hash_map;
use std::ffi::CString;
use std::ptr;
use std::sync::atomic::AtomicUsize;
use std::sync::{atomic, Arc};
use std::{ffi::CStr, marker::PhantomData, mem, slice};

#[derive(Debug)]
pub enum DeviceError {
    VkError(vk::Result),
    ZeroBufferElementSize,
    ZeroBufferSize,
    SwapchainError(String),
    SpirvError(spirv_cross::ErrorCode),
    InvalidShader(String),
    InvalidSignature(&'static str),
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

pub(crate) struct DeviceWrapper {
    pub(crate) native: ash::Device,
    pub(crate) adapter: Arc<Adapter>,
    pub(crate) ts_khr: ash::extensions::khr::TimelineSemaphore,
}

impl DeviceWrapper {
    pub(crate) unsafe fn debug_set_object_name(
        &self,
        ty: vk::ObjectType,
        handle: u64,
        name: &str,
    ) -> Result<(), vk::Result> {
        if cfg!(debug_assertions) {
            let c_name = CString::new(name).unwrap();

            let info = vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(ty)
                .object_handle(handle)
                .object_name(c_name.as_c_str());

            if let Some(debug_utils) = &self.adapter.instance.debug_utils_ext {
                debug_utils.set_debug_utils_object_name(self.native.handle(), &info)
            } else {
                unreachable!()
            }
        } else {
            Ok(())
        }
    }

    pub fn create_sampler(
        self: &Arc<Self>,
        mag_filter: SamplerFilter,
        min_filter: SamplerFilter,
        mipmap: SamplerMipmap,
        clamp: SamplerClamp,
        max_anisotropy: f32,
    ) -> Result<Arc<Sampler>, vk::Result> {
        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(mag_filter.0)
            .min_filter(min_filter.0)
            .mipmap_mode(mipmap.0)
            .address_mode_u(clamp.0)
            .address_mode_v(clamp.0)
            .address_mode_w(clamp.0)
            .anisotropy_enable(true)
            .max_anisotropy(max_anisotropy)
            .compare_enable(false)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .unnormalized_coordinates(false);

        Ok(Arc::new(Sampler {
            device_wrapper: Arc::clone(self),
            native: unsafe { self.native.create_sampler(&sampler_info, None)? },
            min_filter,
            mag_filter,
            mipmap,
        }))
    }
}

impl Drop for DeviceWrapper {
    fn drop(&mut self) {
        unsafe {
            self.native.device_wait_idle().unwrap();
            self.native.destroy_device(None);
        }
    }
}

pub struct Device {
    pub(crate) wrapper: Arc<DeviceWrapper>,
    pub(crate) allocator: vma::VmaAllocator,
    pub(crate) host_mem_pool: vma::VmaPool,
    pub(crate) device_mem_pool: vma::VmaPool,
    pub(crate) total_used_dev_memory: Arc<AtomicUsize>,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
    pub(crate) queues: Vec<Arc<Queue>>,
    pub(crate) default_sampler: Arc<Sampler>,
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
        native: unsafe { device_wrapper.native.create_semaphore(&create_info, None)? },
        semaphore_type: sp_type,
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

// pub(crate) fn create_fence(device_wrapper: &Arc<DeviceWrapper>) -> Result<Fence, vk::Result> {
//     let mut create_info = vk::FenceCreateInfo::builder().build();
//     create_info.flags = vk::FenceCreateFlags::SIGNALED;
//     Ok(Fence {
//         device_wrapper: Arc::clone(device_wrapper),
//         native: unsafe { device_wrapper.native.create_fence(&create_info, None)? },
//     })
// }

#[derive(Clone, Copy, Eq, PartialEq)]
enum MemoryUsage {
    Host,
    Device,
}

impl Device {
    pub fn adapter(&self) -> &Arc<Adapter> {
        &self.wrapper.adapter
    }

    pub fn is_extension_supported(&self, name: &str) -> bool {
        self.wrapper.adapter.is_extension_enabled(name)
    }

    pub fn get_queue(&self, queue_type: QueueType) -> &Queue {
        &self.queues[queue_type as usize]
    }

    pub fn calc_real_device_mem_usage(&self) -> u64 {
        self.total_used_dev_memory.load(atomic::Ordering::Relaxed) as u64
    }

    pub fn create_binary_semaphore(self: &Arc<Self>) -> Result<Semaphore, DeviceError> {
        Ok(create_binary_semaphore(&self.wrapper)?)
    }

    pub fn create_timeline_semaphore(self: &Arc<Self>) -> Result<Semaphore, DeviceError> {
        Ok(create_timeline_semaphore(&self.wrapper)?)
    }

    fn create_buffer(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        elem_size: u64,
        size: u64,
        mem_usage: MemoryUsage,
        name: &str,
    ) -> Result<(Buffer, vma::VmaAllocationInfo), DeviceError> {
        if elem_size == 0 {
            return Err(DeviceError::ZeroBufferElementSize);
        }
        if size == 0 {
            return Err(DeviceError::ZeroBufferSize);
        }

        let elem_align = 1;
        let aligned_elem_size = utils::make_mul_of_u64(elem_size, elem_align as u64);
        let bytesize = aligned_elem_size as u64 * size;

        let buffer_info = vk::BufferCreateInfo::builder()
            .usage(usage.0)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(bytesize as vk::DeviceSize);

        let used_dev_memory = bytesize * (mem_usage == MemoryUsage::Device) as u64;

        let (buffer, allocation, alloc_info) = unsafe {
            let native_buffer = self.wrapper.native.create_buffer(&buffer_info, None)?;

            let mut alloc_create_info: vma::VmaAllocationCreateInfo = mem::zeroed();
            if mem_usage == MemoryUsage::Host {
                alloc_create_info.flags = vma::VMA_ALLOCATION_CREATE_MAPPED_BIT as u32;
                alloc_create_info.pool = self.host_mem_pool;
            } else {
                alloc_create_info.pool = self.device_mem_pool;
            };

            let mut allocation: vma::VmaAllocation = mem::zeroed();
            let mut alloc_info: vma::VmaAllocationInfo = mem::zeroed();
            let result = vma::vmaAllocateMemoryForBuffer(
                self.allocator,
                native_buffer.as_raw() as vma::VkBuffer,
                &alloc_create_info,
                &mut allocation,
                &mut alloc_info,
            );

            if result != vma::VK_SUCCESS {
                return Err(vk::Result::from_raw(result).into());
            }

            let result = vma::vmaBindBufferMemory(
                self.allocator,
                allocation,
                native_buffer.as_raw() as vma::VkBuffer,
            );

            if result != vma::VK_SUCCESS {
                return Err(vk::Result::from_raw(result).into());
            }

            self.wrapper
                .debug_set_object_name(vk::ObjectType::BUFFER, native_buffer.as_raw(), name)?;

            (native_buffer, allocation, alloc_info)
        };

        self.total_used_dev_memory
            .fetch_add(used_dev_memory as usize, atomic::Ordering::Relaxed);

        Ok((
            Buffer {
                device: Arc::clone(self),
                native: buffer,
                allocation,
                used_dev_memory,
                elem_size,
                aligned_elem_size: aligned_elem_size as u64,
                size,
                _bytesize: bytesize as u64,
            },
            alloc_info,
        ))
    }

    pub fn create_host_buffer_named<T>(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        size: u64,
        name: &str,
    ) -> Result<HostBuffer<T>, DeviceError> {
        let (buffer, alloc_info) =
            self.create_buffer(usage, mem::size_of::<T>() as u64, size, MemoryUsage::Host, name)?;

        assert_ne!(alloc_info.pMappedData, ptr::null_mut());

        Ok(HostBuffer {
            _type_marker: PhantomData,
            buffer: Arc::new(buffer),
            p_data: alloc_info.pMappedData as *mut u8,
        })
    }

    pub fn create_host_buffer<T>(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        size: u64,
    ) -> Result<HostBuffer<T>, DeviceError> {
        self.create_host_buffer_named(usage, size, "")
    }

    pub fn create_device_buffer_named(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        element_size: u64,
        size: u64,
        name: &str,
    ) -> Result<DeviceBuffer, DeviceError> {
        let (buffer, _) = self.create_buffer(usage, element_size, size, MemoryUsage::Device, name)?;
        Ok(DeviceBuffer {
            buffer: Arc::new(buffer),
        })
    }

    pub fn create_device_buffer(
        self: &Arc<Self>,
        usage: BufferUsageFlags,
        element_size: u64,
        size: u64,
    ) -> Result<DeviceBuffer, DeviceError> {
        self.create_device_buffer_named(usage, element_size, size, "")
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    pub fn create_image(
        self: &Arc<Self>,
        image_type: ImageType,
        is_array: bool,
        format: Format,
        max_mip_levels: u32,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32, u32),
        name: &str,
    ) -> Result<Arc<Image>, DeviceError> {
        if image_type == Image::TYPE_3D {
            assert_eq!(is_array, false);
        }

        if !IMAGE_FORMATS.contains_key(&format)
            && !BC_IMAGE_FORMATS.contains(&format)
            && format != DEPTH_FORMAT
        {
            panic!("Image format {:?} is not supported!", format);
        }

        let format_props = self.wrapper.adapter.get_image_format_properties(
            format.0,
            image_type.0,
            vk::ImageTiling::OPTIMAL,
            usage.0,
        )?;

        let mut size = preferred_size;
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

        let tiling = vk::ImageTiling::OPTIMAL;
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(image_type.0)
            .format(format.0)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(usage.0)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let (image, allocation, alloc_info) = unsafe {
            let native_image = self.wrapper.native.create_image(&image_info, None)?;

            let mut alloc_create_info: vma::VmaAllocationCreateInfo = mem::zeroed();
            alloc_create_info.pool = self.device_mem_pool;

            let mut allocation: vma::VmaAllocation = mem::zeroed();
            let mut alloc_info: vma::VmaAllocationInfo = mem::zeroed();
            let result = vma::vmaAllocateMemoryForImage(
                self.allocator,
                native_image.as_raw() as vma::VkImage,
                &alloc_create_info,
                &mut allocation,
                &mut alloc_info,
            );

            if result != vma::VK_SUCCESS {
                return Err(vk::Result::from_raw(result).into());
            }

            // let memory_block = self.allocator.lock().alloc(
            //     AshMemoryDevice::wrap(&self.wrapper.native),
            //     gpu_alloc::Request {
            //         size: req.size,
            //         align_mask: req.alignment,
            //         usage: mem_usage,
            //         memory_types: self.allowed_memory_types & req.memory_type_bits,
            //     },
            // )?;

            let result =
                vma::vmaBindImageMemory(self.allocator, allocation, native_image.as_raw() as vma::VkImage);

            if result != vma::VK_SUCCESS {
                return Err(vk::Result::from_raw(result).into());
            }

            // let memory_block = self.allocator.lock().alloc(
            //     AshMemoryDevice::wrap(&self.wrapper.native),
            //     gpu_alloc::Request {
            //         size: req.size,
            //         align_mask: req.alignment,
            //         usage: gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS,
            //         memory_types: self.allowed_memory_types & req.memory_type_bits,
            //     },
            // )?;

            // self.wrapper.native.bind_image_memory(
            //     native_image,
            //     *memory_block.memory(),
            //     memory_block.offset(),
            // )?;

            self.wrapper
                .debug_set_object_name(vk::ObjectType::IMAGE, native_image.as_raw(), name)?;

            (native_image, allocation, alloc_info)
        };
        let bytesize = alloc_info.size;

        let aspect = if format == DEPTH_FORMAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        self.total_used_dev_memory
            .fetch_add(bytesize as usize, atomic::Ordering::Relaxed);

        let image_wrapper = Arc::new(ImageWrapper {
            device: Arc::clone(self),
            _swapchain_wrapper: None,
            native: image,
            allocation: Some(allocation),
            bytesize,
            is_array,
            owned_handle: true,
            ty: image_type,
            format,
            aspect,
            tiling,
            name: name.to_owned(),
        });
        let view = image_wrapper.create_view("view").build()?;

        Ok(Arc::new(Image {
            wrapper: image_wrapper,
            view,
            size,
            mip_levels,
        }))
    }

    pub fn create_sampler(
        self: &Arc<Self>,
        mag_filter: SamplerFilter,
        min_filter: SamplerFilter,
        mipmap: SamplerMipmap,
        clamp: SamplerClamp,
        max_anisotropy: f32,
    ) -> Result<Arc<Sampler>, DeviceError> {
        Ok(self
            .wrapper
            .create_sampler(mag_filter, min_filter, mipmap, clamp, max_anisotropy)?)
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    pub fn create_image_2d(
        self: &Arc<Self>,
        format: Format,
        max_mip_levels: u32,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32),
    ) -> Result<Arc<Image>, DeviceError> {
        self.create_image_2d_named(format, max_mip_levels, usage, preferred_size, "")
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    pub fn create_image_2d_named(
        self: &Arc<Self>,
        format: Format,
        max_mip_levels: u32,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32),
        name: &str,
    ) -> Result<Arc<Image>, DeviceError> {
        self.create_image(
            Image::TYPE_2D,
            false,
            format,
            max_mip_levels,
            usage,
            (preferred_size.0, preferred_size.1, 1),
            name,
        )
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    pub fn create_image_2d_array(
        self: &Arc<Self>,
        format: Format,
        max_mip_levels: u32,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32, u32),
    ) -> Result<Arc<Image>, DeviceError> {
        self.create_image_2d_array_named(format, max_mip_levels, usage, preferred_size, "")
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    pub fn create_image_2d_array_named(
        self: &Arc<Self>,
        format: Format,
        max_mip_levels: u32,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32, u32),
        name: &str,
    ) -> Result<Arc<Image>, DeviceError> {
        self.create_image(
            Image::TYPE_2D,
            true,
            format,
            max_mip_levels,
            usage,
            preferred_size,
            name,
        )
    }

    pub fn create_image_3d(
        self: &Arc<Self>,
        format: Format,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32, u32),
    ) -> Result<Arc<Image>, DeviceError> {
        self.create_image(Image::TYPE_3D, false, format, 1, usage, preferred_size, "")
    }

    pub fn create_query_pool(self: &Arc<Self>, query_count: u32) -> Result<Arc<QueryPool>, vk::Result> {
        let create_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(query_count);

        Ok(Arc::new(QueryPool {
            device: Arc::clone(self),
            native: unsafe { self.wrapper.native.create_query_pool(&create_info, None)? },
        }))
    }

    pub fn swapchain_min_max_images(&self, surface: &Surface) -> Result<(u32, u32), DeviceError> {
        let surface_capabs = self.wrapper.adapter.get_surface_capabilities(&surface)?;
        Ok((surface_capabs.min_image_count, surface_capabs.max_image_count))
    }

    pub fn create_swapchain(
        self: &Arc<Self>,
        surface: &Arc<Surface>,
        preferred_size: (u32, u32),
        vsync: bool,
        preferred_n_images: u32,
        old_swapchain: Option<Swapchain>,
    ) -> Result<Swapchain, DeviceError> {
        let surface_capabs = self.wrapper.adapter.get_surface_capabilities(&surface)?;
        let surface_formats = self.wrapper.adapter.get_surface_formats(&surface)?;
        let surface_present_modes = self.wrapper.adapter.get_surface_present_modes(&surface)?;

        let image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
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

        // TODO: use surface_capabs.current_extend as size and preferred_size as backup

        let size = (
            preferred_size
                .0
                .min(surface_capabs.max_image_extent.width)
                .max(surface_capabs.min_image_extent.width),
            preferred_size
                .1
                .min(surface_capabs.max_image_extent.height)
                .max(surface_capabs.min_image_extent.height),
        );

        let mut s_format = surface_formats.iter().find(|&s_format| {
            matches!(
                s_format.format,
                vk::Format::R8G8B8A8_SRGB | vk::Format::B8G8R8A8_SRGB
            ) && s_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
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

        let min_image_count =
            preferred_n_images.clamp(surface_capabs.min_image_count, surface_capabs.max_image_count);

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

        let mut create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.native)
            .min_image_count(min_image_count)
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

        if let Some(old_swapchain) = &old_swapchain {
            for img in &old_swapchain.images {
                if Arc::strong_count(img) > 1 {
                    return Err(DeviceError::SwapchainError(
                        "old_swapchain images must not be used anywhere at the time of retire of old_swapchain!"
                            .to_string(),
                    ));
                }
            }
            create_info = create_info.old_swapchain(*old_swapchain.wrapper.native.lock());
        }

        let swapchain_wrapper = Arc::new(SwapchainWrapper {
            device: Arc::clone(self),
            _surface: Arc::clone(surface),
            native: Mutex::new(unsafe { self.swapchain_khr.create_swapchain(&create_info, None)? }),
        });

        let images: Result<Vec<Arc<Image>>, DeviceError> = unsafe {
            let native_swapchain = swapchain_wrapper.native.lock();
            self.swapchain_khr.get_swapchain_images(*native_swapchain)?
        }
        .iter()
        .enumerate()
        .map(|(i, &native_image)| {
            let name = format!("swapchain-img{}", i);
            unsafe {
                self.wrapper
                    .debug_set_object_name(vk::ObjectType::IMAGE, native_image.as_raw(), &name)?
            };

            let image_wrapper = Arc::new(ImageWrapper {
                device: Arc::clone(self),
                _swapchain_wrapper: Some(Arc::clone(&swapchain_wrapper)),
                native: native_image,
                allocation: None,
                bytesize: 0,
                is_array: false,
                owned_handle: false,
                ty: Image::TYPE_2D,
                format: Format(s_format.format),
                aspect: vk::ImageAspectFlags::COLOR,
                tiling: vk::ImageTiling::OPTIMAL,
                name,
            });
            let view = image_wrapper.create_view("view").build()?;

            Ok(Arc::new(Image {
                wrapper: image_wrapper,
                view,
                size: (size.0, size.1, 1),
                mip_levels: 1,
            }))
        })
        .collect();

        Ok(Swapchain {
            wrapper: swapchain_wrapper,
            readiness_semaphore: Arc::new(create_binary_semaphore(&self.wrapper)?),
            images: images?,
        })
    }

    fn create_shader(
        self: &Arc<Self>,
        code: &[u8],
        vertex_inputs: &HashMap<&str, (Format, VInputRate)>,
        name: &str,
    ) -> Result<Arc<Shader>, DeviceError> {
        #[allow(clippy::cast_ptr_alignment)]
        let code_words = unsafe {
            slice::from_raw_parts(
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

        macro_rules! binding_image_array_size {
            ($res: ident, $img_type: ident, $desc_type: ident) => {{
                let var_type = ast.get_type($res.type_id)?;
                (
                    BindingLoc::new(
                        ast.get_decoration($res.id, spirv::Decoration::DescriptorSet)
                            .unwrap(),
                        ast.get_decoration($res.id, spirv::Decoration::Binding)?,
                    ),
                    ShaderBinding {
                        stage_flags: stage,
                        binding_type: BindingType(vk::DescriptorType::$desc_type),
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
                    },
                )
            }};
        }

        macro_rules! binding_buffer_array_size {
            ($res: ident, $desc_type0: ident) => {{
                let var_type = ast.get_type($res.type_id)?;
                (
                    BindingLoc::new(
                        ast.get_decoration($res.id, spirv::Decoration::DescriptorSet)
                            .unwrap(),
                        ast.get_decoration($res.id, spirv::Decoration::Binding)?,
                    ),
                    ShaderBinding {
                        stage_flags: stage,
                        binding_type: BindingType(vk::DescriptorType::$desc_type0),
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
                    },
                )
            }};
        }

        let mut vertex_loc_inputs =
            HashMap::<u32, (Format, VInputRate)>::with_capacity(resources.stage_inputs.len());

        if stage == ShaderStage::VERTEX {
            for (f, _) in vertex_inputs.values() {
                if !BUFFER_FORMATS[f].contains(FormatFeatureFlags::VERTEX_BUFFER) {
                    panic!("Unsupported vertex format is used: {:?}", f);
                }
            }
            for res in resources.stage_inputs {
                let f = *vertex_inputs
                    .get(res.name.as_str())
                    .ok_or(DeviceError::InvalidShader(format!(
                        "Input format for {} not provided!",
                        res.name
                    )))?;
                let location = ast.get_decoration(res.id, spirv::Decoration::Location).unwrap();

                vertex_loc_inputs.insert(location, f);
            }
        }

        let mut named_bindings = HashMap::new();

        for res in &resources.sampled_images {
            let binding = binding_image_array_size!(res, SampledImage, COMBINED_IMAGE_SAMPLER);
            named_bindings.insert(res.name.clone(), binding);
        }
        for res in &resources.storage_images {
            let binding = binding_image_array_size!(res, Image, STORAGE_IMAGE);
            named_bindings.insert(res.name.clone(), binding);
        }
        for res in &resources.uniform_buffers {
            let binding = binding_buffer_array_size!(res, UNIFORM_BUFFER);
            named_bindings.insert(res.name.clone(), binding);
        }
        for res in &resources.storage_buffers {
            let binding = binding_buffer_array_size!(res, STORAGE_BUFFER);
            named_bindings.insert(res.name.clone(), binding);
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
        let native = unsafe { self.wrapper.native.create_shader_module(&create_info, None)? };

        unsafe {
            self.wrapper
                .debug_set_object_name(vk::ObjectType::SHADER_MODULE, native.as_raw(), name)?;
        }

        Ok(Arc::new(Shader {
            device: Arc::clone(self),
            native,
            stage,
            vertex_location_inputs: vertex_loc_inputs,
            named_bindings,
            _push_constants: push_constants,
            push_constants_size,
        }))
    }

    pub fn create_vertex_shader(
        self: &Arc<Self>,
        code: &[u8],
        input_formats: &[(&str, Format, VInputRate)],
        name: &str,
    ) -> Result<Arc<Shader>, DeviceError> {
        self.create_shader(
            code,
            &input_formats
                .iter()
                .cloned()
                .map(|(name, format, rate)| (name, (format, rate)))
                .collect(),
            name,
        )
    }

    pub fn create_pixel_shader(
        self: &Arc<Self>,
        code: &[u8],
        name: &str,
    ) -> Result<Arc<Shader>, DeviceError> {
        self.create_shader(code, &HashMap::new(), name)
    }

    pub fn create_compute_shader(
        self: &Arc<Self>,
        code: &[u8],
        name: &str,
    ) -> Result<Arc<Shader>, DeviceError> {
        self.create_shader(code, &HashMap::new(), name)
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
                    LoadStore::InitLoad => {
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
                    LoadStore::InitLoadFinalSave => {
                        native_info.load_op = vk::AttachmentLoadOp::LOAD;
                        native_info.store_op = vk::AttachmentStoreOp::STORE;
                    }
                }
                native_info
            })
            .collect();

        let mut max_attachment_refs = 0;
        for subpass in subpasses {
            // + 1 for depth
            max_attachment_refs += subpass.input.len() + subpass.color.len() + 1;
        }

        let mut input_attachments = Vec::with_capacity(attachments.len());
        let mut color_attachments = Vec::with_capacity(attachments.len());
        let mut depth_attachments = Vec::with_capacity(attachments.len());

        let mut native_attachment_refs = Vec::with_capacity(max_attachment_refs);
        let mut native_subpass_descs = Vec::with_capacity(subpasses.len());

        for subpass in subpasses {
            // Input attachments
            let input_att_ref_index = native_attachment_refs.len();
            for attachment in &subpass.input {
                native_attachment_refs.push(
                    vk::AttachmentReference::builder()
                        .attachment(attachment.index)
                        .layout(attachment.layout.0)
                        .build(),
                );
                if !input_attachments.contains(&attachment.index) {
                    input_attachments.push(attachment.index);
                }
            }

            // Color attachments
            let color_att_ref_index = native_attachment_refs.len();
            for attachment in &subpass.color {
                native_attachment_refs.push(
                    vk::AttachmentReference::builder()
                        .attachment(attachment.index)
                        .layout(attachment.layout.0)
                        .build(),
                );
                if !color_attachments.contains(&attachment.index) {
                    color_attachments.push(attachment.index);
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
                .input_attachments(
                    &native_attachment_refs[input_att_ref_index..input_att_ref_index + subpass.input.len()],
                )
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
            native: unsafe { self.wrapper.native.create_render_pass(&create_info, None)? },
            subpasses: subpasses.into(),
            attachments: attachments.to_vec(),
            _input_attachments: input_attachments,
            _color_attachments: color_attachments,
            _depth_attachments: depth_attachments,
        }))
    }

    /// Creates pipelines signature with bindings from `shaders`. Uses `additional_bindings` to override
    /// present bindings from shaders.
    pub fn create_pipeline_signature(
        self: &Arc<Self>,
        shaders: &[Arc<Shader>],
        additional_bindings: &[(BindingLoc, ShaderBinding)],
    ) -> Result<Arc<PipelineSignature>, vk::Result> {
        let mut combined_bindings: HashMap<BindingLoc, ShaderBinding> =
            additional_bindings.iter().cloned().collect();

        let mut push_constants_size = 0u32;

        for shader in shaders {
            for (_name, (loc, binding)) in &shader.named_bindings {
                let set = loc.descriptor_set;

                if let Some(overriding) = combined_bindings.get_mut(loc) {
                    // Apply overridden binding type
                    if binding.binding_type != overriding.binding_type
                        && (!matches!(
                            binding.binding_type,
                            BindingType::UNIFORM_BUFFER | BindingType::UNIFORM_BUFFER_DYNAMIC
                        ) || !matches!(
                            overriding.binding_type,
                            BindingType::UNIFORM_BUFFER | BindingType::UNIFORM_BUFFER_DYNAMIC
                        ))
                    {
                        panic!(
                            "Conflicting binding types (set {}, binding {}) in the shader and additional_bindings",
                            set, loc.id
                        );
                    }
                    if binding.count != overriding.count {
                        panic!(
                            "Conflicting number of descriptors (set {}, binding {}) in the shader and additional_bindings",
                            set, loc.id
                        );
                    }
                    overriding.stage_flags |= binding.stage_flags;
                } else {
                    // Add a new binding
                    combined_bindings.insert(*loc, *binding);
                }
            }

            push_constants_size += shader.push_constants_size;
        }

        // MAX 4 descriptor sets
        let mut native_bindings: [Vec<vk::DescriptorSetLayoutBinding>; 4] = Default::default();
        let mut binding_flags: [Vec<vk::DescriptorBindingFlags>; 4] = Default::default();
        let mut descriptor_sizes: [Vec<vk::DescriptorPoolSize>; 4] = Default::default();
        let mut descriptor_sizes_indices: [HashMap<vk::DescriptorType, usize>; 4] = Default::default();
        let mut binding_types: [HashMap<u32, vk::DescriptorType>; 4] = Default::default();

        for (loc, binding) in &combined_bindings {
            let set = loc.descriptor_set as usize;
            let descriptor_sizes = &mut descriptor_sizes[set];
            let descriptor_sizes_indices = &mut descriptor_sizes_indices[set];
            let native_bindings = &mut native_bindings[set];

            if let hash_map::Entry::Vacant(entry) = descriptor_sizes_indices.entry(binding.binding_type.0) {
                entry.insert(descriptor_sizes.len());
                descriptor_sizes.push(
                    vk::DescriptorPoolSize::builder()
                        .ty(binding.binding_type.0)
                        .descriptor_count(0)
                        .build(),
                );
            }

            descriptor_sizes[descriptor_sizes_indices[&binding.binding_type.0]].descriptor_count +=
                binding.count;
            binding_types[set].insert(loc.id, binding.binding_type.0);

            native_bindings.push(
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(loc.id)
                    .descriptor_type(binding.binding_type.0)
                    .descriptor_count(binding.count)
                    .stage_flags(binding.stage_flags.0)
                    .build(),
            );

            let mut flags = vk::DescriptorBindingFlags::default();
            if binding.count > 1 {
                flags |= vk::DescriptorBindingFlags::PARTIALLY_BOUND;
            }

            binding_flags[set].push(flags);
        }

        let mut native_descriptor_sets = [vk::DescriptorSetLayout::default(); 4];

        for (set_idx, bindings) in native_bindings.iter().enumerate() {
            let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&binding_flags[set_idx]);

            let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(bindings)
                .push_next(&mut binding_flags_info);

            native_descriptor_sets[set_idx] = unsafe {
                self.wrapper
                    .native
                    .create_descriptor_set_layout(&create_info, None)?
            };
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

            unsafe {
                self.wrapper
                    .native
                    .create_pipeline_layout(&layout_create_info, None)?
            }
        };

        Ok(Arc::new(PipelineSignature {
            device: Arc::clone(self),
            native: native_descriptor_sets,
            pipeline_layout,
            descriptor_sizes,
            binding_types,
            _push_constants_size: push_constants_size,
            shaders,
            bindings: combined_bindings,
        }))
    }

    pub fn load_pipeline_cache(&self, data: &[u8]) -> Result<(), vk::Result> {
        let mut pipeline_cache = self.pipeline_cache.lock();

        unsafe { self.wrapper.native.destroy_pipeline_cache(*pipeline_cache, None) };
        let create_info = vk::PipelineCacheCreateInfo::builder().initial_data(data);
        *pipeline_cache = unsafe { self.wrapper.native.create_pipeline_cache(&create_info, None)? };

        Ok(())
    }

    pub fn get_pipeline_cache(&self) -> Result<Vec<u8>, vk::Result> {
        let pipeline_cache = self.pipeline_cache.lock();
        unsafe { self.wrapper.native.get_pipeline_cache_data(*pipeline_cache) }
    }

    pub fn create_graphics_pipeline(
        self: &Arc<Self>,
        render_pass: &Arc<RenderPass>,
        subpass_index: u32,
        primitive_topology: PrimitiveTopology,
        depth_stencil: PipelineDepthStencil,
        rasterization: PipelineRasterization,
        attachment_blends: &[(u32, AttachmentColorBlend)],
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
        let vertex_binding_count = vertex_shader.vertex_location_inputs.len();
        let mut vertex_binding_descs =
            vec![vk::VertexInputBindingDescription::default(); vertex_binding_count];
        let mut vertex_attrib_descs =
            vec![vk::VertexInputAttributeDescription::default(); vertex_binding_count];

        for (location, (format, rate)) in &vertex_shader.vertex_location_inputs {
            let buffer_index = *location;

            if !BUFFER_FORMATS[&format].contains(FormatFeatureFlags::VERTEX_BUFFER) {
                panic!("Unsupported vertex format is used: {:?}", format);
            }

            vertex_binding_descs[buffer_index as usize] = vk::VertexInputBindingDescription {
                binding: buffer_index,
                stride: FORMAT_SIZES[&format] as u32,
                input_rate: rate.0,
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
        let overridden_blends: HashMap<u32, AttachmentColorBlend> =
            attachment_blends.iter().cloned().collect();
        let n_color_attachments = render_pass.subpasses[subpass_index as usize].color.len();
        let blend_attachment_states: Vec<_> = (0..n_color_attachments)
            .map(|i| {
                if let Some(overridden) = overridden_blends.get(&(i as u32)) {
                    overridden.0
                } else {
                    AttachmentColorBlend::default().0
                }
            })
            .collect();
        let color_blend_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&blend_attachment_states);

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

        let pipeline_cache = self.pipeline_cache.lock();
        let native_pipeline = unsafe {
            self.wrapper
                .native
                .create_graphics_pipelines(*pipeline_cache, &[create_info.build()], None)
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

        let pipeline_cache = self.pipeline_cache.lock();
        let native_pipeline = unsafe {
            self.wrapper
                .native
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

    pub fn wait_idle(&self) -> Result<(), vk::Result> {
        for queue in &self.queues {
            queue.wait_idle()?;
        }
        Ok(())
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let pipeline_cache = self.pipeline_cache.lock();

        unsafe {
            self.wrapper.native.destroy_pipeline_cache(*pipeline_cache, None);
            // self.allocator
            //     .lock()
            //     .cleanup(AshMemoryDevice::wrap(&self.wrapper.native));

            vma::vmaDestroyPool(self.allocator, self.host_mem_pool);
            vma::vmaDestroyPool(self.allocator, self.device_mem_pool);
            vma::vmaDestroyAllocator(self.allocator);
        }
    }
}

unsafe impl Send for Device {}

unsafe impl Sync for Device {}
