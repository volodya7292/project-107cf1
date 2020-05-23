use crate::adapter::Adapter;
use crate::queue::{SubmitInfo, SubmitPacket};
use crate::{
    utils, Fence, Format, Image, Queue, QueueType, Semaphore, Surface, Swapchain,
    {Buffer, BufferUsageFlags, DeviceBuffer, HostBuffer}, {ImageType, ImageUsageFlags},
};
use ash::version::DeviceV1_0;
use ash::vk;
use std::cell::Cell;
use std::{cmp, marker::PhantomData, mem, ptr, rc::Rc};

#[derive(Debug)]
pub enum DeviceError {
    VmaError(vk_mem::Error),
    VkError(vk::Result),
    SwapchainError(String),
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

pub struct Device {
    pub(crate) adapter: Rc<Adapter>,
    pub(crate) native: Rc<ash::Device>,
    pub(crate) allocator: vk_mem::Allocator,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
    pub(crate) queues: Vec<Rc<Queue>>,
}

pub(crate) fn create_semaphore(
    native_device: &Rc<ash::Device>,
    sp_type: vk::SemaphoreType,
) -> Result<Semaphore, vk::Result> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::builder()
        .initial_value(0)
        .semaphore_type(sp_type);
    let create_info = vk::SemaphoreCreateInfo::builder().push_next(&mut type_info);
    Ok(Semaphore {
        native_device: Rc::clone(native_device),
        native: unsafe { native_device.create_semaphore(&create_info, None)? },
        semaphore_type: sp_type,
        last_signal_value: Cell::new(0),
    })
}

pub(crate) fn create_binary_semaphore(native_device: &Rc<ash::Device>) -> Result<Semaphore, vk::Result> {
    create_semaphore(native_device, vk::SemaphoreType::BINARY)
}

pub(crate) fn create_timeline_semaphore(native_device: &Rc<ash::Device>) -> Result<Semaphore, vk::Result> {
    create_semaphore(native_device, vk::SemaphoreType::TIMELINE)
}

pub(crate) fn create_fence(native_device: &Rc<ash::Device>, signaled: bool) -> Result<Fence, vk::Result> {
    let mut create_info = vk::FenceCreateInfo::builder().build();
    if signaled {
        create_info.flags |= vk::FenceCreateFlags::SIGNALED;
    }
    Ok(Fence {
        native_device: Rc::clone(native_device),
        native: unsafe { native_device.create_fence(&create_info, None)? },
    })
}

impl Device {
    pub fn is_extension_supported(&self, name: &str) -> bool {
        self.adapter.is_extension_enabled(name)
    }

    pub fn get_queue(&self, queue_type: QueueType) -> &Queue {
        &self.queues[queue_type.0 as usize]
    }

    fn create_buffer<T>(
        self: &Rc<Self>,
        usage: BufferUsageFlags,
        size: u64,
        mem_usage: vk_mem::MemoryUsage,
    ) -> Result<(Buffer<T>, vk_mem::AllocationInfo), DeviceError> {
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
                device: Rc::clone(self),
                _type_marker: PhantomData,
                native: buffer,
                allocation: alloc,
                aligned_elem_size: aligned_elem_size as u64,
                size,
                bytesize: bytesize as u64,
            },
            alloc_info,
        ))
    }

    pub fn create_host_buffer<T>(
        self: &Rc<Self>,
        usage: BufferUsageFlags,
        size: u64,
    ) -> Result<HostBuffer<T>, DeviceError> {
        let (buffer, alloc_info) = self.create_buffer::<T>(usage, size, vk_mem::MemoryUsage::CpuOnly)?;

        let p_data = alloc_info.get_mapped_data();
        if p_data == ptr::null_mut() {}

        Ok(HostBuffer {
            buffer,
            p_data: alloc_info.get_mapped_data(),
        })
    }

    pub fn create_device_buffer<T>(
        self: &Rc<Self>,
        usage: BufferUsageFlags,
        size: u64,
    ) -> Result<DeviceBuffer<T>, DeviceError> {
        let (buffer, _) = self.create_buffer::<T>(usage, size, vk_mem::MemoryUsage::GpuOnly)?;
        Ok(DeviceBuffer { buffer })
    }

    fn create_image(
        self: &Rc<Self>,
        image_type: ImageType,
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

        Ok(Rc::new(Image {
            device: Rc::clone(self),
            native: image,
            allocation: alloc,
            owned_handle: true,
            format,
            size,
        }))
    }

    pub fn create_image_2d(
        self: &Rc<Self>,
        format: Format,
        mipmaps: bool,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32),
    ) -> Result<Rc<Image>, DeviceError> {
        self.create_image(
            Image::TYPE_2D,
            format,
            mipmaps,
            usage,
            (preferred_size.0, preferred_size.1, 1),
        )
    }

    pub fn create_image_3d(
        self: &Rc<Self>,
        format: Format,
        usage: ImageUsageFlags,
        preferred_size: (u32, u32, u32),
    ) -> Result<Rc<Image>, DeviceError> {
        self.create_image(Image::TYPE_2D, format, false, usage, preferred_size)
    }

    pub fn create_swapchain(
        self: &Rc<Self>,
        surface: &Rc<Surface>,
        size: (u32, u32),
        vsync: bool,
    ) -> Result<Swapchain, DeviceError> {
        let surface_capabs = self.adapter.get_surface_capabilities(&surface)?;
        let surface_formats = self.adapter.get_surface_formats(&surface)?;
        let surface_present_modes = self.adapter.get_surface_present_modes(&surface)?;

        // TODO: HDR metadata

        let s_format = surface_formats.iter().find(|&s_format| {
            s_format.format == vk::Format::R16G16B16A16_SFLOAT
                && s_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        });
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
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_capabs.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let native_swapchain = unsafe { self.swapchain_khr.create_swapchain(&create_info, None)? };
        let images: Vec<Image> = unsafe { self.swapchain_khr.get_swapchain_images(native_swapchain)? }
            .iter()
            .map(|&native_image| Image {
                device: Rc::clone(self),
                native: native_image,
                allocation: vk_mem::Allocation::null(),
                owned_handle: false,
                format: Format(s_format.format),
                size: (size.0, size.1, 1),
            })
            .collect();

        Ok(Swapchain {
            device: Rc::clone(self),
            native: native_swapchain,
            semaphore: Rc::new(create_binary_semaphore(&self.native)?),
            images,
            curr_image: Cell::new(None),
        })
    }

    pub fn create_submit_packet(
        self: &Rc<Self>,
        submit_infos: &[SubmitInfo],
    ) -> Result<SubmitPacket, vk::Result> {
        Ok(SubmitPacket {
            infos: submit_infos.to_vec(),
            fence: create_fence(&self.native, false)?,
            submitted: false,
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.allocator.destroy();
        unsafe {
            self.native.device_wait_idle().unwrap();
            self.native.destroy_device(None);
        }
    }
}
