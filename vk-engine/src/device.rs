use crate::adapter::Adapter;
use crate::{
    buffer::{Buffer, BufferUsageFlags, DeviceBuffer, HostBuffer},
    surface::Surface,
    swapchain::Swapchain,
    utils, Image, image::ImageUsageFlags,
};
use ash::version::DeviceV1_0;
use ash::vk;
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
    pub(crate) native: ash::Device,
    pub(crate) allocator: vk_mem::Allocator,
    pub(crate) swapchain_khr: ash::extensions::khr::Swapchain,
}

impl Device {
    pub fn is_extension_supported(&self, name: &str) -> bool {
        self.adapter.is_extension_enabled(name)
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

    pub fn create_image_2d(self: &Rc<Self>, usage: ImageUsageFlags, size: (u32, u32)) {
        
    }

    pub fn create_swapchain(
        self: &Rc<Self>,
        surface: &Rc<Surface>,
        size: (u32, u32),
        vsync: bool,
    ) -> Result<Rc<Swapchain>, DeviceError> {
        let surface_capabs = self.adapter.get_surface_capabilities(&surface)?;
        let surface_formats = self.adapter.get_surface_formats(&surface)?;
        let surface_present_modes = self.adapter.get_surface_present_modes(&surface)?;

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
            .map(|&native_image| Image { native: native_image })
            .collect();

        Ok(Rc::new(Swapchain {
            native: native_swapchain,
            device: Rc::clone(self),
            images,
        }))
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.allocator.destroy();
        unsafe { self.native.destroy_device(None) };
    }
}
