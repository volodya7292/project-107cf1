use crate::adapter::Adapter;
use crate::{
    buffer::{Buffer, BufferUsageFlags, DeviceBuffer, HostBuffer},
    utils, Instance,
};
use ash::version::DeviceV1_0;
use ash::vk;
use std::{marker::PhantomData, mem, ptr, rc::Rc};

#[derive(Debug)]
pub enum DeviceError {
    VmaError(vk_mem::Error),
    VkError(vk::Result),
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
    pub(crate) _instance: Rc<Instance>,
    pub(crate) adapter: Adapter,
    pub(crate) native: ash::Device,
    pub(crate) allocator: vk_mem::Allocator,
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
                _device: Rc::clone(self),
                _type_marker: PhantomData,
                native: unsafe { self.native.create_buffer(&buffer_info, None)? },
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

    pub fn destroy_host_buffer<T>(&self, host_buffer: HostBuffer<T>) -> Result<(), DeviceError> {
        Ok(self
            .allocator
            .destroy_buffer(host_buffer.buffer.native, &host_buffer.buffer.allocation)?)
    }

    pub fn destroy_device_buffer<T>(&self, device_buffer: DeviceBuffer<T>) -> Result<(), DeviceError> {
        Ok(self
            .allocator
            .destroy_buffer(device_buffer.buffer.native, &device_buffer.buffer.allocation)?)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.allocator.destroy();
        unsafe { self.native.destroy_device(None) };
    }
}
