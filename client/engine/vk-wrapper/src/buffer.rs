use std::ops::{Index, IndexMut};
use std::sync::{atomic, Arc};
use std::{marker::PhantomData, mem, ptr, slice};

use ash::vk;

use crate::{AccessFlags, Device, Queue};

#[derive(Copy, Clone)]
pub struct BufferHandle(pub vk::Buffer);

pub trait BufferHandleImpl {
    fn handle(&self) -> BufferHandle;
}

impl BufferHandleImpl for BufferHandle {
    fn handle(&self) -> BufferHandle {
        *self
    }
}

pub(crate) struct Buffer {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::Buffer,
    pub(crate) allocation: vma::VmaAllocation,
    pub(crate) used_dev_memory: u64,
    pub(crate) elem_size: u64,
    pub(crate) len: u64,
    pub(crate) _bytesize: u64,
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.wrapper.native.destroy_buffer(self.native, None);
            vma::vmaFreeMemory(self.device.allocator, self.allocation);
        }

        self.device
            .total_used_dev_memory
            .fetch_sub(self.used_dev_memory as usize, atomic::Ordering::Relaxed);
    }
}

unsafe impl Send for Buffer {}

unsafe impl Sync for Buffer {}

#[derive(Clone)]
#[repr(transparent)]
pub struct BufferBarrier(pub(crate) vk::BufferMemoryBarrier);

impl BufferBarrier {
    pub fn src_access_mask(mut self, src_access_mask: AccessFlags) -> Self {
        self.0.src_access_mask = src_access_mask.0;
        self
    }

    pub fn dst_access_mask(mut self, dst_access_mask: AccessFlags) -> Self {
        self.0.dst_access_mask = dst_access_mask.0;
        self
    }

    pub fn src_queue(mut self, src_queue: &Queue) -> Self {
        self.0.src_queue_family_index = src_queue.family_index;
        self
    }

    pub fn dst_queue(mut self, dst_queue: &Queue) -> Self {
        self.0.dst_queue_family_index = dst_queue.family_index;
        self
    }
}

pub struct HostBuffer<T: Copy> {
    pub(crate) _type_marker: PhantomData<T>,
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) p_data: *mut u8,
}

impl<T: Copy> Index<usize> for HostBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index as u64 >= self.buffer.len {
            panic!(
                "VkBuffer: index out of bounds: the len is {} but the index is {}",
                self.buffer.len, index
            );
        }

        unsafe { &*(self.p_data.add(self.buffer.elem_size as usize * index) as *const T) }
    }
}

impl<T: Copy> IndexMut<usize> for HostBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index as u64 >= self.buffer.len {
            panic!(
                "VkBuffer: index out of bounds: the len is {} but the index is {}",
                self.buffer.len, index
            );
        }
        unsafe { &mut *(self.p_data.add(self.buffer.elem_size as usize * index) as *mut T) }
    }
}

impl<T: Copy> HostBuffer<T> {
    pub fn size(&self) -> u64 {
        self.buffer.len
    }

    pub fn element_size(&self) -> u64 {
        mem::size_of::<T>() as u64
    }

    pub fn as_ptr(&self) -> *const T {
        self.p_data as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.p_data as *mut T
    }

    pub fn as_slice(&self) -> &[T] {
        assert_eq!(mem::size_of::<T>(), self.buffer.elem_size as usize);
        unsafe { slice::from_raw_parts(self.p_data as *const T, self.buffer.len as usize) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        assert_eq!(mem::size_of::<T>(), self.buffer.elem_size as usize);
        unsafe { slice::from_raw_parts_mut(self.p_data as *mut T, self.buffer.len as usize) }
    }

    pub fn read(&self, first_element: u64, elements: &mut [T]) {
        if first_element + elements.len() as u64 > self.buffer.len {
            panic!(
                "VkBuffer: index {} out of range for slice of length {}",
                first_element + elements.len() as u64,
                elements.len()
            );
        }

        if mem::size_of::<T>() == self.buffer.elem_size as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    (self.p_data as *const T).offset(first_element as isize),
                    elements.as_mut_ptr(),
                    elements.len(),
                )
            };
        } else {
            for i in 0..elements.len() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (self.p_data as *const u8)
                            .offset((first_element as isize + i as isize) * self.buffer.elem_size as isize),
                        (elements.as_mut_ptr()).offset(first_element as isize + i as isize) as *mut u8,
                        1,
                    );
                }
            }
        }
    }

    pub fn write(&mut self, first_element: u64, elements: &[T]) {
        if first_element + elements.len() as u64 > self.buffer.len {
            panic!(
                "VkBuffer: index {} out of range for slice of length {}",
                first_element + elements.len() as u64,
                elements.len()
            );
        }

        if mem::size_of::<T>() == self.buffer.elem_size as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    elements.as_ptr(),
                    (self.p_data as *mut T).offset(first_element as isize),
                    elements.len(),
                )
            };
        } else {
            for i in 0..elements.len() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (elements.as_ptr()).offset(first_element as isize + i as isize) as *mut u8,
                        self.p_data
                            .offset((first_element as isize + i as isize) * self.buffer.elem_size as isize),
                        1,
                    );
                }
            }
        }
    }

    pub fn barrier(&self) -> BufferBarrier {
        BufferBarrier(
            vk::BufferMemoryBarrier::builder()
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.buffer.native)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build(),
        )
    }
}

impl<T: Copy> BufferHandleImpl for HostBuffer<T> {
    fn handle(&self) -> BufferHandle {
        BufferHandle(self.buffer.native)
    }
}

unsafe impl<T: Copy> Send for HostBuffer<T> {}

unsafe impl<T: Copy> Sync for HostBuffer<T> {}

pub struct DeviceBuffer {
    pub(crate) buffer: Arc<Buffer>,
}

impl DeviceBuffer {
    pub fn ptr_eq(&self, other: &Self) -> bool {
        self.buffer.native == other.buffer.native
    }

    pub fn size(&self) -> u64 {
        self.buffer.len
    }

    pub fn element_size(&self) -> u64 {
        self.buffer.elem_size
    }

    pub fn barrier(&self) -> BufferBarrier {
        BufferBarrier(
            vk::BufferMemoryBarrier::builder()
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(self.buffer.native)
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build(),
        )
    }
}

unsafe impl Send for DeviceBuffer {}

unsafe impl Sync for DeviceBuffer {}

impl BufferHandleImpl for DeviceBuffer {
    fn handle(&self) -> BufferHandle {
        BufferHandle(self.buffer.native)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferUsageFlags(pub(crate) vk::BufferUsageFlags);
vk_bitflags_impl!(BufferUsageFlags, vk::BufferUsageFlags);

impl BufferUsageFlags {
    pub const TRANSFER_SRC: Self = Self(vk::BufferUsageFlags::TRANSFER_SRC);
    pub const TRANSFER_DST: Self = Self(vk::BufferUsageFlags::TRANSFER_DST);
    pub const VERTEX: Self = Self(vk::BufferUsageFlags::VERTEX_BUFFER);
    pub const INDEX: Self = Self(vk::BufferUsageFlags::INDEX_BUFFER);
    pub const UNIFORM: Self = Self(vk::BufferUsageFlags::UNIFORM_BUFFER);
    pub const INDIRECT: Self = Self(vk::BufferUsageFlags::INDIRECT_BUFFER);
    pub const STORAGE: Self = Self(vk::BufferUsageFlags::STORAGE_BUFFER);
}
