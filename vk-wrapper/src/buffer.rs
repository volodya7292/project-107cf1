use crate::adapter::MemoryBlock;
use crate::{AccessFlags, Device, Queue};
use ash::vk;
use gpu_alloc_ash::AshMemoryDevice;
use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;
use std::ops::{Index, IndexMut};
use std::sync::{atomic, Arc};
use std::{marker::PhantomData, mem, ptr};

#[derive(Copy, Clone)]
pub struct BufferHandle(pub(crate) vk::Buffer);

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
    pub(crate) allocation: ManuallyDrop<MemoryBlock>,
    pub(crate) used_dev_memory: u64,
    pub(crate) elem_size: u64,
    pub(crate) aligned_elem_size: u64,
    pub(crate) size: u64,
    pub(crate) bytesize: u64,
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.native.hash(state);
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.wrapper.native.destroy_buffer(self.native, None);

            self.device.allocator.lock().dealloc(
                AshMemoryDevice::wrap(&self.device.wrapper.native),
                ManuallyDrop::take(&mut self.allocation),
            )
        }

        self.device
            .total_used_dev_memory
            .fetch_sub(self.used_dev_memory as usize, atomic::Ordering::Relaxed);
    }
}

#[derive(Clone)]
pub struct RawHostBuffer(pub(crate) Arc<Buffer>);

impl RawHostBuffer {
    pub fn device(&self) -> &Arc<Device> {
        &self.0.device
    }

    pub fn element_size(&self) -> u64 {
        self.0.elem_size
    }

    pub fn size(&self) -> u64 {
        self.0.size
    }
}

unsafe impl Send for RawHostBuffer {}

unsafe impl Sync for RawHostBuffer {}

#[derive(Clone, Copy)]
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

    pub fn offset(mut self, offset: u64) -> Self {
        self.0.offset = offset;
        self
    }

    pub fn size(mut self, size: u64) -> Self {
        self.0.size = size;
        self
    }
}

unsafe impl Send for BufferBarrier {}
unsafe impl Sync for BufferBarrier {}

pub struct HostBuffer<T> {
    pub(crate) _type_marker: PhantomData<T>,
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) p_data: *mut u8,
}

impl<'a, T> IntoIterator for &'a HostBuffer<T> {
    type Item = &'a mut T;
    type IntoIter = HostBufferIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        HostBufferIterator {
            p_data: self.p_data,
            stride: self.buffer.aligned_elem_size as usize,
            size: self.buffer.size as usize,
            _marker: PhantomData,
            index: 0,
        }
    }
}

pub struct HostBufferIterator<'a, T> {
    p_data: *mut u8,
    stride: usize,
    size: usize,
    index: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Iterator for HostBufferIterator<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.size {
            let curr_index = self.index;
            self.index += 1;
            Some(unsafe { &mut *(self.p_data.offset((self.stride * curr_index) as isize) as *mut T) })
        } else {
            None
        }
    }
}

impl<T> Index<usize> for HostBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index as u64 >= self.buffer.size {
            panic!(
                "VkBuffer: index out of bounds: the len is {} but the index is {}",
                self.buffer.size, index
            );
        }

        unsafe {
            &*(self
                .p_data
                .offset(self.buffer.aligned_elem_size as isize * index as isize) as *const T)
        }
    }
}

impl<T> IndexMut<usize> for HostBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index as u64 >= self.buffer.size {
            panic!(
                "VkBuffer: index out of bounds: the len is {} but the index is {}",
                self.buffer.size, index
            );
        }
        unsafe {
            &mut *(self
                .p_data
                .offset(self.buffer.aligned_elem_size as isize * index as isize) as *mut T)
        }
    }
}

impl<T> HostBuffer<T> {
    pub fn raw(&self) -> RawHostBuffer {
        RawHostBuffer(Arc::clone(&self.buffer))
    }

    pub fn size(&self) -> u64 {
        self.buffer.size
    }

    pub fn as_ptr(&self) -> *const T {
        self.p_data as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.p_data as *mut T
    }

    pub fn read(&self, first_element: u64, elements: &mut [T]) {
        if first_element + elements.len() as u64 > self.buffer.size {
            panic!(
                "VkBuffer: index {} out of range for slice of length {}",
                first_element + elements.len() as u64,
                elements.len()
            );
        }

        if mem::size_of::<T>() == self.buffer.aligned_elem_size as usize {
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
                        (self.p_data as *const u8).offset(
                            (first_element as isize + i as isize) * self.buffer.aligned_elem_size as isize,
                        ),
                        (elements.as_mut_ptr()).offset(first_element as isize + i as isize) as *mut u8,
                        1,
                    );
                }
            }
        }
    }

    pub fn write(&mut self, first_element: u64, elements: &[T]) {
        if first_element + elements.len() as u64 > self.buffer.size {
            panic!(
                "VkBuffer: index {} out of range for slice of length {}",
                first_element + elements.len() as u64,
                elements.len()
            );
        }

        if mem::size_of::<T>() == self.buffer.aligned_elem_size as usize {
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
                        (self.p_data as *mut u8).offset(
                            (first_element as isize + i as isize) * self.buffer.aligned_elem_size as isize,
                        ),
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

impl<T> BufferHandleImpl for HostBuffer<T> {
    fn handle(&self) -> BufferHandle {
        BufferHandle(self.buffer.native)
    }
}

unsafe impl<T> Send for HostBuffer<T> {}

unsafe impl<T> Sync for HostBuffer<T> {}

pub struct DeviceBuffer {
    pub(crate) buffer: Arc<Buffer>,
}

impl DeviceBuffer {
    pub fn size(&self) -> u64 {
        self.buffer.size
    }

    pub fn aligned_element_size(&self) -> u64 {
        self.buffer.aligned_elem_size
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
