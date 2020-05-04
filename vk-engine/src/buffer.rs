use ash::vk;
use std::{marker::PhantomData, mem, ptr};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferUsageFlags(pub(crate) vk::BufferUsageFlags);
vk_bitflags_impl!(BufferUsageFlags, vk::BufferUsageFlags);

pub(crate) struct Buffer<T: ?Sized> {
    pub(crate) _type_marker: PhantomData<T>,
    pub(crate) native: vk::Buffer,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) aligned_elem_size: u64,
    pub(crate) bytesize: u64,
}

pub struct HostBuffer<T> {
    pub(crate) buffer: Buffer<T>,
    pub(crate) p_data: *mut u8,
}

impl<T> HostBuffer<T> {
    fn read(&self, first_element: u64, elements: &mut [T]) {
        let read_count = (elements.len() as isize)
            .min(
                self.buffer.bytesize as isize / self.buffer.aligned_elem_size as isize
                    - first_element as isize,
            )
            .max(0);

        if read_count == 0 {
            return;
        }

        if mem::size_of::<T>() == self.buffer.aligned_elem_size as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    (self.p_data as *const T).offset(first_element as isize),
                    elements.as_mut_ptr(),
                    read_count as usize,
                )
            };
        } else {
            for i in 0..read_count {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (self.p_data as *const u8)
                            .offset((first_element as isize + i) * self.buffer.aligned_elem_size as isize),
                        (elements.as_mut_ptr()).offset(first_element as isize + i) as *mut u8,
                        1,
                    );
                }
            }
        }
    }

    fn write(&self, first_element: u64, elements: &[T]) {
        // TODO: range check
        // TODO: consider element alignment

        unsafe {
            ptr::copy_nonoverlapping(
                elements.as_ptr(),
                (self.p_data as *mut T).offset(first_element as isize),
                elements.len(),
            )
        };
    }
}

pub struct DeviceBuffer<T> {
    pub(crate) buffer: Buffer<T>,
}

impl BufferUsageFlags {
    pub const TRANSFER_SRC: Self = Self(vk::BufferUsageFlags::TRANSFER_SRC);
    pub const TRANSFER_DST: Self = Self(vk::BufferUsageFlags::TRANSFER_DST);
    pub const VERTEX: Self = Self(vk::BufferUsageFlags::VERTEX_BUFFER);
    pub const INDEX: Self = Self(vk::BufferUsageFlags::INDEX_BUFFER);
    pub const UNIFORM: Self = Self(vk::BufferUsageFlags::UNIFORM_BUFFER);
    pub const INDIRECT: Self = Self(vk::BufferUsageFlags::INDIRECT_BUFFER);
    pub const STORAGE: Self = Self(vk::BufferUsageFlags::STORAGE_BUFFER);
}
