use crate::{BufferHandle, Device, Image, ImageLayout, ImageView, PipelineSignature, Sampler};
use ash::version::DeviceV1_0;
use ash::vk;
use bit_set::BitSet;
use smallvec::SmallVec;
use std::ops::Range;
use std::slice;
use std::sync::Arc;

pub enum BindingRes {
    Buffer(BufferHandle),
    BufferRange(BufferHandle, Range<u64>),
    Image(Arc<Image>, ImageLayout),
    ImageView(Arc<ImageView>, ImageLayout),
    ImageViewSampler(Arc<ImageView>, Arc<Sampler>, ImageLayout),
}

pub struct Binding {
    id: u32,
    ty: vk::DescriptorType,
    pub array_index: u32,
    pub res: BindingRes,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct BindingMapping {
    descriptor_set_id: u32,
    binding_id: u32,
    array_index: u32,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct DescriptorSet {
    pub(crate) native: vk::DescriptorSet,
    pub(crate) id: u32,
}

impl DescriptorSet {
    pub const NULL: DescriptorSet = DescriptorSet {
        native: vk::DescriptorSet::null(),
        id: u32::MAX,
    };
}

pub struct NativeDescriptorPool {
    pub(crate) handle: vk::DescriptorPool,
    pub(crate) size: u32,
}

pub struct DescriptorPool {
    pub(crate) device: Arc<Device>,
    pub(crate) signature: Arc<PipelineSignature>,
    pub(crate) set_layout_id: u32,
    pub(crate) native: SmallVec<[NativeDescriptorPool; 16]>,
    pub(crate) allocated: Vec<DescriptorSet>,
    pub(crate) free_sets: BitSet,
}

impl DescriptorPool {
    fn create_native_pool(
        &self,
        set_layout_id: u32,
        max_sets: u32,
    ) -> Result<Option<vk::DescriptorPool>, vk::Result> {
        let mut pool_sizes = self.signature.descriptor_sizes[set_layout_id as usize].clone();
        for mut pool_size in &mut pool_sizes {
            pool_size.descriptor_count *= max_sets;
        }
        if pool_sizes.is_empty() {
            return Ok(None);
        }

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        Ok(Some(unsafe {
            self.device
                .wrapper
                .native
                .create_descriptor_pool(&pool_info, None)?
        }))
    }

    pub(crate) fn alloc_next_pool(&mut self, size: u32) -> Result<(), vk::Result> {
        let new_pool = self.create_native_pool(self.set_layout_id, size)?;
        if new_pool.is_none() {
            return Ok(());
        }

        let next_pool = NativeDescriptorPool {
            handle: new_pool.unwrap(),
            size,
        };
        let layouts = vec![self.signature.native[self.set_layout_id as usize]; size as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(next_pool.handle)
            .set_layouts(&layouts);
        let sets = unsafe { self.device.wrapper.native.allocate_descriptor_sets(&alloc_info)? };

        let start_i = self.allocated.len() as u32;
        self.allocated
            .extend(sets.into_iter().enumerate().map(|(i, v)| DescriptorSet {
                native: v,
                id: start_i + i as u32,
            }));
        self.free_sets
            .extend((start_i as usize..(start_i + size) as usize).into_iter());
        self.native.push(next_pool);
        Ok(())
    }

    /// Allocate a new descriptor from the pool.
    /// If the pool doesn't have layout (its descriptor sizes = 0), this returns `DescriptorSet::NULL`.
    pub fn alloc(&mut self) -> Result<DescriptorSet, vk::Result> {
        if self.native.is_empty() {
            return Ok(DescriptorSet::NULL);
        }

        if let Some(id) = self.free_sets.iter().next() {
            self.free_sets.remove(id);
            Ok(self.allocated[id])
        } else {
            self.alloc_next_pool((self.native.last().unwrap().size + 1).next_power_of_two())?;
            self.alloc()
        }
    }

    pub fn free(&mut self, descriptor_set: DescriptorSet) {
        if descriptor_set != DescriptorSet::NULL {
            self.free_sets.insert(descriptor_set.id as usize);
        }
    }

    pub fn create_binding(&self, id: u32, array_index: u32, res: BindingRes) -> Binding {
        let ty = self.signature.binding_types[self.set_layout_id as usize][&id];
        Binding {
            id,
            array_index,
            ty,
            res,
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        for native in &self.native {
            unsafe {
                self.device
                    .wrapper
                    .native
                    .destroy_descriptor_pool(native.handle, None)
            };
        }
    }
}

impl Device {
    /// Updates descriptor sets with new bindings of respective ranges of `bindings`.
    ///
    /// # Safety
    ///
    /// - Access to `DescriptorSet`s must be synchronized.
    /// - Buffers and images in `bindings` must be valid.
    pub unsafe fn update_descriptor_sets(
        &self,
        bindings: &[Binding],
        updates: &[(DescriptorSet, Range<usize>)],
    ) {
        if updates.is_empty() {
            return;
        }

        // Safety: calculate total buffers size to prevent reallocating them
        let buf_len = updates.iter().fold(0, |c, (_, r)| c + r.len());

        let mut native_buffer_infos = Vec::with_capacity(buf_len);
        let mut native_image_infos = Vec::with_capacity(buf_len);
        let mut native_writes = Vec::with_capacity(buf_len);

        for (descriptor_set, range) in updates {
            if *descriptor_set == DescriptorSet::NULL {
                panic!("descriptor_set = NULL");
            }

            for i in range.clone() {
                let binding = &bindings[i];

                let mut write_info = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set.native)
                    .dst_binding(binding.id)
                    .dst_array_element(binding.array_index)
                    .descriptor_type(binding.ty);

                match &binding.res {
                    BindingRes::Buffer(handle) => {
                        native_buffer_infos.push(vk::DescriptorBufferInfo {
                            buffer: handle.0,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        });
                        write_info =
                            write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));
                    }
                    BindingRes::BufferRange(handle, range) => {
                        native_buffer_infos.push(vk::DescriptorBufferInfo {
                            buffer: handle.0,
                            offset: range.start,
                            range: range.end - range.start,
                        });
                        write_info =
                            write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));
                    }
                    BindingRes::Image(image, layout) => {
                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler: image.sampler.native,
                            image_view: image.view.native,
                            image_layout: layout.0,
                        });
                        write_info =
                            write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                    }
                    BindingRes::ImageView(image_view, layout) => {
                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler: Default::default(),
                            image_view: image_view.native,
                            image_layout: layout.0,
                        });
                        write_info =
                            write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                    }
                    BindingRes::ImageViewSampler(image_view, sampler, layout) => {
                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler: sampler.native,
                            image_view: image_view.native,
                            image_layout: layout.0,
                        });
                        write_info =
                            write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                    }
                }

                native_writes.push(write_info.build());
            }
        }

        self.wrapper.native.update_descriptor_sets(&native_writes, &[]);
    }

    pub unsafe fn update_descriptor_set(&self, set: DescriptorSet, bindings: &[Binding]) {
        self.update_descriptor_sets(bindings, &[(set, 0..bindings.len())])
    }
}
