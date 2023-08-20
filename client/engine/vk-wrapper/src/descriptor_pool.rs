use crate::sampler::SamplerFilter;
use crate::{BufferHandle, Device, Image, ImageLayout, ImageView, PipelineSignature, Sampler, SamplerMipmap};
use ash::vk;
use ash::vk::Handle;
use common::alloc::index_allocator::IndexAllocator;
use smallvec::SmallVec;
use std::ops::Range;
use std::slice;
use std::sync::Arc;

pub enum BindingRes {
    Buffer(BufferHandle),
    BufferRange(BufferHandle, Range<u64>),
    Image(Arc<Image>, Option<Arc<Sampler>>, ImageLayout),
    ImageView(Arc<ImageView>, Option<Arc<Sampler>>, ImageLayout),
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

impl Default for DescriptorSet {
    fn default() -> Self {
        Self::NULL
    }
}

pub struct NativeDescriptorPool {
    pub(crate) handle: vk::DescriptorPool,
    pub(crate) size: u32,
}

pub struct DescriptorPool {
    pub(crate) device: Arc<Device>,
    pub(crate) signature: Arc<PipelineSignature>,
    pub(crate) name: String,
    pub(crate) set_layout_id: u32,
    pub(crate) native: SmallVec<[NativeDescriptorPool; 16]>,
    pub(crate) all_sets: Vec<vk::DescriptorSet>,
    pub(crate) allocator: IndexAllocator,
}

impl DescriptorPool {
    fn create_native_pool(
        &self,
        set_layout_id: u32,
        max_sets: u32,
        name: &str,
    ) -> Result<vk::DescriptorPool, vk::Result> {
        let mut pool_sizes = self.signature.descriptor_sizes[set_layout_id as usize].clone();
        for pool_size in &mut pool_sizes {
            pool_size.descriptor_count *= max_sets;
        }

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_sets)
            .pool_sizes(&pool_sizes);

        let native_pool = unsafe {
            self.device
                .wrapper
                .native
                .create_descriptor_pool(&pool_info, None)?
        };

        unsafe {
            self.device.wrapper.debug_set_object_name(
                vk::ObjectType::DESCRIPTOR_POOL,
                native_pool.as_raw(),
                name,
            )?;
        }

        Ok(native_pool)
    }

    pub(crate) fn alloc_next_pool(&mut self, size: u32) -> Result<(), vk::Result> {
        let new_pool_id = self.native.len();
        let new_pool = self.create_native_pool(
            self.set_layout_id,
            size,
            &format!("{}-{}", &self.name, new_pool_id),
        )?;

        let next_pool = NativeDescriptorPool {
            handle: new_pool,
            size,
        };
        let layouts = vec![self.signature.native[self.set_layout_id as usize]; size as usize];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(next_pool.handle)
            .set_layouts(&layouts);

        let sets = unsafe { self.device.wrapper.native.allocate_descriptor_sets(&alloc_info)? };
        self.all_sets.extend(sets);

        self.native.push(next_pool);
        Ok(())
    }

    /// Allocate a new descriptor from the pool.
    /// If the pool doesn't have layout (its descriptor sizes = 0), this returns `DescriptorSet::NULL`.
    pub fn alloc(&mut self) -> Result<DescriptorSet, vk::Result> {
        let id = self.allocator.alloc();

        if id >= self.all_sets.len() {
            self.alloc_next_pool((self.native.last().unwrap().size + 1).next_power_of_two())?;
        }

        let native = self.all_sets[id];

        Ok(DescriptorSet {
            native,
            id: id as u32,
        })
    }

    pub fn free(&mut self, descriptor_set: DescriptorSet) {
        assert_ne!(descriptor_set, DescriptorSet::NULL);
        self.allocator.free(descriptor_set.id as usize);
    }

    pub fn reset(&mut self) {
        self.allocator.clear();
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

            for binding in &bindings[range.clone()] {
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

                    BindingRes::Image(image, sampler, layout) => {
                        let adapter = self.adapter();
                        let sampler = sampler
                            .as_ref()
                            .map(|sampler| {
                                if (sampler.mag_filter == SamplerFilter::LINEAR
                                    && !adapter.is_linear_filter_supported(
                                        image.wrapper.format.0,
                                        image.wrapper.tiling,
                                    ))
                                    || (sampler.min_filter == SamplerFilter::LINEAR
                                        && !adapter.is_linear_filter_supported(
                                            image.wrapper.format.0,
                                            image.wrapper.tiling,
                                        ))
                                    || (sampler.mipmap == SamplerMipmap::LINEAR
                                        && !adapter.is_linear_filter_supported(
                                            image.wrapper.format.0,
                                            image.wrapper.tiling,
                                        ))
                                {
                                    // The sampler is not supported, do not use sampler
                                    self.default_sampler.native
                                } else {
                                    sampler.native
                                }
                            })
                            .unwrap_or(self.default_sampler.native);

                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler,
                            image_view: image.view.native,
                            image_layout: layout.0,
                        });
                        write_info =
                            write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                    }
                    BindingRes::ImageView(image_view, sampler, layout) => {
                        let adapter = self.adapter();
                        let sampler = sampler
                            .as_ref()
                            .map(|sampler| {
                                if (sampler.mag_filter == SamplerFilter::LINEAR
                                    && !adapter.is_linear_filter_supported(
                                        image_view.image_wrapper.format.0,
                                        image_view.image_wrapper.tiling,
                                    ))
                                    || (sampler.min_filter == SamplerFilter::LINEAR
                                        && !adapter.is_linear_filter_supported(
                                            image_view.image_wrapper.format.0,
                                            image_view.image_wrapper.tiling,
                                        ))
                                    || (sampler.mipmap == SamplerMipmap::LINEAR
                                        && !adapter.is_linear_filter_supported(
                                            image_view.image_wrapper.format.0,
                                            image_view.image_wrapper.tiling,
                                        ))
                                {
                                    // The sampler is not supported, do not use sampler
                                    self.default_sampler.native
                                } else {
                                    sampler.native
                                }
                            })
                            .unwrap_or(self.default_sampler.native);

                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler,
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

    /// # Safety
    /// See [update_descriptor_sets].
    pub unsafe fn update_descriptor_set(&self, set: DescriptorSet, bindings: &[Binding]) {
        self.update_descriptor_sets(bindings, &[(set, 0..bindings.len())])
    }
}
