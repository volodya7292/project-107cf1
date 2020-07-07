use crate::{DescriptorPool, DeviceBuffer, Image, ImageLayout};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::sync::Mutex;
use std::{slice, sync::Arc};

pub enum BindingRes {
    Buffer(Arc<DeviceBuffer>),
    /// [buffer, offset, range]
    BufferRange(Arc<DeviceBuffer>, u64, u64),
    Image(Arc<Image>, ImageLayout),
}

pub struct Binding {
    pub id: u32,
    pub array_index: u32,
    pub res: BindingRes,
}

pub struct PipelineInput {
    pub(crate) pool: Arc<DescriptorPool>,
    pub(crate) native: vk::DescriptorSet,

    // (u32, u32) : (binding id, array index)
    pub(crate) used_buffers: Mutex<HashMap<(u32, u32), Arc<DeviceBuffer>>>,
    pub(crate) used_images: Mutex<HashMap<(u32, u32), Arc<Image>>>,
}

impl PipelineInput {
    pub fn update(&self, updates: &[Binding]) {
        let mut native_buffer_infos = Vec::<vk::DescriptorBufferInfo>::with_capacity(updates.len());
        let mut native_image_infos = Vec::<vk::DescriptorImageInfo>::with_capacity(updates.len());
        let mut native_writes = Vec::<vk::WriteDescriptorSet>::with_capacity(updates.len());

        for binding in updates {
            let mut write_info = vk::WriteDescriptorSet::builder()
                .dst_set(self.native)
                .dst_binding(binding.id)
                .dst_array_element(binding.array_index)
                .descriptor_type(self.pool.signature.binding_types[&binding.id]);

            match &binding.res {
                BindingRes::Buffer(buffer) => {
                    native_buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: buffer.buffer.native,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    });
                    write_info = write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));

                    self.used_buffers
                        .lock()
                        .unwrap()
                        .insert((binding.id, binding.array_index), Arc::clone(buffer));
                }
                BindingRes::BufferRange(buffer, offset, range) => {
                    native_buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: buffer.buffer.native,
                        offset: *offset,
                        range: *range,
                    });
                    write_info = write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));

                    self.used_buffers
                        .lock()
                        .unwrap()
                        .insert((binding.id, binding.array_index), Arc::clone(buffer));
                }
                BindingRes::Image(image, layout) => {
                    native_image_infos.push(vk::DescriptorImageInfo {
                        sampler: image.sampler,
                        image_view: image.view,
                        image_layout: layout.0,
                    });
                    write_info = write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));

                    self.used_images
                        .lock()
                        .unwrap()
                        .insert((binding.id, binding.array_index), Arc::clone(image));
                }
            }

            native_writes.push(write_info.build());
        }

        unsafe {
            self.pool
                .device
                .wrapper
                .0
                .update_descriptor_sets(&native_writes, &[])
        };
    }
}

impl Drop for PipelineInput {
    fn drop(&mut self) {
        self.pool.free_sets.lock().unwrap().push(self.native);
    }
}
