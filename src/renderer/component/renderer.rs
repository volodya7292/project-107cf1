use crate::renderer::material_pipeline::MaterialPipeline;
use smallvec::SmallVec;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;

pub struct BufferResource {
    pub(in crate::renderer) buffer: Vec<u8>,
    pub(in crate::renderer) device_buffer: Arc<vkw::DeviceBuffer>,
    pub(in crate::renderer) changed: bool,
}

impl BufferResource {
    pub fn set(&mut self, device: &Arc<vkw::Device>, buffer: Vec<u8>) -> Result<(), vkw::DeviceError> {
        self.buffer = buffer;

        let curr_size = self.device_buffer.size();
        let new_size = self.buffer.len() as u64;

        if (new_size > curr_size) || (new_size < curr_size / 2) {
            self.device_buffer = device.create_device_buffer(
                vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::STORAGE,
                new_size,
                1,
            )?;
        }

        self.changed = true;
        Ok(())
    }
}

pub enum Resource {
    Buffer(BufferResource),
}

impl Resource {
    pub fn buffer<T>(device: &Arc<vkw::Device>, buffer: &[T]) -> Result<Resource, vkw::DeviceError> {
        let size = buffer.len() * mem::size_of::<T>();

        let device_buffer = device.create_device_buffer(
            vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::STORAGE,
            size as u64,
            1,
        )?;
        let byte_slice = unsafe { slice::from_raw_parts(buffer.as_ptr() as *const u8, size) };

        Ok(Resource::Buffer(BufferResource {
            buffer: byte_slice.to_vec(),
            device_buffer,
            changed: true,
        }))
    }
}

pub struct Renderer {
    pub(in crate::renderer) mat_pipeline: Arc<MaterialPipeline>,

    pub(in crate::renderer) uniform_buffer: Arc<vkw::DeviceBuffer>,
    // binding id -> Resource
    pub(in crate::renderer) resources: SmallVec<[(u32, Resource); 4]>,
    pub(in crate::renderer) translucent: bool,
}

impl Renderer {
    pub fn new(
        device: &Arc<vkw::Device>,
        mat_pipeline: &Arc<MaterialPipeline>,
        translucent: bool,
    ) -> Renderer {
        Renderer {
            mat_pipeline: Arc::clone(&mat_pipeline),
            uniform_buffer: device
                .create_device_buffer(
                    vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::UNIFORM,
                    mat_pipeline.uniform_buffer_size() as u64,
                    1,
                )
                .unwrap(),
            resources: Default::default(),
            translucent,
        }
    }

    pub fn resources_mut(&mut self) -> &mut SmallVec<[(u32, Resource); 4]> {
        &mut self.resources
    }
}
