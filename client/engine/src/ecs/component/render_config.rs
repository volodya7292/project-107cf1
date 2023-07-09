use crate::module::main_renderer;
use smallvec::SmallVec;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::image::ImageParams;
use vk_wrapper::shader::BindingId;

pub enum Resource {
    Buffer {
        new_source_data: Option<Vec<u8>>,
        buffer: vkw::DeviceBuffer,
    },
    Image {
        new_source_data: Option<Vec<u8>>,
        image: Arc<vkw::Image>,
    },
    None,
}

impl Default for Resource {
    fn default() -> Self {
        Self::None
    }
}

impl Resource {
    pub fn image(
        device: &Arc<vkw::Device>,
        params: ImageParams,
        source_data: Vec<u8>,
    ) -> Result<Resource, vkw::DeviceError> {
        let image = device.create_image(&params.add_usage(vkw::ImageUsageFlags::TRANSFER_DST), "")?;
        let byte_slice = unsafe { slice::from_raw_parts(source_data.as_ptr(), source_data.len()) };

        Ok(Resource::Image {
            new_source_data: Some(byte_slice.to_vec()),
            image,
        })
    }

    pub fn buffer<T: Copy>(
        device: &Arc<vkw::Device>,
        source_data: &[T],
    ) -> Result<Resource, vkw::DeviceError> {
        let size = source_data.len() * mem::size_of::<T>();

        let device_buffer = device.create_device_buffer(
            vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::STORAGE,
            size as u64,
            1,
        )?;
        let byte_slice = unsafe { slice::from_raw_parts(source_data.as_ptr() as *const u8, size) };

        Ok(Resource::Buffer {
            new_source_data: Some(byte_slice.to_vec()),
            buffer: device_buffer,
        })
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RenderLayer {
    /// Renders regular 3D objects.
    Main,
    /// Renders objects after the `MAIN` stage.
    Overlay,
}

pub struct MeshRenderConfigC {
    pub(crate) render_layer: RenderLayer,
    pub(crate) mat_pipeline: u32,
    pub(crate) resources: SmallVec<[(BindingId, Resource); 4]>,
    pub(crate) translucent: bool,
}

impl Default for MeshRenderConfigC {
    fn default() -> Self {
        Self {
            render_layer: RenderLayer::Main,
            mat_pipeline: u32::MAX,
            resources: Default::default(),
            translucent: false,
        }
    }
}

impl MeshRenderConfigC {
    pub fn new(mat_pipeline: u32, translucent: bool) -> MeshRenderConfigC {
        MeshRenderConfigC {
            render_layer: RenderLayer::Main,
            mat_pipeline,
            resources: Default::default(),
            translucent,
        }
    }

    pub fn with_render_layer(mut self, layer: RenderLayer) -> Self {
        self.render_layer = layer;
        self
    }

    pub fn with_shader_resources(mut self, resources: SmallVec<[Resource; 4]>) -> Self {
        self.resources = resources
            .into_iter()
            .enumerate()
            .map(|(idx, res)| {
                let binding_id = main_renderer::CUSTOM_OBJ_BINDING_START_ID + idx as u32;
                (binding_id, res)
            })
            .collect();
        self
    }

    pub fn set_shader_resource(&mut self, idx: usize, res: Resource) {
        self.resources[idx].1 = res;
    }
}
