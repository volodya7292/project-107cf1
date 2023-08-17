use crate::module::main_renderer;
use common::parking_lot::Mutex;
use common::resource_cache::CachedResource;
use smallvec::SmallVec;
use std::any::Any;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::image::ImageParams;
use vk_wrapper::shader::BindingId;

pub struct GPUImageResource {
    pub(crate) new_source_data: Mutex<Option<Vec<u8>>>,
    pub(crate) image: Arc<vkw::Image>,
}

impl GPUImageResource {
    pub fn new(
        device: &Arc<vkw::Device>,
        params: ImageParams,
        source_data: Vec<u8>,
    ) -> Result<Arc<Self>, vkw::DeviceError> {
        let image = device.create_image(&params.add_usage(vkw::ImageUsageFlags::TRANSFER_DST), "")?;
        let byte_slice = unsafe { slice::from_raw_parts(source_data.as_ptr(), source_data.len()) };

        Ok(Arc::new(GPUImageResource {
            new_source_data: Mutex::new(Some(byte_slice.to_vec())),
            image,
        }))
    }

    pub fn size(&self) -> (u32, u32) {
        self.image.size_2d()
    }
}

impl CachedResource for GPUImageResource {
    fn footprint(&self) -> usize {
        self.image.bytesize() as usize
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub enum GPUResource {
    Buffer {
        new_source_data: Option<Vec<u8>>,
        buffer: vkw::DeviceBuffer,
    },
    Image(Arc<GPUImageResource>),
    None,
}

impl Default for GPUResource {
    fn default() -> Self {
        Self::None
    }
}

impl GPUResource {
    pub fn image(
        device: &Arc<vkw::Device>,
        params: ImageParams,
        source_data: Vec<u8>,
    ) -> Result<GPUResource, vkw::DeviceError> {
        Ok(GPUResource::Image(GPUImageResource::new(
            device,
            params,
            source_data,
        )?))
    }

    pub fn buffer<T: Copy>(
        device: &Arc<vkw::Device>,
        source_data: &[T],
    ) -> Result<GPUResource, vkw::DeviceError> {
        let size = source_data.len() * mem::size_of::<T>();

        let device_buffer = device.create_device_buffer(
            vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::STORAGE,
            size as u64,
            1,
        )?;
        let byte_slice = unsafe { slice::from_raw_parts(source_data.as_ptr() as *const u8, size) };

        Ok(GPUResource::Buffer {
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
    pub(crate) resources: SmallVec<[(BindingId, GPUResource); 4]>,
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

    pub fn with_shader_resources(mut self, resources: SmallVec<[GPUResource; 4]>) -> Self {
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

    pub fn set_shader_resource(&mut self, idx: usize, res: GPUResource) {
        self.resources[idx].1 = res;
    }
}
