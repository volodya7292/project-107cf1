use crate::renderer::Renderer;
use crate::resource_file;
use crate::resource_file::ResourceFile;
use std::sync::{Arc, Mutex};
use vk_wrapper as vkw;

#[derive(Debug)]
pub enum Error {
    Load(resource_file::Error),
    Image(image::ImageError),
    Device(vkw::DeviceError),
}

impl From<resource_file::Error> for Error {
    fn from(err: resource_file::Error) -> Self {
        Error::Load(err)
    }
}

impl From<image::ImageError> for Error {
    fn from(err: image::ImageError) -> Self {
        Error::Image(err)
    }
}

impl From<vkw::DeviceError> for Error {
    fn from(err: vkw::DeviceError) -> Self {
        Error::Device(err)
    }
}

pub struct Texture {
    res_file: Arc<Mutex<ResourceFile>>,
    filename: String,
    image: Option<Arc<vkw::Image>>,
}

impl Texture {
    fn load(&self, gen_mipmaps: bool, max_anisotropy: f32) {}

    fn unload(&mut self) {
        self.image = None;
    }
}

pub fn new(
    res_file: &Arc<Mutex<ResourceFile>>,
    device: &vkw::Device,
    filename: &str,
    format: vkw::Format,
) -> Result<Texture, Error> {
    //let bytes = res_file.lock().unwrap().read(filename)?;
    //let img = image::load_from_memory(&bytes)?;
    //let image = device.create_image_2d(format, true, )

    Ok(Texture {
        res_file: Arc::clone(res_file),
        filename: filename.to_owned(),
        image: None,
    })
}
