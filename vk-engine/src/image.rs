use crate::Device;
use ash::vk;
use std::rc::Rc;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageUsageFlags(pub(crate) vk::ImageUsageFlags);
vk_bitflags_impl!(ImageUsageFlags, vk::ImageUsageFlags);

impl ImageUsageFlags {
    pub const TRANSFER_SRC: Self = Self(vk::ImageUsageFlags::TRANSFER_SRC);
    pub const TRANSFER_DST: Self = Self(vk::ImageUsageFlags::TRANSFER_DST);
    pub const SAMPLED: Self = Self(vk::ImageUsageFlags::SAMPLED);
    pub const STORAGE: Self = Self(vk::ImageUsageFlags::STORAGE);
    pub const INPUT_ATTACHMENT: Self = Self(vk::ImageUsageFlags::INPUT_ATTACHMENT);
    pub const COLOR_ATTACHMENT: Self = Self(vk::ImageUsageFlags::COLOR_ATTACHMENT);
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
}
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageType(pub(crate) vk::ImageType);

pub struct Image {
    pub(crate) device: Rc<Device>,
    pub(crate) native: vk::Image,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) owned_handle: bool,
    pub(crate) size: (u32, u32, u32),
}

impl Image {
    pub const TYPE_2D: ImageType = ImageType(vk::ImageType::TYPE_2D);
    pub const TYPE_3D: ImageType = ImageType(vk::ImageType::TYPE_3D);

    pub fn get_size_2d(&self) -> (u32, u32) {
        (self.size.0, self.size.1)
    }

    pub fn get_size(&self) -> (u32, u32, u32) {
        self.size
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        if self.owned_handle {
            self.device
                .allocator
                .destroy_image(self.native, &self.allocation)
                .unwrap();
        }
    }
}
