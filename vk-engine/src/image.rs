use ash::vk;

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

pub struct Image {
    pub(crate) native: vk::Image,
}