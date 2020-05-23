use ash::vk;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineStageFlags(pub(crate) vk::PipelineStageFlags);
vk_bitflags_impl!(PipelineStageFlags, vk::PipelineStageFlags);

impl PipelineStageFlags {
    pub const TOP_OF_PIPE: Self = Self(vk::PipelineStageFlags::TOP_OF_PIPE);
    pub const TRANSFER: Self = Self(vk::PipelineStageFlags::TRANSFER);
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
    pub const PIXEL_SHADER: Self = Self(vk::PipelineStageFlags::FRAGMENT_SHADER);
    pub const BOTTOM_OF_PIPE: Self = Self(vk::PipelineStageFlags::BOTTOM_OF_PIPE);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AccessFlags(pub(crate) vk::AccessFlags);
vk_bitflags_impl!(AccessFlags, vk::AccessFlags);

impl AccessFlags {
    pub const COLOR_ATTACHMENT_READ: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_READ);
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    pub const TRANSFER_READ: Self = Self(vk::AccessFlags::TRANSFER_READ);
    pub const TRANSFER_WRITE: Self = Self(vk::AccessFlags::TRANSFER_WRITE);
    pub const SHADER_READ: Self = Self(vk::AccessFlags::SHADER_READ);
    pub const SHADER_WRITE: Self = Self(vk::AccessFlags::SHADER_WRITE);
}
