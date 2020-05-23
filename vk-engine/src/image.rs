use crate::{format, AccessFlags, Device, Format, Queue};
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
pub struct ImageLayout(pub(crate) vk::ImageLayout);

impl ImageLayout {
    pub const UNDEFINED: Self = Self(vk::ImageLayout::UNDEFINED);
    pub const COLOR_ATTACHMENT: Self = Self(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    pub const DEPTH_STENCIL_ATTACHMENT: Self = Self(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    pub const DEPTH_READ: Self = Self(vk::ImageLayout::DEPTH_READ_ONLY_OPTIMAL);
    pub const TRANSFER_SRC: Self = Self(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    pub const TRANSFER_DST: Self = Self(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    pub const SHADER_READ: Self = Self(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    pub const PRESENT: Self = Self(vk::ImageLayout::PRESENT_SRC_KHR);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageType(pub(crate) vk::ImageType);

#[derive(Clone)]
pub struct ImageBarrier(pub(crate) vk::ImageMemoryBarrier);

pub struct Image {
    pub(crate) device: Rc<Device>,
    pub(crate) native: vk::Image,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) owned_handle: bool,
    pub(crate) format: Format,
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

    pub fn barrier_queue_level(
        &self,
        src_access_mask: AccessFlags,
        dst_access_mask: AccessFlags,
        old_layout: ImageLayout,
        new_layout: ImageLayout,
        src_queue: &Queue,
        dst_queue: &Queue,
        base_mip_level: u32,
        level_count: u32,
    ) -> ImageBarrier {
        let aspect = if self.format == format::DEPTH_FORMAT {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        ImageBarrier(
            vk::ImageMemoryBarrier::builder()
                .src_access_mask(src_access_mask.0)
                .dst_access_mask(dst_access_mask.0)
                .old_layout(old_layout.0)
                .new_layout(new_layout.0)
                .src_queue_family_index(src_queue.family_index)
                .dst_queue_family_index(dst_queue.family_index)
                .image(self.native)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(aspect)
                        .base_mip_level(base_mip_level)
                        .level_count(level_count)
                        .base_array_layer(0)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS)
                        .build(),
                )
                .build(),
        )
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
