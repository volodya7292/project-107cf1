use crate::swapchain::SwapchainWrapper;
use crate::{AccessFlags, Device, DeviceError, Format, ImageView, Queue};
use ash::vk;
use ash::vk::Handle;
use std::sync::{atomic, Arc};

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
    pub const DEPTH_STENCIL_READ: Self = Self(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL);
    pub const TRANSFER_SRC: Self = Self(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
    pub const TRANSFER_DST: Self = Self(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    pub const SHADER_READ: Self = Self(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    pub const GENERAL: Self = Self(vk::ImageLayout::GENERAL);
    pub const PRESENT: Self = Self(vk::ImageLayout::PRESENT_SRC_KHR);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImageType(pub(crate) vk::ImageType);

#[derive(Clone)]
pub struct ImageBarrier {
    pub(crate) native: vk::ImageMemoryBarrier,
}

impl ImageBarrier {
    pub fn src_access_mask(mut self, src_access_mask: AccessFlags) -> Self {
        self.native.src_access_mask = src_access_mask.0;
        self
    }

    pub fn dst_access_mask(mut self, dst_access_mask: AccessFlags) -> Self {
        self.native.dst_access_mask = dst_access_mask.0;
        self
    }

    pub fn old_layout(mut self, old_layout: ImageLayout) -> Self {
        self.native.old_layout = old_layout.0;
        self
    }

    pub fn new_layout(mut self, new_layout: ImageLayout) -> Self {
        self.native.new_layout = new_layout.0;
        self
    }

    pub fn layout(self, layout: ImageLayout) -> Self {
        self.old_layout(layout).new_layout(layout)
    }

    pub fn base_mip_level(mut self, base_mip_level: u32) -> Self {
        self.native.subresource_range.base_mip_level = base_mip_level;
        self
    }

    pub fn level_count(mut self, level_count: u32) -> Self {
        self.native.subresource_range.level_count = level_count;
        self
    }

    pub fn mip_levels(self, base_mip_level: u32, level_count: u32) -> Self {
        self.base_mip_level(base_mip_level).level_count(level_count)
    }

    pub fn src_queue(mut self, src_queue: &Queue) -> Self {
        self.native.src_queue_family_index = src_queue.family_index;
        self
    }

    pub fn dst_queue(mut self, dst_queue: &Queue) -> Self {
        self.native.dst_queue_family_index = dst_queue.family_index;
        self
    }
}

pub(crate) struct ImageWrapper {
    pub(crate) device: Arc<Device>,
    pub(crate) _swapchain_wrapper: Option<Arc<SwapchainWrapper>>,
    pub(crate) native: vk::Image,
    pub(crate) allocation: Option<vma::VmaAllocation>,
    pub(crate) bytesize: u64,
    pub(crate) is_array: bool,
    pub(crate) owned_handle: bool,
    pub(crate) ty: ImageType,
    pub(crate) format: Format,
    pub(crate) aspect: vk::ImageAspectFlags,
    pub(crate) tiling: vk::ImageTiling,
    pub(crate) name: String,
}

impl ImageWrapper {
    pub fn create_view(self: &Arc<Self>, name: &str) -> ImageViewBuilder {
        let ty = match self.ty {
            ImageType(vk::ImageType::TYPE_2D) => {
                if self.is_array {
                    vk::ImageViewType::TYPE_2D_ARRAY
                } else {
                    vk::ImageViewType::TYPE_2D
                }
            }
            ImageType(vk::ImageType::TYPE_3D) => vk::ImageViewType::TYPE_3D,
            _ => {
                unreachable!()
            }
        };
        let name = format!("{}-{}", self.name, name);

        ImageViewBuilder {
            image_wrapper: Arc::clone(self),
            info: vk::ImageViewCreateInfo::builder()
                .image(self.native)
                .view_type(ty)
                .format(self.format.0)
                .components(vk::ComponentMapping {
                    r: vk::ComponentSwizzle::IDENTITY,
                    g: vk::ComponentSwizzle::IDENTITY,
                    b: vk::ComponentSwizzle::IDENTITY,
                    a: vk::ComponentSwizzle::IDENTITY,
                })
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: self.aspect,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                })
                .build(),
            name,
        }
    }
}

impl Drop for ImageWrapper {
    fn drop(&mut self) {
        if !self.owned_handle {
            return;
        }
        unsafe {
            self.device.wrapper.native.destroy_image(self.native, None);

            if let Some(allocation) = self.allocation.take() {
                vma::vmaFreeMemory(self.device.allocator, allocation);
            }

            self.device
                .total_used_dev_memory
                .fetch_sub(self.bytesize as usize, atomic::Ordering::Relaxed);
        }
    }
}

unsafe impl Send for ImageWrapper {}

unsafe impl Sync for ImageWrapper {}

pub struct Image {
    pub(crate) wrapper: Arc<ImageWrapper>,
    pub(crate) view: Arc<ImageView>,
    pub(crate) size: (u32, u32, u32),
    pub(crate) mip_levels: u32,
}

pub struct ImageViewBuilder {
    image_wrapper: Arc<ImageWrapper>,
    info: vk::ImageViewCreateInfo,
    name: String,
}

impl ImageViewBuilder {
    pub fn base_mip_level(mut self, level: u32) -> Self {
        self.info.subresource_range.base_mip_level = level;
        self
    }

    pub fn mip_level_count(mut self, count: u32) -> Self {
        self.info.subresource_range.level_count = count;
        self
    }

    pub fn base_array_layer(mut self, level: u32) -> Self {
        self.info.subresource_range.base_array_layer = level;
        self
    }

    pub fn array_layer_count(mut self, count: u32) -> Self {
        self.info.subresource_range.layer_count = count;
        self
    }

    pub fn format(mut self, format: Format) -> Self {
        self.info.format = format.0;
        self
    }

    pub fn build(self) -> Result<Arc<ImageView>, DeviceError> {
        let native = unsafe {
            let view = self
                .image_wrapper
                .device
                .wrapper
                .native
                .create_image_view(&self.info, None)?;

            self.image_wrapper.device.wrapper.debug_set_object_name(
                vk::ObjectType::IMAGE_VIEW,
                view.as_raw(),
                &self.name,
            )?;

            view
        };

        Ok(Arc::new(ImageView {
            image_wrapper: self.image_wrapper,
            native,
        }))
    }
}

impl Image {
    pub const TYPE_2D: ImageType = ImageType(vk::ImageType::TYPE_2D);
    pub const TYPE_3D: ImageType = ImageType(vk::ImageType::TYPE_3D);

    pub fn size_2d(&self) -> (u32, u32) {
        (self.size.0, self.size.1)
    }

    pub fn size(&self) -> (u32, u32, u32) {
        self.size
    }

    pub fn mip_size(&self, level: u32) -> (u32, u32) {
        ((self.size.0 >> level).max(1), (self.size.1 >> level).max(1))
    }

    pub fn format(&self) -> Format {
        self.wrapper.format
    }

    pub fn mip_levels(&self) -> u32 {
        self.mip_levels
    }

    pub fn view(&self) -> &Arc<ImageView> {
        &self.view
    }

    pub fn create_view(&self) -> ImageViewBuilder {
        self.wrapper.create_view("view")
    }

    pub fn create_view_named(&self, name: &str) -> ImageViewBuilder {
        self.wrapper.create_view(name)
    }

    pub fn barrier(self: &Arc<Self>) -> ImageBarrier {
        ImageBarrier {
            native: vk::ImageMemoryBarrier::builder()
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.wrapper.native)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(self.wrapper.aspect)
                        .base_mip_level(0)
                        .level_count(self.mip_levels)
                        .base_array_layer(0)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS)
                        .build(),
                )
                .build(),
        }
    }
}

impl PartialEq for Image {
    fn eq(&self, other: &Self) -> bool {
        self.wrapper.native == other.wrapper.native
    }
}
