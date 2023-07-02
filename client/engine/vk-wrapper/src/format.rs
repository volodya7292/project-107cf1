use ash::vk;
use common::types::HashMap;
use lazy_static::lazy_static;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Format(pub(crate) vk::Format);

impl Format {
    pub const UNDEFINED: Self = Self(vk::Format::UNDEFINED);
    pub const D32_FLOAT: Self = Self(vk::Format::D32_SFLOAT);
    pub const R32_UINT: Self = Self(vk::Format::R32_UINT);
    pub const R32_FLOAT: Self = Self(vk::Format::R32_SFLOAT);
    pub const RG8_UNORM: Self = Self(vk::Format::R8G8_UNORM);
    pub const RG16_UNORM: Self = Self(vk::Format::R16G16_UNORM);
    pub const RG16_FLOAT: Self = Self(vk::Format::R16G16_SFLOAT);
    pub const RG32_FLOAT: Self = Self(vk::Format::R32G32_SFLOAT);
    pub const RG32_UINT: Self = Self(vk::Format::R32G32_UINT);
    pub const RGB16_FLOAT: Self = Self(vk::Format::R16G16B16_SFLOAT);
    pub const RGB32_FLOAT: Self = Self(vk::Format::R32G32B32_SFLOAT);
    pub const RGB32_UINT: Self = Self(vk::Format::R32G32B32_UINT);
    pub const RGBA8_UNORM: Self = Self(vk::Format::R8G8B8A8_UNORM);
    pub const RGBA8_SRGB: Self = Self(vk::Format::R8G8B8A8_SRGB);
    pub const RGBA16_UNORM: Self = Self(vk::Format::R16G16B16A16_UNORM);
    pub const RGBA16_FLOAT: Self = Self(vk::Format::R16G16B16A16_SFLOAT);
    pub const RGBA32_FLOAT: Self = Self(vk::Format::R32G32B32A32_SFLOAT);
    pub const RGBA32_UINT: Self = Self(vk::Format::R32G32B32A32_UINT);

    pub const BC3_RGBA_UNORM: Self = Self(vk::Format::BC3_UNORM_BLOCK);
    pub const BC5_RG_UNORM: Self = Self(vk::Format::BC5_UNORM_BLOCK);
    pub const BC7_UNORM: Self = Self(vk::Format::BC7_UNORM_BLOCK);
}

pub const BC_IMAGE_FORMATS: [Format; 3] = [Format::BC3_RGBA_UNORM, Format::BC5_RG_UNORM, Format::BC7_UNORM];
pub const DEPTH_FORMAT: Format = Format::D32_FLOAT;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FormatFeatureFlags(pub(crate) vk::FormatFeatureFlags);
vk_bitflags_impl!(FormatFeatureFlags, vk::FormatFeatureFlags);

impl FormatFeatureFlags {
    pub const COLOR_ATTACHMENT: Self = Self(vk::FormatFeatureFlags::COLOR_ATTACHMENT);
    pub const COLOR_ATTACHMENT_BLEND: Self = Self(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND);
    pub const SAMPLED_IMAGE: Self = Self(vk::FormatFeatureFlags::SAMPLED_IMAGE);
    pub const STORAGE_IMAGE: Self = Self(vk::FormatFeatureFlags::STORAGE_IMAGE);
    pub const BLIT_SRC: Self = Self(vk::FormatFeatureFlags::BLIT_SRC);
    pub const BLIT_DST: Self = Self(vk::FormatFeatureFlags::BLIT_DST);
    pub const TRANSFER_SRC: Self = Self(vk::FormatFeatureFlags::TRANSFER_SRC);
    pub const TRANSFER_DST: Self = Self(vk::FormatFeatureFlags::TRANSFER_DST);
    pub const VERTEX_BUFFER: Self = Self(vk::FormatFeatureFlags::VERTEX_BUFFER);
}

lazy_static! {
    pub static ref DEFAULT_IMAGE_FEATURES: FormatFeatureFlags = FormatFeatureFlags::COLOR_ATTACHMENT
        | FormatFeatureFlags::SAMPLED_IMAGE
        | FormatFeatureFlags::STORAGE_IMAGE
        | FormatFeatureFlags::BLIT_SRC
        | FormatFeatureFlags::BLIT_DST
        | FormatFeatureFlags::TRANSFER_SRC
        | FormatFeatureFlags::TRANSFER_DST;
    pub static ref BUFFER_FORMATS: HashMap<Format, FormatFeatureFlags> = [
        (Format::R32_UINT, FormatFeatureFlags::VERTEX_BUFFER),
        (Format::R32_FLOAT, FormatFeatureFlags::VERTEX_BUFFER),
        (Format::RG32_FLOAT, FormatFeatureFlags::VERTEX_BUFFER),
        (Format::RGB32_FLOAT, FormatFeatureFlags::VERTEX_BUFFER),
        (Format::RGBA8_UNORM, FormatFeatureFlags::VERTEX_BUFFER),
        (Format::RGBA32_UINT, FormatFeatureFlags::VERTEX_BUFFER),
    ]
    .into_iter()
    .collect();
    pub static ref IMAGE_FORMATS: HashMap<Format, FormatFeatureFlags> = [
        (Format::R32_UINT, *DEFAULT_IMAGE_FEATURES),
        (Format::R32_FLOAT, *DEFAULT_IMAGE_FEATURES),
        (Format::RG8_UNORM, *DEFAULT_IMAGE_FEATURES),
        (Format::RG16_UNORM, *DEFAULT_IMAGE_FEATURES),
        (Format::RG16_FLOAT, *DEFAULT_IMAGE_FEATURES),
        (Format::RG32_UINT, *DEFAULT_IMAGE_FEATURES),
        (Format::RGBA8_UNORM, *DEFAULT_IMAGE_FEATURES),
        (Format::RGBA8_SRGB, *DEFAULT_IMAGE_FEATURES),
        (Format::RGBA16_FLOAT, *DEFAULT_IMAGE_FEATURES),
        (Format::RGBA32_FLOAT, *DEFAULT_IMAGE_FEATURES),
    ]
    .into_iter()
    .collect();
    pub static ref FORMAT_SIZES: HashMap<Format, u8> = [
        (Format::UNDEFINED, 0),
        (Format::D32_FLOAT, 4),
        (Format::R32_UINT, 4),
        (Format::R32_FLOAT, 4),
        (Format::RG8_UNORM, 2),
        (Format::RG16_UNORM, 4),
        (Format::RG32_FLOAT, 8),
        (Format::RG32_UINT, 8),
        (Format::RGB16_FLOAT, 6),
        (Format::RGB32_FLOAT, 12),
        (Format::RGB32_UINT, 12),
        (Format::RGBA8_UNORM, 4),
        (Format::RGBA8_SRGB, 4),
        (Format::RGBA16_UNORM, 8),
        (Format::RGBA16_FLOAT, 8),
        (Format::RGBA32_FLOAT, 16),
        (Format::RGBA32_UINT, 16),
        (Format::BC3_RGBA_UNORM, 1),
        (Format::BC5_RG_UNORM, 1),
        (Format::BC7_UNORM, 1),
    ]
    .into_iter()
    .collect();
}
