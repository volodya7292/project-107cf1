use ash::vk;
use lazy_static::lazy_static;
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Format(pub(crate) vk::Format);

impl Format {
    pub const UNDEFINED: Self = Self(vk::Format::UNDEFINED);
    pub const D32_FLOAT: Self = Self(vk::Format::D32_SFLOAT);
    pub const R32_UINT: Self = Self(vk::Format::R32_UINT);
    pub const RG16_UNORM: Self = Self(vk::Format::R16G16_UNORM);
    pub const RG32_FLOAT: Self = Self(vk::Format::R32G32_SFLOAT);
    pub const RG32_UINT: Self = Self(vk::Format::R32G32_UINT);
    pub const RGB16_FLOAT: Self = Self(vk::Format::R16G16B16_SFLOAT);
    pub const RGB32_FLOAT: Self = Self(vk::Format::R32G32B32_SFLOAT);
    pub const RGB32_UINT: Self = Self(vk::Format::R32G32B32_UINT);
    pub const RGBA8_UNORM: Self = Self(vk::Format::R8G8B8A8_UNORM);
    pub const RGBA16_UNORM: Self = Self(vk::Format::R16G16B16A16_UNORM);
    pub const RGBA16_FLOAT: Self = Self(vk::Format::R16G16B16A16_SFLOAT);
    pub const RGBA32_FLOAT: Self = Self(vk::Format::R32G32B32A32_SFLOAT);

    pub const BC3_RGBA_UNORM: Self = Self(vk::Format::BC3_UNORM_BLOCK);
    pub const BC5_RG_UNORM: Self = Self(vk::Format::BC5_UNORM_BLOCK);
    pub const BC7_UNORM: Self = Self(vk::Format::BC7_UNORM_BLOCK);
}

// TODO
pub const TEXEL_BUFFER_FORMATS: [Format; 0] = [];
// TODO
pub const IMAGE_FORMATS: [Format; 3] = [Format::RGBA8_UNORM, Format::RG16_UNORM, Format::RG32_UINT];
pub const BC_IMAGE_FORMATS: [Format; 3] = [Format::BC3_RGBA_UNORM, Format::BC5_RG_UNORM, Format::BC7_UNORM];
pub const DEPTH_FORMAT: Format = Format::D32_FLOAT;

lazy_static! {
    pub static ref FORMAT_SIZES: HashMap<Format, u8> = [
        (Format::UNDEFINED, 0),
        (Format::D32_FLOAT, 4),
        (Format::R32_UINT, 4),
        (Format::RG16_UNORM, 4),
        (Format::RG32_FLOAT, 8),
        (Format::RG32_UINT, 8),
        (Format::RGB16_FLOAT, 6),
        (Format::RGB32_FLOAT, 12),
        (Format::RGB32_UINT, 12),
        (Format::RGBA8_UNORM, 4),
        (Format::RGBA16_UNORM, 8),
        (Format::RGBA16_FLOAT, 8),
        (Format::RGBA32_FLOAT, 16),
        (Format::BC3_RGBA_UNORM, 1),
        (Format::BC5_RG_UNORM, 1),
        (Format::BC7_UNORM, 1),
    ]
    .iter()
    .cloned()
    .collect();
}
