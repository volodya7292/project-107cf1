use crate::resource_file::ResourceRef;
use std::sync::Arc;
use vk_wrapper as vkw;

#[derive(Copy, Clone)]
pub struct Cell {
    level: u32,
    inner: [u32; 4],
}

pub struct TextureAtlas {
    image: Arc<vkw::Image>,
    staging_image: Arc<vkw::Image>,
    size: u32,
    size_in_cells: u32,
    max_texture_size: u32,
    resource_refs: Vec<ResourceRef>,
    cells: Vec<Cell>,
}

impl TextureAtlas {
    pub fn add_texture(&mut self, res_ref: ResourceRef) -> u32 {
        self.resource_refs.push(res_ref);
        (self.resource_refs.len() - 1) as u32
    }

    pub fn load_texture(&self, index: u32) {
        // 1. find cell
        // 2. defragment cell if necessary
    }

    pub fn unload_texture(&self, index: u32) {}

    pub fn get_cell(&self, pos: (u32, u32)) -> &Cell {
        &self.cells[(pos.1 * self.size_in_cells + pos.0) as usize]
    }
}

fn next_power_of_two(mut n: u32) -> u32 {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    n
}

fn make_mul_of(number: u32, multiplier: u32) -> u32 {
    ((number + multiplier - 1) / multiplier) * multiplier
}

/// Creates a new texture atlas with resolution (size x size) and
/// max texture resolution (max_texture_size x max_texture_size)
pub fn new(
    device: Arc<vkw::Device>,
    format: vkw::Format,
    gen_mipmaps: bool,
    max_anisotropy: f32,
    size: u32,
    max_texture_size: u32,
) -> Result<TextureAtlas, vkw::DeviceError> {
    let max_texture_size = next_power_of_two(max_texture_size);
    let size = make_mul_of(size, max_texture_size);
    let size_in_cells = size / max_texture_size;
    let cell_count = size_in_cells * size_in_cells;

    let image = device.create_image_2d(
        format,
        gen_mipmaps,
        max_anisotropy,
        vkw::ImageUsageFlags::TRANSFER_SRC
            | vkw::ImageUsageFlags::TRANSFER_DST
            | vkw::ImageUsageFlags::SAMPLED,
        (size, size),
    )?;
    let staging_image = device.create_image_2d(
        format,
        false,
        1.0,
        vkw::ImageUsageFlags::TRANSFER_SRC | vkw::ImageUsageFlags::TRANSFER_DST,
        (max_texture_size, max_texture_size),
    )?;

    Ok(TextureAtlas {
        image,
        staging_image,
        size,
        size_in_cells,
        max_texture_size,
        resource_refs: vec![],
        cells: vec![
            Cell {
                level: 0,
                inner: [u32::MAX; 4]
            };
            cell_count as usize
        ],
    })
}
