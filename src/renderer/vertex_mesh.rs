use nalgebra as na;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::{mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::{Device, DeviceError, Format};

#[derive(Debug)]
pub enum Error {
    VertexMemberNotFound(String),
    IncorrectVertexMemberFormat(String),
}

pub trait VertexMember {
    fn vk_format() -> vkw::Format;
}

pub trait Vertex {
    fn get_member_info(name: &str) -> Option<(u32, vkw::Format)>;
}

macro_rules! vertex_impl {
    ($vertex: ty $(, $member_name: ident)*) => (
        impl $crate::renderer::vertex_mesh::Vertex for $vertex {
            fn get_member_info(name: &str) -> Option<(u32, vkw::Format)> {
                use crate::renderer::vertex_mesh::VertexMember;

                $(
                    if name == stringify!($member_name) {
                        fn get_format<T: VertexMember>(_: &T) -> vkw::Format { T::vk_format() }

                        let dummy = <$vertex>::default();
                        let offset = ((&dummy.$member_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                        let format = get_format(&dummy.$member_name);

                        Some((
                            offset as u32,
                            format,
                        ))
                    } else {
                        None
                    }
                )*
            }
        }
    )
}

impl VertexMember for na::Vector3<f32> {
    fn vk_format() -> Format {
        vkw::Format::RGB32_FLOAT
    }
}

pub struct RawVertexMesh {
    indexed: bool,
    staging_buffer: Option<vkw::HostBuffer<u8>>,
    buffer: Option<Arc<vkw::DeviceBuffer>>,
    vertex_count: u32,
    changed: bool,
}

pub struct VertexMesh<VertexT: Vertex> {
    _type_marker: PhantomData<VertexT>,
    device: Arc<Device>,
    raw: Arc<Mutex<RawVertexMesh>>,
}

impl<VertexT: Vertex> VertexMesh<VertexT> {
    pub fn get_raw(&self) -> Arc<Mutex<RawVertexMesh>> {
        Arc::clone(&self.raw)
    }

    pub fn set_vertices(&self, vertices: &[VertexT], indices: &[u16]) {
        if vertices.is_empty() {
            return;
        }

        let mut raw = self.raw.lock().unwrap();

        let vertex_size = mem::size_of::<VertexT>();
        let index_size = mem::size_of::<u16>();
        let buffer_size = vertices.len() * vertex_size + indices.len() * index_size;

        let vertices_offset = 0;
        let indices_offset = vertex_size * vertices.len();

        // Create host buffer
        raw.staging_buffer = Some(
            self.device
                .create_host_buffer::<u8>(vkw::BufferUsageFlags::TRANSFER_SRC, buffer_size as u64)
                .unwrap(),
        );

        // Copy vertices
        raw.staging_buffer
            .as_mut()
            .unwrap()
            .write(vertices_offset, unsafe {
                slice::from_raw_parts(vertices.as_ptr() as *const u8, vertex_size * vertices.len())
            });

        let mut usage_flags = vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::VERTEX;

        // Copy indices
        if indices.len() > 0 {
            raw.staging_buffer
                .as_mut()
                .unwrap()
                .write(indices_offset as u64, unsafe {
                    slice::from_raw_parts(indices.as_ptr() as *const u8, index_size * indices.len())
                });
            raw.indexed = true;
            usage_flags |= vkw::BufferUsageFlags::INDEX;
        }

        // Create device buffer
        raw.buffer = Some(
            self.device
                .create_device_buffer(usage_flags, 1, buffer_size as u64)
                .unwrap(),
        );

        raw.changed = true;
    }

    /*pub fn set_component<T: VertexMember>(&mut self, member_name: &str, values: &[T]) -> Result<(), Error> {
        let member_info = VertexT::get_member_info(member_name)
            .ok_or_else(|| Error::VertexMemberNotFound(member_name.to_owned()))?;

        if member_info.1 != T::vk_format() {
            return Err(Error::IncorrectVertexMemberFormat(format!(
                "{:?} != {:?}",
                member_info.1,
                T::vk_format()
            )));
        }

        Ok(())
    }*/
}

pub trait VertexMeshCreate {
    fn create_vertex_mesh<VertexT: Vertex>(self: &Arc<Self>)
        -> Result<VertexMesh<VertexT>, vkw::DeviceError>;
}

impl VertexMeshCreate for vkw::Device {
    fn create_vertex_mesh<VertexT: Vertex>(self: &Arc<Self>) -> Result<VertexMesh<VertexT>, DeviceError> {
        Ok(VertexMesh {
            _type_marker: PhantomData,
            device: Arc::clone(self),
            raw: Arc::new(Mutex::new(RawVertexMesh {
                indexed: false,
                staging_buffer: None,
                buffer: None,
                vertex_count: 0,
                changed: false,
            })),
        })
    }
}
