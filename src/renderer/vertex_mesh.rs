use nalgebra as na;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};
use std::{mem, ptr, slice};
use vk_wrapper as vkw;
use vk_wrapper::{Device, Format};

#[derive(Debug)]
pub enum Error {
    VertexMemberNotFound(String),
    IncorrectVertexMemberFormat(String),
}

pub trait VertexMember {
    fn vk_format() -> vkw::Format;
}

pub trait Vertex {
    fn attributes() -> Vec<(u32, vkw::Format)>;
    fn member_info(name: &str) -> Option<(u32, vkw::Format)>;
    fn position(&self) -> &na::Vector3<f32>;
}

macro_rules! vertex_impl {
    ($vertex: ty $(, $member_name: ident)*) => (
        impl $crate::renderer::vertex_mesh::Vertex for $vertex {
            fn attributes() -> Vec<(u32, vk_wrapper::Format)> {
                use crate::renderer::vertex_mesh::VertexMember;

                fn get_format<T: VertexMember>(_: &T) -> vk_wrapper::Format { T::vk_format() }

                let mut attribs = vec![];
                let dummy = <$vertex>::default();

                $(
                    let offset = ((&dummy.$member_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                    let format = get_format(&dummy.$member_name);

                    attribs.push((
                        offset as u32,
                        format,
                    ));
                )*

                attribs
            }

            fn member_info(name: &str) -> Option<(u32, vk_wrapper::Format)> {
                use crate::renderer::vertex_mesh::VertexMember;

                $(
                    if name == stringify!($member_name) {
                        fn get_format<T: VertexMember>(_: &T) -> vk_wrapper::Format { T::vk_format() }

                        let dummy = <$vertex>::default();
                        let offset = ((&dummy.$member_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                        let format = get_format(&dummy.$member_name);

                        return Some((
                            offset as u32,
                            format,
                        ));
                    }
                )*

                return None;
            }

            fn position(&self) -> &nalgebra::Vector3<f32> {
                &self.position
            }
        }
    )
}

impl VertexMember for u32 {
    fn vk_format() -> Format {
        vkw::Format::R32_UINT
    }
}

impl VertexMember for na::Vector2<f32> {
    fn vk_format() -> Format {
        vkw::Format::RG32_FLOAT
    }
}

impl VertexMember for na::Vector3<f32> {
    fn vk_format() -> Format {
        vkw::Format::RGB32_FLOAT
    }
}

impl VertexMember for na::Vector4<u32> {
    fn vk_format() -> Format {
        vkw::Format::RGBA32_UINT
    }
}

pub struct RawVertexMesh {
    indexed: bool,
    pub(in crate::renderer) staging_buffer: Option<vkw::HostBuffer<u8>>,
    pub(in crate::renderer) buffer: Option<Arc<vkw::DeviceBuffer>>,
    _vertex_size: u32,
    pub(in crate::renderer) vertex_count: u32,
    index_count: u32,
    aabb: (na::Vector3<f32>, na::Vector3<f32>),
    bindings: Vec<(Arc<vkw::DeviceBuffer>, u64)>,
    indices_offset: u64,
    pub(in crate::renderer) changed: bool,
}

impl RawVertexMesh {
    pub fn aabb(&self) -> &(na::Vector3<f32>, na::Vector3<f32>) {
        &self.aabb
    }
}

pub struct VertexMesh<VertexT: Vertex> {
    _type_marker: PhantomData<VertexT>,
    device: Arc<Device>,
    raw: Arc<Mutex<RawVertexMesh>>,
}

impl<VertexT> VertexMesh<VertexT>
where
    VertexT: Vertex + Clone + Default,
{
    pub fn raw(&self) -> Arc<Mutex<RawVertexMesh>> {
        Arc::clone(&self.raw)
    }

    pub fn get_vertices(&self, first_vertex: u32, count: u32) -> Vec<VertexT> {
        let raw = self.raw.as_ref().lock().unwrap();
        let attribs = VertexT::attributes();

        let mut vertices =
            vec![Default::default(); count.min(raw.vertex_count.max(first_vertex) - first_vertex) as usize];

        for (vertex_offset, format) in attribs {
            let buffer_offset = vertex_offset as isize * vertices.len() as isize;
            let format_size = vkw::FORMAT_SIZES[&format] as isize;

            for (i, vertex) in vertices.iter_mut().enumerate() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        raw.staging_buffer
                            .as_ref()
                            .unwrap()
                            .as_ptr()
                            .offset(buffer_offset + format_size * (first_vertex + i as u32) as isize),
                        (vertex as *mut VertexT as *mut u8).offset(vertex_offset as isize),
                        format_size as usize,
                    );
                }
            }
        }

        vertices
    }

    pub fn get_indices(&self, first_index: u32, count: u32) -> Vec<u32> {
        let raw = self.raw.as_ref().lock().unwrap();

        let mut indices = vec![0u32; count.min(raw.index_count.max(first_index) - first_index) as usize];

        raw.staging_buffer
            .as_ref()
            .unwrap()
            .read(raw.indices_offset, unsafe {
                slice::from_raw_parts_mut(indices.as_mut_ptr() as *mut u8, indices.len())
            });

        indices
    }

    pub fn set_vertices(&mut self, vertices: &[VertexT], indices: &[u32]) {
        if vertices.is_empty() {
            return;
        }

        let indexed = indices.len() > 0;

        let vertex_size = mem::size_of::<VertexT>();
        let index_size = mem::size_of::<u32>();
        let buffer_size = vertices.len() * vertex_size + indices.len() * index_size;

        let indices_offset = vertex_size * vertices.len();

        // Create host buffer
        let mut staging_buffer = self
            .device
            .create_host_buffer::<u8>(vkw::BufferUsageFlags::TRANSFER_SRC, buffer_size as u64)
            .unwrap();

        let mut usage_flags = vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::VERTEX;
        if indexed {
            usage_flags |= vkw::BufferUsageFlags::INDEX;
        }

        // Create device buffer
        let buffer = self
            .device
            .create_device_buffer(usage_flags, 1, buffer_size as u64)
            .unwrap();

        // Copy vertices
        let attribs = VertexT::attributes();
        let mut bindings = vec![];

        for (vertex_offset, format) in attribs {
            let buffer_offset = vertex_offset as isize * vertices.len() as isize;
            let format_size = vkw::FORMAT_SIZES[&format] as isize;

            for (i, vertex) in vertices.iter().enumerate() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (vertex as *const VertexT as *const u8).offset(vertex_offset as isize),
                        staging_buffer
                            .as_mut_ptr()
                            .offset(buffer_offset + format_size * i as isize),
                        format_size as usize,
                    );
                }
            }

            // Set binging buffers
            bindings.push((Arc::clone(&buffer), buffer_offset as u64));
        }

        // Copy indices
        if indices.len() > 0 {
            staging_buffer.write(indices_offset as u64, unsafe {
                slice::from_raw_parts(indices.as_ptr() as *const u8, index_size * indices.len())
            });
        }

        // Calculate bounds
        let mut aabb = (
            (*vertices[0].position()).clone(),
            (*vertices[0].position()).clone(),
        );
        for vertex in &vertices[1..] {
            aabb.0 = aabb.0.inf(vertex.position());
            aabb.1 = aabb.1.sup(vertex.position());
        }

        let mut raw = self.raw.as_ref().lock().unwrap();
        *raw = RawVertexMesh {
            indexed,
            staging_buffer: Some(staging_buffer),
            buffer: Some(buffer),
            _vertex_size: mem::size_of::<VertexT>() as u32,
            vertex_count: vertices.len() as u32,
            index_count: indices.len() as u32,
            aabb,
            bindings,
            indices_offset: indices_offset as u64,
            changed: true,
        };
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
    fn create_vertex_mesh<VertexT: Vertex>(self: &Arc<Self>) -> Result<VertexMesh<VertexT>, Error>;
}

impl VertexMeshCreate for vkw::Device {
    fn create_vertex_mesh<VertexT: Vertex>(self: &Arc<Self>) -> Result<VertexMesh<VertexT>, Error> {
        const POS_FORMAT: vkw::Format = vkw::Format::RGB32_FLOAT;

        let pos_info =
            VertexT::member_info("position").ok_or(Error::VertexMemberNotFound("position".to_owned()))?;
        if pos_info.1 != POS_FORMAT {
            return Err(Error::IncorrectVertexMemberFormat(format!(
                "{:?} != {:?}",
                pos_info.1, POS_FORMAT
            )));
        }

        Ok(VertexMesh {
            _type_marker: PhantomData,
            device: Arc::clone(self),
            raw: Arc::new(Mutex::new(RawVertexMesh {
                indexed: false,
                staging_buffer: None,
                buffer: None,
                _vertex_size: 0,
                vertex_count: 0,
                index_count: 0,
                aabb: Default::default(),
                bindings: vec![],
                indices_offset: 0,
                changed: false,
            })),
        })
    }
}

pub trait VertexMeshCmdList {
    fn bind_and_draw_vertex_mesh(&mut self, vertex_mesh: &Arc<Mutex<RawVertexMesh>>);
}

impl VertexMeshCmdList for vkw::CmdList {
    fn bind_and_draw_vertex_mesh(&mut self, vertex_mesh: &Arc<Mutex<RawVertexMesh>>) {
        let vertex_mesh = vertex_mesh.lock().unwrap();

        self.bind_vertex_buffers(0, &vertex_mesh.bindings);

        if vertex_mesh.indexed {
            self.bind_index_buffer(&vertex_mesh.buffer.as_ref().unwrap(), vertex_mesh.indices_offset);
            self.draw_indexed(vertex_mesh.index_count, 0, 0);
        } else {
            self.draw(vertex_mesh.vertex_count, 0);
        }
    }
}
