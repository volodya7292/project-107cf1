use nalgebra as na;
use nalgebra_glm::{U8Vec4, UVec3, UVec4, Vec2, Vec3};
use std::marker::PhantomData;
use std::sync::Arc;
use std::{mem, ptr, slice};
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{BufferHandle, Format};

#[derive(Debug)]
pub enum Error {
    VertexMemberNotFound(String),
    IncorrectVertexMemberFormat(String),
}

pub struct VertexAttribute {
    pub format: Format,
    pub member_offset: u32,
}

pub trait VertexMember {
    fn vk_format() -> Format;
}

pub trait AttributesImpl {
    fn attributes() -> Vec<VertexAttribute>;
    fn member_info(name: &str) -> Option<VertexAttribute>;
}

pub trait VertexPositionImpl {
    fn position(&self) -> Vec3;

    fn set_position(&self, _pos: na::Vector3<f32>) {
        unimplemented!()
    }
}

pub trait VertexNormalImpl {
    fn normal(&self) -> &na::Vector3<f32>;

    fn set_normal(&self, _normal: na::Vector3<f32>) {
        unimplemented!()
    }
}

#[macro_export]
macro_rules! vertex_impl_position {
    ($vertex: ty) => {
        impl $crate::renderer::vertex_mesh::VertexPositionImpl for $vertex {
            fn position(&self) -> nalgebra_glm::Vec3 {
                self.position
            }

            fn set_position(&mut self, pos: nalgebra::Vector3<f32>) {
                self.position = pos;
            }
        }
    };
}

#[macro_export]
macro_rules! vertex_impl_normal {
    ($vertex: ty) => {
        impl $crate::renderer::vertex_mesh::VertexNormalImpl for $vertex {
            fn normal(&self) -> &nalgebra::Vector3<f32> {
                &self.normal
            }

            fn normal_mut(&mut self, normal: nalgebra::Vector3<f32>) {
                self.normal = normal;
            }
        }
    };
}

#[macro_export]
macro_rules! attributes_impl {
    ($vertex: ty $(, $member_name: ident)*) => (
        impl $crate::renderer::vertex_mesh::AttributesImpl for $vertex {
            fn attributes() -> Vec<$crate::renderer::vertex_mesh::VertexAttribute> {
                use $crate::renderer::vertex_mesh::{VertexMember, VertexAttribute};

                fn get_format<T: VertexMember>(_: &T) -> vk_wrapper::Format { T::vk_format() }

                let mut attribs = vec![];
                let dummy = <$vertex>::default();

                $(
                    let offset = ((&dummy.$member_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                    let format = get_format(&dummy.$member_name);

                    attribs.push(VertexAttribute {
                        member_offset: offset as u32,
                        format,
                    });
                )*

                attribs
            }

            fn member_info(name: &str) -> Option<$crate::renderer::vertex_mesh::VertexAttribute> {
                use $crate::renderer::vertex_mesh::{VertexMember, VertexAttribute};

                $(
                    if name == stringify!($member_name) {
                        fn get_format<T: VertexMember>(_: &T) -> vk_wrapper::Format { T::vk_format() }

                        let dummy = <$vertex>::default();
                        let offset = ((&dummy.$member_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                        let format = get_format(&dummy.$member_name);

                        return Some(VertexAttribute {
                            member_offset: offset as u32,
                            format,
                        });
                    }
                )*

                return None;
            }
        }
    )
}

impl VertexMember for u32 {
    fn vk_format() -> Format {
        Format::R32_UINT
    }
}

impl VertexMember for f32 {
    fn vk_format() -> Format {
        Format::R32_FLOAT
    }
}

impl VertexMember for Vec2 {
    fn vk_format() -> Format {
        Format::RG32_FLOAT
    }
}

impl VertexMember for Vec3 {
    fn vk_format() -> Format {
        Format::RGB32_FLOAT
    }
}

impl VertexMember for UVec3 {
    fn vk_format() -> Format {
        Format::RGBA32_UINT
    }
}

impl VertexMember for U8Vec4 {
    fn vk_format() -> Format {
        Format::RGBA8_UNORM
    }
}

impl VertexMember for UVec4 {
    fn vk_format() -> Format {
        Format::RGBA32_UINT
    }
}

pub trait InstanceImpl: AttributesImpl {
    fn has_aabb() -> bool {
        false
    }

    fn aabb(&self) -> (Vec3, Vec3) {
        (Vec3::default(), Vec3::default())
    }
}

impl<T: AttributesImpl> InstanceImpl for T {}

impl AttributesImpl for () {
    fn attributes() -> Vec<VertexAttribute> {
        vec![]
    }

    fn member_info(_: &str) -> Option<VertexAttribute> {
        None
    }
}

impl VertexPositionImpl for () {
    fn position(&self) -> Vec3 {
        panic!("() is not vertex struct");
    }
}

#[derive(Default, Copy, Clone)]
pub struct Sphere {
    center: na::Vector3<f32>,
    radius: f32,
}

impl Sphere {
    pub fn center(&self) -> &na::Vector3<f32> {
        &self.center
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }
}

#[derive(Default)]
pub struct RawVertexMesh {
    indexed: bool,
    pub(crate) staging_buffer: Option<vkw::HostBuffer<u8>>,
    pub(crate) buffer: Option<vkw::DeviceBuffer>,
    pub(crate) vertex_count: u32,
    pub(crate) instance_count: u32,
    index_count: u32,
    aabb: Option<(na::Vector3<f32>, na::Vector3<f32>)>,
    sphere: Option<Sphere>,
    bindings: Vec<(BufferHandle, u64)>,
    indices_offset: u64,
}

impl RawVertexMesh {
    pub fn without_data(vertices: u32, instances: u32) -> Arc<Self> {
        Arc::new(Self {
            indexed: false,
            staging_buffer: None,
            buffer: None,
            vertex_count: vertices,
            instance_count: instances,
            index_count: 0,
            aabb: None,
            sphere: None,
            bindings: vec![],
            indices_offset: 0,
        })
    }

    pub fn is_empty(&self) -> bool {
        self.vertex_count == 0
    }

    /// Returns `None` if `self` has no data to infer AABB from.
    pub fn aabb(&self) -> Option<&(na::Vector3<f32>, na::Vector3<f32>)> {
        self.aabb.as_ref()
    }

    /// Returns `None` if `self` has no data to infer sphere from.
    pub fn sphere(&self) -> Option<&Sphere> {
        self.sphere.as_ref()
    }

    pub fn bindings(&self) -> &[(BufferHandle, u64)] {
        &self.bindings
    }
}

#[derive(Default)]
pub struct VertexMesh<VertexT: AttributesImpl, InstanceT: InstanceImpl> {
    _type_marker: PhantomData<VertexT>,
    _type_marker2: PhantomData<InstanceT>,
    raw: Arc<RawVertexMesh>,
}

impl<VertexT, InstanceT> VertexMesh<VertexT, InstanceT>
where
    VertexT: AttributesImpl + Clone + Default,
    InstanceT: InstanceImpl,
{
    pub fn raw(&self) -> Arc<RawVertexMesh> {
        Arc::clone(&self.raw)
    }

    pub fn get_vertices(&self, first_vertex: u32, count: u32) -> Vec<VertexT> {
        let raw = &self.raw;
        let attribs = VertexT::attributes();

        let first_vertex = first_vertex.min(raw.vertex_count);
        let mut vertices = vec![Default::default(); count.min(raw.vertex_count - first_vertex) as usize];

        if let Some(ref staging_buffer) = raw.staging_buffer {
            for attrib in attribs {
                let buffer_offset = attrib.member_offset as isize * raw.vertex_count as isize;
                let format_size = vkw::FORMAT_SIZES[&attrib.format] as isize;

                for (i, vertex) in vertices.iter_mut().enumerate() {
                    unsafe {
                        ptr::copy_nonoverlapping(
                            staging_buffer
                                .as_ptr()
                                .offset(buffer_offset + format_size * (first_vertex + i as u32) as isize),
                            (vertex as *mut VertexT as *mut u8).offset(attrib.member_offset as isize),
                            format_size as usize,
                        );
                    }
                }
            }
        }

        vertices
    }

    pub fn get_indices(&self, first_index: u32, count: u32) -> Vec<u32> {
        let raw = &self.raw;

        let first_index = first_index.min(raw.index_count);
        let mut indices = vec![0u32; count.min(raw.index_count - first_index) as usize];

        if let Some(ref staging_buffer) = raw.staging_buffer {
            staging_buffer.read(
                raw.indices_offset + first_index as u64 * mem::size_of::<u32>() as u64,
                unsafe {
                    slice::from_raw_parts_mut(
                        indices.as_mut_ptr() as *mut u8,
                        indices.len() * mem::size_of::<u32>(),
                    )
                },
            );
        }

        indices
    }

    pub fn vertex_count(&self) -> u32 {
        self.raw.vertex_count
    }

    pub fn index_count(&self) -> u32 {
        self.raw.index_count
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

pub enum VAttributes<'a, T = ()> {
    Slice(&'a [T]),
    /// An object may not have an vertices to render, in that case the shader can infer
    /// necessary vertices from gl_VertexIndex (useful for rendering a single quad).
    WithoutData(u32),
}

impl<'a, T> VAttributes<'a, T> {
    pub fn data(&self) -> Option<&'a [T]> {
        if let Self::Slice(arr) = self {
            Some(*arr)
        } else {
            None
        }
    }

    pub fn attributes_len(&self) -> u32 {
        match self {
            Self::Slice(a) => a.len() as u32,
            Self::WithoutData(n) => *n,
        }
    }
}

pub trait VertexMeshCreate {
    fn create_instanced_vertex_mesh<VertexT, InstanceT: AttributesImpl>(
        self: &Arc<Self>,
        vertices: VAttributes<VertexT>,
        instances: VAttributes<InstanceT>,
        indices: Option<&[u32]>,
    ) -> Result<VertexMesh<VertexT, InstanceT>, Error>
    where
        VertexT: AttributesImpl + VertexPositionImpl;

    fn create_vertex_mesh<VertexT>(
        self: &Arc<Self>,
        vertices: VAttributes<VertexT>,
        indices: Option<&[u32]>,
    ) -> Result<VertexMesh<VertexT, ()>, Error>
    where
        VertexT: AttributesImpl + VertexPositionImpl,
    {
        self.create_instanced_vertex_mesh::<VertexT, ()>(vertices, VAttributes::WithoutData(1), indices)
    }
}

impl VertexMeshCreate for vkw::Device {
    fn create_instanced_vertex_mesh<VertexT, InstanceT>(
        self: &Arc<Self>,
        vertices: VAttributes<VertexT>,
        instances: VAttributes<InstanceT>,
        indices: Option<&[u32]>,
    ) -> Result<VertexMesh<VertexT, InstanceT>, Error>
    where
        VertexT: AttributesImpl + VertexPositionImpl,
        InstanceT: AttributesImpl,
    {
        let indexed = indices.is_some();
        let vertices_data = vertices.data().unwrap_or(&[]);
        let instances_data = instances.data().unwrap_or(&[]);
        let indices_data = indices.unwrap_or(&[]);

        if vertices_data.is_empty() && !indices_data.is_empty() {
            panic!("if `vertices` slice is empty, `indices` slice must also be empty");
        }

        if vertices_data.is_empty() && instances_data.is_empty() && indices_data.is_empty() {
            return Ok(VertexMesh {
                _type_marker: PhantomData,
                _type_marker2: PhantomData,
                raw: Default::default(),
            });
        }

        let vertex_size = mem::size_of::<VertexT>();
        let instance_size = mem::size_of::<InstanceT>();
        let index_size = mem::size_of::<u32>();
        let buffer_size = vertices_data.len() * vertex_size
            + instances_data.len() * instance_size
            + indices_data.len() * index_size;

        let instances_offset = vertices_data.len() * vertex_size;
        let indices_offset = instances_offset + instances_data.len() * instance_size;

        // Create host buffer
        let mut staging_buffer = self
            .create_host_buffer::<u8>(vkw::BufferUsageFlags::TRANSFER_SRC, buffer_size as u64)
            .unwrap();

        let mut usage_flags = vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::VERTEX;
        if indexed {
            usage_flags |= vkw::BufferUsageFlags::INDEX;
        }

        // Create device buffer
        let buffer = self
            .create_device_buffer(usage_flags, 1, buffer_size as u64)
            .unwrap();

        // Copy vertices
        let attribs = VertexT::attributes();
        let mut bindings = vec![];

        for attrib in attribs {
            let buffer_offset = attrib.member_offset as isize * vertices_data.len() as isize;
            let format_size = vkw::FORMAT_SIZES[&attrib.format] as isize;

            for (i, vertex) in vertices_data.iter().enumerate() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (vertex as *const VertexT as *const u8).offset(attrib.member_offset as isize),
                        staging_buffer
                            .as_mut_ptr()
                            .offset(buffer_offset + format_size * i as isize),
                        format_size as usize,
                    );
                }
            }

            // Set binging buffers
            bindings.push((buffer.handle(), buffer_offset as u64));
        }

        // Copy instances
        let attribs = InstanceT::attributes();

        for attrib in attribs {
            let member_offset = attrib.member_offset as usize;
            let buffer_offset = instances_offset + member_offset * instances_data.len();
            let format_size = vkw::FORMAT_SIZES[&attrib.format] as usize;

            for (i, instance) in instances_data.iter().enumerate() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (instance as *const InstanceT as *const u8).add(member_offset),
                        staging_buffer.as_mut_ptr().add(buffer_offset + format_size * i),
                        format_size as usize,
                    );
                }
            }

            // Set binging buffers
            bindings.push((buffer.handle(), buffer_offset as u64));
        }

        // Copy indices
        if indices_data.len() > 0 {
            staging_buffer.write(indices_offset as u64, unsafe {
                slice::from_raw_parts(
                    indices_data.as_ptr() as *const u8,
                    index_size * indices_data.len(),
                )
            });
        }

        // Calculate bounds
        let aabb = if InstanceT::has_aabb() && !instances_data.is_empty() {
            let mut aabb = instances_data[0].aabb();

            for instance in &instances_data[1..] {
                let i_aabb = instance.aabb();
                aabb.0 = aabb.0.inf(&i_aabb.0);
                aabb.1 = aabb.1.sup(&i_aabb.1);
            }
            Some(aabb)
        } else if !vertices_data.is_empty() {
            let mut aabb = (vertices_data[0].position(), vertices_data[0].position());

            for vertex in &vertices_data[1..] {
                aabb.0 = aabb.0.inf(&vertex.position());
                aabb.1 = aabb.1.sup(&vertex.position());
            }
            Some(aabb)
        } else {
            None
        };

        // Calculate sphere if bounds present
        let sphere = aabb.map(|aabb| {
            let center = (aabb.0 + aabb.1) / 2.0;
            let mut radius = 0.0;

            if InstanceT::has_aabb() && !instances_data.is_empty() {
                for instance in instances_data {
                    let i_aabb = instance.aabb();
                    radius = (center - i_aabb.0).magnitude().max(radius);
                    radius = (center - i_aabb.1).magnitude().max(radius);
                }
            } else if !vertices_data.is_empty() {
                for vertex in vertices_data {
                    radius = (center - vertex.position()).magnitude().max(radius);
                }
            }

            Sphere { center, radius }
        });

        let raw = RawVertexMesh {
            indexed,
            staging_buffer: Some(staging_buffer),
            buffer: Some(buffer),
            vertex_count: vertices.attributes_len(),
            instance_count: instances.attributes_len(),
            index_count: indices_data.len() as u32,
            aabb,
            sphere,
            bindings,
            indices_offset: indices_offset as u64,
        };
        Ok(VertexMesh {
            _type_marker: PhantomData,
            _type_marker2: PhantomData,
            raw: Arc::new(raw),
        })
    }
}

pub trait VertexMeshCmdList {
    fn bind_and_draw_vertex_mesh(&mut self, vertex_mesh: &Arc<RawVertexMesh>);
}

impl VertexMeshCmdList for vkw::CmdList {
    fn bind_and_draw_vertex_mesh(&mut self, vertex_mesh: &Arc<RawVertexMesh>) {
        if vertex_mesh.is_empty() {
            return;
        }
        self.bind_vertex_buffers(0, &vertex_mesh.bindings);

        if vertex_mesh.indexed && vertex_mesh.index_count > 0 {
            self.bind_index_buffer(vertex_mesh.buffer.as_ref().unwrap(), vertex_mesh.indices_offset);
            self.draw_indexed_instanced(vertex_mesh.index_count, 0, 0, 0, vertex_mesh.instance_count);
        } else {
            self.draw_instanced(vertex_mesh.vertex_count, 0, 0, vertex_mesh.instance_count);
        }
    }
}
