use crate::render_engine::scene::Entity;
use crate::render_engine::vertex_mesh::RawVertexMesh;
use smallvec::SmallVec;
use std::sync::Arc;
use vk_wrapper::{BufferHandle, CopyRegion, DescriptorSet, DeviceBuffer};

pub mod component {
    use crate::component::Transform;
    use crate::render_engine::scene::Entity;
    use crate::utils::IndexSet;
    use std::ops::Deref;

    pub struct Parent(pub Entity);

    #[derive(Default)]
    pub struct Children {
        pub children: IndexSet<Entity>,
    }

    #[derive(Copy, Clone, Default)]
    pub struct WorldTransform(Transform);

    impl From<Transform> for WorldTransform {
        fn from(transform: Transform) -> Self {
            Self(transform)
        }
    }

    impl Deref for WorldTransform {
        type Target = Transform;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
}

pub struct Renderable {
    pub buffers: SmallVec<[DeviceBuffer; 4]>,
    pub material_pipe: u32,
    pub uniform_buf_index: usize,
    pub descriptor_sets: SmallVec<[DescriptorSet; 4]>,
}

pub struct BufferUpdate1 {
    pub buffer: BufferHandle,
    pub offset: u64,
    pub data: Vec<u8>,
}

pub struct BufferUpdate2 {
    pub buffer: BufferHandle,
    pub data: Vec<u8>,
    pub regions: Vec<CopyRegion>,
}

pub enum BufferUpdate {
    Type1(BufferUpdate1),
    Type2(BufferUpdate2),
}

pub struct VMBufferUpdate {
    pub entity: Entity,
    pub mesh: Arc<RawVertexMesh>,
}
