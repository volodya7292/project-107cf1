use common::glm::Vec3;
use gltf::mesh::util::ReadIndices;

#[derive(Debug)]
pub enum GltfLoadError {
    InvalidData(gltf::Error),
    FileNotBinary,
    BufferNotLocal,
    NoMeshes,
    NoPrimitivesInMesh,
    NoPositions,
    SparceIndices,
}

pub struct SimpleGltfMesh {
    pub positions: Vec<Vec3>,
    pub normals: Option<Vec<Vec3>>,
    pub indices: Option<Vec<u32>>,
}

/// Loads a single primitive of a single mesh from the file data.
pub fn load_simple_gltf(file_data: &[u8]) -> Result<SimpleGltfMesh, GltfLoadError> {
    let file = gltf::Gltf::from_slice(file_data).map_err(GltfLoadError::InvalidData)?;
    let raw_data = file.blob.as_deref().ok_or(GltfLoadError::FileNotBinary)?;

    let mesh = file.meshes().next().ok_or(GltfLoadError::NoMeshes)?;

    let primitive = mesh
        .primitives()
        .next()
        .ok_or(GltfLoadError::NoPrimitivesInMesh)?;
    let reader = primitive.reader(|_| Some(raw_data));

    let positions: Vec<Vec3> = reader
        .read_positions()
        .ok_or(GltfLoadError::NoPositions)
        .map(|iter| iter.map(Vec3::from).collect())?;

    let normals: Option<Vec<Vec3>> = reader.read_normals().map(|iter| iter.map(Vec3::from).collect());

    let indices: Option<Vec<u32>> = reader.read_indices().map(|indices| match indices {
        ReadIndices::U8(iter) => iter.map(|x| x as u32).collect(),
        ReadIndices::U16(iter) => iter.map(|v| v as u32).collect(),
        ReadIndices::U32(iter) => iter.collect(),
    });

    Ok(SimpleGltfMesh {
        positions,
        indices,
        normals,
    })
}
