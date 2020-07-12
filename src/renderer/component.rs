use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::vertex_mesh::RawVertexMesh;
use nalgebra as na;
use nalgebra::{
    Affine3, Isometry, Isometry3, Matrix4, Perspective3, Rotation3, Similarity3, Transform3, Translation3,
    UnitQuaternion, Vector3, Vector4,
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vk_wrapper as vkw;
use vk_wrapper::{DeviceBuffer, PipelineInput};

pub struct Transform {
    pub(in crate::renderer) position: Vector3<f32>,
    pub(in crate::renderer) rotation: Vector3<f32>,
    pub(in crate::renderer) scale: Vector3<f32>,
    pub(in crate::renderer) changed: bool,
}

impl Transform {
    pub fn matrix(&self) -> Matrix4<f32> {
        let mut mat = Matrix4::<f32>::identity();
        mat.prepend_translation_mut(&self.position);
        mat *=
            Rotation3::from_euler_angles(self.rotation.x, self.rotation.y, self.rotation.z).to_homogeneous();
        mat.prepend_nonuniform_scaling_mut(&self.scale);
        mat
    }

    pub fn position(&self) -> &Vector3<f32> {
        &self.position
    }

    pub fn rotation(&self) -> &Vector3<f32> {
        &self.rotation
    }

    pub fn scale(&self) -> &Vector3<f32> {
        &self.scale
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            changed: true,
        }
    }
}

pub struct Camera {
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    aspect: f32,
    fovy: f32,
    near: f32,
    far: f32,
    frustum: [Vector4<f32>; 6],
    changed: bool,
}

impl Camera {
    pub fn new(aspect: f32, fovy: f32, near: f32, far: f32) -> Camera {
        let projection = Perspective3::new(aspect, fovy, near, far);
        let view = Isometry3::identity();
        let proj_view_mat = projection.as_matrix() * view.to_homogeneous();

        // Calculate frustum
        let mut frustum = [Vector4::<f32>::default(); 6];

        for i in 0..6 {
            let plane = &mut frustum[i];
            let sign = ((1 - i as i32 % 2) * 2 - 1) as f32;

            for j in 0..4 {
                plane[j] = proj_view_mat.column(j)[3] + (proj_view_mat.column(j)[i / 2] * sign);
            }

            *plane /= (*plane).fixed_rows::<na::U3>(0).magnitude();
        }

        Camera {
            position: Vector3::default(),
            rotation: Vector3::default(),
            aspect,
            fovy,
            near,
            far,
            frustum,
            changed: true,
        }
    }

    pub fn projection(&self) -> Matrix4<f32> {
        *Perspective3::new(self.aspect, self.fovy, self.near, self.far).as_matrix()
    }

    pub fn view(&self) -> Matrix4<f32> {
        let translation = Translation3::from(self.position);
        let rotation = UnitQuaternion::from_scaled_axis(self.rotation);
        Isometry3::from_parts(translation, rotation).to_homogeneous()
    }

    pub fn position(&self) -> &Vector3<f32> {
        &self.position
    }

    pub fn direction(&self) -> Vector3<f32> {
        let view = self.view();
        let identity = Vector4::<f32>::new(0.0, 0.0, 1.0, 1.0);
        let dir = view * identity;
        Vector3::from(dir.fixed_rows::<na::U3>(0))
    }

    pub fn fovy(&self) -> f32 {
        self.fovy
    }

    pub fn is_sphere_visible(&self, pos: Vector3<f32>, radius: f32) -> bool {
        for i in 0..6 {
            if self.frustum[i].x * pos.x
                + self.frustum[i].y * pos.y
                + self.frustum[i].z * pos.z
                + self.frustum[i].w
                <= -radius
            {
                return false;
            }
        }

        return true;
    }
}

impl specs::Component for Camera {
    type Storage = specs::VecStorage<Self>;
}

impl specs::Component for Transform {
    type Storage = specs::VecStorage<Self>;
}

pub struct VertexMeshRef {
    pub(in crate::renderer) vertex_mesh: Arc<Mutex<RawVertexMesh>>,
    changed: bool,
}

impl VertexMeshRef {
    pub fn new(vertex_mesh: &Arc<Mutex<RawVertexMesh>>) -> VertexMeshRef {
        Self {
            vertex_mesh: Arc::clone(vertex_mesh),
            changed: true,
        }
    }
}

impl specs::Component for VertexMeshRef {
    type Storage = specs::VecStorage<Self>;
}

pub struct Renderer {
    pub(in crate::renderer) mat_pipeline: Arc<MaterialPipeline>,
    pub(in crate::renderer) pipeline_inputs: Vec<Arc<PipelineInput>>,

    pub(in crate::renderer) uniform_buffer: Arc<DeviceBuffer>,
    //buffers: HashMap<u32, vkw::RawHostBuffer>,
    // binding id -> renderer impl-specific res index
    pub(in crate::renderer) translucent: bool,
    pub(in crate::renderer) changed: bool,
}

impl Renderer {
    pub fn new(
        device: &Arc<vkw::Device>,
        mat_pipeline: &Arc<MaterialPipeline>,
        translucent: bool,
    ) -> Renderer {
        Self {
            mat_pipeline: Arc::clone(&mat_pipeline),
            pipeline_inputs: vec![],
            uniform_buffer: device
                .create_device_buffer(
                    vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::UNIFORM,
                    mat_pipeline.uniform_buffer_size() as u64,
                    1,
                )
                .unwrap(),
            translucent,
            changed: true,
        }
    }
}

impl specs::Component for Renderer {
    type Storage = specs::FlaggedStorage<Self, specs::VecStorage<Self>>;
}
