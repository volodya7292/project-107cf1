use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::vertex_mesh::RawVertexMesh;
use nalgebra as na;
use nalgebra::{
    Affine3, Isometry, Isometry3, Matrix4, Perspective3, Rotation3, SimdComplexField, Similarity3,
    Transform3, Translation3, UnitQuaternion, Vector3, Vector4,
};
use nalgebra_glm as glm;
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
    frustum: [Vector4<f32>; 6],
    changed: bool,
}

impl Camera {
    pub fn new(aspect: f32, fovy: f32, near: f32) -> Camera {
        let projection = glm::infinite_perspective_rh_zo(aspect, fovy, near);
        let view = Isometry3::identity();
        let proj_view_mat = projection * view.to_homogeneous();

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
            rotation: Vector3::new(0.0, 0.0, 0.0),
            aspect,
            fovy,
            near,
            frustum,
            changed: true,
        }
    }

    pub fn projection(&self) -> Matrix4<f32> {
        glm::infinite_perspective_rh_zo(self.aspect, self.fovy, self.near)
    }

    pub fn view(&self) -> Matrix4<f32> {
        let mut mat = Matrix4::<f32>::identity();
        mat *= Rotation3::from_axis_angle(&Vector3::x_axis(), self.rotation.x).to_homogeneous();
        mat *= Rotation3::from_axis_angle(&Vector3::y_axis(), self.rotation.y).to_homogeneous();
        mat *= Rotation3::from_axis_angle(&Vector3::z_axis(), self.rotation.z).to_homogeneous();
        mat *= Translation3::from(-self.position).to_homogeneous();
        mat
    }

    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn position(&self) -> Vector3<f32> {
        self.position.clone()
    }

    pub fn set_position(&mut self, position: Vector3<f32>) {
        self.position = position;
        self.changed = true;
    }

    pub fn rotation(&self) -> Vector3<f32> {
        self.rotation.clone()
    }

    pub fn set_rotation(&mut self, rotation: Vector3<f32>) {
        self.rotation = rotation.map(|x| x % (std::f32::consts::PI * 2.0));
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

    pub fn move2(&mut self, front_back: f32, left_right: f32) {
        let d = Vector3::new((-self.rotation.y).sin(), 0.0, (-self.rotation.y).cos());
        self.position -= d * front_back;
        self.position -= d.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize() * left_right;
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
