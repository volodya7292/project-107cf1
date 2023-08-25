use common::glm;
use common::glm::{DVec3, Mat4, Vec3, Vec4};

#[derive(Copy, Clone)]
pub struct PerspectiveCamera {
    position: DVec3,
    rotation: Vec3,
    aspect: f32,
    fov_y: f32,
    z_near: f32,
}

pub fn create_view_matrix(position: &Vec3, rotation: &Vec3) -> Mat4 {
    let mut mat = Mat4::identity();
    mat = Mat4::new_translation(&(-position)) * mat;
    mat = Mat4::from_axis_angle(&Vec3::z_axis(), rotation.z) * mat;
    mat = Mat4::from_axis_angle(&Vec3::y_axis(), rotation.y) * mat;
    mat = Mat4::from_axis_angle(&Vec3::x_axis(), rotation.x) * mat;
    mat
}

pub fn rotation_to_direction(rotation: &Vec3) -> Vec3 {
    let mut mat = Mat4::identity();
    mat = Mat4::from_axis_angle(&Vec3::z_axis(), rotation.z) * mat;
    mat = Mat4::from_axis_angle(&Vec3::y_axis(), rotation.y) * mat;
    mat = Mat4::from_axis_angle(&Vec3::x_axis(), rotation.x) * mat;

    let d: Vec3 = mat.row(2).transpose().fixed_rows::<3>(0).into();
    -d
}

impl PerspectiveCamera {
    pub fn new(aspect: f32, fov_y: f32, z_near: f32) -> Self {
        Self {
            position: DVec3::default(),
            rotation: Vec3::new(0.0, 0.0, 0.0),
            aspect,
            fov_y,
            z_near,
        }
    }

    pub fn projection(&self) -> Mat4 {
        glm::reversed_infinite_perspective_rh_zo(self.aspect, self.fov_y, self.z_near)
    }

    pub fn z_near(&self) -> f32 {
        self.z_near
    }

    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn set_aspect(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn position(&self) -> &DVec3 {
        &self.position
    }

    pub fn set_position(&mut self, position: DVec3) {
        self.position = position;
    }

    pub fn rotation(&self) -> &Vec3 {
        &self.rotation
    }

    pub fn set_rotation(&mut self, rotation: Vec3) {
        self.rotation = rotation.map(|x| x % (std::f32::consts::PI * 2.0));
    }

    pub fn direction(&self) -> Vec3 {
        rotation_to_direction(&self.rotation)
    }

    pub fn fovy(&self) -> f32 {
        self.fov_y
    }

    pub fn set_fovy(&mut self, fov_y: f32) {
        self.fov_y = fov_y;
    }

    pub fn move2(&mut self, front_back: f64, left_right: f64) {
        let d = DVec3::new(
            (-self.rotation.y).sin() as f64,
            0.0,
            (-self.rotation.y).cos() as f64,
        );
        self.position -= d * front_back;
        self.position -= d.cross(&DVec3::new(0.0, 1.0, 0.0)).normalize() * left_right;
    }
}

#[derive(Copy, Clone)]
pub struct OrthoCamera {
    position: DVec3,
    rotation: Vec3,
    z_near: f32,
    z_far: f32,
}

impl OrthoCamera {
    pub fn new() -> Self {
        Self {
            position: Default::default(),
            rotation: Default::default(),
            z_near: 0.0,
            z_far: 1024.0,
        }
    }

    pub fn projection(&self) -> Mat4 {
        // Note: using reversed-z
        glm::ortho_lh_zo(0.0, 1.0, 0.0, 1.0, self.z_far, self.z_near)
    }

    pub fn position(&self) -> &DVec3 {
        &self.position
    }

    pub fn z_near(&self) -> f32 {
        self.z_near
    }

    pub fn set_position(&mut self, position: DVec3) {
        self.position = position;
    }

    pub fn rotation(&self) -> &Vec3 {
        &self.rotation
    }

    pub fn direction(&self) -> Vec3 {
        rotation_to_direction(&self.rotation)
    }

    pub fn set_rotation(&mut self, rotation: Vec3) {
        self.rotation = rotation.map(|x| x % (std::f32::consts::PI * 2.0));
    }
}

impl Default for OrthoCamera {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn new(proj_view_mat: &Mat4) -> Self {
        let mut frustum = Self {
            planes: [Vec4::default(); 6],
        };

        for i in 0..6 {
            let plane = &mut frustum.planes[i];
            let sign = ((1 - i as i32 % 2) * 2 - 1) as f32;

            for j in 0..4 {
                plane[j] = proj_view_mat.column(j)[3] + (proj_view_mat.column(j)[i / 2] * sign);
            }

            *plane /= (*plane).fixed_rows::<3>(0).magnitude();
        }

        frustum
    }

    pub fn is_sphere_visible(&self, pos: &Vec3, radius: f32) -> bool {
        for i in 0..6 {
            if self.planes[i].x * pos.x
                + self.planes[i].y * pos.y
                + self.planes[i].z * pos.z
                + self.planes[i].w
                <= -radius
            {
                return false;
            }
        }

        true
    }
}

/// Calculates movement delta according to camera orientation.
pub fn move_xz(rotation: Vec3, front_back: f64, left_right: f64) -> DVec3 {
    let d = DVec3::new((-rotation.y).sin() as f64, 0.0, (-rotation.y).cos() as f64);
    let mut motion_delta = -d * front_back;
    motion_delta -= d.cross(&DVec3::new(0.0, 1.0, 0.0)).normalize() * left_right;
    motion_delta
}
