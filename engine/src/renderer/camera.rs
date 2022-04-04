use nalgebra::{Isometry3, Rotation3, Translation3};
use nalgebra_glm as glm;
use nalgebra_glm::{DMat4, DVec3, DVec4, Mat4, Vec3, Vec4};

#[derive(Copy, Clone)]
pub struct Camera {
    position: DVec3,
    rotation: Vec3,
    aspect: f32,
    fovy: f32,
    z_near: f32,
}

pub fn create_view_matrix(position: Vec3, rotation: Vec3) -> Mat4 {
    let mut mat = Mat4::identity();
    mat *= Rotation3::from_axis_angle(&Vec3::x_axis(), rotation.x).to_homogeneous();
    mat *= Rotation3::from_axis_angle(&Vec3::y_axis(), rotation.y).to_homogeneous();
    mat *= Rotation3::from_axis_angle(&Vec3::z_axis(), rotation.z).to_homogeneous();
    mat *= Translation3::from(-position).to_homogeneous();
    mat
}

impl Camera {
    pub fn new(aspect: f32, fovy: f32, near: f32) -> Camera {
        let projection = glm::infinite_perspective_rh_zo(aspect, fovy, near);
        let view = Isometry3::identity();
        let proj_view_mat = projection * view.to_homogeneous();

        Camera {
            position: DVec3::default(),
            rotation: Vec3::new(0.0, 0.0, 0.0),
            aspect,
            fovy,
            z_near: near,
        }
    }

    pub fn projection(&self) -> Mat4 {
        glm::infinite_perspective_rh_zo(self.aspect, self.fovy, self.z_near)
    }

    pub fn view(&self) -> DMat4 {
        let mut mat = DMat4::identity();
        mat *= Rotation3::from_axis_angle(&DVec3::x_axis(), self.rotation.x as f64).to_homogeneous();
        mat *= Rotation3::from_axis_angle(&DVec3::y_axis(), self.rotation.y as f64).to_homogeneous();
        mat *= Rotation3::from_axis_angle(&DVec3::z_axis(), self.rotation.z as f64).to_homogeneous();
        mat *= Translation3::from(-self.position).to_homogeneous();
        mat
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

    pub fn position(&self) -> DVec3 {
        self.position
    }

    pub fn set_position(&mut self, position: DVec3) {
        self.position = position;
    }

    pub fn rotation(&self) -> Vec3 {
        self.rotation.clone()
    }

    pub fn set_rotation(&mut self, rotation: Vec3) {
        self.rotation = rotation.map(|x| x % (std::f32::consts::PI * 2.0));
    }

    pub fn direction(&self) -> Vec3 {
        let view = self.view();
        let identity = DVec4::new(0.0, 0.0, 1.0, 1.0);
        let dir = view * identity;
        glm::convert(DVec3::from(dir.fixed_rows::<3>(0)))
    }

    pub fn fovy(&self) -> f32 {
        self.fovy
    }

    pub fn move2(&mut self, front_back: f64, left_right: f64) {
        let d = DVec3::new(
            (-self.rotation.y).sin() as f64,
            0.0,
            (-self.rotation.y).cos() as f64,
        );
        self.position -= d * (front_back as f64);
        self.position -= d.cross(&DVec3::new(0.0, 1.0, 0.0)).normalize() * (left_right) as f64;
    }
}

pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn new(proj_view_mat: Mat4) -> Self {
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

        return true;
    }
}
