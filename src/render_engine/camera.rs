use nalgebra::{Isometry3, Rotation3, Translation3};
use nalgebra_glm as glm;
use nalgebra_glm::{Mat4, Vec3, Vec4};

#[derive(Copy, Clone)]
pub struct Camera {
    position: Vec3,
    rotation: Vec3,
    aspect: f32,
    fovy: f32,
    z_near: f32,
    frustum: [Vec4; 6],
}

impl Camera {
    fn calc_frustum(proj_view_mat: &Mat4) -> [Vec4; 6] {
        let mut frustum = [Vec4::default(); 6];

        for i in 0..6 {
            let plane = &mut frustum[i];
            let sign = ((1 - i as i32 % 2) * 2 - 1) as f32;

            for j in 0..4 {
                plane[j] = proj_view_mat.column(j)[3] + (proj_view_mat.column(j)[i / 2] * sign);
            }

            *plane /= (*plane).fixed_rows::<3>(0).magnitude();
        }

        frustum
    }

    pub fn new(aspect: f32, fovy: f32, near: f32) -> Camera {
        let projection = glm::infinite_perspective_rh_zo(aspect, fovy, near);
        let view = Isometry3::identity();
        let proj_view_mat = projection * view.to_homogeneous();
        let frustum = Self::calc_frustum(&proj_view_mat);

        Camera {
            position: Vec3::default(),
            rotation: Vec3::new(0.0, 0.0, 0.0),
            aspect,
            fovy,
            z_near: near,
            frustum,
        }
    }

    pub fn update_frustum(&mut self) {
        let proj_view_mat = self.projection() * self.view();
        self.frustum = Self::calc_frustum(&proj_view_mat);
    }

    pub fn projection(&self) -> Mat4 {
        glm::infinite_perspective_rh_zo(self.aspect, self.fovy, self.z_near)
    }

    pub fn view(&self) -> Mat4 {
        let mut mat = Mat4::identity();
        mat *= Rotation3::from_axis_angle(&Vec3::x_axis(), self.rotation.x).to_homogeneous();
        mat *= Rotation3::from_axis_angle(&Vec3::y_axis(), self.rotation.y).to_homogeneous();
        mat *= Rotation3::from_axis_angle(&Vec3::z_axis(), self.rotation.z).to_homogeneous();
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
        self.update_frustum();
    }

    pub fn position(&self) -> Vec3 {
        self.position
    }

    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
        self.update_frustum();
    }

    pub fn rotation(&self) -> Vec3 {
        self.rotation.clone()
    }

    pub fn set_rotation(&mut self, rotation: Vec3) {
        self.rotation = rotation.map(|x| x % (std::f32::consts::PI * 2.0));
        self.update_frustum();
    }

    pub fn direction(&self) -> Vec3 {
        let view = self.view();
        let identity = Vec4::new(0.0, 0.0, 1.0, 1.0);
        let dir = view * identity;
        Vec3::from(dir.fixed_rows::<3>(0))
    }

    pub fn fovy(&self) -> f32 {
        self.fovy
    }

    pub fn move2(&mut self, front_back: f32, left_right: f32) {
        let d = Vec3::new((-self.rotation.y).sin(), 0.0, (-self.rotation.y).cos());
        self.position -= d * front_back;
        self.position -= d.cross(&Vec3::new(0.0, 1.0, 0.0)).normalize() * left_right;
        self.update_frustum();
    }

    pub fn is_sphere_visible(&self, pos: &Vec3, radius: f32) -> bool {
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
