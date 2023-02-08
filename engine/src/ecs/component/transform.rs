use nalgebra_glm::{DVec3, Vec3};

#[derive(Copy, Clone)]
pub struct TransformC {
    // Note: use DVec3 to support large distances
    pub position: DVec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub use_parent_transform: bool,
}

impl Default for TransformC {
    fn default() -> Self {
        Self {
            position: Default::default(),
            rotation: Default::default(),
            scale: Vec3::from_element(1.0),
            use_parent_transform: true,
        }
    }
}

impl TransformC {
    pub fn new() -> TransformC {
        Default::default()
    }

    pub fn with_position(mut self, position: DVec3) -> Self {
        self.position = position;
        self
    }

    pub fn with_rotation(mut self, rotation: Vec3) -> Self {
        self.rotation = rotation;
        self
    }

    pub fn with_scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    pub fn with_use_parent_transform(mut self, use_parent_transform: bool) -> Self {
        self.use_parent_transform = use_parent_transform;
        self
    }

    pub fn position(&self) -> &DVec3 {
        &self.position
    }

    pub fn rotation(&self) -> &Vec3 {
        &self.rotation
    }

    pub fn scale(&self) -> &Vec3 {
        &self.scale
    }
}
