use common::glm::Vec4;

#[derive(Copy, Clone)]
pub enum MatComponent {
    Texture(u16),
    Color(Vec4),
}

const TEXTURE_ID_NONE: u16 = u16::MAX;

pub struct Material {
    diffuse: MatComponent,
    specular: MatComponent,
    normal_tex_id: u16,
    emission: Vec4,
    translucent: bool,
}

impl Material {
    pub fn new(diffuse: MatComponent) -> Material {
        Self {
            diffuse,
            specular: MatComponent::Color(Default::default()),
            normal_tex_id: TEXTURE_ID_NONE,
            emission: Default::default(),
            translucent: false,
        }
    }

    pub fn with_specular(mut self, specular: MatComponent) -> Self {
        self.specular = specular;
        self
    }

    pub fn with_normal_tex(mut self, tex_id: u16) -> Self {
        self.normal_tex_id = tex_id;
        self
    }

    pub fn with_emission(mut self, emission: Vec4) -> Self {
        self.emission = emission;
        self
    }

    pub fn with_translucent(mut self, translucent: bool) -> Self {
        self.translucent = translucent;
        self
    }

    pub fn translucent(&self) -> bool {
        self.translucent
    }

    // pub fn info(&self) -> MaterialInfo {
    //     MaterialInfo::new(self.diffuse, self.specular, self.normal_tex_id, self.emission)
    // }
}
