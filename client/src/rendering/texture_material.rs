use common::glm::Vec4;
use engine::module::main_renderer::material::MatComponent;
use engine::module::main_renderer::{MaterialInfo, TEXTURE_ID_NONE};

const TEXTURE_ID_NAME: u16 = u16::MAX;

pub struct TextureMaterial {
    diffuse: MatComponent,
    specular: MatComponent,
    normal_tex_id: u16,
    emission: Vec4,
    translucent: bool,
}

impl TextureMaterial {
    pub fn new(diffuse: MatComponent) -> TextureMaterial {
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

    pub fn info(&self) -> MaterialInfo {
        MaterialInfo::new(self.diffuse, self.specular, self.normal_tex_id, self.emission)
    }
}
