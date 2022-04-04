#[derive(Copy, Clone)]
pub struct Block {
    archetype: u16,
    textured_model: u16,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            archetype: u16::MAX,
            textured_model: u16::MAX,
        }
    }
}

impl Block {
    pub fn new(archetype: u16, textured_model: u16) -> Block {
        Block {
            archetype,
            textured_model,
        }
    }

    pub fn archetype(&self) -> u16 {
        self.archetype
    }

    pub fn textured_model(&self) -> u16 {
        self.textured_model
    }

    pub fn has_textured_model(&self) -> bool {
        self.textured_model != u16::MAX
    }
}
