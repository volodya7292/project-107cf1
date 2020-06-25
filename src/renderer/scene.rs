use crate::renderer::component;
use specs;
use specs::Builder;
use specs::Entities;
use specs::WorldExt;

pub struct Entity(specs::Entity);

pub struct Scene {
    pub(crate) world: specs::World,
}

impl Scene {
    pub fn create_entity(&mut self) -> specs::EntityBuilder {
        self.world.create_entity()
    }
}

pub fn new() -> Scene {
    let mut world = specs::World::new();
    world.register::<component::Transform>();
    world.register::<component::VertexMeshRef>();
    world.register::<component::Renderer>();

    Scene { world }
}
