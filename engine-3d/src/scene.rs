use specs;
use specs::Builder;
use specs::Entities;
use specs::WorldExt;

pub struct Entity(specs::Entity);

pub struct Scene {
    world: specs::World,
}

pub struct PosComponent(f32);

impl specs::Component for PosComponent {
    type Storage = specs::VecStorage<Self>;
}

impl Scene {
    pub fn register<T: specs::Component>(&mut self)
    where
        T::Storage: Default,
    {
        self.world.register::<T>();
    }

    pub fn create_entity(&mut self) -> specs::EntityBuilder {
        self.world.create_entity()
    }
}

pub fn new() -> Scene {
    let mut world = specs::World::new();
    world.register::<PosComponent>();

    Scene { world }
}
