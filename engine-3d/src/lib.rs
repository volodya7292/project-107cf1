pub use specs;
pub use specs::Builder;
pub use specs::Entities;
pub use specs::WorldExt;

pub mod scene;

pub trait Entry {
    fn pri();
}
