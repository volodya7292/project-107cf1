use crate::render_engine::scene::Entity;
use crate::utils::IndexSet;

pub struct Parent(pub(in crate::render_engine) Entity);

#[derive(Default)]
pub struct Children {
    pub children: IndexSet<Entity>,
}
