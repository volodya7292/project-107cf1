pub struct Parent(pub(in crate::render_engine) u32);

#[derive(Default)]
pub struct Children(pub(super) Vec<u32>);

impl Children {
    pub fn get(&self) -> &[u32] {
        &self.0
    }
}
