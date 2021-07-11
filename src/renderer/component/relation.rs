pub struct Parent(pub(in crate::renderer) u32);

#[derive(Default)]
pub struct Children(pub(super) Vec<u32>);

impl Children {
    pub fn get(&self) -> &[u32] {
        &self.0
    }
}
