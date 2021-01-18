pub struct Parent(pub(in crate::renderer) u32);

#[derive(Default)]
pub struct Children(pub Vec<u32>);
