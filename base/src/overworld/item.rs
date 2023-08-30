pub struct Item {
    on_place: fn(),
    on_destruct: fn(),
}

impl Item {
    pub fn new() -> Self {
        Self {
            on_place: || {},
            on_destruct: || {},
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct ItemStack {
    item_id: u32,
    count: u32,
}
