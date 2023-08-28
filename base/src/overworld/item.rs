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
