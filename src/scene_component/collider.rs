use std::sync::Arc;

pub trait CustomCollider {}

pub enum Collider {
    Custom(Arc<dyn CustomCollider>),
}
