use std::sync::Arc;

pub struct MaterialPipeline {}

pub fn new() -> Arc<MaterialPipeline> {
    Arc::new(MaterialPipeline {})
}
