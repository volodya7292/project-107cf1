use common::{shader_compiler::ShaderVariantConfig, types::HashSet};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref DEPTH_SOLID: ShaderVariantConfig =
        ShaderVariantConfig::new(vec!["RENDER_DEPTH_ONLY".to_string()]);
    pub static ref DEPTH_TRANSPARENT: ShaderVariantConfig = ShaderVariantConfig::new(vec![
        "RENDER_DEPTH_ONLY".to_string(),
        "RENDER_CLOSEST_DEPTHS".to_string()
    ]);
    pub static ref GBUFFER_SOLID: ShaderVariantConfig = ShaderVariantConfig::new(vec![]);
    pub static ref GBUFFER_TRANSPARENT: ShaderVariantConfig =
        ShaderVariantConfig::new(vec!["RENDER_GBUFFER_TRANSPARENCY".to_string()]);
    pub static ref GBUFFER_OVERLAY: ShaderVariantConfig = ShaderVariantConfig::new(vec![]);
    pub static ref ALL: HashSet<ShaderVariantConfig> = [
        &*DEPTH_SOLID,
        &*DEPTH_TRANSPARENT,
        &*GBUFFER_SOLID,
        &*GBUFFER_TRANSPARENT,
        &*GBUFFER_OVERLAY
    ]
    .into_iter()
    .cloned()
    .collect();
}
