use common::{
    shader_compiler::{self, ShaderVariantConfig},
    types::HashMap,
};
use std::sync::Arc;
use vk_wrapper::{Device, DeviceError, Format, Shader, shader::VInputRate};

pub struct VkwShaderBundle {
    pub variants: HashMap<ShaderVariantConfig, Arc<Shader>>,
}

pub trait VkwShaderBundleDeviceExt {
    fn load_vertex_shader_bundle(
        self: &Arc<Self>,
        bundle_data: &[u8],
        input_formats: &[(&str, Format, VInputRate)],
        name: &str,
    ) -> Result<Arc<VkwShaderBundle>, DeviceError>;
    fn load_pixel_shader_bundle(
        self: &Arc<Self>,
        bundle_data: &[u8],
        name: &str,
    ) -> Result<Arc<VkwShaderBundle>, DeviceError>;
}

impl VkwShaderBundleDeviceExt for Device {
    fn load_vertex_shader_bundle(
        self: &Arc<Self>,
        bundle_data: &[u8],
        input_formats: &[(&str, Format, VInputRate)],
        name: &str,
    ) -> Result<Arc<VkwShaderBundle>, DeviceError> {
        let bundle = shader_compiler::read_shader_bundle(bundle_data);
        let variants: Result<HashMap<ShaderVariantConfig, Arc<Shader>>, DeviceError> = bundle
            .variants
            .iter()
            .map(|(config, code)| {
                self.create_vertex_shader(code, input_formats, &format!("{}-{}", name, config.uid()))
                    .map(|shader| (config.clone(), shader))
            })
            .collect();

        Ok(Arc::new(VkwShaderBundle { variants: variants? }))
    }

    fn load_pixel_shader_bundle(
        self: &Arc<Self>,
        bundle_data: &[u8],
        name: &str,
    ) -> Result<Arc<VkwShaderBundle>, DeviceError> {
        let bundle = shader_compiler::read_shader_bundle(bundle_data);
        let variants: Result<HashMap<ShaderVariantConfig, Arc<Shader>>, DeviceError> = bundle
            .variants
            .iter()
            .map(|(config, code)| {
                self.create_pixel_shader(code, &format!("{}-{}", name, config.uid()))
                    .map(|shader| (config.clone(), shader))
            })
            .collect();

        Ok(Arc::new(VkwShaderBundle { variants: variants? }))
    }
}
