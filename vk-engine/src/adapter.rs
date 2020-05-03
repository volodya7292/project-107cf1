use ash::vk;
use std::ffi::CString;

#[derive(Clone)]
pub struct Adapter {
    pub(crate) native: vk::PhysicalDevice,
    pub(crate) props: vk::PhysicalDeviceProperties,
    pub(crate) enabled_extensions: Vec<CString>,
    pub(crate) props12: vk::PhysicalDeviceVulkan12Properties,
    pub(crate) features: vk::PhysicalDeviceFeatures,
    pub(crate) features12: vk::PhysicalDeviceVulkan12Features,
    pub(crate) queue_family_indices: [[u8; 2]; 4],
}

impl Adapter {
    pub fn is_extension_enabled(&self, name: &str) -> bool {
        self.enabled_extensions.contains(&CString::new(name).unwrap())
    }
}
