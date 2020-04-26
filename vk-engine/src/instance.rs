use crate::adapter::Adapter;
use crate::{format, utils, Format};
use ash::version::{InstanceV1_0, InstanceV1_1};
use ash::vk;
use std::collections::HashMap;
use std::mem;
use std::os::raw::c_void;

pub struct Instance {
    pub(crate) native: ash::Instance,
    pub(crate) debug_utils_ext: Option<ash::extensions::ext::DebugUtils>,
    pub(crate) debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub(crate) surface_khr: Option<ash::extensions::khr::Surface>,
}

impl Instance {
    pub fn create_vk_surface(&mut self, window: &windows::Window) -> Result<vk::SurfaceKHR, vk::Result> {
        let instance_handle: usize = unsafe { mem::transmute(self.native.handle()) };
        let mut surface_handle: u64 = 0;

        let result: vk::Result =
            unsafe { mem::transmute(window.create_surface(instance_handle, &mut surface_handle)) };
        let vk_surface: vk::SurfaceKHR = unsafe { mem::transmute(surface_handle) };

        if result == vk::Result::SUCCESS {
            Ok(vk_surface)
        } else {
            Err(result)
        }
    }

    fn enumerate_device_extension_names(
        &self,
        p_device: vk::PhysicalDevice,
    ) -> Result<Vec<String>, vk::Result> {
        Ok(
            unsafe { self.native.enumerate_device_extension_properties(p_device) }?
                .iter()
                .map(|ext| unsafe { utils::c_ptr_to_string(ext.extension_name.as_ptr()) })
                .collect(),
        )
    }

    pub fn enumerate_adapters(&self, surface: vk::SurfaceKHR) -> Result<Vec<Adapter>, vk::Result> {
        let physical_devices = unsafe { self.native.enumerate_physical_devices()? };

        Ok(physical_devices
            .iter()
            .filter_map(|&p_device| {
                // Check queue families
                // ------------------------------------------------------------------------------------
                let queue_families =
                    unsafe { self.native.get_physical_device_queue_family_properties(p_device) };
                let mut family_indices: [[u8; 2]; 4] = [[u8::MAX, 0]; 4];

                for (i, fam_prop) in queue_families.iter().enumerate() {
                    // Check for present usage
                    let surface_supported = unsafe {
                        self.surface_khr
                            .as_ref()
                            .unwrap()
                            .get_physical_device_surface_support(p_device, i as u32, surface)
                            .unwrap()
                    };

                    if surface_supported && family_indices[3][0] == u8::MAX {
                        family_indices[3] = [i as u8, 0]; // present
                    }
                    if (fam_prop.queue_flags & vk::QueueFlags::GRAPHICS) == vk::QueueFlags::GRAPHICS
                        && family_indices[0][0] == u8::MAX
                    {
                        family_indices[0] = [i as u8, 0]; // graphics
                    } else if (fam_prop.queue_flags & vk::QueueFlags::COMPUTE) == vk::QueueFlags::COMPUTE
                        && family_indices[1][0] == u8::MAX
                    {
                        family_indices[1] = [i as u8, 0]; // compute
                    } else if (fam_prop.queue_flags & vk::QueueFlags::TRANSFER) == vk::QueueFlags::TRANSFER
                        && family_indices[2][0] == u8::MAX
                    {
                        family_indices[2] = [i as u8, 0]; // transfer
                    }

                    if family_indices[0][0] != u8::MAX
                        && family_indices[1][0] != u8::MAX
                        && family_indices[2][0] != u8::MAX
                        && family_indices[3][0] != u8::MAX
                    {
                        break;
                    }
                }

                if family_indices[1][0] == u8::MAX {
                    family_indices[1] = family_indices[0]; // compute -> graphics
                }
                if family_indices[2][0] == u8::MAX {
                    family_indices[2] = family_indices[1]; // transfer -> compute
                }

                // The same queue(graphics) is used for rest if separate queues not available
                if family_indices[0][0] == u8::MAX || family_indices[3][0] == u8::MAX {
                    return None;
                }

                // Check extensions
                // ------------------------------------------------------------------------------------
                let available_extensions = self.enumerate_device_extension_names(p_device).unwrap();
                let required_extensions = ["VK_KHR_swapchain"];
                let preferred_extensions = ["VK_EXT_memory_budget"];

                let enabled_extensions_res =
                    utils::filter_names(&available_extensions, &required_extensions, true);
                if enabled_extensions_res.is_err() {
                    return None;
                }
                let mut enabled_extensions = enabled_extensions_res.unwrap();
                enabled_extensions.extend(
                    utils::filter_names(&available_extensions, &preferred_extensions, false).unwrap(),
                );

                // Get properties
                // ------------------------------------------------------------------------------------
                let mut props12 = vk::PhysicalDeviceVulkan12Properties::builder();
                let mut props2 = vk::PhysicalDeviceProperties2::builder().push_next(&mut props12);
                unsafe { self.native.get_physical_device_properties2(p_device, &mut props2) };

                let api_version = props2.properties.api_version;
                if vk::version_major(api_version) != 1 || vk::version_minor(api_version) < 2 {
                    return None;
                }

                // Check features
                // ------------------------------------------------------------------------------------
                let mut available_features12 = vk::PhysicalDeviceVulkan12Features::builder().build();
                let mut available_features2 = vk::PhysicalDeviceFeatures2 {
                    p_next: &mut available_features12 as *mut vk::PhysicalDeviceVulkan12Features
                        as *mut c_void,
                    ..Default::default()
                };
                unsafe {
                    self.native
                        .get_physical_device_features2(p_device, &mut available_features2);
                }
                let available_features = available_features2.features;

                let mut enabled_features12 = vk::PhysicalDeviceVulkan12Features::builder().build();
                let mut enabled_features = vk::PhysicalDeviceFeatures::builder().build();

                macro_rules! require_feature {
                    ($name:ident) => {
                        if available_features.$name == 0 {
                            return None;
                        } else {
                            enabled_features.$name = available_features.$name;
                        }
                    };
                }
                macro_rules! require_feature12 {
                    ($name:ident) => {
                        if available_features12.$name == 0 {
                            return None;
                        } else {
                            enabled_features12.$name = available_features12.$name;
                        }
                    };
                }

                require_feature!(sampler_anisotropy);
                require_feature!(shader_uniform_buffer_array_dynamic_indexing);
                require_feature!(shader_sampled_image_array_dynamic_indexing);
                require_feature!(shader_storage_buffer_array_dynamic_indexing);
                require_feature!(shader_storage_image_array_dynamic_indexing);

                require_feature12!(descriptor_indexing);
                require_feature12!(shader_uniform_buffer_array_non_uniform_indexing);
                require_feature12!(shader_sampled_image_array_non_uniform_indexing);
                require_feature12!(shader_storage_buffer_array_non_uniform_indexing);
                require_feature12!(shader_storage_image_array_non_uniform_indexing);
                require_feature12!(descriptor_binding_uniform_buffer_update_after_bind);
                require_feature12!(descriptor_binding_sampled_image_update_after_bind);
                require_feature12!(descriptor_binding_storage_image_update_after_bind);
                require_feature12!(descriptor_binding_storage_buffer_update_after_bind);
                require_feature12!(descriptor_binding_partially_bound);
                require_feature12!(runtime_descriptor_array);
                require_feature12!(separate_depth_stencil_layouts);

                enabled_features.robust_buffer_access =
                    available_features.robust_buffer_access & props12.robust_buffer_access_update_after_bind;

                enabled_features12.vulkan_memory_model = available_features12.vulkan_memory_model;
                enabled_features12.vulkan_memory_model_device_scope =
                    available_features12.vulkan_memory_model_device_scope;
                enabled_features12.vulkan_memory_model_availability_visibility_chains =
                    available_features12.vulkan_memory_model_availability_visibility_chains;

                // Check formats
                // ------------------------------------------------------------------------------------
                for format in format::BufferFormats.iter() {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format.0)
                    };
                    let flags = props.buffer_features;

                    if (flags & vk::FormatFeatureFlags::VERTEX_BUFFER)
                        != vk::FormatFeatureFlags::VERTEX_BUFFER
                        || (flags & vk::FormatFeatureFlags::TRANSFER_SRC)
                            != vk::FormatFeatureFlags::TRANSFER_SRC
                        || (flags & vk::FormatFeatureFlags::TRANSFER_DST)
                            != vk::FormatFeatureFlags::TRANSFER_DST
                    {
                        return None;
                    }
                }

                let image_format_features = vk::FormatFeatureFlags::COLOR_ATTACHMENT
                    | vk::FormatFeatureFlags::SAMPLED_IMAGE
                    | vk::FormatFeatureFlags::STORAGE_IMAGE
                    | vk::FormatFeatureFlags::BLIT_SRC
                    | vk::FormatFeatureFlags::BLIT_DST
                    | vk::FormatFeatureFlags::TRANSFER_SRC
                    | vk::FormatFeatureFlags::TRANSFER_DST;
                let depth_format_features = vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::FormatFeatureFlags::SAMPLED_IMAGE
                    | vk::FormatFeatureFlags::BLIT_SRC
                    | vk::FormatFeatureFlags::BLIT_DST
                    | vk::FormatFeatureFlags::TRANSFER_SRC
                    | vk::FormatFeatureFlags::TRANSFER_DST;

                for format in format::ImageFormats.iter() {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format.0)
                    };
                    if (props.optimal_tiling_features & image_format_features) != image_format_features
                        && (props.linear_tiling_features & image_format_features) != image_format_features
                    {
                        return None;
                    }
                }

                {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format::DepthFormat.0)
                    };
                    if (props.optimal_tiling_features & depth_format_features) != depth_format_features
                        && (props.linear_tiling_features & depth_format_features) != depth_format_features
                    {
                        return None;
                    }
                }

                Some(Adapter { native: p_device })
            })
            .collect())
    }

    pub fn destroy_surface(&self, surface: vk::SurfaceKHR) {
        unsafe {
            self.surface_khr.as_ref().unwrap().destroy_surface(surface, None);
        }
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.debug_utils_ext
                .as_ref()
                .unwrap()
                .destroy_debug_utils_messenger(self.debug_utils_messenger.unwrap(), None);
            self.native.destroy_instance(None);
        }
    }
}
