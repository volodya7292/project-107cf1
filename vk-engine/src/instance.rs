use crate::adapter::Adapter;
use crate::{format, surface::Surface, utils, Entry};
use ash::version::{InstanceV1_0, InstanceV1_1};
use ash::vk;
use std::{os::raw::c_void, rc::Rc};

pub struct Instance {
    pub(crate) entry: Rc<Entry>,
    pub(crate) native: ash::Instance,
    pub(crate) debug_utils_ext: Option<ash::extensions::ext::DebugUtils>,
    pub(crate) debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub(crate) surface_khr: ash::extensions::khr::Surface,
}

pub fn enumerate_required_window_extensions(
    window_handle: &impl raw_window_handle::HasRawWindowHandle,
) -> Result<Vec<String>, vk::Result> {
    let names = ash_window::enumerate_required_extensions(window_handle)?;
    Ok(names
        .iter()
        .map(|&name| unsafe { utils::c_ptr_to_string(name.as_ptr()) })
        .collect())
}

impl Instance {
    pub fn handle(&self) -> vk::Instance {
        self.native.handle()
    }

    pub fn create_surface(
        self: &Rc<Self>,
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Rc<Surface>, vk::Result> {
        Ok(Rc::new(Surface {
            instance: Rc::clone(self),
            native: unsafe {
                ash_window::create_surface(&self.entry.ash_entry, &self.native, window_handle, None)?
            },
        }))
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

    pub fn enumerate_adapters(
        self: &Rc<Self>,
        surface: &Rc<Surface>,
    ) -> Result<Vec<Rc<Adapter>>, vk::Result> {
        let physical_devices = unsafe { self.native.enumerate_physical_devices()? };

        Ok(physical_devices
            .iter()
            .filter_map(|&p_device| {
                // Check queue families
                // ------------------------------------------------------------------------------------
                let queue_families =
                    unsafe { self.native.get_physical_device_queue_family_properties(p_device) };
                let mut queue_fam_indices: [[u32; 2]; 4] = [[u32::MAX, 0]; 4];

                for (i, fam_prop) in queue_families.iter().enumerate() {
                    // Check for present usage

                    // TODO: surface.get_physical_device_surface_support

                    let surface_supported = unsafe {
                        self.surface_khr
                            .get_physical_device_surface_support(p_device, i as u32, surface.native)
                            .unwrap()
                    };

                    if surface_supported && queue_fam_indices[3][0] == u32::MAX {
                        queue_fam_indices[3] = [i as u32, 0]; // present
                    }
                    if fam_prop.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && queue_fam_indices[0][0] == u32::MAX
                    {
                        queue_fam_indices[0] = [i as u32, 0]; // graphics
                    } else if fam_prop.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && queue_fam_indices[1][0] == u32::MAX
                    {
                        queue_fam_indices[1] = [i as u32, 0]; // compute
                    } else if fam_prop.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && queue_fam_indices[2][0] == u32::MAX
                    {
                        queue_fam_indices[2] = [i as u32, 0]; // transfer
                    }

                    if queue_fam_indices[0][0] != u32::MAX
                        && queue_fam_indices[1][0] != u32::MAX
                        && queue_fam_indices[2][0] != u32::MAX
                        && queue_fam_indices[3][0] != u32::MAX
                    {
                        break;
                    }
                }

                if queue_fam_indices[1][0] == u32::MAX {
                    queue_fam_indices[1] = queue_fam_indices[0]; // compute -> graphics
                }
                if queue_fam_indices[2][0] == u32::MAX {
                    queue_fam_indices[2] = queue_fam_indices[1]; // transfer -> compute
                }

                // The same queue(graphics) is used for rest if separate queues not available
                if queue_fam_indices[0][0] == u32::MAX || queue_fam_indices[3][0] == u32::MAX {
                    return None;
                }

                // Check extensions
                // ------------------------------------------------------------------------------------
                let available_extensions = self.enumerate_device_extension_names(p_device).unwrap();
                let required_extensions = ["VK_KHR_swapchain"];
                let preferred_extensions = [""];

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
                let props = props2.properties;

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
                        .get_physical_device_features2(p_device, &mut available_features2)
                };
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
                require_feature12!(timeline_semaphore);

                enabled_features.robust_buffer_access =
                    available_features.robust_buffer_access & props12.robust_buffer_access_update_after_bind;

                enabled_features12.vulkan_memory_model = available_features12.vulkan_memory_model;
                enabled_features12.vulkan_memory_model_device_scope =
                    available_features12.vulkan_memory_model_device_scope;
                enabled_features12.vulkan_memory_model_availability_visibility_chains =
                    available_features12.vulkan_memory_model_availability_visibility_chains;

                // Check formats
                // ------------------------------------------------------------------------------------
                // Buffer formats
                for format in format::BUFFER_FORMATS.iter() {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format.0)
                    };
                    let flags = props.buffer_features;

                    if !flags.contains(
                        vk::FormatFeatureFlags::VERTEX_BUFFER
                            | vk::FormatFeatureFlags::TRANSFER_SRC
                            | vk::FormatFeatureFlags::TRANSFER_DST,
                    ) {
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

                // Image formats
                for format in format::IMAGE_FORMATS.iter() {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format.0)
                    };
                    if !props.optimal_tiling_features.contains(image_format_features) {
                        return None;
                    }
                }

                // Depth format
                {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format::DEPTH_FORMAT.0)
                    };
                    if !props.optimal_tiling_features.contains(depth_format_features) {
                        return None;
                    }
                }

                Some(Rc::new(Adapter {
                    instance: Rc::clone(self),
                    native: p_device,
                    props,
                    enabled_extensions,
                    props12: props12.build(),
                    features: enabled_features,
                    features12: enabled_features12,
                    queue_family_indices: queue_fam_indices,
                }))
            })
            .collect())
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
        };
    }
}
