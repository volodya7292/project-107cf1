use crate::adapter::Adapter;
use crate::FORMAT_SIZES;
use crate::{format, surface::Surface, utils, Entry};
use ash::vk;
use std::sync::Arc;
use std::{collections::HashMap, os::raw::c_void};

pub(crate) const VK_API_VERSION: u32 = vk::API_VERSION_1_0;

pub struct Instance {
    pub(crate) entry: Arc<Entry>,
    pub(crate) native: ash::Instance,
    pub(crate) debug_utils_ext: Option<ash::extensions::ext::DebugUtils>,
    pub(crate) debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub(crate) surface_khr: ash::extensions::khr::Surface,
}

impl Instance {
    pub fn create_surface(
        self: &Arc<Self>,
        window_handle: &impl raw_window_handle::HasRawWindowHandle,
    ) -> Result<Arc<Surface>, vk::Result> {
        Ok(Arc::new(Surface {
            instance: Arc::clone(self),
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
        self: &Arc<Self>,
        surface: Option<&Arc<Surface>>,
    ) -> Result<Vec<Arc<Adapter>>, vk::Result> {
        let physical_devices = unsafe { self.native.enumerate_physical_devices()? };
        let mut adapters = Vec::<Arc<Adapter>>::new();

        'g: for p_device in physical_devices {
            let props = unsafe { self.native.get_physical_device_properties(p_device) };

            if vk::api_version_major(props.api_version) != vk::api_version_major(VK_API_VERSION)
                || vk::api_version_minor(props.api_version) < vk::api_version_minor(VK_API_VERSION)
            {
                continue;
            }

            // Check queue families
            // ------------------------------------------------------------------------------------
            let queue_families = unsafe { self.native.get_physical_device_queue_family_properties(p_device) };
            let mut queue_fam_indices: [[u32; 2]; 4] = [[u32::MAX, 0]; 4];

            for (i, fam_prop) in queue_families.iter().enumerate() {
                // Check for present usage

                let surface_supported = if let Some(surface) = surface {
                    unsafe {
                        self.surface_khr
                            .get_physical_device_surface_support(p_device, i as u32, surface.native)
                            .unwrap()
                    }
                } else {
                    false
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

            // The same queue(graphics) is used for the rest if separate queues not available
            if queue_fam_indices[1][0] == u32::MAX {
                queue_fam_indices[1] = queue_fam_indices[0]; // compute -> graphics
            }
            if queue_fam_indices[2][0] == u32::MAX {
                queue_fam_indices[2] = queue_fam_indices[1]; // transfer -> compute
            }

            if queue_fam_indices[0][0] == u32::MAX
                || (queue_fam_indices[3][0] == u32::MAX && surface.is_some())
            {
                continue;
            }

            // Check extensions
            // ------------------------------------------------------------------------------------
            let available_extensions = self.enumerate_device_extension_names(p_device).unwrap();
            let required_extensions = [
                "VK_KHR_swapchain",
                "VK_KHR_timeline_semaphore",
                "VK_KHR_8bit_storage",
                "VK_KHR_shader_float16_int8",
                "VK_EXT_descriptor_indexing",
                "VK_EXT_scalar_block_layout",
            ];
            let preferred_extensions = ["VK_KHR_portability_subset"];

            let enabled_extensions_res =
                utils::filter_names(&available_extensions, &required_extensions, true);
            if enabled_extensions_res.is_err() {
                continue;
            }
            let mut enabled_extensions = enabled_extensions_res.unwrap();
            enabled_extensions
                .extend(utils::filter_names(&available_extensions, &preferred_extensions, false).unwrap());

            // Check features
            // ------------------------------------------------------------------------------------
            let mut available_scalar_features = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::default();

            let mut available_shader_float16_int8_features =
                vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR::default();
            available_shader_float16_int8_features.p_next = &mut available_scalar_features
                as *mut vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT
                as *mut c_void;

            let mut available_8bit_storage_features = vk::PhysicalDevice8BitStorageFeaturesKHR::default();
            available_8bit_storage_features.p_next = &mut available_shader_float16_int8_features
                as *mut vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR
                as *mut c_void;

            let mut available_desc_features = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::default();
            available_desc_features.p_next = &mut available_8bit_storage_features
                as *mut vk::PhysicalDevice8BitStorageFeaturesKHR
                as *mut c_void;

            let mut available_ts_features = vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::default();
            available_ts_features.p_next = &mut available_desc_features
                as *mut vk::PhysicalDeviceDescriptorIndexingFeaturesEXT
                as *mut c_void;

            let mut available_features2 = vk::PhysicalDeviceFeatures2 {
                p_next: &mut available_ts_features as *mut vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR
                    as *mut c_void,
                ..Default::default()
            };
            unsafe {
                self.native
                    .get_physical_device_features2(p_device, &mut available_features2)
            };
            let available_features = available_features2.features;

            let mut enabled_features = vk::PhysicalDeviceFeatures::default();
            let mut enabled_scalar_features = vk::PhysicalDeviceScalarBlockLayoutFeaturesEXT::default();
            let mut enabled_shader_float16_int8_features =
                vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR::default();
            let mut enabled_8bit_storage_features = vk::PhysicalDevice8BitStorageFeaturesKHR::default();
            let mut enabled_desc_features = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT::default();
            let mut enabled_ts_features = vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR::default();

            macro_rules! require_features {
                ($available: ident, $to_set: ident, $($name: ident),+) => {
                    $(
                        if $available.$name == 0 {
                            continue;
                        } else {
                            $to_set.$name = vk::TRUE;
                        }
                    )*
                };
            }

            require_features!(
                available_features,
                enabled_features,
                sampler_anisotropy,
                shader_uniform_buffer_array_dynamic_indexing,
                shader_sampled_image_array_dynamic_indexing,
                shader_storage_buffer_array_dynamic_indexing,
                shader_storage_image_array_dynamic_indexing,
                texture_compression_bc
            );
            require_features!(
                available_scalar_features,
                enabled_scalar_features,
                scalar_block_layout
            );
            require_features!(available_ts_features, enabled_ts_features, timeline_semaphore);
            require_features!(
                available_desc_features,
                enabled_desc_features,
                descriptor_binding_partially_bound,
                runtime_descriptor_array
            );
            require_features!(
                available_8bit_storage_features,
                enabled_8bit_storage_features,
                storage_buffer8_bit_access
            );
            require_features!(
                available_shader_float16_int8_features,
                enabled_shader_float16_int8_features,
                shader_int8
            );

            // Check formats
            // ------------------------------------------------------------------------------------

            // Buffer formats
            for (format, usage_bits) in format::BUFFER_FORMATS.iter() {
                let props = unsafe {
                    self.native
                        .get_physical_device_format_properties(p_device, format.0)
                };
                if !props.buffer_features.intersects(*usage_bits) {
                    continue 'g;
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
                    continue 'g;
                }
            }

            // Depth format
            {
                let props = unsafe {
                    self.native
                        .get_physical_device_format_properties(p_device, format::DEPTH_FORMAT.0)
                };
                if !props.optimal_tiling_features.contains(depth_format_features) {
                    continue 'g;
                }
            }

            let mut formats_props = HashMap::<vk::Format, vk::FormatProperties>::new();

            // Get properties for each format
            for (format, _size) in FORMAT_SIZES.iter() {
                let props = unsafe {
                    self.native
                        .get_physical_device_format_properties(p_device, format.0)
                };
                formats_props.insert(format.0, props);
            }

            adapters.push(Arc::new(Adapter {
                instance: Arc::clone(self),
                native: p_device,
                props,
                enabled_extensions,
                features: enabled_features,
                scalar_features: enabled_scalar_features,
                desc_features: enabled_desc_features,
                ts_features: enabled_ts_features,
                storage8bit_features: enabled_8bit_storage_features,
                shader_float16_int8_features: enabled_shader_float16_int8_features,
                queue_family_indices: queue_fam_indices,
                formats_props,
            }));
        }

        Ok(adapters)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let Some(debug_utils_ext) = &self.debug_utils_ext {
                debug_utils_ext.destroy_debug_utils_messenger(self.debug_utils_messenger.unwrap(), None);
            }
            self.native.destroy_instance(None);
        };
    }
}
