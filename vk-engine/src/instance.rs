use crate::adapter::Adapter;
use crate::device::Device;
use crate::{format, utils};
use ash::version::{InstanceV1_0, InstanceV1_1};
use ash::vk;
use std::mem;
use std::{rc::Rc, os::{raw::c_void, raw::c_char}};

pub struct Instance {
    pub(crate) native: ash::Instance,
    pub(crate) debug_utils_ext: Option<ash::extensions::ext::DebugUtils>,
    pub(crate) debug_utils_messenger: Option<vk::DebugUtilsMessengerEXT>,
    pub(crate) surface_khr: Option<ash::extensions::khr::Surface>,
}

impl Instance {
    pub fn create_surface(&self, window: &windows::Window) -> Result<vk::SurfaceKHR, vk::Result> {
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
                let mut queue_fam_indices: [[u8; 2]; 4] = [[u8::MAX, 0]; 4];

                for (i, fam_prop) in queue_families.iter().enumerate() {
                    // Check for present usage
                    let surface_supported = unsafe {
                        self.surface_khr
                            .as_ref()
                            .unwrap()
                            .get_physical_device_surface_support(p_device, i as u32, surface)
                            .unwrap()
                    };

                    if surface_supported && queue_fam_indices[3][0] == u8::MAX {
                        queue_fam_indices[3] = [i as u8, 0]; // present
                    }
                    if fam_prop.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && queue_fam_indices[0][0] == u8::MAX
                    {
                        queue_fam_indices[0] = [i as u8, 0]; // graphics
                    } else if fam_prop.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && queue_fam_indices[1][0] == u8::MAX
                    {
                        queue_fam_indices[1] = [i as u8, 0]; // compute
                    } else if fam_prop.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && queue_fam_indices[2][0] == u8::MAX
                    {
                        queue_fam_indices[2] = [i as u8, 0]; // transfer
                    }

                    if queue_fam_indices[0][0] != u8::MAX
                        && queue_fam_indices[1][0] != u8::MAX
                        && queue_fam_indices[2][0] != u8::MAX
                        && queue_fam_indices[3][0] != u8::MAX
                    {
                        break;
                    }
                }

                if queue_fam_indices[1][0] == u8::MAX {
                    queue_fam_indices[1] = queue_fam_indices[0]; // compute -> graphics
                }
                if queue_fam_indices[2][0] == u8::MAX {
                    queue_fam_indices[2] = queue_fam_indices[1]; // transfer -> compute
                }

                // The same queue(graphics) is used for rest if separate queues not available
                if queue_fam_indices[0][0] == u8::MAX || queue_fam_indices[3][0] == u8::MAX {
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
                for format in format::BufferFormats.iter() {
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
                for format in format::ImageFormats.iter() {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format.0)
                    };
                    if !props.optimal_tiling_features.contains(image_format_features)
                        && !props.linear_tiling_features.contains(image_format_features)
                    {
                        return None;
                    }
                }

                // Depth format
                {
                    let props = unsafe {
                        self.native
                            .get_physical_device_format_properties(p_device, format::DepthFormat.0)
                    };
                    if !props.optimal_tiling_features.contains(depth_format_features)
                        && !props.linear_tiling_features.contains(depth_format_features)
                    {
                        return None;
                    }
                }

                Some(Adapter {
                    native: p_device,
                    props,
                    enabled_extensions,
                    props12: props12.build(),
                    features: enabled_features,
                    features12: enabled_features12,
                    queue_family_indices: queue_fam_indices,
                })
            })
            .collect())
    }

    pub fn create_device(self: &Rc<Self>, adapter: &Adapter) -> Result<Rc<Device>, vk::Result> {
        let mut queue_infos: Vec<vk::DeviceQueueCreateInfo> = vec![];
        let priorities = [1.0_f32; u8::MAX as usize];

        for i in 0..adapter.queue_family_indices.len() {
            let mut used_indices = Vec::<u8>::with_capacity(u8::MAX as usize);

            for fam_index in adapter.queue_family_indices.iter() {
                if fam_index[0] == i as u8 && !used_indices.contains(&fam_index[1]) {
                    used_indices.push(fam_index[1]);
                }
            }
            if used_indices.is_empty() {
                continue;
            }

            let mut queue_info = vk::DeviceQueueCreateInfo::builder();
            queue_info = queue_info
                .queue_family_index(i as u32)
                .queue_priorities(&priorities[0..used_indices.len()]);
            queue_infos.push(queue_info.build());
        }

        let mut features12 = adapter.features12;
        let enabled_extensions_raw: Vec<*const c_char> = adapter
            .enabled_extensions
            .iter()
            .map(|name| name.as_ptr())
            .collect();

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions_raw)
            .enabled_features(&adapter.features)
            .push_next(&mut features12);

        let native_device = unsafe { self.native.create_device(adapter.native, &create_info, None)? };

        // Create allocator
        let allocator_info = vk_mem::AllocatorCreateInfo {
            physical_device: adapter.native,
            device: native_device.clone(),
            instance: self.native.clone(),
            flags: Default::default(),
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };
        let allocator = vk_mem::Allocator::new(&allocator_info).unwrap();

        Ok(Rc::new(Device {
            _instance: self.clone(),
            adapter: adapter.clone(),
            native: native_device,
            allocator,
        }))
    }

    pub fn destroy_surface(&self, surface: vk::SurfaceKHR) {
        unsafe { self.surface_khr.as_ref().unwrap().destroy_surface(surface, None) };
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
