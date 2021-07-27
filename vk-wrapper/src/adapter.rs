use crate::device::DeviceWrapper;
use crate::{
    device::{self, Device},
    surface::Surface,
    Instance, Queue,
};
use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use std::sync::{Arc, Mutex, RwLock};
use std::{collections::HashMap, ffi::CString, os::raw::c_char};

pub(crate) const QUEUE_TYPE_COUNT: usize = 4;

pub struct Adapter {
    pub(crate) instance: Arc<Instance>,
    pub(crate) native: vk::PhysicalDevice,
    pub(crate) _props: vk::PhysicalDeviceProperties,
    pub(crate) enabled_extensions: Vec<CString>,
    pub(crate) _props12: vk::PhysicalDeviceVulkan12Properties,
    pub(crate) features: vk::PhysicalDeviceFeatures,
    pub(crate) features12: vk::PhysicalDeviceVulkan12Features,
    pub(crate) queue_family_indices: [[u32; 2]; QUEUE_TYPE_COUNT],
    pub(crate) formats_props: HashMap<vk::Format, vk::FormatProperties>,
}

unsafe impl Send for Adapter {}

unsafe impl Sync for Adapter {}

impl Adapter {
    pub fn create_device(self: &Arc<Self>) -> Result<Arc<Device>, vk::Result> {
        let mut queue_infos = Vec::<vk::DeviceQueueCreateInfo>::with_capacity(QUEUE_TYPE_COUNT);
        let priorities = [1.0_f32; u8::MAX as usize];

        for i in 0..QUEUE_TYPE_COUNT {
            let mut used_indices = Vec::<u32>::with_capacity(256);

            for fam_index in self.queue_family_indices.iter() {
                if fam_index[0] == i as u32 && !used_indices.contains(&fam_index[1]) {
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

        let mut features12 = self.features12;
        let enabled_extensions_raw: Vec<*const c_char> =
            self.enabled_extensions.iter().map(|name| name.as_ptr()).collect();

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions_raw)
            .enabled_features(&self.features)
            .push_next(&mut features12);

        let device_wrapper = Arc::new(DeviceWrapper(unsafe {
            self.instance
                .native
                .create_device(self.native, &create_info, None)?
        }));
        let swapchain_khr = ash::extensions::khr::Swapchain::new(&self.instance.native, &device_wrapper.0);

        // Get queues
        let mut queues = Vec::with_capacity(QUEUE_TYPE_COUNT);
        for queue_info in &queue_infos {
            for i in 0..queue_info.queue_count {
                queues.push(Arc::new(Queue {
                    device_wrapper: Arc::clone(&device_wrapper),
                    swapchain_khr: swapchain_khr.clone(),
                    native: RwLock::new(unsafe {
                        device_wrapper
                            .0
                            .get_device_queue(queue_info.queue_family_index, i)
                    }),
                    semaphore: Arc::new(device::create_binary_semaphore(&device_wrapper)?),
                    timeline_sp: Arc::new(device::create_timeline_semaphore(&device_wrapper)?),
                    family_index: queue_info.queue_family_index,
                }));
            }
        }

        // Graphics queue always exists. Compute, transfer, present queues may be the same as the graphics queue.
        for i in queues.len()..QUEUE_TYPE_COUNT {
            queues.push(Arc::clone(
                &queues
                    .iter()
                    .find(|v| v.family_index == self.queue_family_indices[i][0])
                    .unwrap(),
            ));
        }

        // Create allocator
        let allocator_info = vk_mem::AllocatorCreateInfo {
            physical_device: self.native,
            device: device_wrapper.0.clone(),
            instance: self.instance.native.clone(),
            flags: Default::default(),
            preferred_large_heap_block_size: 0,
            frame_in_use_count: 0,
            heap_size_limits: None,
        };
        let allocator = vk_mem::Allocator::new(&allocator_info).unwrap();

        let pipeline_cache_info = vk::PipelineCacheCreateInfo::builder();
        let pipeline_cache = unsafe {
            device_wrapper
                .0
                .create_pipeline_cache(&pipeline_cache_info, None)?
        };

        Ok(Arc::new(Device {
            adapter: Arc::clone(self),
            wrapper: device_wrapper,
            allocator,
            swapchain_khr,
            queues,
            pipeline_cache: Mutex::new(pipeline_cache),
        }))
    }

    pub(crate) fn get_surface_capabilities(
        &self,
        surface: &Surface,
    ) -> Result<vk::SurfaceCapabilitiesKHR, vk::Result> {
        let mut surface_capabs = unsafe {
            self.instance
                .surface_khr
                .get_physical_device_surface_capabilities(self.native, surface.native)?
        };
        if surface_capabs.max_image_count == 0 {
            surface_capabs.max_image_count = u32::MAX;
        }
        Ok(surface_capabs)
    }

    pub(crate) fn get_surface_formats(
        &self,
        surface: &Surface,
    ) -> Result<Vec<vk::SurfaceFormatKHR>, vk::Result> {
        Ok(unsafe {
            self.instance
                .surface_khr
                .get_physical_device_surface_formats(self.native, surface.native)?
        })
    }

    pub(crate) fn get_surface_present_modes(
        &self,
        surface: &Surface,
    ) -> Result<Vec<vk::PresentModeKHR>, vk::Result> {
        Ok(unsafe {
            self.instance
                .surface_khr
                .get_physical_device_surface_present_modes(self.native, surface.native)?
        })
    }

    pub(crate) fn get_image_format_properties(
        &self,
        format: vk::Format,
        image_type: vk::ImageType,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
    ) -> Result<vk::ImageFormatProperties, vk::Result> {
        Ok(unsafe {
            self.instance.native.get_physical_device_image_format_properties(
                self.native,
                format,
                image_type,
                tiling,
                usage,
                vk::ImageCreateFlags::empty(),
            )?
        })
    }

    pub fn is_extension_enabled(&self, name: &str) -> bool {
        self.enabled_extensions.contains(&CString::new(name).unwrap())
    }

    pub fn is_surface_valid(&self, surface: &Surface) -> Result<bool, vk::Result> {
        let capabs = self.get_surface_capabilities(surface)?;
        Ok(capabs.min_image_extent.width > 0 && capabs.min_image_extent.height > 0)
    }

    pub(crate) fn is_linear_filter_supported(&self, format: vk::Format, tiling: vk::ImageTiling) -> bool {
        return if tiling == vk::ImageTiling::OPTIMAL {
            self.formats_props[&format].optimal_tiling_features
        } else {
            self.formats_props[&format].linear_tiling_features
        }
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
    }
}
