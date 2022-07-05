use crate::device::DeviceWrapper;
use crate::instance::VK_API_VERSION;
use crate::{
    device::{self, Device},
    surface::Surface,
    Instance, Queue, SamplerFilter, SamplerMipmap,
};
use ash::vk;
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::{collections::HashMap, ffi::CString, os::raw::c_char};

pub(crate) type MemoryBlock = gpu_alloc::MemoryBlock<vk::DeviceMemory>;

pub(crate) const QUEUE_TYPE_COUNT: usize = 4;

pub struct Adapter {
    pub(crate) instance: Arc<Instance>,
    pub(crate) native: vk::PhysicalDevice,
    pub(crate) _props: vk::PhysicalDeviceProperties,
    pub(crate) enabled_extensions: Vec<CString>,
    pub(crate) features: vk::PhysicalDeviceFeatures,
    pub(crate) desc_features: vk::PhysicalDeviceDescriptorIndexingFeaturesEXT,
    pub(crate) ts_features: vk::PhysicalDeviceTimelineSemaphoreFeaturesKHR,
    pub(crate) storage8bit_features: vk::PhysicalDevice8BitStorageFeaturesKHR,
    pub(crate) shader_float16_int8_features: vk::PhysicalDeviceShaderFloat16Int8FeaturesKHR,
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

        let mut desc_features = self.desc_features;
        let mut ts_features = self.ts_features;
        let mut storage8bit_features = self.storage8bit_features;
        let mut shader_float16_int8_features = self.shader_float16_int8_features;

        let enabled_extensions_raw: Vec<*const c_char> =
            self.enabled_extensions.iter().map(|name| name.as_ptr()).collect();

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions_raw)
            .enabled_features(&self.features)
            .push_next(&mut desc_features)
            .push_next(&mut ts_features)
            .push_next(&mut storage8bit_features)
            .push_next(&mut shader_float16_int8_features);

        let native_device = unsafe {
            self.instance
                .native
                .create_device(self.native, &create_info, None)?
        };

        let ts_khr = ash::extensions::khr::TimelineSemaphore::new(&self.instance.native, &native_device);
        let swapchain_khr = ash::extensions::khr::Swapchain::new(&self.instance.native, &native_device);

        let device_wrapper = Arc::new(DeviceWrapper {
            native: native_device,
            adapter: Arc::clone(self),
            ts_khr,
        });

        // Get queues
        let mut queues = Vec::with_capacity(QUEUE_TYPE_COUNT);
        for queue_info in &queue_infos {
            for i in 0..queue_info.queue_count {
                queues.push(Arc::new(Queue {
                    device_wrapper: Arc::clone(&device_wrapper),
                    swapchain_khr: swapchain_khr.clone(),
                    native: RwLock::new(unsafe {
                        device_wrapper
                            .native
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

        let memory_props =
            unsafe { gpu_alloc_ash::device_properties(&self.instance.native, VK_API_VERSION, self.native)? };

        let allowed_memory_types = memory_props
            .memory_types
            .iter()
            .enumerate()
            .filter_map(|(i, v)| {
                // HOST_VISIBLE memory types must also be HOST_COHERENT
                // to avoid explicit flushing/invalidating after mapping and alignment issues
                if !v.props.contains(gpu_alloc::MemoryPropertyFlags::HOST_VISIBLE)
                    || v.props.contains(gpu_alloc::MemoryPropertyFlags::HOST_COHERENT)
                {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .fold(0_u32, |acc, v| acc | (1 << v));

        let allocator = gpu_alloc::GpuAllocator::<vk::DeviceMemory>::new(
            gpu_alloc::Config {
                dedicated_threshold: 32 * 1024 * 1024,
                preferred_dedicated_threshold: 32 * 1024 * 1024,
                transient_dedicated_threshold: 128 * 1024 * 1024,
                starting_free_list_chunk: 16 * 1024 * 1024,
                final_free_list_chunk: 128 * 1024 * 1024,
                minimal_buddy_size: 1 * 256,
                initial_buddy_dedicated_size: 32 * 1024 * 1024,
            },
            memory_props,
        );

        // TODO: use reserved memory to reduce stutters due to lots of vkAllocateMemory

        let pipeline_cache_info = vk::PipelineCacheCreateInfo::builder();
        let pipeline_cache = unsafe {
            device_wrapper
                .native
                .create_pipeline_cache(&pipeline_cache_info, None)?
        };

        let default_sampler = device_wrapper.create_sampler(
            SamplerFilter::NEAREST,
            SamplerFilter::NEAREST,
            SamplerMipmap::NEAREST,
            1.0,
        )?;

        Ok(Arc::new(Device {
            wrapper: device_wrapper,
            allocator: Mutex::new(allocator),
            allowed_memory_types,
            total_used_dev_memory: Arc::new(Default::default()),
            swapchain_khr,
            queues,
            default_sampler,
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
