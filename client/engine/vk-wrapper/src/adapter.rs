use crate::device::DeviceWrapper;
use crate::entry::VK_API_VERSION;
use crate::sampler::SamplerClamp;
use crate::{device::Device, surface::Surface, Instance, Queue, QueueType, SamplerFilter, SamplerMipmap};
use ash::vk;
use ash::vk::Handle;
use common::parking_lot::{Mutex, RwLock};
use std::sync::Arc;
use std::{collections::HashMap, ffi::CString, os::raw::c_char};
use std::{mem, ptr};

#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) struct QueueId {
    pub family_index: u32,
    /// A queue family may have multiple queues in it. This identifies queue array index.
    pub index: u32,
}

impl QueueId {
    pub fn null() -> Self {
        Self {
            family_index: u32::MAX,
            index: 0,
        }
    }
}

pub struct Adapter {
    pub(crate) instance: Arc<Instance>,
    pub(crate) native: vk::PhysicalDevice,
    pub(crate) props: vk::PhysicalDeviceProperties,
    pub(crate) enabled_extensions: Vec<CString>,
    pub(crate) features: vk::PhysicalDeviceFeatures,
    pub(crate) vulkan12_features: vk::PhysicalDeviceVulkan12Features,
    pub(crate) queue_family_indices: [QueueId; QueueType::TOTAL_QUEUES],
    pub(crate) formats_props: HashMap<vk::Format, vk::FormatProperties>,
}

unsafe impl Send for Adapter {}

unsafe impl Sync for Adapter {}

impl Adapter {
    pub fn create_device(self: &Arc<Self>) -> Result<Arc<Device>, vk::Result> {
        let mut queue_infos = Vec::<vk::DeviceQueueCreateInfo>::with_capacity(QueueType::TOTAL_QUEUES);
        let priorities = [1.0_f32; u8::MAX as usize];

        let mut queue_count_by_family = HashMap::<u32, u32>::new();

        for id in &self.queue_family_indices {
            if *id == QueueId::null() {
                continue;
            }
            let count = queue_count_by_family.entry(id.family_index).or_default();
            *count = (*count).max(id.index + 1);
        }

        for (family_idx, count) in queue_count_by_family {
            let mut queue_info = vk::DeviceQueueCreateInfo::builder();
            queue_info = queue_info
                .queue_family_index(family_idx)
                .queue_priorities(&priorities[0..count as usize]);
            queue_infos.push(queue_info.build());
        }

        let enabled_extensions_raw: Vec<*const c_char> =
            self.enabled_extensions.iter().map(|name| name.as_ptr()).collect();
        let mut vulkan12_features = self.vulkan12_features;

        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&enabled_extensions_raw)
            .enabled_features(&self.features)
            .push_next(&mut vulkan12_features);

        let native_device = unsafe {
            self.instance
                .native
                .create_device(self.native, &create_info, None)?
        };

        let swapchain_khr = ash::extensions::khr::Swapchain::new(&self.instance.native, &native_device);

        let device_wrapper = Arc::new(DeviceWrapper {
            native: native_device,
            adapter: Arc::clone(self),
        });

        // Get queues
        let unique_queues: HashMap<u32, Vec<Arc<Queue>>> = queue_infos
            .iter()
            .map(|info| {
                let queues: Vec<_> = (0..info.queue_count)
                    .map(|idx| {
                        Arc::new(Queue {
                            device_wrapper: Arc::clone(&device_wrapper),
                            swapchain_khr: swapchain_khr.clone(),
                            native: RwLock::new(unsafe {
                                device_wrapper
                                    .native
                                    .get_device_queue(info.queue_family_index, idx)
                            }),
                            family_index: info.queue_family_index,
                            ty: QueueType::from_idx(
                                self.queue_family_indices
                                    .iter()
                                    .position(|id| {
                                        id.family_index == info.queue_family_index && id.index == idx
                                    })
                                    .unwrap(),
                            ),
                        })
                    })
                    .collect();

                (info.queue_family_index, queues)
            })
            .collect();

        let queues: Vec<_> = self
            .queue_family_indices
            .iter()
            .map(|id| {
                unique_queues
                    .get(&id.family_index)
                    .map(|vec| Arc::clone(&vec[id.index as usize]))
            })
            .collect();

        let memory_props = unsafe {
            self.instance
                .native
                .get_physical_device_memory_properties(self.native)
        };

        // Create memory allocator

        let mut vma_vulkan_funcs: vma::VmaVulkanFunctions = unsafe { mem::zeroed() };
        vma_vulkan_funcs.vkGetInstanceProcAddr = Some(unsafe {
            mem::transmute::<
                _,
                unsafe extern "C" fn(vma::VkInstance, *const i8) -> Option<unsafe extern "C" fn()>,
            >(self.instance.entry.ash_entry.static_fn().get_instance_proc_addr)
        });
        vma_vulkan_funcs.vkGetDeviceProcAddr = Some(unsafe {
            mem::transmute::<
                _,
                unsafe extern "C" fn(vma::VkDevice, *const i8) -> Option<unsafe extern "C" fn()>,
            >(self.instance.native.fp_v1_0().get_device_proc_addr)
        });

        let mut allocator_info: vma::VmaAllocatorCreateInfo = unsafe { mem::zeroed() };
        allocator_info.instance = self.instance.native.handle().as_raw() as vma::VkInstance;
        allocator_info.physicalDevice = self.native.as_raw() as vma::VkPhysicalDevice;
        allocator_info.device = device_wrapper.native.handle().as_raw() as vma::VkDevice;
        allocator_info.vulkanApiVersion = VK_API_VERSION;
        allocator_info.pVulkanFunctions = &vma_vulkan_funcs;

        let mut allocator: vma::VmaAllocator = ptr::null_mut();
        let result = unsafe { vma::vmaCreateAllocator(&allocator_info, &mut allocator) };

        if result != vma::VK_SUCCESS {
            return Err(vk::Result::from_raw(result));
        }

        // Create host pool

        let mut host_alloc_info: vma::VmaAllocationCreateInfo = unsafe { mem::zeroed() };
        host_alloc_info.usage = vma::VMA_MEMORY_USAGE_UNKNOWN;
        host_alloc_info.preferredFlags = vk::MemoryPropertyFlags::HOST_CACHED.as_raw();
        host_alloc_info.requiredFlags =
            (vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT).as_raw();

        let mut host_memory_type_index = 0;
        let result = unsafe {
            vma::vmaFindMemoryTypeIndex(allocator, u32::MAX, &host_alloc_info, &mut host_memory_type_index)
        };

        if result != vma::VK_SUCCESS {
            return Err(vk::Result::from_raw(result));
        }

        let mut host_pool_info: vma::VmaPoolCreateInfo = unsafe { mem::zeroed() };
        host_pool_info.memoryTypeIndex = host_memory_type_index;
        host_pool_info.minBlockCount = 2;

        let mut host_mem_pool: vma::VmaPool = ptr::null_mut();
        let result = unsafe { vma::vmaCreatePool(allocator, &host_pool_info, &mut host_mem_pool) };

        if result != vma::VK_SUCCESS {
            return Err(vk::Result::from_raw(result));
        }

        // Create device pool

        let mut device_alloc_info: vma::VmaAllocationCreateInfo = unsafe { mem::zeroed() };
        device_alloc_info.usage = vma::VMA_MEMORY_USAGE_UNKNOWN;
        device_alloc_info.requiredFlags = vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw();

        let mut device_memory_type_index = 0;
        let result = unsafe {
            vma::vmaFindMemoryTypeIndex(
                allocator,
                u32::MAX,
                &device_alloc_info,
                &mut device_memory_type_index,
            )
        };

        if result != vma::VK_SUCCESS {
            return Err(vk::Result::from_raw(result));
        }

        let device_memory_heap =
            memory_props.memory_types[host_pool_info.memoryTypeIndex as usize].heap_index;
        let device_memory_size = memory_props.memory_heaps[device_memory_heap as usize].size;

        let mut device_pool_info: vma::VmaPoolCreateInfo = unsafe { mem::zeroed() };
        device_pool_info.memoryTypeIndex = device_memory_type_index;
        device_pool_info.blockSize = 1 << 28; // 256 MB

        // Reserve memory to reduce stutters due to lots of vkAllocateMemory
        device_pool_info.minBlockCount =
            16.min((device_memory_size as f64 / device_pool_info.blockSize as f64).ceil() as usize / 2);

        let mut device_mem_pool: vma::VmaPool = ptr::null_mut();
        let result = unsafe { vma::vmaCreatePool(allocator, &device_pool_info, &mut device_mem_pool) };

        if result != vma::VK_SUCCESS {
            return Err(vk::Result::from_raw(result));
        }

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
            SamplerClamp::REPEAT,
            1.0,
        )?;

        Ok(Arc::new(Device {
            wrapper: device_wrapper,
            allocator,
            host_mem_pool,
            device_mem_pool,
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

    pub fn get_surface_size(&self, surface: &Surface) -> Result<(u32, u32), vk::Result> {
        let capabs = self.get_surface_capabilities(surface)?;
        Ok((capabs.current_extent.width, capabs.current_extent.height))
    }

    pub(crate) fn is_linear_filter_supported(&self, format: vk::Format, tiling: vk::ImageTiling) -> bool {
        if tiling == vk::ImageTiling::OPTIMAL {
            self.formats_props[&format].optimal_tiling_features
        } else {
            self.formats_props[&format].linear_tiling_features
        }
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    }
}
