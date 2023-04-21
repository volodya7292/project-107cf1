use common::parking_lot::Mutex;
use common::types::{HashMap, HashSet};
use std::any::Any;
use std::collections::hash_map;
use std::hash::Hash;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper::{
    AccessFlags, Binding, BindingRes, BufferUsageFlags, CmdList, DescriptorPool, DescriptorSet, Device,
    DeviceBuffer, Format, HostBuffer, Image, ImageType, ImageUsageFlags, PipelineSignature,
    PipelineStageFlags, QueueType,
};

pub struct OwnedDescriptorSets {
    pool: DescriptorPool,
    sets: Vec<DescriptorSet>,
}

impl OwnedDescriptorSets {
    pub fn create_binding(&self, id: u32, array_index: u32, res: BindingRes) -> Binding {
        self.pool.create_binding(id, array_index, res)
    }

    pub fn sets(&self) -> &[DescriptorSet] {
        &self.sets
    }

    pub fn get(&self, idx: usize) -> DescriptorSet {
        self.sets[idx]
    }
}

type Name = String;

pub struct ResourceManager {
    device: Arc<Device>,
    res_params: Mutex<HashMap<Name, Box<dyn Any + Send + Sync>>>,
    resources: Mutex<HashMap<Name, Box<dyn Any + Send + Sync>>>,
}

impl ResourceManager {
    pub fn new(device: &Arc<Device>) -> Self {
        Self {
            device: Arc::clone(device),
            res_params: Mutex::new(HashMap::with_capacity(128)),
            resources: Mutex::new(HashMap::with_capacity(128)),
        }
    }

    pub fn scope(&mut self) -> ResourceManagementScope {
        ResourceManagementScope {
            manager: self,
            used_names: Mutex::new(HashSet::with_capacity(128)),
        }
    }
}

pub struct ResourceManagementScope<'a> {
    manager: &'a ResourceManager,
    used_names: Mutex<HashSet<String>>,
}

impl ResourceManagementScope<'_> {
    pub fn add_used_name(&self, name: &str) {
        let name_is_new = self.used_names.lock().insert(name.to_owned());
        if !name_is_new {
            panic!("Resource name \"{}\" is already used", name);
        }
    }

    pub fn get<Res: 'static>(&self, name: &str) -> Arc<Res> {
        let resources = self.manager.resources.lock();
        let res = resources.get(name).unwrap();
        let res = res.downcast_ref::<Arc<Res>>().unwrap();
        Arc::clone(res)
    }

    pub fn get_host_buffer<T: Copy + 'static>(&self, name: &str) -> Arc<Mutex<HostBuffer<T>>> {
        self.get(name)
    }

    pub fn get_image(&self, name: &str) -> Arc<Image> {
        self.get(name)
    }

    /// Creates a resource.
    /// Panics if the resource with `name` has already been requested.
    pub fn create<Res: Send + Sync + 'static>(&self, name: &str, value: Arc<Res>) -> Arc<Res> {
        self.add_used_name(name);

        let mut data = self.manager.resources.lock();

        match data.entry(name.to_owned()) {
            hash_map::Entry::Vacant(res) => {
                res.insert(Box::new(Arc::clone(&value)));
                value
            }
            hash_map::Entry::Occupied(mut curr_res) => {
                let curr_res = curr_res.get_mut().downcast_mut::<Arc<Res>>().unwrap();
                *curr_res = Arc::clone(&value);
                value
            }
        }
    }

    /// Requests a resource. If `key_params` has been changed, `on_create` is called.
    /// Panics if the resource with `name` has already been requested.
    pub fn request<
        Params: PartialEq + Send + Sync + 'static,
        Res: Send + Sync + 'static,
        F: FnOnce(&Params, &str) -> Arc<Res>,
    >(
        &self,
        name: &str,
        key_params: Params,
        on_create: F,
    ) -> Arc<Res> {
        self.add_used_name(name);

        let mut data = self.manager.resources.lock();
        let mut params = self.manager.res_params.lock();

        match (params.entry(name.to_owned()), data.entry(name.to_owned())) {
            (hash_map::Entry::Vacant(params), hash_map::Entry::Vacant(res)) => {
                let val = on_create(&key_params, name);
                params.insert(Box::new(key_params));
                res.insert(Box::new(Arc::clone(&val)));
                val
            }
            (hash_map::Entry::Occupied(mut curr_params), hash_map::Entry::Occupied(mut curr_res)) => {
                let curr_params = curr_params.get_mut().downcast_mut::<Params>().unwrap();
                let curr_res = curr_res.get_mut().downcast_mut::<Arc<Res>>().unwrap();

                if curr_params != &key_params {
                    *curr_res = on_create(&key_params, name);
                    *curr_params = key_params;
                }
                Arc::clone(curr_res)
            }
            _ => unreachable!(),
        }
    }

    pub fn request_cmd_lists(&self, name: &str, params: CmdListParams) -> Arc<Mutex<Vec<CmdList>>> {
        self.request(name, params, |params, name| {
            let queue = self.manager.device.get_queue(params.queue_type);
            let cmd_lists = (0..params.count)
                .map(|i| {
                    if params.secondary {
                        queue.create_secondary_cmd_list(&format!("{name}-{i}")).unwrap()
                    } else {
                        queue.create_primary_cmd_list(&format!("{name}-{i}")).unwrap()
                    }
                })
                .collect::<Vec<_>>();
            Arc::new(Mutex::new(cmd_lists))
        })
    }

    pub fn request_cmd_list(&self, name: &str, params: CmdListParams) -> Arc<Mutex<CmdList>> {
        assert_eq!(params.count, 1);

        self.request(name, params, |params, name| {
            let queue = self.manager.device.get_queue(params.queue_type);
            let cmd_list = if params.secondary {
                queue.create_secondary_cmd_list(name).unwrap()
            } else {
                queue.create_primary_cmd_list(name).unwrap()
            };
            Arc::new(Mutex::new(cmd_list))
        })
    }

    pub fn request_host_buffer<T: Copy + 'static>(
        &self,
        name: &str,
        params: HostBufferParams,
    ) -> Arc<Mutex<HostBuffer<T>>> {
        self.request(name, params, |params, name| {
            let buffer = self
                .manager
                .device
                .create_host_buffer_named(params.usage, params.len, name)
                .unwrap();
            Arc::new(Mutex::new(buffer))
        })
    }

    pub fn request_device_buffer(&self, name: &str, params: DeviceBufferParams) -> Arc<DeviceBuffer> {
        self.request(name, params, |params, name| {
            let buffer = self
                .manager
                .device
                .create_device_buffer_named(params.usage, params.element_size, params.len, name)
                .unwrap();
            Arc::new(buffer)
        })
    }

    pub fn request_uniform_buffer<T: Copy + 'static>(
        &self,
        name: &str,
        data: T,
        copy_cl: &mut CmdList,
    ) -> Arc<DeviceBuffer> {
        let len = mem::size_of_val(&data);

        let staging_buffer = self.request_host_buffer(
            &format!("{name}-staging"),
            HostBufferParams::new(BufferUsageFlags::TRANSFER_SRC, len as u64),
        );

        let device_buffer = self.request_device_buffer(
            name,
            DeviceBufferParams::new(
                BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM,
                1,
                len as u64,
            ),
        );

        let mut staging_buffer = staging_buffer.lock();
        staging_buffer.write(0, unsafe {
            slice::from_raw_parts(&data as *const _ as *const u8, len)
        });

        copy_cl.copy_buffer(&*staging_buffer, 0, &*device_buffer, 0, len as u64);
        copy_cl.barrier_buffer(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::VERTEX_SHADER
                | PipelineStageFlags::PIXEL_SHADER
                | PipelineStageFlags::COMPUTE,
            &[device_buffer
                .barrier()
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_READ)],
        );

        device_buffer
    }

    pub fn request_image(&self, name: &str, params: ImageParams) -> Arc<Image> {
        self.request(name, params, |params, name| {
            self.manager
                .device
                .create_image(
                    params.ty,
                    params.is_array,
                    params.format,
                    params.preferred_mip_levels,
                    params.usage,
                    params.preferred_size,
                    name,
                )
                .unwrap()
        })
    }

    pub fn request_descriptors(
        &self,
        name: &str,
        signature: &Arc<PipelineSignature>,
        set_layout_id: u32,
        num_descriptors: usize,
    ) -> Arc<OwnedDescriptorSets> {
        self.request(
            name,
            (Arc::clone(signature), set_layout_id, num_descriptors),
            |(signature, set_layout_id, num_descriptors), name| {
                let mut pool = signature.create_pool(*set_layout_id, 1, name).unwrap();
                let sets = (0..*num_descriptors)
                    .map(|_| pool.alloc().unwrap())
                    .collect::<Vec<_>>();

                Arc::new(OwnedDescriptorSets { pool, sets })
            },
        )
    }
}

impl Drop for ResourceManagementScope<'_> {
    fn drop(&mut self) {
        let used_names = self.used_names.lock();
        self.manager
            .resources
            .lock()
            .retain(|name, _| used_names.contains(name));
    }
}

#[derive(Eq, PartialEq, Hash)]
pub struct CmdListParams {
    queue_type: QueueType,
    secondary: bool,
    count: usize,
}

impl CmdListParams {
    pub fn secondary(queue: QueueType) -> Self {
        Self {
            queue_type: queue,
            secondary: true,
            count: 1,
        }
    }

    pub fn with_count(mut self, count: usize) -> Self {
        self.count = count;
        self
    }
}

#[derive(Eq, PartialEq, Hash)]
pub struct HostBufferParams {
    usage: BufferUsageFlags,
    len: u64,
}

impl HostBufferParams {
    pub fn new(usage: BufferUsageFlags, len: u64) -> Self {
        Self { usage, len }
    }
}

#[derive(Eq, PartialEq, Hash)]
pub struct DeviceBufferParams {
    usage: BufferUsageFlags,
    element_size: u64,
    len: u64,
}

impl DeviceBufferParams {
    pub fn new(usage: BufferUsageFlags, element_size: u64, len: u64) -> Self {
        Self {
            usage,
            element_size,
            len,
        }
    }
}

#[derive(Eq, PartialEq, Hash)]
pub struct ImageParams {
    ty: ImageType,
    format: Format,
    usage: ImageUsageFlags,
    preferred_size: (u32, u32, u32),
    preferred_mip_levels: u32,
    is_array: bool,
}

impl ImageParams {
    pub fn d2(format: Format, usage: ImageUsageFlags, preferred_size: (u32, u32)) -> Self {
        Self {
            ty: Image::TYPE_2D,
            format,
            usage,
            preferred_size: (preferred_size.0, preferred_size.1, 1),
            preferred_mip_levels: 1,
            is_array: false,
        }
    }

    pub fn d2_array(format: Format, usage: ImageUsageFlags, preferred_size: (u32, u32, u32)) -> Self {
        Self {
            ty: Image::TYPE_2D,
            format,
            usage,
            preferred_size,
            preferred_mip_levels: 1,
            is_array: true,
        }
    }

    pub fn d3(format: Format, usage: ImageUsageFlags, preferred_size: (u32, u32, u32)) -> Self {
        Self {
            ty: Image::TYPE_3D,
            format,
            usage,
            preferred_size,
            preferred_mip_levels: 1,
            is_array: false,
        }
    }

    pub fn with_preferred_mip_levels(mut self, max_mip_levels: u32) -> Self {
        self.preferred_mip_levels = max_mip_levels;
        self
    }
}
