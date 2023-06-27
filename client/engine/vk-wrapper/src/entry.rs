use crate::utils;
use crate::Instance;
use ash::vk;
use common::log::{error, info, warn};
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

pub(crate) const VK_API_VERSION: u32 = vk::API_VERSION_1_2;

#[derive(Debug)]
pub enum InstanceError {
    AshError(ash::LoadingError),
    VkError(vk::Result),
}

impl From<ash::LoadingError> for InstanceError {
    fn from(err: ash::LoadingError) -> Self {
        InstanceError::AshError(err)
    }
}

impl From<vk::Result> for InstanceError {
    fn from(err: vk::Result) -> Self {
        InstanceError::VkError(err)
    }
}

pub struct Entry {
    pub(crate) ash_entry: ash::Entry,
}

unsafe extern "system" fn vk_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let msg = CStr::from_ptr((*p_callback_data).p_message);

    let msg_type = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VAL",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERF",
        _ => "",
    };

    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => warn!(target: "vulkan", "[{}] {:?}", msg_type, msg),
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(target: "vulkan", "[{}] {:?}", msg_type, msg);
            panic!("Vulkan validation");
        }
        _ => info!(target: "vulkan", "[{}] {:?}", msg_type, msg),
    }

    0
}

pub fn enumerate_required_window_extensions(
    window_handle: &dyn HasRawWindowHandle,
) -> Result<Vec<String>, vk::Result> {
    let extensions = match window_handle.raw_window_handle() {
        RawWindowHandle::Win32(_) => {
            const WINDOWS_EXTS: [&CStr; 2] = [
                ash::extensions::khr::Surface::name(),
                ash::extensions::khr::Win32Surface::name(),
            ];
            &WINDOWS_EXTS
        }
        RawWindowHandle::Wayland(_) => {
            const WAYLAND_EXTS: [&CStr; 2] = [
                ash::extensions::khr::Surface::name(),
                ash::extensions::khr::WaylandSurface::name(),
            ];
            &WAYLAND_EXTS
        }
        RawWindowHandle::Xlib(_) => {
            const XLIB_EXTS: [&CStr; 2] = [
                ash::extensions::khr::Surface::name(),
                ash::extensions::khr::XlibSurface::name(),
            ];
            &XLIB_EXTS
        }
        RawWindowHandle::Xcb(_) => {
            const XCB_EXTS: [&CStr; 2] = [
                ash::extensions::khr::Surface::name(),
                ash::extensions::khr::XcbSurface::name(),
            ];
            &XCB_EXTS
        }
        RawWindowHandle::AppKit(_) => {
            const MACOS_EXTS: [&CStr; 2] = [
                ash::extensions::khr::Surface::name(),
                ash::extensions::ext::MetalSurface::name(),
            ];
            &MACOS_EXTS
        }
        _ => return Err(vk::Result::ERROR_EXTENSION_NOT_PRESENT),
    };

    Ok(extensions
        .iter()
        .map(|&name| String::from(name.to_str().unwrap()))
        .collect())
}

impl Entry {
    pub fn new() -> Result<Arc<Entry>, ash::LoadingError> {
        Ok(Arc::new(Entry {
            ash_entry: unsafe { ash::Entry::load()? },
        }))
    }

    fn enumerate_instance_layer_names(&self) -> Result<Vec<String>, vk::Result> {
        Ok(self
            .ash_entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|layer| unsafe { utils::c_ptr_to_string(layer.layer_name.as_ptr()) })
            .collect())
    }

    fn enumerate_instance_extension_names(&self) -> Result<Vec<String>, vk::Result> {
        Ok(self
            .ash_entry
            .enumerate_instance_extension_properties(None)?
            .iter()
            .map(|ext| unsafe { utils::c_ptr_to_string(ext.extension_name.as_ptr()) })
            .collect())
    }

    pub fn create_instance(
        self: &Arc<Self>,
        app_name: &str,
        window: Option<&dyn HasRawWindowHandle>,
    ) -> Result<Arc<Instance>, InstanceError> {
        let c_app_name = CString::new(app_name).unwrap();
        let c_engine_name = CString::new("VULKAN").unwrap();
        let app_info = vk::ApplicationInfo::builder()
            .application_name(c_app_name.as_c_str())
            .engine_name(c_engine_name.as_c_str())
            .api_version(VK_API_VERSION);

        let required_extensions =
            window.map_or(Ok(vec![]), |window| enumerate_required_window_extensions(window))?;
        let available_layers = self.enumerate_instance_layer_names()?;
        let available_extensions = self.enumerate_instance_extension_names()?;

        let mut required_layers: Vec<&str> = vec![];
        let mut required_extensions: Vec<&str> = required_extensions.iter().map(|a| a.as_ref()).collect();
        let preferred_extensions: Vec<&str> = vec!["VK_KHR_portability_enumeration"];

        if cfg!(debug_assertions) {
            required_layers.push("VK_LAYER_KHRONOS_validation");
            required_extensions.push("VK_EXT_debug_utils");
        }
        required_extensions.push("VK_KHR_surface");

        // Filter layers
        let enabled_layers = utils::filter_names(&available_layers, &required_layers, true).unwrap();
        let enabled_layers_raw: Vec<*const c_char> =
            enabled_layers.iter().map(|name| name.as_ptr()).collect();

        // Filter extensions
        let mut enabled_extensions =
            utils::filter_names(&available_extensions, &required_extensions, true).unwrap();
        enabled_extensions
            .extend(utils::filter_names(&available_extensions, &preferred_extensions, false).unwrap());
        let enabled_extensions_raw: Vec<*const c_char> =
            enabled_extensions.iter().map(|name| name.as_ptr()).collect();

        // Infos
        let mut info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_layer_names(&enabled_layers_raw)
            .enabled_extension_names(&enabled_extensions_raw);

        if enabled_extensions.contains(&CString::new("VK_KHR_portability_enumeration").unwrap()) {
            info = info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        let mut debug_msg_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vk_debug_callback));

        // Push extension structures
        if cfg!(debug_assertions) {
            info = info.push_next(&mut debug_msg_info);
        }

        let native_instance = unsafe { self.ash_entry.create_instance(&info, None)? };

        // Init DebugUtils extension & create DebugUtilsMessenger
        let (debug_utils_ext, debug_utils_messenger) = if cfg!(debug_assertions) {
            let debug_utils = ash::extensions::ext::DebugUtils::new(&self.ash_entry, &native_instance);
            let debug_utils_messenger =
                unsafe { debug_utils.create_debug_utils_messenger(&debug_msg_info, None)? };
            (Some(debug_utils), Some(debug_utils_messenger))
        } else {
            (None, None)
        };

        // Init Surface extension
        let surface_khr = ash::extensions::khr::Surface::new(&self.ash_entry, &native_instance);

        Ok(Arc::new(Instance {
            entry: Arc::clone(self),
            native: native_instance,
            debug_utils_ext,
            debug_utils_messenger,
            surface_khr,
        }))
    }
}
