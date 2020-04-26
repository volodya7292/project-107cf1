use crate::Window;
use glfw::Context;
use std::ptr;

type VkInstance = usize;
type VkSurfaceKHR = u64;
type VkResult = i32;

impl Window {
    /// Creates new vulkan-compatible surface
    /// # Safety
    /// p_surface: valid pointer to VkSurfaceKHR
    pub unsafe fn create_surface(&self, instance: VkInstance, p_surface: *mut VkSurfaceKHR) -> VkResult {
        glfw::ffi::glfwCreateWindowSurface(instance, self.native.window_ptr(), ptr::null(), p_surface)
            as VkResult
    }
}
