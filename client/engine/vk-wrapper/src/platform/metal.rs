use core_graphics::base::CGFloat;
use objc2::runtime::{AnyObject, Bool};
use objc2::{class, msg_send};
use raw_window_handle::AppKitWindowHandle;
use std::ffi::c_void;
use std::mem;

pub type CAMetalLayer = *mut AnyObject;

pub enum Layer {
    Existing(CAMetalLayer),
    Allocated(CAMetalLayer),
    None,
}

pub unsafe fn metal_layer_from_handle(handle: AppKitWindowHandle) -> Layer {
    unsafe { metal_layer_from_ns_view(handle.ns_view.as_ptr()) }
}

pub unsafe fn metal_layer_from_ns_view(view: *mut c_void) -> Layer {
    unsafe {
        let view: *mut AnyObject = mem::transmute(view);

        // Check if the view is a CAMetalLayer
        let class = class!(CAMetalLayer);
        let is_actually_layer: bool = msg_send![view, isKindOfClass: class];
        if is_actually_layer {
            return Layer::Existing(view);
        }

        // Check if the view contains a valid CAMetalLayer
        let existing: CAMetalLayer = msg_send![view, layer];
        let use_current = if existing.is_null() {
            false
        } else {
            let result: bool = msg_send![existing, isKindOfClass: class];
            result
        };

        let render_layer = if use_current {
            Layer::Existing(existing)
        } else {
            // Allocate a new CAMetalLayer for the current view
            let layer: CAMetalLayer = msg_send![class, new];
            let () = msg_send![view, setLayer: layer];
            let () = msg_send![view, setWantsLayer: Bool::YES];

            let window: *mut AnyObject = msg_send![view, window];
            if !window.is_null() {
                let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
                let () = msg_send![layer, setContentsScale: scale_factor];
            }

            Layer::Allocated(layer)
        };

        let _: *mut AnyObject = msg_send![view, retain];
        render_layer
    }
}

pub unsafe fn metal_layer_update(handle: AppKitWindowHandle) {
    unsafe {
        let view: *mut AnyObject = mem::transmute(handle.ns_view.as_ptr());
        let contents_scale: CGFloat = msg_send![view, backingScaleFactor];

        let Layer::Existing(layer) = metal_layer_from_handle(handle) else {
            panic!("CAMetalLayer does not exist");
        };

        let () = msg_send![layer, setContentsScale: contents_scale];
    }
}
