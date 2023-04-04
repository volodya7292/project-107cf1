use core_graphics::base::CGFloat;
use objc::runtime::{Object, BOOL, YES};
use objc::{class, msg_send, sel, sel_impl};
use raw_window_handle::AppKitWindowHandle;
use std::ffi::c_void;
use std::mem;

pub type CAMetalLayer = *mut Object;

pub enum Layer {
    Existing(CAMetalLayer),
    Allocated(CAMetalLayer),
    None,
}

pub unsafe fn metal_layer_from_handle(handle: AppKitWindowHandle) -> Layer {
    if !handle.ns_view.is_null() {
        metal_layer_from_ns_view(handle.ns_view)
    } else if !handle.ns_window.is_null() {
        metal_layer_from_ns_window(handle.ns_window)
    } else {
        Layer::None
    }
}

pub unsafe fn metal_layer_from_ns_view(view: *mut c_void) -> Layer {
    let view: cocoa::base::id = mem::transmute(view);

    // Check if the view is a CAMetalLayer
    let class = class!(CAMetalLayer);
    let is_actually_layer: BOOL = msg_send![view, isKindOfClass: class];
    if is_actually_layer == YES {
        return Layer::Existing(view);
    }

    // Check if the view contains a valid CAMetalLayer
    let existing: CAMetalLayer = msg_send![view, layer];
    let use_current = if existing.is_null() {
        false
    } else {
        let result: BOOL = msg_send![existing, isKindOfClass: class];
        result == YES
    };

    let render_layer = if use_current {
        Layer::Existing(existing)
    } else {
        // Allocate a new CAMetalLayer for the current view
        let layer: CAMetalLayer = msg_send![class, new];
        let () = msg_send![view, setLayer: layer];
        let () = msg_send![view, setWantsLayer: YES];

        let window: cocoa::base::id = msg_send![view, window];
        if !window.is_null() {
            let scale_factor: CGFloat = msg_send![window, backingScaleFactor];
            let () = msg_send![layer, setContentsScale: scale_factor];
        }

        Layer::Allocated(layer)
    };

    let _: *mut c_void = msg_send![view, retain];
    render_layer
}

pub unsafe fn metal_layer_from_ns_window(window: *mut c_void) -> Layer {
    let ns_window = window as *mut Object;
    let ns_view = msg_send![ns_window, contentView];
    metal_layer_from_ns_view(ns_view)
}

pub unsafe fn metal_layer_update(handle: AppKitWindowHandle) {
    let contents_scale: CGFloat = if !handle.ns_view.is_null() {
        let view: cocoa::base::id = mem::transmute(handle.ns_view);
        msg_send![view, backingScaleFactor]
    } else if !handle.ns_window.is_null() {
        let window: cocoa::base::id = mem::transmute(handle.ns_window);
        let ns_view: cocoa::base::id = msg_send![window, contentView];
        msg_send![ns_view, backingScaleFactor]
    } else {
        panic!("Invalid appkit handle");
    };

    let Layer::Existing(layer) = metal_layer_from_handle(handle) else {
        panic!("CAMetalLayer does not exist");
    };

    let () = msg_send![layer, setContentsScale: contents_scale];
}
