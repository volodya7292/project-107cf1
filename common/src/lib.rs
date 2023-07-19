pub mod alloc;
pub mod any;
pub mod function_storage;
pub mod instant_meter;
pub mod lrc;
pub mod nd_range;
pub mod resource_encoder;
pub mod resource_file;
pub mod scene;
pub mod shader_compiler;
pub mod slice_split;
pub mod threading;
pub mod types;
pub mod unsafe_slice;
pub mod utils;

pub use futures_lite;
pub use log;
#[cfg(target_os = "macos")]
pub use macos;
pub use memoffset;
pub use moka;
pub use nalgebra;
pub use nalgebra_glm as glm;
pub use parking_lot;
pub use rayon;
pub use resource_encoder::encode_resources;
pub use shader_compiler::compile_shaders;
use std::sync::atomic;
pub use tokio;

pub const MO_RELAXED: atomic::Ordering = atomic::Ordering::Relaxed;
pub const MO_ACQUIRE: atomic::Ordering = atomic::Ordering::Acquire;
pub const MO_RELEASE: atomic::Ordering = atomic::Ordering::Release;
pub const MO_SEQCST: atomic::Ordering = atomic::Ordering::SeqCst;

/// `make_static_id()` returns unique identifier (a `String` consisting of file name, line and column numbers)
/// of in-code position of the call to this macro.
/// `make_static_id(additional_id)` returns the same static
/// identifier but parametrized with `additional_id`.
#[macro_export]
macro_rules! make_static_id {
    () => {
        concat!(std::file!(), "-", std::line!(), "-", std::column!())
    };
    ($additional_id: expr) => {
        format!(
            "{}-{}-{}-{}",
            std::file!(),
            std::line!(),
            std::column!(),
            $additional_id
        )
    };
}
