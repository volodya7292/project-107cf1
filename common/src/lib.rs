pub mod instant_meter;
pub mod lrc;
pub mod resource_encoder;
pub mod resource_file;
pub mod scene;
pub mod shader_compiler;
pub mod slice_split;
pub mod threading;
pub mod timer;
pub mod types;
pub mod unsafe_slice;
pub mod utils;

pub use async_executor;
pub use crossbeam_channel;
pub use futures_lite;
pub use log;
pub use macos;
pub use nalgebra;
pub use nalgebra_glm as glm;
pub use parking_lot;
pub use rayon;
pub use resource_encoder::encode_resources;
pub use shader_compiler::compile_shaders;
use std::sync::atomic;

pub const MO_RELAXED: atomic::Ordering = atomic::Ordering::Relaxed;
pub const MO_ACQUIRE: atomic::Ordering = atomic::Ordering::Acquire;
pub const MO_RELEASE: atomic::Ordering = atomic::Ordering::Release;
