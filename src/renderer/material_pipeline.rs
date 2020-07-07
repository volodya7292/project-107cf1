use std::mem;
use std::sync::Arc;

pub trait UniformStruct {
    fn model_offset() -> u32;
}

macro_rules! uniform_struct_impl {
    ($uniform_struct: ty, $model_name: ident) => {
        impl $crate::material_pipeline::UniformStruct for $uniform_struct {
            fn model_offset() -> u32 {
                let dummy = <$uniform_struct>::default();
                let offset = ((&dummy.$model_name) as *const _ as usize) - ((&dummy) as *const _ as usize);
                offset as u32
            }
        }
    };
}

pub struct MaterialPipeline {
    uniform_buffer_size: u32,
    uniform_buffer_model_offset: u32,
}

impl MaterialPipeline {
    pub fn uniform_buffer_size(&self) -> u32 {
        self.uniform_buffer_size
    }

    /// Get model matrix (mat4) offset in uniform buffer struct
    pub fn uniform_buffer_offset_model(&self) -> u32 {
        self.uniform_buffer_model_offset
    }
}

pub fn new<T: UniformStruct>() -> Arc<MaterialPipeline> {
    Arc::new(MaterialPipeline {
        uniform_buffer_size: mem::size_of::<T>() as u32,
        uniform_buffer_model_offset: T::model_offset(),
    })
}
