use crate::module::main_renderer::camera::PerspectiveCamera;
use crate::module::main_renderer::material_pipeline::{MaterialPipelineSet, PipelineKindId};
use crate::module::main_renderer::resource_manager::ResourceManagementScope;
use crate::module::main_renderer::resources::{MaterialPipelineParams, Renderable};
use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use common::glm::Vec3;
use common::types::HashMap;
use entity_data::{EntityId, EntityStorage};
use std::any::{Any, TypeId};
use std::sync::Arc;
use vk_wrapper::{
    CmdList, DescriptorPool, DescriptorSet, DeviceBuffer, Semaphore, SignalSemaphore, Swapchain,
    SwapchainImage, WaitSemaphore,
};

pub mod compose;
pub mod depth;
pub mod g_buffer;
pub mod present_queue_transition;

pub struct StageContext<'a> {
    pub storage: &'a EntityStorage,
    pub material_pipelines: &'a [MaterialPipelineSet],
    pub ordered_entities: &'a [EntityId],
    pub active_camera: &'a PerspectiveCamera,
    pub relative_camera_pos: Vec3,
    pub curr_vertex_meshes: &'a HashMap<EntityId, Arc<RawVertexMesh>>,
    pub renderables: &'a HashMap<EntityId, Renderable>,
    pub g_per_frame_pool: &'a DescriptorPool,
    pub g_per_frame_in: DescriptorSet,
    pub per_frame_ub: &'a DeviceBuffer,
    pub uniform_buffer_basic: &'a DeviceBuffer,
    pub render_size: (u32, u32),
    pub swapchain: &'a Arc<Swapchain>,
    pub render_sw_image: &'a SwapchainImage,
    pub frame_completion_semaphore: &'a Arc<Semaphore>,
}

pub struct StageRunResult {
    /// The stage waits for these semaphores before execution.
    pub(crate) wait_semaphores: Vec<WaitSemaphore>,
    pub(crate) signal_semaphores: Vec<SignalSemaphore>,
}

impl StageRunResult {
    fn new() -> Self {
        Self {
            wait_semaphores: vec![],
            signal_semaphores: vec![],
        }
    }

    pub fn with_wait_semaphores(mut self, semaphores: Vec<WaitSemaphore>) -> Self {
        self.wait_semaphores = semaphores;
        self
    }

    pub fn with_signal_semaphores(mut self, semaphores: Vec<SignalSemaphore>) -> Self {
        self.signal_semaphores = semaphores;
        self
    }
}

pub type RenderStageId = TypeId;

pub trait RenderStage: Any + Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn num_pipeline_kinds(&self) -> u32 {
        0
    }

    /// Recorded `CmdList` of this gpu_executor is run on GPU
    /// only after the returned dependencies have been completed.
    fn execution_dependencies(&self) -> Vec<RenderStageId> {
        vec![]
    }
    /// [RenderStage::run] will be called
    /// only after the returned dependencies have been submitted and completed.
    fn record_dependencies(&self) -> Vec<RenderStageId> {
        vec![]
    }

    fn setup(&mut self, _pipeline_kinds: &[PipelineKindId]) {}
    fn register_pipeline_kind(
        &self,
        _params: MaterialPipelineParams,
        _material_pipeline_set: &mut MaterialPipelineSet,
    ) {
    }

    /// Records a `CmdList` for this gpu_executor.
    fn run(
        &mut self,
        cmd_list: &mut CmdList,
        resources: &ResourceManagementScope,
        ctx: &StageContext,
    ) -> StageRunResult;
}
