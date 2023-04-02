use crate::module::main_renderer::gpu_executor::{GPUJob, GPUJobDeviceExt, GPUJobExecInfo};
use crate::module::main_renderer::resource_manager::ResourceManager;
use crate::module::main_renderer::stage::{RenderStage, RenderStageId};
use common::any::AsAny;
use common::parking_lot::Mutex;
use common::rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use common::types::{HashMap, HashSet};
use std::any::Any;
use std::sync::Arc;
use vk_wrapper::{Device, DeviceError, PipelineStageFlags, QueueType, WaitSemaphore};

struct StageInfo {
    execution_dependencies: Vec<RenderStageId>,
    job: Mutex<GPUJob>,
}

struct SubmitBatch {
    stages: Vec<RenderStageId>,
}

pub struct StageManager {
    device: Arc<Device>,
    res_manager: ResourceManager,
    stages: HashMap<RenderStageId, Mutex<Box<dyn RenderStage>>>,
    stages_infos: HashMap<RenderStageId, StageInfo>,
    submit_batches: Vec<SubmitBatch>,
}

/// Partitions unrelated stages into batches.
fn partition_dependencies(
    stages: HashSet<RenderStageId>,
    k_depends_on_v: HashMap<RenderStageId, Vec<RenderStageId>>,
) -> Vec<Vec<RenderStageId>> {
    let mut stages_left = stages;
    let mut batches = Vec::with_capacity(stages_left.len());

    while !stages_left.is_empty() {
        let mut batch = Vec::with_capacity(stages_left.len());

        for step in &stages_left {
            let deps = k_depends_on_v
                .get(step)
                .map_or(<&[RenderStageId]>::default(), |v| v.as_slice());

            let ready = deps.iter().all(|d| !stages_left.contains(d));
            if ready {
                batch.push(*step);
            }
        }

        for stage in &batch {
            stages_left.remove(stage);
        }

        if batch.is_empty() {
            panic!("Invalid dependencies!");
        }
        batches.push(batch);
    }

    batches
}

impl StageManager {
    /// Valid usage: Later stages must not be dependencies of previous stages.
    pub fn new(device: &Arc<Device>, stages: Vec<Box<dyn RenderStage>>) -> Self {
        let stages: HashMap<_, _> = stages
            .into_iter()
            .map(|v| (v.as_any().type_id(), Mutex::new(v)))
            .collect();

        let execution_dependencies: HashMap<RenderStageId, Vec<RenderStageId>> = stages
            .iter()
            .map(|(id, stage)| (*id, stage.lock().execution_dependencies().to_vec()))
            .collect();

        let stages_infos: HashMap<_, _> = stages
            .iter()
            .map(|(id, _)| {
                (
                    *id,
                    StageInfo {
                        execution_dependencies: execution_dependencies.get(id).cloned().unwrap_or(vec![]),
                        job: Mutex::new(
                            GPUJob::new(stages[id].lock().name(), device, QueueType::Graphics).unwrap(),
                        ),
                    },
                )
            })
            .collect();

        let record_dependencies: HashMap<RenderStageId, Vec<RenderStageId>> = stages
            .iter()
            .map(|(id, stage)| (*id, stage.lock().record_dependencies().to_vec()))
            .collect();

        let submit_batches: Vec<_> =
            partition_dependencies(stages.keys().cloned().collect(), record_dependencies)
                .into_iter()
                .map(|submit_batch_stages| SubmitBatch {
                    stages: submit_batch_stages,
                })
                .collect();

        Self {
            device: Arc::clone(device),
            res_manager: ResourceManager::new(device),
            stages,
            stages_infos,
            submit_batches,
        }
    }

    pub unsafe fn run(&mut self, ctx: &(dyn Any + Sync)) -> Result<(), DeviceError> {
        // Safety: waiting for completion of pending cmd lists
        // is done inside Device::run_jobs()

        let scope = self.res_manager.scope();

        for submit_batch in &self.submit_batches {
            let run_results: Vec<_> = submit_batch
                .stages
                .par_iter()
                .map(|id| {
                    let mut stage = self.stages[id].lock();
                    let info = &self.stages_infos[id];

                    let mut job = info.job.lock();
                    let cmd_list = job.get_cmd_list_for_recording();

                    stage.run(cmd_list, &scope, ctx)
                })
                .collect();

            let mut submit_jobs: Vec<_> = submit_batch
                .stages
                .iter()
                .map(|id| (*id, self.stages_infos[id].job.lock()))
                .collect();

            let mut exec_infos: Vec<_> = submit_jobs
                .iter_mut()
                .zip(run_results)
                .map(|((stage_id, job), run_result)| {
                    let info = &self.stages_infos[stage_id];

                    GPUJobExecInfo {
                        job: &mut *job,
                        wait_semaphores: info
                            .execution_dependencies
                            .iter()
                            .map(|dep_id| {
                                let dep_info = &self.stages_infos[dep_id];
                                let dep_job = dep_info.job.lock();
                                dep_job.wait_semaphore()
                            })
                            .chain(run_result.wait_semaphores)
                            .collect(),
                        signal_semaphores: Default::default(),
                    }
                })
                .collect();

            // Use _sync to wait for the completion because the next batch recording depends on these jobs.
            self.device.run_jobs_sync(&mut exec_infos)?;
        }

        Ok(())
    }
}

impl Drop for StageManager {
    fn drop(&mut self) {
        let queue = self.device.get_queue(QueueType::Graphics);
        queue.wait_idle().unwrap();
    }
}
