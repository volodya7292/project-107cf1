use crate::module::main_renderer::gpu_executor::{GPUJob, GPUJobDeviceExt, GPUJobExecInfo};
use crate::module::main_renderer::resource_manager::ResourceManager;
use crate::module::main_renderer::stage::{RenderStage, RenderStageId, StageContext};
use common::parking_lot::Mutex;
use common::types::{HashMap, HashSet};
use std::sync::Arc;
use vk_wrapper::{Device, DeviceError, QueueType};

struct StageInfo {
    execution_dependencies: HashSet<RenderStageId>,
    job: Mutex<GPUJob>,
}

#[derive(Debug)]
struct SubmitStages {
    /// This is sorted in execution-dependencies order
    stages: Vec<RenderStageId>,
}

pub(crate) struct StageManager {
    device: Arc<Device>,
    res_manager: ResourceManager,
    stages: HashMap<RenderStageId, Mutex<Box<dyn RenderStage>>>,
    stages_infos: HashMap<RenderStageId, StageInfo>,
    submits: Vec<SubmitStages>,
}

/// Partitions unrelated stages into batches.
fn partition_dependencies(
    stages: HashSet<RenderStageId>,
    k_depends_on_v: &HashMap<RenderStageId, HashSet<RenderStageId>>,
) -> Vec<Vec<RenderStageId>> {
    let mut stages_left = stages;
    let mut batches = Vec::with_capacity(stages_left.len());

    while !stages_left.is_empty() {
        let mut batch = Vec::with_capacity(stages_left.len());

        for step in &stages_left {
            let ready = k_depends_on_v
                .get(step)
                .map_or(true, |deps| deps.iter().all(|d| !stages_left.contains(d)));

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
    pub(super) fn new(device: &Arc<Device>, stages: Vec<Box<dyn RenderStage>>) -> Self {
        let stages: HashMap<_, _> = stages
            .into_iter()
            .map(|v| (v.as_ref().type_id(), Mutex::new(v)))
            .collect();

        let execution_dependencies: HashMap<RenderStageId, HashSet<RenderStageId>> = stages
            .iter()
            .map(|(id, stage)| {
                (
                    *id,
                    stage
                        .lock()
                        .execution_dependencies()
                        .into_iter()
                        .collect::<HashSet<_>>(),
                )
            })
            .collect();

        let record_dependencies: HashMap<RenderStageId, HashSet<RenderStageId>> = stages
            .iter()
            .map(|(id, stage)| {
                (
                    *id,
                    stage
                        .lock()
                        .record_dependencies()
                        .into_iter()
                        .collect::<HashSet<_>>(),
                )
            })
            .collect();

        let combined_dependencies: HashMap<_, _> =
            execution_dependencies.iter().chain(&record_dependencies).fold(
                HashMap::<RenderStageId, HashSet<RenderStageId>>::new(),
                |mut acc, (id, deps)| {
                    let entry = acc.entry(*id).or_default();
                    entry.extend(deps);
                    acc
                },
            );

        let partitioned = partition_dependencies(stages.keys().cloned().collect(), &combined_dependencies);
        let mut submits = vec![SubmitStages { stages: vec![] }];

        for stage_id in partitioned.into_iter().flatten() {
            if record_dependencies
                .get(&stage_id)
                .map_or(false, |deps| !deps.is_empty())
            {
                submits.push(SubmitStages { stages: vec![] });
            }
            submits.last_mut().unwrap().stages.push(stage_id);
        }

        let stages_infos: HashMap<_, _> = stages
            .iter()
            .map(|(id, _)| {
                (
                    *id,
                    StageInfo {
                        execution_dependencies: execution_dependencies.get(id).cloned().unwrap_or_default(), //,.unwrap_or(vec![]),
                        job: Mutex::new(
                            GPUJob::new(stages[id].lock().name(), device, QueueType::Graphics).unwrap(),
                        ),
                    },
                )
            })
            .collect();

        Self {
            device: Arc::clone(device),
            res_manager: ResourceManager::new(device),
            stages,
            stages_infos,
            submits,
        }
    }

    pub fn stages(&self) -> &HashMap<RenderStageId, Mutex<Box<dyn RenderStage>>> {
        &self.stages
    }

    pub unsafe fn run(&mut self, ctx: &StageContext) -> Result<(), DeviceError> {
        // Safety: waiting for completion of pending cmd lists
        // is done inside Device::run_jobs()

        let scope = self.res_manager.scope();

        for submit_batch in &self.submits {
            let run_results: Vec<_> = submit_batch
                .stages
                .iter()
                .map(|id| {
                    let mut stage = self.stages[id].lock();
                    let info = &self.stages_infos[id];

                    let mut job = info.job.lock();
                    let cmd_list = job.get_cmd_list_for_recording();

                    stage.run(cmd_list, &scope, ctx)
                })
                .collect();

            let stages_wait_semaphores: Vec<_> = submit_batch
                .stages
                .iter()
                .zip(&run_results)
                .map(|(id, run_result)| {
                    let info = &self.stages_infos[id];
                    info.execution_dependencies
                        .iter()
                        .map(|dep_id| {
                            let dep_info = &self.stages_infos[dep_id];
                            let dep_job = dep_info.job.lock();
                            dep_job.wait_semaphore()
                        })
                        .chain(run_result.wait_semaphores.iter().cloned())
                        .collect::<Vec<_>>()
                })
                .collect();
            let mut submit_jobs: Vec<_> = submit_batch
                .stages
                .iter()
                .map(|id| self.stages_infos[id].job.lock())
                .collect();

            let mut exec_infos: Vec<_> = submit_jobs
                .iter_mut()
                .zip(stages_wait_semaphores)
                .zip(run_results)
                .map(|((job, wait_semaphores), run_results)| GPUJobExecInfo {
                    job: &mut *job,
                    wait_semaphores: wait_semaphores.into(),
                    signal_semaphores: run_results.signal_semaphores.into(),
                })
                .collect();

            // Use _sync to wait for the completion because the next batch recording depends on these jobs.
            self.device.run_jobs_sync(&mut exec_infos)?;
        }

        Ok(())
    }

    pub fn wait_idle(&self) {
        let queue = self.device.get_queue(QueueType::Graphics);
        queue.wait_idle().unwrap();
    }
}

impl Drop for StageManager {
    fn drop(&mut self) {
        self.wait_idle();
    }
}
