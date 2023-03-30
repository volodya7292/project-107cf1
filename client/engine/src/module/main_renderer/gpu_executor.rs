use common::any::AsAny;
use common::parking_lot::Mutex;
use common::rayon::prelude::*;
use common::types::{HashMap, HashSet};
use common::utils::AllSameBy;
use smallvec::SmallVec;
use std::any::TypeId;
use std::sync::Arc;
use std::{iter};
use vk_wrapper::{
    CmdList, Device, DeviceError, PipelineStageFlags, QueueType, Semaphore, SignalSemaphore, SubmitInfo,
    WaitSemaphore,
};

pub struct GPUJob {
    queue_type: QueueType,
    completion_semaphore: Arc<Semaphore>,
    completion_value: u64,
    cmd_list: CmdList,
}

impl GPUJob {
    pub fn new(name: &str, device: &Arc<Device>, queue_type: QueueType) -> Result<Self, DeviceError> {
        let queue = device.get_queue(queue_type);

        Ok(Self {
            queue_type,
            completion_semaphore: Arc::new(device.create_timeline_semaphore()?),
            completion_value: 0,
            cmd_list: queue.create_primary_cmd_list(name)?,
        })
    }

    pub fn wait_semaphore(&self) -> WaitSemaphore {
        WaitSemaphore {
            semaphore: Arc::clone(&self.completion_semaphore),
            wait_dst_mask: PipelineStageFlags::ALL_COMMANDS,
            wait_value: self.completion_value,
        }
    }

    pub fn wait(&self) -> Result<(), DeviceError> {
        self.completion_semaphore.wait(self.completion_value)
    }

    pub fn get_cmd_list_for_recording(&mut self) -> &mut CmdList {
        self.wait().unwrap();
        &mut self.cmd_list
    }
}

pub struct GPUJobExecInfo<'a> {
    pub job: &'a mut GPUJob,
    pub wait_semaphores: SmallVec<[WaitSemaphore; 4]>,
    pub signal_semaphores: SmallVec<[SignalSemaphore; 4]>,
}

impl<'a> GPUJobExecInfo<'a> {
    pub fn new(job: &'a mut GPUJob) -> Self {
        Self {
            job,
            wait_semaphores: Default::default(),
            signal_semaphores: Default::default(),
        }
    }

    pub fn with_wait_semaphores(mut self, wait_semaphores: &[WaitSemaphore]) -> Self {
        self.wait_semaphores.extend(wait_semaphores.iter().cloned());
        self
    }

    pub fn with_signal_semaphores(mut self, signal_semaphores: &[SignalSemaphore]) -> Self {
        self.signal_semaphores.extend(signal_semaphores.iter().cloned());
        self
    }
}

pub trait GPUJobDeviceExt {
    fn create_job(self: &Arc<Self>, name: &str, queue_type: QueueType) -> Result<GPUJob, DeviceError>;
    unsafe fn run_jobs(&self, jobs: &mut [GPUJobExecInfo]) -> Result<(), DeviceError>;
    unsafe fn run_jobs_sync(&self, jobs: &mut [GPUJobExecInfo]) -> Result<(), DeviceError>;
}

impl GPUJobDeviceExt for Device {
    fn create_job(self: &Arc<Self>, name: &str, queue_type: QueueType) -> Result<GPUJob, DeviceError> {
        let queue = self.get_queue(queue_type);
        Ok(GPUJob {
            queue_type,
            completion_semaphore: Arc::new(self.create_timeline_semaphore()?),
            completion_value: 0,
            cmd_list: queue.create_primary_cmd_list(name)?,
        })
    }

    /// Runs jobs asynchronously.
    unsafe fn run_jobs(&self, jobs: &mut [GPUJobExecInfo]) -> Result<(), DeviceError> {
        assert!(jobs.iter().all_same_by_key(|v| v.job.queue_type));

        if jobs.is_empty() {
            return Ok(());
        }

        // Wait for potential previous task completion
        for info in &*jobs {
            info.job.wait()?;
        }

        // Set new completion value for each job
        for info in &mut *jobs {
            info.job.completion_value += 1;
        }

        let signal_semaphores: Vec<_> = jobs
            .iter()
            .map(|info| {
                info.signal_semaphores
                    .iter()
                    .cloned()
                    .chain(iter::once(SignalSemaphore {
                        semaphore: Arc::clone(&info.job.completion_semaphore),
                        signal_value: info.job.completion_value,
                    }))
                    .collect::<Vec<_>>()
            })
            .collect();

        let work_infos: Vec<_> = jobs
            .iter()
            .zip(&signal_semaphores)
            .map(|(info, signal_semaphores)| SubmitInfo {
                wait_semaphores: info.wait_semaphores.to_vec(),
                cmd_lists: vec![&info.job.cmd_list],
                signal_semaphores: signal_semaphores.clone(),
            })
            .collect();

        let queue = self.get_queue(jobs[0].job.queue_type);
        queue.submit_infos(&work_infos, None)
    }

    /// Runs jobs and wait for completion.
    unsafe fn run_jobs_sync(&self, jobs: &mut [GPUJobExecInfo]) -> Result<(), DeviceError> {
        self.run_jobs(jobs)?;
        for job in jobs {
            job.job.wait()?;
        }
        Ok(())
    }
}

pub type RenderStageId = TypeId;

pub trait RenderStage: AsAny + Send + Sync + 'static {
    fn name(&self) -> &str;
    /// Recorded `CmdList` of this gpu_executor is run on GPU
    /// only after the returned dependencies have been completed.
    fn execution_dependencies(&self) -> &'static [RenderStageId];
    /// [RenderStage::record] will be called
    /// only after the returned dependencies have been completed.
    fn record_dependencies(&self) -> &'static [RenderStageId];
    /// Records a `CmdList` for this gpu_executor.
    fn record(
        &mut self,
        record_dependencies: &HashMap<TypeId, &Mutex<Box<dyn RenderStage>>>,
        cmd_list: &mut CmdList,
    );
}

struct StageInfo {
    execution_dependencies: Vec<RenderStageId>,
    job: Mutex<GPUJob>,
}

struct SubmitBatch {
    stages: Vec<RenderStageId>,
}

pub struct StageManager {
    device: Arc<Device>,
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
            stages,
            stages_infos,
            submit_batches,
        }
    }

    pub unsafe fn run(&mut self) -> Result<(), DeviceError> {
        // Safety: waiting for completion of pending cmd lists
        // is done inside Device::run_jobs()

        for submit_batch in &self.submit_batches {
            submit_batch.stages.par_iter().for_each(|id| {
                let mut stage = self.stages[id].lock();
                let info = &self.stages_infos[id];

                let record_deps: HashMap<_, _> = stage
                    .record_dependencies()
                    .iter()
                    .map(|id| (*id, &self.stages[id]))
                    .collect();

                let mut job = info.job.lock();
                let cmd_list = job.get_cmd_list_for_recording();
                stage.record(&record_deps, cmd_list);
            });

            let mut submit_jobs: Vec<_> = submit_batch
                .stages
                .iter()
                .map(|id| {
                    let info = self.stages_infos.get(id).unwrap();
                    (info, info.job.lock())
                })
                .collect();

            let mut exec_infos: Vec<_> = submit_jobs
                .iter_mut()
                .map(|(info, job)| GPUJobExecInfo {
                    job: &mut *job,
                    wait_semaphores: info
                        .execution_dependencies
                        .iter()
                        .map(|dep_id| {
                            let dep_info = &self.stages_infos[dep_id];
                            let dep_job = dep_info.job.lock();
                            WaitSemaphore {
                                semaphore: Arc::clone(&dep_job.completion_semaphore),
                                wait_dst_mask: PipelineStageFlags::ALL_COMMANDS,
                                wait_value: dep_job.completion_value,
                            }
                        })
                        .collect(),
                    signal_semaphores: Default::default(),
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
