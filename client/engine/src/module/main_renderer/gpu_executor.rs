use common::utils::AllSameBy;
use smallvec::SmallVec;
use std::iter;
use std::sync::Arc;
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
