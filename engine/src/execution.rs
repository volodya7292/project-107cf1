use base::utils::threading::{SafeThreadPool, TaskPriority};

pub fn init(n_realtime_threads: usize) {
    SafeThreadPool::init_global(n_realtime_threads, TaskPriority::Realtime).unwrap();
}
