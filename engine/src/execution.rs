use core::once_cell::sync::OnceCell;
use core::utils::threading::{SafeThreadPool, TaskPriority};

static REALTIME_THREAD_POOL: OnceCell<SafeThreadPool> = OnceCell::new();

pub fn init(n_realtime_threads: usize) {
    let realtime_thread_pool = SafeThreadPool::new(n_realtime_threads, TaskPriority::Realtime).unwrap();
    REALTIME_THREAD_POOL.set(realtime_thread_pool).unwrap();
}

pub fn realtime_queue() -> &'static SafeThreadPool {
    REALTIME_THREAD_POOL.get().unwrap()
}
