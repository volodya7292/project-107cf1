use std::ffi::c_void;
use std::mem;
use std::ops::Deref;
use std::sync::Arc;

use parking_lot::{Condvar, Mutex};
use rayon::ThreadBuilder;

/// Safe thread pool that joins all threads on drop.
#[derive(Debug)]
pub struct SafeThreadPool {
    inner: Option<rayon::ThreadPool>,
    threads: Vec<Arc<(Mutex<bool>, Condvar)>>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum TaskPriority {
    /// Used for realtime tasks such as rendering or updating per-frame state.
    Realtime,
    /// Intensive workloads, coroutines.
    Default,
}

impl TaskPriority {
    #[cfg(target_os = "macos")]
    fn macos_priority(&self) -> u32 {
        match self {
            TaskPriority::Realtime => macos::QOS_CLASS_USER_INTERACTIVE,
            TaskPriority::Default => macos::QOS_CLASS_DEFAULT,
        }
    }
}

#[cfg(target_os = "macos")]
fn context_and_function<F>(closure: F) -> (*mut c_void, macos::dispatch_function_t)
where
    F: FnOnce(),
{
    extern "C" fn work_execute_closure<F: FnOnce()>(context: Box<F>) {
        context();
    }

    let func: extern "C" fn(Box<F>) = work_execute_closure::<F>;
    let closure = Box::new(closure);

    unsafe { (mem::transmute(closure), mem::transmute(func)) }
}

#[cfg(not(target_os = "macos"))]
pub fn spawn_thread<F: FnOnce()>(f: F, _priority: TaskPriority)
where
    F: Send + 'static,
{
    std::thread::spawn(f);
}

#[cfg(target_os = "macos")]
pub fn spawn_thread<F: FnOnce()>(f: F, priority: TaskPriority)
where
    F: Send + 'static,
{
    unsafe {
        let queue = macos::dispatch_get_global_queue(priority.macos_priority() as isize, 0);
        let (context, work) = context_and_function(f);
        macos::dispatch_async_f(queue, context, work);
    }
}

fn schedule_worker(worker: ThreadBuilder, priority: TaskPriority) -> Arc<(Mutex<bool>, Condvar)> {
    let notify = Arc::new((Mutex::new(false), Condvar::new()));

    let closure = {
        let notify = Arc::clone(&notify);
        move || {
            worker.run();

            let mut ended = notify.0.lock();
            *ended = true;
            notify.1.notify_one();
        }
    };
    spawn_thread(closure, priority);

    notify
}

impl SafeThreadPool {
    pub fn new(
        num_threads: usize,
        priority: TaskPriority,
    ) -> Result<SafeThreadPool, rayon::ThreadPoolBuildError> {
        let mut threads = vec![];
        let inner = Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .spawn_handler(|thread| {
                    threads.push(schedule_worker(thread, priority));
                    Ok(())
                })
                .build()?,
        );

        Ok(SafeThreadPool { inner, threads })
    }
}

impl Deref for SafeThreadPool {
    type Target = rayon::ThreadPool;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl Drop for SafeThreadPool {
    fn drop(&mut self) {
        drop(self.inner.take());
        for notify in mem::take(&mut self.threads) {
            let mut ended = notify.0.lock();

            if !*ended {
                notify.1.wait(&mut ended);
            }
        }
    }
}
