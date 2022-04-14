use parking_lot::{Condvar, Mutex};
use rayon::ThreadBuilder;
use std::ffi::c_void;
use std::mem;
use std::ops::Deref;
use std::sync::Arc;

/// Safe thread pool that joins all threads on drop.
pub struct SafeThreadPool {
    inner: Option<rayon::ThreadPool>,
    threads: Vec<Arc<(Mutex<bool>, Condvar)>>,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum TaskPriority {
    /// Used for realtime tasks such as rendering or updating per-frame state
    Realtime,
    /// One separate thread for urgent small tasks
    Coroutine,
    /// Used for intensive workloads
    Intensive,
}

#[cfg(target_os = "macos")]
fn context_and_function<F>(closure: F) -> (*mut c_void, macos::dispatch_function_t)
where
    F: FnOnce(),
{
    extern "C" fn work_execute_closure<F: FnOnce()>(context: Box<F>) {
        (*context)();
    }

    let func: extern "C" fn(Box<F>) = work_execute_closure::<F>;
    let closure = Box::new(closure);

    unsafe { (mem::transmute(closure), mem::transmute(func)) }
}

#[cfg(target_os = "macos")]
fn schedule_worker(worker: ThreadBuilder, priority: TaskPriority) -> Arc<(Mutex<bool>, Condvar)> {
    let macos_priority = match priority {
        TaskPriority::Realtime => macos::QOS_CLASS_USER_INTERACTIVE,
        TaskPriority::Coroutine => macos::QOS_CLASS_USER_INITIATED,
        TaskPriority::Intensive => macos::QOS_CLASS_DEFAULT,
    };

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

    unsafe {
        let queue = macos::dispatch_get_global_queue(macos_priority as isize, 0);
        let (context, work) = context_and_function(closure);
        macos::dispatch_async_f(queue, context, work);
    }

    notify
}

#[cfg(not(target_os = "macos"))]
fn schedule_worker(worker: ThreadBuilder, _: TaskPriority) -> Arc<(Mutex<bool>, Condvar)> {
    let notify = Arc::new((Mutex::new(false), Condvar::new()));

    {
        let notify = Arc::clone(&notify);

        std::thread::spawn(move || {
            worker.run();

            let mut ended = notify.0.lock();
            *ended = true;
            notify.1.notify_one();
        });
    }

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
