//! Generic task execution infrastructure.
//!
//! This module owns persistent worker threads and exposes a small task API that
//! can be shared by the system scheduler and query execution.

use std::{
    any::Any,
    collections::VecDeque,
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{sync_channel, SyncSender, TryRecvError},
        Arc, Condvar, Mutex,
    },
    thread::{self, JoinHandle},
};

/// Result state returned by an asynchronous task.
pub(crate) enum TaskOutcome<R> {
    /// The task finished and produced a value.
    Completed(R),
    /// The task panicked; the payload is forwarded to the owner of the task.
    Panicked(Box<dyn Any + Send + 'static>),
}

/// Message sent back to the thread that submitted a task.
pub(crate) struct TaskResult<R> {
    /// Stable caller-provided identifier.
    pub(crate) task_id: usize,
    /// Completion or panic state for the task.
    pub(crate) outcome: TaskOutcome<R>,
}

/// Boxed task body submitted to the worker queue.
type TaskFn = Box<dyn FnOnce() + Send + 'static>;

/// Shared FIFO queue protected by the executor mutex.
struct Queue {
    /// Pending tasks in submission order.
    tasks: VecDeque<TaskFn>,
    /// Set when the executor drops; workers drain and exit.
    shutdown: bool,
}

/// State shared by every worker thread.
///
/// The mutex only protects task hand-off. Once a task is popped, it runs
/// without holding the queue lock.
struct Shared {
    queue: Mutex<Queue>,
    /// Notifies one waiting worker per enqueued task, or all on shutdown.
    available: Condvar,
    /// Set when a task panicked outside a result channel; poisons the executor.
    poisoned: AtomicBool,
}

impl Shared {
    /// Creates shared state with an open queue and no poison.
    fn new() -> Self {
        Self {
            queue: Mutex::new(Queue {
                tasks: VecDeque::new(),
                shutdown: false,
            }),
            available: Condvar::new(),
            poisoned: AtomicBool::new(false),
        }
    }

    /// Pushes one task and wakes a worker. Panics after shutdown.
    fn enqueue(&self, task: TaskFn) {
        let mut queue = self.queue.lock().unwrap_or_else(|err| err.into_inner());

        assert!(
            !queue.shutdown,
            "cannot enqueue a task while TaskExecutor is shutting down",
        );

        queue.tasks.push_back(task);
        self.available.notify_one();
    }

    /// Blocks until a task is available or the executor shuts down.
    fn take_task(&self) -> Option<TaskFn> {
        let mut queue = self.queue.lock().unwrap_or_else(|err| err.into_inner());

        loop {
            if let Some(task) = queue.tasks.pop_front() {
                return Some(task);
            }

            if queue.shutdown {
                return None;
            }

            queue = self
                .available
                .wait(queue)
                .unwrap_or_else(|err| err.into_inner());
        }
    }

    /// Pops one task without blocking, for caller-thread stealing.
    fn try_take_task(&self) -> Option<TaskFn> {
        let mut queue = self.queue.lock().unwrap_or_else(|err| err.into_inner());
        queue.tasks.pop_front()
    }

    /// Marks the queue closed and wakes every worker.
    fn shutdown(&self) {
        let mut queue = self.queue.lock().unwrap_or_else(|err| err.into_inner());
        queue.shutdown = true;
        self.available.notify_all();
    }
}

/// Join handle for one persistent worker thread.
struct Worker {
    /// Kept as an `Option` so drop can take it while joining.
    handle: Option<JoinHandle<()>>,
}

impl Worker {
    /// Spawns a named worker running `worker_loop` over the shared state.
    fn new(index: usize, shared: Arc<Shared>) -> Self {
        let handle = thread::Builder::new()
            .name(format!("ecs-worker-{index}"))
            .spawn(move || worker_loop(shared))
            .expect("failed to spawn ECS worker");

        Self {
            handle: Some(handle),
        }
    }
}

/// Worker main loop: takes tasks until the queue shuts down.
fn worker_loop(shared: Arc<Shared>) {
    while let Some(task) = shared.take_task() {
        run_task(&shared, task);
    }
}

/// Runs one task, poisoning the shared state if it panics.
fn run_task(shared: &Shared, task: TaskFn) {
    if catch_unwind(AssertUnwindSafe(task)).is_err() {
        shared.poisoned.store(true, Ordering::SeqCst);
    }
}

/// Cheap handle for submitting tasks to a `TaskExecutor` without owning or
/// joining its worker threads.
pub(crate) struct TaskExecutorHandle {
    shared: Arc<Shared>,
    worker_count: usize,
}

impl TaskExecutorHandle {
    #[inline]
    /// Returns whether a worker-level failure made the executor unreliable.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.shared.poisoned.load(Ordering::SeqCst)
    }

    /// Enqueues one independent task and sends its result through `sender`.
    ///
    /// Panics inside the task are caught and returned as `TaskOutcome::Panicked`
    /// so the caller can decide when to resume unwinding.
    pub(crate) fn spawn<R, F>(&self, task_id: usize, sender: SyncSender<TaskResult<R>>, task: F)
    where
        R: Send + 'static,
        F: FnOnce() -> R + Send + 'static,
    {
        self.shared.enqueue(Box::new(move || {
            let outcome = match catch_unwind(AssertUnwindSafe(task)) {
                Ok(value) => TaskOutcome::Completed(value),
                Err(payload) => TaskOutcome::Panicked(payload),
            };

            let _ = sender.send(TaskResult { task_id, outcome });
        }));
    }

    #[inline]
    /// Returns the number of worker threads owned by the executor.
    pub(crate) fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Runs `task_count` indexed jobs using workers plus the caller thread.
    ///
    /// This is the intended primitive for query work units: each index
    /// represents one unit, workers pull indices atomically, and the caller also
    /// participates so a small amount of work does not wait entirely on helper
    /// threads. While waiting for helper results, the caller steals queued
    /// tasks instead of blocking. The first panic is captured and resumed only
    /// after every helper has finished, keeping the closure's borrow sound.
    pub(crate) fn scope<F>(&self, task_count: usize, func: F)
    where
        F: Fn(usize) + Send + Sync,
    {
        if task_count == 0 {
            return;
        }

        let helper_count = self
            .worker_count()
            .saturating_sub(1)
            .min(task_count.saturating_sub(1));

        if helper_count == 0 {
            for index in 0..task_count {
                func(index);
            }

            return;
        }

        let state = ScopeState::new(task_count, &func);
        let (sender, receiver) = sync_channel(helper_count);

        for helper_id in 0..helper_count {
            let task = ErasedScopeTask::new(&state);
            let sender = sender.clone();

            self.spawn(helper_id, sender, move || unsafe {
                task.run();
            });
        }

        drop(sender);

        state.run();

        let mut completed_helpers = 0;

        while completed_helpers < helper_count {
            match receiver.try_recv() {
                Ok(TaskResult {
                    outcome: TaskOutcome::Completed(()),
                    ..
                }) => completed_helpers += 1,
                Ok(TaskResult {
                    outcome: TaskOutcome::Panicked(payload),
                    ..
                }) => {
                    completed_helpers += 1;
                    state.record_panic(payload);
                }
                Err(TryRecvError::Disconnected) => {
                    state.record_panic(Box::new("query task worker stopped unexpectedly"));
                    break;
                }
                Err(TryRecvError::Empty) => {
                    if let Some(task) = self.shared.try_take_task() {
                        run_task(&self.shared, task);
                    } else {
                        match receiver.recv() {
                            Ok(TaskResult {
                                outcome: TaskOutcome::Completed(()),
                                ..
                            }) => completed_helpers += 1,
                            Ok(TaskResult {
                                outcome: TaskOutcome::Panicked(payload),
                                ..
                            }) => {
                                completed_helpers += 1;
                                state.record_panic(payload);
                            }
                            Err(_) => {
                                state.record_panic(Box::new(
                                    "query task worker stopped unexpectedly",
                                ));
                                break;
                            }
                        }
                    }
                }
            }
        }

        if let Some(payload) = state.take_panic() {
            resume_unwind(payload);
        }
    }
}

/// Persistent worker pool used by schedules and query execution.
///
/// The executor owns the worker lifetime. `spawn` is for independent
/// `'static` tasks that report through a channel, while the handle's `scope` splits a fixed
/// range of small tasks across workers and the caller thread.
pub(crate) struct TaskExecutor {
    handle: TaskExecutorHandle,
    workers: Vec<Worker>,
}

impl TaskExecutor {
    /// Creates an executor with at least one persistent worker.
    pub(crate) fn new(worker_count: usize) -> Self {
        let worker_count = worker_count.max(1);
        let shared = Arc::new(Shared::new());

        let handle = TaskExecutorHandle {
            shared: Arc::clone(&shared),
            worker_count,
        };

        let workers = (0..worker_count)
            .map(|index| Worker::new(index, Arc::clone(&shared)))
            .collect();

        Self { handle, workers }
    }

    /// Returns the shared handle for submitting tasks.
    pub(crate) fn handle(&self) -> &TaskExecutorHandle {
        &self.handle
    }

    #[inline]
    /// Returns the number of worker threads owned by the executor.
    pub(crate) fn worker_count(&self) -> usize {
        self.workers.len()
    }

    #[inline]
    /// Returns whether a worker-level failure made the executor unreliable.
    pub(crate) fn is_poisoned(&self) -> bool {
        self.handle.is_poisoned()
    }

    /// Enqueues one independent task and sends its result through `sender`.
    ///
    /// Panics inside the task are caught and returned as `TaskOutcome::Panicked`
    /// so the caller can decide when to resume unwinding.
    pub(crate) fn spawn<R, F>(&self, task_id: usize, sender: SyncSender<TaskResult<R>>, task: F)
    where
        R: Send + 'static,
        F: FnOnce() -> R + Send + 'static,
    {
        self.handle.spawn(task_id, sender, task);
    }
}

impl Drop for TaskExecutor {
    /// Shuts the queue down, then joins every worker.
    fn drop(&mut self) {
        // Wake every worker to exit, then join them. A worker panic during
        // normal executor drop is resumed; during unwinding it is swallowed
        // to avoid aborting.
        self.handle.shared.shutdown();

        for worker in &mut self.workers {
            let Some(handle) = worker.handle.take() else {
                continue;
            };

            if let Err(payload) = handle.join() {
                if !thread::panicking() {
                    resume_unwind(payload);
                }
            }
        }
    }
}

/// Shared state for one scoped indexed run.
///
/// Workers never own the closure. They receive erased pointers whose validity
/// is bounded by `TaskExecutorHandle::scope`, which waits for every helper before the
/// referenced closure can go out of scope.
struct ScopeState<'scope, F> {
    /// Next index to claim; values at or beyond `end` are exhausted.
    next: AtomicUsize,
    /// Exclusive upper bound of indices.
    end: usize,
    /// Set after a panic so other runners stop claiming indices.
    cancelled: AtomicBool,
    /// First panic payload; later panics are discarded.
    panic: Mutex<Option<Box<dyn Any + Send + 'static>>>,
    func: &'scope F,
}

impl<'scope, F> ScopeState<'scope, F>
where
    F: Fn(usize) + Send + Sync,
{
    /// Creates state for `end` indices starting from zero.
    fn new(end: usize, func: &'scope F) -> Self {
        Self {
            next: AtomicUsize::new(0),
            end,
            cancelled: AtomicBool::new(false),
            panic: Mutex::new(None),
            func,
        }
    }

    /// Claims and runs indices until exhaustion or a panic cancels the run.
    fn run(&self) {
        while !self.cancelled.load(Ordering::Acquire) {
            let index = self.next.fetch_add(1, Ordering::Relaxed);

            if index >= self.end {
                break;
            }

            let result = catch_unwind(AssertUnwindSafe(|| (self.func)(index)));

            if let Err(payload) = result {
                self.record_panic(payload);
                break;
            }
        }
    }

    /// Cancels the run and keeps the first panic payload.
    fn record_panic(&self, payload: Box<dyn Any + Send + 'static>) {
        self.cancelled.store(true, Ordering::Release);

        let mut panic = self.panic.lock().unwrap_or_else(|err| err.into_inner());

        if panic.is_none() {
            *panic = Some(payload);
        }
    }

    /// Returns the stored panic payload, if any, for the caller to resume.
    fn take_panic(&self) -> Option<Box<dyn Any + Send + 'static>> {
        self.panic
            .lock()
            .unwrap_or_else(|err| err.into_inner())
            .take()
    }
}

#[derive(Clone, Copy)]
/// Type-erased handle to scoped task state.
///
/// This keeps `TaskExecutor::spawn` simple while still allowing `scope` to run
/// a non-`'static` closure. The lifetime is enforced by `scope`, not by this
/// raw pointer container.
struct ErasedScopeTask {
    state: *const (),
    run: unsafe fn(*const ()),
}

// SAFETY: The raw state pointer is only dereferenced inside `scope`, which
// waits for every helper before the referenced state can be dropped.
unsafe impl Send for ErasedScopeTask {}

impl ErasedScopeTask {
    /// Erases the scope state behind a type-checked run function.
    fn new<F>(state: &ScopeState<'_, F>) -> Self
    where
        F: Fn(usize) + Send + Sync,
    {
        Self {
            state: state as *const ScopeState<'_, F> as *const (),
            run: run_erased_scope_task::<F>,
        }
    }

    /// Runs the erased scope state; validity is guaranteed by `scope`.
    ///
    /// # Safety
    ///
    /// The erased state must still be alive, which only `scope` guarantees.
    unsafe fn run(self) {
        unsafe { (self.run)(self.state) };
    }
}

/// Monomorphized trampoline calling `ScopeState::run` through erased state.
///
/// # Safety
///
/// `state` must have been erased from a live `ScopeState<'_, F>`.
unsafe fn run_erased_scope_task<F>(state: *const ())
where
    F: Fn(usize) + Send + Sync,
{
    let state = unsafe { &*(state as *const ScopeState<'_, F>) };
    state.run();
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::TaskExecutor;

    #[test]
    fn scope_runs_every_task_once() {
        let executor = TaskExecutor::new(4);
        let calls = Arc::new((0..64).map(|_| AtomicUsize::new(0)).collect::<Vec<_>>());
        let observed = Arc::clone(&calls);

        executor.handle().scope(64, move |index| {
            observed[index].fetch_add(1, Ordering::SeqCst);
        });

        assert!(calls.iter().all(|call| call.load(Ordering::SeqCst) == 1));
    }

    #[test]
    fn scope_propagates_panics_after_joining_helpers() {
        let executor = TaskExecutor::new(4);

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            executor.handle().scope(32, |index| {
                if index == 7 {
                    panic!("intentional query task panic");
                }
            });
        }));

        assert!(result.is_err());
    }
}
