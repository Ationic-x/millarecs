//! Schedule execution backends.
//!
//! The executor can run a schedule on the current thread or dispatch ready
//! systems onto a shared task executor. Both modes use the same execution state
//! machine so dependencies, conflicts, and deferred world mutations are handled
//! identically.

use std::{
    any::Any,
    marker::PhantomData,
    panic::resume_unwind,
    sync::mpsc::{sync_channel, Receiver, SyncSender},
    thread,
};

use crate::{
    task::{TaskExecutor, TaskOutcome, TaskResult},
    world::{execution_world::ExecutionWorld, World, WorldError},
};

use super::{
    graph::{ExecutionGraph, SystemId},
    state::ExecutionState,
    ScheduleError, SystemNode,
};

/// Failure captured while a parallel execution phase is being drained.
enum ExecutionFailure {
    /// A system returned an error from its run.
    Run {
        system: SystemId,
        source: WorldError,
    },

    /// A system panicked; the payload is resumed by the coordinator later.
    Panicked {
        system: SystemId,
        payload: Box<dyn Any + Send + 'static>,
    },
}

impl ExecutionFailure {
    /// Returns the system that produced the failure.
    fn system(&self) -> SystemId {
        match self {
            Self::Run { system, .. } | Self::Panicked { system, .. } => *system,
        }
    }
}

/// Keeps the failure of the lowest topological rank when several workers fail
/// during the same phase.
fn record_failure(
    current: &mut Option<ExecutionFailure>,
    candidate: ExecutionFailure,
    graph: &ExecutionGraph,
) {
    let candidate_rank = graph.node(candidate.system()).topological_rank();

    let replace = current
        .as_ref()
        .map(|current| candidate_rank < graph.node(current.system()).topological_rank())
        .unwrap_or(true);

    if replace {
        *current = Some(candidate);
    }
}

/// Data moved into one worker task for system execution.
///
/// The raw pointers borrow the schedule's system slice and the execution
/// world; both stay valid until the owning phase has drained every task.
struct WorkerTask {
    system_id: SystemId,
    /// Pointer to this task's node in the system slice; unique per dispatch.
    system: *mut SystemNode,
    /// Execution world whose lifetime was extended for transport only.
    world: ExecutionWorld<'static>,
}

// SAFETY:
// WorkerTask values are created only by ExecutionPhase. The phase keeps the
// original system slice and ExecutionWorld borrows alive until every dispatched
// task has returned. ExecutionState dispatches each SystemId at most once, so
// active tasks never receive the same SystemNode pointer.
unsafe impl Send for WorkerTask {}

/// Runs the wrapped system on the worker thread.
///
/// # Safety
///
/// `task.system` must point to a `SystemNode` no other task accesses, and the
/// extended `'static` borrow of `task.world` must still be backed by a live
/// execution phase.
unsafe fn execute_worker_task(task: WorkerTask) -> SystemRunOutcome {
    let system = unsafe { &mut *task.system };

    run_system(task.system_id, system, task.world)
}

/// Result produced by running one system body.
struct SystemRunOutcome {
    system_id: SystemId,
    result: Result<(), WorldError>,
    /// Whether a successful run queued deferred commands.
    has_deferred_work: bool,
}

/// Executes compiled schedules.
///
/// `ScheduleExecutor` chooses a single-threaded or multi-threaded backend at
/// construction time. Multi-threaded execution reuses persistent worker
/// threads through `TaskExecutor`.
pub struct ScheduleExecutor {
    inner: ExecutorKind,
}

/// Concrete executor selected by `ScheduleExecutor`.
enum ExecutorKind {
    /// Runs every system on the calling thread.
    Single(SingleThreadExecutor),
    /// Dispatches ready systems to persistent worker threads.
    Multi(MultiThreadExecutor),
}

/// Schedule backend that runs every system on the caller thread.
pub struct SingleThreadExecutor {
    state: ExecutionState,
}

/// Schedule backend that dispatches ready systems to worker threads.
pub struct MultiThreadExecutor {
    tasks: TaskExecutor,
    state: ExecutionState,
    /// Set when a dispatch phase ended abnormally; the backend refuses
    /// further runs.
    poisoned: bool,
}

impl Default for ScheduleExecutor {
    /// Creates an executor with one worker per available core.
    fn default() -> Self {
        let thread_count = thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .max(1);

        ScheduleExecutor::new(thread_count)
    }
}

impl ScheduleExecutor {
    /// Creates an executor with exactly `thread_count` worker threads.
    ///
    /// A count of one selects the single-threaded backend; otherwise the
    /// calling thread coordinates execution and does not run systems.
    ///
    /// # Panics
    ///
    /// Panics when `worker_count` is zero or a worker thread cannot be created.
    pub fn new(thread_count: usize) -> Self {
        assert!(
            thread_count > 0,
            "ScheduleExecutor requires at least one thread",
        );

        if thread_count == 1 {
            return Self::single_threaded();
        }

        Self::multi_threaded(thread_count)
    }

    /// Creates a schedule executor that always runs on the caller thread.
    pub fn single_threaded() -> Self {
        Self {
            inner: ExecutorKind::Single(SingleThreadExecutor::new()),
        }
    }

    /// Creates a schedule executor backed by persistent worker threads.
    pub fn multi_threaded(thread_count: usize) -> Self {
        assert!(
            thread_count > 1,
            "MultiThreadExecutor requires at least two threads",
        );

        Self {
            inner: ExecutorKind::Multi(MultiThreadExecutor::new(thread_count)),
        }
    }

    /// Returns the number of persistent worker threads owned by this executor.
    #[inline]
    pub fn worker_count(&self) -> usize {
        match &self.inner {
            ExecutorKind::Single(_) => 1,
            ExecutorKind::Multi(executor) => executor.worker_count(),
        }
    }

    /// Returns whether an internal coordination failure made this executor
    /// unusable.
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        match &self.inner {
            ExecutorKind::Single(_) => false,
            ExecutorKind::Multi(executor) => executor.is_poisoned(),
        }
    }

    /// Executes one compiled schedule against `world`.
    pub(crate) fn execute(
        &mut self,
        systems: &mut [SystemNode],
        graph: &ExecutionGraph,
        world: &mut World,
    ) -> Result<(), ScheduleError> {
        match &mut self.inner {
            ExecutorKind::Single(executor) => executor.execute(systems, graph, world),
            ExecutorKind::Multi(executor) => executor.execute(systems, graph, world),
        }
    }
}

impl SingleThreadExecutor {
    /// Creates an empty single-threaded backend.
    fn new() -> Self {
        Self {
            state: ExecutionState::new(),
        }
    }

    /// Runs systems in dependency order on the caller thread.
    fn execute(
        &mut self,
        systems: &mut [SystemNode],
        graph: &ExecutionGraph,
        world: &mut World,
    ) -> Result<(), ScheduleError> {
        self.state.reset(graph);
        self.state.evaluate_conditions(systems, world);

        while !self.state.is_finished(graph) {
            while let Some(system_id) = self.state.take_runnable(graph) {
                if !self.state.condition_passed(system_id) {
                    // A failed run_if condition skips the system while its
                    // dependents still run.
                    self.state.complete_run(graph, system_id, false);
                    continue;
                }

                if systems[system_id.index()].system.is_exclusive() {
                    // Exclusive systems run inline with the whole world:
                    // their access conflicts with everything, so nothing
                    // else can be running here anyway.
                    let result = systems[system_id.index()].system.run_exclusive(world);

                    match result {
                        Ok(()) => {
                            self.state.complete_run(graph, system_id, false);
                        }

                        Err(source) => {
                            self.state.cancel_run(graph, system_id);
                            discard_all_deferred(systems, world);

                            let system_name = systems[system_id.index()].system.name();

                            return Err(ScheduleError::SystemRun {
                                system: system_name,
                                source,
                            });
                        }
                    }

                    continue;
                }

                let outcome = {
                    let execution_world = world.execution_world();
                    let node = &mut systems[system_id.index()];

                    run_system(system_id, node, execution_world)
                };

                match outcome.result {
                    Ok(()) => {
                        self.state.complete_run(
                            graph,
                            outcome.system_id,
                            outcome.has_deferred_work,
                        );
                    }

                    Err(source) => {
                        self.state.cancel_run(graph, outcome.system_id);
                        discard_all_deferred(systems, world);

                        let system_name = systems[outcome.system_id.index()].system.name();

                        return Err(ScheduleError::SystemRun {
                            system: system_name,
                            source,
                        });
                    }
                }
            }

            if self.state.has_pending_deferred() {
                apply_pending_deferred(systems, graph, &mut self.state, world)?;
                continue;
            }

            if self.state.is_finished(graph) {
                break;
            }

            unreachable!("execution state stalled without runnable systems or deferred work");
        }

        Ok(())
    }
}

impl MultiThreadExecutor {
    /// Creates a multi-threaded backend with persistent workers.
    fn new(worker_count: usize) -> Self {
        assert!(
            worker_count > 0,
            "MultiThreadExecutor requires at least one worker",
        );

        Self {
            tasks: TaskExecutor::new(worker_count),
            state: ExecutionState::new(),
            poisoned: false,
        }
    }

    #[inline]
    /// Returns the number of worker threads owned by this backend.
    fn worker_count(&self) -> usize {
        self.tasks.worker_count()
    }

    #[inline]
    /// Returns whether a worker failure poisoned this backend.
    fn is_poisoned(&self) -> bool {
        self.poisoned || self.tasks.is_poisoned()
    }

    /// Runs systems across parallel phases with inline exclusive handling.
    fn execute(
        &mut self,
        systems: &mut [SystemNode],
        graph: &ExecutionGraph,
        world: &mut World,
    ) -> Result<(), ScheduleError> {
        let Self {
            tasks,
            state,
            poisoned,
        } = self;

        if *poisoned || tasks.is_poisoned() {
            return Err(ScheduleError::ExecutorPoisoned);
        }

        state.reset(graph);
        state.evaluate_conditions(systems, world);

        while !state.is_finished(graph) {
            // Exclusive systems run inline with exclusive world access; they
            // conflict with everything, so no worker task can be active here.
            while let Some(system_id) = state.peek_runnable(graph) {
                if !state.condition_passed(system_id) {
                    state.take_runnable(graph);
                    state.complete_run(graph, system_id, false);
                    continue;
                }

                if !systems[system_id.index()].system.is_exclusive() {
                    // The head runnable is a normal system: hand control to
                    // the parallel phase below.
                    break;
                }

                state.take_runnable(graph);

                let result = systems[system_id.index()].system.run_exclusive(world);

                if let Err(source) = result {
                    state.cancel_run(graph, system_id);
                    discard_all_deferred(systems, world);

                    let system_name = systems[system_id.index()].system.name();

                    return Err(ScheduleError::SystemRun {
                        system: system_name,
                        source,
                    });
                }

                state.complete_run(graph, system_id, false);
            }

            if state.is_finished(graph) {
                break;
            }

            let phase_result = {
                let execution_world = world.execution_world_with_tasks(Some(tasks.handle()));

                run_execution_phase(tasks, poisoned, systems, graph, state, execution_world)
            };

            if let Err(failure) = phase_result {
                match failure {
                    ExecutionFailure::Run { system, source } => {
                        discard_all_deferred(systems, world);

                        let system_name = systems[system.index()].system.name();

                        return Err(ScheduleError::SystemRun {
                            system: system_name,
                            source,
                        });
                    }

                    ExecutionFailure::Panicked { payload, .. } => {
                        resume_unwind(payload);
                    }
                }
            }

            debug_assert!(!state.has_running_systems(),);

            if state.has_pending_deferred() {
                apply_pending_deferred(systems, graph, state, world)?;

                continue;
            }

            if state.is_finished(graph) {
                break;
            }

            // A phase ends when the runnable head is exclusive: loop back so
            // the coordinator runs it inline.
            debug_assert!(
                state.peek_runnable(graph).is_some(),
                "execution state stalled without runnable systems \
                 or deferred work"
            );
        }

        Ok(())
    }
}

/// Runs one parallel phase until all dispatched systems finish.
///
/// A phase ends when no runnable systems remain and every active worker task has
/// reported back. Deferred work is not applied here; the caller performs that
/// in an exclusive world phase.
fn run_execution_phase(
    tasks: &TaskExecutor,
    executor_poisoned: &mut bool,
    systems: &mut [SystemNode],
    graph: &ExecutionGraph,
    state: &mut ExecutionState,
    world: ExecutionWorld<'_>,
) -> Result<(), ExecutionFailure> {
    let mut phase = ExecutionPhase::new(tasks, executor_poisoned, systems, world);

    let mut failure = None;

    loop {
        if failure.is_none() {
            while phase.has_dispatch_capacity() {
                let Some(system_id) = state.peek_runnable(graph) else {
                    break;
                };

                if !state.condition_passed(system_id) {
                    // A failed run_if condition skips the system while its
                    // dependents still run.
                    state.take_runnable(graph);
                    state.complete_run(graph, system_id, false);
                    continue;
                }

                if state.is_exclusive(system_id) {
                    // Exclusive systems cannot enter a parallel phase: end
                    // the phase so the coordinator runs them inline.
                    break;
                }

                state.take_runnable(graph);
                phase.dispatch(system_id);
            }
        }

        if !phase.has_active_tasks() {
            return match failure {
                Some(failure) => Err(failure),
                None => Ok(()),
            };
        }

        let task_result = phase.receive();

        match task_result.outcome {
            TaskOutcome::Completed(outcome) => match outcome.result {
                Ok(()) => {
                    state.complete_run(graph, outcome.system_id, outcome.has_deferred_work);
                }

                Err(source) => {
                    state.cancel_run(graph, outcome.system_id);

                    record_failure(
                        &mut failure,
                        ExecutionFailure::Run {
                            system: outcome.system_id,
                            source,
                        },
                        graph,
                    );
                }
            },

            TaskOutcome::Panicked(payload) => {
                let system_id = SystemId::new(task_result.task_id);
                state.cancel_run(graph, system_id);

                record_failure(
                    &mut failure,
                    ExecutionFailure::Panicked {
                        system: system_id,
                        payload,
                    },
                    graph,
                );
            }
        }
    }
}

/// Applies queued deferred commands in topological order.
fn apply_pending_deferred(
    systems: &mut [SystemNode],
    graph: &ExecutionGraph,
    state: &mut ExecutionState,
    world: &mut World,
) -> Result<(), ScheduleError> {
    debug_assert!(
        !state.has_running_systems(),
        "deferred work requires exclusive World access",
    );

    while let Some(system_id) = state.take_pending_deferred(graph) {
        let (system_name, result) = {
            let system = &mut systems[system_id.index()].system;

            (system.name(), system.apply_deferred(world))
        };

        if let Err(source) = result {
            discard_all_deferred(systems, world);

            return Err(ScheduleError::SystemApplyDeferred {
                system: system_name,
                source,
            });
        }

        state.complete_deferred(graph, system_id);
    }

    Ok(())
}

/// Drops queued deferred work after a system failure.
fn discard_all_deferred(systems: &mut [SystemNode], world: &mut World) {
    for node in systems {
        node.system.discard_deferred(world);
    }
}

/// Runs one system body and records whether it produced deferred work.
fn run_system(
    system_id: SystemId,
    node: &mut SystemNode,
    world: ExecutionWorld<'_>,
) -> SystemRunOutcome {
    let result = node.system.run(world);

    let has_deferred_work = result.is_ok() && node.system.has_deferred_work();

    SystemRunOutcome {
        system_id,
        result,
        has_deferred_work,
    }
}

/// Owner for one multi-threaded dispatch phase.
///
/// The phase borrows systems and the execution world, extends those lifetimes
/// only for transport through worker tasks, and drains all active tasks before
/// it can be dropped.
struct ExecutionPhase<'executor, 'systems, 'world> {
    tasks: &'executor TaskExecutor,
    /// Executor-level poison flag, raised when the phase is abandoned.
    executor_poisoned: &'executor mut bool,
    /// Result channel bounded by the worker count, so undelivered results
    /// apply backpressure instead of buffering without limit.
    result_sender: SyncSender<TaskResult<SystemRunOutcome>>,
    result_receiver: Receiver<TaskResult<SystemRunOutcome>>,

    /// Base pointer of the borrowed system slice.
    systems: *mut SystemNode,
    /// Length of that slice; keeps raw-pointer indexing in bounds.
    system_count: usize,

    /// Execution world transmuted to `'static` for transport through workers.
    world: ExecutionWorld<'static>,

    /// Dispatched tasks that have not reported back yet.
    active_count: usize,

    _systems: PhantomData<&'systems mut [SystemNode]>,
    _world: PhantomData<ExecutionWorld<'world>>,
}

impl<'executor, 'systems, 'world> ExecutionPhase<'executor, 'systems, 'world> {
    /// Creates a dispatch phase over a borrowed system slice and execution world.
    fn new(
        tasks: &'executor TaskExecutor,
        executor_poisoned: &'executor mut bool,
        systems: &'systems mut [SystemNode],
        world: ExecutionWorld<'world>,
    ) -> Self {
        // SAFETY:
        // The lifetime is extended only for transport through persistent worker
        // threads. ExecutionPhase retains the original borrows through its
        // PhantomData fields and cannot finish until all active tasks return.
        // Its Drop implementation also drains tasks during unwinding.
        let world = unsafe {
            std::mem::transmute::<ExecutionWorld<'world>, ExecutionWorld<'static>>(world)
        };

        let (result_sender, result_receiver) = sync_channel(tasks.worker_count());

        Self {
            tasks,
            executor_poisoned,
            result_sender,
            result_receiver,

            systems: systems.as_mut_ptr(),
            system_count: systems.len(),

            world,

            active_count: 0,

            _systems: PhantomData,
            _world: PhantomData,
        }
    }

    #[inline]
    /// Returns whether another task can be dispatched without exceeding workers.
    fn has_dispatch_capacity(&self) -> bool {
        self.active_count < self.tasks.worker_count()
    }

    #[inline]
    /// Returns whether any dispatched task has not reported back yet.
    fn has_active_tasks(&self) -> bool {
        self.active_count != 0
    }

    /// Dispatches a ready system to the task executor.
    fn dispatch(&mut self, system_id: SystemId) {
        assert!(
            system_id.index() < self.system_count,
            "system index out of bounds",
        );

        let system = unsafe { self.systems.add(system_id.index()) };
        let sender = self.result_sender.clone();
        let task = WorkerTask {
            system_id,
            system,
            world: self.world,
        };

        self.tasks.spawn(system_id.index(), sender, move || unsafe {
            execute_worker_task(task)
        });

        self.active_count += 1;
    }

    /// Waits for one task result and updates the active-task count.
    fn receive(&mut self) -> TaskResult<SystemRunOutcome> {
        assert!(self.active_count > 0, "cannot receive without active tasks",);

        let result = self
            .result_receiver
            .recv()
            .expect("all ECS workers unexpectedly stopped");

        self.active_count -= 1;

        result
    }

    /// Drains unfinished tasks during abnormal exit.
    ///
    /// Outcomes are forgotten because a panic payload's destructor may panic
    /// again while unwinding; a closed worker channel poisons the executor
    /// instead of blocking forever.
    fn drain_remaining(&mut self) {
        while self.active_count != 0 {
            let result = match self.result_receiver.recv() {
                Ok(result) => result,

                Err(_) => {
                    *self.executor_poisoned = true;
                    self.active_count = 0;
                    break;
                }
            };

            self.active_count -= 1;

            // A panic payload may own a value whose destructor also panics.
            // Forgetting outcomes while already unwinding prevents a second
            // panic from aborting the process. This path is bounded by the
            // worker count and permanently poisons the executor.
            std::mem::forget(result.outcome);
        }
    }
}

impl Drop for ExecutionPhase<'_, '_, '_> {
    /// Drains outstanding tasks on abnormal exits, so borrowed systems and the
    /// world are never released while a worker still uses them.
    fn drop(&mut self) {
        let abnormal_exit = thread::panicking() || self.active_count != 0;

        if abnormal_exit {
            *self.executor_poisoned = true;
            self.drain_remaining();
        }

        debug_assert_eq!(
            self.active_count, 0,
            "execution phase released while tasks were active",
        );
    }
}
