//! Dependency-aware parallel system scheduling.
//!
//! A schedule owns systems and their compiled dependency graph.
//! [`ScheduleExecutor`] owns persistent worker threads and can be shared by
//! multiple schedules executed sequentially.
//!
//! Systems without dependencies may run concurrently when their declared
//! accesses do not conflict. Structural changes remain deferred until an
//! exclusive world phase.

mod executor;
mod graph;
mod state;

pub use executor::ScheduleExecutor;
use millarecs_macros::all_tuples;

use std::{
    any::TypeId,
    collections::HashMap,
    fmt::{self},
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
};

use crate::{
    system::{
        schedule::graph::{DependencyGraph, ExecutionGraph, SystemId},
        IntoSystem, IntoSystems, System,
    },
    world::{World, WorldError, WorldId},
};

/// Type-identity label a system can be ordered against.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SystemLabel {
    type_id: TypeId,
    /// Type name kept for diagnostics; identity comes from `type_id`.
    name: &'static str,
}

impl SystemLabel {
    /// Creates the label of the type `T`.
    fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
        }
    }
}

/// Type-identity label for a named group of systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SystemSetLabel {
    type_id: TypeId,
    /// Type name kept for diagnostics; identity comes from `type_id`.
    name: &'static str,
}

impl SystemSetLabel {
    /// Creates the label of the set type `T`.
    fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
        }
    }
}

/// Deferred-work barrier inserted at a fixed schedule position.
#[derive(Clone, Copy, Debug)]
struct DeferredBarrier {
    /// Number of systems added before the barrier.
    system_index: usize,
}

/// Condition evaluated once per execution cycle with shared access to the
/// whole world; a system runs only when its condition returns `true`.
pub(crate) type ConditionFn = Box<dyn Fn(&World) -> bool + Send + Sync>;

/// One registered system with its condition, label, and ordering constraints.
pub(crate) struct SystemNode {
    system: Box<dyn System>,
    /// Optional condition evaluated once per cycle; the system is skipped when it fails.
    condition: Option<ConditionFn>,
    /// Optional label other systems can order against.
    label: Option<SystemLabel>,
    sets: Vec<SystemSetLabel>,
    /// Labels this system must run before.
    before: Vec<SystemLabel>,
    /// Labels this system must run after.
    after: Vec<SystemLabel>,
}

/// Builder returned by [`Schedule::add_system`], offering ordering, labeling,
/// and condition configuration for the just-added system.
pub struct SystemConfig<'a> {
    schedule: &'a mut Schedule,
    index: usize,
}

impl<'a> SystemConfig<'a> {
    /// Assigns a unique type-based label to the configured system.
    pub fn label<L: 'static>(self) -> Self {
        self.schedule.systems[self.index].label = Some(SystemLabel::of::<L>());
        self.schedule.invalidate_graph();
        self
    }

    /// Runs the configured system after the system labelled `L` completes,
    /// including its deferred work.
    pub fn after<L: 'static>(self) -> Self {
        self.schedule.systems[self.index]
            .after
            .push(SystemLabel::of::<L>());
        self.schedule.invalidate_graph();
        self
    }

    /// Runs the configured system before the system labelled `L`.
    pub fn before<L: 'static>(self) -> Self {
        self.schedule.systems[self.index]
            .before
            .push(SystemLabel::of::<L>());
        self.schedule.invalidate_graph();
        self
    }

    /// Runs the configured system only when `condition` returns `true`.
    ///
    /// Conditions are evaluated once per execution cycle, before any system
    /// of the cycle runs, with shared access to the whole world. A system
    /// whose condition fails is skipped for that cycle; its dependents still
    /// run. Calling this again replaces the previous condition.
    ///
    /// Conditions never invalidate the compiled graph: they are evaluated at
    /// run time, so setting one costs no recompilation.
    pub fn run_if<F>(self, condition: F) -> Self
    where
        F: Fn(&World) -> bool + Send + Sync + 'static,
    {
        self.schedule.systems[self.index].condition = Some(Box::new(condition));
        self
    }

    /// Registers the configured system in the set `S`.
    pub fn in_sets<S: 'static>(self) -> Self {
        self.schedule.systems[self.index]
            .sets
            .push(SystemSetLabel::of::<S>());

        self.schedule.invalidate_graph();
        self
    }
}

/// Ordering configuration accumulated for one system set.
#[derive(Default)]
struct SystemSetConfig {
    /// Labels this set must complete before.
    before: Vec<SystemSetLabel>,
    /// Labels that must complete before this set runs.
    after: Vec<SystemSetLabel>,
    /// Chains member systems in registration order.
    chain: bool,
}

/// Builder returned by [`Schedule::configure_set`], offering ordering and
/// chaining configuration for the set.
pub struct SystemSetConfigHandle<'a> {
    schedule: &'a mut Schedule,
    set: SystemSetLabel,
}

impl<'a> SystemSetConfigHandle<'a> {
    /// Runs every system in this set after the system labelled `S`.
    pub fn after<S: 'static>(self) -> Self {
        self.schedule
            .set_configs
            .entry(self.set)
            .or_default()
            .after
            .push(SystemSetLabel::of::<S>());

        self.schedule.invalidate_graph();

        self
    }

    /// Runs every system in this set before the system labelled `S`.
    pub fn before<S: 'static>(self) -> Self {
        self.schedule
            .set_configs
            .entry(self.set)
            .or_default()
            .before
            .push(SystemSetLabel::of::<S>());

        self.schedule.invalidate_graph();

        self
    }

    /// Chains the systems of this set in registration order.
    pub fn chain(self) -> Self {
        self.schedule.set_configs.entry(self.set).or_default().chain = true;

        self.schedule.invalidate_graph();

        self
    }
}

/// Batch configuration for one or more systems.
///
/// This type is produced by calling methods such as `.after::<L>()` or
/// `.before::<L>()` on a single system or on a tuple of systems before passing
/// them to [`Schedule::add_systems`].
pub struct SystemBatchConfig<S> {
    systems: S,
    sets: Vec<SystemSetLabel>,
    before: Vec<SystemLabel>,
    after: Vec<SystemLabel>,
    chain: bool,
}

impl<S> SystemBatchConfig<S> {
    /// Wraps `systems` in a batch with no constraints yet.
    fn new(systems: S) -> Self {
        Self {
            systems,
            sets: Vec::new(),
            before: Vec::new(),
            after: Vec::new(),
            chain: false,
        }
    }

    /// Registers every system in this batch in the set `Set`.
    pub fn in_set<Set: 'static>(mut self) -> Self {
        self.sets.push(SystemSetLabel::of::<Set>());
        self
    }

    /// Runs every system in this batch after the system labelled `L` completes,
    /// including its deferred work.
    pub fn after<L: 'static>(mut self) -> Self {
        self.after.push(SystemLabel::of::<L>());
        self
    }

    /// Runs every system in this batch before the system labelled `L`.
    pub fn before<L: 'static>(mut self) -> Self {
        self.before.push(SystemLabel::of::<L>());
        self
    }

    /// Chains the systems of this batch in registration order.
    pub fn chain(mut self) -> Self {
        self.chain = true;
        self
    }
}

/// Extension trait that allows configuring a single system or a tuple of
/// systems before adding them to a schedule.
///
/// This makes expressions such as `(a, b, c).after::<SomeLabel>()` or
/// `(a, b, c).chain()` valid arguments to `Schedule::add_systems`.
pub trait SystemBatchConfigExt: Sized {
    /// Registers every system in this batch in the set `S`.
    fn in_set<S: 'static>(self) -> SystemBatchConfig<Self> {
        SystemBatchConfig::new(self).in_set::<S>()
    }

    /// Runs every system in this batch after the system labelled `L`.
    fn after<L: 'static>(self) -> SystemBatchConfig<Self> {
        SystemBatchConfig::new(self).after::<L>()
    }

    /// Runs every system in this batch before the system labelled `L`.
    fn before<L: 'static>(self) -> SystemBatchConfig<Self> {
        SystemBatchConfig::new(self).before::<L>()
    }

    /// Chains the systems of this batch in registration order.
    fn chain(self) -> SystemBatchConfig<Self> {
        SystemBatchConfig::new(self).chain()
    }
}

macro_rules! impl_system_batch_config_ext_tuple {
    ($(#[$meta:meta])* $($S:ident),*) => {
        $(#[$meta])*
        impl<$($S),*> SystemBatchConfigExt for ($($S,)*) {}
    };
}

all_tuples!(impl_system_batch_config_ext_tuple, 1, 12, S);

impl<S, Marker> IntoSystems<Marker> for SystemBatchConfig<S>
where
    S: IntoSystems<Marker>,
{
    /// Adds the batched systems, then applies the batch configuration to them.
    ///
    /// A chained batch links consecutive systems through edges recorded
    /// directly in `schedule.chain_edges`; only the first system inherits the
    /// batch's `after` labels and only the last inherits its `before` labels.
    /// An unchained batch applies every constraint to every member.
    fn add_to_schedule(self, schedule: &mut Schedule) {
        let start = schedule.systems.len();

        let SystemBatchConfig {
            systems,
            sets,
            before,
            after,
            chain,
        } = self;

        systems.add_to_schedule(schedule);

        let end = schedule.systems.len();

        for node in &mut schedule.systems[start..end] {
            node.sets.extend(sets.iter().copied());
        }

        if chain && start < end {
            let end = end - 1;

            if let Some(first) = schedule.systems.get_mut(start) {
                first.after.extend(after.iter().copied());
            }

            if let Some(last) = schedule.systems.get_mut(end) {
                last.before.extend(before.iter().copied());
            }

            for index in start..end {
                schedule
                    .chain_edges
                    .push((SystemId::new(index), SystemId::new(index + 1)));
            }
        } else {
            for node in &mut schedule.systems[start..end] {
                node.before.extend(before.iter().copied());
                node.after.extend(after.iter().copied());
            }
        }

        schedule.invalidate_graph();
    }
}

/// Errors produced while building or running a schedule.
#[derive(Debug)]
pub enum ScheduleError {
    /// The same label was assigned to two systems.
    DuplicateLabel(&'static str),
    /// A system referenced a label no system carries.
    UnknownDependency {
        /// Name of the system declaring the dependency.
        system: &'static str,
        /// Label that no system provides.
        label: &'static str,
    },
    /// The explicit ordering constraints contain a cycle.
    DependencyCycle,
    /// A system failed to initialize its parameters.
    SystemInitialize {
        /// Name of the failing system.
        system: &'static str,
        /// Underlying error reported by initialization.
        source: WorldError,
    },
    /// A system failed during execution.
    SystemRun {
        /// Name of the failing system.
        system: &'static str,
        /// Underlying error reported by the system.
        source: WorldError,
    },
    /// A system failed while applying its deferred work.
    SystemApplyDeferred {
        /// Name of the failing system.
        system: &'static str,
        /// Underlying error reported by the world.
        source: WorldError,
    },
    /// A system without parameter access was scheduled for parallel execution.
    MissingSystemAccess {
        /// Name of the offending system.
        system: &'static str,
    },
    /// A worker thread panicked and the executor cannot recover.
    ExecutorPoisoned,
    /// A previous schedule run panicked; this schedule is unusable.
    SchedulePoisoned,
    /// The schedule was bound to one world but run against another.
    WorldMismatch {
        /// Identifier of the world the schedule was bound to.
        expected: WorldId,
        /// Identifier of the world it was run against.
        actual: WorldId,
    },
}

impl fmt::Display for ScheduleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateLabel(label) => {
                write!(f, "duplicate system label `{label}`")
            }
            Self::UnknownDependency { system, label } => {
                write!(f, "system `{system}` references unknown label `{label}`")
            }
            Self::DependencyCycle => f.write_str("system dependencies contain a cycle"),
            Self::SystemInitialize { system, source } => {
                write!(f, "failed to initialize system `{system}`: {source}")
            }
            Self::SystemRun { system, source } => {
                write!(f, "failed to run system `{system}`: {source}")
            }
            Self::SystemApplyDeferred { system, source } => {
                write!(
                    f,
                    "failed to apply deferred work for system `{system}`: {source}"
                )
            }
            Self::MissingSystemAccess { system } => {
                write!(
                    f,
                    "initialized system `{system}` did not expose its access set"
                )
            }
            Self::ExecutorPoisoned => {
                f.write_str("schedule executor is poisoned by a coordination failure")
            }
            Self::SchedulePoisoned => {
                f.write_str("schedule is poisoned by a previous system panic")
            }
            Self::WorldMismatch { expected, actual } => {
                write!(
                    f,
                    "world id mismatch. Expected world id: {expected} Received world id: {actual}"
                )
            }
        }
    }
}

impl std::error::Error for ScheduleError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SystemInitialize { source, .. }
            | Self::SystemRun { source, .. }
            | Self::SystemApplyDeferred { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Adds an edge from every member of `from` to every member of `to`.
///
/// Missing sets contribute nothing, and a system belonging to both sets is
/// skipped so no self-edge is created.
fn add_set_edges(
    edges: &mut Vec<(SystemId, SystemId)>,
    set_members: &HashMap<SystemSetLabel, Vec<SystemId>>,
    from: SystemSetLabel,
    to: SystemSetLabel,
) {
    let Some(from_members) = set_members.get(&from) else {
        return;
    };

    let Some(to_members) = set_members.get(&to) else {
        return;
    };

    for &left in from_members {
        for &right in to_members {
            if left != right {
                edges.push((left, right));
            }
        }
    }
}

/// Owns systems and their compiled dependency graph.
///
/// Worker threads are supplied externally through [`ScheduleExecutor`], allowing
/// multiple schedules to share the same execution infrastructure.
pub struct Schedule {
    /// World this schedule was first initialized against; `None` until then.
    world_id: Option<WorldId>,
    systems: Vec<SystemNode>,
    set_configs: HashMap<SystemSetLabel, SystemSetConfig>,
    /// Direct ordering edges recorded by chained batches, kept apart from
    /// `SystemNode` because they address systems by dense index at add time.
    chain_edges: Vec<(SystemId, SystemId)>,
    deferred_barriers: Vec<DeferredBarrier>,
    /// Compiled graph; `None` while dirty or before the first initialize.
    execution_graph: Option<ExecutionGraph>,
    /// Set after a panic in `initialize` or `run`; the schedule never recovers.
    poisoned: bool,
}

impl Default for Schedule {
    /// Creates an empty schedule, like [`Schedule::new`].
    fn default() -> Self {
        Self::new()
    }
}

impl Schedule {
    /// Creates an empty schedule.
    ///
    /// The schedule becomes permanently associated with the first world used
    /// to initialize it.
    pub fn new() -> Self {
        Self {
            world_id: None,
            systems: Vec::new(),
            set_configs: HashMap::new(),
            chain_edges: Vec::new(),
            deferred_barriers: Vec::new(),
            execution_graph: None,
            poisoned: false,
        }
    }

    /// Returns the number of systems owned by this schedule.
    #[inline]
    pub fn len(&self) -> usize {
        self.systems.len()
    }

    /// Whether this schedule contains no systems.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.systems.is_empty()
    }

    /// `true` when a previous system panic made this schedule unusable.
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    /// Forgets the compiled graph so the next initialize rebuilds it.
    ///
    /// Every structural change calls this; `run_if` does not, because
    /// conditions are evaluated at run time and leave the graph valid.
    fn invalidate_graph(&mut self) {
        self.execution_graph = None;
    }

    /// Returns a builder for the ordering constraints of the set `S`.
    pub fn configure_set<S: 'static>(&mut self) -> SystemSetConfigHandle<'_> {
        let set = SystemSetLabel::of::<S>();

        self.set_configs.entry(set).or_default();

        SystemSetConfigHandle {
            schedule: self,
            set,
        }
    }

    /// Adds a system to the end of the execution list.
    pub fn add_system<S, Marker>(&mut self, system: S) -> SystemConfig<'_>
    where
        S: IntoSystem<Marker>,
        S::System: 'static,
    {
        let index = self.systems.len();

        self.systems.push(SystemNode {
            system: Box::new(system.into_system()),
            condition: None,
            label: None,
            sets: Vec::new(),
            before: Vec::new(),
            after: Vec::new(),
        });

        self.invalidate_graph();

        SystemConfig {
            schedule: self,
            index,
        }
    }

    /// Adds a tuple of systems in registration order, with optional batch
    /// configuration.
    pub fn add_systems<S, Marker>(&mut self, systems: S)
    where
        S: IntoSystems<Marker>,
    {
        systems.add_to_schedule(self);
    }

    /// Inserts an explicit deferred-work barrier at the current schedule position.
    ///
    /// All systems added before this barrier must complete, including their
    /// deferred work, before systems added after this barrier may start.
    ///
    /// This is useful when a whole block of systems produces deferred structural
    /// changes or events that the following block must observe.
    pub fn add_deferred_barrier(&mut self) -> &mut Self {
        self.deferred_barriers.push(DeferredBarrier {
            system_index: self.systems.len(),
        });

        self.invalidate_graph();

        self
    }

    /// Adds the ordering edges implied by the deferred barriers.
    ///
    /// Barriers cut the system list into consecutive segments; every system of
    /// one non-empty segment gains an edge to every system of the next
    /// non-empty segment, so empty segments never break the chain.
    fn add_deferred_barrier_edges(&self, edges: &mut Vec<(SystemId, SystemId)>) {
        if self.deferred_barriers.is_empty() {
            return;
        }

        let mut cuts = self
            .deferred_barriers
            .iter()
            .map(|barrier| barrier.system_index)
            .collect::<Vec<_>>();

        cuts.sort();
        cuts.dedup();
        cuts.push(self.systems.len());

        let mut start = 0usize;
        let mut previous_non_empty: Option<std::ops::Range<usize>> = None;

        for end in cuts {
            let current = start..end;

            if !current.is_empty() {
                if let Some(previous) = previous_non_empty.clone() {
                    for left in previous {
                        for right in current.clone() {
                            edges.push((SystemId::new(left), SystemId::new(right)));
                        }
                    }
                }

                previous_non_empty = Some(current);
            }

            start = end;
        }
    }

    /// Assembles every explicit ordering edge into a dependency graph.
    ///
    /// The edge sources are, in order: chained batches, per-system
    /// `before`/`after` labels, set chaining and set-to-set ordering, and
    /// deferred barriers. Duplicate labels and unknown labels are rejected.
    fn build_dependency_graph(&self) -> Result<DependencyGraph, ScheduleError> {
        let mut labels = HashMap::with_capacity(self.systems.len());

        for (index, node) in self.systems.iter().enumerate() {
            let Some(label) = node.label else {
                continue;
            };

            if labels.insert(label, SystemId::new(index)).is_some() {
                return Err(ScheduleError::DuplicateLabel(label.name));
            }
        }

        let mut edges = Vec::new();
        let mut set_members: HashMap<SystemSetLabel, Vec<SystemId>> = HashMap::new();

        edges.extend(self.chain_edges.iter().copied());

        for (index, node) in self.systems.iter().enumerate() {
            let current = SystemId::new(index);

            for &set in &node.sets {
                set_members.entry(set).or_default().push(current);
            }

            for label in &node.before {
                let Some(&following) = labels.get(label) else {
                    return Err(ScheduleError::UnknownDependency {
                        system: node.system.name(),
                        label: label.name,
                    });
                };

                edges.push((current, following));
            }

            for label in &node.after {
                let Some(&previous) = labels.get(label) else {
                    return Err(ScheduleError::UnknownDependency {
                        system: node.system.name(),
                        label: label.name,
                    });
                };

                edges.push((previous, current));
            }
        }

        for (&set, config) in &self.set_configs {
            if config.chain {
                if let Some(members) = set_members.get(&set) {
                    for pair in members.windows(2) {
                        edges.push((pair[0], pair[1]));
                    }
                }
            }

            for &before in &config.before {
                add_set_edges(&mut edges, &set_members, set, before);
            }

            for &after in &config.after {
                add_set_edges(&mut edges, &set_members, after, set);
            }
        }

        self.add_deferred_barrier_edges(&mut edges);

        Ok(DependencyGraph::from_edges(self.systems.len(), edges))
    }

    /// Initializes all systems that belong to this schedule.
    ///
    /// A panic from custom system initialization is resumed after marking the
    /// schedule as poisoned.
    pub fn initialize(&mut self, world: &mut World) -> Result<(), ScheduleError> {
        if self.poisoned {
            return Err(ScheduleError::SchedulePoisoned);
        }

        self.bind_or_validate_world(world.id())?;
        let result = catch_unwind(AssertUnwindSafe(|| self.initialize_inner(world)));

        match result {
            Ok(result) => result,
            Err(payload) => {
                self.poisoned = true;
                resume_unwind(payload);
            }
        }
    }

    /// Initializes every uninitialized system in topological order and
    /// compiles the execution graph; a no-op while the graph stays valid.
    fn initialize_inner(&mut self, world: &mut World) -> Result<(), ScheduleError> {
        if self.execution_graph.is_some() {
            return Ok(());
        }

        let dependencies = self.build_dependency_graph()?;

        let topological_order = dependencies
            .kahn_sort()
            .map_err(|_| ScheduleError::DependencyCycle)?;

        for system_id in topological_order.iter().copied() {
            let system = &mut self.systems[system_id.index()].system;

            if system.is_initialized() {
                continue;
            }

            system
                .initialize(world)
                .map_err(|source| ScheduleError::SystemInitialize {
                    system: system.name(),
                    source,
                })?;
        }

        let accesses = self
            .systems
            .iter()
            .map(|node| {
                node.system
                    .access()
                    .ok_or(ScheduleError::MissingSystemAccess {
                        system: node.system.name(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let execution_graph = ExecutionGraph::compile(dependencies, topological_order, &accesses);

        self.execution_graph = Some(execution_graph);

        Ok(())
    }

    /// Runs every system once using the supplied executor.
    ///
    /// Explicit dependencies determine ordering and deferred-command visibility.
    /// Independent systems may run concurrently when their access sets are
    /// compatible.
    ///
    /// The call does not return while a worker still borrows a system or the
    /// world. A system panic is resumed on the calling thread after outstanding
    /// tasks finish and deferred state is discarded; the schedule is then
    /// poisoned, while the executor remains reusable.
    pub fn run(
        &mut self,
        world: &mut World,
        executor: &mut ScheduleExecutor,
    ) -> Result<(), ScheduleError> {
        if self.poisoned {
            return Err(ScheduleError::SchedulePoisoned);
        }

        world.event_update();
        world
            .dispatch_changed()
            .expect("change observation dispatch must remain valid for its world");
        self.initialize(world)?;
        world.advance_tick();

        let graph = self
            .execution_graph
            .as_ref()
            .expect("schedule must be initialized");

        let result = catch_unwind(AssertUnwindSafe(|| {
            executor.execute(&mut self.systems, graph, world)
        }));

        match result {
            Ok(result) => result,

            Err(payload) => {
                self.poisoned = true;

                // The execution phase has already joined every borrowed task,
                // so exclusive world access is available again. Clearing every
                // queue prevents reservations from surviving a failed run.
                for node in &mut self.systems {
                    node.system.discard_deferred(world);
                }

                resume_unwind(payload);
            }
        }
    }

    /// Binds the schedule to `actual` on first use and rejects any other world
    /// afterwards.
    fn bind_or_validate_world(&mut self, actual: WorldId) -> Result<(), ScheduleError> {
        match self.world_id {
            Some(expected) => {
                if expected != actual {
                    Err(ScheduleError::WorldMismatch { expected, actual })
                } else {
                    Ok(())
                }
            }
            None => {
                self.world_id = Some(actual);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests;
