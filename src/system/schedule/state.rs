//! Mutable execution state for one schedule run.
//!
//! The state tracks which systems are waiting, ready, running, waiting for
//! deferred work, or completed. It is rebuilt from the compiled graph before
//! each schedule execution.

use crate::{system::schedule::SystemNode, world::World};

use super::graph::{ExecutionGraph, SystemId};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Runtime state of one system during a schedule execution.
enum NodeState {
    /// The node cannot run yet because dependencies or conflicts block it.
    #[default]
    Waiting,
    /// The node can be dispatched.
    Ready,
    /// The node is currently running.
    Running,
    /// The node finished its run but still has deferred world work to apply.
    Deferred,
    /// The node and its deferred work are complete.
    Completed,
}

#[derive(Default)]
/// Bit set ordered by topological rank.
///
/// `first_non_empty` avoids scanning from the start of the set after every
/// insertion/removal, which keeps ready-node selection cheap for larger
/// schedules.
struct RankedSet {
    words: Vec<u64>,
    /// Index of the first word that may still hold a set bit.
    first_non_empty: usize,
}

impl RankedSet {
    /// Clears all bits and resets the scan cursor.
    fn reset(&mut self, system_count: usize) {
        let word_count = system_count.div_ceil(u64::BITS as usize);

        self.words.resize(word_count, 0);
        self.words.fill(0);
        self.first_non_empty = word_count;
    }

    /// Inserts one rank, panicking on duplicates.
    fn insert(&mut self, rank: usize) {
        let word = rank / u64::BITS as usize;
        let bit = rank % u64::BITS as usize;
        let mask = 1_u64 << bit;

        assert_eq!(
            self.words[word] & mask,
            0,
            "topological rank inserted twice",
        );

        self.words[word] |= mask;
        self.first_non_empty = self.first_non_empty.min(word);
    }

    /// Removes one rank, returning whether it was present.
    fn remove(&mut self, rank: usize) -> bool {
        let word = rank / u64::BITS as usize;
        let bit = rank % u64::BITS as usize;

        if word >= self.words.len() {
            return false;
        }

        let mask = 1_u64 << bit;

        if self.words[word] & mask == 0 {
            return false;
        }

        self.words[word] &= !mask;

        if word == self.first_non_empty {
            self.advance_first();
        }

        true
    }

    /// Returns the smallest rank without removing it.
    fn first(&self) -> Option<usize> {
        if self.is_empty() {
            return None;
        }

        let bit = self.words[self.first_non_empty].trailing_zeros() as usize;

        Some(self.first_non_empty * u64::BITS as usize + bit)
    }

    /// Removes and returns the smallest rank.
    fn pop_first(&mut self) -> Option<usize> {
        self.advance_first();

        if self.first_non_empty >= self.words.len() {
            return None;
        }

        let word_index = self.first_non_empty;
        let bit = self.words[word_index].trailing_zeros() as usize;

        self.words[word_index] &= !(1_u64 << bit);

        if self.words[word_index] == 0 {
            self.advance_first();
        }

        Some(word_index * u64::BITS as usize + bit)
    }

    /// Whether no rank is set.
    fn is_empty(&self) -> bool {
        self.first_non_empty >= self.words.len()
    }

    /// Advances the scan cursor past empty words.
    fn advance_first(&mut self) {
        while self.first_non_empty < self.words.len() && self.words[self.first_non_empty] == 0 {
            self.first_non_empty += 1;
        }
    }
}

/// Mutable state machine for dispatching one compiled schedule.
///
/// The state is reset before each run. It enforces dependency completion,
/// temporarily blocks access-conflicting systems while one of them is running,
/// and separates deferred work into an exclusive phase.
pub(crate) struct ExecutionState {
    /// Uncompleted dependency predecessors per system.
    remaining_predecessors: Vec<usize>,
    /// Number of currently running systems this system conflicts with; it
    /// stays blocked until the count returns to zero.
    active_conflicts: Vec<usize>,
    node_states: Vec<NodeState>,
    condition_passed: Vec<bool>,
    exclusive_systems: Vec<bool>,

    /// Ready systems, keyed by topological rank.
    runnable: RankedSet,
    /// Systems whose deferred work awaits the exclusive phase, by rank.
    pending_deferred: RankedSet,

    running_count: usize,
    completed_count: usize,
}
impl ExecutionState {
    /// Rebuilds state for a fresh execution of `graph`.
    pub(crate) fn reset(&mut self, graph: &ExecutionGraph) {
        let system_count = graph.len();

        self.remaining_predecessors.resize(system_count, 0);
        self.active_conflicts.resize(system_count, 0);
        self.node_states.resize(system_count, NodeState::Waiting);

        self.active_conflicts.fill(0);
        self.node_states.fill(NodeState::Waiting);
        self.condition_passed.clear();
        self.condition_passed.resize(system_count, true);

        for index in 0..system_count {
            let system = SystemId::new(index);

            self.remaining_predecessors[index] = graph.node(system).initial_predecessor_count();
        }

        self.runnable.reset(system_count);
        self.pending_deferred.reset(system_count);

        self.running_count = 0;
        self.completed_count = 0;

        for &system in graph.roots() {
            self.try_make_runnable(graph, system);
        }
    }

    /// Moves a waiting node into the runnable set when all constraints are clear.
    fn try_make_runnable(&mut self, graph: &ExecutionGraph, system: SystemId) {
        let index = system.index();

        if self.node_states[index] != NodeState::Waiting
            || self.remaining_predecessors[index] != 0
            || self.active_conflicts[index] != 0
        {
            return;
        }

        self.node_states[index] = NodeState::Ready;

        self.runnable.insert(graph.node(system).topological_rank());
    }

    /// Creates an empty execution state.
    pub(crate) fn new() -> Self {
        Self {
            remaining_predecessors: Vec::new(),
            active_conflicts: Vec::new(),
            node_states: Vec::new(),
            condition_passed: Vec::new(),
            exclusive_systems: Vec::new(),

            runnable: RankedSet::default(),
            pending_deferred: RankedSet::default(),

            running_count: 0,
            completed_count: 0,
        }
    }

    /// Evaluates every system condition once, with exclusive world access,
    /// and captures per-system exclusivity for the whole run.
    pub(crate) fn evaluate_conditions(&mut self, systems: &[SystemNode], world: &World) {
        self.condition_passed = systems
            .iter()
            .map(|node| {
                node.condition
                    .as_ref()
                    .map(|condition| condition(world))
                    .unwrap_or(true)
            })
            .collect();

        self.exclusive_systems = systems
            .iter()
            .map(|node| node.system.is_exclusive())
            .collect();
    }

    #[inline]
    /// Whether the system's run condition passed this cycle.
    pub(crate) fn condition_passed(&self, system: SystemId) -> bool {
        self.condition_passed[system.index()]
    }

    #[inline]
    /// Whether the system requires exclusive world access.
    pub(crate) fn is_exclusive(&self, system: SystemId) -> bool {
        self.exclusive_systems[system.index()]
    }

    /// Returns the earliest runnable system without consuming it.
    pub(crate) fn peek_runnable(&self, graph: &ExecutionGraph) -> Option<SystemId> {
        let rank = self.runnable.first()?;

        Some(graph.topological_order()[rank])
    }

    /// Takes the earliest runnable system and blocks its conflicts.
    ///
    /// Every conflicting `Waiting` system gains one conflict lock, and a
    /// conflicting `Ready` system is demoted back to `Waiting` and removed
    /// from the runnable set until its lock count drops to zero.
    pub(crate) fn take_runnable(&mut self, graph: &ExecutionGraph) -> Option<SystemId> {
        let rank = self.runnable.pop_first()?;
        let system = graph.topological_order()[rank];
        let index = system.index();

        assert_eq!(self.node_states[index], NodeState::Ready);
        assert_eq!(self.remaining_predecessors[index], 0);
        assert_eq!(self.active_conflicts[index], 0);

        self.node_states[index] = NodeState::Running;
        self.running_count += 1;

        for conflict in graph.node(system).conflicts().iter() {
            let conflict_index = conflict.index();

            assert_ne!(
                self.node_states[conflict_index],
                NodeState::Running,
                "conflicting systems cannot run concurrently",
            );

            match self.node_states[conflict_index] {
                NodeState::Waiting => {
                    self.active_conflicts[conflict_index] += 1;
                }

                NodeState::Ready => {
                    self.active_conflicts[conflict_index] += 1;
                    self.node_states[conflict_index] = NodeState::Waiting;

                    let removed = self
                        .runnable
                        .remove(graph.node(conflict).topological_rank());

                    assert!(removed, "ready system was missing from runnable set");
                }

                NodeState::Running | NodeState::Deferred | NodeState::Completed => {}
            }
        }

        Some(system)
    }

    /// Marks a running system as finished.
    ///
    /// Systems with deferred commands move to the deferred set. Systems without
    /// deferred work complete immediately and may unlock successors.
    pub(crate) fn complete_run(
        &mut self,
        graph: &ExecutionGraph,
        system: SystemId,
        has_deferred_work: bool,
    ) {
        let index = system.index();

        assert_eq!(self.node_states[index], NodeState::Running);
        assert!(self.running_count > 0);

        self.running_count -= 1;
        self.release_conflicts(graph, system);

        if has_deferred_work {
            self.node_states[index] = NodeState::Deferred;

            self.pending_deferred
                .insert(graph.node(system).topological_rank());
        } else {
            self.mark_completed(graph, system);
        }
    }

    /// Takes deferred work in topological order.
    ///
    /// Asserts that no system is still running: deferred work needs exclusive
    /// world access, so it can only be applied between parallel phases.
    pub(crate) fn take_pending_deferred(&mut self, graph: &ExecutionGraph) -> Option<SystemId> {
        assert_eq!(
            self.running_count, 0,
            "deferred work requires an exclusive World phase",
        );

        let rank = self.pending_deferred.pop_first()?;
        let system = graph.topological_order()[rank];

        assert_eq!(self.node_states[system.index()], NodeState::Deferred,);

        Some(system)
    }

    /// Marks deferred work as applied and unlocks successors.
    pub(crate) fn complete_deferred(&mut self, graph: &ExecutionGraph, system: SystemId) {
        assert_eq!(self.node_states[system.index()], NodeState::Deferred,);

        self.mark_completed(graph, system);
    }

    /// Marks a node completed and reduces predecessor counts of successors.
    fn mark_completed(&mut self, graph: &ExecutionGraph, system: SystemId) {
        let index = system.index();

        assert!(matches!(
            self.node_states[index],
            NodeState::Running | NodeState::Deferred,
        ));

        self.node_states[index] = NodeState::Completed;
        self.completed_count += 1;

        assert!(self.completed_count <= graph.len());

        for &successor in graph.node(system).successors() {
            let successor_index = successor.index();
            let remaining = &mut self.remaining_predecessors[successor_index];

            assert!(*remaining > 0, "predecessor count underflow");

            *remaining -= 1;

            if *remaining == 0 {
                self.try_make_runnable(graph, successor);
            }
        }
    }

    /// Releases temporary conflict locks created when `system` started running.
    ///
    /// Each blocked system's lock count is decremented; a system whose count
    /// reaches zero becomes runnable again if its dependencies are done.
    fn release_conflicts(&mut self, graph: &ExecutionGraph, system: SystemId) {
        for conflict in graph.node(system).conflicts().iter() {
            let index = conflict.index();

            assert_ne!(
                self.node_states[index],
                NodeState::Running,
                "conflicting systems cannot run concurrently",
            );

            if self.active_conflicts[index] == 0 {
                continue;
            }

            self.active_conflicts[index] -= 1;

            if self.active_conflicts[index] == 0 {
                self.try_make_runnable(graph, conflict);
            }
        }
    }

    /// Cancels a running system after a failure while keeping the state drainable.
    pub(crate) fn cancel_run(&mut self, graph: &ExecutionGraph, system: SystemId) {
        let index = system.index();

        assert_eq!(self.node_states[index], NodeState::Running);
        assert!(self.running_count > 0);

        self.running_count -= 1;
        self.node_states[index] = NodeState::Waiting;

        self.release_conflicts(graph, system);
    }

    /// `true` while at least one system is running.
    pub(crate) fn has_running_systems(&self) -> bool {
        self.running_count != 0
    }

    /// Whether deferred work is waiting for the exclusive phase.
    pub(crate) fn has_pending_deferred(&self) -> bool {
        !self.pending_deferred.is_empty()
    }

    /// Whether all systems and deferred work are complete.
    pub(crate) fn is_finished(&self, graph: &ExecutionGraph) -> bool {
        self.completed_count == graph.len()
            && self.running_count == 0
            && self.runnable.is_empty()
            && self.pending_deferred.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        resource::ResourceId,
        system::{access::SystemAccess, schedule::graph::DependencyGraph},
    };

    use super::{
        super::graph::{ExecutionGraph, SystemId},
        ExecutionState, RankedSet,
    };

    /// Test helper compiling a graph and conflict matrix from accesses.
    fn compile_graph(
        system_count: usize,
        edges: impl IntoIterator<Item = (SystemId, SystemId)>,
        accesses: &[SystemAccess],
    ) -> ExecutionGraph {
        let dependencies = DependencyGraph::from_edges(system_count, edges);
        let order = dependencies.kahn_sort().unwrap();
        let accesses: Vec<_> = accesses.iter().collect();

        ExecutionGraph::compile(dependencies, order, &accesses)
    }

    #[test]
    fn ranked_set_orders_entries_across_word_boundaries() {
        let mut set = RankedSet::default();
        set.reset(130);

        for rank in [129, 65, 64, 63, 0] {
            set.insert(rank);
        }

        assert_eq!(set.pop_first(), Some(0));
        assert_eq!(set.pop_first(), Some(63));
        assert_eq!(set.pop_first(), Some(64));
        assert_eq!(set.pop_first(), Some(65));
        assert_eq!(set.pop_first(), Some(129));
        assert_eq!(set.pop_first(), None);
        assert!(set.is_empty());
    }

    #[test]
    fn ranked_set_removal_and_reset_keep_the_first_word_valid() {
        let mut set = RankedSet::default();
        set.reset(130);
        set.insert(64);
        set.insert(129);

        assert!(set.remove(64));
        assert!(!set.remove(64));
        assert_eq!(set.pop_first(), Some(129));

        set.reset(3);

        assert!(set.is_empty());
        assert_eq!(set.pop_first(), None);
    }

    #[test]
    fn a_system_waits_until_every_active_conflict_finishes() {
        struct FirstResource;
        struct SecondResource;

        let first = SystemAccess::from_resource_write(ResourceId::of::<FirstResource>()).unwrap();
        let second = SystemAccess::from_resource_write(ResourceId::of::<SecondResource>()).unwrap();
        let mut both = first.clone();
        both.extend_from(&second).unwrap();

        let graph = compile_graph(3, [], &[first, second, both]);
        let mut state = ExecutionState::new();
        state.reset(&graph);

        let first = state.take_runnable(&graph).unwrap();
        let second = state.take_runnable(&graph).unwrap();

        assert_eq!(first, SystemId::new(0));
        assert_eq!(second, SystemId::new(1));
        assert_eq!(state.take_runnable(&graph), None);

        state.complete_run(&graph, first, false);
        assert_eq!(state.take_runnable(&graph), None);

        state.complete_run(&graph, second, false);
        assert_eq!(state.take_runnable(&graph), Some(SystemId::new(2)));
    }

    #[test]
    fn deferred_work_completes_before_unlocking_successors() {
        let first = SystemId::new(0);
        let second = SystemId::new(1);
        let graph = compile_graph(
            2,
            [(first, second)],
            &[SystemAccess::default(), SystemAccess::default()],
        );
        let mut state = ExecutionState::new();
        state.reset(&graph);

        assert_eq!(state.take_runnable(&graph), Some(first));

        state.complete_run(&graph, first, true);

        assert_eq!(state.take_runnable(&graph), None);
        assert_eq!(state.take_pending_deferred(&graph), Some(first));

        state.complete_deferred(&graph, first);

        assert_eq!(state.take_runnable(&graph), Some(second));
    }
}
