//! Compiled dependency and conflict graph for schedules.
//!
//! Explicit ordering constraints are represented as dependencies. Access
//! conflicts are stored separately so the executor can run independent systems
//! in parallel while preventing incompatible reads and writes from overlapping.

use std::collections::BTreeSet;

use crate::system::access::SystemAccess;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// Dense identifier assigned to a system inside one compiled schedule.
pub(crate) struct SystemId(usize);

impl SystemId {
    #[inline]
    /// Creates a system id from its dense schedule index.
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    #[inline]
    /// Returns the dense schedule index.
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

#[derive(Clone, Debug)]
/// Compact bit set of system ids backed by `u64` words.
pub(crate) struct SystemSet {
    /// Bit `i` of word `w` represents dense system index `w * 64 + i`.
    words: Box<[u64]>,
}

impl SystemSet {
    /// Creates an empty set sized for `system_count` systems.
    pub(crate) fn new(system_count: usize) -> Self {
        let word_count = system_count.div_ceil(u64::BITS as usize);

        Self {
            words: vec![0; word_count].into_boxed_slice(),
        }
    }

    #[inline]
    /// Inserts `system` into the set.
    pub(crate) fn insert(&mut self, system: SystemId) {
        let index = system.index();
        let word = index / u64::BITS as usize;
        let bit = index % u64::BITS as usize;

        self.words[word] |= 1_u64 << bit;
    }

    /// Iterates system ids in ascending dense-index order.
    pub(crate) fn iter(&self) -> SystemSetIter<'_> {
        SystemSetIter {
            words: &self.words,
            word_index: 0,
            remaining: self.words.first().copied().unwrap_or(0),
        }
    }
}

/// Iterator over a `SystemSet`.
pub(crate) struct SystemSetIter<'a> {
    words: &'a [u64],
    word_index: usize,
    /// Bits of the current word that have not been yielded yet.
    remaining: u64,
}

impl Iterator for SystemSetIter<'_> {
    type Item = SystemId;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.remaining != 0 {
                let bit = self.remaining.trailing_zeros() as usize;
                self.remaining &= self.remaining - 1;
                return Some(SystemId::new(self.word_index * u64::BITS as usize + bit));
            }

            self.word_index += 1;

            if self.word_index >= self.words.len() {
                return None;
            }

            self.remaining = self.words[self.word_index]
        }
    }
}

#[derive(Debug)]
/// Error returned when explicit schedule dependencies contain a cycle.
pub(crate) struct DependencyCycle;

/// Directed dependency graph built from explicit schedule ordering edges.
pub(crate) struct DependencyGraph {
    /// Adjacency lists: for each system, the systems that run after it.
    /// `BTreeSet` collapses duplicate edges and iterates in a fixed order.
    successors: Vec<BTreeSet<SystemId>>,
    /// Number of unresolved predecessors per system.
    predecessor_counts: Vec<usize>,
}

impl DependencyGraph {
    /// Builds a dependency graph from `(predecessor, successor)` edges.
    ///
    /// Duplicate edges are ignored, so predecessor counts stay exact.
    ///
    /// # Panics
    ///
    /// Panics if an edge references a system id outside `system_count`.
    pub(crate) fn from_edges<I>(system_count: usize, edges: I) -> Self
    where
        I: IntoIterator<Item = (SystemId, SystemId)>,
    {
        let mut successors = vec![BTreeSet::new(); system_count];

        let mut predecessor_counts = vec![0usize; system_count];

        for (predecessor, successor) in edges {
            assert!(predecessor.index() < system_count);
            assert!(successor.index() < system_count);

            if successors[predecessor.index()].insert(successor) {
                predecessor_counts[successor.index()] += 1;
            }
        }

        Self {
            successors,
            predecessor_counts,
        }
    }

    /// Returns a deterministic topological order or reports a cycle.
    ///
    /// Among the currently ready systems, `BTreeSet::pop_first` always emits
    /// the smallest dense index first, so the order is reproducible.
    pub(crate) fn kahn_sort(&self) -> Result<Box<[SystemId]>, DependencyCycle> {
        let mut remaining = self.predecessor_counts.clone();

        let mut ready = BTreeSet::new();

        for (index, predecessor_count) in remaining.iter().copied().enumerate() {
            if predecessor_count == 0 {
                ready.insert(SystemId::new(index));
            }
        }

        let mut order = Vec::with_capacity(self.successors.len());

        while let Some(system) = ready.pop_first() {
            order.push(system);

            for &successor in &self.successors[system.index()] {
                let count = &mut remaining[successor.index()];

                *count -= 1;

                if *count == 0 {
                    ready.insert(successor);
                }
            }
        }

        if order.len() != self.successors.len() {
            return Err(DependencyCycle);
        }

        Ok(order.into_boxed_slice())
    }
}

/// Compiled data for one system node.
pub(crate) struct ExecutionNode {
    /// Systems that become eligible when this node completes.
    successors: Box<[SystemId]>,
    /// Dependency predecessors the node starts with each run.
    initial_predecessor_count: usize,
    /// Systems that cannot overlap with this node.
    conflicts: SystemSet,
    /// Position in the deterministic topological order.
    topological_rank: usize,
}

impl ExecutionNode {
    #[inline]
    /// Returns systems that depend on this node.
    pub(crate) fn successors(&self) -> &[SystemId] {
        &self.successors
    }

    #[inline]
    /// Returns how many dependency predecessors this node starts with.
    pub(crate) fn initial_predecessor_count(&self) -> usize {
        self.initial_predecessor_count
    }

    #[inline]
    /// Returns systems that cannot overlap with this node because of access conflicts.
    pub(crate) fn conflicts(&self) -> &SystemSet {
        &self.conflicts
    }

    #[inline]
    /// Returns this node position in the deterministic topological order,
    /// used by the executor to break dispatch-order ties.
    pub(crate) fn topological_rank(&self) -> usize {
        self.topological_rank
    }
}

/// Graph used directly by schedule executors.
///
/// It combines explicit dependency edges with conflict sets derived from system
/// access metadata. The executor treats dependencies as completion
/// requirements and conflicts as temporary runtime locks.
pub(crate) struct ExecutionGraph {
    /// Per-system compiled data.
    nodes: Box<[ExecutionNode]>,
    /// Systems with no explicit predecessors.
    roots: Box<[SystemId]>,
    /// Deterministic topological order over all systems.
    topological_order: Box<[SystemId]>,
}

impl ExecutionGraph {
    /// Compiles dependency and access information into executor-ready nodes.
    ///
    /// Access conflicts are analyzed pairwise and inserted in both directions:
    /// unlike directed dependency edges, conflicts are symmetric exclusion
    /// constraints. Each node also records its rank in the topological order
    /// so the executor can always pick the earliest eligible system.
    pub(crate) fn compile(
        dependencies: DependencyGraph,
        topological_order: Box<[SystemId]>,
        accesses: &[&SystemAccess],
    ) -> Self {
        let system_count = accesses.len();

        assert_eq!(dependencies.successors.len(), system_count);

        assert_eq!(topological_order.len(), system_count);

        let mut conflicts = (0..system_count)
            .map(|_| SystemSet::new(system_count))
            .collect::<Vec<_>>();

        // Conflict analysis only runs when recompiling the schedule.
        for left_index in 0..system_count {
            for right_index in (left_index + 1)..system_count {
                if accesses[left_index].conflicts_with(accesses[right_index]) {
                    let left = SystemId::new(left_index);
                    let right = SystemId::new(right_index);

                    // Conflicts are symmetric execution constraints, not dependencies.
                    conflicts[left_index].insert(right);
                    conflicts[right_index].insert(left);
                }
            }
        }

        let mut ranks = vec![usize::MAX; system_count];

        for (rank, system) in topological_order.iter().copied().enumerate() {
            ranks[system.index()] = rank;
        }

        let roots = dependencies
            .predecessor_counts
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, count)| (count == 0).then_some(SystemId::new(index)))
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let nodes = dependencies
            .successors
            .into_iter()
            .zip(dependencies.predecessor_counts)
            .zip(conflicts)
            .enumerate()
            .map(
                |(index, ((successors, predecessor_count), conflicts))| ExecutionNode {
                    successors: successors
                        .into_iter()
                        .collect::<Vec<_>>()
                        .into_boxed_slice(),

                    initial_predecessor_count: predecessor_count,

                    conflicts,

                    topological_rank: ranks[index],
                },
            )
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            nodes,
            roots,
            topological_order,
        }
    }

    #[inline]
    /// Returns compiled data for `system`.
    pub(crate) fn node(&self, system: SystemId) -> &ExecutionNode {
        &self.nodes[system.index()]
    }

    #[inline]
    /// Returns systems with no explicit predecessors.
    pub(crate) fn roots(&self) -> &[SystemId] {
        &self.roots
    }

    #[inline]
    /// Returns the deterministic topological order.
    pub(crate) fn topological_order(&self) -> &[SystemId] {
        &self.topological_order
    }

    #[inline]
    /// Returns the number of systems in the graph.
    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
}
