//! Version tracking for world topology and archetype row counts.
//!
//! Query states compare these versions to decide when their cached route and
//! work splits need refreshing. Changes are accumulated in
//! [`PlanningChanges`] while world operations run and published through
//! [`WorldPlanning::apply`] once per cycle.

use crate::{
    component::ArchetypeId,
    planning::{changes::Changes, version::Version},
};

mod changes;
mod version;

/// Marker for the world topology (archetype set) version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Topology;
/// Marker for the archetype row-count version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArchetypeRows;
/// Marker for the query work structure version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct QueryWorkStructure;

/// Tracked planning state of one archetype.
#[derive(Clone, Debug)]
struct ArchetypeNode {
    /// Bumped whenever the live row count changes.
    rows_version: Version<ArchetypeRows>,
    /// Current live row count.
    rows: usize,
}

impl ArchetypeNode {
    /// Creates the node for a newly registered archetype.
    fn new(rows: usize) -> Self {
        Self {
            rows_version: Version::new(),
            rows,
        }
    }

    /// Sets the row count, bumping the version and reporting a change.
    fn set_rows(&mut self, rows: usize) -> bool {
        if self.rows == rows {
            return false;
        }

        self.rows = rows;
        self.rows_version.bump();
        true
    }
}

/// Cost-model version: the archetype row counts route estimation depends on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PlanningCostVersion {
    archetype_rows: u64,
}

impl PlanningCostVersion {
    /// Returns the tracked archetype-rows version.
    #[inline]
    pub(crate) const fn archetype_rows(self) -> u64 {
        self.archetype_rows
    }
}

/// Point-in-time planning state of one archetype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ArchetypePlanningSnapshot {
    /// Identifier of the archetype.
    pub(crate) id: ArchetypeId,
    /// Row-count version when the snapshot was taken.
    pub(crate) rows_version: u64,
    /// Live row count when the snapshot was taken.
    pub(crate) rows: usize,
}

/// Point-in-time state of the world topology.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TopologyPlanningSnapshot {
    /// Current topology version.
    pub(crate) version: u64,
    /// Number of registered archetypes.
    pub(crate) archetype_count: usize,
}

/// World-side planner: version counters plus per-archetype row tracking.
pub struct WorldPlanning {
    topology_version: Version<Topology>,
    archetype_rows_version: Version<ArchetypeRows>,
    query_work_version: Version<QueryWorkStructure>,
    /// Per-archetype nodes indexed densely by `ArchetypeId::index()`.
    archetypes: Vec<ArchetypeNode>,
}

impl WorldPlanning {
    /// Creates an empty planner.
    pub fn new() -> Self {
        Self {
            topology_version: Version::new(),
            archetype_rows_version: Version::new(),
            query_work_version: Version::new(),
            archetypes: Vec::new(),
        }
    }

    /// Returns the current query-work structure version.
    #[inline]
    pub(crate) fn query_work_version(&self) -> u64 {
        self.query_work_version.current()
    }

    /// Returns the cost-model version for route re-estimation.
    #[inline]
    pub(crate) fn route_version(&self) -> PlanningCostVersion {
        PlanningCostVersion {
            archetype_rows: self.archetype_rows_version.current(),
        }
    }

    /// Registers a new archetype or updates the rows of a known one.
    fn add_archetype(&mut self, change: ArchetypeRowsChange) {
        let id = change.id;
        let index = id.index();

        if index < self.archetypes.len() {
            self.archetypes[index].rows = change.rows;
            return;
        }

        assert_eq!(
            index,
            self.archetypes.len(),
            "archetype ids must be registered densely in WorldPlanning"
        );

        self.archetypes.push(ArchetypeNode::new(change.rows));
    }

    /// Returns the mutable node for `id`, if registered.
    fn get_mut_archetype_node(&mut self, id: ArchetypeId) -> Option<&mut ArchetypeNode> {
        self.archetypes.get_mut(id.index())
    }

    /// Returns the node for `id`, if registered.
    fn get_archetype_node(&self, id: ArchetypeId) -> Option<&ArchetypeNode> {
        self.archetypes.get(id.index())
    }

    /// Snapshots the planning state of one archetype.
    #[inline]
    pub(crate) fn archetype_snapshot(&self, id: ArchetypeId) -> Option<ArchetypePlanningSnapshot> {
        let node = self.get_archetype_node(id)?;

        Some(ArchetypePlanningSnapshot {
            id,
            rows_version: node.rows_version.current(),
            rows: node.rows,
        })
    }

    /// Snapshots the world topology state.
    #[inline]
    pub(crate) fn topology_snapshot(&self) -> TopologyPlanningSnapshot {
        TopologyPlanningSnapshot {
            version: self.topology_version.current(),
            archetype_count: self.archetypes.len(),
        }
    }

    /// Applies one archetype row change directly: the rows version bumps only
    /// when the count changed, while the query-work version always advances.
    #[inline]
    fn apply_archetype_rows(&mut self, id: ArchetypeId, rows: usize) {
        let node = &mut self.archetypes[id.index()];

        if node.set_rows(rows) {
            self.archetype_rows_version.bump();
        }

        self.query_work_version.bump();
    }

    /// Publishes accumulated changes into the version counters.
    ///
    /// The single-row-change fast path avoids the general topology pass when
    /// only one archetype's row count moved.
    pub(crate) fn apply(&mut self, changes: PlanningChanges) {
        let PlanningChanges {
            topology,
            changed_archetypes,
            query_work_changed,
        } = changes;

        if query_work_changed && topology.is_none() {
            if let Changes::One(change) = &changed_archetypes {
                self.apply_archetype_rows(change.id, change.rows);
                return;
            }
        }

        if let Some(topology) = topology {
            let TopologyChanges { created_archetypes } = *topology;
            let topology_changed = !created_archetypes.is_empty();

            created_archetypes.for_each(|change| {
                self.add_archetype(change);
            });

            if topology_changed {
                self.topology_version.bump();
            }
        }

        let mut archetype_rows_changed = false;

        changed_archetypes.for_each(|change| {
            let node = self
                .get_mut_archetype_node(change.id)
                .expect("changed archetype must exist in WorldPlanning");

            archetype_rows_changed |= node.set_rows(change.rows);
        });

        if archetype_rows_changed {
            self.archetype_rows_version.bump();
        }

        if query_work_changed {
            self.query_work_version.bump();
        }
    }
}

impl Default for WorldPlanning {
    fn default() -> Self {
        Self::new()
    }
}

/// One archetype row-count change awaiting publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ArchetypeRowsChange {
    /// Identifier of the changed archetype.
    pub(crate) id: ArchetypeId,
    /// Live row count after the change.
    pub(crate) rows: usize,
}

/// Archetype registrations produced by one world operation batch.
#[derive(Default, Debug)]
struct TopologyChanges {
    created_archetypes: Changes<ArchetypeRowsChange>,
}

/// Changes accumulated by world operations and published once per cycle.
#[derive(Default, Debug)]
pub(crate) struct PlanningChanges {
    /// Set when archetypes were created (allocates lazily).
    topology: Option<Box<TopologyChanges>>,

    /// Row-count changes for already-known archetypes.
    changed_archetypes: Changes<ArchetypeRowsChange>,

    /// Set when any structural activity invalidates query work.
    query_work_changed: bool,
}

impl PlanningChanges {
    /// Returns the topology accumulator, allocating it on first use.
    #[inline]
    fn topology_mut(&mut self) -> &mut TopologyChanges {
        self.topology
            .get_or_insert_with(|| Box::new(TopologyChanges::default()))
    }

    /// Flags cached query work as needing a rebuild.
    #[inline]
    pub(crate) fn mark_query_work_changed(&mut self) {
        self.query_work_changed = true;
    }

    /// Records the creation of a new archetype.
    #[inline]
    pub(crate) fn mark_archetype_created(&mut self, id: ArchetypeId, rows: usize) {
        self.mark_query_work_changed();
        self.topology_mut()
            .created_archetypes
            .upsert(ArchetypeRowsChange { id, rows });
    }

    /// Flags a sparse-set registration as structural activity.
    #[inline]
    pub(crate) fn mark_sparse_set_registered(&mut self) {
        self.mark_query_work_changed();
    }

    /// Records a row-count change, merging into the creation entry when the
    /// archetype was created in the same cycle.
    #[inline]
    pub(crate) fn mark_archetype_rows(&mut self, id: ArchetypeId, rows: usize) {
        self.mark_query_work_changed();

        let change = ArchetypeRowsChange { id, rows };

        if let Some(topology) = self.topology.as_deref_mut() {
            if topology.created_archetypes.replace_existing(change) {
                return;
            }
        }

        self.changed_archetypes.upsert(change);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_work_version_observes_net_zero_structural_changes() {
        let archetype = ArchetypeId::new(0);
        let mut planning = WorldPlanning::new();

        let mut initial = PlanningChanges::default();
        initial.mark_archetype_created(archetype, 1);
        planning.apply(initial);

        let previous_work_version = planning.query_work_version();
        let previous_cost_version = planning.route_version().archetype_rows();

        let mut changes = PlanningChanges::default();
        changes.mark_archetype_rows(archetype, 0);
        changes.mark_archetype_rows(archetype, 1);
        planning.apply(changes);

        assert_eq!(
            planning.route_version().archetype_rows(),
            previous_cost_version,
            "net-zero changes must not invalidate route cost"
        );
        assert_ne!(
            planning.query_work_version(),
            previous_work_version,
            "query work must observe structural activity even when row count is unchanged"
        );
    }
}
