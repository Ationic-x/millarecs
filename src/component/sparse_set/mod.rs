//! Sparse-set component storage.
//!
//! Sparse components are stored outside archetype tables and are addressed by
//! entity id. This is useful for components with frequent insertion/removal or
//! for query filters where a sparse driver can reduce the candidate set.

use crate::{
    change_detection::ChangeTick,
    component::{
        archetype::plan::batch_transition::BatchTransitionInput, bundle::BatchBundleItem,
        ComponentId,
    },
    entity::{metadata::EntityLocation, Entity},
};

mod dense;
mod driver;
mod manager;
mod set;

pub(crate) use driver::{
    SparseDriverBucket, SparseDriverIndex, SparseDriverLane, SparseDriverRow, SparseLaneLocality,
    SparseMembershipLinks,
};
pub(crate) use manager::SparseSetManager;
pub(crate) use set::ComponentSparseSet;

/// Membership probe captured before a sparse batch removal.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseBatchProbe {
    /// Dense position of the entity in the set.
    pub(crate) dense_pos: u32,
    /// Membership links captured before the removal.
    pub(crate) links: SparseMembershipLinks,
}

/// One prevalidated removal in a component-major sparse batch.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseBatchRemoval {
    /// Entity being removed from the set.
    pub(crate) entity: Entity,
    /// Dense position the entity occupies.
    pub(crate) dense_pos: u32,
}

/// One prevalidated sparse location repair.
///
/// The dense position is captured while traversing the entity membership
/// chain, allowing the execution phase to update each sparse set without
/// resolving the entity again.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseLocationUpdate {
    /// Table location that will replace the stored one.
    pub(crate) location: EntityLocation,
    /// Dense position of the entity inside its sparse set.
    pub(crate) dense_pos: u32,
}

/// Component-major location repairs prepared by structural batch operations.
#[derive(Default)]
pub(crate) struct SparseLocationUpdateGroups {
    /// Prepared repairs per component, indexed by `ComponentId::index`.
    groups: Vec<Vec<SparseLocationUpdate>>,
    /// Components with at least one queued repair.
    touched: Vec<ComponentId>,
    /// Total queued repairs across every component, backing `is_empty`.
    update_count: usize,
}

impl SparseLocationUpdateGroups {
    /// Appends one repair for `component`, tracking it as touched.
    #[inline]
    pub(crate) fn push(&mut self, component: ComponentId, update: SparseLocationUpdate) {
        let index = component.index();

        if index >= self.groups.len() {
            self.groups.resize_with(index + 1, Vec::new);
        }

        if self.groups[index].is_empty() {
            self.touched.push(component);
        }

        self.groups[index].push(update);
        self.update_count += 1;
    }

    /// Returns the prepared repairs for one component.
    #[inline]
    pub(crate) fn updates(&self, component: ComponentId) -> &[SparseLocationUpdate] {
        &self.groups[component.index()]
    }

    /// Returns whether no repair is accumulated.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.update_count == 0
    }

    /// Drains the touched component list, leaving the groups in place.
    #[inline]
    pub(crate) fn take_touched(&mut self) -> Vec<ComponentId> {
        std::mem::take(&mut self.touched)
    }

    /// Clears logical contents while retaining per-component allocations.
    pub(crate) fn finish(&mut self, mut touched: Vec<ComponentId>) {
        for &component in &touched {
            self.groups[component.index()].clear();
        }

        touched.clear();
        self.touched = touched;
        self.update_count = 0;
    }
}

/// One prevalidated value in a component-major sparse spawn batch.
pub(crate) struct SparseSpawnBatchEntry {
    /// Entity receiving the value.
    pub(crate) entity: Entity,
    /// Final table location of the entity.
    pub(crate) location: EntityLocation,
    /// Erased bundle value to move into the set.
    pub(crate) item: BatchBundleItem,
    /// Prelinked membership chain position.
    pub(crate) links: SparseMembershipLinks,
}

/// One prevalidated value in a component-major sparse insertion batch.
///
/// Unlike spawn insertion, an existing component is replaced in place. A new
/// dense position is published here so membership linking can happen after the
/// typed sparse-set borrow has been released.
pub(crate) struct SparseInsertBatchEntry {
    /// Entity receiving the value.
    pub(crate) entity: Entity,
    /// Final table location of the entity.
    pub(crate) location: EntityLocation,
    /// Erased bundle value to move into the set.
    pub(crate) item: BatchBundleItem,
    /// Dense position assigned when the value was newly inserted.
    pub(crate) inserted_dense_pos: Option<u32>,
}

/// Component-grouped removals prepared before structural mutation starts.
#[derive(Default)]
pub(crate) struct SparseRemovalGroups {
    /// Removal plans per component, indexed by `ComponentId::index`.
    groups: Vec<SparseRemovalGroup>,
    /// Components with at least one planned removal.
    touched: Vec<ComponentId>,
    /// Total planned removals across every component, backing `is_empty`.
    removal_count: usize,
}

/// Per-component removal plan for one destruction batch.
///
/// Memberships are first counted without per-entity metadata. Once counting
/// finishes, each group is classified: a full group covers its complete sparse
/// set, which can then be cleared wholesale, while a partial group records
/// per-entity removals and compacts the set individually.
#[derive(Default)]
struct SparseRemovalGroup {
    /// Per-entity removals, recorded only when the batch cannot clear the
    /// complete set.
    removals: Vec<SparseBatchRemoval>,
    /// Batch members that own this component.
    target_count: usize,
    /// Sparse-set length captured when counting began.
    set_len: usize,
    /// Set while the group only counts candidates, before classification.
    deferred: bool,
    /// Set once classification proved the batch covers the complete set.
    full_set: bool,
}

impl SparseRemovalGroups {
    /// Returns the group of `component`, growing the table for a new component
    /// index.
    #[inline]
    fn group_mut_or_insert(&mut self, component: ComponentId) -> &mut SparseRemovalGroup {
        let index = component.index();

        if index >= self.groups.len() {
            self.groups
                .resize_with(index + 1, SparseRemovalGroup::default);
        }

        &mut self.groups[index]
    }

    /// Marks `component` as touched unless its group already holds a plan.
    #[inline]
    fn touch_if_empty(&mut self, component: ComponentId) {
        if self.groups[component.index()].target_count == 0 {
            self.touched.push(component);
        }
    }

    /// Records a membership whose set is too large to be completely covered
    /// by the destruction batch. Its dense position is retained immediately.
    #[inline]
    pub(crate) fn push_known_partial(
        &mut self,
        component: ComponentId,
        removal: SparseBatchRemoval,
    ) {
        self.group_mut_or_insert(component);
        self.touch_if_empty(component);

        let group = &mut self.groups[component.index()];
        debug_assert!(!group.deferred);
        group.target_count += 1;
        group.removals.push(removal);
        self.removal_count += 1;
    }

    /// Counts one membership without storing per-entity removal metadata.
    ///
    /// This is used only when the batch is large enough to cover the complete
    /// sparse set. A second pass is required only if coverage is incomplete.
    #[inline]
    pub(crate) fn count_full_candidate(&mut self, component: ComponentId, set_len: usize) {
        self.group_mut_or_insert(component);
        self.touch_if_empty(component);

        let group = &mut self.groups[component.index()];
        if group.target_count == 0 {
            group.deferred = true;
            group.set_len = set_len;
        } else {
            debug_assert!(group.deferred);
            debug_assert_eq!(group.set_len, set_len);
        }

        group.target_count += 1;
        self.removal_count += 1;
    }

    /// Classifies deferred groups after every target membership has been
    /// counted. Complete sets require no per-entity removal vector.
    pub(crate) fn classify_full_candidates(&mut self) {
        for &component in &self.touched {
            let group = &mut self.groups[component.index()];

            if group.deferred {
                debug_assert!(group.target_count <= group.set_len);
                group.full_set = group.target_count == group.set_len;
            }
        }
    }

    /// Returns whether any deferred group turned out to be partial.
    #[inline]
    pub(crate) fn has_deferred_partial(&self) -> bool {
        self.touched.iter().any(|component| {
            let group = &self.groups[component.index()];
            group.deferred && !group.full_set
        })
    }

    /// Returns whether `component` is a deferred group needing a second,
    /// per-entity traversal.
    #[inline]
    pub(crate) fn needs_deferred_removal(&self, component: ComponentId) -> bool {
        let group = &self.groups[component.index()];
        group.deferred && !group.full_set
    }

    /// Records one removal for a deferred group classified as partial.
    #[inline]
    pub(crate) fn push_deferred_removal(
        &mut self,
        component: ComponentId,
        removal: SparseBatchRemoval,
    ) {
        let group = &mut self.groups[component.index()];
        debug_assert!(group.deferred && !group.full_set);
        debug_assert!(group.removals.len() < group.target_count);
        group.removals.push(removal);
    }

    /// Returns whether the batch removes the complete set of `component`.
    #[inline]
    pub(crate) fn is_full_set(&self, component: ComponentId) -> bool {
        self.groups[component.index()].full_set
    }

    /// Returns the recorded per-entity removals of `component`.
    #[inline]
    pub(crate) fn removals_mut(&mut self, component: ComponentId) -> &mut Vec<SparseBatchRemoval> {
        &mut self.groups[component.index()].removals
    }

    /// Returns how many batch members own `component`.
    #[inline]
    pub(crate) fn target_count(&self, component: ComponentId) -> usize {
        self.groups[component.index()].target_count
    }

    /// Returns whether no removal is planned.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.removal_count == 0
    }

    /// Drains the touched component list, leaving the groups in place.
    #[inline]
    pub(crate) fn take_touched(&mut self) -> Vec<ComponentId> {
        std::mem::take(&mut self.touched)
    }

    /// Clears logical contents while retaining per-component allocations.
    #[inline]
    pub(crate) fn finish(&mut self, mut touched: Vec<ComponentId>) {
        for &component in &touched {
            let group = &mut self.groups[component.index()];
            debug_assert!(group.removals.is_empty());
            group.target_count = 0;
            group.set_len = 0;
            group.deferred = false;
            group.full_set = false;
        }

        touched.clear();
        self.touched = touched;
        self.removal_count = 0;
    }
}

/// Result of moving one erased value into a sparse set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SparseInsertOutcome {
    /// A new dense entry was created at `dense_pos`.
    Inserted { dense_pos: u32 },

    /// The value of an existing membership was replaced in place.
    Replaced,
}

/// Metadata captured while removing one sparse membership.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseRemoveOutcome {
    /// Links held by the removed node before dense storage was compacted.
    pub(crate) links: SparseMembershipLinks,
}

/// Type-erased interface shared by all typed sparse sets.
pub(crate) trait SparseSet: 'static + Send + Sync {
    /// Returns the number of stored components.
    fn len(&self) -> usize;

    /// Inserts or replaces a component by moving it from a raw pointer.
    ///
    /// # Safety
    /// `value` must point to a valid, aligned `T` for the concrete sparse set.
    unsafe fn try_insert_move(
        &mut self,
        entity: Entity,
        location: EntityLocation,
        value: *mut u8,
        change_tick: ChangeTick,
    ) -> Option<SparseInsertOutcome>;

    /// Validates entity vacancies and reserves storage for a spawn batch.
    fn prepare_spawn_batch(&mut self, entities: &[Entity]) -> bool;

    /// Reserves storage for a batch that may insert or replace values.
    fn prepare_insert_batch(&mut self, inputs: &[BatchTransitionInput]);

    /// Moves one prepared component-major spawn batch into this set.
    ///
    /// # Safety
    ///
    /// `prepare_spawn_batch` must have succeeded for the same entities. Every
    /// item pointer must contain this set's concrete component type and each
    /// value may be moved exactly once.
    unsafe fn insert_spawn_batch_prepared(
        &mut self,
        entries: &[SparseSpawnBatchEntry],
        change_tick: ChangeTick,
    );

    /// Moves a prepared insertion batch, recording only newly-created dense
    /// positions for later entity-membership linking.
    ///
    /// # Safety
    ///
    /// `prepare_insert_batch` must have run for the same entities. Every item
    /// pointer must contain this set's concrete component type.
    unsafe fn insert_batch_prepared(
        &mut self,
        entries: &mut [SparseInsertBatchEntry],
        change_tick: ChangeTick,
    );

    /// Removes the component and returns the metadata needed to repair or
    /// continue traversing the entity-local membership chain.
    ///
    /// Implementations must resolve the sparse-to-dense position only once.
    fn remove_with_links(&mut self, entity: Entity) -> Option<SparseRemoveOutcome>;

    /// Returns the sparse-driver grouping, building it on first use.
    fn driver_buckets(&self) -> &[SparseDriverBucket];

    /// Returns the structural revision, building the driver on first use.
    fn driver_revision(&self) -> u64;

    /// Updates the indexed archetype row and returns the next sparse membership
    /// component id for entity-local traversal.
    fn update_driver_location_and_next(
        &mut self,
        entity: Entity,
        location: EntityLocation,
    ) -> Option<u32>;

    /// Applies component-major location repairs captured before mutation.
    fn update_locations_batch(&mut self, updates: &[SparseLocationUpdate]);

    /// Exposes the set for immutable downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Exposes the set for mutable downcasting.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Returns the membership links stored for `entity`, if it belongs to the
    /// set.
    fn membership_links(&self, entity: Entity) -> Option<SparseMembershipLinks>;

    /// Overwrites both membership links of the entry at `dense_pos`; returns
    /// whether the position is initialized.
    fn set_membership_links_dense(&mut self, dense_pos: u32, prev: u32, next: u32) -> bool;

    /// Points the `prev` link of `entity` at a component index; returns whether
    /// the entity belongs to the set.
    fn set_membership_prev(&mut self, entity: Entity, prev: u32) -> bool;

    /// Points the `next` link of `entity` at a component index; returns whether
    /// the entity belongs to the set.
    fn set_membership_next(&mut self, entity: Entity, next: u32) -> bool;

    /// Resolves everything required to remove one entity without mutation.
    fn prepare_batch_remove(&self, entity: Entity) -> Option<SparseBatchProbe>;

    /// Removes entries whose dense positions were captured before mutation.
    ///
    /// Entries belong to this sparse set and contain no duplicates.
    fn remove_batch_prepared(&mut self, removals: &mut [SparseBatchRemoval]);

    /// Clears the complete set without requiring per-entity removal records.
    fn clear_all_entries(&mut self);
}
