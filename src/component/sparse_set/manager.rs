//! Sparse-set registry indexed by component id.
//!
//! Each registered sparse component owns one typed dense/sparse pair hidden
//! behind a dynamic trait object. Sparse membership is additionally linked per
//! entity so structural operations only visit the sparse sets an entity really
//! owns instead of scanning every registered sparse component type.

use crate::{
    change_detection::ChangeTick,
    component::{
        archetype::plan::batch_transition::BatchTransitionInput,
        sparse_set::{
            set::{ComponentSparseSet, INVALID_SPARSE_COMPONENT},
            SparseBatchRemoval, SparseDriverBucket, SparseInsertBatchEntry, SparseInsertOutcome,
            SparseLocationUpdate, SparseLocationUpdateGroups, SparseRemovalGroups,
            SparseRemoveOutcome, SparseSpawnBatchEntry,
        },
        Component, ComponentId, SparseSet,
    },
    entity::{metadata::EntityLocation, Entity},
};

/// Result of an insertion routed through [`SparseSetManager`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseInsertResult {
    /// `true` when this operation created a new sparse membership rather than
    /// replacing the value of an existing component.
    pub(crate) inserted: bool,
}

/// One membership created by a deferred bundle, awaiting chain linking.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingSparseMembership {
    /// Component whose sparse set received the value.
    component: ComponentId,
    /// Dense position the value received inside that set.
    dense_pos: u32,
}

/// Collection of all sparse sets indexed by component id.
pub struct SparseSetManager {
    /// Registered sparse sets indexed by component index; `None` marks
    /// components without sparse storage.
    sparse_sets: Vec<Option<Box<dyn SparseSet>>>,

    /// Head sparse component for each entity index.
    ///
    /// Every concrete sparse membership stores its `prev`/`next` component ids
    /// in canonical membership metadata independent from the optional query
    /// driver. This makes entity-local sparse traversal O(K), where K is the
    /// number of sparse components actually owned by the entity.
    entity_heads: Vec<u32>,

    /// Reusable storage for memberships created by one prepared bundle.
    /// Linking them together after all insertions avoids repeatedly updating
    /// the entity head and the previous head node.
    pending_memberships: Vec<PendingSparseMembership>,
}

impl Default for SparseSetManager {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseSetManager {
    /// Creates an empty manager.
    pub fn new() -> Self {
        Self {
            sparse_sets: Vec::new(),
            entity_heads: Vec::new(),
            pending_memberships: Vec::new(),
        }
    }

    /// Registers sparse storage for component `T` at `index`.
    pub fn register<C: Component>(&mut self, index: usize) {
        if index >= self.sparse_sets.len() {
            self.sparse_sets.resize_with(index + 1, || None);
        }

        if self.sparse_sets[index].is_none() {
            self.sparse_sets[index] = Some(Box::new(ComponentSparseSet::<C>::new()));
        }
    }

    /// Returns a typed immutable sparse set if the index and type match.
    pub fn get_sparse_set<C: Component>(&self, index: usize) -> Option<&ComponentSparseSet<C>> {
        let boxed = self.sparse_sets.get(index)?.as_ref()?;
        boxed.as_any().downcast_ref::<ComponentSparseSet<C>>()
    }

    /// Returns a typed mutable sparse set if the index and type match.
    pub fn get_sparse_set_mut<C: Component>(
        &mut self,
        index: usize,
    ) -> Option<&mut ComponentSparseSet<C>> {
        let boxed = self.sparse_sets.get_mut(index)?.as_mut()?;
        boxed.as_any_mut().downcast_mut::<ComponentSparseSet<C>>()
    }

    /// Returns whether a sparse set exists at `index`.
    #[inline]
    pub fn is_registered(&self, index: usize) -> bool {
        self.sparse_sets
            .get(index)
            .and_then(|x| x.as_ref())
            .is_some()
    }

    /// Returns a sparse set through dynamic dispatch.
    pub(crate) fn get_sparse_set_dyn(&self, index: usize) -> Option<&dyn SparseSet> {
        let boxed = self.sparse_sets.get(index)?.as_ref()?;
        Some(&**boxed)
    }

    /// Returns a mutable sparse set through dynamic dispatch.
    pub(crate) fn get_sparse_set_dyn_mut(&mut self, index: usize) -> Option<&mut dyn SparseSet> {
        let boxed = self.sparse_sets.get_mut(index)?.as_mut()?;
        Some(&mut **boxed)
    }

    /// Starts one deferred sparse-membership insertion group.
    #[inline]
    pub(crate) fn begin_deferred_insertions(&mut self, additional: usize) {
        self.pending_memberships.clear();
        self.pending_memberships.reserve(additional);
    }

    /// Moves one value into its sparse set without changing the entity-local
    /// membership chain.
    ///
    /// # Safety
    ///
    /// `value` must satisfy the concrete sparse set's `try_insert_move` contract
    /// for `component`.
    #[inline]
    unsafe fn insert_move_unlinked(
        &mut self,
        component: ComponentId,
        entity: Entity,
        location: EntityLocation,
        value: *mut u8,
        change_tick: ChangeTick,
    ) -> Option<SparseInsertOutcome> {
        let set = self.get_sparse_set_dyn_mut(component.index())?;

        unsafe { set.try_insert_move(entity, location, value, change_tick) }
    }

    /// Validates and reserves one sparse component set before batch values move.
    pub(crate) fn prepare_spawn_component_batch(
        &mut self,
        component: ComponentId,
        entities: &[Entity],
    ) -> bool {
        self.get_sparse_set_dyn_mut(component.index())
            .is_some_and(|set| set.prepare_spawn_batch(entities))
    }

    /// Validates a sparse component and reserves one mixed insertion batch.
    pub(crate) fn prepare_insert_component_batch(
        &mut self,
        component: ComponentId,
        inputs: &[BatchTransitionInput],
    ) -> bool {
        let Some(set) = self.get_sparse_set_dyn_mut(component.index()) else {
            return false;
        };

        set.prepare_insert_batch(inputs);
        true
    }

    /// Executes one already prepared component-major spawn batch.
    ///
    /// # Safety
    ///
    /// The target set must have been prepared for every entry and each erased
    /// value must match `component`.
    pub(crate) unsafe fn insert_spawn_component_batch(
        &mut self,
        component: ComponentId,
        entries: &[SparseSpawnBatchEntry],
        change_tick: ChangeTick,
    ) {
        let set = self
            .get_sparse_set_dyn_mut(component.index())
            .expect("prepared sparse spawn set must exist");

        unsafe {
            set.insert_spawn_batch_prepared(entries, change_tick);
        }
    }

    /// Executes one prepared component-major insertion batch and links only
    /// the memberships created by this operation.
    ///
    /// # Safety
    ///
    /// The target set and every erased item must satisfy
    /// `SparseSet::insert_batch_prepared`.
    pub(crate) unsafe fn insert_component_batch(
        &mut self,
        component: ComponentId,
        entries: &mut [SparseInsertBatchEntry],
        change_tick: ChangeTick,
    ) -> usize {
        {
            let set = self
                .get_sparse_set_dyn_mut(component.index())
                .expect("prepared sparse insertion set must exist");

            unsafe {
                set.insert_batch_prepared(entries, change_tick);
            }
        }

        let mut inserted = 0;

        for entry in entries {
            let Some(dense_pos) = entry.inserted_dense_pos else {
                continue;
            };

            self.link_membership(entry.entity, component, dense_pos);
            inserted += 1;
        }

        inserted
    }

    /// Publishes the shared membership head for a homogeneous spawn batch.
    pub(crate) fn commit_spawn_batch_memberships(
        &mut self,
        entities: &[Entity],
        head: ComponentId,
    ) {
        let Some(max_index) = entities.iter().map(|entity| entity.index() as usize).max() else {
            return;
        };

        if max_index >= self.entity_heads.len() {
            self.entity_heads
                .resize(max_index + 1, INVALID_SPARSE_COMPONENT);
        }

        let head = head.index() as u32;
        for entity in entities {
            let entry = &mut self.entity_heads[entity.index() as usize];
            debug_assert_eq!(*entry, INVALID_SPARSE_COMPONENT);
            *entry = head;
        }
    }

    /// Inserts or replaces one sparse component and links a newly-created
    /// membership immediately.
    ///
    /// This path avoids deferred-membership bookkeeping for bundles containing
    /// exactly one sparse component.
    ///
    /// # Safety
    ///
    /// `value` must satisfy the concrete sparse set's `try_insert_move` contract
    /// for `component`.
    pub(crate) unsafe fn insert_move_immediate(
        &mut self,
        component: ComponentId,
        entity: Entity,
        location: EntityLocation,
        value: *mut u8,
        change_tick: ChangeTick,
    ) -> Option<SparseInsertResult> {
        let outcome =
            unsafe { self.insert_move_unlinked(component, entity, location, value, change_tick)? };

        match outcome {
            SparseInsertOutcome::Inserted { dense_pos } => {
                self.link_membership(entity, component, dense_pos);

                Some(SparseInsertResult { inserted: true })
            }

            SparseInsertOutcome::Replaced => Some(SparseInsertResult { inserted: false }),
        }
    }

    /// Inserts or replaces one sparse component while deferring entity-local
    /// membership linking until the complete prepared bundle has been moved.
    ///
    /// # Safety
    ///
    /// `value` must satisfy the concrete sparse set's `try_insert_move`
    /// contract for `component`.
    pub(crate) unsafe fn insert_move_deferred(
        &mut self,
        component: ComponentId,
        entity: Entity,
        location: EntityLocation,
        value: *mut u8,
        change_tick: ChangeTick,
    ) -> Option<SparseInsertResult> {
        let outcome =
            unsafe { self.insert_move_unlinked(component, entity, location, value, change_tick)? };

        match outcome {
            SparseInsertOutcome::Inserted { dense_pos } => {
                self.pending_memberships.push(PendingSparseMembership {
                    component,
                    dense_pos,
                });

                Some(SparseInsertResult { inserted: true })
            }
            SparseInsertOutcome::Replaced => Some(SparseInsertResult { inserted: false }),
        }
    }

    /// Commits all memberships inserted since `begin_deferred_insertions` as
    /// one linked chain at the head of `entity`'s sparse membership list.
    pub(crate) fn finish_deferred_insertions(&mut self, entity: Entity) {
        if self.pending_memberships.is_empty() {
            return;
        }

        let mut pending = std::mem::take(&mut self.pending_memberships);
        self.link_memberships(entity, &pending);
        pending.clear();
        self.pending_memberships = pending;
    }

    /// Removes one sparse component and unlinks its entity-local membership.
    pub(crate) fn remove_component(&mut self, component: ComponentId, entity: Entity) -> bool {
        self.remove_component_entry(component, entity).is_some()
    }

    /// Removes one sparse component from a prevalidated entity batch.
    ///
    /// All dense positions and membership links are captured before the first
    /// storage mutation. The component set is then compacted once, allowing
    /// its driver and dense metadata to be repaired component-major instead of
    /// re-entering the generic single-entity path for every item.
    pub(crate) fn remove_component_batch(
        &mut self,
        component: ComponentId,
        entities: &[Entity],
    ) -> usize {
        let Some(set) = self.get_sparse_set_dyn(component.index()) else {
            return 0;
        };

        let mut removals = Vec::with_capacity(entities.len().min(set.len()));
        let mut links = Vec::with_capacity(removals.capacity());

        for &entity in entities {
            let Some(probe) = set.prepare_batch_remove(entity) else {
                continue;
            };

            removals.push(SparseBatchRemoval {
                entity,
                dense_pos: probe.dense_pos,
            });
            links.push((entity, probe.links));
        }

        let removed = removals.len();
        if removed == 0 {
            return 0;
        }

        // Release the immutable set borrow before repairing neighbouring
        // membership nodes, which can belong to any other sparse set.
        for (entity, links) in links {
            self.unlink_membership(entity, links);
        }

        self.get_sparse_set_dyn_mut(component.index())
            .expect("prepared sparse set must exist")
            .remove_batch_prepared(&mut removals);

        removed
    }

    /// Removes one sparse value and unlinks its membership chain entry.
    #[inline]
    fn remove_component_entry(
        &mut self,
        component: ComponentId,
        entity: Entity,
    ) -> Option<SparseRemoveOutcome> {
        let removed = {
            let set = self.get_sparse_set_dyn_mut(component.index())?;
            set.remove_with_links(entity)?
        };

        self.unlink_membership(entity, removed.links);
        Some(removed)
    }

    /// Removes every sparse component owned by `entity` and reports each
    /// affected component.
    ///
    /// This walks the entity's sparse-membership chain instead of scanning all
    /// registered sparse component types.
    pub(crate) fn remove_entity_from_all<F>(&mut self, entity: Entity, mut removed: F)
    where
        F: FnMut(ComponentId),
    {
        let entity_index = entity.index() as usize;
        if entity_index >= self.entity_heads.len() {
            return;
        }

        let mut component = self.entity_heads[entity_index];
        self.entity_heads[entity_index] = INVALID_SPARSE_COMPONENT;

        while component != INVALID_SPARSE_COMPONENT {
            let component_id = ComponentId::from_index(component as usize);

            let outcome = {
                let set = self
                    .get_sparse_set_dyn_mut(component_id.index())
                    .expect("sparse membership references a missing sparse set");

                set.remove_with_links(entity)
                    .expect("sparse membership references a missing entity")
            };

            removed(component_id);
            component = outcome.links.next;
        }
    }

    /// Resolves sparse memberships for a destruction batch without mutating
    /// any set. Groups are component-major so each type-erased lookup happens
    /// once during execution rather than once per entity membership.
    pub(crate) fn prepare_despawn_batch(
        &self,
        entities: &[Entity],
        groups: &mut SparseRemovalGroups,
    ) {
        debug_assert!(groups.is_empty());

        for &entity in entities {
            let mut component = self.entity_head(entity);

            while component != INVALID_SPARSE_COMPONENT {
                let component_id = ComponentId::from_index(component as usize);
                let set = self
                    .get_sparse_set_dyn(component_id.index())
                    .expect("sparse membership must reference an existing set");

                if set.len() <= entities.len() {
                    let links = set
                        .membership_links(entity)
                        .expect("sparse membership must reference an existing entity");

                    groups.count_full_candidate(component_id, set.len());
                    component = links.next;
                } else {
                    let probe = set
                        .prepare_batch_remove(entity)
                        .expect("sparse membership must reference an existing entity");

                    groups.push_known_partial(
                        component_id,
                        SparseBatchRemoval {
                            entity,
                            dense_pos: probe.dense_pos,
                        },
                    );

                    component = probe.links.next;
                }
            }
        }

        groups.classify_full_candidates();

        if !groups.has_deferred_partial() {
            return;
        }

        // Only ambiguous groups need a second traversal. Groups known to be
        // partial were prepared above, while complete groups remain metadata-free.
        for &entity in entities {
            let mut component = self.entity_head(entity);

            while component != INVALID_SPARSE_COMPONENT {
                let component_id = ComponentId::from_index(component as usize);
                let set = self
                    .get_sparse_set_dyn(component_id.index())
                    .expect("sparse membership must reference an existing set");

                if groups.needs_deferred_removal(component_id) {
                    let probe = set
                        .prepare_batch_remove(entity)
                        .expect("sparse membership must reference an existing entity");

                    groups.push_deferred_removal(
                        component_id,
                        SparseBatchRemoval {
                            entity,
                            dense_pos: probe.dense_pos,
                        },
                    );

                    component = probe.links.next;
                } else {
                    component = set
                        .membership_links(entity)
                        .expect("sparse membership must reference an existing entity")
                        .next;
                }
            }
        }
    }

    /// Executes a component-major sparse destruction batch prepared by
    /// `prepare_despawn_batch`.
    pub(crate) fn execute_despawn_batch(
        &mut self,
        entities: &[Entity],
        groups: &mut SparseRemovalGroups,
    ) {
        for &entity in entities {
            let entity_index = entity.index() as usize;

            if let Some(head) = self.entity_heads.get_mut(entity_index) {
                *head = INVALID_SPARSE_COMPONENT;
            }
        }

        let touched = groups.take_touched();

        for &component in &touched {
            if groups.is_full_set(component) {
                debug_assert_eq!(
                    groups.target_count(component),
                    self.get_sparse_set_dyn(component.index())
                        .expect("prepared sparse set must exist")
                        .len()
                );

                self.get_sparse_set_dyn_mut(component.index())
                    .expect("prepared sparse set must exist")
                    .clear_all_entries();
            } else {
                let mut removals = std::mem::take(groups.removals_mut(component));
                debug_assert_eq!(removals.len(), groups.target_count(component));

                self.get_sparse_set_dyn_mut(component.index())
                    .expect("prepared sparse set must exist")
                    .remove_batch_prepared(&mut removals);

                removals.clear();
                *groups.removals_mut(component) = removals;
            }
        }

        groups.finish(touched);
    }

    /// Removes every sparse component of `entity` rejected by `keep`.
    ///
    /// Only sparse components currently owned by `entity` are visited.
    pub(crate) fn retain_entity_components<K, F>(
        &mut self,
        entity: Entity,
        mut keep: K,
        mut removed: F,
    ) where
        K: FnMut(ComponentId) -> bool,
        F: FnMut(ComponentId),
    {
        let mut component = self.entity_head(entity);

        while component != INVALID_SPARSE_COMPONENT {
            let component_id = ComponentId::from_index(component as usize);

            component = if keep(component_id) {
                self.get_sparse_set_dyn(component_id.index())
                    .and_then(|set| set.membership_links(entity))
                    .expect("sparse membership is inconsistent")
                    .next
            } else {
                let outcome = self
                    .remove_component_entry(component_id, entity)
                    .expect("sparse membership is inconsistent");
                removed(component_id);
                outcome.links.next
            };
        }
    }

    /// Returns the query-driver buckets of the sparse set at `index`, if any.
    pub(crate) fn driver_buckets(&self, index: usize) -> Option<&[SparseDriverBucket]> {
        Some(self.get_sparse_set_dyn(index)?.driver_buckets())
    }

    /// Returns the driver revision of the sparse set at `index`, if any.
    #[inline]
    pub(crate) fn driver_revision(&self, index: usize) -> Option<u64> {
        Some(self.get_sparse_set_dyn(index)?.driver_revision())
    }

    /// Updates canonical locations, and active query drivers, for the sparse
    /// components actually owned by `entity`.
    pub(crate) fn update_entity_location(&mut self, entity: Entity, location: EntityLocation) {
        let mut component = self.entity_head(entity);

        while component != INVALID_SPARSE_COMPONENT {
            let component_id = ComponentId::from_index(component as usize);

            let next = self
                .get_sparse_set_dyn_mut(component_id.index())
                .expect("sparse membership references a missing sparse set")
                .update_driver_location_and_next(entity, location)
                .expect("sparse membership references a missing entity");

            component = next;
        }
    }

    /// Captures one entity's sparse location repairs without mutating a set.
    ///
    /// The membership traversal resolves dense positions once and groups the
    /// resulting work by component for the later execution phase.
    pub(crate) fn prepare_entity_location_update(
        &self,
        entity: Entity,
        location: EntityLocation,
        groups: &mut SparseLocationUpdateGroups,
    ) {
        let mut component = self.entity_head(entity);

        while component != INVALID_SPARSE_COMPONENT {
            let component_id = ComponentId::from_index(component as usize);
            let probe = self
                .get_sparse_set_dyn(component_id.index())
                .expect("sparse membership references a missing sparse set")
                .prepare_batch_remove(entity)
                .expect("sparse membership references a missing entity");

            groups.push(
                component_id,
                SparseLocationUpdate {
                    location,
                    dense_pos: probe.dense_pos,
                },
            );

            component = probe.links.next;
        }
    }

    /// Applies previously captured repairs component-major.
    pub(crate) fn execute_entity_location_updates(
        &mut self,
        groups: &mut SparseLocationUpdateGroups,
    ) {
        if groups.is_empty() {
            return;
        }

        let touched = groups.take_touched();

        for &component in &touched {
            let updates = groups.updates(component);
            self.get_sparse_set_dyn_mut(component.index())
                .expect("prepared sparse set must exist")
                .update_locations_batch(updates);
        }

        groups.finish(touched);
    }

    #[inline]
    /// Returns the head of `entity`'s sparse-membership chain.
    fn entity_head(&self, entity: Entity) -> u32 {
        self.entity_heads
            .get(entity.index() as usize)
            .copied()
            .unwrap_or(INVALID_SPARSE_COMPONENT)
    }

    #[inline]
    /// Returns whether `entity` owns at least one sparse component.
    pub(crate) fn entity_has_components(&self, entity: Entity) -> bool {
        self.entity_head(entity) != INVALID_SPARSE_COMPONENT
    }

    #[inline]
    /// Sets the head of `entity`'s membership chain, growing the table.
    fn set_entity_head(&mut self, entity: Entity, component: u32) {
        let index = entity.index() as usize;

        if index >= self.entity_heads.len() {
            self.entity_heads
                .resize(index + 1, INVALID_SPARSE_COMPONENT);
        }

        self.entity_heads[index] = component;
    }

    /// Links one newly-created sparse membership at the head of the entity-local
    /// membership chain.
    fn link_membership(&mut self, entity: Entity, component: ComponentId, dense_pos: u32) {
        let component_raw = component.index() as u32;
        let old_head = self.entity_head(entity);

        {
            let set = self
                .get_sparse_set_dyn_mut(component.index())
                .expect("inserted sparse set must exist");

            let linked =
                set.set_membership_links_dense(dense_pos, INVALID_SPARSE_COMPONENT, old_head);

            debug_assert!(linked, "new sparse membership dense position must exist");
        }

        if old_head != INVALID_SPARSE_COMPONENT {
            let old_head_id = ComponentId::from_index(old_head as usize);

            let set = self
                .get_sparse_set_dyn_mut(old_head_id.index())
                .expect("sparse membership head must exist");

            let linked = set.set_membership_prev(entity, component_raw);

            debug_assert!(linked, "sparse membership head entity must exist");
        }

        self.set_entity_head(entity, component_raw);
    }

    /// Links a group of newly-created memberships with one entity-head update.
    fn link_memberships(&mut self, entity: Entity, inserted: &[PendingSparseMembership]) {
        debug_assert!(!inserted.is_empty());

        if let [membership] = inserted {
            self.link_membership(entity, membership.component, membership.dense_pos);
            return;
        }

        let old_head = self.entity_head(entity);

        for (index, current) in inserted.iter().copied().enumerate() {
            let prev = inserted
                .get(index + 1)
                .map_or(INVALID_SPARSE_COMPONENT, |entry| {
                    entry.component.index() as u32
                });

            let next = if index == 0 {
                old_head
            } else {
                inserted[index - 1].component.index() as u32
            };

            let set = self
                .get_sparse_set_dyn_mut(current.component.index())
                .expect("inserted sparse set must exist");

            let linked = set.set_membership_links_dense(current.dense_pos, prev, next);

            debug_assert!(linked, "new sparse membership dense position must exist");
        }

        if old_head != INVALID_SPARSE_COMPONENT {
            let old_head_id = ComponentId::from_index(old_head as usize);

            let set = self
                .get_sparse_set_dyn_mut(old_head_id.index())
                .expect("sparse membership head must exist");

            let linked = set.set_membership_prev(entity, inserted[0].component.index() as u32);

            debug_assert!(linked, "sparse membership head entity must exist");
        }

        let new_head = inserted
            .last()
            .expect("inserted memberships cannot be empty")
            .component
            .index() as u32;

        self.set_entity_head(entity, new_head);
    }

    /// Removes one node from the entity's doubly-linked sparse membership list.
    fn unlink_membership(&mut self, entity: Entity, links: super::SparseMembershipLinks) {
        if links.previous == INVALID_SPARSE_COMPONENT {
            self.set_entity_head(entity, links.next);
        } else {
            let prev = ComponentId::from_index(links.previous as usize);
            let set = self
                .get_sparse_set_dyn_mut(prev.index())
                .expect("sparse membership prev set must exist");
            let linked = set.set_membership_next(entity, links.next);
            debug_assert!(linked, "sparse membership prev entity must exist");
        }

        if links.next != INVALID_SPARSE_COMPONENT {
            let next = ComponentId::from_index(links.next as usize);
            let set = self
                .get_sparse_set_dyn_mut(next.index())
                .expect("sparse membership next set must exist");
            let linked = set.set_membership_prev(entity, links.previous);
            debug_assert!(linked, "sparse membership next entity must exist");
        }
    }
}
