//! Typed sparse-set implementation for one component.
//!
//! The sparse array maps entity indices to dense positions, while dense arrays
//! store component values, change ticks, and the entity occupying each slot.

use core::ptr;
use std::{
    cmp::Reverse,
    sync::{Mutex, OnceLock, PoisonError},
};

use crate::{
    change_detection::{ChangeTick, ComponentTicks},
    component::{
        archetype::plan::batch_transition::BatchTransitionInput,
        sparse_set::{
            driver::SparseMembershipLinks, SparseBatchProbe, SparseBatchRemoval,
            SparseInsertBatchEntry, SparseInsertOutcome, SparseLocationUpdate, SparseRemoveOutcome,
            SparseSpawnBatchEntry,
        },
        Component,
    },
    entity::{metadata::EntityLocation, Entity},
};

use super::{dense::SparseDenseStorage, SparseDriverBucket, SparseDriverIndex, SparseSet};

/// Component-index sentinel for empty entity heads and membership-chain ends.
pub(crate) const INVALID_SPARSE_COMPONENT: u32 = u32::MAX;
/// Sparse-index sentinel for an entity slot with no dense entry.
const INVALID_DENSE_POS: u32 = u32::MAX;
/// Typed sparse-set storage for one component type.
pub struct ComponentSparseSet<T> {
    /// Entity-indexed map into `dense`; `Entity::INVALID.index()` marks empty slots.
    sparse: Vec<u32>,
    /// Compact values, identities, and auxiliary metadata kept in aligned
    /// parallel arrays. Entity validation only touches the identity array.
    dense: SparseDenseStorage<T>,
    /// Canonical locations retained only while the query driver is dormant.
    ///
    /// The mutex permits one-time activation through `&self`. Structural
    /// mutation owns `&mut self` and therefore accesses the vector through
    /// `Mutex::get_mut` without acquiring the lock.
    dormant_locations: Mutex<Vec<EntityLocation>>,
    /// Query-only metadata, built when first requested.
    driver_index: OnceLock<SparseDriverIndex>,
}

impl<T> ComponentSparseSet<T> {
    /// Borrows the dormant locations without locking, recovering the array even
    /// from a poisoned mutex.
    #[inline]
    fn dormant_locations_mut(&mut self) -> &mut Vec<EntityLocation> {
        self.dormant_locations
            .get_mut()
            .unwrap_or_else(PoisonError::into_inner)
    }

    /// Returns the dense position of `entity`, validating the sparse slot
    /// against the dense identity array so stale generations read as absent.
    #[inline]
    fn dense_pos_of_raw(&self, entity: Entity) -> Option<u32> {
        let &dense_pos = self.sparse.get(entity.index() as usize)?;

        if dense_pos == INVALID_DENSE_POS
            || self.dense.entity_at(dense_pos as usize) != Some(entity)
        {
            return None;
        }

        Some(dense_pos)
    }

    /// Returns the membership links recorded for `entity`, if it belongs to
    /// this set.
    #[inline]
    pub(crate) fn membership_links(&self, entity: Entity) -> Option<SparseMembershipLinks> {
        let dense_pos = self.dense_pos_of_raw(entity)?;
        self.dense
            .auxiliary_at(dense_pos as usize)
            .map(|metadata| metadata.links)
    }

    /// Warms the sparse-index line that the next membership check of `entity`
    /// will load, hiding the first of its two dependent reads.
    #[inline]
    pub(crate) fn prefetch_membership_line(&self, entity: Entity) {
        let index = entity.index() as usize;
        if index < self.sparse.len() {
            crate::utils::prefetch::prefetch_element(self.sparse.as_ptr(), index);
        }
    }

    /// Overwrites both membership links of the entry at `dense_pos`; returns
    /// whether the position is initialized.
    #[inline]
    pub(crate) fn set_membership_links_dense(
        &mut self,
        dense_pos: u32,
        previous: u32,
        next: u32,
    ) -> bool {
        let Some(metadata) = self.dense.auxiliary_at_mut(dense_pos as usize) else {
            return false;
        };

        metadata.links = SparseMembershipLinks { previous, next };
        true
    }

    /// Points the `previous` membership link of `entity` at a component index;
    /// returns whether the entity belongs to this set.
    #[inline]
    pub(crate) fn set_membership_prev(&mut self, entity: Entity, previous: u32) -> bool {
        let Some(dense_pos) = self.dense_pos_of_raw(entity) else {
            return false;
        };

        self.dense.auxiliary[dense_pos as usize].links.previous = previous;
        true
    }

    /// Points the `next` membership link of `entity` at a component index;
    /// returns whether the entity belongs to this set.
    #[inline]
    pub(crate) fn set_membership_next(&mut self, entity: Entity, next: u32) -> bool {
        let Some(dense_pos) = self.dense_pos_of_raw(entity) else {
            return false;
        };

        self.dense.auxiliary[dense_pos as usize].links.next = next;
        true
    }

    /// Swap-removes the entry at `dense_pos`, refreshing the canonical location
    /// store and repairing the sparse slot of the entity moved into the hole.
    /// Returns the removed value with its membership links.
    fn remove_dense_pos(&mut self, dense_pos: usize) -> Option<(T, SparseMembershipLinks)> {
        if dense_pos >= self.dense.len() {
            return None;
        }

        if let Some(driver) = self.driver_index.get_mut() {
            let removed = driver
                .remove_dense(u32::try_from(dense_pos).expect("dense position exceeds u32::MAX"));

            debug_assert!(
                removed,
                "active sparse driver must contain every dense entry"
            );
        } else {
            self.dormant_locations_mut().swap_remove(dense_pos);
        }

        let removal = self.dense.swap_remove(dense_pos)?;
        self.sparse[removal.entity.index() as usize] = INVALID_DENSE_POS;

        if let Some(moved_entity) = removal.moved_entity {
            self.sparse[moved_entity.index() as usize] =
                u32::try_from(dense_pos).expect("dense position exceeds u32::MAX");
        }

        Some((removal.value, removal.links))
    }

    /// Updates the stored location of one member and returns the next
    /// component in its membership chain, if the entity belongs to the set.
    pub(crate) fn update_location(
        &mut self,
        entity: Entity,
        location: EntityLocation,
    ) -> Option<u32> {
        let dense_pos = self.dense_pos_of_raw(entity)? as usize;
        let next_component = self.dense.auxiliary[dense_pos].links.next;

        if let Some(driver_index) = self.driver_index.get_mut() {
            let updated = driver_index.update_location(
                u32::try_from(dense_pos).expect("dense position exceeds u32::MAX"),
                entity,
                location,
            );
            debug_assert!(
                updated,
                "active sparse driver must contain every dense entry"
            );
        } else {
            self.dormant_locations_mut()[dense_pos] = location;
        }

        Some(next_component)
    }
}

// SAFETY:
// Sparse membership is immutable while systems run. Each mutable query pointer
// addresses one UnsafeCell element and requires filtered-access exclusivity.
// Structural methods require `&mut self`.
unsafe impl<C: Component> Sync for ComponentSparseSet<C> {}

impl<C: Component> Default for ComponentSparseSet<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Component> ComponentSparseSet<C> {
    /// Creates an empty sparse set.
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: SparseDenseStorage::new(),
            dormant_locations: Mutex::new(Vec::new()),
            driver_index: OnceLock::new(),
        }
    }

    /// Returns the component stored at `dense_pos` without validating the
    /// position or the entity.
    ///
    /// # Safety
    ///
    /// `dense_pos` must identify `expected_entity` in this sparse set. The
    /// caller must hold a filtered-access set that excludes a writer for this
    /// entity while the returned reference is alive.
    #[inline]
    pub(crate) unsafe fn get_ref_by_dense_pos_unchecked(
        &self,
        dense_pos: u32,
        expected_entity: Entity,
    ) -> &C {
        let dense_pos = dense_pos as usize;
        debug_assert!(dense_pos < self.dense.len());
        debug_assert_eq!(self.dense.entity_at(dense_pos), Some(expected_entity));

        let cell = unsafe { self.dense.values.get_unchecked(dense_pos) };
        unsafe { &*cell.get() }
    }

    /// Returns immutable access to the component owned by `entity`.
    #[inline]
    pub fn get_ref(&self, entity: Entity) -> Option<&C> {
        let dense_pos = self.dense_pos_of(entity)? as usize;
        let cell = self.dense.values.get(dense_pos)?;

        // SAFETY: the validated filtered access set excludes a writer for this
        // entity while the returned reference is alive.
        Some(unsafe { &*cell.get() })
    }

    /// Returns the dense position of `entity`, if it belongs to this set.
    #[inline]
    fn dense_pos_of(&self, entity: Entity) -> Option<u32> {
        self.dense_pos_of_raw(entity)
    }

    /// Returns immutable change ticks for `entity`.
    pub(crate) fn get_ticks(&self, entity: Entity) -> Option<&ComponentTicks> {
        let dense_pos = self.dense_pos_of(entity)? as usize;
        let metadata = self.dense.auxiliary_at(dense_pos)?;

        // SAFETY: tick writers are excluded for this entity while the returned
        // reference is alive.
        Some(unsafe { &*metadata.ticks.get() })
    }

    /// Returns mutable value and tick pointers while holding `&mut self`.
    pub(crate) fn get_mut(&mut self, entity: Entity) -> Option<(*mut C, *mut ComponentTicks)> {
        let dense_pos = self.dense_pos_of(entity)? as usize;
        Some((
            self.dense.values[dense_pos].get(),
            self.dense.auxiliary[dense_pos].ticks.get(),
        ))
    }

    /// Returns mutable value and tick pointers without mutably borrowing the
    /// sparse set as a whole.
    ///
    /// # Safety
    ///
    /// The caller must hold value and tick write access for `T`, must prevent
    /// structural mutation while the pointers are alive, and must not create
    /// another incompatible access to the same component.
    pub(crate) unsafe fn get_mut_ptrs(
        &self,
        entity: Entity,
    ) -> Option<(*mut C, *mut ComponentTicks)> {
        let dense_pos = self.dense_pos_of(entity)? as usize;
        Some((
            self.dense.values.get(dense_pos)?.get(),
            self.dense.auxiliary.get(dense_pos)?.ticks.get(),
        ))
    }

    /// Returns mutable value and tick pointers using a position supplied by
    /// this sparse set's driver index.
    ///
    /// # Safety
    ///
    /// `dense_pos` must belong to this sparse set and identify
    /// `expected_entity`. The caller must hold exclusive component access and
    /// prevent structural mutation while either pointer is alive.
    #[inline]
    pub(crate) unsafe fn get_mut_ptrs_by_dense_pos_unchecked(
        &self,
        dense_pos: u32,
        expected_entity: Entity,
    ) -> (*mut C, *mut ComponentTicks) {
        let dense_pos = dense_pos as usize;
        debug_assert!(dense_pos < self.dense.len());
        debug_assert_eq!(self.dense.entity_at(dense_pos), Some(expected_entity));

        let value = unsafe { self.dense.values.get_unchecked(dense_pos).get() };
        let ticks = unsafe { self.dense.auxiliary.get_unchecked(dense_pos).ticks.get() };
        (value, ticks)
    }

    /// Returns change ticks using a position supplied by this set's driver.
    ///
    /// # Safety
    ///
    /// `dense_pos` must identify `expected_entity` in this sparse set, and
    /// structural mutation must remain disabled while the result is alive.
    #[inline]
    pub(crate) unsafe fn get_ticks_by_dense_pos_unchecked(
        &self,
        dense_pos: u32,
        expected_entity: Entity,
    ) -> &ComponentTicks {
        let dense_pos = dense_pos as usize;
        debug_assert!(dense_pos < self.dense.len());
        debug_assert_eq!(self.dense.entity_at(dense_pos), Some(expected_entity));

        unsafe { &*self.dense.auxiliary.get_unchecked(dense_pos).ticks.get() }
    }

    /// Returns whether `entity` has a value in this sparse set.
    #[inline]
    pub fn contains_entity(&self, entity: Entity) -> bool {
        self.dense_pos_of(entity).is_some()
    }

    /// Builds the driver index from the dormant locations, completing the
    /// one-time dormant-to-active transition.
    ///
    /// Once built, the driver is the canonical location store; the dormant
    /// vector is released instead of retaining duplicate per-entity metadata.
    #[cold]
    #[inline(never)]
    fn activate_driver_index(&self) -> SparseDriverIndex {
        let mut locations = self
            .dormant_locations
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        assert_eq!(
            self.dense.len(),
            locations.len(),
            "sparse entities and locations must remain aligned"
        );
        let driver_index = SparseDriverIndex::build(
            self.dense
                .entities
                .iter()
                .copied()
                .zip(locations.iter().copied()),
        );

        // Once the driver is active it becomes the canonical location index.
        // Replacing the vector releases its allocation instead of retaining
        // duplicate per-entity metadata for the lifetime of the sparse set.
        *locations = Vec::new();

        driver_index
    }

    /// Returns the active driver index, activating it on first use.
    #[inline]
    fn driver_index(&self) -> &SparseDriverIndex {
        self.driver_index
            .get_or_init(|| self.activate_driver_index())
    }

    /// Returns the driver's structural revision, building the driver if needed.
    #[inline]
    pub(crate) fn driver_revision(&self) -> u64 {
        self.driver_index().revision()
    }

    /// Returns the driver buckets, building the driver on first use.
    #[inline]
    pub(crate) fn driver_buckets(&self) -> &[SparseDriverBucket] {
        self.driver_index().buckets()
    }

    /// Location-update entry point used by the sparse-set manager.
    pub(crate) fn update_driver_location_and_next(
        &mut self,
        entity: Entity,
        location: EntityLocation,
    ) -> Option<u32> {
        self.update_location(entity, location)
    }

    /// Applies prevalidated location repairs to whichever location
    /// representation is currently canonical.
    fn update_locations_batch(&mut self, updates: &[SparseLocationUpdate]) {
        if updates.is_empty() {
            return;
        }

        debug_assert!(updates
            .iter()
            .all(|update| self.dense.entity_at(update.dense_pos as usize).is_some()));

        if let Some(driver) = self.driver_index.get_mut() {
            driver.update_locations_batch(updates);
            return;
        }

        let locations = self.dormant_locations_mut();
        for update in updates {
            locations[update.dense_pos as usize] = update.location;
        }
    }

    /// Removes `entity` by swap-removing from dense storage.
    #[inline]
    pub fn remove_entity(&mut self, entity: Entity) -> bool {
        let Some(dense_pos) = self.dense_pos_of(entity) else {
            return false;
        };

        self.remove_dense_pos(dense_pos as usize);
        true
    }
}

impl<C: Component> ComponentSparseSet<C> {
    /// Captures the dense position and membership links of `entity` without
    /// mutating the set.
    #[inline]
    fn prepare_batch_remove(&self, entity: Entity) -> Option<SparseBatchProbe> {
        let dense_pos = self.dense_pos_of(entity)?;
        let links = self.dense.auxiliary[dense_pos as usize].links;

        Some(SparseBatchProbe { dense_pos, links })
    }

    /// Compacts one prepared removal batch, clearing the whole set when the
    /// batch covers it and applying descending-order removals otherwise.
    fn remove_batch_prepared(&mut self, removals: &mut [SparseBatchRemoval]) {
        if removals.is_empty() {
            return;
        }

        debug_assert!(removals.len() <= self.dense.len());

        // The complete-set path is particularly important for destruction
        // of worlds, scenes, chunks and benchmark populations.
        if removals.len() == self.dense.len() {
            self.clear_all_entries();
            return;
        }

        // Descending positions remain valid under swap removal. Removing a
        // higher position cannot change any lower pending position.
        removals.sort_unstable_by_key(|entry| Reverse(entry.dense_pos));

        debug_assert!(
            removals
                .windows(2)
                .all(|pair| pair[0].dense_pos != pair[1].dense_pos),
            "prepared sparse removals must be unique"
        );

        self.remove_partial_prepared(removals);
    }

    /// Removes every entry without per-entity records, retaining allocations.
    fn clear_all_entries(&mut self) {
        for &entity in &self.dense.entities {
            self.sparse[entity.index() as usize] = INVALID_DENSE_POS;
        }

        if let Some(driver) = self.driver_index.get_mut() {
            driver.clear();
        } else {
            self.dormant_locations_mut().clear();
        }

        // Drops every component value while retaining allocations for reuse.
        self.dense.clear();
    }

    /// Applies one descending-sorted partial batch, refreshing the location
    /// store once before compacting dense storage.
    fn remove_partial_prepared(&mut self, removals: &[SparseBatchRemoval]) {
        // Update the canonical location representation once as a group.
        if let Some(driver) = self.driver_index.get_mut() {
            driver.remove_dense_batch(removals.iter().map(|entry| entry.dense_pos));
        } else {
            let locations = self.dormant_locations_mut();

            for removal in removals {
                locations.swap_remove(removal.dense_pos as usize);
            }
        }

        for removal in removals {
            let dense_pos = removal.dense_pos as usize;

            debug_assert_eq!(self.dense.entity_at(dense_pos), Some(removal.entity),);

            let removed = self
                .dense
                .swap_remove(dense_pos)
                .expect("prepared dense position must remain valid");

            debug_assert_eq!(removed.entity, removal.entity);

            self.sparse[removed.entity.index() as usize] = INVALID_DENSE_POS;

            if let Some(moved_entity) = removed.moved_entity {
                self.sparse[moved_entity.index() as usize] = removal.dense_pos;
            }

            // `removed.value` is dropped here exactly once.
            drop(removed);
        }
    }
}

impl<C: Component> SparseSet for ComponentSparseSet<C> {
    fn len(&self) -> usize {
        self.dense.len()
    }

    unsafe fn try_insert_move(
        &mut self,
        entity: Entity,
        location: EntityLocation,
        value: *mut u8,
        change_tick: ChangeTick,
    ) -> Option<SparseInsertOutcome> {
        if value.is_null() {
            return None;
        }

        let index = entity.index() as usize;

        if index >= self.sparse.len() {
            self.sparse.resize(index + 1, INVALID_DENSE_POS);
        }

        let pos = self.sparse[index];

        if pos == INVALID_DENSE_POS {
            // Reserve dormant location storage before ownership moves out of
            // the erased bundle. Dense storage reserves all of its parallel
            // arrays atomically from the caller's perspective.
            if self.driver_index.get().is_none() {
                self.dormant_locations_mut().reserve(1);
            }

            let moved: C = unsafe { ptr::read(value as *mut C) };
            let dense_pos = self
                .dense
                .push(entity, moved, ComponentTicks::new(change_tick));

            // Exactly one location representation is maintained. Dormant sets
            // keep compact source locations; active sets update the driver.
            if let Some(driver_index) = self.driver_index.get_mut() {
                driver_index.insert_dense(dense_pos, entity, location);
            } else {
                self.dormant_locations_mut().push(location);
            }

            self.sparse[index] = dense_pos;

            return Some(SparseInsertOutcome::Inserted { dense_pos });
        }

        let dense_pos = pos as usize;
        assert_eq!(
            self.dense.entity_at(dense_pos),
            Some(entity),
            "sparse index points to another entity generation"
        );

        // Replacing a sparse value cannot change its archetype location. Any
        // preceding table transition has already synchronized every sparse set.
        let moved: C = unsafe { ptr::read(value as *mut C) };
        *self.dense.values[dense_pos].get_mut() = moved;
        self.dense.auxiliary[dense_pos]
            .ticks
            .get_mut()
            .mark_changed(change_tick);

        Some(SparseInsertOutcome::Replaced)
    }

    fn prepare_spawn_batch(&mut self, entities: &[Entity]) -> bool {
        let Some(max_index) = entities.iter().map(|entity| entity.index() as usize).max() else {
            return true;
        };

        if max_index >= self.sparse.len() {
            self.sparse.resize(max_index + 1, INVALID_DENSE_POS);
        }

        if entities
            .iter()
            .any(|entity| self.sparse[entity.index() as usize] != INVALID_DENSE_POS)
        {
            return false;
        }

        self.dense.reserve(entities.len());

        if self.driver_index.get().is_none() {
            self.dormant_locations_mut().reserve(entities.len());
        }

        true
    }

    fn prepare_insert_batch(&mut self, inputs: &[BatchTransitionInput]) {
        let Some(max_index) = inputs
            .iter()
            .map(|input| input.entity.index() as usize)
            .max()
        else {
            return;
        };

        if max_index >= self.sparse.len() {
            self.sparse.resize(max_index + 1, INVALID_DENSE_POS);
        }

        let inserted = inputs
            .iter()
            .filter(|input| self.dense_pos_of_raw(input.entity).is_none())
            .count();

        self.dense.reserve(inserted);

        if self.driver_index.get().is_none() {
            self.dormant_locations_mut().reserve(inserted);
        }
    }

    unsafe fn insert_spawn_batch_prepared(
        &mut self,
        entries: &[SparseSpawnBatchEntry],
        change_tick: ChangeTick,
    ) {
        if entries.is_empty() {
            return;
        }

        let first_dense_pos =
            u32::try_from(self.dense.len()).expect("sparse component count exceeds u32::MAX");

        for entry in entries {
            let index = entry.entity.index() as usize;
            debug_assert!(index < self.sparse.len());
            debug_assert_eq!(self.sparse[index], INVALID_DENSE_POS);

            let moved: C = unsafe { ptr::read(entry.item.ptr as *mut C) };
            unsafe {
                entry.item.mark_consumed();
            }

            let dense_pos = self.dense.push_reserved_linked(
                entry.entity,
                moved,
                ComponentTicks::new(change_tick),
                entry.links,
            );
            self.sparse[index] = dense_pos;
        }

        if let Some(driver_index) = self.driver_index.get_mut() {
            driver_index.insert_dense_batch(
                first_dense_pos,
                entries.len(),
                entries.iter().map(|entry| (entry.entity, entry.location)),
            );
        } else {
            self.dormant_locations_mut()
                .extend(entries.iter().map(|entry| entry.location));
        }
    }

    unsafe fn insert_batch_prepared(
        &mut self,
        entries: &mut [SparseInsertBatchEntry],
        change_tick: ChangeTick,
    ) {
        if entries.is_empty() {
            return;
        }

        let first_dense_pos =
            u32::try_from(self.dense.len()).expect("sparse component count exceeds u32::MAX");
        let mut inserted = 0usize;

        for entry in entries.iter_mut() {
            let index = entry.entity.index() as usize;
            debug_assert!(index < self.sparse.len());
            entry.inserted_dense_pos = None;

            let dense_pos = self.sparse[index];

            if dense_pos == INVALID_DENSE_POS {
                let moved: C = unsafe { ptr::read(entry.item.ptr as *mut C) };
                unsafe {
                    entry.item.mark_consumed();
                }

                let dense_pos = self.dense.push_reserved_linked(
                    entry.entity,
                    moved,
                    ComponentTicks::new(change_tick),
                    SparseMembershipLinks::UNLINKED,
                );
                self.sparse[index] = dense_pos;
                entry.inserted_dense_pos = Some(dense_pos);
                inserted += 1;
                continue;
            }

            let dense_pos = dense_pos as usize;
            assert_eq!(
                self.dense.entity_at(dense_pos),
                Some(entry.entity),
                "sparse index points to another entity generation"
            );

            let moved: C = unsafe { ptr::read(entry.item.ptr as *mut C) };
            unsafe {
                entry.item.mark_consumed();
            }
            *self.dense.values[dense_pos].get_mut() = moved;
            self.dense.auxiliary[dense_pos]
                .ticks
                .get_mut()
                .mark_changed(change_tick);
        }

        if let Some(driver_index) = self.driver_index.get_mut() {
            driver_index.insert_dense_batch(
                first_dense_pos,
                inserted,
                entries.iter().filter_map(|entry| {
                    entry
                        .inserted_dense_pos
                        .map(|_| (entry.entity, entry.location))
                }),
            );
        } else {
            self.dormant_locations_mut().extend(
                entries
                    .iter()
                    .filter(|entry| entry.inserted_dense_pos.is_some())
                    .map(|entry| entry.location),
            );
        }
    }

    #[inline]
    fn remove_with_links(&mut self, entity: Entity) -> Option<SparseRemoveOutcome> {
        let dense_pos = self.dense_pos_of(entity)?;
        let (_, links) = self.remove_dense_pos(dense_pos as usize)?;

        Some(SparseRemoveOutcome { links })
    }

    #[inline]
    fn driver_buckets(&self) -> &[SparseDriverBucket] {
        ComponentSparseSet::driver_buckets(self)
    }

    #[inline]
    fn driver_revision(&self) -> u64 {
        ComponentSparseSet::driver_revision(self)
    }

    #[inline]
    fn update_driver_location_and_next(
        &mut self,
        entity: Entity,
        location: EntityLocation,
    ) -> Option<u32> {
        ComponentSparseSet::update_driver_location_and_next(self, entity, location)
    }

    fn update_locations_batch(&mut self, updates: &[SparseLocationUpdate]) {
        ComponentSparseSet::update_locations_batch(self, updates)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn membership_links(&self, entity: Entity) -> Option<SparseMembershipLinks> {
        ComponentSparseSet::membership_links(self, entity)
    }

    fn set_membership_links_dense(&mut self, dense_pos: u32, prev: u32, next: u32) -> bool {
        ComponentSparseSet::set_membership_links_dense(self, dense_pos, prev, next)
    }

    fn set_membership_prev(&mut self, entity: Entity, prev: u32) -> bool {
        ComponentSparseSet::set_membership_prev(self, entity, prev)
    }

    fn set_membership_next(&mut self, entity: Entity, next: u32) -> bool {
        ComponentSparseSet::set_membership_next(self, entity, next)
    }

    fn prepare_batch_remove(&self, entity: Entity) -> Option<super::SparseBatchProbe> {
        ComponentSparseSet::prepare_batch_remove(self, entity)
    }

    fn remove_batch_prepared(&mut self, removals: &mut [super::SparseBatchRemoval]) {
        ComponentSparseSet::remove_batch_prepared(self, removals)
    }

    fn clear_all_entries(&mut self) {
        ComponentSparseSet::clear_all_entries(self)
    }
}
