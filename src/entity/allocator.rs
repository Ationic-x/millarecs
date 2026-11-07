//! Thread-safe reservation allocator for entity handles.
//!
//! Reservations can be made while systems are running. Actual metadata
//! activation is delayed until an exclusive world phase claims the reserved
//! handles and assigns storage locations.

use std::{
    num::{NonZero, NonZeroU32},
    sync::{Mutex, MutexGuard},
};

use super::{Entity, EntityError};

/// Generation stored in the pending table for slots still awaiting
/// materialization; zero is unreachable for live handles.
const NO_PENDING_GENERATION: u32 = 0;

/// Dense generation table for reservations not yet materialized in entity
/// metadata. Entity indices are allocated densely, so this avoids one tree
/// node allocation and an O(log N) lookup for every deferred entity.
#[derive(Default)]
struct PendingReservations {
    generations: Vec<u32>,
    len: usize,
}

impl PendingReservations {
    /// Records `entity` as pending, growing the table with sentinel slots as
    /// needed. Returns `false` without storing anything when the index already
    /// has a pending reservation.
    #[inline]
    fn insert(&mut self, entity: Entity) -> bool {
        let index = entity.index() as usize;

        if self.generations.len() <= index {
            self.generations.resize(index + 1, NO_PENDING_GENERATION);
        }

        let generation = &mut self.generations[index];
        if *generation != NO_PENDING_GENERATION {
            return false;
        }

        *generation = entity.generation();
        self.len += 1;
        true
    }

    /// Returns whether the exact handle is pending; a stale generation on the
    /// same index is not.
    #[inline]
    fn contains(&self, entity: &Entity) -> bool {
        self.generations
            .get(entity.index() as usize)
            .is_some_and(|&generation| generation == entity.generation())
    }

    /// Clears the pending entry matching `entity`'s index and generation,
    /// resetting its slot to the sentinel. Returns `false` when no matching
    /// entry exists.
    #[inline]
    fn remove(&mut self, entity: &Entity) -> bool {
        let Some(generation) = self.generations.get_mut(entity.index() as usize) else {
            return false;
        };

        if *generation != entity.generation() {
            return false;
        }

        *generation = NO_PENDING_GENERATION;
        self.len -= 1;
        true
    }

    /// Returns the number of pending reservations.
    #[inline]
    const fn len(&self) -> usize {
        self.len
    }

    #[cfg(test)]
    fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.generations
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(index, generation)| {
                NonZeroU32::new(generation)
                    .map(|generation| Entity::from_index_and_generation(index as u32, generation))
            })
    }
}

/// Mutex-guarded allocator state: the fresh-index cursor, recycled handles,
/// and pending reservations.
#[derive(Default)]
struct AllocationState {
    /// Index handed out next when the recycled list is empty.
    next_index: u32,
    /// Recycled handles whose generation already advanced, reused LIFO before
    /// new indices are minted.
    free: Vec<Entity>,
    pending: PendingReservations,
}

/// Thread-safe allocator for entity identifiers.
///
/// Reserving an identifier does not touch entity metadata or component
/// storage. The exclusive world phase later claims the reservation and
/// materializes its metadata before attaching components.
#[derive(Default)]
pub(crate) struct EntityAllocator {
    state: Mutex<AllocationState>,
}

impl EntityAllocator {
    /// Creates an empty allocator.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Locks and returns the allocator state, recovering from a poisoned mutex.
    fn state(&self) -> MutexGuard<'_, AllocationState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Returns the allocator state without locking; `&mut self` guarantees
    /// exclusive access.
    #[inline]
    fn state_mut(&mut self) -> &mut AllocationState {
        match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    /// Returns the same entity index with a fresh generation.
    fn next_generation(entity: Entity) -> Entity {
        let generation =
            NonZeroU32::new(entity.generation().wrapping_add(1)).unwrap_or(NonZero::<u32>::MIN);

        Entity::from_index_and_generation(entity.index(), generation)
    }

    /// Pops a recycled handle or mints one from the next unused index.
    ///
    /// Panics when the u32 index space is exhausted.
    fn allocate_from(state: &mut AllocationState) -> Entity {
        match state.free.pop() {
            Some(entity) => entity,
            None => {
                let index = state.next_index;

                state.next_index = state.next_index.checked_add(1).expect("Too many entities");

                Entity::from_index(index)
            }
        }
    }

    /// Allocates one entity while the caller has exclusive access.
    ///
    /// Unlike `reserve`, this does not enter the pending-reservation set.
    /// The caller must immediately materialize or otherwise account for
    /// the returned handle before releasing exclusive access.
    #[inline]
    pub(crate) fn allocate_exclusive(&mut self) -> Entity {
        let state = self.state_mut();

        Self::allocate_from(state)
    }

    /// Allocates several entities while borrowing allocator state once.
    ///
    /// The returned handles are unpublished: the exclusive caller must either
    /// commit all of them to metadata or release all of them before returning.
    pub(crate) fn allocate_exclusive_many(&mut self, amount: u32) -> Vec<Entity> {
        let state = self.state_mut();
        let mut entities = Vec::with_capacity(amount as usize);

        for _ in 0..amount {
            entities.push(Self::allocate_from(state));
        }

        entities
    }

    /// Reserves one entity from free storage or from the next unused index.
    fn reserve_from(state: &mut AllocationState) -> Entity {
        let entity = Self::allocate_from(state);

        let inserted = state.pending.insert(entity);

        debug_assert!(
            inserted,
            "entity allocator returned a duplicate reservation"
        );

        entity
    }

    /// Reserves one entity handle.
    pub(crate) fn reserve(&self) -> Entity {
        Self::reserve_from(&mut self.state())
    }

    /// Reserves `amount` entity handles while acquiring the lock once.
    pub(crate) fn reserve_many(&self, amount: u32) -> Vec<Entity> {
        let mut state = self.state();
        let mut entities = Vec::with_capacity(amount as usize);

        for _ in 0..amount {
            entities.push(Self::reserve_from(&mut state));
        }

        entities
    }

    /// Claims a pending reservation so metadata can be materialized.
    pub(crate) fn claim(&self, entity: Entity) -> Result<(), EntityError> {
        if self.state().pending.remove(&entity) {
            Ok(())
        } else {
            Err(EntityError::NotReserved(entity))
        }
    }

    /// Claims a batch produced by one allocator reservation.
    ///
    /// The allocator guarantees that the input is unique. Validation remains a
    /// separate pass so a missing or stale reservation cannot partially claim
    /// the batch.
    pub(crate) fn claim_many_unique(&self, entities: &[Entity]) -> Result<(), EntityError> {
        let mut state = self.state();
        for &entity in entities {
            if !state.pending.contains(&entity) {
                return Err(EntityError::NotReserved(entity));
            }
        }

        for entity in entities {
            let removed = state.pending.remove(entity);
            debug_assert!(removed, "allocator reservation batches must be unique");
        }

        Ok(())
    }

    /// Cancels one pending reservation and recycles its index with a new generation.
    pub(crate) fn cancel(&self, entity: Entity) -> Result<(), EntityError> {
        let mut state = self.state();

        if !state.pending.remove(&entity) {
            return Err(EntityError::NotReserved(entity));
        }

        state.free.push(Self::next_generation(entity));
        Ok(())
    }

    /// Cancels a unique batch produced by one allocator reservation.
    pub(crate) fn cancel_many_unique(&self, entities: &[Entity]) -> Result<(), EntityError> {
        let mut state = self.state();
        for &entity in entities {
            if !state.pending.contains(&entity) {
                return Err(EntityError::NotReserved(entity));
            }
        }

        for &entity in entities {
            let removed = state.pending.remove(&entity);
            debug_assert!(removed, "allocator reservation batches must be unique");
            state.free.push(Self::next_generation(entity));
        }

        Ok(())
    }

    /// Returns one unpublished exclusive allocation to the free list.
    ///
    /// The generation advances so the abandoned handle can never be
    /// revalidated later.
    pub(crate) fn release_unpublished_exclusive(&mut self, entity: Entity) {
        let recycled = Self::next_generation(entity);

        let state = self.state_mut();

        debug_assert!(!state.pending.contains(&entity));

        debug_assert!(!state.free.contains(&recycled));

        state.free.push(recycled);
    }

    /// Releases several unpublished exclusive allocations in one allocator
    /// access. Every generation advances so none of the abandoned handles can
    /// become valid later.
    pub(crate) fn release_unpublished_exclusive_many(&mut self, entities: &[Entity]) {
        let state = self.state_mut();

        state
            .free
            .extend(entities.iter().copied().map(Self::next_generation));
    }

    /// Recycles an entity handle whose metadata was already released.
    pub(crate) fn recycle(&self, entity: Entity) {
        let mut state = self.state();

        debug_assert!(!state.pending.contains(&entity));
        debug_assert!(!state.free.contains(&entity));

        state.free.push(entity);
    }

    /// Recycles a handle whose metadata was already released, without locking.
    #[inline]
    pub(crate) fn recycle_exclusive(&mut self, entity: Entity) {
        let state = self.state_mut();

        debug_assert!(!state.pending.contains(&entity));
        debug_assert!(!state.free.contains(&entity));

        state.free.push(entity);
    }

    /// Recycles prevalidated handles while borrowing allocator state once.
    #[inline]
    pub(crate) fn recycle_exclusive_many<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
    {
        let state = self.state_mut();
        state.free.extend(entities);
    }

    /// Returns which entities are still pending while acquiring the allocator
    /// lock only once for the complete batch.
    pub(crate) fn pending_mask(&self, entities: &[Entity]) -> Vec<bool> {
        let state = self.state();

        entities
            .iter()
            .map(|entity| state.pending.contains(entity))
            .collect()
    }

    /// Returns whether `entity` is currently reserved but not materialized.
    pub(crate) fn is_pending(&self, entity: Entity) -> bool {
        self.state().pending.contains(&entity)
    }

    /// Returns the number of pending reservations.
    pub(crate) fn pending_len(&self) -> usize {
        self.state().pending.len()
    }

    #[cfg(test)]
    /// Returns the next unused index for tests.
    pub(crate) fn next_index(&self) -> u32 {
        self.state().next_index
    }

    #[cfg(test)]
    /// Returns recycled handles for tests.
    pub(crate) fn free_entities(&self) -> Vec<Entity> {
        self.state().free.clone()
    }

    #[cfg(test)]
    /// Returns pending reservations for tests.
    pub(crate) fn pending_entities(&self) -> Vec<Entity> {
        self.state().pending.entities().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeSet, sync::Arc, thread};

    use super::EntityAllocator;

    #[test]
    fn concurrent_reservations_are_unique() {
        let allocator = Arc::new(EntityAllocator::new());
        let mut workers = Vec::new();

        for _ in 0..4 {
            let allocator = Arc::clone(&allocator);
            workers.push(thread::spawn(move || allocator.reserve_many(256)));
        }

        let entities: Vec<_> = workers
            .into_iter()
            .flat_map(|worker| worker.join().unwrap())
            .collect();
        let unique: BTreeSet<_> = entities.iter().copied().collect();

        assert_eq!(unique.len(), entities.len());
        assert_eq!(allocator.pending_len(), entities.len());
    }

    #[test]
    fn cancelling_a_reservation_invalidates_its_generation() {
        let allocator = EntityAllocator::new();
        let first = allocator.reserve();

        allocator.cancel(first).unwrap();
        let reused = allocator.reserve();

        assert_eq!(reused.index(), first.index());
        assert_ne!(reused.generation(), first.generation());
    }

    #[test]
    fn claiming_removes_only_the_selected_pending_reservation() {
        let allocator = EntityAllocator::new();
        let entities = allocator.reserve_many(2);

        allocator.claim(entities[0]).unwrap();

        assert!(!allocator.is_pending(entities[0]));
        assert!(allocator.is_pending(entities[1]));
        assert_eq!(allocator.pending_len(), 1);
    }
}
