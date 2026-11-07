//! Entity metadata manager.
//!
//! The manager owns paged metadata for alive and reserved entities, delegates
//! concurrent handle reservation to `EntityAllocator`, and keeps storage
//! locations in sync when archetypes move rows.

use std::num::NonZeroU32;

use crate::{
    entity::{
        metadata::{EntityLocation, EntityMeta, EntityState},
        paging::{index_in_page, page_of_entity, PageManager, VecPage},
        Entity, EntityAllocator, EntityError,
    },
    utils::ecs_vec::EcsVec,
};

/// Owns entity slot metadata and recycles freed indices.
///
/// The manager has two phases:
///
/// - reservation returns [`Entity`] values whose slots are marked as reserved;
/// - activation assigns a valid storage location and makes those handles alive.
///
/// Freed slots are reused before new indices are created. Reuse increments the
/// slot generation so stale entity handles stop matching metadata.
pub(crate) struct EntityManager {
    allocator: EntityAllocator,
    /// Active metadata pages indexed by entity page id.
    pages: VecPage,
    alive_len: usize,
    /// Materialized reservations only; `reserved_len` adds allocator-pending ones.
    reserved_len: usize,
    page_manager: PageManager,
}

// SAFETY:
// `EntityManager` owns raw pointers to metadata pages, but those pointers are
// stable heap allocations owned by `PageManager`. Shared access only reads
// metadata and delegates concurrent reservations to `EntityAllocator`, whose
// state is protected by a mutex. Any operation that mutates metadata, activates
// entities, frees slots, or grows pages requires `&mut EntityManager`.
unsafe impl Send for EntityManager {}

// SAFETY: see the `Send` implementation above. During system execution the
// world exposes only `&EntityManager`, so metadata pages are read-only while
// query workers resolve sparse-driver candidates back to entity locations.
unsafe impl Sync for EntityManager {}

impl Default for EntityManager {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityManager {
    /// Creates an empty entity manager with the first metadata page active.
    pub(crate) fn new() -> Self {
        let mut pages = EcsVec::default();
        let page_manager = PageManager::new(&mut pages);
        EntityManager {
            allocator: EntityAllocator::new(),
            pages,
            alive_len: 0,
            reserved_len: 0,
            page_manager,
        }
    }

    /// Returns the shared reservation allocator.
    pub(crate) fn allocator(&self) -> &EntityAllocator {
        &self.allocator
    }

    /// Allocates one entity for a direct spawn, marking its metadata page
    /// active. The slot stays free until `commit_spawn` publishes it.
    pub(crate) fn allocate_for_spawn(&mut self) -> Entity {
        let entity = self.allocator.allocate_exclusive();

        self.ensure_entity_range_active(entity.index(), entity.index() + 1);

        let metadata = self
            .get_metadata(entity.index())
            .expect("allocated entity page must be active");

        debug_assert_eq!(metadata.state, EntityState::Free);

        entity
    }

    /// Allocates a homogeneous direct-spawn batch without entering the shared
    /// pending-reservation set.
    pub(crate) fn allocate_many_for_spawn(&mut self, amount: u32) -> Vec<Entity> {
        let entities = self.allocator.allocate_exclusive_many(amount);

        let Some(max_index) = entities.iter().map(Entity::index).max() else {
            return entities;
        };

        self.ensure_entity_range_active(0, max_index + 1);

        debug_assert!(entities.iter().all(|entity| self
            .get_metadata(entity.index())
            .is_some_and(|metadata| metadata.state == EntityState::Free)));

        entities
    }

    /// Marks a direct-spawned entity as alive at `location`.
    pub(crate) fn commit_spawn(&mut self, entity: Entity, location: EntityLocation) {
        debug_assert!(entity.is_valid());
        debug_assert!(location.is_valid());

        let metadata = self
            .get_metadata_mut(entity.index())
            .expect("allocated entity metadata must exist");

        debug_assert_eq!(metadata.state, EntityState::Free);

        metadata.generation =
            NonZeroU32::new(entity.generation()).expect("entity generation must be non-zero");

        metadata.state = EntityState::Alive;
        metadata.location = location;

        self.alive_len += 1;
    }

    /// Publishes a preallocated direct-spawn batch without sorting or repeated
    /// entity validation. The allocator guarantees that all handles are unique.
    pub(crate) fn commit_spawn_many(&mut self, entries: &[(Entity, EntityLocation)]) {
        for &(entity, location) in entries {
            debug_assert!(entity.is_valid());
            debug_assert!(location.is_valid());

            let metadata = self
                .get_metadata_mut(entity.index())
                .expect("allocated entity metadata must exist");

            debug_assert_eq!(metadata.state, EntityState::Free);
            metadata.generation =
                NonZeroU32::new(entity.generation()).expect("entity generation must be non-zero");
            metadata.state = EntityState::Alive;
            metadata.location = location;
        }

        self.alive_len += entries.len();
    }

    /// Returns the logical state for a handle, including pending reservations.
    fn state_of(&self, entity: Entity) -> Option<EntityState> {
        let metadata_state = self
            .checked_metadata(entity)
            .ok()
            .map(|metadata| metadata.state);

        if matches!(
            metadata_state,
            Some(EntityState::Alive | EntityState::Reserved)
        ) {
            return metadata_state;
        }

        if self.allocator.is_pending(entity) {
            Some(EntityState::Reserved)
        } else {
            metadata_state
        }
    }

    #[cfg(test)]
    /// Whether the entity is alive or reserved.
    pub fn exists(&self, entity: Entity) -> bool {
        matches!(
            self.state_of(entity),
            Some(EntityState::Reserved | EntityState::Alive)
        )
    }

    /// `true` when the entity is reserved but not alive.
    pub fn is_reserved(&self, entity: Entity) -> bool {
        self.state_of(entity) == Some(EntityState::Reserved)
    }

    /// Whether `entity` is alive.
    #[cfg(test)]
    pub fn contains(&self, entity: Entity) -> bool {
        self.state_of(entity) == Some(EntityState::Alive)
    }

    /// Updates the storage location of a live entity.
    pub(crate) fn set_location(
        &mut self,
        entity: Entity,
        location: EntityLocation,
    ) -> Result<(), EntityError> {
        if !location.is_valid() {
            return Err(EntityError::InvalidLocation(location));
        }

        let metadata = self
            .alive_metadata_of_mut(entity)
            .ok_or(EntityError::NotAlive(entity))?;

        metadata.location = location;
        Ok(())
    }

    /// Returns metadata for an entity index if its page is active.
    #[inline]
    fn get_metadata(&self, entity_id: u32) -> Option<&EntityMeta> {
        let local_id = index_in_page(entity_id as usize);

        self.page_manager
            .get(&self.pages, entity_id)
            .and_then(|page| page.metalist().get(local_id))
    }

    /// Returns mutable metadata for an entity index if its page is active.
    #[inline]
    fn get_metadata_mut(&mut self, entity_id: u32) -> Option<&mut EntityMeta> {
        let local_id = index_in_page(entity_id as usize);

        self.page_manager
            .get_mut(&mut self.pages, entity_id)
            .and_then(|page| page.mut_metalist().get_mut(local_id))
    }

    /// Returns metadata when the handle generation matches.
    fn metadata_of(&self, entity: Entity) -> Option<&EntityMeta> {
        let metadata = self.get_metadata(entity.index())?;

        (metadata.generation.get() == entity.generation()).then_some(metadata)
    }

    /// Returns mutable metadata when the handle generation matches.
    fn metadata_of_mut(&mut self, entity: Entity) -> Option<&mut EntityMeta> {
        let metadata = self.get_metadata_mut(entity.index())?;

        (metadata.generation.get() == entity.generation()).then_some(metadata)
    }

    /// Returns metadata only for a live entity.
    fn alive_metadata_of(&self, entity: Entity) -> Option<&EntityMeta> {
        let metadata = self.metadata_of(entity)?;

        (metadata.state == EntityState::Alive).then_some(metadata)
    }

    /// Returns mutable metadata only for a live entity.
    fn alive_metadata_of_mut(&mut self, entity: Entity) -> Option<&mut EntityMeta> {
        let metadata = self.metadata_of_mut(entity)?;

        (metadata.state == EntityState::Alive).then_some(metadata)
    }

    /// Returns the current storage location for a live entity.
    pub(crate) fn location(&self, entity: Entity) -> Option<EntityLocation> {
        Some(self.alive_metadata_of(entity)?.location)
    }

    /// Returns the current storage location mutably for a live entity.
    pub(crate) fn location_mut(&mut self, entity: Entity) -> Option<&mut EntityLocation> {
        Some(&mut self.alive_metadata_of_mut(entity)?.location)
    }

    /// Returns metadata or a precise handle validation error.
    fn checked_metadata(&self, entity: Entity) -> Result<&EntityMeta, EntityError> {
        if !entity.is_valid() {
            return Err(EntityError::Invalid(entity));
        }

        let metadata = self
            .get_metadata(entity.index())
            .ok_or(EntityError::NotFound(entity))?;

        if metadata.generation.get() != entity.generation() {
            return Err(EntityError::Stale(entity));
        }

        Ok(metadata)
    }

    /// Returns mutable metadata or a precise handle validation error.
    fn checked_metadata_mut(&mut self, entity: Entity) -> Result<&mut EntityMeta, EntityError> {
        if !entity.is_valid() {
            return Err(EntityError::Invalid(entity));
        }

        let metadata = self
            .get_metadata_mut(entity.index())
            .ok_or(EntityError::NotFound(entity))?;

        if metadata.generation.get() != entity.generation() {
            return Err(EntityError::Stale(entity));
        }

        Ok(metadata)
    }

    /// Ensures the handle's metadata page is active and its slot is still free.
    ///
    /// Returns `Invalid` for the sentinel handle and `AlreadyAlive` when the
    /// slot is already reserved or alive.
    fn prepare_free_slot(&mut self, entity: Entity) -> Result<(), EntityError> {
        if !entity.is_valid() {
            return Err(EntityError::Invalid(entity));
        }

        self.ensure_entity_range_active(entity.index(), entity.index() + 1);

        let metadata = self
            .get_metadata(entity.index())
            .expect("entity metadata page must be active");

        if metadata.state != EntityState::Free {
            return Err(EntityError::AlreadyAlive(entity));
        }

        Ok(())
    }

    /// Marks a known-free slot as reserved with the handle's generation and
    /// bumps the reserved counter. The slot's page must already be active.
    #[inline]
    fn materialize_reserved_slot(&mut self, entity: Entity) {
        let metadata = self
            .get_metadata_mut(entity.index())
            .expect("entity metadata page must be active");

        metadata.generation =
            NonZeroU32::new(entity.generation()).expect("entity generation must be non-zero");

        metadata.state = EntityState::Reserved;
        metadata.location = EntityLocation::INVALID;

        self.reserved_len += 1;
    }

    /// Materializes an unpublished exclusive allocation as a reserved slot.
    fn materialize_exclusive(&mut self, entity: Entity) -> Result<(), EntityError> {
        self.prepare_free_slot(entity)?;

        self.materialize_reserved_slot(entity);

        Ok(())
    }

    /// Converts one allocator-pending reservation into metadata state.
    fn materialize_pending(&mut self, entity: Entity) -> Result<(), EntityError> {
        self.prepare_free_slot(entity)?;

        self.allocator.claim(entity)?;

        self.materialize_reserved_slot(entity);

        Ok(())
    }

    /// Converts several allocator-pending reservations into metadata state.
    fn materialize_pending_many(&mut self, entities: &[Entity]) -> Result<(), EntityError> {
        if entities.is_empty() {
            return Ok(());
        }

        for &entity in entities {
            if !entity.is_valid() {
                return Err(EntityError::Invalid(entity));
            }
        }

        let max_index = entities
            .iter()
            .map(Entity::index)
            .max()
            .expect("non-empty entity list");
        self.ensure_entity_range_active(0, max_index + 1);

        for &entity in entities {
            let metadata = self
                .get_metadata(entity.index())
                .expect("reserved entity metadata page must be active");

            if metadata.state != EntityState::Free {
                return Err(EntityError::AlreadyAlive(entity));
            }
        }

        self.allocator.claim_many_unique(entities)?;

        for &entity in entities {
            let metadata = self
                .get_metadata_mut(entity.index())
                .expect("reserved entity metadata page must be active");

            metadata.generation =
                NonZeroU32::new(entity.generation()).expect("entity generation must be non-zero");
            metadata.state = EntityState::Reserved;
            metadata.location = EntityLocation::INVALID;
        }

        self.reserved_len += entities.len();
        Ok(())
    }

    /// Ensures a reserved handle has materialized metadata.
    pub(crate) fn ensure_reserved(&mut self, entity: Entity) -> Result<(), EntityError> {
        if self
            .metadata_of(entity)
            .is_some_and(|metadata| metadata.state == EntityState::Reserved)
        {
            return Ok(());
        }

        self.materialize_pending(entity)
    }

    /// Ensures several reserved handles have materialized metadata.
    pub(crate) fn ensure_reserved_many(&mut self, entities: &[Entity]) -> Result<(), EntityError> {
        if entities.is_empty() {
            return Ok(());
        }

        let pending_mask = self.allocator.pending_mask(entities);
        let pending: Vec<Entity> = entities
            .iter()
            .copied()
            .zip(pending_mask.iter().copied())
            .filter_map(|(entity, is_pending)| is_pending.then_some(entity))
            .collect();

        for (&entity, is_pending) in entities.iter().zip(pending_mask) {
            if is_pending {
                continue;
            }

            let is_materialized = self
                .metadata_of(entity)
                .is_some_and(|metadata| metadata.state == EntityState::Reserved);

            if !is_materialized {
                return Err(EntityError::NotReserved(entity));
            }
        }

        self.materialize_pending_many(&pending)
    }

    /// Reserves and immediately materializes one entity.
    pub fn reserve(&mut self) -> Entity {
        let entity = self.allocator.allocate_exclusive();

        self.materialize_exclusive(entity)
            .expect("fresh exclusive entity allocation must materialize");

        entity
    }

    /// Ensures metadata pages exist for the half-open entity index range.
    fn ensure_entity_range_active(&mut self, start: u32, end: u32) {
        if start == end {
            return;
        }

        let amount = (end - start) as usize;

        self.page_manager
            .reserve_from_entities(&mut self.pages, amount, start as usize);

        let last_page = page_of_entity((end - 1) as usize);

        while self.active_pages() <= last_page {
            self.page_manager.pop(&mut self.pages);
        }
    }

    #[cfg(test)]
    /// Reserves many entities for tests.
    pub fn reserve_many(&mut self, amount: u32) -> Vec<Entity> {
        let entities = self.allocator.reserve_many(amount);
        self.materialize_pending_many(&entities)
            .expect("fresh entity reservations must materialize");
        entities
    }

    /// Turns a reserved entity into a live entity at `location`.
    pub fn activate(
        &mut self,
        entity: Entity,
        location: EntityLocation,
    ) -> Result<(), EntityError> {
        if !entity.is_valid() {
            return Err(EntityError::Invalid(entity));
        }

        if !location.is_valid() {
            return Err(EntityError::InvalidLocation(location));
        }

        let metadata = self.checked_metadata_mut(entity)?;

        match metadata.state {
            EntityState::Reserved => {
                metadata.state = EntityState::Alive;
                metadata.location = location;

                self.reserved_len -= 1;
                self.alive_len += 1;

                Ok(())
            }
            EntityState::Free => Err(EntityError::NotFound(entity)),
            EntityState::Alive => Err(EntityError::AlreadyAlive(entity)),
        }
    }

    /// Activates every reservation after validating the complete input.
    ///
    /// On error, no slot state or location is modified.
    pub fn activate_many<I>(&mut self, entries: I) -> Result<(), EntityError>
    where
        I: IntoIterator<Item = (Entity, EntityLocation)>,
    {
        let mut entries: Vec<(Entity, EntityLocation)> = entries.into_iter().collect();

        entries.sort_unstable_by_key(|entry| entry.0.index());

        for pair in entries.windows(2) {
            if pair[0].0.index() == pair[1].0.index() {
                return Err(EntityError::Duplicate(pair[1].0));
            }
        }

        for &(entity, location) in &entries {
            if !entity.is_valid() {
                return Err(EntityError::Invalid(entity));
            }

            if !location.is_valid() {
                return Err(EntityError::InvalidLocation(location));
            }

            let metadata = self
                .metadata_of(entity)
                .ok_or(EntityError::NotFound(entity))?;

            match metadata.state {
                EntityState::Reserved => (),
                EntityState::Free => return Err(EntityError::NotFound(entity)),
                EntityState::Alive => return Err(EntityError::AlreadyAlive(entity)),
            }
        }

        self.activate_many_prevalidated(&entries);
        Ok(())
    }

    /// Activates an already validated reservation batch without sorting it.
    ///
    /// The caller must guarantee unique entities, valid locations, and that
    /// every entity is currently reserved.
    pub(crate) fn activate_many_prevalidated(&mut self, entries: &[(Entity, EntityLocation)]) {
        for &(entity, location) in entries {
            let metadata = self
                .metadata_of_mut(entity)
                .expect("entity was validated before activation");

            debug_assert_eq!(metadata.state, EntityState::Reserved);
            metadata.state = EntityState::Alive;
            metadata.location = location;
        }

        self.reserved_len -= entries.len();
        self.alive_len += entries.len();
    }

    /// Clears metadata and returns the recycled handle for the allocator.
    fn recycle_entity_slot(index: u32, meta: &mut EntityMeta) -> Entity {
        meta.generation = meta.generation.checked_add(1).unwrap_or(NonZeroU32::MIN);

        meta.state = EntityState::Free;
        meta.location = EntityLocation::INVALID;

        Entity::from_index_and_generation(index, meta.generation)
    }

    /// Frees one live entity and returns its previous storage location.
    pub fn free_alive(&mut self, entity: Entity) -> Result<EntityLocation, EntityError> {
        if !entity.is_valid() {
            return Err(EntityError::Invalid(entity));
        }

        // One metadata access serves both validation and recycling; the page
        // is already warm after the caller's location read, so the removed
        // cost is the second page walk and generation compare.
        let metadata = self
            .get_metadata_mut(entity.index())
            .ok_or(EntityError::NotFound(entity))?;

        if metadata.generation.get() != entity.generation() {
            return Err(EntityError::Stale(entity));
        }

        match metadata.state {
            EntityState::Alive => {
                let old_location = metadata.location;
                let recycled = Self::recycle_entity_slot(entity.index(), metadata);

                self.alive_len -= 1;
                self.allocator.recycle_exclusive(recycled);
                Ok(old_location)
            }
            EntityState::Reserved => Err(EntityError::NotAlive(entity)),
            EntityState::Free => Err(EntityError::NotFound(entity)),
        }
    }

    /// Frees handles that were validated before structural batch mutation.
    pub(crate) fn free_alive_many_prevalidated(&mut self, entities: &[Entity]) {
        let mut recycled = Vec::with_capacity(entities.len());

        for &entity in entities {
            let metadata = self
                .metadata_of_mut(entity)
                .expect("prevalidated live entity metadata must exist");

            debug_assert_eq!(metadata.state, EntityState::Alive);
            recycled.push(Self::recycle_entity_slot(entity.index(), metadata));
        }

        self.alive_len -= entities.len();
        self.allocator.recycle_exclusive_many(recycled);
    }

    /// Discards one reservation and returns its slot to the free list.
    pub fn discard_reserved(&mut self, entity: Entity) -> Result<(), EntityError> {
        if self.allocator.is_pending(entity) {
            return self.allocator.cancel(entity);
        }

        let metadata = self
            .get_metadata_mut(entity.index())
            .ok_or(EntityError::NotFound(entity))?;

        if metadata.generation.get() != entity.generation() {
            return Err(EntityError::Stale(entity));
        }

        if metadata.state != EntityState::Reserved {
            return Err(EntityError::NotReserved(entity));
        }

        let recycled = Self::recycle_entity_slot(entity.index(), metadata);

        self.reserved_len -= 1;
        self.allocator.recycle(recycled);

        Ok(())
    }

    /// Returns a failed direct-spawn allocation to the allocator.
    pub(crate) fn abort_spawn(&mut self, entity: Entity) {
        debug_assert!(self
            .get_metadata(entity.index())
            .is_some_and(|meta| meta.state == EntityState::Free));

        self.allocator.release_unpublished_exclusive(entity);
    }

    /// Returns an unpublished direct-spawn batch to the allocator.
    pub(crate) fn abort_spawn_many(&mut self, entities: &[Entity]) {
        debug_assert!(entities.iter().all(|entity| self
            .get_metadata(entity.index())
            .is_some_and(|metadata| metadata.state == EntityState::Free)));

        self.allocator.release_unpublished_exclusive_many(entities);
    }

    #[inline]
    /// Returns the number of active metadata pages.
    pub(super) fn active_pages(&self) -> usize {
        self.pages.len()
    }

    /// Returns the entity-index capacity backed by materialized metadata.
    #[inline]
    pub(crate) fn materialized_slot_capacity(&self) -> usize {
        self.active_pages() * crate::entity::paging::PAGE_SIZE
    }

    /// Returns the number of live entities.
    pub fn alive_len(&self) -> usize {
        self.alive_len
    }

    /// Returns materialized and allocator-pending reservations.
    pub fn reserved_len(&self) -> usize {
        self.reserved_len + self.allocator.pending_len()
    }

    /// Whether no live or reserved entity exists.
    pub fn is_empty(&self) -> bool {
        self.alive_len == 0 && self.reserved_len() == 0
    }

    /// Whether no live entity exists, ignoring pending reservations.
    #[cfg(test)]
    pub fn has_no_alive(&self) -> bool {
        self.alive_len == 0
    }

    /// Returns the current logical state for `entity`.
    pub fn state(&self, entity: Entity) -> Option<EntityState> {
        self.state_of(entity)
    }

    /// Whether `entity` is alive.
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.state(entity) == Some(EntityState::Alive)
    }

    #[cfg(test)]
    /// Verifies allocator and metadata counters agree.
    pub(super) fn assert_invariants(&self) {
        use std::collections::HashSet;

        let mut reserved: usize = 0usize;
        let mut alive: usize = 0usize;
        let mut free_set = HashSet::new();
        let pending: HashSet<u32> = self
            .allocator
            .pending_entities()
            .into_iter()
            .map(|entity| entity.index())
            .collect();

        for entity in self.allocator.free_entities() {
            let index = entity.index();
            assert!(free_set.insert(index), "duplicated free index {index}");

            if let Some(meta) = self.get_metadata(index) {
                assert_eq!(meta.state, EntityState::Free);
                assert!(!meta.location.is_valid());
            }
        }

        for index in 0..self.allocator.next_index() {
            let Some(meta) = self.get_metadata(index) else {
                assert!(
                    free_set.contains(&index) || pending.contains(&index),
                    "issued entity index {index} has neither metadata nor allocator ownership"
                );
                continue;
            };

            match meta.state {
                EntityState::Free => {
                    assert!(
                        free_set.contains(&index) || pending.contains(&index),
                        "free entity index {index} is missing from the allocator"
                    );
                    assert!(!meta.location.is_valid());
                }
                EntityState::Reserved => {
                    reserved += 1;
                    assert!(!meta.location.is_valid());
                }
                EntityState::Alive => {
                    alive += 1;
                    assert!(meta.location.is_valid());
                }
            }
        }

        assert_eq!(self.reserved_len(), reserved + pending.len());
        assert_eq!(self.alive_len, alive);
    }
}
