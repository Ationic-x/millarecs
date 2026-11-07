//! Entity identifiers and allocation state.
//!
//! `EntityManager` separates reservation from activation so systems such as a
//! command buffer can hand out stable entity handles before their storage
//! location is known. Each slot records whether it is free, reserved, or alive.

pub(crate) mod allocator;
mod manager;
pub(crate) mod metadata;
mod paging;

pub(crate) use allocator::EntityAllocator;
pub(crate) use manager::EntityManager;
pub use metadata::{ArchetypeStorageKind, EntityLocation};

use std::{fmt, num::NonZero};

/// Errors produced by entity lifecycle and hierarchy operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityError {
    /// The handle does not resolve to a live or reserved entity.
    NotFound(Entity),
    /// The handle is alive but was never reserved.
    NotReserved(Entity),
    /// The handle is already alive and cannot be activated again.
    AlreadyAlive(Entity),
    /// The operation requires a live entity and the handle is not one.
    NotAlive(Entity),
    /// The slot was freed and reused after this handle was created.
    Stale(Entity),
    /// The handle is structurally invalid (e.g. the sentinel).
    Invalid(Entity),
    /// The same handle appears more than once in a batch input.
    Duplicate(Entity),
    /// The entity location is not a valid storage position.
    InvalidLocation(EntityLocation),
}

impl std::fmt::Display for EntityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(entity) => write!(f, "Entity not found: {entity:?}"),
            Self::NotReserved(entity) => write!(f, "Entity not reserved: {entity:?}"),
            Self::AlreadyAlive(entity) => write!(f, "Entity already alive: {entity:?}"),
            Self::NotAlive(entity) => write!(f, "Entity not alive: {entity:?}"),
            Self::Stale(entity) => write!(f, "Stale entity handle: {entity:?}"),
            Self::Invalid(entity) => write!(f, "Invalid entity: {entity:?}"),
            Self::Duplicate(entity) => write!(f, "Duplicated entity: {entity:?}"),
            Self::InvalidLocation(location) => {
                write!(f, "Invalid entity location: {location:?}")
            }
        }
    }
}
impl std::error::Error for EntityError {}

/// Opaque entity handle made of a sparse index and a generation.
///
/// The index addresses a slot in the metadata pages. The generation is compared
/// with the slot generation to reject handles that were created before the slot
/// was freed and reused.
#[derive(Clone, Copy)]
#[repr(C, align(8))]
pub struct Entity {
    index: u32,
    generation: NonZero<u32>,
}

impl Entity {
    /// Sentinel used when an API needs to represent the absence of an entity.
    pub const INVALID: Self = Self::from_index(u32::MAX);

    /// Creates an entity handle for `index` using the initial generation.
    #[inline(always)]
    pub const fn from_index(index: u32) -> Entity {
        Self::from_index_and_generation(index, NonZero::<u32>::MIN)
    }

    /// Creates an entity handle from an explicit index and generation.
    #[inline(always)]
    pub const fn from_index_and_generation(index: u32, generation: NonZero<u32>) -> Entity {
        Entity { index, generation }
    }

    /// Returns the sparse slot index stored by this handle.
    #[inline]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// Returns `false` only for the reserved invalid sentinel.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.index != u32::MAX
    }

    /// Returns the generation stored by this handle.
    #[inline]
    pub const fn generation(&self) -> u32 {
        self.generation.get()
    }
}

impl PartialEq for Entity {
    #[inline]
    fn eq(&self, other: &Entity) -> bool {
        // Bitwise and keeps the comparison branch-free.
        (self.index == other.index) & (self.generation == other.generation)
    }
}

impl Eq for Entity {}

impl PartialOrd for Entity {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entity {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.index
            .cmp(&other.index)
            .then(self.generation.cmp(&other.generation))
    }
}

impl fmt::Debug for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self == &Self::INVALID {
            write!(f, "Entity(INVALID)")
        } else {
            write!(f, "Entity({}:{})", self.index(), self.generation())
        }
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Entity(id: {}, gen: {}, valid: {})",
            self.index(),
            self.generation(),
            self != &Self::INVALID
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        component::archetype::ArchetypeId,
        entity::{
            metadata::{ArchetypeStorageKind, EntityLocation},
            EntityError, EntityManager,
        },
    };

    #[test]
    fn reserve_marks_entity_as_reserved_not_alive() {
        let mut entities = EntityManager::new();

        let entity = entities.reserve();

        assert!(entities.exists(entity));
        assert!(entities.is_reserved(entity));
        assert!(!entities.contains(entity));
        assert_eq!(entities.alive_len(), 0);
        assert_eq!(entities.reserved_len(), 1);
    }
    #[test]
    fn activate_moves_reserved_entity_to_alive() {
        let mut entities = EntityManager::new();
        let entity = entities.reserve();
        let location = location(1, 7);

        entities.activate(entity, location).unwrap();

        assert!(entities.contains(entity));
        assert!(!entities.is_reserved(entity));
        assert_eq!(entities.location(entity), Some(location));
        assert_eq!(entities.alive_len(), 1);
        assert_eq!(entities.reserved_len(), 0);
    }
    #[test]
    fn free_alive_invalidates_old_handle() {
        let mut entities = EntityManager::new();
        let entity = entities.reserve();

        entities.activate(entity, location(1, 0)).unwrap();
        entities.free_alive(entity).unwrap();

        assert!(!entities.exists(entity));
        assert!(!entities.contains(entity));

        let reused = entities.reserve();

        assert_eq!(reused.index(), entity.index());
        assert_ne!(reused.generation(), entity.generation());
    }

    fn location(id: usize, row: u32) -> EntityLocation {
        EntityLocation {
            archetype_id: ArchetypeId::new(id),
            storage: ArchetypeStorageKind::Contiguous,
            row,
        }
    }

    #[test]
    fn reserve_many_crosses_page_boundaries() {
        use crate::entity::paging::PAGE_SIZE;

        let mut entities = EntityManager::new();
        let batch = entities.reserve_many(PAGE_SIZE as u32 + 3);

        assert_eq!(batch.len(), PAGE_SIZE + 3);
        assert_eq!(entities.reserved_len(), PAGE_SIZE + 3);
        assert_eq!(entities.alive_len(), 0);

        for entity in batch {
            assert!(entities.is_reserved(entity));
            assert!(!entities.is_alive(entity));
        }

        entities.assert_invariants();
    }

    #[test]
    fn shared_reservations_do_not_materialize_metadata_until_claimed() {
        use crate::entity::paging::PAGE_SIZE;

        let mut entities = EntityManager::new();
        let initial_pages = entities.active_pages();
        let reserved = entities.allocator().reserve_many(PAGE_SIZE as u32 + 3);

        assert_eq!(entities.active_pages(), initial_pages);
        assert_eq!(entities.reserved_len(), PAGE_SIZE + 3);
        assert!(reserved
            .iter()
            .copied()
            .all(|entity| entities.is_reserved(entity)));

        entities.ensure_reserved_many(&reserved).unwrap();

        assert!(entities.active_pages() > initial_pages);
        assert_eq!(entities.reserved_len(), PAGE_SIZE + 3);
        entities.assert_invariants();
    }

    #[test]
    fn reserve_many_reuses_freed_indices_with_new_generation() {
        let mut entities = EntityManager::new();

        let first = entities.reserve_many(4);
        for (row, entity) in first.iter().copied().enumerate() {
            entities.activate(entity, location(1, row as u32)).unwrap();
        }

        entities.free_alive(first[1]).unwrap();
        entities.free_alive(first[3]).unwrap();

        let reused = entities.reserve_many(2);

        assert_eq!(reused[0].index(), first[3].index());
        assert_eq!(reused[1].index(), first[1].index());
        assert_ne!(reused[0].generation(), first[3].generation());
        assert_ne!(reused[1].generation(), first[1].generation());

        entities.assert_invariants();
    }

    #[test]
    fn activate_many_rejects_duplicates_without_changing_reservations() {
        let mut entities = EntityManager::new();
        let batch = entities.reserve_many(2);

        let result =
            entities.activate_many([(batch[0], location(1, 0)), (batch[0], location(1, 1))]);

        assert!(matches!(result, Err(EntityError::Duplicate(entity)) if entity == batch[0]));
        assert!(entities.is_reserved(batch[0]));
        assert!(entities.is_reserved(batch[1]));
        assert_eq!(entities.alive_len(), 0);
        assert_eq!(entities.reserved_len(), 2);
        entities.assert_invariants();
    }

    #[test]
    fn empty_distinguishes_reservations_from_live_entities() {
        let mut entities = EntityManager::new();
        let entity = entities.reserve();

        assert!(!entities.is_empty());
        assert!(entities.has_no_alive());

        entities.discard_reserved(entity).unwrap();

        assert!(entities.is_empty());
        assert!(entities.has_no_alive());
        entities.assert_invariants();
    }
}
