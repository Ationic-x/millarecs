//! Metadata stored alongside each entity identifier.
//!
//! Entity metadata is intentionally small and cache-friendly: the generation
//! protects against stale handles, while the location points to the row that
//! owns the entity data inside the archetype storage.

use std::num::NonZero;

use crate::component::archetype::ArchetypeId;

/// Lifecycle state of one entity slot in the metadata pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum EntityState {
    /// Slot unused; no live or reserved handle refers to it.
    Free,
    /// Handle issued and materialized, but no storage location is assigned yet.
    Reserved,
    /// Entity attached to a valid archetype row.
    Alive,
}

/// Per-entity metadata kept in the paged sparse array.
#[repr(C)]
#[derive(Debug)]
pub(crate) struct EntityMeta {
    /// Current generation for this entity slot.
    ///
    /// A stale [`crate::entity::Entity`] is rejected when its stored generation
    /// no longer matches this value.
    pub(crate) generation: NonZero<u32>,
    /// Current storage position for the entity.
    pub(crate) location: EntityLocation,
    /// Lifecycle state of this slot.
    pub(crate) state: EntityState,
}

impl EntityMeta {
    /// Metadata value used for unoccupied slots and freshly initialized pages.
    pub const INVALID: EntityMeta = EntityMeta {
        generation: NonZero::<u32>::MIN,
        location: EntityLocation::INVALID,
        state: EntityState::Free,
    };
}

/// Storage lane that owns an entity's components.
///
/// Table components live in one of the two archetype lanes; sparse components
/// are not covered by this selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ArchetypeStorageKind {
    /// Dense lane: live rows form a prefix and removals backfill from the tail.
    Contiguous,
    /// Hole-tolerant lane: removals leave gaps tracked by page free lists.
    Fragmented,
}

/// Position of an entity inside an archetype table.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct EntityLocation {
    /// Archetype that owns the entity row.
    pub(crate) archetype_id: ArchetypeId,
    /// Lane of the archetype that owns the row.
    pub storage: ArchetypeStorageKind,
    /// Row inside the archetype table.
    ///
    /// `u32::MAX` is reserved as the invalid row sentinel.
    pub(crate) row: u32,
}

impl EntityLocation {
    /// Sentinel used when the entity is not attached to any archetype row.
    pub const INVALID: EntityLocation = EntityLocation {
        archetype_id: ArchetypeId::INVALID,
        storage: ArchetypeStorageKind::Contiguous,
        row: u32::MAX,
    };

    /// Returns `true` when both the archetype id and row are usable values.
    pub fn is_valid(&self) -> bool {
        self.archetype_id.is_valid() && self.row != u32::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_location_uses_sentinel_values() {
        assert_eq!(EntityLocation::INVALID.archetype_id, ArchetypeId::INVALID);
        assert_eq!(EntityLocation::INVALID.row, u32::MAX);
        assert!(!EntityLocation::INVALID.is_valid());
    }

    #[test]
    fn location_requires_valid_archetype_and_row() {
        let valid = EntityLocation {
            archetype_id: ArchetypeId::new(3),
            storage: ArchetypeStorageKind::Contiguous,
            row: 7,
        };

        let invalid_arch = EntityLocation {
            archetype_id: ArchetypeId::INVALID,
            storage: ArchetypeStorageKind::Contiguous,
            row: 7,
        };

        let invalid_row = EntityLocation {
            archetype_id: ArchetypeId::new(3),
            storage: ArchetypeStorageKind::Contiguous,
            row: u32::MAX,
        };

        assert!(valid.is_valid());
        assert!(!invalid_arch.is_valid());
        assert!(!invalid_row.is_valid());
    }

    #[test]
    fn invalid_metadata_starts_at_first_generation_without_a_location() {
        assert_eq!(EntityMeta::INVALID.generation, NonZero::<u32>::MIN);
        assert_eq!(EntityMeta::INVALID.location, EntityLocation::INVALID);
        assert!(!EntityMeta::INVALID.location.is_valid());
    }
}
