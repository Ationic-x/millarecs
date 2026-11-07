//! Minimal parent-child hierarchy built on regular components.
//!
//! [`Parent`] and [`Children`] are ordinary table components: no engine
//! machinery is involved in storing them. The [`crate::world::World`]
//! hierarchy methods keep both sides consistent, and scalar
//! [`crate::world::World::despawn`] detaches dying entities so surviving
//! relatives never observe stale handles.

// TODO: batch the cleanup; a whole subtree dying still walks and
// detaches one entity at a time.

use millarecs_macros::Component;

use crate::entity::Entity;

/// Marks the parent of an entity. Added and maintained by
/// [`crate::world::World::add_child`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Component)]
pub struct Parent(pub Entity);

/// Lists the live children of an entity. Maintained by
/// [`crate::world::World::add_child`] and [`crate::world::World::despawn`].
#[derive(Debug, Default, Component)]
pub struct Children(pub Vec<Entity>);

impl Children {
    /// Returns the live children of the entity.
    pub fn get(&self) -> &[Entity] {
        &self.0
    }
}
