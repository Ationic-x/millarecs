//! Query building blocks and access validation.
//!
//! This module owns the low-level query model used by `World` and by systems.
//! A query is split into data (`QueryData`) and filters (`QueryFilter`), then
//! cached in `QueryState` so iteration does not have to recompute component ids
//! or archetype matches every frame.

pub(crate) mod access;
pub(crate) mod cost;
pub mod debug;
pub(crate) mod execution;
pub(crate) mod fetch;
pub mod filter;
pub(crate) mod iter;
pub mod planning;
pub(crate) mod state;
pub(crate) mod world_query;

use std::any::TypeId;

use crate::{
    component::{archetype::ArchetypeError, ComponentId},
    world::WorldId,
};

pub use access::{FilteredAccessSet, QueryAccess};
pub use debug::{
    QueryAccessDebug, QueryArchetypeDebug, QueryDebugComposition, QueryDebugItem, QueryDebugSource,
    QueryDriverCandidateDebug, QueryPlanDebug,
};
pub use fetch::{QueryData, ReadQueryData, WriteQueryData};
pub use iter::{QueryIter, QueryIterMut};
pub use planning::{
    QueryRoute, QueryRouteCandidate, QueryRoutePolicy, RouteCandidateRole, SparseRouteReason,
};
pub use state::{
    AutoQueryPolicy, AutoQueryState, CachedQueryState, QueryCacheDebug, QueryCacheKind, QueryState,
    UncachedQueryState,
};
pub use world_query::{WorldQuery, WorldQueryMut, WorldQueryMutRef, WorldQueryRef};

pub use execution::parallel::{
    QueryParallelChoice, QueryParallelConfig, QueryParallelDecision, QueryParallelMode,
    QueryParallelReason,
};

pub use execution::debug::{
    QueryExecutionDebug, QueryExecutionPath, QueryExecutionRequest, QueryParallelConfigDebug,
    QueryWorkDebug,
};
pub use execution::work::QueryWorkMode;

/// Error returned when building or executing a query.
#[derive(Debug)]
pub enum QueryError {
    /// The requested component type has not been registered.
    ComponentNotRegistered(TypeId),

    /// The component exists as an id, but its storage kind is unknown.
    StorageUnknown(ComponentId),

    /// A forced sparse route does not belong to this query's valid drivers.
    InvalidSparseRouteDriver(ComponentId),

    /// The query requests incompatible access to the same component.
    ///
    /// For example, `(&mut T, &T)` or `(&mut T, &mut T)`.
    AccessConflict {
        /// The component whose declared accesses collide.
        component: ComponentId,
        /// Access already held by the query.
        current: AccessMask,
        /// Access the new parameter tried to add.
        requested: AccessMask,
    },

    /// Error propagated from the archetype layer.
    Archetype(ArchetypeError),

    /// A state or system bound to one world was used against another.
    WorldMismatch {
        /// Identifier of the world the item was created with.
        expected: WorldId,
        /// Identifier of the world it was used against.
        actual: WorldId,
    },
}

impl From<ArchetypeError> for QueryError {
    fn from(value: ArchetypeError) -> Self {
        Self::Archetype(value)
    }
}

/// Independently tracked region of a component.
///
/// Each category occupies two adjacent bits in [`AccessMask`]: the even bit is
/// shared access and the odd bit is exclusive access. New ordinary categories
/// can therefore be added without changing the conflict algorithm.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessCategory {
    /// The stored component values.
    Value = 0,
    /// The per-row [`ComponentTicks`](crate::change_detection::ComponentTicks).
    Ticks = 1,
    /// The existence and position of the component (add/remove/move).
    Structure = 2,
}

/// Compact description of every access requested for one component.
#[repr(transparent)]
#[derive(Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AccessMask(u64);

impl AccessMask {
    /// Bit pattern selecting every shared-access bit (even positions).
    const READ_BITS: u64 = 0x5555_5555_5555_5555;
    /// Bit pattern selecting every exclusive-access bit (odd positions).
    const WRITE_BITS: u64 = 0xAAAA_AAAA_AAAA_AAAA;

    /// No access declared.
    pub const NONE: Self = Self(0);
    /// Shared read of the component values.
    pub const VALUE_READ: Self = Self::read(AccessCategory::Value);
    /// Exclusive write of the component values.
    pub const VALUE_WRITE: Self = Self::write(AccessCategory::Value);
    /// Shared read of the change ticks.
    pub const TICKS_READ: Self = Self::read(AccessCategory::Ticks);
    /// Exclusive write of the change ticks.
    pub const TICKS_WRITE: Self = Self::write(AccessCategory::Ticks);
    /// Shared read of component presence or position.
    pub const STRUCTURE_READ: Self = Self::read(AccessCategory::Structure);
    /// Exclusive write of component presence or position.
    pub const STRUCTURE_WRITE: Self = Self::write(AccessCategory::Structure);

    /// Builds the shared-access bit of `category`.
    #[inline]
    pub const fn read(category: AccessCategory) -> Self {
        Self(1_u64 << (category as u32 * 2))
    }

    /// Builds the exclusive-access bit of `category`.
    #[inline]
    pub const fn write(category: AccessCategory) -> Self {
        Self(1_u64 << (category as u32 * 2 + 1))
    }

    /// The raw bit pattern behind the mask.
    #[inline]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// `true` when no access is declared.
    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether every bit of `other` is present in `self`.
    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Any bit shared between the two masks.
    #[inline]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Combines both masks.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether any shared-read bit is set.
    #[inline]
    pub const fn has_reads(self) -> bool {
        self.0 & Self::READ_BITS != 0
    }

    /// Returns whether any exclusive-write bit is set.
    #[inline]
    pub const fn has_writes(self) -> bool {
        self.0 & Self::WRITE_BITS != 0
    }

    /// Returns whether two masks cannot be active independently.
    ///
    /// All ordinary categories use the same rule: a write conflicts with any
    /// access in its category. Structural writes are stronger because moving a
    /// component can invalidate value and tick pointers for that component.
    #[inline]
    pub const fn conflicts_with(self, other: Self) -> bool {
        if self.contains(Self::STRUCTURE_WRITE) && !other.is_empty()
            || other.contains(Self::STRUCTURE_WRITE) && !self.is_empty()
        {
            return true;
        }

        let self_writes = (self.0 & Self::WRITE_BITS) >> 1;
        let other_writes = (other.0 & Self::WRITE_BITS) >> 1;
        let self_accesses = (self.0 | (self.0 >> 1)) & Self::READ_BITS;
        let other_accesses = (other.0 | (other.0 >> 1)) & Self::READ_BITS;

        self_writes & other_accesses != 0 || other_writes & self_accesses != 0
    }
}

impl std::fmt::Debug for AccessMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccessMask")
            .field("bits", &format_args!("{:#018b}", self.0))
            .field("value_read", &self.contains(Self::VALUE_READ))
            .field("value_write", &self.contains(Self::VALUE_WRITE))
            .field("ticks_read", &self.contains(Self::TICKS_READ))
            .field("ticks_write", &self.contains(Self::TICKS_WRITE))
            .field("structure_read", &self.contains(Self::STRUCTURE_READ))
            .field("structure_write", &self.contains(Self::STRUCTURE_WRITE))
            .finish()
    }
}

impl std::ops::BitOr for AccessMask {
    type Output = Self;

    /// Unions both masks.
    #[inline]
    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for AccessMask {
    /// Unions `rhs` into this mask.
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// Access mask associated with one component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComponentAccess {
    component: ComponentId,
    mask: AccessMask,
}

impl ComponentAccess {
    /// Returns the accessed component.
    #[inline]
    pub const fn component(self) -> ComponentId {
        self.component
    }

    /// Returns the declared access categories.
    #[inline]
    pub const fn mask(self) -> AccessMask {
        self.mask
    }
}

/// Sorted component access declarations requested by a query or system.
///
/// Each component appears at most once and stores all of its access categories
/// in a single mask. Keeping the vector sorted makes conflict checks linear in
/// the number of accessed components and avoids hashing during scheduling.
#[derive(Debug, Default, Clone)]
pub struct AccessSet {
    accesses: Vec<ComponentAccess>,
}

impl AccessSet {
    /// Returns the index of `id`, or the sorted insertion position when absent.
    fn search(&self, id: ComponentId) -> Result<usize, usize> {
        self.accesses
            .binary_search_by_key(&id, |access| access.component)
    }

    /// Registers an access while enforcing independent aliasing rules.
    pub(crate) fn add(&mut self, id: ComponentId, requested: AccessMask) -> Result<(), QueryError> {
        debug_assert!(!requested.is_empty());

        match self.search(id) {
            Ok(index) => {
                let current = self.accesses[index].mask;

                if current.conflicts_with(requested) {
                    return Err(QueryError::AccessConflict {
                        component: id,
                        current,
                        requested,
                    });
                }

                self.accesses[index].mask |= requested;
            }
            Err(index) => self.accesses.insert(
                index,
                ComponentAccess {
                    component: id,
                    mask: requested,
                },
            ),
        }

        Ok(())
    }

    /// Combines the data and filter declarations of one query.
    ///
    /// Filters are evaluated by the same iterator before its data item is
    /// yielded, so a filter may read ticks that the yielded `Mut<T>` can later
    /// write. Both input sets have already validated their own internal aliases.
    pub(crate) fn compose_query(data: Self, filters: Self) -> Self {
        let mut accesses = Vec::with_capacity(data.len() + filters.len());
        let mut data_index = 0;
        let mut filter_index = 0;

        while data_index < data.accesses.len() && filter_index < filters.accesses.len() {
            let data_access = data.accesses[data_index];
            let filter_access = filters.accesses[filter_index];

            match data_access.component.cmp(&filter_access.component) {
                std::cmp::Ordering::Less => {
                    accesses.push(data_access);
                    data_index += 1;
                }
                std::cmp::Ordering::Greater => {
                    accesses.push(filter_access);
                    filter_index += 1;
                }
                std::cmp::Ordering::Equal => {
                    accesses.push(ComponentAccess {
                        component: data_access.component,
                        mask: data_access.mask | filter_access.mask,
                    });
                    data_index += 1;
                    filter_index += 1;
                }
            }
        }

        accesses.extend_from_slice(&data.accesses[data_index..]);
        accesses.extend_from_slice(&filters.accesses[filter_index..]);

        Self { accesses }
    }

    /// Adds every access from `other` without validating conflicts.
    ///
    /// This is used only for coarse, read-only summaries after each individual
    /// query has already been validated.
    pub(crate) fn union_from(&mut self, other: &Self) {
        let current = std::mem::take(self);
        *self = Self::compose_query(current, other.clone());
    }

    /// Returns the first conflicting component pair between two sets, if any.
    pub(crate) fn first_conflict(
        &self,
        other: &Self,
    ) -> Option<(ComponentAccess, ComponentAccess)> {
        let mut left = 0;
        let mut right = 0;

        while left < self.accesses.len() && right < other.accesses.len() {
            let current = self.accesses[left];
            let requested = other.accesses[right];

            match current.component.cmp(&requested.component) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => {
                    if current.mask.conflicts_with(requested.mask) {
                        return Some((current, requested));
                    }

                    left += 1;
                    right += 1;
                }
            }
        }

        None
    }

    /// Returns whether two independent access sets conflict.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.first_conflict(other).is_some()
    }

    /// Returns the declared access mask for `id`, if any.
    #[inline]
    pub fn get(&self, id: ComponentId) -> Option<AccessMask> {
        self.search(id).ok().map(|index| self.accesses[index].mask)
    }

    /// Returns the sorted component access declarations.
    #[inline]
    pub fn accesses(&self) -> &[ComponentAccess] {
        &self.accesses
    }

    /// Returns the number of components with declared access.
    #[inline]
    pub fn len(&self) -> usize {
        self.accesses.len()
    }

    /// Returns whether no component access is declared.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.accesses.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::AccessMask;

    #[test]
    fn shared_accesses_never_conflict() {
        let left = AccessMask::VALUE_READ | AccessMask::TICKS_READ;
        let right = AccessMask::VALUE_READ | AccessMask::TICKS_READ;

        assert!(!left.conflicts_with(right));
        assert!(!right.conflicts_with(left));
    }

    #[test]
    fn writes_only_conflict_inside_their_category() {
        assert!(AccessMask::VALUE_WRITE.conflicts_with(AccessMask::VALUE_READ));
        assert!(AccessMask::TICKS_WRITE.conflicts_with(AccessMask::TICKS_READ));
        assert!(!AccessMask::VALUE_WRITE.conflicts_with(AccessMask::TICKS_READ));
        assert!(!AccessMask::TICKS_WRITE.conflicts_with(AccessMask::STRUCTURE_READ));
    }

    #[test]
    fn structural_writes_invalidate_every_access_for_the_component() {
        assert!(AccessMask::STRUCTURE_WRITE.conflicts_with(AccessMask::VALUE_READ));
        assert!(AccessMask::STRUCTURE_WRITE.conflicts_with(AccessMask::TICKS_READ));
        assert!(AccessMask::VALUE_WRITE.conflicts_with(AccessMask::STRUCTURE_WRITE));
    }

    #[test]
    fn bit_pairs_are_stable_and_non_overlapping() {
        let all = AccessMask::VALUE_READ
            | AccessMask::VALUE_WRITE
            | AccessMask::TICKS_READ
            | AccessMask::TICKS_WRITE
            | AccessMask::STRUCTURE_READ
            | AccessMask::STRUCTURE_WRITE;

        assert_eq!(all.bits().count_ones(), 6);
    }
}
