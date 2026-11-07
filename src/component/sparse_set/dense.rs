//! Dense storage shared by typed sparse sets.
//!
//! Values, entity identities, and auxiliary metadata use parallel arrays so
//! ordinary entity validation only touches the compact identity stream. All
//! structural mutation is centralized here to preserve array alignment.

use std::cell::UnsafeCell;

use crate::{change_detection::ComponentTicks, entity::Entity};

use super::SparseMembershipLinks;

/// Metadata used by change detection and structural sparse-set operations.
#[repr(C)]
pub(super) struct SparseAuxiliaryMeta {
    pub(super) ticks: UnsafeCell<ComponentTicks>,
    pub(super) links: SparseMembershipLinks,
}

impl SparseAuxiliaryMeta {
    /// Builds one auxiliary entry from `ticks` and `links`.
    #[inline]
    const fn with_links(ticks: ComponentTicks, links: SparseMembershipLinks) -> Self {
        Self {
            ticks: UnsafeCell::new(ticks),
            links,
        }
    }
}

/// Data returned after compacting one dense sparse-set position.
pub(super) struct DenseRemoval<T> {
    /// The removed component value, dropped by the caller.
    pub(super) value: T,
    /// Entity the value belonged to.
    pub(super) entity: Entity,
    /// Membership links of the removed entry.
    pub(super) links: SparseMembershipLinks,
    /// Entity swapped into the removed position, if any.
    pub(super) moved_entity: Option<Entity>,
}

/// Parallel dense arrays kept aligned behind one mutation boundary.
pub(super) struct SparseDenseStorage<T> {
    /// Component value stored at each dense position.
    pub(super) values: Vec<UnsafeCell<T>>,
    /// Entity owning the entry at each dense position.
    pub(super) entities: Vec<Entity>,
    /// Change ticks and membership links of the entry at each dense position.
    pub(super) auxiliary: Vec<SparseAuxiliaryMeta>,
}

impl<T> Default for SparseDenseStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> SparseDenseStorage<T> {
    /// Creates storage with all three parallel arrays empty.
    #[inline]
    pub(super) const fn new() -> Self {
        Self {
            values: Vec::new(),
            entities: Vec::new(),
            auxiliary: Vec::new(),
        }
    }

    /// Returns the number of initialized dense positions.
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.values.len()
    }

    /// Drops every stored value while retaining the three allocations.
    #[inline]
    pub(super) fn clear(&mut self) {
        self.debug_assert_aligned();
        self.values.clear();
        self.entities.clear();
        self.auxiliary.clear();
        self.debug_assert_aligned();
    }

    /// Returns the entity stored at `dense_pos`, if the position is initialized.
    #[inline]
    pub(super) fn entity_at(&self, dense_pos: usize) -> Option<Entity> {
        self.entities.get(dense_pos).copied()
    }

    /// Returns the auxiliary metadata stored at `dense_pos`, if the position is
    /// initialized.
    #[inline]
    pub(super) fn auxiliary_at(&self, dense_pos: usize) -> Option<&SparseAuxiliaryMeta> {
        self.auxiliary.get(dense_pos)
    }

    /// Returns the auxiliary metadata stored at `dense_pos` mutably, if the
    /// position is initialized.
    #[inline]
    pub(super) fn auxiliary_at_mut(
        &mut self,
        dense_pos: usize,
    ) -> Option<&mut SparseAuxiliaryMeta> {
        self.auxiliary.get_mut(dense_pos)
    }

    /// Reserves matching capacity in every parallel dense array.
    pub(super) fn reserve(&mut self, additional: usize) {
        self.debug_assert_aligned();
        self.values.reserve(additional);
        self.entities.reserve(additional);
        self.auxiliary.reserve(additional);
    }

    /// Appends one scalar entry after reserving every parallel array together.
    ///
    /// Keeping capacity growth behind the same mutation boundary preserves the
    /// alignment invariant even when one of the allocations fails.
    #[inline]
    pub(super) fn push(&mut self, entity: Entity, value: T, ticks: ComponentTicks) -> u32 {
        self.reserve(1);
        self.push_reserved_linked(entity, value, ticks, SparseMembershipLinks::UNLINKED)
    }

    /// Appends into capacity reserved by a batch preparation step.
    #[inline]
    pub(super) fn push_reserved_linked(
        &mut self,
        entity: Entity,
        value: T,
        ticks: ComponentTicks,
        links: SparseMembershipLinks,
    ) -> u32 {
        self.debug_assert_aligned();
        assert!(
            self.len() < u32::MAX as usize,
            "sparse dense storage exhausted u32 positions"
        );
        debug_assert!(self.values.len() < self.values.capacity());
        debug_assert!(self.entities.len() < self.entities.capacity());
        debug_assert!(self.auxiliary.len() < self.auxiliary.capacity());

        let dense_pos = self.len() as u32;
        self.values.push(UnsafeCell::new(value));
        self.entities.push(entity);
        self.auxiliary
            .push(SparseAuxiliaryMeta::with_links(ticks, links));
        self.debug_assert_aligned();
        dense_pos
    }

    /// Removes one entry and compacts every parallel array identically.
    #[inline]
    pub(super) fn swap_remove(&mut self, dense_pos: usize) -> Option<DenseRemoval<T>> {
        self.debug_assert_aligned();
        if dense_pos >= self.len() {
            return None;
        }

        let last_dense_pos = self.len() - 1;
        let entity = self.entities[dense_pos];
        let links = self.auxiliary[dense_pos].links;
        let value = self.values.swap_remove(dense_pos).into_inner();
        self.entities.swap_remove(dense_pos);
        self.auxiliary.swap_remove(dense_pos);
        let moved_entity = (dense_pos != last_dense_pos).then(|| self.entities[dense_pos]);

        self.debug_assert_aligned();
        Some(DenseRemoval {
            value,
            entity,
            links,
            moved_entity,
        })
    }

    /// Asserts in debug builds that all three parallel arrays share one length.
    #[inline]
    fn debug_assert_aligned(&self) {
        debug_assert_eq!(self.values.len(), self.entities.len());
        debug_assert_eq!(self.values.len(), self.auxiliary.len());
    }
}
