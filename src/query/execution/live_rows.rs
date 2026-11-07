//! Cursor over live rows inside prepared query work units.
//!
//! A unit can cover several pages in either contiguous or fragmented storage.
//! The cursor hides the page policy by yielding only live `(row, entity)` pairs
//! using the page batch live mask.

use crate::{
    component::archetype::{Archetype, ARCHETYPE_PAGE_ROWS},
    entity::{metadata::ArchetypeStorageKind, Entity},
};

/// Iterates live rows for one prepared work unit.
///
/// The cursor stores raw pointers so prepared read/write execution can avoid
/// borrowing the whole archetype for the lifetime of the iterator. Callers must
/// guarantee that the archetype and entity slice outlive the cursor and that no
/// structural mutation changes the targeted page range while it is being used.
///
/// Row production runs a three-level machine: a dense run of consecutive rows
/// for a fully live page, the page's live bit mask when only part of it is
/// live, and a page loader that advances one batch at a time. A page whose
/// mask is complete switches the machine to its dense run, skipping per-row
/// mask arithmetic until the run ends.
pub(crate) struct LiveRowCursor {
    archetype: *const Archetype,
    storage: ArchetypeStorageKind,
    entities: *const Entity,
    /// Entity count, used only for debug assertions.
    entity_len: usize,
    /// Past-the-last page of the range.
    page_end: usize,
    next_page: usize,
    /// Absolute row of bit zero in the current live mask.
    base_row: usize,
    /// Remaining live bits of the current page; zero while a dense run is
    /// active or no page is loaded.
    live_mask: u64,
    /// Next row of the active dense run.
    run_row: usize,
    /// Past-the-last row of the active dense run; an empty run has
    /// `run_row == run_end`.
    run_end: usize,
}

impl LiveRowCursor {
    /// Creates a cursor over an exclusive page range.
    pub(crate) fn new_range(
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        entities: *const Entity,
        entity_len: usize,
        page_start: usize,
        page_end: usize,
    ) -> Self {
        debug_assert!(page_start <= page_end);

        Self {
            archetype,
            storage,
            entities,
            entity_len,
            page_end,
            next_page: page_start,
            base_row: 0,
            live_mask: 0,
            run_row: 0,
            run_end: 0,
        }
    }

    /// Returns the next live row and its entity.
    ///
    /// # Safety
    ///
    /// The pointers supplied to `new_range` must still be valid, and the
    /// corresponding storage lane must not be structurally mutated while the
    /// cursor is active.
    pub(crate) unsafe fn next(&mut self) -> Option<(usize, Entity)> {
        loop {
            if self.run_row < self.run_end {
                let row = self.run_row;
                self.run_row += 1;
                debug_assert!(row < self.entity_len);

                let entity = unsafe { *self.entities.add(row) };
                return Some((row, entity));
            }

            if self.live_mask != 0 {
                let offset = self.live_mask.trailing_zeros() as usize;
                self.live_mask &= self.live_mask - 1;

                let row = self.base_row + offset;
                debug_assert!(row < self.entity_len);

                let entity = unsafe { *self.entities.add(row) };
                return Some((row, entity));
            }

            if self.next_page >= self.page_end {
                return None;
            }

            let archetype = unsafe { &*self.archetype };
            let Some(batch) = archetype.page_batch(self.storage, self.next_page) else {
                self.next_page += 1;
                continue;
            };

            self.next_page += 1;
            self.base_row = batch.base_row;
            let mask = row_count_mask(batch.row_count);
            self.live_mask = batch.live_mask & mask;

            if self.live_mask == mask && mask != 0 {
                // Full pages (the contiguous-storage common case) iterate as
                // one dense run, skipping per-row mask arithmetic.
                self.run_row = batch.base_row;
                self.run_end = batch.base_row + batch.row_count;
                self.live_mask = 0;
            }
        }
    }

    /// Returns the next live row without loading its entity handle.
    ///
    /// Mirror of [`Self::next`] for queries that never observe the entity;
    /// skipping the handle load removes one streamed access per row.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::next`].
    pub(crate) unsafe fn next_row(&mut self) -> Option<usize> {
        loop {
            if self.run_row < self.run_end {
                let row = self.run_row;
                self.run_row += 1;
                debug_assert!(row < self.entity_len);

                return Some(row);
            }

            if self.live_mask != 0 {
                let offset = self.live_mask.trailing_zeros() as usize;
                self.live_mask &= self.live_mask - 1;

                let row = self.base_row + offset;
                debug_assert!(row < self.entity_len);

                return Some(row);
            }

            if self.next_page >= self.page_end {
                return None;
            }

            let archetype = unsafe { &*self.archetype };
            let Some(batch) = archetype.page_batch(self.storage, self.next_page) else {
                self.next_page += 1;
                continue;
            };

            self.next_page += 1;
            self.base_row = batch.base_row;
            let mask = row_count_mask(batch.row_count);
            self.live_mask = batch.live_mask & mask;

            if self.live_mask == mask && mask != 0 {
                // Full pages (the contiguous-storage common case) iterate as
                // one dense run, skipping per-row mask arithmetic.
                self.run_row = batch.base_row;
                self.run_end = batch.base_row + batch.row_count;
                self.live_mask = 0;
            }
        }
    }
}

/// Builds the liveness bit mask covering `row_count` offsets of one page.
#[inline]
fn row_count_mask(row_count: usize) -> u64 {
    match row_count {
        0 => 0,
        ARCHETYPE_PAGE_ROWS => u64::MAX,
        rows => (1_u64 << rows) - 1,
    }
}
