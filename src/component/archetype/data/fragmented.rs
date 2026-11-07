//! Fragmented archetype storage lane.
//!
//! Rows may contain holes. Allocation reuses a page-local free list before
//! growing storage, and removal never moves another entity unless an explicit
//! compaction pass is requested.

use super::*;

/// Hole-preserving storage lane: live rows are tracked by page masks.
pub(crate) struct FragmentedData {
    /// Number of live rows summed across all pages.
    len: usize,
    /// Per-page liveness masks and allocation metadata.
    pages: Vec<FragmentedPage>,
    /// Head of the intrusive list of pages with spare rows, linked by `next_non_full`.
    first_non_full: Option<usize>,
    /// Entity handles indexed by global row; holes hold `Entity::INVALID`.
    entities: Vec<Entity>,
    /// Component columns in signature order.
    columns: Vec<ArchetypeColumn>,
}

impl FragmentedData {
    /// Frees a live row without moving any other row.
    pub(crate) fn free_row(&mut self, row: usize) {
        let PageRow { page, offset } = page_row(row);
        let was_full = self.pages[page].is_full();

        self.pages[page].free(offset);
        self.entities[row] = Entity::INVALID;
        self.len -= 1;

        if was_full {
            self.link_non_full_page(page);
        }
    }

    /// Removes prevalidated fragmented rows in column-major order.
    ///
    /// `rows` must be unique, live and sorted by ascending row number.
    pub(crate) fn remove_rows_prepared<I>(
        &mut self,
        rows: I,
        removals: &mut Vec<FragmentedPageRemoval>,
    ) -> usize
    where
        I: IntoIterator<Item = usize>,
    {
        removals.clear();

        let rows = rows.into_iter();
        let (minimum_rows, _) = rows.size_hint();

        removals.reserve(minimum_rows.div_ceil(ARCHETYPE_PAGE_ROWS).max(1));

        for row in rows {
            let PageRow { page, offset } = page_row(row);
            let bit = 1_u64 << offset;

            debug_assert!(
                self.pages
                    .get(page)
                    .is_some_and(|page| page.live_mask() & bit != 0),
                "prepared fragmented row must remain live"
            );

            if let Some(last) = removals.last_mut() {
                if last.page == page {
                    debug_assert_eq!(
                        last.mask & bit,
                        0,
                        "prepared fragmented rows must be unique"
                    );

                    last.mask |= bit;
                    continue;
                }

                debug_assert!(
                    last.page < page,
                    "fragmented batch rows must be page ordered"
                );
            }

            removals.push(FragmentedPageRemoval { page, mask: bit });
        }

        let removed = removals
            .iter()
            .map(|removal| removal.mask.count_ones() as usize)
            .sum::<usize>();

        if removed == 0 {
            return 0;
        }

        // Column-major traversal is the essential optimisation.
        //
        // Each concrete column remains active while all affected pages are
        // processed, improving locality and reducing dynamic dispatch.
        for column in &mut self.columns {
            let column = column.data_mut();

            for removal in removals.iter().copied() {
                unsafe {
                    column.drop_page_mask(removal.page, removal.mask);
                }
            }
        }

        // Commit entity and page metadata only after every component value has
        // been dropped.
        for removal in removals.iter().copied() {
            let was_full = self.pages[removal.page].is_full();
            let freed = self.pages[removal.page].free_mask(removal.mask);

            debug_assert_eq!(freed, removal.mask.count_ones() as usize);

            let base_row = removal.page * ARCHETYPE_PAGE_ROWS;
            let mut mask = removal.mask;

            while mask != 0 {
                let offset = mask.trailing_zeros() as usize;
                mask &= mask - 1;

                self.entities[base_row + offset] = Entity::INVALID;
            }

            // A full page was absent from the non-full list. It must be linked
            // exactly once after the first batch removal.
            if was_full {
                self.link_non_full_page(removal.page);
            }
        }

        self.len = self
            .len
            .checked_sub(removed)
            .expect("fragmented batch removal exceeded lane length");

        debug_assert!(self.columns.iter().all(|column| column.len() == self.len));

        removals.clear();
        removed
    }

    /// Updates entity and page metadata after compaction moved a row.
    pub(crate) fn mark_row_moved(&mut self, src: usize, dst: usize, entity: Entity) {
        let src_pos = page_row(src);
        let dst_pos = page_row(dst);

        if src_pos.page == dst_pos.page {
            let page = &mut self.pages[src_pos.page];
            page.mark_moved_from(src_pos.offset);
            page.mark_moved_to(dst_pos.offset);
        } else {
            self.pages[src_pos.page].mark_moved_from(src_pos.offset);
            self.pages[dst_pos.page].mark_moved_to(dst_pos.offset);
        }

        self.entities[dst] = entity;
        self.entities[src] = Entity::INVALID;
    }

    /// Counts initialized row slots that are currently holes.
    pub(crate) fn dead_rows(&self) -> usize {
        self.pages.iter().map(FragmentedPage::dead_rows).sum()
    }

    /// Rebuilds all page masks and free lists after live rows were compacted.
    pub(crate) fn rebuild_dense_pages(&mut self, live_len: usize) {
        let needed_pages = pages_for_rows(live_len);
        self.first_non_full = None;

        for page_index in 0..needed_pages {
            let base = page_index * ARCHETYPE_PAGE_ROWS;
            let live_in_page = live_len.saturating_sub(base).min(ARCHETYPE_PAGE_ROWS);
            self.pages[page_index].rebuild_dense(live_in_page);

            if live_in_page < ARCHETYPE_PAGE_ROWS {
                self.link_non_full_page(page_index);
            }
        }

        self.pages.truncate(needed_pages);
        self.entities
            .truncate(needed_pages.saturating_mul(ARCHETYPE_PAGE_ROWS));
    }

    /// Adds `page_index` to the allocation list when the page has spare space.
    fn link_non_full_page(&mut self, page_index: usize) {
        let page = &mut self.pages[page_index];

        if page.in_non_full_list || page.is_full() {
            return;
        }

        page.next_non_full = self.first_non_full;
        page.in_non_full_list = true;
        self.first_non_full = Some(page_index);
    }

    /// Removes the current allocation head after it became full.
    fn unlink_first_non_full_page(&mut self, page_index: usize) {
        debug_assert_eq!(self.first_non_full, Some(page_index));

        let page = &mut self.pages[page_index];
        self.first_non_full = page.next_non_full;
        page.next_non_full = None;
        page.in_non_full_list = false;
    }
}

impl Drop for FragmentedData {
    /// Drops only rows selected by page live masks.
    fn drop(&mut self) {
        for column in &mut self.columns {
            let column = column.data_mut();

            for (page_index, page) in self.pages.iter().enumerate() {
                let mask = page.live_mask();

                if mask == 0 {
                    continue;
                }

                unsafe {
                    column.drop_page_mask(page_index, mask);
                }
            }
        }
    }
}

impl ArchetypeData for FragmentedData {
    fn new(columns: Vec<ArchetypeColumn>) -> Self {
        Self {
            len: 0,
            pages: Vec::new(),
            first_non_full: None,
            entities: Vec::new(),
            columns,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn page_count(&self) -> usize {
        self.pages.len()
    }

    #[inline]
    fn capacity_rows(&self) -> usize {
        self.pages.len() * ARCHETYPE_PAGE_ROWS
    }

    #[inline]
    fn page_batch(&self, page: usize) -> Option<DataPageBatch> {
        let page_ref = self.pages.get(page)?;
        let live_mask = page_ref.live_mask();

        if live_mask == 0 {
            return None;
        }

        Some(DataPageBatch {
            page,
            base_row: page * ARCHETYPE_PAGE_ROWS,
            row_count: ARCHETYPE_PAGE_ROWS,
            live_mask: self.pages[page].live_mask(),
        })
    }

    #[inline]
    fn entity_ptrs(&self) -> (*const Entity, usize) {
        (self.entities.as_ptr(), self.entities.len())
    }

    #[inline]
    fn column(&self, index: usize) -> &ArchetypeColumn {
        self.columns.get(index).unwrap()
    }

    #[inline]
    fn column_mut(&mut self, index: usize) -> &mut ArchetypeColumn {
        self.columns.get_mut(index).unwrap()
    }

    #[inline]
    fn columns(&self) -> &[ArchetypeColumn] {
        &self.columns
    }

    #[inline]
    fn columns_mut(&mut self) -> &mut [ArchetypeColumn] {
        &mut self.columns
    }

    fn add_page(&mut self) {
        let page_index = self.pages.len();
        self.pages.push(FragmentedPage::new());
        self.entities
            .extend(std::iter::repeat_n(Entity::INVALID, ARCHETYPE_PAGE_ROWS));

        for column in &mut self.columns {
            column.data_mut().add_page();
        }

        self.link_non_full_page(page_index);
    }

    #[inline]
    fn allocate_row(&mut self) -> usize {
        if self.first_non_full.is_none() {
            self.add_page();
        }

        let page_index = self
            .first_non_full
            .expect("fragmented storage must track a non-full page");

        let offset = self.pages[page_index]
            .allocate()
            .expect("non-full fragmented page must allocate");

        if self.pages[page_index].is_full() {
            self.unlink_first_non_full_page(page_index);
        }

        self.len += 1;
        page_index * ARCHETYPE_PAGE_ROWS + offset
    }

    #[inline]
    fn write_entity(&mut self, row: usize, entity: Entity) {
        debug_assert!(row < self.entities.len());
        self.entities[row] = entity;
    }

    #[inline]
    fn entity(&self, row: usize) -> Option<Entity> {
        self.is_live(row).then_some(self.entities[row])
    }

    #[inline]
    fn is_live(&self, row: usize) -> bool {
        let PageRow { page, offset } = page_row(row);
        self.pages
            .get(page)
            .map(|page| page.live_mask() & (1_u64 << offset) != 0)
            .unwrap_or(false)
    }

    fn remove_row(&mut self, archetype_id: ArchetypeId, row: usize) -> Option<MovedEntity> {
        assert!(self.is_live(row), "row is not live in fragmented storage");

        let backfill_row = self.migration_backfill_row(row);

        for column in &mut self.columns {
            unsafe {
                column.data_mut().drop_row_and_backfill(row, backfill_row);
            }
        }

        self.finish_migrated_row(archetype_id, row, backfill_row)
    }

    fn migration_backfill_row(&self, row: usize) -> Option<usize> {
        debug_assert!(row < self.capacity_rows());
        None
    }

    fn finish_migrated_row(
        &mut self,
        _archetype_id: ArchetypeId,
        row: usize,
        backfill_row: Option<usize>,
    ) -> Option<MovedEntity> {
        debug_assert!(backfill_row.is_none());

        self.free_row(row);
        None
    }
}
