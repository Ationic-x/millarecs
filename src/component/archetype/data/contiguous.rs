//! Contiguous archetype storage lane.
//!
//! Live rows are always the prefix `0..len`. Removing a non-tail row moves the
//! last live row into the hole and returns the moved entity so metadata can be
//! repaired by the caller.

use crate::entity::ArchetypeStorageKind::Contiguous;

use super::*;

/// Dense storage lane: live rows are always `0..len`.
pub(crate) struct ContiguousData {
    /// Number of live rows; live rows are exactly the prefix `0..len`.
    len: usize,
    /// Per-page allocation metadata, one entry per allocated page.
    pages: Vec<ContiguousPage>,
    /// Entity handles indexed by global row; rows at or beyond `len` hold `Entity::INVALID`.
    entities: Vec<Entity>,
    /// Component columns in signature order.
    columns: Vec<ArchetypeColumn>,
}

impl ContiguousData {
    /// Updates page metadata after the global tail row was removed.
    fn mark_tail_removed(&mut self, row: usize) {
        let PageRow { page, offset } = page_row(row);
        self.pages[page].mark_tail_removed(offset);
    }
}

impl Drop for ContiguousData {
    /// Drops every live component value still owned by the lane.
    fn drop(&mut self) {
        for row in 0..self.len {
            for column in &mut self.columns {
                unsafe {
                    column.data_mut().drop_row(row);
                }
            }
        }
    }
}

impl ArchetypeData for ContiguousData {
    fn new(columns: Vec<ArchetypeColumn>) -> Self {
        Self {
            len: 0,
            pages: Vec::new(),
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
        if page >= self.pages.len() {
            return None;
        }

        let base_row = page * ARCHETYPE_PAGE_ROWS;

        if base_row >= self.len {
            return None;
        }

        let row_count = (self.len - base_row).min(ARCHETYPE_PAGE_ROWS);

        Some(DataPageBatch {
            page,
            base_row,
            row_count,
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
        self.pages.push(ContiguousPage::new());
        self.entities
            .extend(std::iter::repeat_n(Entity::INVALID, ARCHETYPE_PAGE_ROWS));

        for column in &mut self.columns {
            column.data_mut().add_page();
        }
    }

    #[inline]
    fn allocate_row(&mut self) -> usize {
        if self.len == self.capacity_rows() {
            self.add_page();
        }

        let row = self.len;
        let PageRow {
            page,
            offset: expected_offset,
        } = page_row(row);
        let offset = self.pages[page]
            .allocate()
            .expect("contiguous page must have a trailing free row");

        self.len += 1;
        debug_assert_eq!(offset, expected_offset);
        row
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
        row < self.len
    }

    fn remove_row(&mut self, archetype_id: ArchetypeId, row: usize) -> Option<MovedEntity> {
        assert!(row < self.len, "row is outside the contiguous table");

        let backfill_row = self.migration_backfill_row(row);

        for column in &mut self.columns {
            unsafe {
                column.data_mut().drop_row_and_backfill(row, backfill_row);
            }
        }

        self.finish_migrated_row(archetype_id, row, backfill_row)
    }

    fn migration_backfill_row(&self, row: usize) -> Option<usize> {
        debug_assert!(row < self.len);

        let last_row = self.len - 1;

        if row == last_row {
            None
        } else {
            Some(last_row)
        }
    }

    fn finish_migrated_row(
        &mut self,
        archetype_id: ArchetypeId,
        row: usize,
        backfill_row: Option<usize>,
    ) -> Option<MovedEntity> {
        debug_assert!(row < self.len);

        let last_row = self.len - 1;
        let expected_backfill = (row != last_row).then_some(last_row);

        debug_assert_eq!(backfill_row, expected_backfill);

        let moved_entity = backfill_row.map(|source_row| {
            let entity = self
                .entity(source_row)
                .expect("contiguous backfill row must contain a live entity");

            self.entities[row] = entity;

            MovedEntity {
                entity,
                location: EntityLocation {
                    archetype_id,
                    storage: Contiguous,
                    row: row as u32,
                },
            }
        });

        self.mark_tail_removed(last_row);
        self.entities[last_row] = Entity::INVALID;
        self.len -= 1;

        moved_entity
    }
}
