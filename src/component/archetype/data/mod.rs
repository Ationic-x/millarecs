//! Storage-lane abstraction for archetypes.
//!
//! `Archetype` owns one contiguous lane and one fragmented lane. Both expose the
//! same logical operations, but each implementation keeps its own allocation,
//! removal, and page-liveness invariants.

use crate::{
    component::MovedEntity,
    entity::{metadata::EntityLocation, Entity},
};

use super::{
    page::{
        page_row, pages_for_rows, ContiguousPage, FragmentedPage, PageRow, ARCHETYPE_PAGE_ROWS,
    },
    ArchetypeColumn, ArchetypeId,
};

mod contiguous;
mod fragmented;

pub(super) use contiguous::ContiguousData;
pub(super) use fragmented::FragmentedData;

/// Common read and write surface exposed by concrete archetype storage lanes.
///
/// The concrete lane owns the invariants. Contiguous storage never has holes;
/// fragmented storage may have holes and tracks them with page-local free
/// lists. This trait only exists so `Archetype` can dispatch by user-selected
/// storage without reintroducing a shared `kind` field.
pub(crate) trait ArchetypeData {
    /// Creates an empty storage lane using the provided typed columns.
    fn new(columns: Vec<ArchetypeColumn>) -> Self
    where
        Self: Sized;
    /// Returns the number of live entities in this lane.
    fn len(&self) -> usize;
    /// Returns the number of allocated pages.
    fn page_count(&self) -> usize;
    /// Returns the total row capacity across allocated pages.
    fn capacity_rows(&self) -> usize;

    /// Returns a page batch if the requested page contains live rows.
    fn page_batch(&self, page: usize) -> Option<DataPageBatch>;

    /// Returns the raw entity array pointer and its allocated length.
    fn entity_ptrs(&self) -> (*const Entity, usize);
    /// Returns one type-erased component column by signature index.
    fn column(&self, index: usize) -> &ArchetypeColumn;
    /// Returns one mutable type-erased component column by signature index.
    fn column_mut(&mut self, index: usize) -> &mut ArchetypeColumn;
    /// Returns all type-erased columns in signature order.
    fn columns(&self) -> &[ArchetypeColumn];
    /// Returns all mutable type-erased columns in signature order.
    fn columns_mut(&mut self) -> &mut [ArchetypeColumn];

    /// Grows the lane by one page in every column and in entity metadata.
    fn add_page(&mut self);
    /// Allocates one row according to the lane policy.
    fn allocate_row(&mut self) -> usize;
    /// Writes the entity handle stored at `row`.
    fn write_entity(&mut self, row: usize, entity: Entity);
    /// Returns the entity at `row` when the row is currently live.
    fn entity(&self, row: usize) -> Option<Entity>;
    /// Returns whether `row` is currently live.
    fn is_live(&self, row: usize) -> bool;

    /// Removes a live row and returns any entity moved to repair metadata.
    fn remove_row(&mut self, archetype_id: ArchetypeId, row: usize) -> Option<MovedEntity>;

    /// Returns the live tail row that must fill `row`, when this lane compacts
    /// removals. Fragmented storage leaves holes and therefore returns `None`.
    fn migration_backfill_row(&self, row: usize) -> Option<usize>;

    /// Finalizes only lane metadata after every column has already moved or
    /// dropped the source row and applied the same optional backfill.
    fn finish_migrated_row(
        &mut self,
        archetype_id: ArchetypeId,
        row: usize,
        backfill_row: Option<usize>,
    ) -> Option<MovedEntity>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Page batch produced by one storage lane.
///
/// Contiguous lanes report a shortened `row_count` and a dense live mask.
/// Fragmented lanes report the full page width and use `live_mask` to identify
/// occupied offsets.
pub(crate) struct DataPageBatch {
    /// Page index inside the lane.
    pub(crate) page: usize,
    /// Global row index represented by local offset zero.
    pub(crate) base_row: usize,
    /// Number of offsets covered by the batch.
    pub(crate) row_count: usize,
    /// Live offsets within the covered range.
    pub(crate) live_mask: u64,
}

/// Rows removed from one fragmented page.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FragmentedPageRemoval {
    /// Page index inside the lane.
    pub(crate) page: usize,
    /// Bitmask of freed offsets within `page`.
    pub(crate) mask: u64,
}
