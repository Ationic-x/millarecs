//! Archetype tables and type-erased column storage.
//!
//! Each archetype owns two storage lanes with the same table-component
//! signature:
//!
//! - contiguous storage keeps live rows dense and removes holes by moving the
//!   tail row;
//! - fragmented storage keeps row locations stable and recycles holes through
//!   page-local free lists.
//!
//! Query execution receives page batches from both lanes through the same
//! abstraction, so iteration code can reason about live rows without knowing
//! which storage policy produced them.

use crate::{
    change_detection::{ChangeTick, ComponentTicks},
    component::{archetype::data::FragmentedPageRemoval, Component, ComponentId, MovedEntity},
    entity::{
        metadata::{ArchetypeStorageKind, EntityLocation},
        Entity,
    },
};

mod column;
mod data;
mod manager;
mod page;
pub(crate) mod plan;
mod signature;

pub use column::ColumnFactoryError;
pub(crate) use column::{ArchetypeColumn, BatchTransitionRow, Column, ColumnFactory, ColumnPaged};
pub(crate) use data::DataPageBatch;
use data::{ArchetypeData, ContiguousData, FragmentedData};
pub(crate) use manager::ArchetypeManager;
pub(crate) use page::ARCHETYPE_PAGE_ROWS;
pub use signature::ArchetypeSignature;

/// Errors raised while creating or retrieving archetype storage.
#[derive(Debug)]
pub enum ArchetypeManagerError {
    /// A column could not be created for one of the signature components.
    Factory(ColumnFactoryError),
    /// An archetype rejected the requested layout or row operation.
    Archetype(ArchetypeError),
}

/// Stable index of an archetype inside `ArchetypeManager`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArchetypeId(u32);

impl ArchetypeId {
    /// Identifier reserved for the empty signature archetype.
    pub const EMPTY: ArchetypeId = ArchetypeId(0);

    /// Sentinel value used before an entity has a real archetype location.
    pub const INVALID: Self = Self(u32::MAX);

    /// Creates an id from a dense archetype index.
    #[inline]
    pub const fn new(index: usize) -> Self {
        assert!(
            index < u32::MAX as usize,
            "archetype id exhausted its valid range"
        );
        Self(index as u32)
    }

    /// Returns the dense archetype index.
    #[inline]
    pub const fn index(&self) -> usize {
        self.0 as usize
    }

    /// Returns `false` only for the `INVALID` sentinel.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.0 != u32::MAX
    }
}

/// Errors raised by archetype row, signature, or column operations.
#[derive(Debug)]
pub enum ArchetypeError {
    /// A row index was outside the current entity range.
    RowOutOfBounds {
        /// The rejected row index.
        row: usize,
        /// Live rows of the lane at the time of the access.
        len: usize,
    },
    /// A signature, column order, or concrete column type did not match.
    SignatureMismatch,
}

impl From<ArchetypeError> for ArchetypeManagerError {
    fn from(e: ArchetypeError) -> Self {
        Self::Archetype(e)
    }
}

impl From<ColumnFactoryError> for ArchetypeManagerError {
    fn from(e: ColumnFactoryError) -> Self {
        Self::Factory(e)
    }
}

/// Table containing entities that share the same table-component signature.
pub struct Archetype {
    id: ArchetypeId,
    signature: ArchetypeSignature,
    contiguous: ContiguousData,
    fragmented: FragmentedData,
}

/// Snapshot of one archetype storage lane.
///
/// `dead_rows` only counts initialized row slots that used to contain an entity
/// and are now holes. Uninitialized capacity at the end of a page is reported as
/// free capacity, not as fragmentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchetypeStorageStats {
    /// Storage lane described by this snapshot.
    pub storage: ArchetypeStorageKind,
    /// Live rows currently addressable by entity metadata.
    pub live_rows: usize,
    /// Allocated row slots across all pages.
    pub capacity_rows: usize,
    /// Allocated page count.
    pub page_count: usize,
    /// Initialized holes inside the lane.
    pub dead_rows: usize,
}

impl ArchetypeStorageStats {
    /// Returns initialized row slots, including live rows and holes.
    #[inline]
    pub const fn initialized_rows(self) -> usize {
        self.live_rows + self.dead_rows
    }

    /// Returns allocated row slots that have never been initialized.
    #[inline]
    pub const fn free_rows(self) -> usize {
        self.capacity_rows
            .saturating_sub(self.live_rows + self.dead_rows)
    }

    /// Returns the percentage of initialized slots that are holes.
    #[inline]
    pub const fn dead_ratio_percent(self) -> usize {
        let initialized = self.initialized_rows();
        match self.dead_rows.saturating_mul(100).checked_div(initialized) {
            Some(ratio) => ratio,
            None => 0,
        }
    }
}

/// Policy deciding when fragmented pages should be compacted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CompactionPolicy {
    /// Minimum number of holes required before compacting.
    pub min_dead_rows: usize,
    /// Minimum hole ratio over initialized rows required before compacting.
    pub min_dead_ratio_percent: usize,
}

impl Default for CompactionPolicy {
    /// Compacts only after one page of holes covers a quarter of initialized rows.
    fn default() -> Self {
        Self {
            min_dead_rows: ARCHETYPE_PAGE_ROWS,
            min_dead_ratio_percent: 25,
        }
    }
}

impl CompactionPolicy {
    /// Builds a policy from explicit thresholds.
    pub const fn new(min_dead_rows: usize, min_dead_ratio_percent: usize) -> Self {
        Self {
            min_dead_rows,
            min_dead_ratio_percent,
        }
    }

    /// Compacts as soon as there is any real fragmentation.
    pub const fn aggressive() -> Self {
        Self::new(1, 1)
    }

    /// Disables automatic compaction decisions.
    pub const fn disabled() -> Self {
        Self::new(usize::MAX, usize::MAX)
    }

    /// Returns whether the policy accepts the provided storage statistics.
    pub fn should_compact(self, stats: ArchetypeStorageStats) -> bool {
        stats.storage == ArchetypeStorageKind::Fragmented
            && stats.dead_rows > 0
            && stats.dead_rows >= self.min_dead_rows
            && stats.dead_ratio_percent() >= self.min_dead_ratio_percent
    }
}

/// Result of a fragmented-page compaction pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FragmentedCompactionReport {
    /// Number of archetypes whose fragmented lane was compacted.
    pub compacted_archetypes: usize,
    /// Number of entity locations that had to be updated.
    pub moved_rows: usize,
}

impl Default for Archetype {
    /// Creates the empty-signature archetype.
    fn default() -> Self {
        Self {
            id: ArchetypeId::EMPTY,
            signature: ArchetypeSignature::default(),
            contiguous: ContiguousData::new(Vec::new()),
            fragmented: FragmentedData::new(Vec::new()),
        }
    }
}

impl Archetype {
    /// Creates an archetype and verifies that its columns match the signature order.
    pub(crate) fn new(
        id: ArchetypeId,
        signature: ArchetypeSignature,
        contiguous_columns: Vec<ArchetypeColumn>,
        fragmented_columns: Vec<ArchetypeColumn>,
    ) -> Result<Self, ArchetypeError> {
        if signature.components().len() != contiguous_columns.len()
            || signature.components().len() != fragmented_columns.len()
        {
            return Err(ArchetypeError::SignatureMismatch);
        }

        for (c, col) in signature.components().iter().zip(contiguous_columns.iter()) {
            if c != &col.component_id {
                return Err(ArchetypeError::SignatureMismatch);
            }
        }

        for (c, col) in signature.components().iter().zip(fragmented_columns.iter()) {
            if c != &col.component_id {
                return Err(ArchetypeError::SignatureMismatch);
            }
        }

        Ok(Self {
            id,
            signature,
            contiguous: ContiguousData::new(contiguous_columns),
            fragmented: FragmentedData::new(fragmented_columns),
        })
    }

    /// Returns the sorted component signature for this archetype.
    pub fn signature(&self) -> &ArchetypeSignature {
        &self.signature
    }

    /// Returns immutable access to raw archetype storage.
    pub(crate) fn data(&self, storage: ArchetypeStorageKind) -> &dyn ArchetypeData {
        match storage {
            ArchetypeStorageKind::Contiguous => &self.contiguous,
            ArchetypeStorageKind::Fragmented => &self.fragmented,
        }
    }

    /// Returns mutable access to raw archetype storage.
    pub(crate) fn data_mut(&mut self, storage: ArchetypeStorageKind) -> &mut dyn ArchetypeData {
        match storage {
            ArchetypeStorageKind::Contiguous => &mut self.contiguous,
            ArchetypeStorageKind::Fragmented => &mut self.fragmented,
        }
    }

    /// Returns the amount of live entities currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.contiguous.len() + self.fragmented.len()
    }

    /// Returns whether this archetype contains no live rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns page and fragmentation metrics for one storage lane.
    pub fn storage_stats(&self, storage: ArchetypeStorageKind) -> ArchetypeStorageStats {
        match storage {
            ArchetypeStorageKind::Contiguous => ArchetypeStorageStats {
                storage,
                live_rows: self.contiguous.len(),
                capacity_rows: self.contiguous.capacity_rows(),
                page_count: self.contiguous.page_count(),
                dead_rows: 0,
            },
            ArchetypeStorageKind::Fragmented => ArchetypeStorageStats {
                storage,
                live_rows: self.fragmented.len(),
                capacity_rows: self.fragmented.capacity_rows(),
                page_count: self.fragmented.page_count(),
                dead_rows: self.fragmented.dead_rows(),
            },
        }
    }

    #[inline]
    /// Returns how many pages are allocated by the selected storage lane.
    pub(crate) fn page_count(&self, storage: ArchetypeStorageKind) -> usize {
        self.data(storage).page_count()
    }

    #[inline]
    /// Returns a raw page batch from the selected storage lane.
    pub(crate) fn page_batch(
        &self,
        storage: ArchetypeStorageKind,
        page: usize,
    ) -> Option<DataPageBatch> {
        self.data(storage).page_batch(page)
    }

    #[inline]
    /// Returns a page batch enriched with archetype and storage identity.
    pub(crate) fn archetype_page_batch(
        &self,
        storage: ArchetypeStorageKind,
        page: usize,
    ) -> Option<ArchetypePageBatch> {
        let batch = self.page_batch(storage, page)?;

        Some(ArchetypePageBatch {
            archetype_id: self.id,
            storage,
            page: batch.page,
            base_row: batch.base_row,
            row_count: batch.row_count,
            live_mask: batch.live_mask,
        })
    }

    #[inline]
    /// Returns the raw entity array pointer for the selected storage lane.
    pub(crate) fn entity_ptrs(&self, storage: ArchetypeStorageKind) -> (*const Entity, usize) {
        self.data(storage).entity_ptrs()
    }

    #[inline]
    /// Returns whether a row is live in the selected storage lane.
    pub(crate) fn is_row_live(&self, storage: ArchetypeStorageKind, row: usize) -> bool {
        self.data(storage).is_live(row)
    }

    /// Returns the column index for a component in this archetype.
    #[inline]
    pub fn column_index(&self, component_id: ComponentId) -> Result<usize, ArchetypeError> {
        self.signature.column_index(component_id)
    }

    /// Appends an entity to an archetype that has no component columns.
    pub(crate) fn push_empty_row(
        &mut self,
        storage: ArchetypeStorageKind,
        entity: Entity,
    ) -> Result<usize, ArchetypeError> {
        let data = self.data_mut(storage);

        if !data.columns().is_empty() {
            return Err(ArchetypeError::SignatureMismatch);
        }

        let row = data.allocate_row();
        data.write_entity(row, entity);

        Ok(row)
    }

    /// Errors with `RowOutOfBounds` unless the row is live in the selected lane.
    fn validate_live_row(
        &self,
        storage: ArchetypeStorageKind,
        row: usize,
    ) -> Result<(), ArchetypeError> {
        let data = self.data(storage);

        if data.is_live(row) {
            Ok(())
        } else {
            Err(ArchetypeError::RowOutOfBounds {
                row,
                len: data.len(),
            })
        }
    }

    /// Removes a row using the selected storage policy.
    ///
    /// Contiguous storage may move the last live row into the removed row and
    /// returns that moved entity so metadata can be repaired. Fragmented
    /// storage only marks the row as free and therefore returns `None`.
    pub(crate) fn remove_row(
        &mut self,
        storage: ArchetypeStorageKind,
        row: usize,
    ) -> Result<Option<MovedEntity>, ArchetypeError> {
        self.validate_live_row(storage, row)?;
        let archetype_id = self.id;

        Ok(self.data_mut(storage).remove_row(archetype_id, row))
    }

    /// Removes prevalidated fragmented rows without performing row movement.
    pub(crate) fn remove_fragmented_rows_prepared<I>(
        &mut self,
        rows: I,
        removals: &mut Vec<FragmentedPageRemoval>,
    ) -> usize
    where
        I: IntoIterator<Item = usize>,
    {
        self.fragmented.remove_rows_prepared(rows, removals)
    }

    /// Compacts fragmented storage into a dense prefix and returns moved rows.
    ///
    /// Component data is moved inside each column, page metadata is rebuilt, and
    /// every moved entity is returned so the entity manager can update
    /// locations.
    pub(crate) fn compact_pages(&mut self) -> Vec<MovedEntity> {
        let data = &mut self.fragmented;

        if data.len() == 0 {
            data.rebuild_dense_pages(0);
            for column in data.columns_mut() {
                column.data_mut().rebuild_dense_metadata(0);
            }
            return Vec::new();
        }

        let live_len = data.len();
        let capacity = data.capacity_rows();
        let mut moved = Vec::new();
        let mut dst = 0usize;

        for src in 0..capacity {
            if !data.is_live(src) {
                continue;
            }

            if src == dst {
                dst += 1;
                continue;
            }

            debug_assert!(dst < src);
            let entity = data.entity(src).expect("live row must have entity");

            for column in data.columns_mut() {
                unsafe {
                    column.data_mut().move_row_within(src, dst);
                }
            }

            data.mark_row_moved(src, dst, entity);

            moved.push(MovedEntity {
                entity,
                location: EntityLocation {
                    archetype_id: self.id,
                    storage: ArchetypeStorageKind::Fragmented,
                    row: dst as u32,
                },
            });

            dst += 1;
        }

        debug_assert_eq!(dst, live_len);

        data.rebuild_dense_pages(live_len);
        for column in data.columns_mut() {
            column.data_mut().rebuild_dense_metadata(live_len);
        }

        moved
    }

    /// Compacts fragmented storage only when the provided policy selects it.
    pub(crate) fn compact_fragmented_with_policy(
        &mut self,
        policy: CompactionPolicy,
    ) -> Option<Vec<MovedEntity>> {
        if !policy.should_compact(self.storage_stats(ArchetypeStorageKind::Fragmented)) {
            return None;
        }

        Some(self.compact_pages())
    }

    /// Returns raw component storage used by shared query fetches.
    ///
    /// # Safety
    ///
    /// The lane must stay structurally unchanged and free of exclusive
    /// writers while the pointers are used, and only rows below the returned
    /// capacity may be addressed.
    pub(crate) unsafe fn get_column_read_ptrs<C: Component>(
        &self,
        storage: ArchetypeStorageKind,
        component_id: ComponentId,
    ) -> Result<(*const C, usize), ArchetypeError> {
        let index = self.column_index(component_id)?;
        let column = self
            .data(storage)
            .column(index)
            .data()
            .as_any()
            .downcast_ref::<ColumnPaged<C>>()
            .ok_or(ArchetypeError::SignatureMismatch)?;

        Ok(column.read_ptrs())
    }

    /// Returns raw component and tick pointers for query mutation.
    ///
    /// # Safety
    ///
    /// The caller must hold exclusive value and tick access (normally a
    /// validated query access set), keep the lane structurally unchanged
    /// while the pointers are live, and address only rows below the returned
    /// capacity.
    pub(crate) unsafe fn get_column_write_ptrs<C: Component>(
        &self,
        storage: ArchetypeStorageKind,
        component_id: ComponentId,
    ) -> Result<(*mut C, *mut ComponentTicks, usize), ArchetypeError> {
        let index = self.column_index(component_id)?;
        let column = self
            .data(storage)
            .column(index)
            .data()
            .as_any()
            .downcast_ref::<ColumnPaged<C>>()
            .ok_or(ArchetypeError::SignatureMismatch)?;

        Ok(unsafe { column.write_ptrs() })
    }

    /// Records that a query write fetch is active on one column.
    ///
    /// # Safety
    ///
    /// Every path that grants mutable access to the column at `tick` must call
    /// this, or change filters may skip the column as untouched.
    pub(crate) unsafe fn note_column_write_access(
        &self,
        storage: ArchetypeStorageKind,
        component_id: ComponentId,
        tick: ChangeTick,
    ) {
        let Ok(index) = self.column_index(component_id) else {
            return;
        };
        self.data(storage)
            .column(index)
            .data()
            .note_write_access(tick);
    }

    /// Returns raw component and tick pointers for one live row.
    ///
    /// # Safety
    ///
    /// The caller must hold exclusive access to both slots for the row, and
    /// the pointers must not outlive structural changes to the lane.
    pub(crate) unsafe fn get_component_mut_ptrs<C: Component>(
        &self,
        storage: ArchetypeStorageKind,
        component_id: ComponentId,
        row: usize,
    ) -> Result<(*mut C, *mut ComponentTicks), ArchetypeError> {
        self.validate_live_row(storage, row)?;

        let index = self.column_index(component_id)?;
        let column = self
            .data(storage)
            .column(index)
            .data()
            .as_any()
            .downcast_ref::<ColumnPaged<C>>()
            .ok_or(ArchetypeError::SignatureMismatch)?;

        Ok(unsafe { column.row_mut_ptrs(row) })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Page batch with archetype identity attached.
///
/// Query work stores these fields so prepared execution can dispatch a unit
/// without having to recover parent metadata from the lower-level
/// `DataPageBatch`.
pub(crate) struct ArchetypePageBatch {
    /// Archetype that owns the page.
    pub(crate) archetype_id: ArchetypeId,
    /// Storage lane that produced the batch.
    pub(crate) storage: ArchetypeStorageKind,
    /// Page index inside the selected lane.
    pub(crate) page: usize,
    /// Global row for local offset zero.
    pub(crate) base_row: usize,
    /// Number of offsets covered by the batch.
    pub(crate) row_count: usize,
    /// Live offsets inside the batch.
    pub(crate) live_mask: u64,
}
