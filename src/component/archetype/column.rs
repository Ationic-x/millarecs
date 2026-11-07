//! Type-erased component columns used by archetype storage.
//!
//! Columns allocate fixed-size pages for values and change ticks. The archetype
//! storage lanes decide which rows are live; columns only own the per-component
//! memory and the unsafe row movement primitives required by migration and
//! compaction.

use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{
    change_detection::{ChangeTick, ComponentTicks},
    component::{Component, ComponentId},
    entity::Entity,
};

use super::page::{page_row, pages_for_rows, PageRow, ARCHETYPE_PAGE_ROWS};

/// Creates one page-sized array of uninitialized values.
fn uninit_array<T, const N: usize>() -> [MaybeUninit<T>; N] {
    // SAFETY: `[MaybeUninit<T>; N]` may be left fully uninitialized.
    unsafe { MaybeUninit::<[MaybeUninit<T>; N]>::uninit().assume_init() }
}

/// Type-erased storage column for one component inside an archetype.
pub(crate) struct ArchetypeColumn {
    pub(crate) component_id: ComponentId,
    data: Box<dyn Column>,
}

impl ArchetypeColumn {
    /// Creates a column for `component_id` using an already constructed storage backend.
    pub(crate) fn new(component_id: ComponentId, data: Box<dyn Column>) -> Self {
        Self { component_id, data }
    }

    /// Returns mutable access to the erased column implementation.
    pub(crate) fn data_mut(&mut self) -> &mut dyn Column {
        &mut *self.data
    }

    /// Returns immutable access to the erased column implementation.
    pub(crate) fn data(&self) -> &dyn Column {
        &*self.data
    }

    /// Returns the amount of live component values stored in this column.
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns a raw pointer to the first change tick slot in this column.
    pub(crate) fn ticks_ptr(&self) -> *const ComponentTicks {
        self.data.ticks_ptr()
    }
}

/// Errors returned when a column constructor cannot be found.
#[derive(Debug)]
pub enum ColumnFactoryError {
    /// No column constructor has been registered for the component id.
    NotRegistered(ComponentId),
}

/// Row coordinates for one entity inside a batch transition.
///
/// Migrations read `source_row`, write the reserved `destination_row`, and then
/// repair the source hole with `backfill_row`. Pure drops only use
/// `source_row` and `backfill_row`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BatchTransitionRow {
    /// Entity occupying the moved or dropped row.
    pub(crate) entity: Entity,
    /// Row the entity currently occupies in its archetype.
    pub(crate) source_row: usize,
    /// Reserved, uninitialized row receiving the entity on migrations.
    pub(crate) destination_row: usize,
    /// Tail row moved into `source_row` to repair its hole, if any.
    pub(crate) backfill_row: Option<usize>,
}

/// Type-erased operations required by archetype columns.
pub(crate) trait Column: 'static + Send + Sync {
    /// Returns the number of live values.
    fn len(&self) -> usize;

    /// Adds one uninitialized page to the value and tick storage.
    fn add_page(&mut self);

    /// Drops one initialized row.
    ///
    /// # Safety
    ///
    /// `row` must contain an initialized value.
    unsafe fn drop_row(&mut self, row: usize);

    /// Drops every initialized row selected by `mask` in one page.
    ///
    /// # Safety
    ///
    /// Every selected offset must contain an initialized value. Masks supplied
    /// to the same column operation must not select the same row twice.
    unsafe fn drop_page_mask(&mut self, page: usize, mask: u64);

    /// Moves several rows to a compatible destination column.
    ///
    /// # Safety
    ///
    /// Every source row must be initialized. Destination rows must be reserved
    /// and uninitialized. The sequence must describe the same row-removal order
    /// later applied to the owning archetype lane.
    unsafe fn move_rows_to_and_backfill(
        &mut self,
        destination: &mut dyn Column,
        rows: &[BatchTransitionRow],
    );

    /// Drops several rows and applies their precomputed backfills.
    ///
    /// # Safety
    ///
    /// Every source row must be initialized and each row must occur once.
    unsafe fn drop_rows_and_backfill(&mut self, rows: &[BatchTransitionRow]);

    /// Moves a component between two rows of the same column.
    ///
    /// # Safety
    ///
    /// `src_row` must be initialized and `dst_row` must be reserved for the
    /// moved value.
    unsafe fn move_row_within(&mut self, src_row: usize, dst_row: usize);

    /// Appends or writes a new row by moving a value from a raw pointer.
    ///
    /// # Safety
    ///
    /// `src` must point to a valid `T` for this column and ownership is moved
    /// into the column.
    unsafe fn push_move_from_ptr_at(&mut self, row: usize, src: *mut u8, change_tick: ChangeTick);

    /// Replaces an existing row by moving a value from a raw pointer.
    ///
    /// # Safety
    ///
    /// `row` must be initialized and `src` must point to a valid replacement
    /// value of the same concrete component type.
    unsafe fn write_move_from_ptr(&mut self, row: usize, src: *mut u8, change_tick: ChangeTick);

    /// Moves one row to a compatible column and repairs the source hole in the
    /// same type-erased dispatch.
    ///
    /// # Safety
    ///
    /// Both columns must store the same concrete component type. `row` and an
    /// optional `backfill_row` must be distinct initialized source rows, while
    /// `destination_row` must be reserved but uninitialized in `destination`.
    unsafe fn move_row_to_and_backfill(
        &mut self,
        row: usize,
        destination: &mut dyn Column,
        destination_row: usize,
        backfill_row: Option<usize>,
    );

    /// Drops one row and repairs its hole in the same type-erased dispatch.
    ///
    /// # Safety
    ///
    /// `row` and an optional `backfill_row` must be distinct initialized rows.
    /// The caller must update the owning lane metadata after every column has
    /// completed the same operation.
    unsafe fn drop_row_and_backfill(&mut self, row: usize, backfill_row: Option<usize>);

    /// Rebuilds column metadata after rows have been compacted into a dense prefix.
    fn rebuild_dense_metadata(&mut self, live_len: usize);

    /// Returns a raw pointer to the first tick slot.
    fn ticks_ptr(&self) -> *const ComponentTicks;

    /// Exposes the column for immutable downcasting.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Exposes the column for mutable downcasting.
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;

    /// Returns the most recent tick at which this column was written.
    fn last_write_access(&self) -> ChangeTick;

    /// Records write activity on this column at `tick`.
    fn note_write_access(&self, tick: ChangeTick);
}

/// Fixed-page implementation of a table column.
pub(crate) struct ColumnPaged<T> {
    /// Component values, one fixed-size page per entry.
    values: Vec<[MaybeUninit<UnsafeCell<T>>; ARCHETYPE_PAGE_ROWS]>,
    /// Change-tick pairs, parallel to `values`.
    ticks: Vec<[MaybeUninit<UnsafeCell<ComponentTicks>>; ARCHETYPE_PAGE_ROWS]>,
    /// Live values stored across all pages.
    len: usize,
    /// Monotonic tick of the last write access granted on this column.
    ///
    /// Change filters consult it to skip whole work units over archetypes
    /// where nothing can have changed since the reader's window. Written via
    /// `fetch_max` so out-of-order writers cannot move it backwards; u64 tick
    /// wrap is beyond practical reach.
    last_write_access: AtomicU64,
}

impl<T> Default for ColumnPaged<T> {
    /// Creates a column with no pages and no recorded write access.
    fn default() -> Self {
        Self {
            values: Vec::new(),
            ticks: Vec::new(),
            len: 0,
            last_write_access: AtomicU64::new(0),
        }
    }
}

// SAFETY:
// Query pointers address individual `UnsafeCell` slots. Their caller must
// enforce filtered access exclusivity and structural immutability. Structural
// methods require `&mut self`.
unsafe impl<C: Component> Sync for ColumnPaged<C> {}

impl<C: Component> ColumnPaged<C> {
    /// Returns total row capacity across allocated pages.
    fn capacity_rows(&self) -> usize {
        self.values.len() * ARCHETYPE_PAGE_ROWS
    }

    /// Ensures that `page` exists in both value and tick storage.
    fn ensure_page(&mut self, page: usize) {
        while self.values.len() <= page {
            self.add_page();
        }
    }

    /// Converts a global row into page-local coordinates.
    fn slot(row: usize) -> PageRow {
        page_row(row)
    }

    /// Returns a mutable pointer to the value slot at `row`.
    ///
    /// # Safety
    ///
    /// The page holding `row` must already be allocated.
    unsafe fn value_ptr(&self, row: usize) -> *mut C {
        let PageRow { page, offset } = Self::slot(row);
        self.values[page][offset]
            .as_ptr()
            .cast::<UnsafeCell<C>>()
            .cast::<C>()
            .cast_mut()
    }

    /// Returns a mutable pointer to the tick slot at `row`.
    ///
    /// # Safety
    ///
    /// The page holding `row` must already be allocated.
    unsafe fn ticks_ptr_at(&self, row: usize) -> *mut ComponentTicks {
        let PageRow { page, offset } = Self::slot(row);
        self.ticks[page][offset]
            .as_ptr()
            .cast::<UnsafeCell<ComponentTicks>>()
            .cast::<ComponentTicks>()
            .cast_mut()
    }

    /// Writes a moved value and tick pair into a reserved destination row.
    ///
    /// # Safety
    ///
    /// `row` must be reserved but uninitialized; the moved-in pair replaces
    /// no previous occupant.
    unsafe fn write_existing_value(&mut self, row: usize, value: C, ticks: ComponentTicks) {
        let PageRow { page, offset } = Self::slot(row);
        self.ensure_page(page);

        self.values[page][offset].write(UnsafeCell::new(value));
        self.ticks[page][offset].write(UnsafeCell::new(ticks));

        self.note_write_access(ticks.latest_tick());
        self.len += 1;
    }

    /// Moves the value and ticks out of a live row.
    ///
    /// # Safety
    ///
    /// `row` must be initialized; both slots are left uninitialized.
    unsafe fn read_move(&mut self, row: usize) -> (C, ComponentTicks) {
        let value = unsafe { core::ptr::read(self.value_ptr(row)) };
        let ticks = unsafe { core::ptr::read(self.ticks_ptr_at(row)) };

        self.len -= 1;

        (value, ticks)
    }

    /// Fills `row` from the source lane tail after `row` was moved or dropped.
    /// Keeping this operation concrete avoids a second virtual call per column.
    ///
    /// # Safety
    ///
    /// `row` must be an uninitialized hole and `backfill_row`, when present,
    /// an initialized live row distinct from it.
    unsafe fn backfill_from(&mut self, row: usize, backfill_row: Option<usize>) {
        let Some(last_row) = backfill_row else {
            return;
        };

        debug_assert_ne!(row, last_row);

        let (value, ticks) = unsafe { self.read_move(last_row) };
        unsafe {
            self.write_existing_value(row, value, ticks);
        }
    }

    #[inline]
    /// Returns raw read pointers and row capacity for query fetches.
    pub(crate) fn read_ptrs(&self) -> (*const C, usize) {
        let ptr = self
            .values
            .first()
            .map(|page| page.as_ptr().cast::<C>())
            .unwrap_or_else(|| std::ptr::NonNull::<C>::dangling().as_ptr());

        (ptr, self.capacity_rows())
    }

    /// Returns pointers used by mutable query fetches.
    ///
    /// # Safety
    ///
    /// The caller must hold exclusive value and tick access to the column and
    /// keep it structurally unchanged while the pointers are live.
    pub(crate) unsafe fn write_ptrs(&self) -> (*mut C, *mut ComponentTicks, usize) {
        let value_ptr = self
            .values
            .first()
            .map(|page| page.as_ptr().cast::<C>().cast_mut())
            .unwrap_or_else(|| std::ptr::NonNull::<C>::dangling().as_ptr());

        let ticks_ptr = self
            .ticks
            .first()
            .map(|page| page.as_ptr().cast::<ComponentTicks>().cast_mut())
            .unwrap_or_else(|| std::ptr::NonNull::<ComponentTicks>::dangling().as_ptr());

        (value_ptr, ticks_ptr, self.capacity_rows())
    }

    /// Returns raw pointers for a single row.
    ///
    /// # Safety
    ///
    /// `row` must be initialized, and the caller must hold exclusive access to
    /// both slots.
    pub(crate) unsafe fn row_mut_ptrs(&self, row: usize) -> (*mut C, *mut ComponentTicks) {
        (unsafe { self.value_ptr(row) }, unsafe { self.ticks_ptr_at(row) })
    }
}

impl<C: Component> Column for ColumnPaged<C> {
    fn len(&self) -> usize {
        self.len
    }

    fn add_page(&mut self) {
        self.values.push(uninit_array());
        self.ticks.push(uninit_array());
    }

    unsafe fn drop_page_mask(&mut self, page: usize, mut mask: u64) {
        debug_assert!(page < self.values.len());
        debug_assert!(page < self.ticks.len());

        let removed = mask.count_ones() as usize;
        let values = &mut self.values[page];
        let ticks = &mut self.ticks[page];

        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            mask &= mask - 1; // Clears the lowest set bit just consumed.

            unsafe {
                values[offset].assume_init_drop();
                ticks[offset].assume_init_drop();
            }
        }

        self.len = self
            .len
            .checked_sub(removed)
            .expect("column batch removal exceeded its live length");
    }

    unsafe fn drop_row(&mut self, row: usize) {
        let PageRow { page, offset } = Self::slot(row);

        unsafe {
            self.drop_page_mask(page, 1_u64 << offset);
        }
    }

    unsafe fn move_row_within(&mut self, src_row: usize, dst_row: usize) {
        if src_row == dst_row {
            return;
        }

        let (value, ticks) = unsafe { self.read_move(src_row) };
        unsafe { self.write_existing_value(dst_row, value, ticks) };
    }

    unsafe fn move_rows_to_and_backfill(
        &mut self,
        destination: &mut dyn Column,
        rows: &[BatchTransitionRow],
    ) {
        // The concrete type is resolved once for the whole column batch.
        let destination = destination
            .as_any_mut()
            .downcast_mut::<ColumnPaged<C>>()
            .expect("batch transition paired incompatible column types");

        for movement in rows {
            let (value, ticks) = unsafe { self.read_move(movement.source_row) };

            unsafe {
                destination.write_existing_value(movement.destination_row, value, ticks);

                self.backfill_from(movement.source_row, movement.backfill_row);
            }
        }
    }

    unsafe fn drop_rows_and_backfill(&mut self, rows: &[BatchTransitionRow]) {
        for movement in rows {
            let slot = Self::slot(movement.source_row);

            unsafe {
                self.values[slot.page][slot.offset].assume_init_drop();
                self.ticks[slot.page][slot.offset].assume_init_drop();
            }

            self.len -= 1;

            unsafe {
                self.backfill_from(movement.source_row, movement.backfill_row);
            }
        }
    }

    unsafe fn push_move_from_ptr_at(&mut self, row: usize, src: *mut u8, change_tick: ChangeTick) {
        debug_assert!(!src.is_null());
        debug_assert!((src as usize).is_multiple_of(core::mem::align_of::<C>()));

        let value = unsafe { core::ptr::read(src as *mut C) };
        unsafe { self.write_existing_value(row, value, ComponentTicks::new(change_tick)) };
    }

    unsafe fn write_move_from_ptr(&mut self, row: usize, src: *mut u8, change_tick: ChangeTick) {
        debug_assert!(!src.is_null());
        debug_assert!((src as usize).is_multiple_of(core::mem::align_of::<C>()));

        let value = unsafe { core::ptr::read(src as *mut C) };
        unsafe { *self.value_ptr(row) = value };

        self.note_write_access(change_tick);
        (unsafe { *self.ticks_ptr_at(row) }).mark_changed(change_tick);
    }

    unsafe fn move_row_to_and_backfill(
        &mut self,
        row: usize,
        destination: &mut dyn Column,
        destination_row: usize,
        backfill_row: Option<usize>,
    ) {
        let destination = destination
            .as_any_mut()
            .downcast_mut::<ColumnPaged<C>>()
            .expect("transition plan paired incompatible column types");

        let (value, ticks) = unsafe { self.read_move(row) };

        unsafe {
            destination.write_existing_value(destination_row, value, ticks);
        }

        unsafe {
            self.backfill_from(row, backfill_row);
        }
    }

    unsafe fn drop_row_and_backfill(&mut self, row: usize, backfill_row: Option<usize>) {
        let slot = Self::slot(row);

        unsafe {
            self.values[slot.page][slot.offset].assume_init_drop();
            self.ticks[slot.page][slot.offset].assume_init_drop();
        }

        self.len -= 1;

        unsafe {
            self.backfill_from(row, backfill_row);
        }
    }

    fn rebuild_dense_metadata(&mut self, live_len: usize) {
        let needed_pages = pages_for_rows(live_len);

        self.values.truncate(needed_pages);
        self.ticks.truncate(needed_pages);
        self.len = live_len;
    }

    fn ticks_ptr(&self) -> *const ComponentTicks {
        self.ticks
            .first()
            .map(|page| page.as_ptr().cast::<ComponentTicks>())
            .unwrap_or_else(|| std::ptr::NonNull::<ComponentTicks>::dangling().as_ptr())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn last_write_access(&self) -> ChangeTick {
        ChangeTick::from_u64(self.last_write_access.load(Ordering::Relaxed))
    }

    #[inline]
    fn note_write_access(&self, tick: ChangeTick) {
        self.last_write_access
            .fetch_max(tick.as_u64(), Ordering::Relaxed);
    }
}

/// Monomorphized zero-argument constructor for one erased column.
type ColumnFn = fn() -> Box<dyn Column>;

/// Factory that creates empty table columns for registered component ids.
pub(crate) struct ColumnFactory {
    constructors: Vec<Option<ColumnFn>>,
}

impl Default for ColumnFactory {
    /// Creates a factory with no registered constructors.
    fn default() -> Self {
        Self::new()
    }
}

impl ColumnFactory {
    /// Creates an empty factory with no registered component constructors.
    pub(crate) fn new() -> Self {
        Self {
            constructors: Vec::new(),
        }
    }

    /// Registers the `ColumnPaged<T>` constructor at a component index.
    ///
    /// Each call monomorphizes a distinct constructor carrying the concrete
    /// type, so the factory itself stays fully type-erased.
    pub(crate) fn register<C: Component>(&mut self, component_index: usize) {
        fn make<C: Component>() -> Box<dyn Column> {
            Box::new(ColumnPaged::<C>::default())
        }

        if self.constructors.len() <= component_index {
            self.constructors.resize(component_index + 1, None);
        }

        self.constructors[component_index] = Some(make::<C>);
    }

    /// Creates a fresh column for a registered component index.
    pub(crate) fn make(&self, component_index: usize) -> Option<Box<dyn Column>> {
        self.constructors
            .get(component_index)
            .and_then(|f| f.map(|m| m()))
    }

    /// Returns whether a column constructor exists for `component_index`.
    pub(crate) fn is_registered(&self, component_index: usize) -> bool {
        self.constructors
            .get(component_index)
            .map(|constructor| constructor.is_some())
            .unwrap_or(false)
    }
}
