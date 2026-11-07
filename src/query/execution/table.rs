//! Table-only query unit preparation.

use crate::{
    component::archetype::Archetype,
    entity::Entity,
    query::{
        execution::{
            work::{ArchetypePageWork, PreparedSparseRows, SparseRowsWork},
            SPARSE_GATHER_PREFETCH_DISTANCE,
        },
        fetch::{mixed as f_mixed, table as f_table},
        filter::QueryFilter,
    },
};

use super::{
    change_filter_skips_unit, matches_filter, matches_filter_sparse_driver, panic_fetch_error,
    LiveRowCursor, QueryReadContext, QueryWriteContext,
};

/// Prepared table fetches for the currently active read-only work unit.
pub(crate) struct PreparedReadUnit<'w, D, F>
where
    D: f_table::ReadQueryData,
    F: QueryFilter,
{
    rows: LiveRowCursor,
    data: D::TableFetch<'w>,
    filter: F::Fetch<'w>,
}

impl<'w, D, F> PreparedReadUnit<'w, D, F>
where
    D: f_table::ReadQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the unit, or `None` when drained.
    ///
    /// When no per-row filter applies and the data never observes the entity,
    /// rows stream through `next_row` with `Entity::INVALID`, dropping the
    /// entity load from the hot loop.
    ///
    /// # Safety
    ///
    /// The prepared fetches must stay valid: no structural mutation may change
    /// the page range, entity slice, or table columns while items are
    /// produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::ReadQueryData>::Item<'w>> {
        if !APPLY_FILTER && !<D as f_table::ReadQueryData>::USES_ENTITY {
            if let Some(row) = unsafe { self.rows.next_row() } {
                return Some(unsafe {
                    <D as f_table::ReadQueryData>::fetch(&mut self.data, row, Entity::INVALID)
                });
            }

            return None;
        }

        while let Some((row, entity)) = unsafe { self.rows.next() } {
            if APPLY_FILTER && !unsafe { matches_filter::<F>(&mut self.filter, row, entity) } {
                continue;
            }

            return Some(unsafe {
                <D as f_table::ReadQueryData>::fetch(&mut self.data, row, entity)
            });
        }

        None
    }

    /// Drains the remaining items of the unit through `func`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::next`].
    pub(crate) unsafe fn for_each_remaining<Func, const APPLY_FILTER: bool>(
        &mut self,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        while let Some(item) = unsafe { self.next::<APPLY_FILTER>() } {
            func(item);
        }
    }
}

/// Prepared table fetches for a sparse-driver unit already resolved to rows.
pub(crate) struct PreparedSparseReadUnit<'w, D, F>
where
    D: f_table::ReadQueryData,
    F: QueryFilter,
{
    rows: PreparedSparseRows,
    data: D::TableFetch<'w>,
    filter: F::Fetch<'w>,
}

impl<'w, D, F> PreparedSparseReadUnit<'w, D, F>
where
    D: f_table::ReadQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the sparse unit, or `None` when drained.
    ///
    /// Each candidate prefetches the table row of the candidate
    /// `SPARSE_GATHER_PREFETCH_DISTANCE` positions ahead when the data supports
    /// row prefetch.
    ///
    /// # Safety
    ///
    /// The prepared fetches and the resolved row slice must stay valid: no
    /// structural mutation may change the bucket lane or the table columns
    /// while items are produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::ReadQueryData>::Item<'w>> {
        while let Some(candidate) = unsafe { self.rows.next() } {
            let row = candidate.row as usize;
            let entity = candidate.entity;

            if <D as f_table::ReadQueryData>::HAS_ROW_PREFETCH {
                if let Some(future) = unsafe { self.rows.peek(SPARSE_GATHER_PREFETCH_DISTANCE) } {
                    unsafe { <D as f_table::ReadQueryData>::prefetch_row(&self.data, future.row as usize) };
                }
            }

            if APPLY_FILTER
                && !unsafe {
                    matches_filter_sparse_driver::<F>(
                        &mut self.filter,
                        row,
                        entity,
                        candidate.in_page_jumps,
                    )
                }
            {
                continue;
            }

            return Some(unsafe {
                <D as f_table::ReadQueryData>::fetch(&mut self.data, row, entity)
            });
        }

        None
    }

    /// Drains the remaining items of the unit through `func`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::next`].
    pub(crate) unsafe fn for_each_remaining<Func, const APPLY_FILTER: bool>(
        &mut self,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        while let Some(item) = unsafe { self.next::<APPLY_FILTER>() } {
            func(item);
        }
    }
}

impl<'w, 's, D, F> QueryReadContext<'w, 's, D, F>
where
    D: f_table::ReadQueryData,
    F: QueryFilter,
{
    /// Prepares the row cursor and table fetches for one archetype page unit.
    ///
    /// A provably cold change-filter column yields an empty page range, so the
    /// unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// `unit` must come from validated work built against this context's
    /// managers, and no structural mutation may run while the prepared unit is
    /// alive.
    pub(crate) unsafe fn prepare_table_read_unit(
        &self,
        unit: ArchetypePageWork,
    ) -> PreparedReadUnit<'w, D, F> {
        debug_assert!(unit.page_count() > 0);
        debug_assert!(unit.estimated_rows() > 0);

        let archetypes = unsafe { &*self.archetypes };
        let arch = archetypes.get_archetype(unit.archetype_id).unwrap();
        let (entities, entity_len) = arch.entity_ptrs(unit.storage);

        if change_filter_skips_unit::<F>(self.filter_state, arch, unit.storage, self.tick_window) {
            // Cold column: no tick inside can be newer than the reader's
            // window, so the unit yields nothing. An empty page range skips
            // every row without touching the data columns.
            return PreparedReadUnit {
                rows: LiveRowCursor::new_range(
                    arch as *const Archetype,
                    unit.storage,
                    entities,
                    entity_len,
                    0,
                    0,
                ),
                data: unsafe { <D as f_table::ReadQueryData>::init_fetch(
                    self.data_state,
                    arch as *const Archetype,
                    unit.storage,
                )
                .unwrap_or_else(|err| panic_fetch_error::<D>("table data", unit.archetype_id, err)) },
                filter: unsafe { F::init_fetch(
                    self.filter_state,
                    arch as *const Archetype,
                    unit.storage,
                    self.sparse_sets,
                    self.tick_window,
                )
                .unwrap_or_else(|err| panic_fetch_error::<F>("filter", unit.archetype_id, err)) },
            };
        }

        PreparedReadUnit {
            rows: LiveRowCursor::new_range(
                arch as *const Archetype,
                unit.storage,
                entities,
                entity_len,
                unit.page_start(),
                unit.page_end(),
            ),
            data: unsafe { <D as f_table::ReadQueryData>::init_fetch(
                self.data_state,
                arch as *const Archetype,
                unit.storage,
            )
            .unwrap_or_else(|err| panic_fetch_error::<D>("table data", unit.archetype_id, err)) },
            filter: unsafe { F::init_fetch(
                self.filter_state,
                arch as *const Archetype,
                unit.storage,
                self.sparse_sets,
                self.tick_window,
            )
            .unwrap_or_else(|err| panic_fetch_error::<F>("filter", unit.archetype_id, err)) },
        }
    }

    /// Prepares the resolved row cursor and table fetches for one sparse-driver
    /// unit.
    ///
    /// A provably cold change-filter column swaps the resolved rows for an
    /// empty slice, so the unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::prepare_table_read_unit`].
    pub(crate) unsafe fn prepare_table_sparse_read_unit(
        &self,
        unit: SparseRowsWork,
    ) -> PreparedSparseReadUnit<'w, D, F> {
        debug_assert!(unit.row_count() > 0);

        let archetypes = unsafe { &*self.archetypes };
        let arch = archetypes.get_archetype(unit.archetype_id).unwrap();
        let storage = unit.storage();

        let rows =
            if change_filter_skips_unit::<F>(self.filter_state, arch, storage, self.tick_window) {
                // Cold table column: no row the driver lane yields can match the
                // change filter, so the unit iterates nothing.
                PreparedSparseRows::indexed(&[])
            } else {
                unsafe { self.prepare_sparse_rows(unit) }
            };

        PreparedSparseReadUnit {
            rows,
            data: unsafe { <D as f_table::ReadQueryData>::init_fetch(
                self.data_state,
                arch as *const Archetype,
                storage,
            )
            .unwrap_or_else(|err| panic_fetch_error::<D>("table data", unit.archetype_id, err)) },
            filter: unsafe { F::init_sparse_driver_fetch(
                self.filter_state,
                arch as *const Archetype,
                storage,
                self.sparse_sets,
                self.tick_window,
                unit.driver,
            )
            .unwrap_or_else(|err| panic_fetch_error::<F>("filter", unit.archetype_id, err)) },
        }
    }
}

/// Prepared table fetches for the currently active mutable work unit.
pub(crate) struct PreparedWriteUnit<'w, D, F>
where
    D: f_table::WriteQueryData,
    F: QueryFilter,
{
    rows: LiveRowCursor,
    data: D::TableFetch<'w>,
    filter: F::Fetch<'w>,
}

impl<'w, D, F> PreparedWriteUnit<'w, D, F>
where
    D: f_table::WriteQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the unit, or `None` when drained.
    ///
    /// When no per-row filter applies and the data never observes the entity,
    /// rows stream through `next_row` with `Entity::INVALID`, dropping the
    /// entity load from the hot loop.
    ///
    /// # Safety
    ///
    /// The prepared fetches must stay valid: no structural mutation may change
    /// the page range, entity slice, or table columns while items are
    /// produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::WriteQueryData>::Item<'w>> {
        if !APPLY_FILTER && !<D as f_table::WriteQueryData>::USES_ENTITY {
            if let Some(row) = unsafe { self.rows.next_row() } {
                return Some(unsafe {
                    <D as f_table::WriteQueryData>::fetch(&mut self.data, row, Entity::INVALID)
                });
            }

            return None;
        }

        while let Some((row, entity)) = unsafe { self.rows.next() } {
            if APPLY_FILTER && !unsafe { matches_filter::<F>(&mut self.filter, row, entity) } {
                continue;
            }

            return Some(unsafe {
                <D as f_table::WriteQueryData>::fetch(&mut self.data, row, entity)
            });
        }

        None
    }

    /// Drains the remaining items of the unit through `func`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::next`].
    pub(crate) unsafe fn for_each_remaining<Func, const APPLY_FILTER: bool>(
        &mut self,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        while let Some(item) = unsafe { self.next::<APPLY_FILTER>() } {
            func(item);
        }
    }
}

/// Prepared mutable table fetches for a sparse-driver unit already resolved to rows.
pub(crate) struct PreparedSparseWriteUnit<'w, D, F>
where
    D: f_table::WriteQueryData,
    F: QueryFilter,
{
    rows: PreparedSparseRows,
    data: D::TableFetch<'w>,
    filter: F::Fetch<'w>,
}

impl<'w, D, F> PreparedSparseWriteUnit<'w, D, F>
where
    D: f_table::WriteQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the sparse unit, or `None` when drained.
    ///
    /// Each candidate prefetches the table row of the candidate
    /// `SPARSE_GATHER_PREFETCH_DISTANCE` positions ahead when the data supports
    /// row prefetch.
    ///
    /// # Safety
    ///
    /// The prepared fetches and the resolved row slice must stay valid: no
    /// structural mutation may change the bucket lane or the table columns
    /// while items are produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::WriteQueryData>::Item<'w>> {
        while let Some(candidate) = unsafe { self.rows.next() } {
            let row = candidate.row as usize;
            let entity = candidate.entity;

            if <D as f_table::WriteQueryData>::HAS_ROW_PREFETCH {
                if let Some(future) = unsafe { self.rows.peek(SPARSE_GATHER_PREFETCH_DISTANCE) } {
                    unsafe { <D as f_table::WriteQueryData>::prefetch_row(&self.data, future.row as usize) };
                }
            }

            if APPLY_FILTER
                && !unsafe {
                    matches_filter_sparse_driver::<F>(
                        &mut self.filter,
                        row,
                        entity,
                        candidate.in_page_jumps,
                    )
                }
            {
                continue;
            }

            return Some(unsafe {
                <D as f_table::WriteQueryData>::fetch(&mut self.data, row, entity)
            });
        }

        None
    }

    /// Drains the remaining items of the unit through `func`.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::next`].
    pub(crate) unsafe fn for_each_remaining<Func, const APPLY_FILTER: bool>(
        &mut self,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        while let Some(item) = unsafe { self.next::<APPLY_FILTER>() } {
            func(item);
        }
    }
}

impl<'w, 's, D, F> QueryWriteContext<'w, 's, D, F>
where
    D: f_table::WriteQueryData,
    F: QueryFilter,
{
    /// Prepares the row cursor and table fetches for one archetype page unit.
    ///
    /// A provably cold change-filter column yields an empty page range, so the
    /// unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// `unit` must come from validated work built against this context's
    /// managers, and no structural mutation may run while the prepared unit is
    /// alive.
    pub(crate) unsafe fn prepare_table_write_unit(
        &self,
        unit: ArchetypePageWork,
    ) -> PreparedWriteUnit<'w, D, F> {
        debug_assert!(unit.page_count() > 0);
        debug_assert!(unit.estimated_rows() > 0);

        let archetypes = unsafe { &*self.archetypes };
        let arch = archetypes.get_archetype(unit.archetype_id).unwrap();
        let (entities, entity_len) = arch.entity_ptrs(unit.storage);

        if change_filter_skips_unit::<F>(self.filter_state, arch, unit.storage, self.tick_window) {
            // Cold column: no tick inside can be newer than the reader's
            // window, so the unit yields nothing.
            return PreparedWriteUnit {
                rows: LiveRowCursor::new_range(
                    arch as *const Archetype,
                    unit.storage,
                    entities,
                    entity_len,
                    0,
                    0,
                ),
                data: unsafe { <D as f_table::WriteQueryData>::init_fetch(
                    self.data_state,
                    arch as *const Archetype,
                    unit.storage,
                    self.tick_window.current_tick,
                )
                .unwrap_or_else(|err| panic_fetch_error::<D>("table data", unit.archetype_id, err)) },
                filter: unsafe { F::init_fetch(
                    self.filter_state,
                    arch as *const Archetype,
                    unit.storage,
                    self.sparse_sets,
                    self.tick_window,
                )
                .unwrap_or_else(|err| panic_fetch_error::<F>("filter", unit.archetype_id, err)) },
            };
        }

        PreparedWriteUnit {
            rows: LiveRowCursor::new_range(
                arch as *const Archetype,
                unit.storage,
                entities,
                entity_len,
                unit.page_start(),
                unit.page_end(),
            ),
            data: unsafe { <D as f_table::WriteQueryData>::init_fetch(
                self.data_state,
                arch as *const Archetype,
                unit.storage,
                self.tick_window.current_tick,
            )
            .unwrap_or_else(|err| panic_fetch_error::<D>("table data", unit.archetype_id, err)) },
            filter: unsafe { F::init_fetch(
                self.filter_state,
                arch as *const Archetype,
                unit.storage,
                self.sparse_sets,
                self.tick_window,
            )
            .unwrap_or_else(|err| panic_fetch_error::<F>("filter", unit.archetype_id, err)) },
        }
    }

    /// Prepares the resolved row cursor and table fetches for one sparse-driver
    /// unit.
    ///
    /// A provably cold change-filter column swaps the resolved rows for an
    /// empty slice, so the unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::prepare_table_write_unit`].
    pub(crate) unsafe fn prepare_table_sparse_write_unit(
        &self,
        unit: SparseRowsWork,
    ) -> PreparedSparseWriteUnit<'w, D, F> {
        debug_assert!(unit.row_count() > 0);

        let archetypes = unsafe { &*self.archetypes };
        let arch = archetypes.get_archetype(unit.archetype_id).unwrap();
        let storage = unit.storage();

        let rows =
            if change_filter_skips_unit::<F>(self.filter_state, arch, storage, self.tick_window) {
                // Cold table column: no row the driver lane yields can match the
                // change filter, so the unit iterates nothing.
                PreparedSparseRows::indexed(&[])
            } else {
                unsafe { self.prepare_sparse_rows(unit) }
            };

        PreparedSparseWriteUnit {
            rows,
            data: unsafe { <D as f_table::WriteQueryData>::init_fetch(
                self.data_state,
                arch as *const Archetype,
                storage,
                self.tick_window.current_tick,
            )
            .unwrap_or_else(|err| panic_fetch_error::<D>("table data", unit.archetype_id, err)) },
            filter: unsafe { F::init_sparse_driver_fetch(
                self.filter_state,
                arch as *const Archetype,
                storage,
                self.sparse_sets,
                self.tick_window,
                unit.driver,
            )
            .unwrap_or_else(|err| panic_fetch_error::<F>("filter", unit.archetype_id, err)) },
        }
    }
}
