//! Mixed table/sparse query unit preparation.

use crate::{
    component::archetype::Archetype,
    query::{
        execution::{
            work::{ArchetypePageWork, PreparedSparseRows, SparseRowsWork},
            SPARSE_GATHER_PREFETCH_DISTANCE,
        },
        fetch::{mixed as f_mixed, ReadQueryData, WriteQueryData},
        filter::QueryFilter,
    },
};

use super::{
    change_filter_skips_unit, matches_filter, matches_filter_sparse_driver, panic_fetch_error,
    LiveRowCursor, QueryReadContext, QueryWriteContext,
};

/// Prepared mixed fetches for the currently active read-only work unit.
pub(crate) struct PreparedReadUnit<'w, D, F>
where
    D: f_mixed::ReadQueryData,
    F: QueryFilter,
{
    rows: LiveRowCursor,
    data: D::Fetch<'w>,
    filter: F::Fetch<'w>,
}

impl<'w, D, F> PreparedReadUnit<'w, D, F>
where
    D: f_mixed::ReadQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the unit, or `None` when drained.
    ///
    /// Rows the mixed fetch rejects, such as an optional component that is
    /// absent on the row, are skipped without surfacing to the caller.
    ///
    /// # Safety
    ///
    /// The prepared fetches must stay valid: no structural mutation may change
    /// the page range, entity slice, table columns, or sparse sets while items
    /// are produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::ReadQueryData>::Item<'w>> {
        while let Some((row, entity)) = unsafe { self.rows.next() } {
            if APPLY_FILTER && !unsafe { matches_filter::<F>(&mut self.filter, row, entity) } {
                continue;
            }

            if let Some(item) = unsafe { D::fetch(&mut self.data, row, entity) } {
                return Some(item);
            }
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

/// Prepared mixed fetches for a sparse-driver unit already resolved to rows.
pub(crate) struct PreparedSparseReadUnit<'w, D, F>
where
    D: f_mixed::ReadQueryData,
    F: QueryFilter,
{
    rows: PreparedSparseRows,
    data: D::Fetch<'w>,
    filter: F::Fetch<'w>,
    /// Cached `has_row_prefetch` result of the initialized fetch, hoisted out
    /// of the candidate loop.
    prefetch: bool,
}

impl<'w, D, F> PreparedSparseReadUnit<'w, D, F>
where
    D: f_mixed::ReadQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the sparse unit, or `None` when drained.
    ///
    /// Each candidate prefetches the row and entity of the candidate
    /// `SPARSE_GATHER_PREFETCH_DISTANCE` positions ahead when the prepared
    /// fetch supports it; rows the fetch rejects are skipped.
    ///
    /// # Safety
    ///
    /// The prepared fetches and the resolved row slice must stay valid: no
    /// structural mutation may change the bucket lane, table columns, or
    /// sparse sets while items are produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::ReadQueryData>::Item<'w>> {
        while let Some(candidate) = unsafe { self.rows.next() } {
            let row = candidate.row as usize;
            let entity = candidate.entity;

            if self.prefetch {
                if let Some(future) = unsafe { self.rows.peek(SPARSE_GATHER_PREFETCH_DISTANCE) } {
                    unsafe { <D as f_mixed::ReadQueryData>::prefetch_sparse_driver_row(
                        &self.data,
                        future.row as usize,
                        future.entity,
                    ) };
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

            if let Some(item) = unsafe {
                <D as f_mixed::ReadQueryData>::fetch_sparse_driver(
                    &mut self.data,
                    row,
                    entity,
                    candidate.in_page_jumps,
                )
            } {
                return Some(item);
            }
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
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Prepares the row cursor and mixed fetches for one archetype page unit.
    ///
    /// A provably cold change-filter column yields an empty page range, so the
    /// unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// `unit` must come from validated work built against this context's
    /// managers, and no structural mutation may run while the prepared unit is
    /// alive.
    pub(crate) unsafe fn prepare_mixed_read_unit(
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
                data: unsafe { <D as f_mixed::ReadQueryData>::init_fetch(
                    self.data_state,
                    arch as *const Archetype,
                    self.sparse_sets,
                    unit.storage,
                )
                .unwrap_or_else(|err| panic_fetch_error::<D>("data", unit.archetype_id, err)) },
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
            data: unsafe { <D as f_mixed::ReadQueryData>::init_fetch(
                self.data_state,
                arch as *const Archetype,
                self.sparse_sets,
                unit.storage,
            )
            .unwrap_or_else(|err| panic_fetch_error::<D>("data", unit.archetype_id, err)) },
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

    /// Prepares the resolved row cursor and mixed fetches for one sparse-driver
    /// unit, caching the fetch's prefetch capability.
    ///
    /// A provably cold change-filter column swaps the resolved rows for an
    /// empty slice, so the unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::prepare_mixed_read_unit`].
    pub(crate) unsafe fn prepare_mixed_sparse_read_unit(
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

        let data = unsafe { <D as f_mixed::ReadQueryData>::init_sparse_driver_fetch(
            self.data_state,
            arch as *const Archetype,
            self.sparse_sets,
            storage,
            unit.driver,
        )
        .unwrap_or_else(|err| panic_fetch_error::<D>("data", unit.archetype_id, err)) };

        let filter = unsafe { F::init_sparse_driver_fetch(
            self.filter_state,
            arch as *const Archetype,
            storage,
            self.sparse_sets,
            self.tick_window,
            unit.driver,
        )
        .unwrap_or_else(|err| panic_fetch_error::<F>("filter", unit.archetype_id, err)) };

        let prefetch = <D as f_mixed::ReadQueryData>::has_row_prefetch(&data);

        PreparedSparseReadUnit {
            rows,
            data,
            filter,
            prefetch,
        }
    }
}

/// Prepared mixed fetches for the currently active mutable work unit.
pub(crate) struct PreparedWriteUnit<'w, D, F>
where
    D: f_mixed::WriteQueryData,
    F: QueryFilter,
{
    rows: LiveRowCursor,
    data: D::Fetch<'w>,
    filter: F::Fetch<'w>,
}

impl<'w, D, F> PreparedWriteUnit<'w, D, F>
where
    D: f_mixed::WriteQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the unit, or `None` when drained.
    ///
    /// Rows the mixed fetch rejects, such as an optional component that is
    /// absent on the row, are skipped without surfacing to the caller.
    ///
    /// # Safety
    ///
    /// The prepared fetches must stay valid: no structural mutation may change
    /// the page range, entity slice, table columns, or sparse sets while items
    /// are produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::WriteQueryData>::Item<'w>> {
        while let Some((row, entity)) = unsafe { self.rows.next() } {
            if APPLY_FILTER && !unsafe { matches_filter::<F>(&mut self.filter, row, entity) } {
                continue;
            }

            if let Some(item) = unsafe { D::fetch(&mut self.data, row, entity) } {
                return Some(item);
            }
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

/// Prepared mutable mixed fetches for a sparse-driver unit already resolved to rows.
pub(crate) struct PreparedSparseWriteUnit<'w, D, F>
where
    D: f_mixed::WriteQueryData,
    F: QueryFilter,
{
    rows: PreparedSparseRows,
    data: D::Fetch<'w>,
    filter: F::Fetch<'w>,
}

impl<'w, D, F> PreparedSparseWriteUnit<'w, D, F>
where
    D: f_mixed::WriteQueryData,
    F: QueryFilter,
{
    /// Returns the next item of the sparse unit, or `None` when drained.
    ///
    /// Rows the fetch rejects are skipped; the mutable path performs no gather
    /// prefetching.
    ///
    /// # Safety
    ///
    /// The prepared fetches and the resolved row slice must stay valid: no
    /// structural mutation may change the bucket lane, table columns, or
    /// sparse sets while items are produced.
    pub(crate) unsafe fn next<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::WriteQueryData>::Item<'w>> {
        while let Some(candidate) = unsafe { self.rows.next() } {
            let row = candidate.row as usize;
            let entity = candidate.entity;
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

            if let Some(item) = unsafe {
                <D as f_mixed::WriteQueryData>::fetch_sparse_driver(
                    &mut self.data,
                    row,
                    entity,
                    candidate.in_page_jumps,
                )
            } {
                return Some(item);
            }
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
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Prepares the row cursor and mixed fetches for one archetype page unit.
    ///
    /// A provably cold change-filter column yields an empty page range, so the
    /// unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// `unit` must come from validated work built against this context's
    /// managers, and no structural mutation may run while the prepared unit is
    /// alive.
    pub(crate) unsafe fn prepare_mixed_write_unit(
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
                data: unsafe { <D as f_mixed::WriteQueryData>::init_fetch(
                    self.data_state,
                    arch as *const Archetype,
                    self.sparse_sets,
                    unit.storage,
                    self.tick_window.current_tick,
                )
                .unwrap_or_else(|err| panic_fetch_error::<D>("data", unit.archetype_id, err)) },
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
            data: unsafe { <D as f_mixed::WriteQueryData>::init_fetch(
                self.data_state,
                arch as *const Archetype,
                self.sparse_sets,
                unit.storage,
                self.tick_window.current_tick,
            )
            .unwrap_or_else(|err| panic_fetch_error::<D>("data", unit.archetype_id, err)) },
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

    /// Prepares the resolved row cursor and mixed fetches for one sparse-driver
    /// unit.
    ///
    /// A provably cold change-filter column swaps the resolved rows for an
    /// empty slice, so the unit drains without touching any data column.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::prepare_mixed_write_unit`].
    pub(crate) unsafe fn prepare_mixed_sparse_write_unit(
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
            data: unsafe { <D as f_mixed::WriteQueryData>::init_sparse_driver_fetch(
                self.data_state,
                arch as *const Archetype,
                self.sparse_sets,
                storage,
                self.tick_window.current_tick,
                unit.driver,
            )
            .unwrap_or_else(|err| panic_fetch_error::<D>("data", unit.archetype_id, err)) },
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
