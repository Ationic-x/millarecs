//! Table-only iterator implementation.
//!
//! This path assumes the data fetch never fails for rows in a matched archetype.
//! It is used only after `QueryState` has proven that the data part is
//! archetypal.

use crate::query::{
    execution::{
        table::{
            PreparedReadUnit, PreparedSparseReadUnit, PreparedSparseWriteUnit, PreparedWriteUnit,
        },
        work::{QueryWorkCursor, QueryWorkUnit},
        Path, QueryReadContext, QueryWriteContext,
    },
    fetch::{mixed as f_mixed, table as f_table},
    filter::QueryFilter,
};

/// Read-only iterator for the table fast path.
///
/// Selected by `super::QueryIter` when the data fetch is fully archetypal.
/// Rows are produced through prepared units pulled lazily from the work
/// cursor: a unit is prepared only once both unit slots are empty, then
/// drained row by row, so preparation cost is paid per unit instead of per
/// row.
pub struct QueryIter<'w, 's, D, F>
where
    D: f_table::ReadQueryData,
    F: QueryFilter,
{
    context: QueryReadContext<'w, 's, D, F>,
    work: QueryWorkCursor<'s>,
    /// Prepared archetype-pages unit being drained, if any.
    current: Option<PreparedReadUnit<'w, D, F>>,
    /// Prepared sparse-driver unit being drained, if any; at most one of
    /// `current` and `current_sparse` holds a unit at a time.
    current_sparse: Option<PreparedSparseReadUnit<'w, D, F>>,
}

impl<'w, 's, D, F> QueryIter<'w, 's, D, F>
where
    D: f_table::ReadQueryData,
    F: QueryFilter,
{
    /// Creates a new read-only query iterator.
    pub(crate) fn new(context: QueryReadContext<'w, 's, D, F>, work: QueryWorkCursor<'s>) -> Self {
        Self {
            context,
            work,
            current: None,
            current_sparse: None,
        }
    }

    /// Pulls the next work unit from the cursor and prepares it into the
    /// matching slot.
    ///
    /// Returns whether a unit was prepared; both slots stay `None` when the
    /// cursor is exhausted.
    ///
    /// # Safety
    ///
    /// The cursor must only yield units validated against this context's
    /// storage managers, and no structural mutation may run while the
    /// prepared unit is alive.
    unsafe fn prepare_next_unit(&mut self) -> bool {
        let Some(unit) = self.work.next_unit() else {
            return false;
        };

        match unit {
            QueryWorkUnit::ArchetypePages(unit) => {
                self.current = Some(unsafe { self.context.prepare_table_read_unit(unit) });
            }
            QueryWorkUnit::SparseRows(unit) => {
                self.current_sparse =
                    Some(unsafe { self.context.prepare_table_sparse_read_unit(unit) });
            }
        }

        true
    }

    /// Visits all remaining rows, optionally applying per-row filters.
    ///
    /// Units already prepared by a partially driven `next_impl` chain are
    /// drained first; fresh units then stream straight from the work cursor.
    pub(crate) fn for_each<Func, const APPLY_FILTER: bool>(mut self, mut func: Func)
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        if let Some(mut current) = self.current.take() {
            unsafe { current.for_each_remaining::<Func, APPLY_FILTER>(&mut func) };
        }

        if let Some(unit) = self.current_sparse.take() {
            unsafe {
                let mut unit = unit;
                unit.for_each_remaining::<Func, APPLY_FILTER>(&mut func);
            }
        }

        while let Some(unit) = self.work.next_unit() {
            unsafe {
                self.context
                    .for_each_read_unit::<Func, APPLY_FILTER, { Path::TABLE }>(unit, &mut func);
            }
        }
    }

    /// Returns the next row, optionally applying per-row filters and
    /// preparing further work units lazily until the cursor is exhausted.
    pub(crate) fn next_impl<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::ReadQueryData>::Item<'w>> {
        loop {
            if self.current.is_none()
                && self.current_sparse.is_none()
                && !unsafe { self.prepare_next_unit() }
            {
                return None;
            }

            if let Some(unit) = self.current_sparse.as_mut() {
                if let Some(item) = unsafe { unit.next::<APPLY_FILTER>() } {
                    return Some(item);
                }

                self.current_sparse = None;
                continue;
            }

            let current = self.current.as_mut().unwrap();

            if let Some(item) = unsafe { current.next::<APPLY_FILTER>() } {
                return Some(item);
            }

            self.current = None;
        }
    }
}

/// Mutable iterator for the table fast path.
///
/// Selected by `super::QueryIterMut` when the data fetch is fully
/// archetypal. Row production follows the read-only variant: units are
/// prepared lazily from the work cursor and drained one at a time.
pub struct QueryIterMut<'w, 's, D, F>
where
    D: f_table::WriteQueryData,
    F: QueryFilter,
{
    context: QueryWriteContext<'w, 's, D, F>,
    work: QueryWorkCursor<'s>,
    /// Prepared archetype-pages unit being drained, if any.
    current: Option<PreparedWriteUnit<'w, D, F>>,
    /// Prepared sparse-driver unit being drained, if any; at most one of
    /// `current` and `current_sparse` holds a unit at a time.
    current_sparse: Option<PreparedSparseWriteUnit<'w, D, F>>,
}

impl<'w, 's, D, F> QueryIterMut<'w, 's, D, F>
where
    D: f_table::WriteQueryData,
    F: QueryFilter,
{
    /// Creates a new mutable query iterator.
    pub(crate) fn new(context: QueryWriteContext<'w, 's, D, F>, work: QueryWorkCursor<'s>) -> Self {
        Self {
            context,
            work,
            current: None,
            current_sparse: None,
        }
    }

    /// Pulls the next work unit from the cursor and prepares it into the
    /// matching slot.
    ///
    /// Returns whether a unit was prepared; both slots stay `None` when the
    /// cursor is exhausted.
    ///
    /// # Safety
    ///
    /// The cursor must only yield units validated against this context's
    /// storage managers, and no structural mutation may run while the
    /// prepared unit is alive.
    unsafe fn prepare_next_unit(&mut self) -> bool {
        let Some(unit) = self.work.next_unit() else {
            return false;
        };

        match unit {
            QueryWorkUnit::ArchetypePages(unit) => {
                self.current = Some(unsafe { self.context.prepare_table_write_unit(unit) });
            }
            QueryWorkUnit::SparseRows(unit) => {
                self.current_sparse =
                    Some(unsafe { self.context.prepare_table_sparse_write_unit(unit) });
            }
        }

        true
    }

    /// Visits all remaining rows, optionally applying per-row filters.
    ///
    /// Units already prepared by a partially driven `next_impl` chain are
    /// drained first; fresh units then stream straight from the work cursor.
    pub(crate) fn for_each<Func, const APPLY_FILTER: bool>(mut self, mut func: Func)
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        if let Some(mut current) = self.current.take() {
            unsafe { current.for_each_remaining::<Func, APPLY_FILTER>(&mut func) };
        }

        if let Some(unit) = self.current_sparse.take() {
            unsafe {
                let mut unit = unit;
                unit.for_each_remaining::<Func, APPLY_FILTER>(&mut func);
            }
        }

        while let Some(unit) = self.work.next_unit() {
            unsafe {
                self.context
                    .for_each_write_unit::<Func, APPLY_FILTER, { Path::TABLE }>(unit, &mut func);
            }
        }
    }

    /// Returns the next row, optionally applying per-row filters and
    /// preparing further work units lazily until the cursor is exhausted.
    pub(crate) fn next_impl<const APPLY_FILTER: bool>(
        &mut self,
    ) -> Option<<D as f_mixed::WriteQueryData>::Item<'w>> {
        loop {
            if self.current.is_none()
                && self.current_sparse.is_none()
                && !unsafe { self.prepare_next_unit() }
            {
                return None;
            }

            if let Some(unit) = self.current_sparse.as_mut() {
                if let Some(item) = unsafe { unit.next::<APPLY_FILTER>() } {
                    return Some(item);
                }

                self.current_sparse = None;
                continue;
            }

            let current = self.current.as_mut().unwrap();

            if let Some(item) = unsafe { current.next::<APPLY_FILTER>() } {
                return Some(item);
            }

            self.current = None;
        }
    }
}
