//! Query iterator dispatch.
//!
//! `QueryIter` and `QueryIterMut` choose between a table-only fast path and the
//! mixed table/sparse path. They also skip per-row filter checks when both data
//! and filters are fully archetypal.

pub(crate) mod mixed;
pub(crate) mod table;

use crate::{
    change_detection::TickWindow,
    component::{archetype::ArchetypeManager, sparse_set::SparseSetManager},
    query::fetch::{ReadQueryData, WriteQueryData},
};

use super::{
    execution::{work::QueryWorkCursor, QueryReadContext, QueryWriteContext},
    fetch::mixed as f_mixed,
    filter::QueryFilter,
};

/// Read-only query iterator.
///
/// Dispatcher enum: construction resolves once whether the data fetch and
/// the filter are fully archetypal and stores the most specific
/// implementation. `next` and `for_each` then forward a const filter flag,
/// so the row loops never re-test the dispatch decision and per-row filter
/// checks compile away for archetypal filters.
///
/// The `'w` lifetime borrows the world's component storage that yielded
/// items point into; `'s` borrows the cached query state supplying the fetch
/// state and the work-unit list. Splitting the two lets a reusable state
/// outlive any single world borrow, as system-held states do.
pub enum QueryIter<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Table route with per-row filter evaluation.
    TableFiltered(table::QueryIter<'w, 's, D, F>),
    /// Table route with the filter resolved at archetype level.
    TableArchetypal(table::QueryIter<'w, 's, D, F>),
    /// Mixed route with per-row filter evaluation.
    MixedFiltered(mixed::QueryIter<'w, 's, D, F>),
    /// Mixed route with the filter resolved at archetype level.
    MixedArchetypal(mixed::QueryIter<'w, 's, D, F>),
}

/// Namespace for the const flags forwarded as the iterators' `APPLY_FILTER`
/// parameter.
struct Filter;
impl Filter {
    /// The query has a non-archetypal filter: evaluate it per row.
    const APPLY: bool = true;
    /// The filter was resolved at archetype level: bypass evaluation.
    const BYPASS: bool = false;
}

impl<'w, 's, D, F> Iterator for QueryIter<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    type Item = <D as f_mixed::ReadQueryData>::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            QueryIter::TableFiltered(iter) => iter.next_impl::<{ Filter::APPLY }>(),
            QueryIter::TableArchetypal(iter) => iter.next_impl::<{ Filter::BYPASS }>(),
            QueryIter::MixedFiltered(iter) => iter.next_impl::<{ Filter::APPLY }>(),
            QueryIter::MixedArchetypal(iter) => iter.next_impl::<{ Filter::BYPASS }>(),
        }
    }
}

impl<'w, 's, D, F> QueryIter<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Selects the iterator implementation for the cached query state,
    /// crossing table-vs-mixed data access with archetypal-vs-per-row filter
    /// evaluation.
    pub(crate) fn new(
        archetypes: *const ArchetypeManager,
        sparse_sets: *const SparseSetManager,
        data_state: &'s D::State,
        filter_state: &'s F::State,
        work: QueryWorkCursor<'s>,
        tick_window: TickWindow,
    ) -> Self {
        let context = QueryReadContext::new(
            archetypes,
            sparse_sets,
            data_state,
            filter_state,
            tick_window,
        );
        match (D::is_archetypal(data_state), F::is_archetypal(filter_state)) {
            (true, true) => QueryIter::TableArchetypal(table::QueryIter::new(context, work)),
            (true, false) => QueryIter::TableFiltered(table::QueryIter::new(context, work)),
            (false, true) => QueryIter::MixedArchetypal(mixed::QueryIter::new(context, work)),
            (false, false) => QueryIter::MixedFiltered(mixed::QueryIter::new(context, work)),
        }
    }

    /// Visits every matching item using the selected iterator path.
    pub fn for_each<Func>(self, func: Func)
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        match self {
            QueryIter::MixedFiltered(iter) => iter.for_each::<Func, { Filter::APPLY }>(func),
            QueryIter::MixedArchetypal(iter) => iter.for_each::<Func, { Filter::BYPASS }>(func),
            QueryIter::TableFiltered(iter) => iter.for_each::<Func, { Filter::APPLY }>(func),
            QueryIter::TableArchetypal(iter) => iter.for_each::<Func, { Filter::BYPASS }>(func),
        }
    }
}

/// Mutable query iterator.
///
/// Mirrors [`QueryIter`] with the same dispatch variants, const filter flag,
/// and `'w`/`'s` lifetime split, but limits mutation to the component
/// regions declared by the query access set.
pub enum QueryIterMut<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Table route with per-row filter evaluation.
    TableFiltered(table::QueryIterMut<'w, 's, D, F>),
    /// Table route with the filter resolved at archetype level.
    TableArchetypalFilter(table::QueryIterMut<'w, 's, D, F>),
    /// Mixed route with per-row filter evaluation.
    MixedFiltered(mixed::QueryIterMut<'w, 's, D, F>),
    /// Mixed route with the filter resolved at archetype level.
    MixedArchetypalFilter(mixed::QueryIterMut<'w, 's, D, F>),
}

impl<'w, 's, D, F> Iterator for QueryIterMut<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    type Item = <D as f_mixed::WriteQueryData>::Item<'w>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            QueryIterMut::TableFiltered(iter) => iter.next_impl::<{ Filter::APPLY }>(),
            QueryIterMut::TableArchetypalFilter(iter) => iter.next_impl::<{ Filter::BYPASS }>(),
            QueryIterMut::MixedFiltered(iter) => iter.next_impl::<{ Filter::APPLY }>(),
            QueryIterMut::MixedArchetypalFilter(iter) => iter.next_impl::<{ Filter::BYPASS }>(),
        }
    }
}

impl<'w, 's, D, F> QueryIterMut<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Selects the mutable iterator implementation for the cached query
    /// state, crossing table-vs-mixed data access with archetypal-vs-per-row
    /// filter evaluation.
    pub(crate) fn new(
        archetypes: *const ArchetypeManager,
        sparse_sets: *const SparseSetManager,
        data_state: &'s D::State,
        filter_state: &'s F::State,
        work: QueryWorkCursor<'s>,
        tick_window: TickWindow,
    ) -> Self {
        let context = QueryWriteContext::new(
            archetypes,
            sparse_sets,
            data_state,
            filter_state,
            tick_window,
        );
        match (D::is_archetypal(data_state), F::is_archetypal(filter_state)) {
            (true, true) => {
                QueryIterMut::TableArchetypalFilter(table::QueryIterMut::new(context, work))
            }
            (true, false) => QueryIterMut::TableFiltered(table::QueryIterMut::new(context, work)),
            (false, true) => {
                QueryIterMut::MixedArchetypalFilter(mixed::QueryIterMut::new(context, work))
            }
            (false, false) => QueryIterMut::MixedFiltered(mixed::QueryIterMut::new(context, work)),
        }
    }

    /// Visits every matching mutable item using the selected iterator path.
    pub fn for_each<Func>(self, func: Func)
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        match self {
            QueryIterMut::MixedFiltered(iter) => iter.for_each::<Func, { Filter::APPLY }>(func),
            QueryIterMut::MixedArchetypalFilter(iter) => {
                iter.for_each::<Func, { Filter::BYPASS }>(func)
            }
            QueryIterMut::TableFiltered(iter) => iter.for_each::<Func, { Filter::APPLY }>(func),
            QueryIterMut::TableArchetypalFilter(iter) => {
                iter.for_each::<Func, { Filter::BYPASS }>(func)
            }
        }
    }
}
