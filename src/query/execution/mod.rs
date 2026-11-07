//! Query execution over prepared work units.
//!
//! This module owns the runtime execution layer used by `QueryState::for_each`.
//! Sequential and parallel dispatch share the same `QueryWork` representation.

pub(crate) mod live_rows;
pub(crate) mod mixed;
pub(crate) mod parallel;
pub(crate) mod table;
pub(crate) mod work;

pub mod debug;

pub(crate) use live_rows::LiveRowCursor;

use std::{any::type_name, borrow::Cow, marker::PhantomData};

use crate::{
    change_detection::TickWindow,
    component::{
        archetype::{ArchetypeId, ArchetypeManager},
        sparse_set::SparseSetManager,
        Archetype, ComponentManager,
    },
    entity::{ArchetypeStorageKind, Entity},
    query::{
        execution::work::{
            PreparedSparseRows, QueryWorkConfig, QueryWorkUnit, SparseDriverLaneKind,
            SparseRowsSource, SparseRowsWork, ValidatedQueryWork,
        },
        fetch::{mixed as f_mixed, ReadQueryData, WriteQueryData},
        filter::QueryFilter,
        QueryError, QueryExecutionDebug, QueryExecutionPath, QueryParallelChoice,
        QueryParallelConfig, QueryParallelDecision, QueryParallelReason,
    },
    task::TaskExecutorHandle,
};

/// How many rows ahead sparse-driver units prefetch their table gathers.
///
/// The strided gather cases measure 8-14ns per row, so ten rows of lookahead
/// keep roughly one memory latency of work in flight; dense lanes gain
/// nothing from larger distances because the hardware prefetcher already
/// streams them.
pub(crate) const SPARSE_GATHER_PREFETCH_DISTANCE: usize = 10;

/// Zero-sized namespace holding the `TABLE` and `MIXED` markers that name the
/// two fetch paths selected by the `TABLE` const generic.
pub(crate) struct Path;
impl Path {
    /// Const marker selecting the table-only fetch path.
    pub(crate) const TABLE: bool = true;
    /// Const marker selecting the mixed table/sparse fetch path.
    pub(crate) const MIXED: bool = false;
}

/// Validated work leased to one execution.
///
/// Borrowed work covers runs that stay inside the owner's borrow; owned copies
/// cover executions that must outlive it.
type QueryWorkLease<'s> = Cow<'s, ValidatedQueryWork>;

/// Evaluates entity membership and the per-row filter for one table row.
///
/// # Safety
///
/// `fetch` must be initialized for the archetype and storage lane owning
/// `row`, and `row` must address a live row of that lane.
#[inline(always)]
pub(crate) unsafe fn matches_filter<'w, F: QueryFilter>(
    fetch: &mut F::Fetch<'w>,
    row: usize,
    entity: Entity,
) -> bool {
    (unsafe { F::matches_entity(fetch, row, entity) }) && (unsafe { F::filter(fetch, row, entity) })
}

/// Evaluates entity membership and the per-row filter for one sparse-driver
/// candidate, addressing it through its pre-resolved dense position.
///
/// # Safety
///
/// `fetch` must be initialized for the unit's archetype, storage lane, and
/// driver, and `row` must address a live row of the resolved range.
#[inline(always)]
pub(crate) unsafe fn matches_filter_sparse_driver<'w, F: QueryFilter>(
    fetch: &mut F::Fetch<'w>,
    row: usize,
    entity: Entity,
    sparse_dense_pos: u32,
) -> bool {
    (unsafe { F::matches_entity_sparse_driver(fetch, row, entity, sparse_dense_pos) })
        && (unsafe { F::filter_sparse_driver(fetch, row, entity, sparse_dense_pos) })
}

/// Shared raw context for one read-only execution.
///
/// Bundles the storage manager pointers, query and filter state, and tick
/// window that every prepared unit needs. Worker tasks share the context by
/// pointer, which is sound because structural mutation is forbidden while the
/// execution world is alive.
pub(crate) struct QueryReadContext<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Raw archetype manager pointer, valid while the execution world is alive.
    pub(crate) archetypes: *const ArchetypeManager,
    /// Raw sparse set manager pointer, valid while the execution world is alive.
    pub(crate) sparse_sets: *const SparseSetManager,
    /// Data fetch initialization state for `D`.
    pub(crate) data_state: &'s D::State,
    /// Filter fetch initialization state for `F`.
    pub(crate) filter_state: &'s F::State,
    /// Tick window that change filters evaluate against.
    pub(crate) tick_window: TickWindow,
    /// Phantom shared borrow of the world's component manager, tying the raw
    /// pointers to `'w`.
    _marker: PhantomData<&'w ComponentManager>,
}

// SAFETY:
// Query contexts only share immutable pointers to storage managers and immutable
// query/filter state. Structural mutation is forbidden while an execution world
// is alive, and mutable value access is controlled by validated query access
// plus disjoint query work units.
unsafe impl<'w, 's, D, F> Send for QueryReadContext<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
    D::State: Sync,
    F::State: Sync,
{
}

// SAFETY: see the `Send` implementation above.
unsafe impl<'w, 's, D, F> Sync for QueryReadContext<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
    D::State: Sync,
    F::State: Sync,
{
}

impl<'w, 's, D, F> QueryReadContext<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Assembles the read context from raw manager pointers.
    pub(crate) fn new(
        archetypes: *const ArchetypeManager,
        sparse_sets: *const SparseSetManager,
        data_state: &'s D::State,
        filter_state: &'s F::State,
        tick_window: TickWindow,
    ) -> Self {
        Self {
            archetypes,
            sparse_sets,
            data_state,
            filter_state,
            tick_window,
            _marker: PhantomData,
        }
    }

    /// Executes one work unit by preparing the fetch state it requires.
    ///
    /// `APPLY_FILTER` keeps per-row filter evaluation for filters that
    /// archetype metadata cannot decide; `TABLE` selects table-only fetch
    /// preparation instead of the mixed table/sparse one. Both parameters are
    /// const, so the dispatch specializes per instantiation.
    ///
    /// # Safety
    ///
    /// `unit` must come from validated work built against this context's
    /// managers, and no structural mutation may run while the unit executes.
    pub(crate) unsafe fn for_each_read_unit<Func, const APPLY_FILTER: bool, const TABLE: bool>(
        &self,
        unit: QueryWorkUnit,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        match unit {
            QueryWorkUnit::ArchetypePages(unit) => {
                if TABLE {
                    let mut prepared = unsafe { self.prepare_table_read_unit(unit) };
                    unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
                } else {
                    let mut prepared = unsafe { self.prepare_mixed_read_unit(unit) };
                    unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
                }
            }
            QueryWorkUnit::SparseRows(unit) => {
                unsafe { self.for_each_sparse_read_unit::<Func, APPLY_FILTER, TABLE>(unit, func) };
            }
        }
    }

    /// Prepares and drains one sparse-driver unit through the selected path.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::for_each_read_unit`].
    unsafe fn for_each_sparse_read_unit<Func, const APPLY_FILTER: bool, const TABLE: bool>(
        &self,
        unit: SparseRowsWork,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        if TABLE {
            let mut prepared = unsafe { self.prepare_table_sparse_read_unit(unit) };
            unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
        } else {
            let mut prepared = unsafe { self.prepare_mixed_sparse_read_unit(unit) };
            unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
        }
    }

    /// Resolves a sparse unit's source into its driver row slice.
    ///
    /// # Safety
    ///
    /// `self.sparse_sets` must be valid and `unit` must come from validated
    /// work, which guarantees the driver, bucket, and row range exist.
    pub(crate) unsafe fn prepare_sparse_rows(&self, unit: SparseRowsWork) -> PreparedSparseRows {
        unsafe { prepare_sparse_rows(self.sparse_sets, unit) }
    }
}

/// Dispatch envelope over the four read execution specializations.
///
/// Every variant carries the same context and work; the const generics baked
/// into the wrapped unit fix whether the filter runs per row and whether the
/// table-only or mixed fetch machinery is used.
pub(crate) enum QueryReadExecution<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Table data with per-row filters (`APPLY_FILTER = true`, `TABLE = true`).
    TableFiltered(QueryReadExecutionUnit<'w, 's, D, F, true, true>),
    /// Table data with archetypal filters (`APPLY_FILTER = false`, `TABLE = true`).
    TableArchetypal(QueryReadExecutionUnit<'w, 's, D, F, false, true>),
    /// Mixed data with per-row filters (`APPLY_FILTER = true`, `TABLE = false`).
    MixedFiltered(QueryReadExecutionUnit<'w, 's, D, F, true, false>),
    /// Mixed data with archetypal filters (`APPLY_FILTER = false`, `TABLE = false`).
    MixedArchetypal(QueryReadExecutionUnit<'w, 's, D, F, false, false>),
}

impl<'w, 's, D, F> QueryReadExecution<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Selects the execution unit variant from data and filter archetypality.
    pub(crate) fn new(
        archetypes: *const ArchetypeManager,
        sparse_sets: *const SparseSetManager,
        data_state: &'s D::State,
        filter_state: &'s F::State,
        work: QueryWorkLease<'s>,
        tick_window: TickWindow,
    ) -> Self {
        let context = QueryReadContext::new(
            archetypes,
            sparse_sets,
            data_state,
            filter_state,
            tick_window,
        );

        let data_is_archetypal = D::is_archetypal(data_state);

        let filter_is_archetypal = F::is_archetypal(filter_state)
            || work
                .sparse_driver()
                .is_some_and(|driver| F::is_satisfied_by_sparse_driver(filter_state, driver));

        match (data_is_archetypal, filter_is_archetypal) {
            (true, true) => Self::TableArchetypal(QueryReadExecutionUnit { context, work }),
            (true, false) => Self::TableFiltered(QueryReadExecutionUnit { context, work }),
            (false, true) => Self::MixedArchetypal(QueryReadExecutionUnit { context, work }),
            (false, false) => Self::MixedFiltered(QueryReadExecutionUnit { context, work }),
        }
    }

    /// Runs every unit sequentially with the configured filter handling.
    pub(crate) fn for_each<Func>(self, func: Func)
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        match self {
            Self::TableFiltered(exec) => exec.for_each(func),
            Self::TableArchetypal(exec) => exec.for_each(func),
            Self::MixedFiltered(exec) => exec.for_each(func),
            Self::MixedArchetypal(exec) => exec.for_each(func),
        }
    }

    /// Runs every unit sequentially and reports the execution path taken.
    pub(crate) fn for_each_with_debug<Func>(
        self,
        work_config: QueryWorkConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        let debug = QueryExecutionDebug::sequential(self.path(), self.work(), work_config);

        self.for_each(func);

        debug
    }

    /// Runs every unit through the task executor with static partitioning.
    ///
    /// The parallelization decision happens before this call; the split itself
    /// is dispatched one unit per task.
    pub(crate) fn par_for_each<Func>(self, tasks: &TaskExecutorHandle, func: Func)
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::ReadQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        match self {
            Self::TableFiltered(exec) => exec.par_for_each(tasks, func),
            Self::TableArchetypal(exec) => exec.par_for_each(tasks, func),
            Self::MixedFiltered(exec) => exec.par_for_each(tasks, func),
            Self::MixedArchetypal(exec) => exec.par_for_each(tasks, func),
        }
    }

    /// Returns the debug path matching this variant.
    fn path(&self) -> QueryExecutionPath {
        match self {
            Self::TableFiltered(_) => QueryExecutionPath::TableFiltered,
            Self::TableArchetypal(_) => QueryExecutionPath::TableArchetypal,
            Self::MixedFiltered(_) => QueryExecutionPath::MixedFiltered,
            Self::MixedArchetypal(_) => QueryExecutionPath::MixedArchetypal,
        }
    }

    /// Returns the validated work shared by every variant.
    fn work(&self) -> &ValidatedQueryWork {
        match self {
            Self::TableFiltered(exec) => exec.work.as_ref(),
            Self::TableArchetypal(exec) => exec.work.as_ref(),
            Self::MixedFiltered(exec) => exec.work.as_ref(),
            Self::MixedArchetypal(exec) => exec.work.as_ref(),
        }
    }

    /// Runs with an explicit parallel configuration and reports the decision.
    pub(crate) fn par_for_each_with_config<Func>(
        self,
        tasks: &TaskExecutorHandle,
        work_config: QueryWorkConfig,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::ReadQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        let config = config.normalized();
        let worker_count = tasks.worker_count();
        let decision = config.decide(self.work(), worker_count);
        let debug = QueryExecutionDebug::parallel_request(
            self.path(),
            self.work(),
            work_config,
            config,
            decision,
            worker_count,
        );

        match decision.choice {
            QueryParallelChoice::Parallel => self.par_for_each(tasks, func),
            QueryParallelChoice::Sequential => self.for_each(func),
        }

        debug
    }

    /// Runs sequentially, reporting that no executor was available.
    pub(crate) fn par_for_each_without_executor<Func>(
        self,
        work_config: QueryWorkConfig,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        let config = config.normalized();
        let decision = QueryParallelDecision {
            choice: QueryParallelChoice::Sequential,
            reason: QueryParallelReason::NotEnoughWorkers,
        };
        let debug = QueryExecutionDebug::parallel_request(
            self.path(),
            self.work(),
            work_config,
            config,
            decision,
            0,
        );

        self.for_each(func);

        debug
    }
}

/// One specialization of a read execution with fixed filter and fetch paths.
///
/// `APPLY_FILTER` evaluates the filter per row; `TABLE` selects table-only
/// fetches instead of mixed table/sparse ones. Both parameters are const, so
/// each specialization compiles its own inner loop without the dispatch.
pub(crate) struct QueryReadExecutionUnit<'w, 's, D, F, const APPLY_FILTER: bool, const TABLE: bool>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    context: QueryReadContext<'w, 's, D, F>,
    work: QueryWorkLease<'s>,
}

impl<'w, 's, D, F, const APPLY_FILTER: bool, const TABLE: bool>
    QueryReadExecutionUnit<'w, 's, D, F, APPLY_FILTER, TABLE>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Executes every unit sequentially on the current thread.
    pub(crate) fn for_each<Func>(self, mut func: Func)
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        for unit in self.work.units().iter().copied() {
            unsafe {
                self.context
                    .for_each_read_unit::<Func, APPLY_FILTER, TABLE>(unit, &mut func);
            }
        }
    }

    /// Executes the units under a static, one-unit-per-task partition with no
    /// work stealing between tasks.
    ///
    /// A single unit or a single worker skips the executor and runs inline,
    /// since task overhead would dominate; otherwise each task receives one
    /// unit by index.
    pub(crate) fn par_for_each<Func>(self, tasks: &TaskExecutorHandle, func: Func)
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::ReadQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        let Self { context, work } = self;
        let units = work.units();

        if units.len() <= 1 || tasks.worker_count() <= 1 {
            for unit in units.iter().copied() {
                let mut local = |item| func(item);
                unsafe {
                    context.for_each_read_unit::<_, APPLY_FILTER, TABLE>(unit, &mut local);
                }
            }
            return;
        }

        tasks.scope(units.len(), |index| {
            let unit = units[index];
            let mut local = |item| func(item);

            unsafe {
                context.for_each_read_unit::<_, APPLY_FILTER, TABLE>(unit, &mut local);
            }
        });
    }
}

/// Shared raw context for one mutable execution.
///
/// Carries the same state as the read context plus a mutable world marker:
/// writes stay sound because the access validator rejects conflicting query
/// parameters up front and workers receive disjoint work units.
pub(crate) struct QueryWriteContext<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Raw archetype manager pointer, valid while the execution world is alive.
    pub(crate) archetypes: *const ArchetypeManager,
    /// Raw sparse set manager pointer, valid while the execution world is alive.
    pub(crate) sparse_sets: *const SparseSetManager,
    /// Data fetch initialization state for `D`.
    pub(crate) data_state: &'s D::State,
    /// Filter fetch initialization state for `F`.
    pub(crate) filter_state: &'s F::State,
    /// Tick window that change filters evaluate against.
    pub(crate) tick_window: TickWindow,
    /// Phantom exclusive borrow of the world's component manager, tying the
    /// raw pointers to `'w`.
    _marker: PhantomData<&'w mut ComponentManager>,
}

// SAFETY:
// Write contexts are shared between worker tasks, but each task receives a
// disjoint query work unit. The system access validator already rejects
// incompatible query parameters before any context is built.
unsafe impl<'w, 's, D, F> Send for QueryWriteContext<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
    D::State: Sync,
    F::State: Sync,
{
}

// SAFETY: see the `Send` implementation above.
unsafe impl<'w, 's, D, F> Sync for QueryWriteContext<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
    D::State: Sync,
    F::State: Sync,
{
}

impl<'w, 's, D, F> QueryWriteContext<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Assembles the write context from raw manager pointers.
    pub(crate) fn new(
        archetypes: *const ArchetypeManager,
        sparse_sets: *const SparseSetManager,
        data_state: &'s D::State,
        filter_state: &'s F::State,
        tick_window: TickWindow,
    ) -> Self {
        Self {
            archetypes,
            sparse_sets,
            data_state,
            filter_state,
            tick_window,
            _marker: PhantomData,
        }
    }

    /// Executes one work unit by preparing the fetch state it requires.
    ///
    /// `APPLY_FILTER` keeps per-row filter evaluation for filters that
    /// archetype metadata cannot decide; `TABLE` selects table-only fetch
    /// preparation instead of the mixed table/sparse one.
    ///
    /// # Safety
    ///
    /// `unit` must come from validated work built against this context's
    /// managers, and no structural mutation may run while the unit executes.
    pub(crate) unsafe fn for_each_write_unit<Func, const APPLY_FILTER: bool, const TABLE: bool>(
        &self,
        unit: QueryWorkUnit,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        match unit {
            QueryWorkUnit::ArchetypePages(unit) => {
                if TABLE {
                    let mut prepared = unsafe { self.prepare_table_write_unit(unit) };
                    unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
                } else {
                    let mut prepared = unsafe { self.prepare_mixed_write_unit(unit) };
                    unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
                }
            }
            QueryWorkUnit::SparseRows(unit) => {
                unsafe { self.for_each_sparse_write_unit::<Func, APPLY_FILTER, TABLE>(unit, func) };
            }
        }
    }

    /// Prepares and drains one sparse-driver unit through the selected path.
    ///
    /// # Safety
    ///
    /// Same contract as [`Self::for_each_write_unit`].
    unsafe fn for_each_sparse_write_unit<Func, const APPLY_FILTER: bool, const TABLE: bool>(
        &self,
        unit: SparseRowsWork,
        func: &mut Func,
    ) where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        if TABLE {
            let mut prepared = unsafe { self.prepare_table_sparse_write_unit(unit) };
            unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
        } else {
            let mut prepared = unsafe { self.prepare_mixed_sparse_write_unit(unit) };
            unsafe { prepared.for_each_remaining::<Func, APPLY_FILTER>(func) };
        }
    }

    /// Resolves a sparse unit's source into its driver row slice.
    ///
    /// # Safety
    ///
    /// `self.sparse_sets` must be valid and `unit` must come from validated
    /// work, which guarantees the driver, bucket, and row range exist.
    pub(crate) unsafe fn prepare_sparse_rows(&self, unit: SparseRowsWork) -> PreparedSparseRows {
        unsafe { prepare_sparse_rows(self.sparse_sets, unit) }
    }
}

/// Dispatch envelope over the four mutable execution specializations.
///
/// Every variant carries the same context and work; the const generics baked
/// into the wrapped unit fix whether the filter runs per row and whether the
/// table-only or mixed fetch machinery is used.
pub(crate) enum QueryWriteExecution<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Table data with per-row filters (`APPLY_FILTER = true`, `TABLE = true`).
    TableFiltered(QueryWriteExecutionUnit<'w, 's, D, F, true, true>),
    /// Table data with archetypal filters (`APPLY_FILTER = false`, `TABLE = true`).
    TableArchetypal(QueryWriteExecutionUnit<'w, 's, D, F, false, true>),
    /// Mixed data with per-row filters (`APPLY_FILTER = true`, `TABLE = false`).
    MixedFiltered(QueryWriteExecutionUnit<'w, 's, D, F, true, false>),
    /// Mixed data with archetypal filters (`APPLY_FILTER = false`, `TABLE = false`).
    MixedArchetypal(QueryWriteExecutionUnit<'w, 's, D, F, false, false>),
}

impl<'w, 's, D, F> QueryWriteExecution<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Selects the execution unit variant from data and filter archetypality.
    pub(crate) fn new(
        archetypes: *const ArchetypeManager,
        sparse_sets: *const SparseSetManager,
        data_state: &'s D::State,
        filter_state: &'s F::State,
        work: QueryWorkLease<'s>,
        tick_window: TickWindow,
    ) -> Self {
        let context = QueryWriteContext::new(
            archetypes,
            sparse_sets,
            data_state,
            filter_state,
            tick_window,
        );

        let data_is_archetypal = D::is_archetypal(data_state);

        let filter_is_archetypal = F::is_archetypal(filter_state)
            || work
                .sparse_driver()
                .is_some_and(|driver| F::is_satisfied_by_sparse_driver(filter_state, driver));

        match (data_is_archetypal, filter_is_archetypal) {
            (true, true) => Self::TableArchetypal(QueryWriteExecutionUnit { context, work }),
            (true, false) => Self::TableFiltered(QueryWriteExecutionUnit { context, work }),
            (false, true) => Self::MixedArchetypal(QueryWriteExecutionUnit { context, work }),
            (false, false) => Self::MixedFiltered(QueryWriteExecutionUnit { context, work }),
        }
    }

    /// Runs every unit sequentially with the configured filter handling.
    pub(crate) fn for_each<Func>(self, func: Func)
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        match self {
            Self::TableFiltered(exec) => exec.for_each(func),
            Self::TableArchetypal(exec) => exec.for_each(func),
            Self::MixedFiltered(exec) => exec.for_each(func),
            Self::MixedArchetypal(exec) => exec.for_each(func),
        }
    }

    /// Runs every unit sequentially and reports the execution path taken.
    pub(crate) fn for_each_with_debug<Func>(
        self,
        work_config: QueryWorkConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        let debug = QueryExecutionDebug::sequential(self.path(), self.work(), work_config);

        self.for_each(func);

        debug
    }

    /// Runs every unit through the task executor with static partitioning.
    ///
    /// The parallelization decision happens before this call; the split itself
    /// is dispatched one unit per task.
    pub(crate) fn par_for_each<Func>(self, tasks: &TaskExecutorHandle, func: Func)
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::WriteQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        match self {
            Self::TableFiltered(exec) => exec.par_for_each(tasks, func),
            Self::TableArchetypal(exec) => exec.par_for_each(tasks, func),
            Self::MixedFiltered(exec) => exec.par_for_each(tasks, func),
            Self::MixedArchetypal(exec) => exec.par_for_each(tasks, func),
        }
    }

    /// Returns the debug path matching this variant.
    fn path(&self) -> QueryExecutionPath {
        match self {
            Self::TableFiltered(_) => QueryExecutionPath::TableFiltered,
            Self::TableArchetypal(_) => QueryExecutionPath::TableArchetypal,
            Self::MixedFiltered(_) => QueryExecutionPath::MixedFiltered,
            Self::MixedArchetypal(_) => QueryExecutionPath::MixedArchetypal,
        }
    }

    /// Returns the validated work shared by every variant.
    fn work(&self) -> &ValidatedQueryWork {
        match self {
            Self::TableFiltered(exec) => exec.work.as_ref(),
            Self::TableArchetypal(exec) => exec.work.as_ref(),
            Self::MixedFiltered(exec) => exec.work.as_ref(),
            Self::MixedArchetypal(exec) => exec.work.as_ref(),
        }
    }

    /// Runs with an explicit parallel configuration and reports the decision.
    pub(crate) fn par_for_each_with_config<Func>(
        self,
        tasks: &TaskExecutorHandle,
        work_config: QueryWorkConfig,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::WriteQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        let config = config.normalized();
        let worker_count = tasks.worker_count();
        let decision = config.decide(self.work(), worker_count);
        let debug = QueryExecutionDebug::parallel_request(
            self.path(),
            self.work(),
            work_config,
            config,
            decision,
            worker_count,
        );

        match decision.choice {
            QueryParallelChoice::Parallel => self.par_for_each(tasks, func),
            QueryParallelChoice::Sequential => self.for_each(func),
        }

        debug
    }

    /// Runs sequentially, reporting that no executor was available.
    pub(crate) fn par_for_each_without_executor<Func>(
        self,
        work_config: QueryWorkConfig,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        let config = config.normalized();
        let decision = QueryParallelDecision {
            choice: QueryParallelChoice::Sequential,
            reason: QueryParallelReason::NotEnoughWorkers,
        };
        let debug = QueryExecutionDebug::parallel_request(
            self.path(),
            self.work(),
            work_config,
            config,
            decision,
            0,
        );

        self.for_each(func);

        debug
    }
}

/// One specialization of a mutable execution with fixed filter and fetch
/// paths.
///
/// `APPLY_FILTER` evaluates the filter per row; `TABLE` selects table-only
/// fetches instead of mixed table/sparse ones. Both parameters are const, so
/// each specialization compiles its own inner loop without the dispatch.
pub(crate) struct QueryWriteExecutionUnit<'w, 's, D, F, const APPLY_FILTER: bool, const TABLE: bool>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    context: QueryWriteContext<'w, 's, D, F>,
    work: QueryWorkLease<'s>,
}

impl<'w, 's, D, F, const APPLY_FILTER: bool, const TABLE: bool>
    QueryWriteExecutionUnit<'w, 's, D, F, APPLY_FILTER, TABLE>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Executes every unit sequentially on the current thread.
    pub(crate) fn for_each<Func>(self, mut func: Func)
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        for unit in self.work.units().iter().copied() {
            unsafe {
                self.context
                    .for_each_write_unit::<Func, APPLY_FILTER, TABLE>(unit, &mut func);
            }
        }
    }

    /// Executes the units under a static, one-unit-per-task partition with no
    /// work stealing between tasks.
    ///
    /// A single unit or a single worker skips the executor and runs inline,
    /// since task overhead would dominate; otherwise each task receives one
    /// unit by index.
    pub(crate) fn par_for_each<Func>(self, tasks: &TaskExecutorHandle, func: Func)
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::WriteQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        let Self { context, work } = self;
        let units = work.units();

        if units.len() <= 1 || tasks.worker_count() <= 1 {
            for unit in units.iter().copied() {
                let mut local = |item| func(item);
                unsafe {
                    context.for_each_write_unit::<_, APPLY_FILTER, TABLE>(unit, &mut local);
                }
            }
            return;
        }

        tasks.scope(units.len(), |index| {
            let unit = units[index];
            let mut local = |item| func(item);

            unsafe {
                context.for_each_write_unit::<_, APPLY_FILTER, TABLE>(unit, &mut local);
            }
        });
    }
}

/// Resolves a sparse unit's bucket lane into its prepared row cursor.
///
/// # Safety
///
/// `sparse_sets` must be valid for the world and `unit` must come from
/// validated work, which guarantees the driver, bucket, lane, and row range
/// exist.
unsafe fn prepare_sparse_rows(
    sparse_sets: *const SparseSetManager,
    unit: SparseRowsWork,
) -> PreparedSparseRows {
    let SparseRowsSource {
        bucket,
        lane,
        start,
        end,
    } = unit.source;
    let sparse_sets = unsafe { &*sparse_sets };
    let buckets = sparse_sets
        .driver_buckets(unit.driver.index())
        .expect("sparse driver set must exist while executing sparse query work");
    let bucket_ref = buckets
        .get(bucket as usize)
        .expect("sparse driver bucket index must be valid while executing query work");

    debug_assert_eq!(bucket_ref.archetype_id, unit.archetype_id);

    let rows = match lane {
        SparseDriverLaneKind::Contiguous => bucket_ref.contiguous.rows(),
        SparseDriverLaneKind::Fragmented => bucket_ref.fragmented.rows(),
    };

    assert!(
        start <= end && end as usize <= rows.len(),
        "sparse driver row range must be valid while executing query work"
    );

    PreparedSparseRows::indexed(&rows[start as usize..end as usize])
}

/// Panics when fetch initialization fails during query execution.
///
/// This indicates that an internal invariant has been broken: archetypes stored
/// in `matched_archetypes` are expected to remain compatible with the cached
/// query state.
pub(crate) fn panic_fetch_error<T>(kind: &str, archetype_id: ArchetypeId, err: QueryError) -> ! {
    panic!(
        "internal query invariant violated: {} fetch failed for `{}` on archetype {:?}: {:?}",
        kind,
        type_name::<T>(),
        archetype_id,
        err
    )
}

/// Returns whether every row of a unit can be skipped because the filter's
/// change detection provably cannot match since the reader's window.
///
/// Only a filter with a tick-sensitive component can skip, and only when that
/// component lives in the unit's archetype table column and the column's last
/// write is not newer than the reader's window.
pub(crate) fn change_filter_skips_unit<F: QueryFilter>(
    filter_state: &F::State,
    archetype: &Archetype,
    storage: ArchetypeStorageKind,
    tick_window: TickWindow,
) -> bool {
    let Some(component) = F::tick_sensitive_component(filter_state) else {
        return false;
    };

    let Ok(index) = archetype.column_index(component) else {
        return false;
    };

    let column = archetype.data(storage).column(index);
    !column
        .data()
        .last_write_access()
        .is_newer_than(tick_window.last_tick, tick_window.current_tick)
}
