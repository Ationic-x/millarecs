//! Query state and cache policies.
//!
//! The query state is split into three concrete behaviours:
//! cached queries keep the matched archetype list, uncached queries scan the
//! archetype set on demand, and automatic queries start uncached and promote to
//! cached when reuse or world size justifies the retained memory.

use std::{any::TypeId, borrow::Cow};

use crate::{
    change_detection::{ChangeTick, TickWindow},
    component::{
        archetype::{Archetype, ArchetypeId, ArchetypeManager},
        Component, ComponentId, ComponentManager, StorageKind,
    },
    entity::EntityManager,
    planning::WorldPlanning,
    query::{
        access::{DataAccess, FilterAccess},
        execution::{
            work::{QueryWorkConfig, ValidatedQueryWork},
            QueryReadExecution, QueryWriteExecution,
        },
        fetch::{mixed as f_mixed, ReadQueryData, WriteQueryData},
        planning::{QueryPlanning, QueryRoute, QueryRoutePolicy},
        QueryAccess, QueryExecutionDebug, QueryParallelConfig, QueryWorkMode,
    },
    task::TaskExecutorHandle,
    world::WorldId,
};

use super::{
    fetch::QueryData,
    filter::QueryFilter,
    iter::{QueryIter, QueryIterMut},
    QueryError,
};

/// Cached state for a single component used by a query.
///
/// Stores the component id and the storage kind where the component lives.
#[derive(Clone, Copy)]
pub struct ComponentQueryState {
    /// Identifier assigned by `ComponentManager`.
    pub(crate) id: ComponentId,
    /// Storage backend used by this component.
    pub(crate) storage: StorageKind,
}

/// Runtime cache mode currently backing a query state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryCacheKind {
    /// Retains the matched archetype list between runs.
    Cached,
    /// Scans the archetype set on every run.
    Uncached,
    /// Auto policy already promoted to cached.
    AutoCached,
    /// Auto policy still running uncached.
    AutoUncached,
}

/// Read-only cache diagnostics for a query state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryCacheDebug {
    /// Current cache mode.
    pub kind: QueryCacheKind,
    /// Matched archetypes currently retained, for cached modes.
    pub retained_archetypes: Option<usize>,
    /// Highest archetype count ever observed, for promotion decisions.
    pub seen_archetypes: Option<usize>,
    /// Completed runs of this state, for promotion decisions.
    pub runs: Option<u32>,
    /// Promotion thresholds, for auto states.
    pub policy: Option<AutoQueryPolicy>,
}

impl ComponentQueryState {
    /// Creates cached state for component `T`.
    ///
    /// Returns `QueryError::ComponentNotRegistered` if `T` has not been
    /// registered in the `ComponentManager`.
    pub(crate) fn new<C: Component>(cm: &ComponentManager) -> Result<Self, QueryError> {
        let id = cm
            .id_of::<C>()
            .ok_or(QueryError::ComponentNotRegistered(TypeId::of::<C>()))?;

        let storage = cm.storage_of(id).ok_or(QueryError::StorageUnknown(id))?;

        Ok(Self { id, storage })
    }
}

/// Public query-state wrapper used by world and system APIs.
///
/// The enum is intentionally the only state type used outside this module:
/// callers can choose a cache policy without knowing how the concrete state is
/// stored internally.
pub enum QueryState<D, F = ()>
where
    D: QueryData,
    F: QueryFilter,
{
    /// State that retains the matched archetype list between runs.
    Cached(CachedQueryState<D, F>),
    /// State that scans the archetype set on every run.
    Uncached(UncachedQueryState<D, F>),
    /// State that starts uncached and promotes to cached on reuse.
    Auto(AutoQueryState<D, F>),
}

impl<D, F> QueryState<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Creates an automatic query state.
    ///
    /// This is the default constructor used by the higher-level APIs. It starts
    /// without retained archetype matches and promotes itself only when the
    /// configured policy says the cache is worth keeping.
    pub(crate) fn new(
        world_id: WorldId,
        cm: &ComponentManager,
        planning: &WorldPlanning,
    ) -> Result<Self, QueryError> {
        Self::auto(world_id, cm, planning, AutoQueryPolicy::default())
    }

    /// Creates a state that retains matched archetypes between runs.
    pub(crate) fn cached(
        world_id: WorldId,
        cm: &ComponentManager,
        planning: &WorldPlanning,
    ) -> Result<Self, QueryError> {
        Ok(Self::Cached(CachedQueryState::new(world_id, cm, planning)?))
    }

    /// Creates a state that scans archetypes on every run.
    pub(crate) fn uncached(
        world_id: WorldId,
        cm: &ComponentManager,
        planning: &WorldPlanning,
    ) -> Result<Self, QueryError> {
        Ok(Self::Uncached(UncachedQueryState::new(
            world_id, cm, planning,
        )?))
    }

    /// Creates an auto state with the given promotion policy.
    pub(crate) fn auto(
        world_id: WorldId,
        cm: &ComponentManager,
        planning: &WorldPlanning,
        policy: AutoQueryPolicy,
    ) -> Result<Self, QueryError> {
        Ok(Self::Auto(AutoQueryState::new(
            world_id, cm, planning, policy,
        )?))
    }

    /// Whether the data part resolves without sparse-set lookups.
    #[inline]
    pub fn is_archetypal(&self) -> bool {
        self.plan().is_archetypal()
    }

    /// Returns the execution route currently planned for this query.
    #[inline]
    pub fn route(&self) -> QueryRoute {
        self.plan().route()
    }

    /// Returns the route-selection policy used by this query state.
    #[inline]
    pub fn route_policy(&self) -> QueryRoutePolicy {
        self.plan().route_policy()
    }

    /// Changes the route-selection policy without rebuilding the type-level
    /// query state.
    ///
    /// Forcing a sparse driver fails when that component is not an exact,
    /// required sparse candidate for this query. Cached execution work is
    /// rebuilt lazily only when the selected route actually changes.
    #[inline]
    pub fn set_route_policy(&mut self, policy: QueryRoutePolicy) -> Result<bool, QueryError> {
        self.plan_mut().set_route_policy(policy)
    }

    /// Returns the cache mode currently backing this state.
    #[inline]
    pub fn cache_kind(&self) -> QueryCacheKind {
        self.cache_debug().kind
    }

    /// Returns read-only cache diagnostics for this state.
    pub fn cache_debug(&self) -> QueryCacheDebug {
        match self {
            Self::Cached(state) => state.cache_debug(QueryCacheKind::Cached, None, None),
            Self::Uncached(_) => QueryCacheDebug {
                kind: QueryCacheKind::Uncached,
                retained_archetypes: None,
                seen_archetypes: None,
                runs: None,
                policy: None,
            },
            Self::Auto(state) => state.cache_debug(),
        }
    }

    /// Returns the composed data and filter access of this query.
    #[inline]
    pub(crate) fn access(&self) -> &QueryAccess {
        self.plan().access()
    }

    /// Returns an error when the query is bound to a different world.
    #[inline]
    pub(crate) fn validate_world_id(&self, actual: WorldId) -> Result<(), QueryError> {
        self.plan().validate_world_id(actual)
    }

    /// Opens a run with the given tick window, validating the world binding.
    pub(crate) fn begin_run(
        &mut self,
        world_id: WorldId,
        current_tick: ChangeTick,
    ) -> Result<(), QueryError> {
        self.plan_mut().begin_run(world_id, current_tick)
    }

    /// Closes the current run, advancing change detection and promotion counts.
    pub(crate) fn finish_run(&mut self) {
        match self {
            Self::Cached(state) => state.finish_run(),
            Self::Uncached(state) => state.finish_run(),
            Self::Auto(state) => state.finish_run(),
        }
    }

    /// Returns the query plan shared by every cache policy.
    #[inline]
    pub(crate) fn plan(&self) -> &QueryPlan<D, F> {
        match self {
            Self::Cached(state) => state.plan(),
            Self::Uncached(state) => state.plan(),
            Self::Auto(state) => state.plan(),
        }
    }

    /// Returns the query plan shared by every cache policy for mutation.
    #[inline]
    pub(crate) fn plan_mut(&mut self) -> &mut QueryPlan<D, F> {
        match self {
            Self::Cached(state) => state.plan_mut(),
            Self::Uncached(state) => state.plan_mut(),
            Self::Auto(state) => state.plan_mut(),
        }
    }
}

impl<D, F> QueryState<D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Creates a read-only iterator over the entities matching this query.
    pub(crate) fn iter<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryIter<'w, 's, D, F> {
        match self {
            Self::Cached(state) => state.iter(cm, planning, entities),
            Self::Uncached(state) => state.iter(cm, planning, entities),
            Self::Auto(state) => state.iter(cm, planning, entities),
        }
    }

    /// Prepares the read execution for this query with default work config.
    pub(crate) fn read_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryReadExecution<'w, 's, D, F> {
        match self {
            Self::Cached(state) => state.read_execution(cm, planning, entities),
            Self::Uncached(state) => state.read_execution(cm, planning, entities),
            Self::Auto(state) => state.read_execution(cm, planning, entities),
        }
    }

    /// Prepares the read execution with an explicit work configuration.
    pub(crate) fn read_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryReadExecution<'w, 's, D, F> {
        match self {
            Self::Cached(state) => state.read_execution_with_config(cm, planning, entities, config),
            Self::Uncached(state) => {
                state.read_execution_with_config(cm, planning, entities, config)
            }
            Self::Auto(state) => state.read_execution_with_config(cm, planning, entities, config),
        }
    }

    /// Runs every matching item sequentially through the read execution.
    pub(crate) fn for_each<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        func: Func,
    ) where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        self.read_execution(cm, planning, entities).for_each(func);
    }

    /// Runs sequentially and reports the execution path taken.
    pub(crate) fn for_each_debug<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        let work_config = self.plan().work_config();
        let execution = self.read_execution(cm, planning, entities);

        execution.for_each_with_debug(work_config, func)
    }

    /// Runs every matching item with automatic parallelization.
    pub(crate) fn par_for_each<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        tasks: &TaskExecutorHandle,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::ReadQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        self.par_for_each_with_config(
            cm,
            planning,
            entities,
            tasks,
            QueryParallelConfig::auto(),
            func,
        )
    }

    /// Runs with an explicit parallel configuration and reports the decision.
    pub(crate) fn par_for_each_with_config<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        tasks: &TaskExecutorHandle,
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
        let work_config = config.work_config();

        let execution = self.read_execution_with_config(cm, planning, entities, work_config);
        execution.par_for_each_with_config(tasks, work_config, config, func)
    }

    /// Runs sequentially, reporting that no executor was available.
    pub(crate) fn par_for_each_without_executor_with_config<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        let config = config.normalized();
        let work_config = config.work_config();

        let execution = self.read_execution_with_config(cm, planning, entities, work_config);
        execution.par_for_each_without_executor(work_config, config, func)
    }
}

impl<D, F> QueryState<D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Creates a mutable iterator over the entities matching this query.
    pub(crate) fn iter_mut<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryIterMut<'w, 's, D, F> {
        match self {
            Self::Cached(state) => state.iter_mut(cm, planning, entities),
            Self::Uncached(state) => state.iter_mut(cm, planning, entities),
            Self::Auto(state) => state.iter_mut(cm, planning, entities),
        }
    }

    /// Prepares the mutable execution for this query with default work config.
    pub(crate) fn write_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        match self {
            Self::Cached(state) => state.write_execution(cm, planning, entities),
            Self::Uncached(state) => state.write_execution(cm, planning, entities),
            Self::Auto(state) => state.write_execution(cm, planning, entities),
        }
    }

    /// Prepares the mutable execution with an explicit work configuration.
    pub(crate) fn write_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        match self {
            Self::Cached(state) => {
                state.write_execution_with_config(cm, planning, entities, config)
            }
            Self::Uncached(state) => {
                state.write_execution_with_config(cm, planning, entities, config)
            }
            Self::Auto(state) => state.write_execution_with_config(cm, planning, entities, config),
        }
    }

    /// Runs every matching item sequentially through the mutable execution.
    pub(crate) fn for_each_mut<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        func: Func,
    ) where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        self.write_execution(cm, planning, entities).for_each(func)
    }

    /// Runs sequentially and reports the execution path taken.
    pub(crate) fn for_each_mut_debug<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        let work_config = self.plan().work_config();
        let execution = self.write_execution(cm, planning, entities);

        execution.for_each_with_debug(work_config, func)
    }

    /// Runs every matching item with automatic parallelization.
    pub(crate) fn par_for_each_mut<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        tasks: &TaskExecutorHandle,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::WriteQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        self.par_for_each_mut_with_config(
            cm,
            planning,
            entities,
            tasks,
            QueryParallelConfig::auto(),
            func,
        )
    }

    /// Runs with an explicit parallel configuration and reports the decision.
    pub(crate) fn par_for_each_mut_with_config<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        tasks: &TaskExecutorHandle,
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
        let work_config = config.work_config();

        let execution = self.write_execution_with_config(cm, planning, entities, work_config);
        execution.par_for_each_with_config(tasks, work_config, config, func)
    }

    /// Runs sequentially, reporting that no executor was available.
    pub(crate) fn par_for_each_mut_without_executor_with_config<'w, 's, Func>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        let config = config.normalized();
        let work_config = config.work_config();

        let execution = self.write_execution_with_config(cm, planning, entities, work_config);
        execution.par_for_each_without_executor(work_config, config, func)
    }
}

/// Shared query plan.
///
/// This owns the expensive type-level state: component identifiers, access
/// information, storage kind decisions and change-detection window. Cache
/// policies move this plan between concrete states instead of cloning it.
pub(crate) struct QueryPlan<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// World this plan is bound to; runs from other worlds fail validation.
    world_id: WorldId,
    /// Cached type-level state of the data part.
    data_state: D::State,
    /// Cached type-level state of the filter part.
    filter_state: F::State,
    /// Cached result of data and filter `is_archetypal` at build time.
    archetypal: bool,
    /// Composed data and filter access.
    access: QueryAccess,
    /// Route-planning snapshot, refreshed when the world planning changes.
    planning: QueryPlanning,
    /// Work split configured for executions.
    work_config: QueryWorkConfig,
    /// Tick of the most recently completed run; opens the next change window.
    last_tick: ChangeTick,
    /// Change-detection window active while a run is open.
    tick_window: TickWindow,
}

impl<D, F> QueryPlan<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Builds a plan by initializing data and filter states, composing their
    /// access, and running the route planner.
    pub(crate) fn new(
        world_id: WorldId,
        cm: &ComponentManager,
        world_planning: &WorldPlanning,
    ) -> Result<Self, QueryError> {
        let mut data_access = DataAccess::default();
        let mut filter_access = FilterAccess::default();

        let data_state = D::init_state(cm, &mut data_access)?;
        let filter_state = F::init_state(cm, &mut filter_access)?;
        let archetypal = D::is_archetypal(&data_state) && F::is_archetypal(&filter_state);
        let access = QueryAccess::compose_query(data_access, filter_access);

        let planning =
            QueryPlanning::build::<D, F>(cm, world_planning, &data_state, &filter_state, &access);

        Ok(Self {
            world_id,
            data_state,
            filter_state,
            archetypal,
            access,
            planning,
            work_config: QueryWorkConfig::sequential(),
            last_tick: ChangeTick::default(),
            tick_window: TickWindow::default(),
        })
    }

    /// Returns the cached data state.
    #[inline]
    pub(crate) fn data_state(&self) -> &D::State {
        &self.data_state
    }

    /// Returns the cached filter state.
    #[inline]
    pub(crate) fn filter_state(&self) -> &F::State {
        &self.filter_state
    }

    /// Returns the change-detection window of the current run.
    #[inline]
    pub(crate) fn tick_window(&self) -> TickWindow {
        self.tick_window
    }

    /// Returns the composed data and filter access.
    #[inline]
    pub(crate) fn access(&self) -> &QueryAccess {
        &self.access
    }

    /// Returns the execution route currently planned.
    #[inline]
    pub(crate) fn route(&self) -> QueryRoute {
        self.planning.route()
    }

    /// Returns the route-selection policy.
    #[inline]
    pub(crate) fn route_policy(&self) -> QueryRoutePolicy {
        self.planning.route_policy()
    }

    /// Changes the route-selection policy, returning whether the route changed.
    #[inline]
    pub(crate) fn set_route_policy(
        &mut self,
        policy: QueryRoutePolicy,
    ) -> Result<bool, QueryError> {
        self.planning.set_route_policy(policy)
    }

    /// Returns the route-planning snapshot.
    #[inline]
    pub(crate) fn planning(&self) -> &QueryPlanning {
        &self.planning
    }

    /// Refreshes route planning when the world planning changed, returning
    /// whether anything was refreshed.
    pub(crate) fn refresh_planning_if_needed(
        &mut self,
        cm: &ComponentManager,
        world_planning: &WorldPlanning,
    ) -> bool {
        self.planning.refresh::<D, F>(
            cm,
            world_planning,
            &self.data_state,
            &self.filter_state,
            &self.access,
        )
    }

    /// Whether data and filter resolve without sparse-set lookups.
    #[inline]
    pub(crate) fn is_archetypal(&self) -> bool {
        self.archetypal
    }

    /// `true` when the composed access can never match.
    #[inline]
    pub(crate) fn is_impossible(&self) -> bool {
        self.access.is_impossible()
    }

    /// Returns the work split configured for executions.
    #[inline]
    pub(crate) fn work_config(&self) -> QueryWorkConfig {
        self.work_config
    }

    /// Whether both the data and the filter accept the archetype.
    pub(crate) fn matches_archetype(&self, archetype: &Archetype) -> bool {
        D::matches_archetype(&self.data_state, archetype)
            && F::matches_archetype(&self.filter_state, archetype)
    }

    /// Returns an error when the plan is bound to a different world.
    #[inline]
    pub(crate) fn validate_world_id(&self, actual: WorldId) -> Result<(), QueryError> {
        if self.world_id == actual {
            Ok(())
        } else {
            Err(QueryError::WorldMismatch {
                expected: self.world_id,
                actual,
            })
        }
    }

    /// Opens a run with the given tick window, validating the world binding.
    pub(crate) fn begin_run(
        &mut self,
        world_id: WorldId,
        current_tick: ChangeTick,
    ) -> Result<(), QueryError> {
        self.validate_world_id(world_id)?;

        self.tick_window = TickWindow {
            last_tick: self.last_tick,
            current_tick,
        };

        Ok(())
    }

    /// Closes the run, recording its tick as the start of the next one.
    pub(crate) fn finish_run(&mut self) {
        self.last_tick = self.tick_window.current_tick;
    }
}

/// Query state that keeps the matching archetype list between runs.
pub struct CachedQueryState<D, F = ()>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Shared query plan; auto promotion moves it into the cached state.
    plan: QueryPlan<D, F>,
    /// Archetype ids currently retained as matching.
    matched_archetypes: Vec<ArchetypeId>,
    /// Archetype count observed when the list was last rebuilt; a mismatch
    /// forces a refresh even without a planning change.
    seen_archetypes: usize,
    /// Validated work per execution mode, rebuilt in place on key changes.
    work_cache: CachedQueryWork,
}

impl<D, F> CachedQueryState<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Creates the state and runs the initial planning refresh.
    pub(crate) fn new(
        world_id: WorldId,
        cm: &ComponentManager,
        planning: &WorldPlanning,
    ) -> Result<Self, QueryError> {
        let plan = QueryPlan::new(world_id, cm, planning)?;
        let mut state = Self {
            plan,
            matched_archetypes: Vec::new(),
            seen_archetypes: 0,
            work_cache: CachedQueryWork::default(),
        };

        state.refresh(cm, planning);
        Ok(state)
    }

    /// Refreshes matches and returns the plan with validated work for `config`.
    fn prepare_work<'s>(
        &'s mut self,
        cm: &ComponentManager,
        planning: &WorldPlanning,
        config: QueryWorkConfig,
    ) -> (&'s QueryPlan<D, F>, &'s ValidatedQueryWork) {
        self.refresh(cm, planning);

        let Self {
            plan,
            matched_archetypes,
            work_cache,
            ..
        } = self;

        let work = work_cache.get_or_build(cm, planning, plan, matched_archetypes, config);

        (plan, work)
    }

    /// Builds a cached state by moving the plan of an uncached one; used by
    /// auto promotion.
    fn from_uncached(
        uncached: UncachedQueryState<D, F>,
        cm: &ComponentManager,
        planning: &WorldPlanning,
    ) -> Self {
        let UncachedQueryState { plan } = uncached;

        let mut cached = Self {
            plan,
            matched_archetypes: Vec::new(),
            seen_archetypes: 0,
            work_cache: CachedQueryWork::default(),
        };

        cached.refresh(cm, planning);
        cached
    }

    /// Returns the query plan.
    #[inline]
    fn plan(&self) -> &QueryPlan<D, F> {
        &self.plan
    }

    /// Returns the query plan for mutation.
    #[inline]
    fn plan_mut(&mut self) -> &mut QueryPlan<D, F> {
        &mut self.plan
    }

    /// Refreshes planning and rebuilds the retained archetype list when stale;
    /// impossible queries retain no archetypes.
    fn refresh(&mut self, cm: &ComponentManager, planning: &WorldPlanning) {
        let refreshed = self.plan.refresh_planning_if_needed(cm, planning);

        if self.plan.is_impossible() {
            self.seen_archetypes = cm.archetype_count();
            self.matched_archetypes.clear();
            return;
        }

        if refreshed || self.seen_archetypes != self.plan.planning().seen_archetypes() {
            self.matched_archetypes.clear();
            self.matched_archetypes
                .extend_from_slice(self.plan.planning().matched_archetypes());
            self.seen_archetypes = self.plan.planning().seen_archetypes();
        }
    }

    /// Builds diagnostics with the given kind, run count, and policy.
    fn cache_debug(
        &self,
        kind: QueryCacheKind,
        runs: Option<u32>,
        policy: Option<AutoQueryPolicy>,
    ) -> QueryCacheDebug {
        QueryCacheDebug {
            kind,
            retained_archetypes: Some(self.matched_archetypes.len()),
            seen_archetypes: Some(self.seen_archetypes),
            runs,
            policy,
        }
    }

    /// Closes the run on the underlying plan.
    fn finish_run(&mut self) {
        self.plan.finish_run();
    }
}

impl<D, F> CachedQueryState<D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Creates a read-only iterator over the retained archetypes.
    fn iter<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
    ) -> QueryIter<'w, 's, D, F> {
        let config = self.plan.work_config();
        let (plan, work) = self.prepare_work(cm, planning, config);
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryIter::new(
            archetypes,
            sparse_sets,
            plan.data_state(),
            plan.filter_state(),
            work.cursor(),
            plan.tick_window(),
        )
    }

    /// Prepares the read execution with the current work config.
    fn read_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryReadExecution<'w, 's, D, F> {
        self.read_execution_with_config(cm, planning, entities, self.plan.work_config())
    }

    /// Prepares the read execution with an explicit work configuration.
    fn read_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryReadExecution<'w, 's, D, F> {
        let (plan, work) = self.prepare_work(cm, planning, config);
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryReadExecution::new(
            archetypes,
            sparse_sets,
            plan.data_state(),
            plan.filter_state(),
            Cow::Borrowed(work),
            plan.tick_window(),
        )
    }
}

impl<D, F> CachedQueryState<D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Creates a mutable iterator over the retained archetypes.
    fn iter_mut<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
    ) -> QueryIterMut<'w, 's, D, F> {
        let config = self.plan.work_config();
        let (plan, work) = self.prepare_work(cm, planning, config);
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryIterMut::new(
            archetypes,
            sparse_sets,
            plan.data_state(),
            plan.filter_state(),
            work.cursor(),
            plan.tick_window(),
        )
    }

    /// Prepares the mutable execution with the current work config.
    fn write_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        self.write_execution_with_config(cm, planning, entities, self.plan.work_config())
    }

    /// Prepares the mutable execution with an explicit work configuration.
    fn write_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        let (plan, work) = self.prepare_work(cm, planning, config);
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryWriteExecution::new(
            archetypes,
            sparse_sets,
            plan.data_state(),
            plan.filter_state(),
            Cow::Borrowed(work),
            plan.tick_window(),
        )
    }
}

/// Query state that does not retain the matching archetype list.
pub struct UncachedQueryState<D, F = ()>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Shared query plan; the only state this policy retains.
    plan: QueryPlan<D, F>,
}

impl<D, F> UncachedQueryState<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Creates the state; no archetype list is retained.
    pub(crate) fn new(
        world_id: WorldId,
        cm: &ComponentManager,
        planning: &WorldPlanning,
    ) -> Result<Self, QueryError> {
        Ok(Self {
            plan: QueryPlan::new(world_id, cm, planning)?,
        })
    }

    /// Returns the query plan.
    #[inline]
    fn plan(&self) -> &QueryPlan<D, F> {
        &self.plan
    }

    /// Returns the query plan for mutation.
    #[inline]
    fn plan_mut(&mut self) -> &mut QueryPlan<D, F> {
        &mut self.plan
    }

    /// Closes the run on the underlying plan.
    fn finish_run(&mut self) {
        self.plan.finish_run();
    }
}

impl<D, F> UncachedQueryState<D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Creates a read-only iterator, building work from a fresh archetype scan.
    fn iter<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
    ) -> QueryIter<'w, 's, D, F> {
        self.plan.refresh_planning_if_needed(cm, planning);
        let source: QueryArchetypeSource<'_, D, F> =
            QueryArchetypeSource::scan(&self.plan, cm.archetype_count());
        let work = ValidatedQueryWork::build_from_planning(
            cm,
            source,
            self.plan.planning(),
            self.plan.data_state(),
            self.plan.filter_state(),
            self.plan.work_config(),
        );
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryIter::new(
            archetypes,
            sparse_sets,
            self.plan.data_state(),
            self.plan.filter_state(),
            work.into_cursor(),
            self.plan.tick_window(),
        )
    }

    /// Prepares the read execution with the current work config.
    fn read_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryReadExecution<'w, 's, D, F> {
        self.read_execution_with_config(cm, planning, entities, self.plan.work_config())
    }

    /// Prepares the read execution, building work from a fresh archetype scan.
    fn read_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryReadExecution<'w, 's, D, F> {
        self.plan.refresh_planning_if_needed(cm, planning);
        let source: QueryArchetypeSource<'_, D, F> =
            QueryArchetypeSource::scan(&self.plan, cm.archetype_count());
        let work = ValidatedQueryWork::build_from_planning(
            cm,
            source,
            self.plan.planning(),
            self.plan.data_state(),
            self.plan.filter_state(),
            config,
        );
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryReadExecution::new(
            archetypes,
            sparse_sets,
            self.plan.data_state(),
            self.plan.filter_state(),
            Cow::Owned(work),
            self.plan.tick_window(),
        )
    }
}

impl<D, F> UncachedQueryState<D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Creates a mutable iterator, building work from a fresh archetype scan.
    fn iter_mut<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
    ) -> QueryIterMut<'w, 's, D, F> {
        self.plan.refresh_planning_if_needed(cm, planning);
        let source: QueryArchetypeSource<'_, D, F> =
            QueryArchetypeSource::scan(&self.plan, cm.archetype_count());
        let work = ValidatedQueryWork::build_from_planning(
            cm,
            source,
            self.plan.planning(),
            self.plan.data_state(),
            self.plan.filter_state(),
            self.plan.work_config(),
        );
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryIterMut::new(
            archetypes,
            sparse_sets,
            self.plan.data_state(),
            self.plan.filter_state(),
            work.into_cursor(),
            self.plan.tick_window(),
        )
    }

    /// Prepares the mutable execution with the current work config.
    fn write_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        self.write_execution_with_config(cm, planning, entities, self.plan.work_config())
    }

    /// Prepares the mutable execution, building work from a fresh archetype scan.
    fn write_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        _entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        self.plan.refresh_planning_if_needed(cm, planning);
        let source: QueryArchetypeSource<'_, D, F> =
            QueryArchetypeSource::scan(&self.plan, cm.archetype_count());
        let work = ValidatedQueryWork::build_from_planning(
            cm,
            source,
            self.plan.planning(),
            self.plan.data_state(),
            self.plan.filter_state(),
            config,
        );
        let (archetypes, sparse_sets) = cm.storage_ptrs();

        QueryWriteExecution::new(
            archetypes,
            sparse_sets,
            self.plan.data_state(),
            self.plan.filter_state(),
            Cow::Owned(work),
            self.plan.tick_window(),
        )
    }
}

/// Archetype source consumed by query iterators.
pub(crate) enum QueryArchetypeSource<'s, D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Walks a retained list of matched archetype ids.
    Cached {
        /// Retained ids, already filtered to matches.
        ids: &'s [ArchetypeId],
        /// Next position to read within `ids`.
        index: usize,
    },

    /// Scans every archetype id in order, testing each against the plan.
    Scan {
        /// Plan used to test each archetype for a match.
        plan: &'s QueryPlan<D, F>,
        /// Next archetype ordinal to test.
        next: usize,
        /// Exclusive upper bound of the scan.
        end: usize,
    },
}

impl<'s, D, F> QueryArchetypeSource<'s, D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Creates a source over a retained id list.
    fn cached(ids: &'s [ArchetypeId]) -> Self {
        Self::Cached { ids, index: 0 }
    }

    /// Creates a source scanning up to the archetype count; impossible plans
    /// scan nothing.
    fn scan(plan: &'s QueryPlan<D, F>, archetype_count: usize) -> Self {
        let end = if plan.is_impossible() {
            0
        } else {
            archetype_count
        };

        Self::Scan { plan, next: 0, end }
    }

    /// Returns the next matching archetype id, or `None` when exhausted.
    pub(crate) fn next(&mut self, archetypes: &ArchetypeManager) -> Option<ArchetypeId> {
        match self {
            Self::Cached { ids, index } => {
                let id = ids.get(*index).copied();
                *index += usize::from(id.is_some());
                id
            }

            Self::Scan { plan, next, end } => {
                while *next < *end {
                    let id = ArchetypeId::new(*next);
                    *next += 1;

                    let archetype = archetypes.get_archetype(id).unwrap();

                    if plan.matches_archetype(archetype) {
                        return Some(id);
                    }
                }

                None
            }
        }
    }
}

/// Automatic query state.
///
/// It begins as uncached and promotes to cached by moving the owned
/// `QueryPlan`; no cloning or dummy state is needed.
pub struct AutoQueryState<D, F = ()>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Backing storage; empty only transiently while promoting.
    storage: Option<QueryStorage<D, F>>,
    /// Promotion thresholds consulted before each run.
    policy: AutoQueryPolicy,
    /// Completed runs counted toward promotion.
    runs: u32,
}

impl<D, F> AutoQueryState<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Creates the state in uncached mode with the given policy.
    pub(crate) fn new(
        world_id: WorldId,
        cm: &ComponentManager,
        planning: &WorldPlanning,
        policy: AutoQueryPolicy,
    ) -> Result<Self, QueryError> {
        Ok(Self {
            storage: Some(QueryStorage::Uncached(UncachedQueryState::new(
                world_id, cm, planning,
            )?)),
            policy,
            runs: 0,
        })
    }

    /// Returns the backing storage, never empty outside promotion.
    #[inline]
    fn storage(&self) -> &QueryStorage<D, F> {
        self.storage
            .as_ref()
            .expect("AutoQueryState storage cannot be empty outside promotion")
    }

    /// Returns the backing storage for mutation, never empty outside promotion.
    #[inline]
    fn storage_mut(&mut self) -> &mut QueryStorage<D, F> {
        self.storage
            .as_mut()
            .expect("AutoQueryState storage cannot be empty outside promotion")
    }

    /// Returns the query plan of the backing storage.
    #[inline]
    fn plan(&self) -> &QueryPlan<D, F> {
        self.storage().plan()
    }

    /// Returns the query plan of the backing storage for mutation.
    #[inline]
    fn plan_mut(&mut self) -> &mut QueryPlan<D, F> {
        self.storage_mut().plan_mut()
    }

    /// Promotes to cached once runs or archetype count reach the policy
    /// thresholds.
    fn prepare(&mut self, cm: &ComponentManager, planning: &WorldPlanning) {
        if matches!(self.storage(), QueryStorage::Cached(_)) {
            return;
        }

        let should_promote = self.runs >= self.policy.cache_after_runs
            || cm.archetype_count() >= self.policy.cache_after_archetypes;

        if should_promote {
            self.promote_to_cached(cm, planning);
        }
    }

    /// Moves the owned plan into a cached state; already-cached storage is
    /// kept as is.
    fn promote_to_cached(&mut self, cm: &ComponentManager, planning: &WorldPlanning) {
        let storage = self
            .storage
            .take()
            .expect("AutoQueryState storage cannot be empty outside promotion");

        self.storage = Some(match storage {
            QueryStorage::Cached(cached) => QueryStorage::Cached(cached),
            QueryStorage::Uncached(uncached) => {
                QueryStorage::Cached(CachedQueryState::from_uncached(uncached, cm, planning))
            }
        });
    }

    /// Closes the run and counts it toward promotion.
    fn finish_run(&mut self) {
        self.storage_mut().finish_run();
        self.runs = self.runs.saturating_add(1);
    }

    /// Builds diagnostics for the current promotion stage.
    fn cache_debug(&self) -> QueryCacheDebug {
        match self.storage() {
            QueryStorage::Cached(state) => state.cache_debug(
                QueryCacheKind::AutoCached,
                Some(self.runs),
                Some(self.policy),
            ),
            QueryStorage::Uncached(_) => QueryCacheDebug {
                kind: QueryCacheKind::AutoUncached,
                retained_archetypes: None,
                seen_archetypes: None,
                runs: Some(self.runs),
                policy: Some(self.policy),
            },
        }
    }
}

impl<D, F> AutoQueryState<D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Creates a read-only iterator, promoting first when due.
    fn iter<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryIter<'w, 's, D, F> {
        self.prepare(cm, planning);

        match self.storage_mut() {
            QueryStorage::Cached(state) => state.iter(cm, planning, entities),
            QueryStorage::Uncached(state) => state.iter(cm, planning, entities),
        }
    }

    /// Prepares the read execution with the current work config.
    fn read_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryReadExecution<'w, 's, D, F> {
        self.read_execution_with_config(cm, planning, entities, self.plan().work_config())
    }

    /// Prepares the read execution with an explicit work configuration,
    /// promoting first when due.
    fn read_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryReadExecution<'w, 's, D, F> {
        self.prepare(cm, planning);

        match self.storage_mut() {
            QueryStorage::Cached(state) => {
                state.read_execution_with_config(cm, planning, entities, config)
            }
            QueryStorage::Uncached(state) => {
                state.read_execution_with_config(cm, planning, entities, config)
            }
        }
    }
}

impl<D, F> AutoQueryState<D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Creates a mutable iterator, promoting first when due.
    fn iter_mut<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryIterMut<'w, 's, D, F> {
        self.prepare(cm, planning);

        match self.storage_mut() {
            QueryStorage::Cached(state) => state.iter_mut(cm, planning, entities),
            QueryStorage::Uncached(state) => state.iter_mut(cm, planning, entities),
        }
    }

    /// Prepares the mutable execution with the current work config.
    fn write_execution<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        self.write_execution_with_config(cm, planning, entities, self.plan().work_config())
    }

    /// Prepares the mutable execution with an explicit work configuration,
    /// promoting first when due.
    fn write_execution_with_config<'w, 's>(
        &'s mut self,
        cm: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        config: QueryWorkConfig,
    ) -> QueryWriteExecution<'w, 's, D, F> {
        self.prepare(cm, planning);

        match self.storage_mut() {
            QueryStorage::Cached(state) => {
                state.write_execution_with_config(cm, planning, entities, config)
            }
            QueryStorage::Uncached(state) => {
                state.write_execution_with_config(cm, planning, entities, config)
            }
        }
    }
}

/// Concrete storage of an auto query state before promotion.
enum QueryStorage<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// The auto policy has promoted: the matched archetype list is retained.
    Cached(CachedQueryState<D, F>),
    /// The auto policy is still scanning the archetype set on every run.
    Uncached(UncachedQueryState<D, F>),
}

impl<D, F> QueryStorage<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Returns the query plan of the concrete state.
    #[inline]
    fn plan(&self) -> &QueryPlan<D, F> {
        match self {
            Self::Cached(state) => state.plan(),
            Self::Uncached(state) => state.plan(),
        }
    }

    /// Returns the query plan of the concrete state for mutation.
    #[inline]
    fn plan_mut(&mut self) -> &mut QueryPlan<D, F> {
        match self {
            Self::Cached(state) => state.plan_mut(),
            Self::Uncached(state) => state.plan_mut(),
        }
    }

    /// Closes the run on the concrete state.
    fn finish_run(&mut self) {
        match self {
            Self::Cached(state) => state.finish_run(),
            Self::Uncached(state) => state.finish_run(),
        }
    }
}

/// Automatic-cache promotion thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AutoQueryPolicy {
    /// Completed runs after which an auto state promotes to cached.
    pub cache_after_runs: u32,
    /// World archetype count that promotes an auto state immediately.
    pub cache_after_archetypes: usize,
}

/// Promotes after 3 runs or once the world reaches 32 archetypes.
impl Default for AutoQueryPolicy {
    fn default() -> Self {
        Self::new(3, 32)
    }
}

impl AutoQueryPolicy {
    /// Creates a policy from explicit promotion thresholds.
    pub const fn new(cache_after_runs: u32, cache_after_archetypes: usize) -> Self {
        Self {
            cache_after_runs,
            cache_after_archetypes,
        }
    }
}

/// Route identity for cache keys, discarding estimate-only fields of
/// `QueryRoute`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueryRouteKey {
    /// Archetype-order iteration.
    Archetype,
    /// Sparse-driver iteration over the given component.
    SparseDriver(ComponentId),
}

impl From<QueryRoute> for QueryRouteKey {
    fn from(route: QueryRoute) -> Self {
        match route {
            QueryRoute::Archetype { .. } => Self::Archetype,
            QueryRoute::SparseDriver(sparse_route) => Self::SparseDriver(sparse_route.driver()),
        }
    }
}

/// Identifies one validated work build.
///
/// Two runs share a split only when the world structure version, the selected
/// route, and the work configuration all match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueryWorkKey {
    /// World structure version the work was built from.
    structure_version: u64,
    /// Route the work was planned for.
    route: QueryRouteKey,
    /// Work configuration the work was split for.
    config: QueryWorkConfig,
}

/// One cached work split keyed by the world versions it was built from.
struct QueryWorkCacheEntry {
    /// Identity the current split was built from.
    key: QueryWorkKey,
    /// Validated split, reused while the key keeps matching.
    work: ValidatedQueryWork,
}

/// Retains one validated work split per execution mode for a cached query.
///
/// Entries are rebuilt in place when their key changes, so the work
/// allocation survives refreshes.
#[derive(Default)]
struct CachedQueryWork {
    /// Split used by sequential executions.
    sequential: Option<QueryWorkCacheEntry>,
    /// Split used by chunked executions.
    chunked: Option<Box<QueryWorkCacheEntry>>,
}

impl CachedQueryWork {
    /// Builds a fresh entry from the matched archetype list.
    fn build_entry<D, F>(
        cm: &ComponentManager,
        plan: &QueryPlan<D, F>,
        matched_archetypes: &[ArchetypeId],
        key: QueryWorkKey,
        config: QueryWorkConfig,
    ) -> QueryWorkCacheEntry
    where
        D: QueryData,
        F: QueryFilter,
    {
        let source: QueryArchetypeSource<'_, D, F> =
            QueryArchetypeSource::cached(matched_archetypes);
        let work = ValidatedQueryWork::build_from_planning(
            cm,
            source,
            plan.planning(),
            plan.data_state(),
            plan.filter_state(),
            config,
        );

        QueryWorkCacheEntry { key, work }
    }

    /// Rebuilds the entry in place when the key changed; an identical key
    /// keeps the split untouched.
    fn refresh_entry<D, F>(
        entry: &mut QueryWorkCacheEntry,
        cm: &ComponentManager,
        plan: &QueryPlan<D, F>,
        matched_archetypes: &[ArchetypeId],
        key: QueryWorkKey,
        config: QueryWorkConfig,
    ) where
        D: QueryData,
        F: QueryFilter,
    {
        if entry.key == key {
            return;
        }

        let source: QueryArchetypeSource<'_, D, F> =
            QueryArchetypeSource::cached(matched_archetypes);
        entry.work.rebuild_from_planning(
            cm,
            source,
            plan.planning(),
            plan.data_state(),
            plan.filter_state(),
            config,
        );
        entry.key = key;
    }

    /// Returns the validated work for the normalized `config`, building it on
    /// first use and rebuilding in place on key changes.
    fn get_or_build<'a, D, F>(
        &'a mut self,
        cm: &ComponentManager,
        planning: &WorldPlanning,
        plan: &QueryPlan<D, F>,
        matched_archetypes: &[ArchetypeId],
        config: QueryWorkConfig,
    ) -> &'a ValidatedQueryWork
    where
        D: QueryData,
        F: QueryFilter,
    {
        let config = config.normalized();
        let key = QueryWorkKey {
            structure_version: planning.query_work_version(),
            route: plan.route().into(),
            config,
        };

        match config.mode {
            QueryWorkMode::Sequential => {
                let entry = self.sequential.get_or_insert_with(|| {
                    Self::build_entry(cm, plan, matched_archetypes, key, config)
                });
                Self::refresh_entry(entry, cm, plan, matched_archetypes, key, config);
                &entry.work
            }
            QueryWorkMode::Chunked => {
                let entry = self
                    .chunked
                    .get_or_insert_with(|| {
                        Box::new(Self::build_entry(cm, plan, matched_archetypes, key, config))
                    })
                    .as_mut();
                Self::refresh_entry(entry, cm, plan, matched_archetypes, key, config);
                &entry.work
            }
        }
    }
}

#[cfg(test)]
mod tests;
