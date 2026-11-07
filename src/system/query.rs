//! Query parameter exposed to system functions.
//!
//! This wrapper is created from a cached [`QueryState`] plus shared access to the
//! world's component storage. Mutable items are produced through component-local
//! `UnsafeCell` values after the system access set has been validated.

use crate::{
    component::ComponentManager,
    entity::EntityManager,
    planning::WorldPlanning,
    query::{
        fetch::{mixed as f_mixed, QueryData, ReadQueryData, WriteQueryData},
        filter::QueryFilter,
        iter::{QueryIter, QueryIterMut},
        state::QueryState,
        QueryExecutionDebug, QueryParallelConfig,
    },
    task::TaskExecutorHandle,
};

/// System parameter used to iterate entities matching query data and filters.
///
/// `D` defines the data fetched for each entity and `F` defines the filter.
/// Read-only data exposes [`Query::iter`], while mutable data exposes
/// [`Query::iter_mut`].
pub struct Query<'w, 's, D, F = ()>
where
    D: QueryData,
    F: QueryFilter,
{
    components: &'w ComponentManager,
    planning: &'w WorldPlanning,
    entities: &'w EntityManager,
    /// Executor used by the parallel methods; when `None`, they take the
    /// sequential no-executor path.
    tasks: Option<&'w TaskExecutorHandle>,
    state: &'s mut QueryState<D, F>,
}

impl<'w, 's, D, F> Query<'w, 's, D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Creates a query parameter from component storage and cached state.
    ///
    /// The state's run window must already be open: pair this with
    /// [`QueryState::begin_run`] for the current run and close the window with
    /// [`QueryState::finish_run`] once the query is gone. A `tasks` value of
    /// `None` disables the executor-backed parallel paths.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other live query parameter can create
    /// references that conflict with this query's declared access.
    pub(crate) unsafe fn new(
        components: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        tasks: Option<&'w TaskExecutorHandle>,
        state: &'s mut QueryState<D, F>,
    ) -> Self {
        Self {
            components,
            planning,
            entities,
            tasks,
            state,
        }
    }
}

impl<'w, 's, D, F> Query<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Returns an iterator over read-only query items.
    #[inline]
    pub fn iter(&mut self) -> QueryIter<'w, '_, D, F> {
        self.state
            .iter(self.components, self.planning, self.entities)
    }

    /// Runs `func` for every read-only item matched by this query.
    #[inline]
    pub fn for_each<Func>(&mut self, func: Func)
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        self.state
            .for_each(self.components, self.planning, self.entities, func);
    }

    /// Runs `func` sequentially and returns diagnostics for the concrete
    /// execution path.
    #[inline]
    pub fn for_each_debug<Func>(&mut self, func: Func) -> QueryExecutionDebug
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        self.state
            .for_each_debug(self.components, self.planning, self.entities, func)
    }

    /// Runs `func` with the default parallel policy and returns the decision
    /// taken for this execution.
    #[inline]
    pub fn par_for_each<Func>(&mut self, func: Func) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::ReadQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        match self.tasks {
            Some(tasks) => {
                self.state
                    .par_for_each(self.components, self.planning, self.entities, tasks, func)
            }
            None => self.state.par_for_each_without_executor_with_config(
                self.components,
                self.planning,
                self.entities,
                QueryParallelConfig::auto(),
                func,
            ),
        }
    }

    /// Runs `func` with an explicit parallel policy and returns the decision
    /// taken for this execution.
    #[inline]
    pub fn par_for_each_with_config<Func>(
        &mut self,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::ReadQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::ReadQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        match self.tasks {
            Some(tasks) => self.state.par_for_each_with_config(
                self.components,
                self.planning,
                self.entities,
                tasks,
                config,
                func,
            ),
            None => self.state.par_for_each_without_executor_with_config(
                self.components,
                self.planning,
                self.entities,
                config,
                func,
            ),
        }
    }
}

impl<'w, 's, D, F> Query<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Returns an iterator over mutable query items.
    #[inline]
    pub fn iter_mut(&mut self) -> QueryIterMut<'w, '_, D, F> {
        self.state
            .iter_mut(self.components, self.planning, self.entities)
    }

    /// Runs `func` for every mutable item matched by this query.
    #[inline]
    pub fn for_each_mut<Func>(&mut self, func: Func)
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        self.state
            .for_each_mut(self.components, self.planning, self.entities, func);
    }

    /// Runs `func` sequentially and returns diagnostics for the concrete
    /// execution path.
    #[inline]
    pub fn for_each_mut_debug<Func>(&mut self, func: Func) -> QueryExecutionDebug
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'w>),
    {
        self.state
            .for_each_mut_debug(self.components, self.planning, self.entities, func)
    }

    /// Runs `func` with the default parallel policy and returns the decision
    /// taken for this execution.
    #[inline]
    pub fn par_for_each_mut<Func>(&mut self, func: Func) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::WriteQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        match self.tasks {
            Some(tasks) => self.state.par_for_each_mut(
                self.components,
                self.planning,
                self.entities,
                tasks,
                func,
            ),
            None => self.state.par_for_each_mut_without_executor_with_config(
                self.components,
                self.planning,
                self.entities,
                QueryParallelConfig::auto(),
                func,
            ),
        }
    }

    /// Runs `func` with an explicit parallel policy and returns the decision
    /// taken for this execution.
    #[inline]
    pub fn par_for_each_mut_with_config<Func>(
        &mut self,
        config: QueryParallelConfig,
        func: Func,
    ) -> QueryExecutionDebug
    where
        Func: Fn(<D as f_mixed::WriteQueryData>::Item<'w>) + Send + Sync,
        <D as f_mixed::WriteQueryData>::Item<'w>: Send,
        D::State: Sync,
        F::State: Sync,
    {
        match self.tasks {
            Some(tasks) => self.state.par_for_each_mut_with_config(
                self.components,
                self.planning,
                self.entities,
                tasks,
                config,
                func,
            ),
            None => self.state.par_for_each_mut_without_executor_with_config(
                self.components,
                self.planning,
                self.entities,
                config,
                func,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicI32, Ordering},
        Arc,
    };

    use millarecs_macros::Component;

    use crate::{
        component::archetype::ARCHETYPE_PAGE_ROWS,
        entity::{metadata::ArchetypeStorageKind, Entity},
        query::{
            filter::{Changed, With, Without},
            QueryExecutionRequest, QueryParallelChoice, QueryParallelConfig, QueryParallelMode,
            QueryParallelReason, QueryWorkMode,
        },
        system::{param::SystemParam, query::Query as SystemQuery, FunctionSystem, System},
        world::World,
    };

    #[derive(Default, Component, Debug, Clone, Copy, PartialEq, Eq)]
    struct Position {
        x: i32,
        y: i32,
    }

    #[derive(Default, Component, Debug, Clone, Copy, PartialEq, Eq)]
    struct Velocity {
        x: i32,
        y: i32,
    }

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Health(i32);

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Player;

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Frozen;

    fn setup_world() -> World {
        let mut world = World::new();
        world.register_table::<Position>().unwrap();
        world.register_table::<Velocity>().unwrap();
        world.register_table::<Player>().unwrap();
        world.register_sparse::<Health>().unwrap();
        world.register_sparse::<Frozen>().unwrap();
        world
    }

    #[test]
    fn component_storage_and_query_state_are_thread_transferable() {
        fn assert_send<T: Send>() {}
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<crate::component::ComponentManager>();
        assert_send::<crate::query::state::QueryState<&'static mut Position, ()>>();
        assert_send::<crate::query::state::QueryState<&'static mut Health, ()>>();
    }

    fn make_query<'w, 's, D, F>(
        world: &'w mut World,
        state: &'s mut crate::query::state::QueryState<D, F>,
    ) -> SystemQuery<'w, 's, D, F>
    where
        D: crate::query::fetch::QueryData,
        F: crate::query::filter::QueryFilter,
    {
        let execution_world = world.execution_world();
        unsafe {
            SystemQuery::new(
                execution_world.components(),
                execution_world.planning(),
                execution_world.entity_locations(),
                execution_world.task_executor(),
                state,
            )
        }
    }

    fn make_query_with_tasks<'w, 's, D, F>(
        world: &'w mut World,
        tasks: &'w crate::task::TaskExecutor,
        state: &'s mut crate::query::state::QueryState<D, F>,
    ) -> SystemQuery<'w, 's, D, F>
    where
        D: crate::query::fetch::QueryData,
        F: crate::query::filter::QueryFilter,
    {
        let execution_world = world.execution_world_with_tasks(Some(tasks.handle()));
        unsafe {
            SystemQuery::new(
                execution_world.components(),
                execution_world.planning(),
                execution_world.entity_locations(),
                execution_world.task_executor(),
                state,
            )
        }
    }

    #[test]
    fn iter_reads_table_components() {
        let mut world = setup_world();
        world.spawn((Position { x: 1, y: 2 },)).unwrap();
        world.spawn((Position { x: 3, y: 4 },)).unwrap();
        let mut state = world.query_state::<&Position, ()>().unwrap();
        let mut query = make_query(&mut world, &mut state);

        let mut positions = Vec::new();
        for position in query.iter() {
            positions.push(*position);
        }

        positions.sort_by_key(|position| position.x);
        assert_eq!(
            positions,
            vec![Position { x: 1, y: 2 }, Position { x: 3, y: 4 }]
        );
    }

    #[test]
    fn iter_mut_updates_table_components() {
        let mut world = setup_world();
        let entity = world.spawn((Position { x: 1, y: 2 },)).unwrap();
        let mut state = world.query_state::<&mut Position, ()>().unwrap();
        let mut query = make_query(&mut world, &mut state);

        for mut position in query.iter_mut() {
            position.x += 10;
            position.y += 20;
        }

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 11, y: 22 })
        );
    }

    #[test]
    fn iter_mut_updates_sparse_components() {
        let mut world = setup_world();
        let entity = world.spawn((Position { x: 0, y: 0 }, Health(100))).unwrap();
        let mut state = world.query_state::<&mut Health, ()>().unwrap();
        let mut query = make_query(&mut world, &mut state);

        for mut health in query.iter_mut() {
            health.0 -= 25;
        }

        assert_eq!(world.get::<Health>(entity), Some(&Health(75)));
    }

    #[test]
    fn iter_mut_updates_mixed_query_items() {
        let mut world = setup_world();
        let entity = world
            .spawn((Position { x: 1, y: 1 }, Velocity { x: 2, y: 3 }))
            .unwrap();
        let mut state = world
            .query_state::<(&mut Position, &Velocity), ()>()
            .unwrap();
        let mut query = make_query(&mut world, &mut state);

        for (mut position, velocity) in query.iter_mut() {
            position.x += velocity.x;
            position.y += velocity.y;
        }

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 3, y: 4 })
        );
    }

    #[test]
    fn iter_respects_with_filter() {
        let mut world = setup_world();
        let player = world.spawn((Position { x: 1, y: 1 }, Player)).unwrap();
        let other = world.spawn((Position { x: 2, y: 2 },)).unwrap();
        let mut state = world.query_state::<&Position, With<Player>>().unwrap();
        let mut query = make_query(&mut world, &mut state);

        let positions: Vec<_> = query.iter().copied().collect();

        assert_eq!(positions, vec![*world.get::<Position>(player).unwrap()]);
        assert_eq!(world.get::<Position>(other), Some(&Position { x: 2, y: 2 }));
    }

    #[test]
    fn iter_mut_respects_without_filter_for_sparse_components() {
        let mut world = setup_world();
        let normal = world.spawn((Position { x: 0, y: 0 },)).unwrap();
        let frozen = world.spawn((Position { x: 10, y: 10 }, Frozen)).unwrap();
        let mut state = world
            .query_state::<&mut Position, Without<Frozen>>()
            .unwrap();
        let mut query = make_query(&mut world, &mut state);

        for mut position in query.iter_mut() {
            position.x += 1;
            position.y += 1;
        }

        assert_eq!(
            world.get::<Position>(normal),
            Some(&Position { x: 1, y: 1 })
        );
        assert_eq!(
            world.get::<Position>(frozen),
            Some(&Position { x: 10, y: 10 })
        );
    }

    #[test]
    fn param_get_param_builds_working_query() {
        type Param = SystemQuery<'static, 'static, &'static mut Position>;

        let mut world = setup_world();
        let entity = world.spawn((Position { x: 4, y: 5 },)).unwrap();
        let mut state = <Param as SystemParam>::init_state(&mut world).unwrap();
        let execution_world = world.execution_world();
        let mut query =
            unsafe { <Param as SystemParam>::get_param(execution_world, &mut state) }.unwrap();

        for mut position in query.iter_mut() {
            position.x *= 2;
            position.y *= 2;
        }

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 8, y: 10 })
        );
    }

    #[test]
    fn disjoint_mutable_query_iterators_can_be_alive_together() {
        fn update(
            mut positions: SystemQuery<&mut Position>,
            mut velocities: SystemQuery<&mut Velocity>,
        ) {
            let mut positions = positions.iter_mut();
            let mut velocities = velocities.iter_mut();
            let mut position = positions.next().unwrap();
            let mut velocity = velocities.next().unwrap();

            position.x += 10;
            velocity.y += 20;
        }

        let mut world = setup_world();
        let entity = world
            .spawn((Position { x: 1, y: 2 }, Velocity { x: 3, y: 4 }))
            .unwrap();
        let mut system = FunctionSystem::new(update);

        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();
        system.run(execution_world).unwrap();

        assert_eq!(world.get::<Position>(entity).unwrap().x, 11);
        assert_eq!(world.get::<Velocity>(entity).unwrap().y, 24);
    }

    #[test]
    fn sparse_presence_can_coexist_with_sparse_value_mutation() {
        fn update(
            mut values: SystemQuery<&mut Health>,
            mut presence: SystemQuery<Entity, With<Health>>,
        ) {
            let mut values = values.iter_mut();
            let mut presence = presence.iter();
            let mut health = values.next().unwrap();

            assert!(presence.next().is_some());
            health.0 -= 10;
        }

        let mut world = setup_world();
        let entity = world.spawn((Health(50),)).unwrap();
        let mut system = FunctionSystem::new(update);

        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();
        system.run(execution_world).unwrap();

        assert_eq!(world.get::<Health>(entity), Some(&Health(40)));
    }

    #[test]
    fn mutually_exclusive_sparse_filters_split_one_mutable_column() {
        fn update(
            mut frozen: SystemQuery<&mut Position, With<Frozen>>,
            mut active: SystemQuery<&mut Position, Without<Frozen>>,
        ) {
            let mut frozen_iter = frozen.iter_mut();
            let mut active_iter = active.iter_mut();
            let mut frozen_position = frozen_iter.next().unwrap();
            let mut active_position = active_iter.next().unwrap();

            frozen_position.x += 100;
            active_position.x += 10;
        }

        let mut world = setup_world();
        let active = world.spawn((Position { x: 1, y: 0 },)).unwrap();
        let frozen = world.spawn((Position { x: 2, y: 0 }, Frozen)).unwrap();
        let mut system = FunctionSystem::new(update);

        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();
        system.run(execution_world).unwrap();

        assert_eq!(world.get::<Position>(active).unwrap().x, 11);
        assert_eq!(world.get::<Position>(frozen).unwrap().x, 102);
    }

    #[test]
    fn structural_constraints_run_before_change_filters() {
        fn update(
            mut active: SystemQuery<&mut Position, Without<Frozen>>,
            mut changed_frozen: SystemQuery<Entity, (Changed<Position>, With<Frozen>)>,
        ) {
            let mut active_iter = active.iter_mut();
            let mut active_position = active_iter.next().unwrap();

            let changed: Vec<_> = changed_frozen.iter().collect();
            assert_eq!(changed.len(), 1);

            active_position.x += 10;
        }

        let mut world = setup_world();
        let active = world.spawn((Position { x: 1, y: 0 },)).unwrap();
        let frozen = world.spawn((Position { x: 2, y: 0 }, Frozen)).unwrap();
        let mut system = FunctionSystem::new(update);

        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();
        system.run(execution_world).unwrap();

        assert_eq!(world.get::<Position>(active).unwrap().x, 11);
        assert_eq!(world.get::<Position>(frozen).unwrap().x, 2);
    }

    #[test]
    fn par_for_each_reads_table_components() {
        let mut world = setup_world();

        for x in 0..256 {
            world.spawn((Position { x, y: 0 },)).unwrap();
        }

        let tasks = crate::task::TaskExecutor::new(4);
        let sum = Arc::new(AtomicI32::new(0));
        let mut state = world.query_state::<&Position, ()>().unwrap();

        {
            let mut query = make_query_with_tasks(&mut world, &tasks, &mut state);
            let sum = Arc::clone(&sum);

            query.par_for_each(move |position| {
                sum.fetch_add(position.x, Ordering::Relaxed);
            });
        }

        assert_eq!(sum.load(Ordering::Relaxed), (0..256).sum());
    }

    #[test]
    fn par_for_each_reports_missing_task_executor_fallback() {
        let mut world = setup_world();

        for x in 0..8 {
            world.spawn((Position { x, y: 0 },)).unwrap();
        }

        let sum = Arc::new(AtomicI32::new(0));
        let mut state = world.query_state::<&Position, ()>().unwrap();
        let mut query = make_query(&mut world, &mut state);
        let sum_for_query = Arc::clone(&sum);

        let debug = query.par_for_each(move |position| {
            sum_for_query.fetch_add(position.x, Ordering::Relaxed);
        });

        assert_eq!(sum.load(Ordering::Relaxed), (0..8).sum());
        assert_eq!(debug.request, QueryExecutionRequest::Parallel);
        assert_eq!(debug.selected, QueryParallelChoice::Sequential);
        assert_eq!(debug.reason, QueryParallelReason::NotEnoughWorkers);
        assert_eq!(debug.worker_count, 0);
        assert!(debug.parallel_config.is_some());
    }

    #[test]
    fn par_for_each_with_config_reports_parallel_selection() {
        let mut world = setup_world();
        let rows = ARCHETYPE_PAGE_ROWS * 4;

        for x in 0..rows {
            world.spawn((Position { x: x as i32, y: 0 },)).unwrap();
        }

        let tasks = crate::task::TaskExecutor::new(4);
        let sum = Arc::new(AtomicI32::new(0));
        let mut state = world.query_state::<&Position, ()>().unwrap();
        let config = QueryParallelConfig {
            mode: QueryParallelMode::Always,
            min_workers: 2,
            min_units: 2,
            min_estimated_rows: 1,
            target_rows_per_unit: ARCHETYPE_PAGE_ROWS,
            max_pages_per_unit: 1,
        };

        {
            let mut query = make_query_with_tasks(&mut world, &tasks, &mut state);
            let sum_for_query = Arc::clone(&sum);

            let debug = query.par_for_each_with_config(config, move |position| {
                sum_for_query.fetch_add(position.x, Ordering::Relaxed);
            });

            assert_eq!(debug.request, QueryExecutionRequest::Parallel);
            assert_eq!(debug.selected, QueryParallelChoice::Parallel);
            assert_eq!(debug.reason, QueryParallelReason::ForcedParallel);
            assert_eq!(debug.worker_count, 4);
            assert_eq!(debug.work.mode, QueryWorkMode::Chunked);
            assert!(debug.work.unit_count >= 2);
            assert_eq!(
                debug.parallel_config.unwrap().mode,
                QueryParallelMode::Always
            );
        }

        assert_eq!(sum.load(Ordering::Relaxed), (0..rows as i32).sum());
    }

    #[test]
    fn par_for_each_mut_updates_contiguous_and_fragmented_rows() {
        let mut world = setup_world();
        let mut entities = Vec::new();

        for x in 0..256 {
            let storage = if x % 2 == 0 {
                ArchetypeStorageKind::Contiguous
            } else {
                ArchetypeStorageKind::Fragmented
            };

            let entity = world
                .spawn_with_storage(storage, (Position { x, y: 0 },))
                .unwrap();
            entities.push((entity, x));
        }

        let tasks = crate::task::TaskExecutor::new(4);
        let mut state = world.query_state::<&mut Position, ()>().unwrap();

        {
            let mut query = make_query_with_tasks(&mut world, &tasks, &mut state);

            query.par_for_each_mut(|mut position| {
                position.x += 10;
            });
        }

        for (entity, initial_x) in entities {
            assert_eq!(world.get::<Position>(entity).unwrap().x, initial_x + 10);
        }
    }
}
