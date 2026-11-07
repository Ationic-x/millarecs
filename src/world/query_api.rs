//! Query construction APIs on [`World`].
use super::World;
use crate::{
    query::{
        fetch::{QueryData, ReadQueryData, WriteQueryData},
        filter::QueryFilter,
        state::{AutoQueryPolicy, QueryState},
        world_query::{WorldQuery, WorldQueryMut, WorldQueryMutRef, WorldQueryRef},
        QueryPlanDebug,
    },
    world::WorldError,
};
impl World {
    /// Creates an automatic query state that can be reused across calls.
    pub fn query_state<D, F>(&self) -> Result<QueryState<D, F>, WorldError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        Ok(QueryState::new(
            self.id,
            self.components(),
            self.planning(),
        )?)
    }

    /// Creates a query state that always keeps matched archetypes cached.
    pub fn query_state_cached<D, F>(&self) -> Result<QueryState<D, F>, WorldError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        Ok(QueryState::cached(
            self.id,
            self.components(),
            self.planning(),
        )?)
    }

    /// Creates a query state that scans archetypes on demand without retaining
    /// the matched archetype list.
    pub fn query_state_uncached<D, F>(&self) -> Result<QueryState<D, F>, WorldError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        Ok(QueryState::uncached(
            self.id,
            self.components(),
            self.planning(),
        )?)
    }

    /// Creates an automatic query state with explicit promotion thresholds.
    pub fn query_state_auto_with<D, F>(
        &self,
        policy: AutoQueryPolicy,
    ) -> Result<QueryState<D, F>, WorldError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        Ok(QueryState::auto(
            self.id,
            self.components(),
            self.planning(),
            policy,
        )?)
    }

    /// Builds a read-only diagnostic report for a freshly created automatic
    /// query state.
    pub fn query_plan_debug<D, F>(&self) -> Result<QueryPlanDebug, WorldError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        let state: QueryState<D, F> = QueryState::new(self.id, self.components(), self.planning())?;
        self.query_state_debug(&state)
    }

    /// Builds a read-only diagnostic report for an existing query state.
    ///
    /// The report does not refresh cached archetype lists. For cached states it
    /// reports both the retained cache counters and the current world match
    /// count computed by scanning archetypes.
    pub fn query_state_debug<D, F>(
        &self,
        state: &QueryState<D, F>,
    ) -> Result<QueryPlanDebug, WorldError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        state.validate_world_id(self.id)?;
        Ok(QueryPlanDebug::from_query_state(self.components(), state))
    }

    /// Creates an immutable one-shot query using the automatic cache policy.
    ///
    /// One-shot states never run again, so paying the automatic promotion
    /// (second archetype scan plus cached work construction) would be pure
    /// waste. Scan-on-demand iteration matches one run exactly.
    pub fn query<D, F>(&self) -> Result<WorldQuery<'_, D, F>, WorldError>
    where
        D: ReadQueryData,
        F: QueryFilter,
    {
        let world_id = self.id;
        let mut state = QueryState::uncached(world_id, self.components(), self.planning())?;
        state.begin_run(world_id, self.change_tick)?;

        Ok(WorldQuery::new(
            &self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates an immutable one-shot query with retained matched archetypes.
    pub fn query_cached<D, F>(&self) -> Result<WorldQuery<'_, D, F>, WorldError>
    where
        D: ReadQueryData,
        F: QueryFilter,
    {
        let world_id = self.id;
        let mut state = QueryState::cached(world_id, self.components(), self.planning())?;
        state.begin_run(world_id, self.change_tick)?;

        Ok(WorldQuery::new(
            &self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates an immutable one-shot query that scans archetypes on demand.
    pub fn query_uncached<D, F>(&self) -> Result<WorldQuery<'_, D, F>, WorldError>
    where
        D: ReadQueryData,
        F: QueryFilter,
    {
        let world_id = self.id;
        let mut state = QueryState::uncached(world_id, self.components(), self.planning())?;
        state.begin_run(world_id, self.change_tick)?;

        Ok(WorldQuery::new(
            &self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates a mutable one-shot query using the automatic cache policy.
    ///
    /// Scan-on-demand matches a single run exactly; see [`Self::query`].
    pub fn query_mut<D, F>(&mut self) -> Result<WorldQueryMut<'_, D, F>, WorldError>
    where
        D: WriteQueryData,
        F: QueryFilter,
    {
        let world_id = self.id();
        let mut state = QueryState::uncached(world_id, self.components(), self.planning())?;
        let tick = self.advance_tick();
        state.begin_run(world_id, tick)?;

        Ok(WorldQueryMut::new(
            &mut self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates a mutable one-shot query with retained matched archetypes.
    pub fn query_mut_cached<D, F>(&mut self) -> Result<WorldQueryMut<'_, D, F>, WorldError>
    where
        D: WriteQueryData,
        F: QueryFilter,
    {
        let world_id = self.id();
        let mut state = QueryState::cached(world_id, self.components(), self.planning())?;
        let tick = self.advance_tick();
        state.begin_run(world_id, tick)?;

        Ok(WorldQueryMut::new(
            &mut self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates a mutable one-shot query that scans archetypes on demand.
    pub fn query_mut_uncached<D, F>(&mut self) -> Result<WorldQueryMut<'_, D, F>, WorldError>
    where
        D: WriteQueryData,
        F: QueryFilter,
    {
        let world_id = self.id();
        let mut state = QueryState::uncached(world_id, self.components(), self.planning())?;
        let tick = self.advance_tick();
        state.begin_run(world_id, tick)?;

        Ok(WorldQueryMut::new(
            &mut self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates an immutable query from a reusable state.
    pub fn query_with<'w, 's, D, F>(
        &'w self,
        state: &'s mut QueryState<D, F>,
    ) -> Result<WorldQueryRef<'w, 's, D, F>, WorldError>
    where
        D: ReadQueryData,
        F: QueryFilter,
    {
        state.begin_run(self.id, self.change_tick)?;
        Ok(WorldQueryRef::new(
            &self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates a mutable query from a reusable state.
    pub fn query_mut_with<'w, 's, D, F>(
        &'w mut self,
        state: &'s mut QueryState<D, F>,
    ) -> Result<WorldQueryMutRef<'w, 's, D, F>, WorldError>
    where
        D: WriteQueryData,
        F: QueryFilter,
    {
        let world_id = self.id;

        // Validate the world id before the world's change tick
        state.validate_world_id(world_id)?;

        let tick = self.advance_tick();
        state.begin_run(self.id, tick)?;

        Ok(WorldQueryMutRef::new(
            &mut self.components,
            &self.world_planning,
            &self.entities,
            state,
        ))
    }

    /// Creates an immutable query without filters.
    pub fn query_all<D>(&self) -> Result<WorldQuery<'_, D, ()>, WorldError>
    where
        D: ReadQueryData,
    {
        self.query::<D, ()>()
    }

    /// Creates a mutable query without filters.
    pub fn query_all_mut<D>(&mut self) -> Result<WorldQueryMut<'_, D, ()>, WorldError>
    where
        D: WriteQueryData,
    {
        self.query_mut::<D, ()>()
    }

    /// Runs `func` for every item produced by an immutable query.
    pub fn query_each<D, F, Func>(&self, func: Func) -> Result<(), WorldError>
    where
        D: ReadQueryData,
        F: QueryFilter,
        Func: for<'w> FnMut(<D as crate::query::fetch::mixed::ReadQueryData>::Item<'w>),
    {
        let mut query = self.query::<D, F>()?;

        query.for_each(func);

        Ok(())
    }

    /// Runs `func` for every item produced by a mutable query.
    pub fn query_each_mut<D, F, Func>(&mut self, func: Func) -> Result<(), WorldError>
    where
        D: WriteQueryData,
        F: QueryFilter,
        Func: for<'w> FnMut(<D as crate::query::fetch::mixed::WriteQueryData>::Item<'w>),
    {
        let mut query = self.query_mut::<D, F>()?;

        query.for_each_mut(func);

        Ok(())
    }
}
