//! Borrowed query wrappers used by the world/system APIs.
//!
//! The wrappers keep a `QueryState` together with the component manager borrow
//! required to create iterators. They provide the public-facing object returned
//! by higher-level ECS APIs.

use crate::{
    component::ComponentManager,
    entity::EntityManager,
    planning::WorldPlanning,
    query::{
        fetch::{mixed as f_mixed, ReadQueryData, WriteQueryData},
        filter::QueryFilter,
        iter::{QueryIter, QueryIterMut},
        state::QueryState,
    },
};

/// Owned read-only query wrapper.
///
/// Stores a freshly created `QueryState` together with a shared component-manager
/// borrow.
pub struct WorldQuery<'w, D, F = ()>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    components: &'w ComponentManager,
    planning: &'w WorldPlanning,
    entities: &'w EntityManager,
    state: QueryState<D, F>,
}

impl<'w, D, F> WorldQuery<'w, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Creates a read-only world query from a component manager and cached state.
    pub(crate) fn new(
        components: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        state: QueryState<D, F>,
    ) -> Self {
        Self {
            components,
            planning,
            entities,
            state,
        }
    }

    /// Creates an iterator over the current matching entities.
    #[inline]
    pub fn iter(&mut self) -> QueryIter<'w, '_, D, F> {
        self.state
            .iter(self.components, self.planning, self.entities)
    }

    /// Runs `func` for every current matching read-only item.
    #[inline]
    pub fn for_each<Func>(&mut self, func: Func)
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        self.state
            .for_each(self.components, self.planning, self.entities, func);
    }
}

/// Owned mutable query wrapper.
///
/// Keeps exclusive access to the component manager while mutable iterators are
/// created from the stored query state.
pub struct WorldQueryMut<'w, D, F = ()>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    components: &'w mut ComponentManager,
    planning: &'w WorldPlanning,
    entities: &'w EntityManager,
    state: QueryState<D, F>,
}

impl<'w, D, F> WorldQueryMut<'w, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Creates a mutable world query from a component manager and cached state.
    pub(crate) fn new(
        components: &'w mut ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        state: QueryState<D, F>,
    ) -> Self {
        Self {
            components,
            planning,
            entities,
            state,
        }
    }

    /// Creates a mutable iterator over the current matching entities.
    #[inline]
    pub fn iter_mut(&mut self) -> QueryIterMut<'_, '_, D, F> {
        self.state
            .iter_mut(self.components, self.planning, self.entities)
    }

    /// Runs `func` for every current matching mutable item.
    #[inline]
    pub fn for_each_mut<'a, Func>(&'a mut self, func: Func)
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'a>),
    {
        self.state
            .for_each_mut(self.components, self.planning, self.entities, func);
    }
}

/// Borrowed read-only query wrapper.
///
/// Used when an existing query state is owned by another object, such as a
/// system parameter.
pub struct WorldQueryRef<'w, 's, D, F = ()>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    components: &'w ComponentManager,
    planning: &'w WorldPlanning,
    entities: &'w EntityManager,
    state: &'s mut QueryState<D, F>,
}

impl<D: ReadQueryData, F: QueryFilter> Drop for WorldQueryRef<'_, '_, D, F> {
    /// Closes the borrowed run, advancing the state's change-detection window
    /// even when iteration stopped early.
    fn drop(&mut self) {
        self.state.finish_run();
    }
}

impl<'w, 's, D, F> WorldQueryRef<'w, 's, D, F>
where
    D: ReadQueryData,
    F: QueryFilter,
{
    /// Creates a borrowed read-only query wrapper.
    pub(crate) fn new(
        components: &'w ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        state: &'s mut QueryState<D, F>,
    ) -> Self {
        Self {
            components,
            planning,
            entities,
            state,
        }
    }

    /// Creates an iterator using the borrowed state.
    #[inline]
    pub fn iter(&mut self) -> QueryIter<'w, '_, D, F> {
        self.state
            .iter(self.components, self.planning, self.entities)
    }

    /// Runs `func` for every current matching read-only item.
    #[inline]
    pub fn for_each<Func>(&mut self, func: Func)
    where
        Func: FnMut(<D as f_mixed::ReadQueryData>::Item<'w>),
    {
        self.state
            .for_each(self.components, self.planning, self.entities, func);
    }
}

/// Borrowed mutable query wrapper.
///
/// Used when an existing mutable query state is owned by another object, such
/// as a system parameter.
pub struct WorldQueryMutRef<'w, 's, D, F = ()>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    components: &'w mut ComponentManager,
    planning: &'w WorldPlanning,
    entities: &'w EntityManager,
    state: &'s mut QueryState<D, F>,
}

impl<D: WriteQueryData, F: QueryFilter> Drop for WorldQueryMutRef<'_, '_, D, F> {
    /// Closes the borrowed run, advancing the state's change-detection window
    /// even when iteration stopped early.
    fn drop(&mut self) {
        self.state.finish_run();
    }
}

impl<'w, 's, D, F> WorldQueryMutRef<'w, 's, D, F>
where
    D: WriteQueryData,
    F: QueryFilter,
{
    /// Creates a borrowed mutable query wrapper.
    pub(crate) fn new(
        components: &'w mut ComponentManager,
        planning: &'w WorldPlanning,
        entities: &'w EntityManager,
        state: &'s mut QueryState<D, F>,
    ) -> Self {
        Self {
            components,
            planning,
            entities,
            state,
        }
    }

    /// Creates a mutable iterator using the borrowed state.
    #[inline]
    pub fn iter_mut(&mut self) -> QueryIterMut<'_, '_, D, F> {
        self.state
            .iter_mut(self.components, self.planning, self.entities)
    }

    /// Runs `func` for every current matching mutable item.
    #[inline]
    pub fn for_each_mut<'a, Func>(&'a mut self, func: Func)
    where
        Func: FnMut(<D as f_mixed::WriteQueryData>::Item<'a>),
    {
        self.state
            .for_each_mut(self.components, self.planning, self.entities, func);
    }
}
