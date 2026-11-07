//! Query filters.
//!
//! Filters can reject whole archetypes when the component is table-backed, or
//! reject individual entities when the component is stored in a sparse set.

use std::{any::type_name, marker::PhantomData};

use millarecs_macros::all_tuples;

use crate::{
    change_detection::{ComponentTicks, TickWindow},
    component::{
        archetype::Archetype,
        sparse_set::{ComponentSparseSet, SparseSetManager},
        Component, ComponentId, ComponentManager, StorageKind,
    },
    entity::{metadata::ArchetypeStorageKind, Entity},
    query::{
        access::FilterAccess,
        debug::{QueryDebugItem, QueryDebugSource},
        planning::{QueryRouteCandidate, RouteCandidateRole},
        AccessMask,
    },
};

use super::{state::ComponentQueryState, QueryError};

/// Fetch used by component filters.
///
/// Table-stored components do not need stored pointers, because membership can
/// be decided from the archetype signature. Sparse-set components store a
/// pointer to the set so each entity can be checked individually.
pub enum FilterCompFetch<'w, C: Component> {
    /// Filter over a table-stored component.
    Table(PhantomData<&'w C>),

    /// Filter over a sparse-set component.
    Sparse {
        /// The set whose membership decides the filter per entity.
        set: *const ComponentSparseSet<C>,
        /// Binds the set pointer to the fetch lifetime.
        _marker: PhantomData<&'w C>,
    },

    /// Presence is guaranteed by the sparse set driving this query unit.
    SparseDriver(PhantomData<&'w C>),
}

/// Initializes the state of a required component filter.
pub(crate) fn init_required_filter_state<C: Component>(
    cm: &ComponentManager,
    access: &mut FilterAccess,
    mask: AccessMask,
) -> Result<ComponentQueryState, QueryError> {
    let state = ComponentQueryState::new::<C>(cm)?;
    let id = state.id;
    access.add_required(id, mask)?;
    Ok(state)
}

/// Initializes the state of an excluded (`Without`) component filter.
pub(crate) fn init_excluded_filter_state<C: Component>(
    cm: &ComponentManager,
    access: &mut FilterAccess,
    mask: AccessMask,
) -> Result<ComponentQueryState, QueryError> {
    let state = ComponentQueryState::new::<C>(cm)?;
    let id = state.id;
    access.add_excluded(id, mask)?;
    Ok(state)
}

/// Initializes the fetch for a filter.
///
/// # Safety
///
/// `sparse_sets` must point to a valid `SparseSetManager` compatible with the
/// `ComponentManager` used to create the filter state.
pub(crate) unsafe fn init_filter_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    sparse_sets: *const SparseSetManager,
) -> Result<FilterCompFetch<'w, C>, QueryError> {
    Ok(match state.storage {
        StorageKind::Table => FilterCompFetch::Table(PhantomData),
        StorageKind::Sparse => {
            let sets = unsafe { &*sparse_sets };
            let set = sets
                .get_sparse_set(state.id.index())
                .ok_or(QueryError::StorageUnknown(state.id))?;

            FilterCompFetch::Sparse {
                set,
                _marker: PhantomData,
            }
        }
    })
}

/// Builds the presence-filter fetch used for rows produced by a sparse
/// conductor.
///
/// Upgrades the plain sparse variant to [`FilterCompFetch::SparseDriver`] when
/// the filtered component is the driver itself, so membership checks become
/// trivial.
///
/// # Safety
///
/// `sparse_sets` must point to a valid `SparseSetManager` compatible with the
/// `ComponentManager` used to create the filter state, and `driver` must
/// identify the sparse set that produced the rows evaluated with this fetch.
unsafe fn init_sparse_driver_filter_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    sparse_sets: *const SparseSetManager,
    driver: ComponentId,
) -> Result<FilterCompFetch<'w, C>, QueryError> {
    let fetch = unsafe { init_filter_fetch(state, sparse_sets) }?;

    match fetch {
        FilterCompFetch::Sparse { .. } if state.id == driver => {
            Ok(FilterCompFetch::SparseDriver(PhantomData))
        }
        other => Ok(other),
    }
}

/// Returns whether an archetype may satisfy a `With<T>` filter.
///
/// For table-stored components, this checks the archetype signature. For sparse
/// sets, this returns `true` because the real check must be performed per
/// entity.
pub(crate) fn matches_with_archetype(state: &ComponentQueryState, archetype: &Archetype) -> bool {
    match state.storage {
        StorageKind::Table => archetype.signature().contains(state.id),
        StorageKind::Sparse => true,
    }
}

/// Returns whether an archetype may satisfy a `Without<T>` filter.
///
/// For table-stored components, this checks the archetype signature. For sparse
/// sets, this returns `true` because the real check must be performed per
/// entity.
pub(crate) fn matches_without_archetype(
    state: &ComponentQueryState,
    archetype: &Archetype,
) -> bool {
    match state.storage {
        StorageKind::Table => !archetype.signature().contains(state.id),
        StorageKind::Sparse => true,
    }
}

/// Evaluates a `With<T>` filter for a specific entity.
///
/// Table-stored membership was already settled by the archetype signature and
/// the sparse driver guarantees presence, so both variants accept every row;
/// only the plain sparse variant consults the set per entity.
///
/// # Safety
///
/// The fetch must have been initialized for the correct storage.
pub(crate) unsafe fn evaluate_with_filter<'w, C: Component>(
    fetch: &mut FilterCompFetch<'w, C>,
    entity: Entity,
) -> bool {
    match fetch {
        FilterCompFetch::Table(_) => true,
        FilterCompFetch::SparseDriver(_) => true,
        FilterCompFetch::Sparse { set, .. } => {
            let set = unsafe { &**set };
            set.contains_entity(entity)
        }
    }
}

/// Evaluates a `Without<T>` filter for a specific entity.
///
/// The asymmetry with `With<T>` sits in the driver variant: the conductor
/// only produces rows holding its own component, so those rows always fail a
/// `Without<T>` over it and are rejected outright.
///
/// # Safety
///
/// The fetch must have been initialized for the correct storage.
pub(crate) unsafe fn evaluate_without_filter<'w, C: Component>(
    fetch: &mut FilterCompFetch<'w, C>,
    entity: Entity,
) -> bool {
    match fetch {
        FilterCompFetch::Table(_) => true,
        FilterCompFetch::SparseDriver(_) => false,
        FilterCompFetch::Sparse { set, .. } => {
            let set = unsafe { &**set };
            !set.contains_entity(entity)
        }
    }
}

/// Filter that can be applied to a query.
///
/// A filter may reject entire archetypes or individual entities.
/// Implemented for `With<T>`, `Without<T>`, `Added<T>`, `Changed<T>`, `()`,
/// tuples of filters, and combinators such as `Not<T>`, `Or<T>` and
/// `AnyOf<T>`.
///
/// # Safety
///
/// Implementors must declare every component region read by `filter`, and each
/// fetch must remain within those regions for its entire lifetime.
pub unsafe trait QueryFilter {
    /// Cached filter state.
    type State: Send + 'static;

    /// Fetch prepared for evaluating the filter during iteration.
    type Fetch<'w>;

    /// Initializes the filter state.
    fn init_state(
        cm: &ComponentManager,
        access: &mut FilterAccess,
    ) -> Result<Self::State, QueryError>;

    /// Returns whether an archetype may contain entities matching the filter.
    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool;

    /// Returns whether this filter can be resolved using archetype information only.
    fn is_archetypal(state: &Self::State) -> bool;

    /// Returns a component whose change ticks this filter checks, if any.
    ///
    /// Execution consults the column's last write-access tick to skip whole
    /// work units where the filter provably cannot match. Compositions with
    /// any-match semantics (`Or`, `AnyOf`, `Not`) must keep the default
    /// `None`: skipping on one cold member would be unsound for them.
    fn tick_sensitive_component(_state: &Self::State) -> Option<ComponentId> {
        None
    }

    /// Initializes the filter fetch for a specific archetype.
    ///
    /// # Safety
    ///
    /// The received pointers must be valid for `'w`.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        tick_window: TickWindow,
    ) -> Result<Self::Fetch<'w>, QueryError>;

    /// Initializes a filter for work produced by a sparse-set conductor.
    ///
    /// # Safety
    ///
    /// The received pointers must be valid for `'w`, and `driver` must identify
    /// the sparse set that produced the rows evaluated with this fetch.
    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        tick_window: TickWindow,
        _driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { Self::init_fetch(state, archetype, storage, sparse_sets, tick_window) }
    }

    /// Evaluates stable component-presence constraints for one entity.
    ///
    /// Iterators run this phase for the complete filter before evaluating
    /// change predicates or fetching query data. This ordering is required for
    /// filter domains to prove that later accesses target disjoint rows.
    ///
    /// # Safety
    ///
    /// The fetch must have been initialized for the current archetype.
    unsafe fn matches_entity<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> bool {
        true
    }

    /// Evaluates stable constraints using the driver's dense position.
    ///
    /// # Safety
    ///
    /// The fetch must have been initialized for the current archetype and
    /// `sparse_dense_pos` must belong to `entity` in the selected driver.
    unsafe fn matches_entity_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        _sparse_dense_pos: u32,
    ) -> bool {
        unsafe { Self::matches_entity(fetch, row, entity) }
    }

    /// Evaluates whether a specific entity satisfies the filter.
    ///
    /// # Safety
    ///
    /// The fetch must have been initialized for the current archetype.
    unsafe fn filter<'w>(fetch: &mut Self::Fetch<'w>, row: usize, entity: Entity) -> bool;

    /// Evaluates the filter using the driver's dense position.
    ///
    /// # Safety
    ///
    /// The fetch must have been initialized for the current archetype and
    /// `sparse_dense_pos` must belong to `entity` in the selected driver.
    unsafe fn filter_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        _sparse_dense_pos: u32,
    ) -> bool {
        unsafe { Self::filter(fetch, row, entity) }
    }

    /// Reports sparse-route candidacy to the planner.
    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        let _ = state;
        let _ = out;
    }

    /// Returns whether this filter is satisfied by the selected sparse driver,
    /// letting execution skip its per-row evaluation.
    fn is_satisfied_by_sparse_driver(state: &Self::State, driver: ComponentId) -> bool {
        let _ = (state, driver);
        false
    }

    /// Reports structured diagnostics for the debug report.
    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        let _ = state;
        let _ = out;
    }
}

/// Filter that keeps only entities with component `T`.
#[derive(Debug, Clone, Copy, Default)]
pub struct With<T>(PhantomData<fn() -> T>);

unsafe impl<C: Component> QueryFilter for With<C> {
    type State = ComponentQueryState;
    type Fetch<'w> = FilterCompFetch<'w, C>;

    fn init_state(
        cm: &ComponentManager,
        access: &mut FilterAccess,
    ) -> Result<Self::State, QueryError> {
        init_required_filter_state::<C>(cm, access, AccessMask::STRUCTURE_READ)
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        matches_with_archetype(state, archetype)
    }

    fn is_archetypal(state: &Self::State) -> bool {
        matches!(state.storage, StorageKind::Table)
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        _: TickWindow,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_filter_fetch(state, sparse_sets) }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        _: TickWindow,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_sparse_driver_filter_fetch(state, sparse_sets, driver) }
    }

    unsafe fn matches_entity<'w>(fetch: &mut Self::Fetch<'w>, _: usize, entity: Entity) -> bool {
        unsafe { evaluate_with_filter(fetch, entity) }
    }

    unsafe fn filter<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> bool {
        true
    }

    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        out.push(QueryRouteCandidate::required_filter(
            state.id,
            state.storage,
        ));
    }

    fn is_satisfied_by_sparse_driver(state: &Self::State, driver: ComponentId) -> bool {
        state.storage == StorageKind::Sparse && state.id == driver
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::component(
            QueryDebugSource::Filter,
            "With<T>",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::STRUCTURE_READ,
            Some(RouteCandidateRole::RequiredFilter),
        ));
    }
}

/// Filter that keeps only entities without component `T`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Without<T>(PhantomData<fn() -> T>);

unsafe impl<C: Component> QueryFilter for Without<C> {
    type State = ComponentQueryState;
    type Fetch<'w> = FilterCompFetch<'w, C>;

    fn init_state(
        cm: &ComponentManager,
        access: &mut FilterAccess,
    ) -> Result<Self::State, QueryError> {
        init_excluded_filter_state::<C>(cm, access, AccessMask::STRUCTURE_READ)
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        matches_without_archetype(state, archetype)
    }

    fn is_archetypal(state: &Self::State) -> bool {
        matches!(state.storage, StorageKind::Table)
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        _: TickWindow,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_filter_fetch(state, sparse_sets) }
    }

    unsafe fn matches_entity<'w>(fetch: &mut Self::Fetch<'w>, _: usize, entity: Entity) -> bool {
        unsafe { evaluate_without_filter(fetch, entity) }
    }

    unsafe fn filter<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> bool {
        true
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::excluded_filter(
            "Without<T>",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::STRUCTURE_READ,
        ));
    }
}

unsafe impl QueryFilter for () {
    type State = ();
    type Fetch<'w> = ();

    fn init_state(_: &ComponentManager, _: &mut FilterAccess) -> Result<Self::State, QueryError> {
        Ok(())
    }

    fn matches_archetype(_: &Self::State, _: &Archetype) -> bool {
        true
    }

    fn is_archetypal(_: &Self::State) -> bool {
        true
    }

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
        _: *const SparseSetManager,
        _: TickWindow,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        Ok(())
    }

    unsafe fn filter<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> bool {
        true
    }
}

/// Implements `QueryFilter` for tuples of filters.
macro_rules! tuple_query_filter_impl {
    ($(#[$meta:meta])* $($T:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($T),*> QueryFilter for ($($T,)*)
        where
            $($T: QueryFilter),*
        {
            type State = ($($T::State,)*);
            type Fetch<'w> = ($($T::Fetch<'w>,)*);

            fn init_state(cm: &ComponentManager, access: &mut FilterAccess) -> Result<Self::State, QueryError> {
                Ok((
                    $($T::init_state(cm, access)?,)*
                ))
            }

            fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
                let ($(ref $T,)*) = *state;
                $($T::matches_archetype($T, archetype) &&)* true
            }

            fn is_archetypal(state: &Self::State) -> bool {
                let ($(ref $T,)*) = *state;
                $($T::is_archetypal($T) &&)* true
            }

            fn tick_sensitive_component(state: &Self::State) -> Option<ComponentId> {
                let ($(ref $T,)*) = *state;
                let mut found = None;
                $(if found.is_none() {
                    found = $T::tick_sensitive_component($T);
                })*
                found
            }

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                storage: ArchetypeStorageKind,
                sparse_sets: *const SparseSetManager,
                tick_window: TickWindow,
            ) -> Result<Self::Fetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;
                Ok((
                    $(unsafe { $T::init_fetch($T, archetype, storage, sparse_sets, tick_window)? },)*
                ))
            }

            unsafe fn init_sparse_driver_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                storage: ArchetypeStorageKind,
                sparse_sets: *const SparseSetManager,
                tick_window: TickWindow,
                driver: ComponentId,
            ) -> Result<Self::Fetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;
                Ok((
                    $(unsafe {
                        $T::init_sparse_driver_fetch(
                            $T,
                            archetype,
                            storage,
                            sparse_sets,
                            tick_window,
                            driver,
                        )?
                    },)*
                ))
            }

            unsafe fn matches_entity<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
            ) -> bool {
                let ($(ref mut $T,)*) = *fetch;
                unsafe { $($T::matches_entity($T, row, entity) &&)* true }
            }

            unsafe fn matches_entity_sparse_driver<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
                sparse_dense_pos: u32,
            ) -> bool {
                let ($(ref mut $T,)*) = *fetch;
                unsafe {
                        $($T::matches_entity_sparse_driver(
                        $T,
                        row,
                        entity,
                        sparse_dense_pos,
                    ) &&)* true
                }
            }

            unsafe fn filter<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
            ) -> bool {
                let ($(ref mut $T,)*) = *fetch;
                unsafe { $($T::filter($T, row, entity) &&)* true }
            }

            unsafe fn filter_sparse_driver<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
                sparse_dense_pos: u32,
            ) -> bool {
                let ($(ref mut $T,)*) = *fetch;
                unsafe {
                    $($T::filter_sparse_driver(
                        $T,
                        row,
                        entity,
                        sparse_dense_pos,
                    ) &&)* true
                }
            }

            fn is_satisfied_by_sparse_driver(
                state: &Self::State,
                driver: ComponentId,
            ) -> bool {
                let ($(ref $T,)*) = *state;
                $($T::is_satisfied_by_sparse_driver($T, driver) &&)* true
            }

            fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
                let ($(ref $T,)*) = *state;
                $($T::route_candidates($T, out);)*
            }

            fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
                let ($(ref $T,)*) = *state;
                $($T::debug_items($T, out);)*
            }
        }
    };
}

all_tuples!(tuple_query_filter_impl, 1, 12, F);

/// Filter that keeps entities matching at least one filter in `T`.
///
/// `T` is normally a tuple of filters, e.g. `Or<(With<Player>, With<Enemy>)>`
/// keeps entities holding either component.
#[derive(Debug, Clone, Copy, Default)]
pub struct Or<T>(PhantomData<fn() -> T>);

/// Filter that keeps entities matching any filter in `T`.
///
/// This is equivalent to [`Or<T>`], but reads better when the intent is to
/// describe a set of accepted component combinations.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnyOf<T>(PhantomData<fn() -> T>);

/// Filter that keeps entities rejected by `F`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Not<F>(PhantomData<fn() -> F>);

/// Fetch used by `Not<F>`.
///
/// The inner fetch can only be created for archetypes that may satisfy `F`.
/// Archetypes that cannot satisfy `F` automatically satisfy `Not<F>`.
pub enum NotFilterFetch<'w, F: QueryFilter> {
    /// The archetype may contain matches for `F`; evaluate the inner filter
    /// and invert the result.
    Inner(F::Fetch<'w>),
    /// The archetype cannot satisfy `F`; every row matches.
    AlwaysMatch,
}

unsafe impl<F: QueryFilter> QueryFilter for Not<F> {
    type State = F::State;
    type Fetch<'w> = NotFilterFetch<'w, F>;

    fn init_state(
        cm: &ComponentManager,
        access: &mut FilterAccess,
    ) -> Result<Self::State, QueryError> {
        let mut inner_access = FilterAccess::default();
        let state = F::init_state(cm, &mut inner_access)?;

        access.add_negation(inner_access);
        Ok(state)
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        if F::is_archetypal(state) {
            !F::matches_archetype(state, archetype)
        } else {
            true
        }
    }

    fn is_archetypal(state: &Self::State) -> bool {
        F::is_archetypal(state)
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        tick_window: TickWindow,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        let archetype_ref = unsafe { &*archetype };

        if F::matches_archetype(state, archetype_ref) {
            Ok(NotFilterFetch::Inner(unsafe {
                F::init_fetch(state, archetype, storage, sparse_sets, tick_window)?
            }))
        } else {
            Ok(NotFilterFetch::AlwaysMatch)
        }
    }

    unsafe fn matches_entity<'w>(fetch: &mut Self::Fetch<'w>, row: usize, entity: Entity) -> bool {
        match fetch {
            NotFilterFetch::Inner(inner) => {
                !(unsafe { F::matches_entity(inner, row, entity) }
                    && unsafe { F::filter(inner, row, entity) })
            }
            NotFilterFetch::AlwaysMatch => true,
        }
    }

    unsafe fn filter<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> bool {
        true
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        let start = out.len();
        F::debug_items(state, out);

        for item in &mut out[start..] {
            item.mark_negated();
        }
    }
}

/// Implements `QueryFilter` for tuple-backed disjunctive filters.
macro_rules! disjunctive_query_filter_impl {
    ($wrapper:ident, $(#[$meta:meta])* $($T:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($T),*> QueryFilter for $wrapper<($($T,)*)>
        where
            $($T: QueryFilter),*
        {
            type State = ($($T::State,)*);
            type Fetch<'w> = ($($T::Fetch<'w>,)*);

            fn init_state(cm: &ComponentManager, access: &mut FilterAccess) -> Result<Self::State, QueryError> {
                let mut branch_accesses = Vec::new();
                let state = (
                    $({
                        let mut branch_access = FilterAccess::default();
                        let branch_state = $T::init_state(cm, &mut branch_access)?;
                        branch_accesses.push(branch_access);
                        branch_state
                    },)*
                );

                access.add_disjunction(branch_accesses);
                Ok(state)
            }

            fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
                let ($(ref $T,)*) = *state;
                false $(|| $T::matches_archetype($T, archetype))*
            }

            fn is_archetypal(state: &Self::State) -> bool {
                let ($(ref $T,)*) = *state;
                $($T::is_archetypal($T) &&)* true
            }

            fn tick_sensitive_component(state: &Self::State) -> Option<ComponentId> {
                let ($(ref $T,)*) = *state;
                let mut found = None;
                $(if found.is_none() {
                    found = $T::tick_sensitive_component($T);
                })*
                found
            }

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                storage: ArchetypeStorageKind,
                sparse_sets: *const SparseSetManager,
                tick_window: TickWindow,
            ) -> Result<Self::Fetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;
                Ok((
                    $(unsafe { $T::init_fetch($T, archetype, storage, sparse_sets, tick_window)? },)*
                ))
            }

            unsafe fn matches_entity<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
            ) -> bool {
                let ($(ref mut $T,)*) = *fetch;
                false $(|| (unsafe { $T::matches_entity($T, row, entity) && $T::filter($T, row, entity) }))*
            }

            unsafe fn filter<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> bool {
                true
            }

            fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
                let ($(ref $T,)*) = *state;
                $(
                    {
                        let start = out.len();
                        $T::debug_items($T, out);

                        for item in &mut out[start..] {
                            item.mark_disjunctive();
                        }
                    }
                )*
            }
        }
    };
}

/// Implements the disjunctive filter for both `Or` and `AnyOf` over each
/// tuple shape.
macro_rules! or_query_filter_impl {
    ($(#[$meta:meta])* $($T:ident),*) => {
        disjunctive_query_filter_impl!(Or, $(#[$meta])* $($T),*);
        disjunctive_query_filter_impl!(AnyOf, $(#[$meta])* $($T),*);
    };
}

all_tuples!(or_query_filter_impl, 1, 12, F);

/// Filter that keeps only entities whose component `T` was added in the
/// current tick window.
#[derive(Debug, Clone, Copy, Default)]
pub struct Added<C>(PhantomData<fn() -> C>);

/// Filter that keeps only entities whose component `T` changed in the current
/// tick window.
#[derive(Debug, Clone, Copy, Default)]
pub struct Changed<C>(PhantomData<fn() -> C>);

/// Prepared fetch for `Added<C>` / `Changed<C>` filters.
pub enum ChangeFilterFetch<'w, C: Component> {
    /// Table-stored component: a column tick pointer.
    Table {
        /// Column ticks used to evaluate the filter per row.
        component_ticks: *const ComponentTicks,
        /// Window the filter compares ticks against.
        tick_window: TickWindow,
        /// Binds the pointer to the fetch lifetime.
        _marker: PhantomData<&'w C>,
    },
    /// Sparse-stored component: the set plus entity-keyed lookup.
    SparseEntity {
        /// The set owning the component values and ticks.
        set: *const ComponentSparseSet<C>,
        /// Window the filter compares ticks against.
        tick_window: TickWindow,
        /// Binds the set pointer to the fetch lifetime.
        _marker: PhantomData<&'w C>,
    },
    /// Sparse component guaranteed by the selected driver.
    SparseDriver {
        /// The set driving this query unit.
        set: *const ComponentSparseSet<C>,
        /// Window the filter compares ticks against.
        tick_window: TickWindow,
        /// Binds the set pointer to the fetch lifetime.
        _marker: PhantomData<&'w C>,
    },
}

/// Builds the `Added<C>` / `Changed<C>` fetch from the component's storage
/// kind: a column tick pointer for table storage, the sparse set plus its
/// entity-keyed tick lookup for sparse storage.
///
/// # Safety
///
/// `archetype` and `sparse_sets` must be valid for `'w` and compatible with
/// the `ComponentManager` used to create the filter state.
unsafe fn init_change_filter_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    storage: ArchetypeStorageKind,
    sparse_sets: *const SparseSetManager,
    tick_window: TickWindow,
) -> Result<ChangeFilterFetch<'w, C>, QueryError> {
    match state.storage {
        StorageKind::Table => {
            let arch = unsafe { &*archetype };
            let column_index = arch.column_index(state.id)?;
            let column = arch.data(storage).column(column_index);

            Ok(ChangeFilterFetch::Table {
                component_ticks: column.ticks_ptr(),
                tick_window,
                _marker: PhantomData,
            })
        }
        StorageKind::Sparse => {
            let sets = unsafe { &*sparse_sets };
            let set = sets
                .get_sparse_set::<C>(state.id.index())
                .ok_or(QueryError::StorageUnknown(state.id))?;

            Ok(ChangeFilterFetch::SparseEntity {
                set,
                tick_window,
                _marker: PhantomData,
            })
        }
    }
}

/// Builds the change-filter fetch used for rows produced by a sparse
/// conductor.
///
/// Upgrades the entity-keyed sparse variant to
/// [`ChangeFilterFetch::SparseDriver`] when the filtered component is the
/// driver itself, enabling tick lookups by dense position.
///
/// # Safety
///
/// `archetype` and `sparse_sets` must be valid for `'w` and compatible with
/// the `ComponentManager` used to create the filter state, and `driver` must
/// identify the sparse set that produced the rows evaluated with this fetch.
unsafe fn init_sparse_driver_change_filter_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    storage: ArchetypeStorageKind,
    sparse_sets: *const SparseSetManager,
    tick_window: TickWindow,
    driver: ComponentId,
) -> Result<ChangeFilterFetch<'w, C>, QueryError> {
    let fetch =
        unsafe { init_change_filter_fetch(state, archetype, storage, sparse_sets, tick_window) }?;

    match fetch {
        ChangeFilterFetch::SparseEntity {
            set, tick_window, ..
        } if state.id == driver => Ok(ChangeFilterFetch::SparseDriver {
            set,
            tick_window,
            _marker: PhantomData,
        }),
        other => Ok(other),
    }
}

unsafe impl<C: Component> QueryFilter for Changed<C> {
    type State = ComponentQueryState;
    type Fetch<'w> = ChangeFilterFetch<'w, C>;

    fn init_state(
        cm: &ComponentManager,
        access: &mut FilterAccess,
    ) -> Result<Self::State, QueryError> {
        let state = init_required_filter_state::<C>(cm, access, AccessMask::TICKS_READ)?;
        access.mark_domain_inexact();
        Ok(state)
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        match state.storage {
            StorageKind::Table => archetype.signature().contains(state.id),
            StorageKind::Sparse => true,
        }
    }

    fn is_archetypal(_: &Self::State) -> bool {
        false
    }

    fn tick_sensitive_component(state: &Self::State) -> Option<ComponentId> {
        Some(state.id)
    }

    unsafe fn filter<'w>(fetch: &mut Self::Fetch<'w>, row: usize, entity: Entity) -> bool {
        match fetch {
            ChangeFilterFetch::Table {
                component_ticks,
                tick_window,
                ..
            } => (unsafe { *component_ticks.add(row) }).is_changed(*tick_window),
            ChangeFilterFetch::SparseEntity {
                set, tick_window, ..
            } => {
                let Some(component_ticks) = (unsafe { &**set }).get_ticks(entity) else {
                    return false;
                };

                component_ticks.is_changed(*tick_window)
            }
            ChangeFilterFetch::SparseDriver { .. } => {
                unreachable!("sparse driver change filter requires its dense position")
            }
        }
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        tick_window: TickWindow,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            init_change_filter_fetch::<C>(state, archetype, storage, sparse_sets, tick_window)
        }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        tick_window: TickWindow,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            init_sparse_driver_change_filter_fetch(
                state,
                archetype,
                storage,
                sparse_sets,
                tick_window,
                driver,
            )
        }
    }

    unsafe fn filter_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> bool {
        match fetch {
            ChangeFilterFetch::SparseDriver {
                set, tick_window, ..
            } => unsafe {
                (&**set)
                    .get_ticks_by_dense_pos_unchecked(sparse_dense_pos, entity)
                    .is_changed(*tick_window)
            },
            _ => unsafe { Self::filter(fetch, row, entity) },
        }
    }

    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        out.push(QueryRouteCandidate::change_filter(state.id, state.storage));
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::component(
            QueryDebugSource::Filter,
            "Changed<T>",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::TICKS_READ,
            Some(RouteCandidateRole::ChangeFilter),
        ));
    }
}

unsafe impl<C: Component> QueryFilter for Added<C> {
    type State = ComponentQueryState;
    type Fetch<'w> = ChangeFilterFetch<'w, C>;

    fn init_state(
        cm: &ComponentManager,
        access: &mut FilterAccess,
    ) -> Result<Self::State, QueryError> {
        let state = init_required_filter_state::<C>(cm, access, AccessMask::TICKS_READ)?;
        access.mark_domain_inexact();
        Ok(state)
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        match state.storage {
            StorageKind::Table => archetype.signature().contains(state.id),
            StorageKind::Sparse => true,
        }
    }

    fn is_archetypal(_: &Self::State) -> bool {
        false
    }

    fn tick_sensitive_component(state: &Self::State) -> Option<ComponentId> {
        Some(state.id)
    }

    unsafe fn filter<'w>(fetch: &mut Self::Fetch<'w>, row: usize, entity: Entity) -> bool {
        match fetch {
            ChangeFilterFetch::Table {
                component_ticks,
                tick_window,
                ..
            } => (unsafe { *component_ticks.add(row) }).is_added(*tick_window),
            ChangeFilterFetch::SparseEntity {
                set, tick_window, ..
            } => {
                let Some(component_ticks) = (unsafe { &**set }).get_ticks(entity) else {
                    return false;
                };

                component_ticks.is_added(*tick_window)
            }
            ChangeFilterFetch::SparseDriver { .. } => {
                unreachable!("sparse driver addition filter requires its dense position")
            }
        }
    }

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        tick_window: TickWindow,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            init_change_filter_fetch::<C>(state, archetype, storage, sparse_sets, tick_window)
        }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        sparse_sets: *const SparseSetManager,
        tick_window: TickWindow,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            init_sparse_driver_change_filter_fetch(
                state,
                archetype,
                storage,
                sparse_sets,
                tick_window,
                driver,
            )
        }
    }

    unsafe fn filter_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> bool {
        match fetch {
            ChangeFilterFetch::SparseDriver {
                set, tick_window, ..
            } => unsafe {
                (&**set)
                    .get_ticks_by_dense_pos_unchecked(sparse_dense_pos, entity)
                    .is_added(*tick_window)
            },
            _ => unsafe { Self::filter(fetch, row, entity) },
        }
    }

    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        out.push(QueryRouteCandidate::change_filter(state.id, state.storage));
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::component(
            QueryDebugSource::Filter,
            "Added<T>",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::TICKS_READ,
            Some(RouteCandidateRole::ChangeFilter),
        ));
    }
}
