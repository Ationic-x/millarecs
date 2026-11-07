//! System parameter traits and tuple adapters.
//!
//! System parameters are the bridge between a system function signature and the
//! runtime data stored in the world. Each parameter owns a cached `State`,
//! exposes its declared component/resource access, and can materialize the value
//! passed to the function for a single run.

use crate::{
    query::{fetch::QueryData, filter::QueryFilter, state::QueryState},
    system::SystemAccess,
    world::{execution_world::ExecutionWorld, World, WorldError},
};

use super::query::Query;
use millarecs_macros::all_tuples;

/// Cached query state owned by the [`Query`] system parameter across runs.
///
/// The inner [`QueryState`] is built once during initialization and reused for
/// every run of the system; the access snapshot kept next to it makes
/// `SystemParam::access` cheap after initialization.
pub struct QueryParamState<D, F>
where
    D: QueryData,
    F: QueryFilter,
{
    /// Query state reused across runs of the owning system.
    pub(crate) query: QueryState<D, F>,
    /// Component access declared by the query.
    pub(crate) access: SystemAccess,
}

/// Parameter type that can be requested by a system function.
///
/// # Safety
///
/// Implementors must ensure that values returned from [`SystemParam::get_param`]
/// obey the access described by [`SystemParam::access`]. In particular, a
/// parameter that declares mutable access to a component or resource must not
/// create any additional aliasing references to that same value.
pub unsafe trait SystemParam {
    /// Cached data required to build this parameter efficiently.
    type State: Send + 'static;

    /// Value passed to the system function for a single run.
    type Item<'w, 's>;

    /// Creates the cached state for this parameter from the world metadata.
    fn init_state(world: &mut World) -> Result<Self::State, WorldError>;

    /// Returns the component/resource access requested by this parameter state.
    fn access(state: &Self::State) -> &SystemAccess;

    /// Builds the parameter value for a system run.
    ///
    /// The item borrows the world for `'w` and the cached state for `'s`, so it
    /// must not outlive the run that materialized it. Materialization may fail
    /// at run time (for example a missing resource); no value is produced then.
    ///
    /// # Safety
    ///
    /// The caller must guarantee that all parameters for the same system were
    /// validated together and that their combined access set has no conflicts.
    unsafe fn get_param<'w, 's>(
        world: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError>;

    /// Returns whether this parameter state contains deferred work.
    ///
    /// When this returns `false`, the schedule may skip [`SystemParam::apply`].
    fn has_deferred_work(_: &Self::State) -> bool {
        false
    }

    /// Applies deferred work accumulated by the parameter.
    fn apply(_: &mut World, _: &mut Self::State) -> Result<(), WorldError> {
        Ok(())
    }

    /// Discards deferred work when the system cannot complete.
    fn discard(_: &mut World, _: &mut Self::State) {}

    /// Marks the current run as finished for run-scoped parameters.
    fn finish_run(_: &mut Self::State) {}
}

unsafe impl<D, F> SystemParam for Query<'_, '_, D, F>
where
    D: QueryData + 'static,
    F: QueryFilter + 'static,
{
    type State = QueryParamState<D, F>;
    type Item<'w, 's> = Query<'w, 's, D, F>;

    /// Builds the query state and captures its access snapshot.
    fn init_state(world: &mut World) -> Result<Self::State, WorldError> {
        let query = world.query_state::<D, F>()?;

        let access = SystemAccess::from_query(query.access().clone());

        Ok(QueryParamState { query, access })
    }

    /// Returns the access snapshot captured during initialization.
    fn access(state: &Self::State) -> &SystemAccess {
        &state.access
    }

    /// Opens the run window on the query state and materializes the [`Query`].
    unsafe fn get_param<'w, 's>(
        world: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        state.query.begin_run(world.id(), world.change_tick())?;

        // SAFETY: forwarded from `SystemParam::get_param`; the combined access
        // for this system has already rejected conflicting component borrows.
        Ok(unsafe {
            Query::new(
                world.components(),
                world.planning(),
                world.entity_locations(),
                world.task_executor(),
                &mut state.query,
            )
        })
    }

    /// Closes the run window opened by `get_param`.
    fn finish_run(state: &mut Self::State) {
        state.query.finish_run();
    }
}

unsafe impl SystemParam for () {
    type State = SystemAccess;
    type Item<'w, 's> = ();

    /// Returns empty access state.
    fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
        Ok(SystemAccess::default())
    }

    /// Returns the empty access set.
    fn access(state: &Self::State) -> &SystemAccess {
        state
    }

    /// Returns the unit value.
    unsafe fn get_param<'w, 's>(
        _: ExecutionWorld<'w>,
        _: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        Ok(())
    }
}

/// Cached state for a tuple of system parameters.
///
/// The combined access is stored separately from the inner states so callers can
/// inspect conflicts without traversing each parameter again.
pub struct SystemParamTupleState<T> {
    /// Union of every member's access, computed once during initialization.
    access: SystemAccess,
    /// Member states in tuple order.
    inner: T,
}

macro_rules! tuple_system_param_impl {
    ($(#[$meta:meta])* $($P:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($P),*> SystemParam for ($($P,)*)
        where
            $($P: SystemParam),*
        {
            type State = SystemParamTupleState<($($P::State,)*)>;

            type Item<'w, 's> = ($($P::Item<'w, 's>,)*);

            /// Initializes every member and merges their access sets into one;
            /// this is where aliases between tuple members fail.
            fn init_state(world: &mut World) -> Result<Self::State, WorldError> {
                let mut access = SystemAccess::default();

                $(
                    let $P = $P::init_state(world)?;
                    access.extend_from($P::access(&$P))?;
                )*

                Ok(SystemParamTupleState {
                    access,
                    inner: ($($P,)*),
                })
            }

            /// Returns the merged access computed during initialization.
            fn access(state: &Self::State) -> &SystemAccess {
                &state.access
            }

            /// Materializes every member parameter in tuple order.
            unsafe fn get_param<'w, 's>(
                world: ExecutionWorld<'w>,
                state: &'s mut Self::State,
            ) -> Result<Self::Item<'w, 's>, WorldError> {
                let ($($P,)*) = &mut state.inner;

                Ok((
                    $(
                        // SAFETY: the tuple state was initialized by merging all
                        // parameter access sets and rejecting conflicts first.
                        unsafe { $P::get_param(world, $P)? },
                    )*
                ))
            }

            /// Returns whether any member has deferred work.
            fn has_deferred_work(
                state: &Self::State,
            ) -> bool {
                let ($($P,)*) = &state.inner;

                false $(|| $P::has_deferred_work($P))*
            }

            /// Applies deferred work of every member in tuple order.
            fn apply(world: &mut World, state: &mut Self::State) -> Result<(), WorldError> {
                let($($P,)*) = &mut state.inner;

                $(
                    $P::apply(world, $P)?;
                )*

                Ok(())
            }

            /// Discards deferred work of every member in tuple order.
            fn discard(world: &mut World, state: &mut Self::State) {
                let ($($P,)*) = &mut state.inner;

                $(
                    $P::discard(world, $P);
                )*
            }

            /// Finishes the run for every member in tuple order.
            fn finish_run(state: &mut Self::State) {
                let ($($P,)*) = &mut state.inner;
                $(
                    $P::finish_run($P);
                )*
            }
        }
    };
}

all_tuples!(tuple_system_param_impl, 1, 12, P);

/// Callable adapter for a concrete [`SystemParam`] list.
///
/// This trait lets [`FunctionSystem`](crate::system::FunctionSystem) call plain
/// Rust functions whose arguments match the parameter tuple selected by the
/// schedule. The impls require the function to accept the parameter items for
/// any choice of `'w` and `'s`; the bound is higher-ranked (`for<'w, 's>`),
/// which is what lets each run hand the function freshly borrowed values.
pub trait SystemParamFunction<Marker>: Send {
    /// Concrete parameter tuple materialized for each run.
    type Param: SystemParam;

    /// Calls the function with the materialized parameter value.
    fn call<'w, 's>(&mut self, param: <Self::Param as SystemParam>::Item<'w, 's>);
}

impl<F> SystemParamFunction<fn()> for F
where
    F: FnMut() + Send,
{
    type Param = ();

    /// Calls the function with no arguments.
    fn call<'w, 's>(&mut self, _: ()) {
        (self)();
    }
}

macro_rules! impl_system_param_function {
    ($(#[$meta:meta])* $($P:ident),*) => {
        $(#[$meta])*
        impl<F, $($P),*> SystemParamFunction<fn($($P),*)> for F
        where
            $($P: SystemParam,)*
            F: FnMut($($P),*) + Send
                + for<'w, 's> FnMut(
                    $(<$P as SystemParam>::Item<'w, 's>),*
                ),
        {
            type Param = ($($P,)*);

            /// Destructures the tuple and calls the function with each
            /// parameter.
            fn call<'w, 's>(
                &mut self,
                param: <Self::Param as SystemParam>::Item<'w, 's>,
            ) {
                let ($($P,)*) = param;
                (self)($($P),*);
            }
        }
    };
}

all_tuples!(impl_system_param_function, 1, 12, P);

#[cfg(test)]
mod tests {
    use millarecs_macros::Component;

    use crate::{
        query::{
            filter::{Added, Changed, Not, Or, With, Without},
            AccessMask, QueryError,
        },
        system::{param::SystemParam, query::Query as SystemQuery},
        world::{execution_world::ExecutionWorld, World, WorldError},
    };

    use super::SystemParamFunction;

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Position(i32);

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Velocity(i32);

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Health(i32);

    #[derive(Clone, Copy)]
    struct NumberParam(i32);

    #[derive(Default)]
    struct DiscardCount(usize);

    struct DiscardParam;

    unsafe impl SystemParam for NumberParam {
        type State = crate::system::SystemAccess;
        type Item<'w, 's> = NumberParam;

        fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
            Ok(crate::system::SystemAccess::default())
        }

        fn access(state: &Self::State) -> &crate::system::SystemAccess {
            state
        }

        unsafe fn get_param<'w, 's>(
            _: ExecutionWorld<'w>,
            _: &'s mut Self::State,
        ) -> Result<Self::Item<'w, 's>, WorldError> {
            Ok(NumberParam(1))
        }
    }

    unsafe impl SystemParam for DiscardParam {
        type State = crate::system::SystemAccess;
        type Item<'w, 's> = DiscardParam;

        fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
            Ok(crate::system::SystemAccess::default())
        }

        fn access(state: &Self::State) -> &crate::system::SystemAccess {
            state
        }

        unsafe fn get_param<'w, 's>(
            _: ExecutionWorld<'w>,
            _: &'s mut Self::State,
        ) -> Result<Self::Item<'w, 's>, WorldError> {
            Ok(DiscardParam)
        }

        fn discard(world: &mut World, _: &mut Self::State) {
            world.get_resource_or_insert_with(DiscardCount::default).0 += 1;
        }
    }

    fn setup_world() -> World {
        let mut world = World::new();
        world.register_table::<Position>().unwrap();
        world.register_table::<Velocity>().unwrap();
        world.register_sparse::<Health>().unwrap();
        world
    }

    #[test]
    fn unit_param_has_empty_access() {
        let mut world = setup_world();

        let state = <() as SystemParam>::init_state(&mut world).unwrap();
        let access = <() as SystemParam>::access(&state);

        let components = access.components();

        assert!(components.is_empty());
    }

    #[test]
    fn query_param_reports_read_access() {
        type Param = SystemQuery<'static, 'static, &'static Position>;
        let mut world = setup_world();

        let state = <Param as SystemParam>::init_state(&mut world).unwrap();
        let access = <Param as SystemParam>::access(&state);

        let components = access.components();
        let position = world.components().id_of::<Position>().unwrap();

        assert_eq!(components.len(), 1);
        assert_eq!(components.get(position), Some(AccessMask::VALUE_READ));
    }

    #[test]
    fn query_param_reports_write_access() {
        type Param = SystemQuery<'static, 'static, &'static mut Position>;
        let mut world = setup_world();

        let state = <Param as SystemParam>::init_state(&mut world).unwrap();
        let access = <Param as SystemParam>::access(&state);

        let components = access.components();
        let position = world.components().id_of::<Position>().unwrap();

        assert_eq!(components.len(), 1);
        assert_eq!(
            components.get(position),
            Some(AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE)
        );
    }

    #[test]
    fn component_filters_report_their_precise_access_categories() {
        type ChangedParam = SystemQuery<'static, 'static, crate::entity::Entity, Changed<Position>>;
        type AddedParam = SystemQuery<'static, 'static, crate::entity::Entity, Added<Position>>;
        type WithParam = SystemQuery<'static, 'static, crate::entity::Entity, With<Position>>;
        let mut world = setup_world();
        let position = world.components().id_of::<Position>().unwrap();

        let changed = <ChangedParam as SystemParam>::init_state(&mut world).unwrap();
        let added = <AddedParam as SystemParam>::init_state(&mut world).unwrap();
        let with = <WithParam as SystemParam>::init_state(&mut world).unwrap();

        assert_eq!(
            <ChangedParam as SystemParam>::access(&changed)
                .components()
                .get(position),
            Some(AccessMask::TICKS_READ)
        );
        assert_eq!(
            <AddedParam as SystemParam>::access(&added)
                .components()
                .get(position),
            Some(AccessMask::TICKS_READ)
        );
        assert_eq!(
            <WithParam as SystemParam>::access(&with)
                .components()
                .get(position),
            Some(AccessMask::STRUCTURE_READ)
        );
    }

    #[test]
    fn query_composition_allows_changed_filter_with_mutable_data() {
        type Param = SystemQuery<'static, 'static, &'static mut Position, Changed<Position>>;
        let mut world = setup_world();
        let position = world.components().id_of::<Position>().unwrap();

        let state = <Param as SystemParam>::init_state(&mut world).unwrap();
        let mask = <Param as SystemParam>::access(&state)
            .components()
            .get(position)
            .unwrap();

        assert_eq!(
            mask,
            AccessMask::VALUE_WRITE | AccessMask::TICKS_READ | AccessMask::TICKS_WRITE
        );
    }

    #[test]
    fn independent_changed_filter_conflicts_with_mutable_query() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position>,
            SystemQuery<'static, 'static, crate::entity::Entity, Changed<Position>>,
        );
        let mut world = setup_world();

        let result = <Params as SystemParam>::init_state(&mut world);

        assert!(matches!(
            result,
            Err(WorldError::Query(QueryError::AccessConflict {
                current,
                requested,
                ..
            })) if current.contains(AccessMask::TICKS_WRITE)
                && requested.contains(AccessMask::TICKS_READ)
        ));
    }

    #[test]
    fn structural_filter_does_not_conflict_with_value_and_tick_writes() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position>,
            SystemQuery<'static, 'static, crate::entity::Entity, With<Position>>,
        );
        let mut world = setup_world();

        assert!(<Params as SystemParam>::init_state(&mut world).is_ok());
    }

    #[test]
    fn system_compatibility_uses_the_precise_access_categories() {
        type Writer = SystemQuery<'static, 'static, &'static mut Position>;
        type ChangedReader =
            SystemQuery<'static, 'static, crate::entity::Entity, Changed<Position>>;
        type StructureReader = SystemQuery<'static, 'static, crate::entity::Entity, With<Position>>;
        let mut world = setup_world();

        let writer = <Writer as SystemParam>::init_state(&mut world).unwrap();
        let changed = <ChangedReader as SystemParam>::init_state(&mut world).unwrap();
        let structure = <StructureReader as SystemParam>::init_state(&mut world).unwrap();

        let writer_access = <Writer as SystemParam>::access(&writer);
        let changed_access = <ChangedReader as SystemParam>::access(&changed);
        let structure_access = <StructureReader as SystemParam>::access(&structure);

        assert!(writer_access.conflicts_with(changed_access));
        assert!(!writer_access.conflicts_with(structure_access));
    }

    #[test]
    fn tuple_param_combines_non_conflicting_access() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position>,
            SystemQuery<'static, 'static, &'static Velocity>,
            SystemQuery<'static, 'static, &'static mut Health>,
        );
        let mut world = setup_world();

        let state = <Params as SystemParam>::init_state(&mut world).unwrap();
        let access = <Params as SystemParam>::access(&state);

        let components = access.components();
        let position = world.components().id_of::<Position>().unwrap();
        let velocity = world.components().id_of::<Velocity>().unwrap();
        let health = world.components().id_of::<Health>().unwrap();

        assert_eq!(components.len(), 3);
        assert_eq!(
            components.get(position),
            Some(AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE)
        );
        assert_eq!(components.get(velocity), Some(AccessMask::VALUE_READ));
        assert_eq!(
            components.get(health),
            Some(AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE)
        );
    }

    #[test]
    fn tuple_param_allows_multiple_reads_to_same_component() {
        type Params = (
            SystemQuery<'static, 'static, &'static Position>,
            SystemQuery<'static, 'static, &'static Position>,
        );
        let mut world = setup_world();

        let result = <Params as SystemParam>::init_state(&mut world);

        assert!(result.is_ok());
    }

    #[test]
    fn tuple_param_rejects_read_write_conflict() {
        type Params = (
            SystemQuery<'static, 'static, &'static Position>,
            SystemQuery<'static, 'static, &'static mut Position>,
        );
        let mut world = setup_world();

        let result = <Params as SystemParam>::init_state(&mut world);

        assert!(matches!(
            result,
            Err(WorldError::Query(QueryError::AccessConflict { .. }))
        ));
    }

    #[test]
    fn tuple_param_rejects_write_write_conflict() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position>,
            SystemQuery<'static, 'static, &'static mut Position>,
        );
        let mut world = setup_world();

        let result = <Params as SystemParam>::init_state(&mut world);

        assert!(matches!(
            result,
            Err(WorldError::Query(QueryError::AccessConflict { .. }))
        ));
    }

    #[test]
    fn tuple_param_allows_writes_with_mutually_exclusive_filters() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position, With<Health>>,
            SystemQuery<'static, 'static, &'static mut Position, Without<Health>>,
        );
        let mut world = setup_world();

        let state = <Params as SystemParam>::init_state(&mut world).unwrap();
        let components = <Params as SystemParam>::access(&state).components();

        assert_eq!(components.queries().len(), 2);
        assert_eq!(
            components.get(world.components().id_of::<Position>().unwrap()),
            Some(AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE)
        );
    }

    #[test]
    fn tuple_param_rejects_writes_with_overlapping_filters() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position, With<Health>>,
            SystemQuery<'static, 'static, &'static mut Position, With<Health>>,
        );
        let mut world = setup_world();

        assert!(matches!(
            <Params as SystemParam>::init_state(&mut world),
            Err(WorldError::Query(QueryError::AccessConflict { .. }))
        ));
    }

    #[test]
    fn filters_on_different_components_do_not_prove_disjointness() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position, With<Health>>,
            SystemQuery<'static, 'static, &'static mut Position, Without<Velocity>>,
        );
        let mut world = setup_world();

        assert!(matches!(
            <Params as SystemParam>::init_state(&mut world),
            Err(WorldError::Query(QueryError::AccessConflict { .. }))
        ));
    }

    #[test]
    fn or_filter_domains_can_prove_mutual_exclusion() {
        type Params = (
            SystemQuery<
                'static,
                'static,
                &'static mut Position,
                Or<(With<Health>, With<Velocity>)>,
            >,
            SystemQuery<
                'static,
                'static,
                &'static mut Position,
                (Without<Health>, Without<Velocity>),
            >,
        );
        let mut world = setup_world();

        assert!(<Params as SystemParam>::init_state(&mut world).is_ok());
    }

    #[test]
    fn tautological_or_filter_does_not_hide_write_conflicts() {
        type Params = (
            SystemQuery<
                'static,
                'static,
                &'static mut Position,
                Or<(With<Health>, Without<Health>)>,
            >,
            SystemQuery<'static, 'static, &'static mut Position>,
        );
        let mut world = setup_world();

        assert!(matches!(
            <Params as SystemParam>::init_state(&mut world),
            Err(WorldError::Query(QueryError::AccessConflict { .. }))
        ));
    }

    #[test]
    fn not_filter_domains_can_prove_mutual_exclusion() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position, Not<With<Health>>>,
            SystemQuery<'static, 'static, &'static mut Position, With<Health>>,
        );
        let mut world = setup_world();

        assert!(<Params as SystemParam>::init_state(&mut world).is_ok());
    }

    #[test]
    fn not_inexact_filter_domain_does_not_hide_write_conflicts() {
        type Params = (
            SystemQuery<'static, 'static, &'static mut Position, Not<Changed<Health>>>,
            SystemQuery<'static, 'static, &'static mut Position, With<Health>>,
        );
        let mut world = setup_world();

        assert!(matches!(
            <Params as SystemParam>::init_state(&mut world),
            Err(WorldError::Query(QueryError::AccessConflict { .. }))
        ));
    }

    #[test]
    fn impossible_query_has_no_effective_component_access() {
        type Param = SystemQuery<'static, 'static, &'static mut Position, Without<Position>>;
        let mut world = setup_world();

        let state = <Param as SystemParam>::init_state(&mut world).unwrap();
        let components = <Param as SystemParam>::access(&state).components();

        assert!(components.is_empty());
        assert_eq!(components.queries().len(), 1);
        assert!(components.queries()[0].is_impossible());
    }

    #[test]
    fn tuple_param_discards_each_inner_parameter() {
        type Params = (DiscardParam, DiscardParam, DiscardParam);
        let mut world = setup_world();
        let mut state = <Params as SystemParam>::init_state(&mut world).unwrap();

        <Params as SystemParam>::discard(&mut world, &mut state);

        assert_eq!(world.get_resource::<DiscardCount>().unwrap().0, 3);
    }

    #[test]
    fn zero_param_function_adapter_calls_closure() {
        let mut calls = 0;
        let mut function = || {
            calls += 1;
        };

        SystemParamFunction::call(&mut function, ());
        SystemParamFunction::call(&mut function, ());

        assert_eq!(calls, 2);
    }

    #[test]
    fn tuple_function_adapter_expands_arguments() {
        let mut observed = 0;
        let mut function = |a: NumberParam, b: NumberParam, c: NumberParam| {
            observed = a.0 + b.0 + c.0;
        };

        SystemParamFunction::call(
            &mut function,
            (NumberParam(2), NumberParam(3), NumberParam(5)),
        );

        assert_eq!(observed, 10);
    }
}
