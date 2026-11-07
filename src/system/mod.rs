//! Function-based systems and conversion traits.
//!
//! A system is any callable value that can be initialized from a [`World`] and
//! later executed with a set of [`SystemParam`] values. The concrete
//! [`FunctionSystem`] type stores the callable and the cached state required by
//! its parameters.

pub mod access;
pub mod commands;
pub mod event;
pub mod local;
pub mod param;
pub mod query;
pub mod resource;
pub mod schedule;

use std::marker::PhantomData;

use millarecs_macros::all_tuples;

use crate::{
    system::{access::SystemAccess, param::SystemParamFunction, schedule::Schedule},
    world::{execution_world::ExecutionWorld, World, WorldError, WorldId},
};

use self::param::SystemParam;

/// Executable unit owned by a schedule.
///
/// Implementations are expected to cache any parameter state during
/// initialization and reuse it across later runs. Initialization happens in an
/// exclusive world phase before `run` receives its restricted execution view.
pub trait System: Send {
    /// Returns the debug name used by schedule diagnostics.
    fn name(&self) -> &'static str;

    /// Builds or refreshes the system state needed to run against `world`.
    fn initialize(&mut self, world: &mut World) -> Result<(), WorldError>;

    /// Returns whether the system has already created its parameter state.
    fn is_initialized(&self) -> bool;

    /// Returns the access set declared by the initialized parameters.
    ///
    /// Systems that have not been initialized yet do not expose an access set,
    /// because their parameters have not been inspected against a world.
    fn access(&self) -> Option<&SystemAccess>;

    /// Executes the system once against `world`.
    fn run(&mut self, world: ExecutionWorld<'_>) -> Result<(), WorldError>;

    /// Returns whether this system requires exclusive world access.
    ///
    /// Exclusive systems are serialized against every other system by the
    /// conflict machinery and run with `&mut World`.
    fn is_exclusive(&self) -> bool {
        false
    }

    /// Runs the system with exclusive access to the whole world.
    ///
    /// Only called when [`System::is_exclusive`] returns `true`.
    fn run_exclusive(&mut self, world: &mut World) -> Result<(), WorldError> {
        let _ = world;

        unreachable!("non-exclusive systems are never run exclusively");
    }

    /// Returns whether this system currently has deferred work waiting to be
    /// applied.
    ///
    /// When this returns `false`, the schedule may skip [`System::apply_deferred`].
    fn has_deferred_work(&self) -> bool;

    /// Applies deferred work from the system parameters.
    fn apply_deferred(&mut self, world: &mut World) -> Result<(), WorldError>;

    /// Discards deferred work when system execution or application returns an
    /// error.
    fn discard_deferred(&mut self, _: &mut World) {}
}

/// Per-system metadata shared by the concrete system adapters.
pub(crate) struct SystemMeta {
    name: &'static str,
}

impl SystemMeta {
    /// Creates metadata carrying the system's debug `name`.
    pub(crate) fn new(name: &'static str) -> Self {
        Self { name }
    }

    /// Returns the debug name.
    pub(crate) fn name(&self) -> &'static str {
        self.name
    }
}

/// System adapter for ordinary Rust functions and closures.
///
/// `P` describes the parameter list accepted by the callable. The state for
/// those parameters is created once and kept inside the system so that repeated
/// runs can reuse query caches.
pub struct FunctionSystem<F, Marker>
where
    F: SystemParamFunction<Marker>,
{
    func: F,
    meta: SystemMeta,
    /// Parameter state, created by the first `initialize` call and reused for
    /// every run afterwards.
    state: Option<<F::Param as SystemParam>::State>,
    /// World the system was initialized against; runs against any other world
    /// are rejected.
    world_id: Option<WorldId>,
    // `P` is only used at the type level. The function pointer shape avoids
    // implying ownership of a value of type `P`.
    _marker: PhantomData<fn() -> Marker>,
}

impl<F, Marker> FunctionSystem<F, Marker>
where
    F: SystemParamFunction<Marker>,
{
    /// Creates a function system without initializing its parameter state.
    pub fn new(func: F) -> Self {
        Self {
            func,
            meta: SystemMeta::new(std::any::type_name::<F>()),
            state: None,
            world_id: None,
            _marker: PhantomData,
        }
    }
}

impl<F, Marker> System for FunctionSystem<F, Marker>
where
    F: SystemParamFunction<Marker>,
{
    /// Returns the type name of the wrapped function.
    fn name(&self) -> &'static str {
        self.meta.name()
    }

    /// Binds the system to `world` and creates the parameter state on the
    /// first call; later calls only revalidate the world.
    fn initialize(&mut self, world: &mut World) -> Result<(), WorldError> {
        match self.world_id {
            Some(world_id) => world.validate_world_id(world_id)?,
            None => self.world_id = Some(world.id()),
        }

        if self.state.is_none() {
            self.state = Some(<F::Param as SystemParam>::init_state(world)?);
        }

        Ok(())
    }

    /// Returns whether the parameter state has been created.
    fn is_initialized(&self) -> bool {
        self.state.is_some()
    }

    /// Returns the merged parameter access once the state exists.
    fn access(&self) -> Option<&SystemAccess> {
        self.state.as_ref().map(<F::Param as SystemParam>::access)
    }

    /// Validates the world, materializes the parameters, calls the function
    /// and closes the run.
    fn run(&mut self, world: ExecutionWorld) -> Result<(), WorldError> {
        let expected = self
            .world_id
            .expect("system must be initialized before execution");

        if world.id() != expected {
            return Err(WorldError::WorldMismatch {
                expected,
                actual: world.id(),
            });
        }

        let Some(state) = self.state.as_mut() else {
            unreachable!("system must be initialized before execution");
        };

        // Parameter implementations materialize only the values allowed by the
        // access set computed during initialization.
        let param = unsafe { <F::Param as SystemParam>::get_param(world, state)? };

        self.func.call(param);
        <F::Param as SystemParam>::finish_run(state);

        Ok(())
    }

    /// Forwards to the parameter state.
    fn has_deferred_work(&self) -> bool {
        let state = self.state.as_ref().expect("system must be initialized");

        <F::Param as SystemParam>::has_deferred_work(state)
    }

    /// Validates the world, then applies the deferrals held by the parameters.
    fn apply_deferred(&mut self, world: &mut World) -> Result<(), WorldError> {
        let expected = self
            .world_id
            .expect("system must be initialized before applying deferred work");

        world.validate_world_id(expected)?;

        let state = self
            .state
            .as_mut()
            .expect("system must be initialized before applying deferred work");

        <F::Param as SystemParam>::apply(world, state)
    }

    /// Discards parameter deferrals when the state exists.
    fn discard_deferred(&mut self, world: &mut World) {
        if let Some(state) = self.state.as_mut() {
            <F::Param as SystemParam>::discard(world, state);
        }
    }
}

/// A system function with exclusive access to the whole world.
///
/// Exclusive systems conflict with every other system in the schedule, so
/// they always run alone. Use them for operations that cannot be expressed
/// through validated parameters, such as structural maintenance, save/load
/// hooks, or debug tooling.
pub struct ExclusiveSystemFunction<F>
where
    F: Fn(&mut World) -> Result<(), WorldError>,
{
    function: F,
    meta: SystemMeta,
    /// Always the exclusive access set, so this system conflicts with every
    /// other one.
    access: SystemAccess,
    /// World the system was initialized against.
    world_id: Option<WorldId>,
}

impl<F> ExclusiveSystemFunction<F>
where
    F: Fn(&mut World) -> Result<(), WorldError> + Send + Sync + 'static,
{
    /// Creates the system with the exclusive access set and no world binding.
    fn new(function: F) -> Self {
        Self {
            function,
            meta: SystemMeta::new(std::any::type_name::<F>()),
            access: SystemAccess::exclusive(),
            world_id: None,
        }
    }
}

impl<F> System for ExclusiveSystemFunction<F>
where
    F: Fn(&mut World) -> Result<(), WorldError> + Send + Sync + 'static,
{
    /// Returns the type name of the wrapped function.
    fn name(&self) -> &'static str {
        self.meta.name()
    }

    /// Binds the system to `world`; no parameter state is needed.
    fn initialize(&mut self, world: &mut World) -> Result<(), WorldError> {
        match self.world_id {
            Some(world_id) => world.validate_world_id(world_id)?,
            None => self.world_id = Some(world.id()),
        }

        Ok(())
    }

    /// Returns whether the system is bound to a world.
    fn is_initialized(&self) -> bool {
        self.world_id.is_some()
    }

    /// Always returns the exclusive access set.
    fn access(&self) -> Option<&SystemAccess> {
        Some(&self.access)
    }

    /// Always exclusive.
    fn is_exclusive(&self) -> bool {
        true
    }

    /// Validates the world and calls the wrapped function.
    fn run_exclusive(&mut self, world: &mut World) -> Result<(), WorldError> {
        let expected = self
            .world_id
            .expect("system must be initialized before execution");

        world.validate_world_id(expected)?;

        (self.function)(world)
    }

    // Never called: the schedule routes exclusive systems to `run_exclusive`
    // instead of the restricted execution view.
    fn run(&mut self, _: ExecutionWorld<'_>) -> Result<(), WorldError> {
        unreachable!("exclusive systems are always run through run_exclusive");
    }

    /// Does nothing; the function already ran with full world access.
    fn apply_deferred(&mut self, _: &mut World) -> Result<(), WorldError> {
        Ok(())
    }

    /// Always false; the function applies its effects directly.
    fn has_deferred_work(&self) -> bool {
        false
    }
}

/// Wraps a function into an exclusive system.
///
/// ```
/// use millarecs::system::exclusive_system;
/// use millarecs::world::World;
///
/// exclusive_system(|world: &mut World| {
///     // full world access, serialized against every other system
///     Ok(())
/// });
/// ```
pub fn exclusive_system<F>(function: F) -> ExclusiveSystemFunction<F>
where
    F: Fn(&mut World) -> Result<(), WorldError> + Send + Sync + 'static,
{
    ExclusiveSystemFunction::new(function)
}

/// Marker for the exclusive system conversion.
pub struct ExclusiveSystemMarker;

impl<F> IntoSystem<ExclusiveSystemMarker> for ExclusiveSystemFunction<F>
where
    F: Fn(&mut World) -> Result<(), WorldError> + Send + Sync + 'static,
{
    type System = Self;

    /// Returns the system unchanged.
    fn into_system(self) -> Self {
        self
    }
}

/// Converts a callable into the concrete [`System`] type used by the schedule.
pub trait IntoSystem<Marker> {
    /// Concrete system produced by the conversion.
    type System: System;

    /// Wraps `self` into a system value.
    fn into_system(self) -> Self::System;
}

impl<F, Marker> IntoSystem<Marker> for F
where
    F: SystemParamFunction<Marker>,
{
    type System = FunctionSystem<F, Marker>;

    /// Wraps the function in a [`FunctionSystem`].
    fn into_system(self) -> Self::System {
        FunctionSystem::new(self)
    }
}

/// Converts a single system or a tuple of systems into schedulable batch
/// configuration.
pub trait IntoSystems<Marker> {
    /// Adds every wrapped system to `scheduler` in order.
    fn add_to_schedule(self, scheduler: &mut Schedule);
}

macro_rules! impl_into_systems_tuple {
    ($(#[$meta:meta])* $($S:ident, $M:ident),*) => {
        $(#[$meta])*
        impl<$($S, $M),*> IntoSystems<($($M,)*)> for ($($S,)*)
        where
            $(
                $S: IntoSystem<$M>,
                <$S as IntoSystem<$M>>::System: 'static,
            )*
        {
            /// Converts each member through `IntoSystem` and adds it in tuple
            /// order.
            fn add_to_schedule(self, schedule: &mut Schedule) {
                #[allow(non_snake_case)]
                let ($($S,)*) = self;

                $(
                    schedule.add_system($S);
                )*
            }
        }
    };
}

all_tuples!(impl_into_systems_tuple, 1, 12, S, M);

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use millarecs_macros::Component;

    use crate::{
        entity::Entity,
        query::filter::With,
        system::{
            exclusive_system,
            query::Query as SystemQuery,
            resource::Res,
            schedule::{Schedule, ScheduleExecutor},
            FunctionSystem, IntoSystem, System,
        },
        world::{execution_world::ExecutionWorld, World, WorldError},
    };

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Counter(i32);

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Step(i32);

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Marker;

    #[derive(Default)]
    struct DiscardCount(usize);

    struct DiscardParam;

    unsafe impl crate::system::param::SystemParam for DiscardParam {
        type State = crate::system::SystemAccess;
        type Item<'w, 's> = DiscardParam;

        fn init_state(_: &mut World) -> Result<Self::State, crate::world::WorldError> {
            Ok(crate::system::SystemAccess::default())
        }

        fn access(state: &Self::State) -> &crate::system::SystemAccess {
            state
        }

        unsafe fn get_param<'w, 's>(
            _: ExecutionWorld<'w>,
            _: &'s mut Self::State,
        ) -> Result<Self::Item<'w, 's>, crate::world::WorldError> {
            Ok(DiscardParam)
        }

        fn discard(world: &mut World, _: &mut Self::State) {
            world.get_resource_or_insert_with(DiscardCount::default).0 += 1;
        }
    }

    fn register_components(world: &mut World) {
        world.register_table::<Counter>().unwrap();
        world.register_table::<Step>().unwrap();
        world.register_table::<Marker>().unwrap();
    }

    #[test]
    fn new_function_system_starts_uninitialized() {
        let system = FunctionSystem::new(|| {});

        assert!(!system.is_initialized());
        assert!(system.access().is_none());
    }

    #[test]
    fn initialize_creates_state_without_running_function() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_system = Arc::clone(&calls);
        let mut system = FunctionSystem::new(move || {
            calls_for_system.fetch_add(1, Ordering::Relaxed);
        });
        let mut world = World::new();

        system.initialize(&mut world).unwrap();

        assert!(system.is_initialized());
        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(system.access().unwrap().components().is_empty());
        assert!(system.access().unwrap().resources().is_empty());
    }

    #[test]
    fn initialized_system_executes_zero_param_function() {
        let calls = Arc::new(AtomicUsize::new(0));
        let calls_for_system = Arc::clone(&calls);
        let mut system = FunctionSystem::new(move || {
            calls_for_system.fetch_add(1, Ordering::Relaxed);
        });
        let mut world = World::new();
        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();

        system.run(execution_world).unwrap();
        system.run(execution_world).unwrap();

        assert!(system.is_initialized());
        assert_eq!(calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn function_system_mutates_world_through_query_param() {
        fn increment(mut query: SystemQuery<&mut Counter>) {
            for mut counter in query.iter_mut() {
                counter.0 += 1;
            }
        }

        let mut world = World::new();
        register_components(&mut world);
        let entity = world.spawn((Counter(4),)).unwrap();
        let mut system = FunctionSystem::new(increment);
        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();

        system.run(execution_world).unwrap();

        assert_eq!(world.get::<Counter>(entity), Some(&Counter(5)));
    }

    #[test]
    fn exclusive_system_wrapper_reports_and_runs_exclusively() {
        fn touch_world(world: &mut World) -> Result<(), WorldError> {
            world.insert_resource(RanExclusive(true));
            Ok(())
        }

        let mut world = World::new();
        let mut system = exclusive_system(touch_world);
        system.initialize(&mut world).unwrap();
        assert!(system.is_exclusive());
        assert!(system.access().is_some());

        system.run_exclusive(&mut world).unwrap();
        assert!(world.get_resource::<RanExclusive>().unwrap().0);
    }

    #[derive(Default)]
    struct RanExclusive(bool);

    #[test]
    fn function_system_accepts_multiple_query_parameters() {
        fn add_step_for_each_marked_counter(
            mut counters: SystemQuery<(&mut Counter, &Step)>,
            mut marked: SystemQuery<Entity, With<Marker>>,
        ) {
            let mut marked_count = 0;
            for _ in marked.iter() {
                marked_count += 1;
            }

            for (mut counter, step) in counters.iter_mut() {
                counter.0 += step.0 * marked_count;
            }
        }

        let mut world = World::new();
        register_components(&mut world);
        let moved = world.spawn((Counter(1), Step(3))).unwrap();
        world.spawn((Counter(10), Marker)).unwrap();
        world.spawn((Counter(20), Marker)).unwrap();
        let mut system = FunctionSystem::new(add_step_for_each_marked_counter);
        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();

        system.run(execution_world).unwrap();

        assert_eq!(world.get::<Counter>(moved), Some(&Counter(7)));
    }

    #[test]
    fn access_is_available_after_query_system_initialization() {
        fn read_counters(_query: SystemQuery<&Counter>) {}

        let mut world = World::new();
        register_components(&mut world);
        let mut system = FunctionSystem::new(read_counters);

        system.initialize(&mut world).unwrap();

        let access = system.access().unwrap();
        assert_eq!(access.components().len(), 1);
        assert_eq!(
            access.components().accesses()[0].mask(),
            crate::query::AccessMask::VALUE_READ
        );
        assert!(access.resources().is_empty());
    }

    #[test]
    fn access_is_available_after_resource_system_initialization() {
        struct Time;

        fn read_time(_time: Res<Time>) {}

        let mut world = World::new();
        let mut system = FunctionSystem::new(read_time);

        system.initialize(&mut world).unwrap();

        let access = system.access().unwrap();
        assert!(access.components().is_empty());
        assert_eq!(access.resources().reads().len(), 1);
        assert_eq!(access.resources().writes().len(), 0);
    }

    #[test]
    fn into_system_wraps_callable_as_function_system() {
        fn increment(mut query: SystemQuery<&mut Counter>) {
            for mut counter in query.iter_mut() {
                counter.0 += 10;
            }
        }

        let mut world = World::new();
        register_components(&mut world);
        let entity = world.spawn((Counter(2),)).unwrap();
        let mut system = increment.into_system();
        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();

        system.run(execution_world).unwrap();

        assert_eq!(world.get::<Counter>(entity), Some(&Counter(12)));
    }

    #[test]
    fn discard_deferred_forwards_to_initialized_parameters() {
        fn system(_: DiscardParam) {}

        let mut world = World::new();
        let mut system = FunctionSystem::new(system);
        system.initialize(&mut world).unwrap();

        system.discard_deferred(&mut world);

        assert_eq!(world.get_resource::<DiscardCount>().unwrap().0, 1);
    }

    #[test]
    fn systems_keep_query_state_across_schedule_runs() {
        fn increment(mut query: SystemQuery<&mut Counter>) {
            for mut counter in query.iter_mut() {
                counter.0 += 1;
            }
        }

        let mut world = World::new();
        let mut executor = ScheduleExecutor::default();
        register_components(&mut world);
        let entity = world.spawn((Counter(0),)).unwrap();
        let mut schedule = Schedule::new();
        schedule.add_system(increment);

        schedule.run(&mut world, &mut executor).unwrap();
        schedule.run(&mut world, &mut executor).unwrap();
        schedule.run(&mut world, &mut executor).unwrap();

        assert_eq!(world.get::<Counter>(entity), Some(&Counter(3)));
    }
}
