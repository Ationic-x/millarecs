//! Resource system parameters.
//!
//! [`Res<T>`] provides shared access to a world resource and [`ResMut<T>`]
//! provides exclusive access. Parameter state stores access metadata and the
//! previous execution tick so both wrappers can expose change detection. The
//! actual resource value is still looked up every time the system runs, allowing
//! resources to be inserted after schedule initialization.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    change_detection::{ChangeTick, ComponentTicks, Mut, TickWindow},
    resource::{Resource, ResourceError, ResourceId},
    system::{access::SystemAccess, param::SystemParam},
    world::{execution_world::ExecutionWorld, World, WorldError},
};

/// Cached access metadata and change-detection window for a resource parameter.
///
/// Serves as the state of both [`Res`] and [`ResMut`]; the declared access is
/// the only difference between the two.
pub struct ResState<T: Resource> {
    /// Access declared by the parameter (read or write of `T`).
    access: SystemAccess,
    /// Last tick this system itself observed; only changes after it are
    /// reported as new, and `finish_run` advances it.
    last_tick: ChangeTick,
    /// Window used to evaluate change detection during the current run.
    tick_window: TickWindow,
    _marker: PhantomData<T>,
}

/// Shared system parameter for resource `T`.
///
/// Multiple `Res<T>` parameters may coexist. `Res<T>` conflicts with
/// `ResMut<T>` for the same `T` during parameter initialization.
pub struct Res<'w, T: Resource> {
    value: &'w T,
    /// Snapshot of the resource ticks at materialization time.
    ticks: ComponentTicks,
    /// Window used to evaluate `is_added` / `is_changed`.
    tick_window: TickWindow,
}

impl<T: Resource> Res<'_, T> {
    /// Returns whether the resource was inserted since this system last ran.
    #[inline]
    pub fn is_added(&self) -> bool {
        self.ticks.is_added(self.tick_window)
    }

    /// Returns whether the resource was changed since this system last ran.
    #[inline]
    pub fn is_changed(&self) -> bool {
        self.ticks.is_changed(self.tick_window)
    }
}

impl<'w, T: Resource> Deref for Res<'w, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

unsafe impl<T: Resource> SystemParam for Res<'_, T> {
    type State = ResState<T>;
    type Item<'w, 's> = Res<'w, T>;

    /// Declares a read of `T`; the resource does not need to exist yet.
    fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
        Ok(ResState {
            access: SystemAccess::from_resource_read(ResourceId::of::<T>())?,
            last_tick: ChangeTick::ZERO,
            tick_window: TickWindow::default(),
            _marker: PhantomData,
        })
    }

    /// Returns the declared read access.
    fn access(state: &Self::State) -> &SystemAccess {
        &state.access
    }

    /// Opens the change-detection window and borrows the resource; a resource
    /// that is still absent fails here with `Missing`, not during init.
    unsafe fn get_param<'w, 's>(
        world: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        state.tick_window = TickWindow {
            last_tick: state.last_tick,
            current_tick: world.change_tick(),
        };

        let (value, ticks) = world
            .resources()
            .get_with_ticks::<T>()
            .ok_or(ResourceError::Missing(ResourceId::of::<T>()))?;

        Ok(Res {
            value,
            ticks,
            tick_window: state.tick_window,
        })
    }

    /// Fixes the window by advancing `last_tick` to this run's tick.
    fn finish_run(state: &mut Self::State) {
        state.last_tick = state.tick_window.current_tick;
    }
}

/// Exclusive system parameter for resource `T`.
///
/// `ResMut<T>` conflicts with any other `Res<T>` or `ResMut<T>` in the same
/// parameter tuple. It can coexist with reads or writes of different resource
/// types.
pub struct ResMut<'w, R: Resource> {
    /// Change-tracking wrapper over the exclusive resource value.
    value: Mut<'w, R>,
    /// Window used to evaluate `is_added` / `is_changed`.
    tick_window: TickWindow,
}

impl<R: Resource> ResMut<'_, R> {
    /// Returns whether the resource was inserted since this system last ran.
    #[inline]
    pub fn is_added(&self) -> bool {
        self.value.is_added(self.tick_window)
    }

    /// Returns whether the resource was changed since this system last ran.
    #[inline]
    pub fn is_changed(&self) -> bool {
        self.value.is_changed(self.tick_window)
    }
}

impl<'w, R: Resource> Deref for ResMut<'w, R> {
    type Target = R;

    fn deref(&self) -> &Self::Target {
        self.value.deref()
    }
}

impl<'w, T: Resource> DerefMut for ResMut<'w, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.deref_mut()
    }
}

unsafe impl<R: Resource> SystemParam for ResMut<'_, R> {
    type State = ResState<R>;
    type Item<'w, 's> = ResMut<'w, R>;

    /// Declares a write of `R`; the resource does not need to exist yet.
    fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
        Ok(ResState {
            access: SystemAccess::from_resource_write(ResourceId::of::<R>())?,
            last_tick: ChangeTick::ZERO,
            tick_window: TickWindow::default(),
            _marker: PhantomData,
        })
    }

    /// Returns the declared write access.
    fn access(state: &Self::State) -> &SystemAccess {
        &state.access
    }

    /// Opens the change-detection window and takes the resource exclusively; a
    /// resource that is still absent fails here with `Missing`, not during
    /// init.
    unsafe fn get_param<'w, 's>(
        world: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        state.tick_window = TickWindow {
            last_tick: state.last_tick,
            current_tick: world.change_tick(),
        };

        // SAFETY: this parameter declares the only write of `R` for the run,
        // and parameter validation rejected every other accessor.
        let (value, ticks) = unsafe { world.resources().get_mut_ptrs::<R>() }
            .ok_or(ResourceError::Missing(ResourceId::of::<R>()))?;

        // SAFETY: the pointer and ticks come from the same resource cell, and
        // the tick is the one of this run.
        let value = unsafe { Mut::new(value, ticks, world.change_tick()) };

        Ok(ResMut {
            value,
            tick_window: state.tick_window,
        })
    }

    /// Fixes the window by advancing `last_tick` to this run's tick.
    fn finish_run(state: &mut Self::State) {
        state.last_tick = state.tick_window.current_tick;
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        resource::{ResourceError, ResourceId},
        system::param::SystemParam,
        world::{World, WorldError},
    };

    use super::{Res, ResMut};

    #[derive(Debug, PartialEq, Eq)]
    struct Time {
        delta: i32,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Counter(i32);

    #[test]
    fn res_state_declares_resource_read_without_requiring_the_resource_to_exist() {
        type Param = Res<'static, Time>;

        let mut world = World::new();
        let state = <Param as SystemParam>::init_state(&mut world).unwrap();
        let access = <Param as SystemParam>::access(&state);

        assert_eq!(access.resources().reads(), &[ResourceId::of::<Time>()]);
        assert!(access.resources().writes().is_empty());
        assert!(access.components().is_empty());
    }

    #[test]
    fn res_mut_state_declares_resource_write() {
        type Param = ResMut<'static, Counter>;

        let mut world = World::new();
        let state = <Param as SystemParam>::init_state(&mut world).unwrap();
        let access = <Param as SystemParam>::access(&state);

        assert!(access.resources().reads().is_empty());
        assert_eq!(access.resources().writes(), &[ResourceId::of::<Counter>()]);
        assert!(access.components().is_empty());
    }

    #[test]
    fn res_get_param_reads_existing_resource() {
        type Param = Res<'static, Time>;

        let mut world = World::new();
        world.insert_resource(Time { delta: 4 });
        let mut state = <Param as SystemParam>::init_state(&mut world).unwrap();

        let time =
            unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) }
                .unwrap();

        assert_eq!(time.delta, 4);
    }

    #[test]
    fn res_mut_get_param_writes_existing_resource() {
        type Param = ResMut<'static, Counter>;

        let mut world = World::new();
        world.insert_resource(Counter(1));
        let mut state = <Param as SystemParam>::init_state(&mut world).unwrap();

        {
            let mut counter =
                unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) }
                    .unwrap();
            counter.0 += 6;
        }

        assert_eq!(world.get_resource::<Counter>(), Some(&Counter(7)));
    }

    #[test]
    fn res_reports_additions_and_changes_across_runs() {
        type Param = Res<'static, Counter>;

        let mut world = World::new();
        world.insert_resource(Counter(1));
        let mut state = <Param as SystemParam>::init_state(&mut world).unwrap();

        {
            let counter =
                unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) }
                    .unwrap();

            assert!(counter.is_added());
            assert!(counter.is_changed());
        }
        <Param as SystemParam>::finish_run(&mut state);

        {
            let counter =
                unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) }
                    .unwrap();

            assert!(!counter.is_added());
            assert!(!counter.is_changed());
        }
        <Param as SystemParam>::finish_run(&mut state);

        {
            let mut counter = world.get_resource_mut::<Counter>().unwrap();
            counter.0 += 1;
        }

        let counter =
            unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) }
                .unwrap();

        assert!(!counter.is_added());
        assert!(counter.is_changed());
    }

    #[test]
    fn res_mut_observes_writes_performed_through_its_wrapper() {
        type Param = ResMut<'static, Counter>;

        let mut world = World::new();
        world.insert_resource(Counter(1));
        let mut state = <Param as SystemParam>::init_state(&mut world).unwrap();

        {
            let counter =
                unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) }
                    .unwrap();
            assert!(counter.is_added());
            assert!(counter.is_changed());
        }
        <Param as SystemParam>::finish_run(&mut state);

        world.advance_tick();

        {
            let mut counter =
                unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) }
                    .unwrap();

            assert!(!counter.is_added());
            assert!(!counter.is_changed());

            counter.0 += 1;

            assert!(counter.is_changed());
        }
    }

    #[test]
    fn missing_res_returns_resource_error_at_materialization_time() {
        type Param = Res<'static, Time>;

        let mut world = World::new();
        let mut state = <Param as SystemParam>::init_state(&mut world).unwrap();

        let result =
            unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) };

        assert!(matches!(
            result,
            Err(WorldError::Resource(ResourceError::Missing(id))) if id == ResourceId::of::<Time>()
        ));
    }

    #[test]
    fn missing_res_mut_returns_resource_error_at_materialization_time() {
        type Param = ResMut<'static, Counter>;

        let mut world = World::new();
        let mut state = <Param as SystemParam>::init_state(&mut world).unwrap();

        let result =
            unsafe { <Param as SystemParam>::get_param(world.execution_world(), &mut state) };

        assert!(matches!(
            result,
            Err(WorldError::Resource(ResourceError::Missing(id))) if id == ResourceId::of::<Counter>()
        ));
    }

    #[test]
    fn tuple_param_rejects_read_and_write_of_same_resource() {
        type Params = (Res<'static, Time>, ResMut<'static, Time>);

        let mut world = World::new();
        let result = <Params as SystemParam>::init_state(&mut world);

        assert!(matches!(
            result,
            Err(WorldError::Resource(ResourceError::AccessConflict(id)))
                if id == ResourceId::of::<Time>()
        ));
    }

    #[test]
    fn tuple_param_allows_reading_one_resource_and_writing_another() {
        type Params = (Res<'static, Time>, ResMut<'static, Counter>);

        let mut world = World::new();
        world.insert_resource(Time { delta: 3 });
        world.insert_resource(Counter(2));
        let mut state = <Params as SystemParam>::init_state(&mut world).unwrap();

        {
            let (time, mut counter) =
                unsafe { <Params as SystemParam>::get_param(world.execution_world(), &mut state) }
                    .unwrap();
            counter.0 += time.delta;
        }

        assert_eq!(world.get_resource::<Counter>(), Some(&Counter(5)));
    }
}
