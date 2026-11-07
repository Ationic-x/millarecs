//! Per-system local state parameter.
//!
//! `Local<T>` stores one value per system instance. It is not part of the world
//! and declares no component or resource access, so it does not create schedule
//! conflicts.

use std::ops::{Deref, DerefMut};

use crate::{
    system::{param::SystemParam, SystemAccess},
    world::{execution_world::ExecutionWorld, World, WorldError},
};

/// Mutable per-system value persisted between runs of the same system.
///
/// The value lives in the system's parameter state rather than in the world,
/// so every system instance owns an independent copy.
pub struct Local<'s, T> {
    value: &'s mut T,
}

impl<T> Deref for Local<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

impl<T> DerefMut for Local<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value
    }
}

#[doc(hidden)]
/// Stored state backing `Local<T>`.
pub struct LocalState<T> {
    /// The persisted per-system value.
    value: T,
    /// Empty access: locals declare no world visibility.
    access: SystemAccess,
}

unsafe impl<'a, T> SystemParam for Local<'a, T>
where
    T: Default + Send + 'static,
{
    type State = LocalState<T>;
    type Item<'w, 's> = Local<'s, T>;

    /// Creates the value from `T::default` with empty access.
    fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
        Ok(LocalState {
            value: T::default(),
            access: SystemAccess::default(),
        })
    }

    /// Returns the empty access set.
    fn access(state: &Self::State) -> &SystemAccess {
        &state.access
    }

    /// Lends the stored value; the world is untouched.
    unsafe fn get_param<'w, 's>(
        _: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        Ok(Local {
            value: &mut state.value,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    };

    use crate::{
        system::{param::SystemParam, FunctionSystem, System},
        world::World,
    };

    use super::Local;

    #[derive(Default)]
    struct Counter(u32);

    #[test]
    fn local_state_persists_between_system_runs() {
        let observed = Arc::new(AtomicU32::new(0));
        let observed_for_system = Arc::clone(&observed);

        let mut system = FunctionSystem::new(move |mut counter: Local<Counter>| {
            counter.0 += 1;
            observed_for_system.store(counter.0, Ordering::Relaxed);
        });

        let mut world = World::new();
        system.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();

        system.run(execution_world).unwrap();
        assert_eq!(observed.load(Ordering::Relaxed), 1);

        system.run(execution_world).unwrap();
        assert_eq!(observed.load(Ordering::Relaxed), 2);

        system.run(execution_world).unwrap();
        assert_eq!(observed.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn local_declares_no_world_access() {
        let mut world = World::new();

        let state = <Local<Counter> as SystemParam>::init_state(&mut world).unwrap();
        let access = <Local<Counter> as SystemParam>::access(&state);

        assert!(access.components().is_empty());
        assert!(access.resources().is_empty());
        assert!(access.unique_params().is_empty());
    }

    #[test]
    fn each_system_instance_owns_an_independent_local_state() {
        let first_value = Arc::new(AtomicU32::new(0));
        let second_value = Arc::new(AtomicU32::new(0));

        let first_observer = Arc::clone(&first_value);
        let second_observer = Arc::clone(&second_value);

        let mut first = FunctionSystem::new(move |mut counter: Local<Counter>| {
            counter.0 += 1;
            first_observer.store(counter.0, Ordering::Relaxed);
        });

        let mut second = FunctionSystem::new(move |mut counter: Local<Counter>| {
            counter.0 += 1;
            second_observer.store(counter.0, Ordering::Relaxed);
        });

        let mut world = World::new();
        first.initialize(&mut world).unwrap();
        second.initialize(&mut world).unwrap();
        let execution_world = world.execution_world();

        first.run(execution_world).unwrap();
        second.run(execution_world).unwrap();

        assert_eq!(first_value.load(Ordering::Relaxed), 1);
        assert_eq!(second_value.load(Ordering::Relaxed), 1);
    }
}
