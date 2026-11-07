//! Resource management APIs on [`World`].
use super::World;
use crate::{change_detection::Mut, resource::Resource};

impl World {
    /// Inserts a global resource into the world.
    ///
    /// Resources are indexed by their concrete type, so there can be only one
    /// resource of type `T` at a time. If a previous value existed, it is
    /// returned.
    pub fn insert_resource<R: Resource>(&mut self, value: R) -> Option<R> {
        let tick = self.advance_tick();
        self.resources.insert(value, tick)
    }

    /// Returns shared access to the global resource `T`, if it exists.
    pub fn get_resource<R: Resource>(&self) -> Option<&R> {
        self.resources.get::<R>()
    }

    /// Returns exclusive access to the global resource `T`, if it exists.
    ///
    /// Existence is checked before the world tick advances, so a miss leaves
    /// the change-detection windows untouched.
    pub fn get_resource_mut<R: Resource>(&mut self) -> Option<Mut<'_, R>> {
        if !self.resources.contains::<R>() {
            return None;
        }

        let tick = self.advance_tick();
        self.resources.get_mut::<R>(tick)
    }

    /// Removes and returns the global resource `T`, if it exists.
    pub fn remove_resource<R: Resource>(&mut self) -> Option<R> {
        self.resources.remove::<R>()
    }

    /// Returns whether the global resource `T` is currently stored.
    pub fn contains_resource<R: Resource>(&self) -> bool {
        self.resources.contains::<R>()
    }

    /// Ensures that resource `T` exists and returns mutable access to it.
    ///
    /// This inserts `T::default()` only when the resource is missing. Existing
    /// resources are preserved.
    pub fn init_resource<R>(&mut self) -> Mut<'_, R>
    where
        R: Resource + Default,
    {
        self.get_resource_or_insert_with(R::default)
    }

    /// Returns resource `T`, inserting the value produced by `f` if missing.
    pub fn get_resource_or_insert_with<R, F>(&mut self, f: F) -> Mut<'_, R>
    where
        R: Resource,
        F: FnOnce() -> R,
    {
        let tick = self.advance_tick();
        self.resources.get_or_insert_with(tick, f)
    }

    /// Returns the number of resources stored in the world.
    pub fn resource_len(&self) -> usize {
        self.resources.len()
    }
}
