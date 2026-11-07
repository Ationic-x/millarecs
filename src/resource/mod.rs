//! Type-erased storage for global ECS resources.
//!
//! A resource is a single value stored directly in the [`World`](crate::world::World)
//! and indexed by its concrete Rust type. Unlike components, resources are not
//! attached to entities and do not participate in archetype layout. They are
//! intended for global state such as time, configuration, input state, asset
//! registries, or other data that should be requested by systems through
//! `Res<T>` / `ResMut<T>`.
//!
//! Safe access through `ResourceManager::get` and `ResourceManager::get_mut`
//! follows normal Rust borrowing rules and tracks mutable writes. System
//! parameters use the internal unchecked access path only after
//! [`ResourceAccess`](crate::system::access::ResourceAccess) has validated that
//! no aliases can be created for the same resource type.

use std::{
    any::{Any, TypeId},
    cell::UnsafeCell,
    collections::HashMap,
};

use crate::change_detection::{ChangeTick, ComponentTicks, Mut};

/// Marker trait for values that can be stored as global ECS resources.
///
/// Resources must be `'static` because they are stored behind type erasure and
/// indexed by [`TypeId`]. `Send + Sync` keeps the storage compatible with a
/// scheduler that moves or shares systems across worker threads.
pub trait Resource: 'static + Send + Sync {}

impl<T> Resource for T where T: 'static + Send + Sync {}

/// Error produced while accessing or validating resource borrows.
#[derive(Debug)]
pub enum ResourceError {
    /// A system or caller requested a resource that is not currently stored.
    Missing(ResourceId),
    /// A combined access set attempted to read and write, or write twice, the
    /// same resource type.
    AccessConflict(ResourceId),
}

/// Stable identifier for a resource type.
///
/// The [`TypeId`] is used for lookup and ordering. The type name is kept only to
/// make diagnostics and tests readable.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId {
    type_id: TypeId,
    type_name: &'static str,
}

impl std::fmt::Display for ResourceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.type_name)
    }
}

impl ResourceId {
    /// Returns the identifier for resource type `T`.
    pub fn of<T: Resource>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            type_name: std::any::type_name::<T>(),
        }
    }

    /// Returns the raw [`TypeId`] used by the resource map.
    pub fn type_id(self) -> TypeId {
        self.type_id
    }

    /// Returns the fully-qualified Rust type name captured for diagnostics.
    pub fn type_name(self) -> &'static str {
        self.type_name
    }
}

impl std::fmt::Display for ResourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(id) => write!(f, "missing resource `{id}`"),
            Self::AccessConflict(id) => {
                write!(f, "conflicting access to resource `{id}`")
            }
        }
    }
}

impl std::error::Error for ResourceError {}

/// Internal cell used to obtain mutable access to one stored resource without
/// taking `&mut ResourceManager` as a whole.
///
/// This is what allows a system such as `(Res<A>, ResMut<B>)` to be materialized:
/// both parameters can start from shared access to the manager while the mutable
/// borrow is restricted to the concrete `B` resource value.
struct ResourceCell {
    value: UnsafeCell<Box<dyn Any + Send + Sync>>,
    ticks: UnsafeCell<ComponentTicks>,
}

// SAFETY:
// ResourceCell only exposes safe shared access and safe exclusive access through
// methods that require `&mut self`. The only shared-to-mutable escape hatch is
// `get_mut_ptrs`, which is crate-private and must be guarded by
// SystemAccess/ResourceAccess before system parameters are materialized.
unsafe impl Sync for ResourceCell {}

impl ResourceCell {
    /// Boxes `value` and stamps it as inserted at `tick`.
    fn new<R: Resource>(value: R, tick: ChangeTick) -> Self {
        Self {
            value: UnsafeCell::new(Box::new(value)),
            ticks: UnsafeCell::new(ComponentTicks::new(tick)),
        }
    }

    /// Consumes the cell and returns the stored value as `T`.
    ///
    /// # Panics
    ///
    /// Panics when the stored value is not of type `T`.
    fn into_inner<T: Resource>(self) -> T {
        *self
            .value
            .into_inner()
            .downcast::<T>()
            .expect("resource TypeId must match stored type")
    }

    /// Returns the stored value as `R`, or `None` on a type mismatch.
    fn get<R: Resource>(&self) -> Option<&R> {
        // SAFETY: shared access only creates `&T`; mutable access through this
        // cell is guarded by `get_mut` or the unsafe system-param path.
        unsafe { (&*self.value.get()).downcast_ref::<R>() }
    }

    /// Returns the stored value and its ticks mutably, or `None` on a type
    /// mismatch.
    fn get_mut_with_ticks<R: Resource>(&mut self) -> Option<(&mut R, &mut ComponentTicks)> {
        let value = self.value.get_mut().downcast_mut::<R>()?;
        let ticks = self.ticks.get_mut();

        Some((value, ticks))
    }

    /// Returns raw mutable pointers to the stored value and its ticks.
    ///
    /// # Safety
    ///
    /// The caller must guarantee exclusive access to this resource value: no
    /// references derived from the cell may coexist with the returned pointers.
    unsafe fn get_mut_ptrs<R: Resource>(&self) -> Option<(*mut R, *mut ComponentTicks)> {
        let value = unsafe { (&mut *self.value.get()).downcast_mut::<R>()? as *mut R };

        Some((value, self.ticks.get()))
    }
}

/// Type-erased storage for all resources owned by a world.
#[derive(Default)]
pub(crate) struct ResourceManager {
    values: HashMap<ResourceId, ResourceCell>,
}

impl ResourceManager {
    /// Creates an empty resource manager.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Inserts `value` as the unique resource of type `T`.
    ///
    /// Returns the previous value when a resource of the same type was already
    /// present.
    pub(crate) fn insert<R: Resource>(&mut self, value: R, tick: ChangeTick) -> Option<R> {
        self.values
            .insert(ResourceId::of::<R>(), ResourceCell::new(value, tick))
            .map(ResourceCell::into_inner::<R>)
    }

    /// Returns shared access to resource `T`, if it exists.
    pub(crate) fn get<R: Resource>(&self) -> Option<&R> {
        self.values
            .get(&ResourceId::of::<R>())
            .and_then(ResourceCell::get::<R>)
    }

    /// Returns change-tracked mutable access to resource `T`, if it exists.
    pub(crate) fn get_mut<R: Resource>(&mut self, change_tick: ChangeTick) -> Option<Mut<'_, R>> {
        let cell = self.values.get_mut(&ResourceId::of::<R>())?;
        let (value, ticks) = cell.get_mut_with_ticks::<R>()?;

        Some(unsafe { Mut::new(value, ticks, change_tick) })
    }

    /// Returns change-tracked mutable access to resource `T`, inserting the
    /// value produced by `f` if missing.
    pub(crate) fn get_or_insert_with<T, F>(&mut self, change_tick: ChangeTick, f: F) -> Mut<'_, T>
    where
        T: Resource,
        F: FnOnce() -> T,
    {
        let id = ResourceId::of::<T>();

        let cell = self
            .values
            .entry(id)
            .or_insert_with(|| ResourceCell::new(f(), change_tick));
        let (value, ticks) = cell
            .get_mut_with_ticks::<T>()
            .expect("resource TypeId must match stored type");

        unsafe { Mut::new(value, ticks, change_tick) }
    }

    /// Returns a shared resource together with a snapshot of its change ticks.
    ///
    /// The returned tick snapshot is valid for the current system execution.
    /// Scheduler access validation prevents a writer for the same resource from
    /// running concurrently.
    pub(crate) fn get_with_ticks<R: Resource>(&self) -> Option<(&R, ComponentTicks)> {
        let cell = self.values.get(&ResourceId::of::<R>())?;
        let value = cell.get::<R>()?;

        let ticks = unsafe {
            // System access validation prevents a concurrent resource writer.
            *cell.ticks.get()
        };

        Some((value, ticks))
    }

    /// Removes and returns resource `T`, if it exists.
    pub(crate) fn remove<T: Resource>(&mut self) -> Option<T> {
        self.values
            .remove(&ResourceId::of::<T>())
            .map(ResourceCell::into_inner::<T>)
    }

    /// Returns raw mutable pointers to resource `R` and its ticks, if stored.
    ///
    /// # Safety
    ///
    /// The caller must uphold the exclusivity that scheduler access validation
    /// establishes before system parameters are materialized: no other
    /// references to this resource may exist while the pointers are live.
    pub(crate) unsafe fn get_mut_ptrs<R: Resource>(&self) -> Option<(*mut R, *mut ComponentTicks)> {
        let cell = self.values.get(&ResourceId::of::<R>())?;
        unsafe { cell.get_mut_ptrs::<R>() }
    }

    /// Returns whether resource `T` is currently stored.
    pub(crate) fn contains<T: Resource>(&self) -> bool {
        self.values.contains_key(&ResourceId::of::<T>())
    }

    /// Returns the number of resources currently stored.
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns whether no resources are currently stored.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::change_detection::TickWindow;

    #[derive(Debug, PartialEq, Eq)]
    struct Config {
        value: i32,
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Counter(i32);

    fn ticks_of<R: Resource>(manager: &ResourceManager) -> ComponentTicks {
        let cell = manager
            .values
            .get(&ResourceId::of::<R>())
            .expect("resource must exist");

        // SAFETY: tests call this helper only after dropping every mutable
        // handle to the resource.
        unsafe { *cell.ticks.get() }
    }

    #[test]
    fn new_manager_starts_empty() {
        let manager = ResourceManager::new();

        assert!(manager.is_empty());
        assert_eq!(manager.len(), 0);
        assert!(!manager.contains::<Config>());
    }

    #[test]
    fn mutable_access_marks_changed_only_when_written() {
        let inserted_tick = ChangeTick::ZERO.next();
        let read_tick = inserted_tick.next();
        let write_tick = read_tick.next();
        let mut manager = ResourceManager::new();
        manager.insert(Counter(4), inserted_tick);

        {
            let resource = manager.get_mut::<Counter>(read_tick).unwrap();
            assert_eq!(resource.0, 4);
        }

        assert!(!ticks_of::<Counter>(&manager).is_changed(TickWindow {
            last_tick: inserted_tick,
            current_tick: read_tick,
        }));

        {
            let mut resource = manager.get_mut::<Counter>(write_tick).unwrap();
            resource.0 += 3;
        }

        assert!(ticks_of::<Counter>(&manager).is_changed(TickWindow {
            last_tick: read_tick,
            current_tick: write_tick,
        }));
        assert_eq!(manager.get::<Counter>(), Some(&Counter(7)));
    }

    #[test]
    fn get_or_insert_existing_resource_does_not_mark_it_without_a_write() {
        let inserted_tick = ChangeTick::ZERO.next();
        let access_tick = inserted_tick.next();
        let calls = AtomicUsize::new(0);
        let mut manager = ResourceManager::new();
        manager.insert(Counter(11), inserted_tick);

        {
            let resource = manager.get_or_insert_with(access_tick, || {
                calls.fetch_add(1, Ordering::Relaxed);
                Counter(99)
            });
            assert_eq!(resource.0, 11);
        }

        assert_eq!(calls.load(Ordering::Relaxed), 0);
        assert!(!ticks_of::<Counter>(&manager).is_changed(TickWindow {
            last_tick: inserted_tick,
            current_tick: access_tick,
        }));
    }
}
