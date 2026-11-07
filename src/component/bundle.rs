//! Bundle erasure and cleanup helpers.
//!
//! Bundles temporarily expose concrete component values as `ErasedBundleItem`s so
//! the storage layer can route them into archetype columns or sparse sets without
//! knowing the tuple shape at runtime.

use std::any::TypeId;

use millarecs_macros::all_tuples;

use crate::component::{
    archetype::{ArchetypeError, ArchetypeManagerError},
    ArchetypeId, Component, ComponentId, ComponentManager,
};
use core::mem::ManuallyDrop;

/// Errors produced while expanding or applying a component bundle.
#[derive(Debug)]
pub enum BundleError {
    /// The target entity cannot receive components.
    EntityInvalid,
    /// A component type in the bundle has not been registered.
    ComponentNotRegistered(TypeId),
    /// The component id does not resolve to any storage backend.
    StorageUnknown(ComponentId),
    /// The same component appears more than once in a bundle.
    DuplicateComponent(ComponentId),
    /// Sparse storage was expected but no sparse set exists for the id.
    SparseSetMissing(ComponentId),
    /// The sparse set rejected a moved component value.
    SparseInsertFailed(ComponentId),
    /// Archetype creation failed while applying a table component.
    Archetype(ArchetypeManagerError),
    /// An existing archetype rejected a row or column operation.
    ArchetypeOp(ArchetypeError),
    /// A cached transition references an archetype that does not exist.
    InvalidArchetype(ArchetypeId),
    /// A bundle has more items than the compact cached index can represent.
    BundleTooLarge,
    /// A bundle does not match the component layout used to prepare a spawn.
    PreparedLayoutMismatch,
}

impl From<ArchetypeManagerError> for BundleError {
    fn from(e: ArchetypeManagerError) -> Self {
        Self::Archetype(e)
    }
}

impl From<ArchetypeError> for BundleError {
    fn from(e: ArchetypeError) -> Self {
        Self::ArchetypeOp(e)
    }
}

/// Type-erased component value borrowed from a bundle during insertion.
pub struct ErasedBundleItem {
    /// Pointer to the still-owned component value.
    pub(crate) ptr: *mut u8,
    /// Type-specific drop function used if storage does not consume the value.
    pub(crate) drop_fn: unsafe fn(*mut u8),
    /// Set to `true` once storage has moved the value out of `ptr`.
    pub(crate) consumed: bool,
}

/// Compile-time metadata for one bundle type.
///
/// Implemented by the `#[derive(Bundle)]` expansion to report which
/// components a bundle contains, in the order its fields are declared.
pub trait BundleInfo: 'static {
    /// Returns the component identifiers of this bundle, in declaration
    /// order, after validating that every component is registered.
    fn component_ids(component_manager: &ComponentManager)
        -> Result<Vec<ComponentId>, BundleError>;
}

/// Value that can transfer one or more components into ECS storage.
///
/// Manual implementations are unsupported. Use `#[derive(Bundle)]` or one of
/// the tuple implementations provided by this crate.
///
/// # Safety
///
/// Implementors must guarantee that:
///
/// - every [`ErasedBundleItem`] points to a live, initialized, and properly
///   aligned value of the component type identified by its [`ComponentId`];
/// - each item uses the matching type-specific drop function;
/// - every component value is exposed exactly once;
/// - `with_items` must expose exactly the same number of items as
///   `BundleInfo::component_ids`;
/// - item `N` must point to a live value whose concrete component type
///   corresponds exactly to component id `N` returned by
///   `BundleInfo::component_ids`;
/// - this positional relationship is relied upon by cached bundle and
///   archetype plans and is not revalidated during plan execution;
/// - component values remain valid for the entire callback invocation;
/// - a consumed component is never read, recovered, moved, or dropped again;
/// - an unconsumed component remains valid so `BundleItemsGuard` can drop it.
pub unsafe trait Bundle: BundleInfo {
    /// Gives the storage layer temporary access to erased bundle items.
    fn with_items<R, F>(
        self,
        component_manager: &mut ComponentManager,
        f: F,
    ) -> Result<R, BundleError>
    where
        F: FnOnce(&mut ComponentManager, &mut [ErasedBundleItem]) -> Result<R, BundleError>;

    #[doc(hidden)]
    /// Returns one erased field without transferring or dropping its value.
    ///
    /// This entry point supports component-major batch execution. Manual
    /// implementations are unsupported; generated implementations preserve the
    /// same positional contract as [`Bundle::with_items`].
    ///
    /// # Safety
    ///
    /// `item_index` must be in bounds. The caller must ensure that the returned
    /// value is moved or dropped exactly once and that `self` remains alive and
    /// stationary until that happens.
    unsafe fn __batch_item(&mut self, item_index: u16) -> ErasedBundleItem;
}

/// Drops unconsumed bundle items when bundle processing exits early.
struct BundleItemsGuard<'a> {
    items: &'a mut [ErasedBundleItem],
}

impl<'a> BundleItemsGuard<'a> {
    /// Creates a guard that owns cleanup of every unconsumed item in `items`.
    fn new(items: &'a mut [ErasedBundleItem]) -> Self {
        Self { items }
    }

    /// Returns the guarded items while cleanup responsibility stays on the guard.
    fn items_mut(&mut self) -> &mut [ErasedBundleItem] {
        self.items
    }
}

impl Drop for BundleItemsGuard<'_> {
    fn drop(&mut self) {
        unsafe {
            __bundle_drop_unconsumed(self.items);
        }
    }
}

/// One component field retained inside component-major batch storage.
pub(crate) struct BatchBundleItem {
    pub(crate) ptr: *mut u8,
    /// Word of the owning storage's bitset that tracks this field's ownership.
    consumed_word: *mut u64,
    /// Single-bit mask locating this field inside `consumed_word`.
    consumed_mask: u64,
}

impl BatchBundleItem {
    /// Marks the field as moved immediately after ownership leaves the bundle.
    ///
    /// # Safety
    ///
    /// The value at `ptr` must have been moved exactly once and the originating
    /// [`BatchBundleStorage`] must still be alive.
    #[inline]
    pub(crate) unsafe fn mark_consumed(&self) {
        unsafe {
            *self.consumed_word |= self.consumed_mask;
        }
    }
}

/// Owns a homogeneous bundle batch while fields are moved component-major.
///
/// A compact moved-item bitset preserves unwind safety without storing one
/// erased descriptor for every entity/component pair.
pub(crate) struct BatchBundleStorage<B: Bundle> {
    /// Bundle rows whose fields are moved individually; a row must never be
    /// dropped as a whole once storage began consuming its fields.
    bundles: Vec<ManuallyDrop<B>>,
    /// Moved-field bitset with `words_per_bundle` `u64` words per bundle row,
    /// one bit per `(row, item_index)` field.
    consumed: Vec<u64>,
    /// Number of erased fields exposed by each bundle.
    item_count: usize,
    /// Words of `consumed` belonging to each bundle row.
    words_per_bundle: usize,
    /// Whether `Drop` still owns fallback cleanup of unconsumed fields.
    armed: bool,
}

impl<B: Bundle> BatchBundleStorage<B> {
    /// Takes ownership of the bundles and arms the consumed bitset.
    pub(crate) fn new(bundles: Vec<B>, item_count: usize) -> Self {
        let words_per_bundle = item_count.div_ceil(u64::BITS as usize);
        let consumed = vec![0; bundles.len().saturating_mul(words_per_bundle)];
        let bundles = into_manually_drop_vec(bundles);

        Self {
            bundles,
            consumed,
            item_count,
            words_per_bundle,
            armed: true,
        }
    }

    /// Returns the number of bundles retained in the batch.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.bundles.len()
    }

    /// Returns one stable erased field and its ownership marker.
    ///
    /// # Safety
    ///
    /// Each `(row, item_index)` pair may be moved at most once.
    pub(crate) unsafe fn item(&mut self, row: usize, item_index: u16) -> BatchBundleItem {
        let item_index = item_index as usize;
        assert!(row < self.bundles.len());
        assert!(item_index < self.item_count);

        let word = row * self.words_per_bundle + item_index / u64::BITS as usize;
        let mask = 1_u64 << (item_index % u64::BITS as usize);
        debug_assert_eq!(self.consumed[word] & mask, 0);

        let bundle = &mut *self.bundles[row];
        let item = unsafe { B::__batch_item(bundle, item_index as u16) };

        BatchBundleItem {
            ptr: item.ptr,
            consumed_word: &mut self.consumed[word],
            consumed_mask: mask,
        }
    }

    /// Disarms fallback cleanup after every field has moved into ECS storage.
    pub(crate) fn finish(mut self) {
        self.debug_assert_fully_consumed();

        self.armed = false;
    }

    /// Disarms cleanup and returns the bundle allocation as an empty vector.
    ///
    /// Every bundle field has moved into ECS storage, so the original `B`
    /// values no longer exist. Reusing only the allocation avoids rebuilding a
    /// potentially large bundle vector for the next homogeneous run.
    pub(crate) fn finish_reuse(mut self) -> Vec<B> {
        self.debug_assert_fully_consumed();
        self.armed = false;

        let bundles = core::mem::take(&mut self.bundles);
        unsafe { recycle_manually_drop_vec(bundles) }
    }

    /// Asserts that every field of every bundle row has been marked consumed.
    #[inline]
    fn debug_assert_fully_consumed(&self) {
        debug_assert!((0..self.bundles.len()).all(|row| {
            (0..self.item_count).all(|item| {
                let word = row * self.words_per_bundle + item / u64::BITS as usize;
                let mask = 1_u64 << (item % u64::BITS as usize);
                self.consumed[word] & mask != 0
            })
        }));
    }
}

/// Reinterprets a bundle vector without reallocating its backing storage.
#[inline]
fn into_manually_drop_vec<B>(mut bundles: Vec<B>) -> Vec<ManuallyDrop<B>> {
    let ptr = bundles.as_mut_ptr().cast::<ManuallyDrop<B>>();
    let len = bundles.len();
    let capacity = bundles.capacity();
    core::mem::forget(bundles);

    unsafe { Vec::from_raw_parts(ptr, len, capacity) }
}

/// Recovers an empty `Vec<B>` allocation after every bundle field was moved.
///
/// # Safety
///
/// Every field of every logical `B` element must already be consumed, so no
/// element may be observed or dropped after the conversion.
#[inline]
unsafe fn recycle_manually_drop_vec<B>(mut bundles: Vec<ManuallyDrop<B>>) -> Vec<B> {
    let ptr = bundles.as_mut_ptr().cast::<B>();
    let capacity = bundles.capacity();
    core::mem::forget(bundles);

    unsafe { Vec::from_raw_parts(ptr, 0, capacity) }
}

impl<B: Bundle> Drop for BatchBundleStorage<B> {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }

        for row in 0..self.bundles.len() {
            for item_index in 0..self.item_count {
                let word = row * self.words_per_bundle + item_index / u64::BITS as usize;
                let mask = 1_u64 << (item_index % u64::BITS as usize);

                if self.consumed[word] & mask != 0 {
                    continue;
                }

                let bundle = &mut *self.bundles[row];
                let item = unsafe { B::__batch_item(bundle, item_index as u16) };

                unsafe {
                    (item.drop_fn)(item.ptr);
                }
            }
        }
    }
}

#[doc(hidden)]
/// Builds one erased item from a concrete component value.
///
/// # Safety
///
/// `value` must contain a live, initialized `T` and must remain at a stable
/// address until the returned item has been processed by [`__bundle_run`].
/// After that call returns, the caller must never access or recover `value`:
/// it has either been moved into storage or dropped by [`BundleItemsGuard`].
pub unsafe fn __bundle_erased_item_of<C: Component>(
    value: &mut ManuallyDrop<C>,
) -> ErasedBundleItem {
    ErasedBundleItem {
        ptr: (&mut **value as *mut C).cast::<u8>(),
        drop_fn: __bundle_drop_in_place_impl::<C>,
        consumed: false,
    }
}

#[doc(hidden)]
/// Builds one erased item from a component field retained inside a batch.
///
/// # Safety
///
/// The caller must keep `value` alive and stationary until the returned item is
/// moved or dropped, and must prevent the enclosing bundle from dropping the
/// field afterwards.
pub unsafe fn __bundle_erased_item_from_ref<C: Component>(value: &mut C) -> ErasedBundleItem {
    ErasedBundleItem {
        ptr: (value as *mut C).cast::<u8>(),
        drop_fn: __bundle_drop_in_place_impl::<C>,
        consumed: false,
    }
}

#[doc(hidden)]
/// Runs a bundle callback behind a cleanup guard.
pub fn __bundle_run<R, F>(
    component_manager: &mut ComponentManager,
    items: &mut [ErasedBundleItem],
    f: F,
) -> Result<R, BundleError>
where
    F: FnOnce(&mut ComponentManager, &mut [ErasedBundleItem]) -> Result<R, BundleError>,
{
    // The guard owns cleanup responsibility until the callback marks items as consumed.
    let mut guard = BundleItemsGuard::new(items);
    f(component_manager, guard.items_mut())
}

#[doc(hidden)]
/// Drops a component through the concrete type used to build the erased item.
///
/// # Safety
/// `ptr` must point to a valid initialized value of type `T`.
pub unsafe fn __bundle_drop_in_place_impl<T>(ptr: *mut u8) {
    unsafe { core::ptr::drop_in_place(ptr as *mut T) };
}

#[doc(hidden)]
/// Drops all items that were not moved into storage.
///
/// # Safety
/// Every unconsumed item must contain a valid pointer and matching drop function.
pub unsafe fn __bundle_drop_unconsumed(items: &mut [ErasedBundleItem]) {
    for item in items {
        if !item.consumed {
            unsafe { (item.drop_fn)(item.ptr) };
        }
    }
}

/// Implements `Bundle` for tuples generated by `all_tuples!`.
macro_rules! tuple_bundle_impl {
    ($(#[$meta:meta])* $($C:ident),*) => {
        $(#[$meta])*
        impl<$($C: Component),*> BundleInfo for ($($C,)*) {
            fn component_ids(
                component_manager: &ComponentManager,
            ) -> Result<Vec<ComponentId>, BundleError> {
                let _ = component_manager;

                Ok(vec![
                    $(
                        component_manager
                            .id_of::<$C>()
                            .ok_or(
                                BundleError::ComponentNotRegistered(
                                    TypeId::of::<$C>()
                                )
                            )?,
                    )*
                ])
            }
        }

        $(#[$meta])*
        // SAFETY: each tuple field is wrapped in `ManuallyDrop`, converted with
        // the matching component type, and handed to `__bundle_run` exactly once.
        // No field is accessed again after the cleanup guard completes.
        unsafe impl<$($C: Component),*> Bundle for ($($C,)*) {
            fn with_items<R, F>(
                self,
                component_manager: &mut ComponentManager,
                f: F,
            ) -> Result<R, BundleError>
            where
                F: FnOnce(
                    &mut ComponentManager,
                    &mut [ErasedBundleItem],
                ) -> Result<R, BundleError>,
            {
                let ($($C,)*) = self;

                $(
                    let mut $C = ManuallyDrop::new($C);
                )*

                let mut items = [
                    $(
                        // SAFETY: `$T` contains a live value at a stable address.
                        // `__bundle_run` below either moves or drops it, and this
                        // implementation never accesses the field afterwards.
                        unsafe {
                            __bundle_erased_item_of::<$C>(
                                &mut $C,
                            )
                        },
                    )*
                ];

                __bundle_run(component_manager, &mut items, f)
            }

            #[allow(unused_assignments, unused_mut, unused_variables, non_snake_case)]
            unsafe fn __batch_item(
                &mut self,
                item_index: u16,
            ) -> ErasedBundleItem {
                let ($($C,)*) = self;
                let mut current = 0_u16;

                $(
                    if item_index == current {
                        return unsafe {
                            __bundle_erased_item_from_ref::<$C>($C)
                        };
                    }
                    current += 1;
                )*

                panic!("bundle item index out of bounds");
            }
        }
    };
}

all_tuples!(tuple_bundle_impl, 0, 12, B);

#[cfg(test)]
mod tests {
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    use super::*;
    use crate::{component::ComponentManager, planning::PlanningChanges};

    #[derive(Default, Debug, PartialEq, Eq)]
    struct Position {
        x: i32,
        y: i32,
    }
    impl Component for Position {}

    #[derive(Default, Debug, PartialEq, Eq)]
    struct Velocity {
        x: i32,
        y: i32,
    }
    impl Component for Velocity {}

    #[derive(Default, Debug, PartialEq, Eq)]
    struct Health(i32);
    impl Component for Health {}

    #[derive(Clone)]
    struct DropSpy {
        counter: Arc<AtomicUsize>,
    }

    impl Default for DropSpy {
        fn default() -> Self {
            Self {
                counter: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Drop for DropSpy {
        fn drop(&mut self) {
            self.counter.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[derive(Default)]
    struct Droppable {
        _spy: DropSpy,
    }
    impl Component for Droppable {}

    fn registered_manager() -> ComponentManager {
        let mut _planning = PlanningChanges::default();

        let mut cm = ComponentManager::new();
        cm.register_column::<Position>().unwrap();
        cm.register_column::<Velocity>().unwrap();
        cm.register_sparse_set::<Health>(&mut _planning).unwrap();
        cm.register_column::<Droppable>().unwrap();
        cm
    }

    #[test]
    fn empty_tuple_builds_no_erased_items() {
        let mut cm = registered_manager();

        let result = ().with_items(&mut cm, |_cm, items| {
            assert!(items.is_empty());
            Ok::<_, BundleError>(42)
        });

        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn tuple_bundle_preserves_tuple_order() {
        let mut cm = registered_manager();

        let result = (Position { x: 1, y: 2 }, Velocity { x: 3, y: 4 }, Health(50)).with_items(
            &mut cm,
            |_cm, items| {
                assert_eq!(items.len(), 3);

                let position = unsafe { &*(items[0].ptr as *const Position) };

                let velocity = unsafe { &*(items[1].ptr as *const Velocity) };

                let health = unsafe { &*(items[2].ptr as *const Health) };

                assert_eq!(position, &Position { x: 1, y: 2 });

                assert_eq!(velocity, &Velocity { x: 3, y: 4 });

                assert_eq!(health, &Health(50));

                Ok::<_, BundleError>(())
            },
        );

        assert!(result.is_ok());
    }

    #[test]
    fn erased_item_points_to_the_original_component_value() {
        let mut cm = registered_manager();

        let result = (Position { x: 7, y: 8 },).with_items(&mut cm, |_cm, items| {
            let value = unsafe { &*(items[0].ptr as *const Position) };
            assert_eq!(value, &Position { x: 7, y: 8 });
            Ok::<_, BundleError>(())
        });

        assert!(result.is_ok());
    }

    #[test]
    fn bundle_layout_creation_fails_for_unregistered_components() {
        #[derive(Default)]
        struct Unregistered;

        impl Component for Unregistered {}

        let mut cm = ComponentManager::new();

        let result = cm.ensure_bundle_layout::<(Unregistered,)>();

        assert!(matches!(
            result,
            Err(BundleError::ComponentNotRegistered(_))
        ));
    }

    #[test]
    fn bundle_layout_rejects_duplicate_table_and_sparse_components() {
        let mut cm = registered_manager();

        let table = cm.ensure_bundle_layout::<(Position, Position)>();
        let sparse = cm.ensure_bundle_layout::<(Health, Health)>();

        assert!(matches!(table, Err(BundleError::DuplicateComponent(_))));
        assert!(matches!(sparse, Err(BundleError::DuplicateComponent(_))));
    }

    #[test]
    fn unconsumed_bundle_items_are_dropped_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut cm = registered_manager();

        let result = (Droppable {
            _spy: DropSpy {
                counter: counter.clone(),
            },
        },)
            .with_items(&mut cm, |_cm, _items| Ok::<_, BundleError>(()));

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn consumed_bundle_items_are_not_dropped_by_the_guard() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut cm = registered_manager();

        let result = (Droppable {
            _spy: DropSpy {
                counter: counter.clone(),
            },
        },)
            .with_items(&mut cm, |_cm, items| {
                let moved = unsafe { core::ptr::read(items[0].ptr as *mut Droppable) };
                items[0].consumed = true;
                drop(moved);
                Ok::<_, BundleError>(())
            });

        assert!(result.is_ok());
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn callback_errors_still_drop_unconsumed_items() {
        let counter = Arc::new(AtomicUsize::new(0));
        let mut cm = registered_manager();

        let result = (Droppable {
            _spy: DropSpy {
                counter: counter.clone(),
            },
        },)
            .with_items(&mut cm, |_cm, _items| {
                Err::<(), _>(BundleError::EntityInvalid)
            });

        assert!(matches!(result, Err(BundleError::EntityInvalid)));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
