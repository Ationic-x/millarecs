//! Change tracking primitives used by components and resources.
//!
//! Each stored value carries insertion and mutation ticks. Queries compare
//! those ticks against a caller-provided window to decide whether a value was
//! added or changed recently enough to match a filter.

use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

/// Maximum age a tick difference can represent before comparisons clamp.
///
/// Prevents incorrect `changed` results after tick wrap-around.
const MAX_CHANGE_AGE: u64 = u64::MAX / 4;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
/// Monotonic tick used to mark world changes.
pub struct ChangeTick(u64);

impl ChangeTick {
    /// Initial tick value.
    pub const ZERO: Self = Self(0);

    /// Returns the next tick, wrapping on overflow.
    pub fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }

    /// `true` when this tick is newer than `last` relative to `current`.
    ///
    /// Ages are clamped to avoid incorrect comparisons after wrapping.
    pub fn is_newer_than(self, last: Self, current: Self) -> bool {
        let changed_age = current.0.wrapping_sub(self.0).min(MAX_CHANGE_AGE);
        let last_age = current.0.wrapping_sub(last.0).min(MAX_CHANGE_AGE);

        changed_age < last_age
    }

    /// Raw value used by column write-access latches.
    pub(crate) fn as_u64(self) -> u64 {
        self.0
    }

    /// Rebuilds a tick from a raw latch value.
    pub(crate) fn from_u64(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Clone, Copy, Debug)]
/// Change ticks stored beside one component or resource value.
pub struct ComponentTicks {
    added: ChangeTick,
    changed: ChangeTick,
}

impl ComponentTicks {
    /// Creates ticks for a newly inserted value.
    pub fn new(change_tick: ChangeTick) -> Self {
        Self {
            added: change_tick,
            changed: change_tick,
        }
    }

    /// Marks the value as changed at `change_tick`.
    pub fn mark_changed(&mut self, change_tick: ChangeTick) {
        self.changed = change_tick;
    }

    /// Whether the value was added inside `tick_window`.
    pub fn is_added(self, tick_window: TickWindow) -> bool {
        self.added
            .is_newer_than(tick_window.last_tick, tick_window.current_tick)
    }

    /// Whether the value changed inside `tick_window`.
    pub fn is_changed(self, tick_window: TickWindow) -> bool {
        self.changed
            .is_newer_than(tick_window.last_tick, tick_window.current_tick)
    }

    /// Returns the newest tick stored in this record.
    pub(crate) fn latest_tick(&self) -> ChangeTick {
        if self.changed.is_newer_than(self.added, ChangeTick::ZERO) {
            self.changed
        } else {
            self.added
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
/// Tick range used by change filters.
pub struct TickWindow {
    /// Last tick observed by the caller.
    pub last_tick: ChangeTick,
    /// Current world tick.
    pub current_tick: ChangeTick,
}

/// Mutable reference wrapper that records changes on dereference.
pub struct Mut<'w, T> {
    value: *mut T,
    component_ticks: *mut ComponentTicks,
    change_tick: ChangeTick,
    _marker: PhantomData<&'w mut T>,
}

// SAFETY:
// `Mut<T>` represents unique access to one component/resource value. Query and
// resource access validation must guarantee that no incompatible reference to
// the same value exists while the wrapper is alive. Moving that unique access to
// another thread is sound when the wrapped value itself is `Send`.
unsafe impl<T: Send> Send for Mut<'_, T> {}

impl<'w, T> Mut<'w, T> {
    /// Creates a mutable wrapper from raw value and tick pointers.
    ///
    /// # Safety
    ///
    /// The caller must guarantee unique mutable access to `value` and to its
    /// tick metadata for the returned lifetime.
    pub(crate) unsafe fn new(
        value: *mut T,
        component_ticks: *mut ComponentTicks,
        change_tick: ChangeTick,
    ) -> Self {
        Self {
            value,
            component_ticks,
            change_tick,
            _marker: PhantomData,
        }
    }

    /// Marks the wrapped value as changed at the current world tick.
    pub fn set_changed(&mut self) {
        unsafe {
            (*self.component_ticks).mark_changed(self.change_tick);
        }
    }

    /// Whether the value was added inside `tick_window`.
    #[inline]
    pub fn is_added(&self, tick_window: TickWindow) -> bool {
        unsafe { (*self.component_ticks).is_added(tick_window) }
    }

    /// Whether the value was changed inside `tick_window`.
    #[inline]
    pub fn is_changed(&self, tick_window: TickWindow) -> bool {
        unsafe { (*self.component_ticks).is_changed(tick_window) }
    }
}

impl<T> Deref for Mut<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.value }
    }
}

impl<T> DerefMut for Mut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.set_changed();
        unsafe { &mut *self.value }
    }
}
