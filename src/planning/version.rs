//! Non-zero monotonic version counters.
//!
//! The type parameter only tags which version stream a counter belongs to,
//! preventing accidental comparisons between unrelated streams. Counters
//! never reach zero, so consumers can use zero as their uninitialized state.

use std::{marker::PhantomData, num::NonZero};

/// Monotonic counter tagged by the version stream `T`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Version<T> {
    value: NonZero<u64>,
    _marker: PhantomData<T>,
}

impl<T> Copy for Version<T> {}

impl<T> Clone for Version<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Version<T> {
    /// Creates the counter at its initial value.
    pub(super) fn new() -> Self {
        Self {
            value: NonZero::<u64>::MIN,
            _marker: PhantomData,
        }
    }

    /// Returns the current counter value.
    #[inline]
    pub(super) fn current(&self) -> u64 {
        self.value.get()
    }

    /// Advances the counter, wrapping back to one on overflow.
    pub(super) fn bump(&mut self) {
        let next = self.value.get().wrapping_add(1);

        self.value = NonZero::new(if next == 0 { 1 } else { next })
            .expect("value is guaranteed to be non-zero");
    }
}
