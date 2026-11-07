//! Vector wrapper with an independent atomic logical length.
//!
//! [`EcsVec`] is useful for internal ECS structures that need an owned `Vec<T>`
//! plus a counter that can be advanced through shared references. The backing
//! vector still follows normal Rust borrowing rules; only the counter is atomic.

use std::ops::{Deref, DerefMut};

use crate::utils::atomic::{AtomicInt, Ordering};

/// `Vec<T>` paired with an atomic counter of the chosen integer width.
///
/// The counter is not automatically tied to `Vec::len()`. Callers decide when it
/// represents reserved slots, committed elements or another logical measure.
pub(crate) struct EcsVec<T, A: AtomicInt> {
    list: Vec<T>,
    count: A,
}

impl<T, A: AtomicInt> Default for EcsVec<T, A> {
    #[inline]
    fn default() -> Self {
        Self {
            list: Vec::new(),
            count: A::new(A::ZERO),
        }
    }
}

impl<T, A: AtomicInt> EcsVec<T, A> {
    /// Returns the current logical counter.
    #[inline]
    pub fn count(&self) -> A::Int {
        self.count.load(Ordering::Relaxed)
    }

    /// Increments the counter by `amount` and returns the previous value.
    #[inline]
    pub fn inc_by(&self, amount: usize) -> A::Int {
        self.count.add(amount, Ordering::AcqRel, Ordering::Acquire)
    }

    /// Decrements the counter by `1`, saturating non-negative counters at zero.
    #[inline]
    pub fn dec_nonneg(&self) -> A::Int {
        self.dec_nonneg_by(1)
    }

    /// Decrements the counter by `amount`, saturating non-negative counters at zero.
    #[inline]
    pub fn dec_nonneg_by(&self, amount: usize) -> A::Int {
        self.count
            .sub_nonneg_by(amount, Ordering::AcqRel, Ordering::Acquire)
    }
}

impl<T, A: AtomicInt> Deref for EcsVec<T, A> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.list
    }
}

impl<T, A: AtomicInt> DerefMut for EcsVec<T, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.list
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::atomic::{AtomicIsize, AtomicU8, AtomicUsize};
    use std::sync::{Arc, Barrier};
    use std::thread;

    type SignedVec<T> = EcsVec<T, AtomicIsize>;
    type UnsignedVec<T> = EcsVec<T, AtomicUsize>;

    #[test]
    fn default_vec_has_empty_storage_and_zero_counter() {
        let vec: UnsignedVec<u32> = EcsVec::default();

        assert!(vec.is_empty());
        assert_eq!(vec.count(), 0);
    }

    #[test]
    fn counter_updates_return_previous_value() {
        let vec: SignedVec<()> = EcsVec::default();

        assert_eq!(vec.inc_by(1), 0);
        assert_eq!(vec.inc_by(4), 1);
        assert_eq!(vec.dec_nonneg(), 5);
        assert_eq!(vec.dec_nonneg_by(2), 4);
        assert_eq!(vec.count(), 2);
    }

    #[test]
    fn non_negative_decrement_saturates_unsigned_counter() {
        let vec: UnsignedVec<()> = EcsVec::default();

        assert_eq!(vec.dec_nonneg(), 0);
        assert_eq!(vec.count(), 0);
        assert_eq!(vec.inc_by(3), 0);
        assert_eq!(vec.dec_nonneg_by(2), 3);
        assert_eq!(vec.count(), 1);
        assert_eq!(vec.dec_nonneg_by(9), 1);
        assert_eq!(vec.count(), 0);
    }

    #[test]
    fn non_negative_decrement_saturates_signed_counter_from_positive_values() {
        let vec: SignedVec<()> = EcsVec::default();

        assert_eq!(vec.inc_by(2), 0);
        assert_eq!(vec.dec_nonneg_by(5), 2);
        assert_eq!(vec.count(), 0);
    }

    #[test]
    fn vec_operations_are_available_through_deref_mut() {
        let mut vec: UnsignedVec<i32> = EcsVec::default();

        vec.push(3);
        vec.push(1);
        vec.push(2);
        vec.sort();

        assert_eq!(&*vec, &[1, 2, 3]);
    }

    #[test]
    fn counter_can_be_modified_through_shared_references() {
        let vec: UnsignedVec<()> = EcsVec::default();
        let shared = &vec;

        assert_eq!(shared.inc_by(2), 0);
        assert_eq!(vec.count(), 2);
    }

    #[test]
    fn concurrent_unsigned_increments_are_observed_by_the_counter() {
        let threads = 4;
        let iterations = 2_000;
        let vec: Arc<UnsignedVec<()>> = Arc::new(EcsVec::default());
        let start = Arc::new(Barrier::new(threads));

        let mut handles = Vec::new();
        for _ in 0..threads {
            let vec = Arc::clone(&vec);
            let start = Arc::clone(&start);
            handles.push(thread::spawn(move || {
                start.wait();
                for _ in 0..iterations {
                    let _ = vec.inc_by(1);
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(vec.count(), threads * iterations);
    }

    #[test]
    #[should_panic]
    fn unsigned_increment_overflow_panics() {
        let vec: EcsVec<(), AtomicU8> = EcsVec::default();
        let _ = vec.inc_by(u8::MAX as usize);
        let _ = vec.inc_by(1);
    }
}
