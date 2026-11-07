//! Atomic integer abstraction used by internal counters.
//!
//! The standard library exposes one atomic type per integer width. [`AtomicInt`]
//! gives the ECS a small common interface for those counters while preserving
//! the concrete atomic type chosen by each data structure.

pub(crate) use std::sync::atomic::{
    AtomicI16, AtomicI32, AtomicI64, AtomicI8, AtomicIsize, AtomicU16, AtomicU32, AtomicU64,
    AtomicU8, AtomicUsize, Ordering,
};

/// Common operations supported by the atomic integer types used by the ECS.
///
/// The mutating methods mirror the behaviour of the standard `fetch_*` family:
/// they return the value that was stored before the update was applied.
///
/// # Panics
///
/// Implementations panic when the requested arithmetic cannot be represented by
/// the underlying integer type. Passing `0` as the arithmetic amount is also
/// rejected in debug builds, because these helpers are intended to model real
/// counter transitions.
pub(crate) trait AtomicInt {
    /// Primitive integer stored by the atomic type.
    type Int;

    /// Creates a new atomic value.
    fn new(v: Self::Int) -> Self;

    /// Loads the current value with the requested memory ordering.
    fn load(&self, o: Ordering) -> Self::Int;

    /// Adds `amount` and returns the previous value.
    fn add(&self, amount: usize, set_order: Ordering, fetch_order: Ordering) -> Self::Int;

    /// Subtracts `amount` without allowing a non-negative counter to pass below zero.
    fn sub_nonneg_by(&self, amount: usize, set_order: Ordering, fetch_order: Ordering)
        -> Self::Int;

    /// Integer `0` represented as the underlying primitive type.
    const ZERO: Self::Int;
}

// Generate the same counter semantics for every atomic integer type supported by
// `std`. Keeping this as a macro avoids hand-written implementations drifting
// apart when the counter contract changes.
macro_rules! impl_atomic_int {
    ($atomic:ty, $int:ty) => {
        impl AtomicInt for $atomic {
            type Int = $int;

            #[inline]
            fn new(v: $int) -> Self {
                <$atomic>::new(v)
            }
            #[inline]
            fn load(&self, o: Ordering) -> $int {
                <$atomic>::load(self, o)
            }

            #[inline]
            fn add(&self, amount: usize, set_order: Ordering, fetch_order: Ordering) -> $int {
                debug_assert!(
                    amount > 0,
                    "AtomicInt::try_add only accepts positive amounts"
                );
                <$atomic>::try_update(self, set_order, fetch_order, move |old| {
                    #[allow(unused_comparisons)] // for unsinged cases
                    let result = if old < 0 {
                        let new = (old as isize).checked_add(amount as isize);
                        debug_assert!(
                            new.is_some(),
                            "AtomicInt::add overflowed: [{} + {} | {}]",
                            old,
                            amount,
                            isize::MAX
                        );

                        <$int>::try_from(new.unwrap()).ok()
                    } else {
                        let new = (old as usize).checked_add(amount);
                        debug_assert!(
                            new.is_some(),
                            "AtomicInt::add overflowed: [{} + {} | {}]",
                            old,
                            amount,
                            usize::MAX
                        );

                        <$int>::try_from(new.unwrap()).ok()
                    };
                    debug_assert!(
                        result.is_some(),
                        "AtomicInt::add overflowed: [{} + {} | {}]",
                        old,
                        amount,
                        <$int>::MAX
                    );
                    result
                })
                .unwrap() // If this method fails would panic
            }

            #[inline]
            fn sub_nonneg_by(
                &self,
                amount: usize,
                set_order: Ordering,
                fetch_order: Ordering,
            ) -> $int {
                debug_assert!(
                    amount > 0,
                    "AtomicInt::try_sub_nonneg_by only accepts positive amounts"
                );
                <$atomic>::try_update(self, set_order, fetch_order, move |old| {
                    if old as usize <= amount {
                        Some(0)
                    } else {
                        Some(old - amount as $int)
                    }
                })
                .unwrap() // This method never fails, because we always return Some(_)
            }

            const ZERO: $int = 0 as $int;
        }
    };
}

impl_atomic_int!(AtomicI8, i8);
impl_atomic_int!(AtomicI16, i16);
impl_atomic_int!(AtomicI32, i32);
impl_atomic_int!(AtomicI64, i64);
impl_atomic_int!(AtomicIsize, isize);

impl_atomic_int!(AtomicU8, u8);
impl_atomic_int!(AtomicU16, u16);
impl_atomic_int!(AtomicU32, u32);
impl_atomic_int!(AtomicU64, u64);
impl_atomic_int!(AtomicUsize, usize);

#[cfg(test)]
mod tests {
    use super::*;

    fn relaxed_orders() -> (Ordering, Ordering) {
        (Ordering::Relaxed, Ordering::Relaxed)
    }

    #[test]
    fn add_supports_negative_signed_counters() {
        let value = AtomicIsize::new(-10);
        let (set, fetch) = relaxed_orders();

        assert_eq!(value.add(4, set, fetch), -10);
        assert_eq!(value.load(Ordering::Relaxed), -6);
    }

    #[test]
    fn sub_nonneg_by_saturates_unsigned_counters_at_zero() {
        let value = AtomicUsize::new(3);
        let (set, fetch) = relaxed_orders();

        assert_eq!(value.sub_nonneg_by(1, set, fetch), 3);
        assert_eq!(value.load(Ordering::Relaxed), 2);
        assert_eq!(value.sub_nonneg_by(9, set, fetch), 2);
        assert_eq!(value.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn sub_nonneg_by_saturates_non_negative_signed_counters_at_zero() {
        let value = AtomicI16::new(2);
        let (set, fetch) = relaxed_orders();

        assert_eq!(value.sub_nonneg_by(5, set, fetch), 2);
        assert_eq!(value.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn constants_match_underlying_integer_type() {
        assert_eq!(<AtomicU8 as AtomicInt>::ZERO, 0u8);
        assert_eq!(<AtomicIsize as AtomicInt>::ZERO, 0isize);
    }

    macro_rules! assert_basic_lifecycle {
        ($name:ident, $atomic:ty, $initial:expr) => {
            #[test]
            fn $name() {
                let value = <$atomic as AtomicInt>::new($initial);
                let (set, fetch) = relaxed_orders();

                assert_eq!(value.load(Ordering::Relaxed), $initial);
                assert_eq!(value.add(1, set, fetch), $initial);
                assert_eq!(value.load(Ordering::Relaxed), $initial + 1);
            }
        };
    }

    assert_basic_lifecycle!(atomic_i8_implements_atomic_int, AtomicI8, 2i8);
    assert_basic_lifecycle!(atomic_i16_implements_atomic_int, AtomicI16, 2i16);
    assert_basic_lifecycle!(atomic_i32_implements_atomic_int, AtomicI32, 2i32);
    assert_basic_lifecycle!(atomic_i64_implements_atomic_int, AtomicI64, 2i64);
    assert_basic_lifecycle!(atomic_isize_implements_atomic_int, AtomicIsize, 2isize);
    assert_basic_lifecycle!(atomic_u8_implements_atomic_int, AtomicU8, 2u8);
    assert_basic_lifecycle!(atomic_u16_implements_atomic_int, AtomicU16, 2u16);
    assert_basic_lifecycle!(atomic_u32_implements_atomic_int, AtomicU32, 2u32);
    assert_basic_lifecycle!(atomic_u64_implements_atomic_int, AtomicU64, 2u64);
    assert_basic_lifecycle!(atomic_usize_implements_atomic_int, AtomicUsize, 2usize);

    #[test]
    #[should_panic]
    fn unsigned_add_overflow_panics() {
        let value = AtomicU8::new(u8::MAX);
        let (set, fetch) = relaxed_orders();
        let _ = value.add(1, set, fetch);
    }

    #[test]
    #[should_panic]
    fn signed_add_overflow_panics() {
        let value = AtomicI8::new(i8::MAX);
        let (set, fetch) = relaxed_orders();
        let _ = value.add(1, set, fetch);
    }
}
