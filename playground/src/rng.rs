//! Tiny deterministic PRNG (`xorshift64*`) so every demo replays the same
//! scene for the same seed without pulling in a random-number crate.

/// Deterministic pseudo-random generator, usable as a millarecs resource.
pub struct Rng {
    state: u64,
}

impl Rng {
    /// The default seed used by the demos.
    pub const DEFAULT_SEED: u64 = 0x853C_49E6_748F_EA9B;

    /// Creates a generator from `seed`; a zero seed is coerced to a nonzero
    /// state so sequences always advance.
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// Creates a generator with [`Rng::DEFAULT_SEED`].
    pub const fn seeded() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }

    /// Produces the next raw 64-bit value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform `f32` in `[0, 1)`.
    pub fn f32(&mut self) -> f32 {
        // 24 mantissa bits is plenty for visuals and keeps the range exact.
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform `f32` in `[lo, hi)`.
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.f32() * (hi - lo)
    }

    /// Uniform `usize` in `[0, n)`; returns 0 when `n == 0`.
    pub fn usize(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Uniform angle in `[0, 2π)`.
    pub fn angle(&mut self) -> f32 {
        self.f32() * std::f32::consts::TAU
    }

    /// Returns `true` with probability `p` in `[0, 1]`.
    pub fn chance(&mut self, p: f32) -> bool {
        self.f32() < p
    }

    /// Picks a random element of `items`, or `None` when empty.
    pub fn choose<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.usize(items.len())])
        }
    }
}

impl Default for Rng {
    fn default() -> Self {
        Self::seeded()
    }
}

#[cfg(test)]
mod tests {
    use super::Rng;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Rng::new(42);
        let mut b = Rng::new(42);
        for _ in 0..16 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn f32_in_unit_range() {
        let mut rng = Rng::new(7);
        for _ in 0..10_000 {
            let v = rng.f32();
            assert!((0.0..1.0).contains(&v));
        }
    }

    #[test]
    fn range_respects_bounds() {
        let mut rng = Rng::new(9);
        for _ in 0..10_000 {
            let v = rng.range(3.5, 7.25);
            assert!((3.5..7.25).contains(&v));
        }
    }

    #[test]
    fn usize_bounded_and_zero_safe() {
        let mut rng = Rng::new(11);
        assert_eq!(rng.usize(0), 0);
        for _ in 0..1_000 {
            assert!(rng.usize(5) < 5);
        }
    }
}
