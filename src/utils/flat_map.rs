//! Flat open-addressed maps for the ECS hot identity tables.
//!
//! Component ids, bundle layouts and compiled archetype plans are looked up
//! once or more per structural operation. Those lookups previously paid a
//! `std::HashMap` probe: SipHash over the key plus random-state indirection,
//! roughly 12 ns per operation on this crate's benchmark hardware.
//!
//! Every one of those tables is small (tens to low hundreds of entries),
//! append-only for the lifetime of a world, and queried with cheap keys.
//! [`FlatMap`] targets exactly that shape: keys fold to a `u64` through a
//! multiplicative mix, slots live inline in one allocation, and lookups
//! resolve in a single cache-line probe in the common case.

use std::any::TypeId;
use std::hash::{Hash, Hasher};

/// Key contract for [`FlatMap`]: a cheap deterministic fold to `u64`.
///
/// Folding is separated from the map's own mixing so each key type can
/// contribute the cheapest useful entropy: dense ids fold to themselves,
/// while `TypeId` replays its `Hash` implementation into a multiplicative
/// hasher because stable Rust does not expose its internals.
pub(crate) trait FlatKey: Copy + Eq {
    /// Returns this key flattened to the `u64` the map mixes into a home slot.
    fn fold(self) -> u64;
}

/// Mixing constant of the FxHash-family fold.
const FOLD_SEED: u64 = 0x517c_c1b7_2722_0a95;

/// Multiplicative fold hasher (FxHash family) used to flatten `TypeId`s.
#[derive(Default)]
pub(crate) struct FoldHasher {
    hash: u64,
}

impl FoldHasher {
    /// Folds one `u64` word into the running hash.
    #[inline]
    fn mix(&mut self, value: u64) {
        self.hash = (self.hash.rotate_left(5) ^ value).wrapping_mul(FOLD_SEED);
    }
}

impl Hasher for FoldHasher {
    #[inline]
    fn write_u8(&mut self, value: u8) {
        self.mix(value as u64);
    }

    #[inline]
    fn write_u64(&mut self, value: u64) {
        self.mix(value);
    }

    #[inline]
    fn write_u128(&mut self, value: u128) {
        self.mix(value as u64);
        self.mix((value >> 64) as u64);
    }

    #[inline]
    fn write_usize(&mut self, value: usize) {
        self.mix(value as u64);
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            self.mix(u64::from_le_bytes(
                chunk.try_into().expect("chunk of eight bytes"),
            ));
        }
        let remainder = chunks.remainder();
        if !remainder.is_empty() {
            let mut last = [0u8; 8];
            last[..remainder.len()].copy_from_slice(remainder);
            self.mix(u64::from_le_bytes(last));
        }
    }

    fn finish(&self) -> u64 {
        self.hash
    }
}

impl FlatKey for TypeId {
    #[inline]
    fn fold(self) -> u64 {
        let mut hasher = FoldHasher::default();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

/// Append-only open-addressed map for identity tables on the hot paths.
///
/// Keys never repeat and entries are never removed in this crate's domain:
/// registered components, bundle layouts and compiled plans live as long as
/// their world. The map encodes that contract: there is no removal API and
/// duplicate insertion trips a `debug_assert`.
pub(crate) struct FlatMap<K: FlatKey, V> {
    slots: Vec<Option<(K, V)>>,
    /// Bit mask for slot indices; the slot count is always a power of two.
    mask: usize,
    /// Count of low bits discarded from the 64-bit product before masking.
    shift: u32,
    len: usize,
}

/// Fibonacci-hashing multiplier. Sequential dense ids land on consecutive
/// homes after the multiply, so layout and component ids do not cluster.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// Smallest backing capacity after the first growth step.
const MIN_CAPACITY: usize = 8;

impl<K: FlatKey, V> Default for FlatMap<K, V> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            mask: 0,
            shift: 64,
            len: 0,
        }
    }
}

impl<K: FlatKey, V> FlatMap<K, V> {
    /// Returns an empty map without allocating.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the number of entries stored in the map.
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Returns the home slot of a folded key after the Fibonacci multiply.
    #[inline]
    fn home(&self, folded: u64) -> usize {
        (folded.wrapping_mul(GOLDEN_GAMMA) >> self.shift) as usize & self.mask
    }

    /// Returns the value stored under `key`, probing linearly from its home
    /// slot until a match or an empty slot proves absence.
    #[inline]
    pub(crate) fn get(&self, key: K) -> Option<&V> {
        if self.slots.is_empty() {
            return None;
        }

        let mut index = self.home(key.fold());
        loop {
            match &self.slots[index] {
                None => return None,
                Some((occupied, value)) if *occupied == key => return Some(value),
                _ => index = (index + 1) & self.mask,
            }
        }
    }

    /// Returns a copy of the value stored under `key`, if present.
    #[inline]
    pub(crate) fn get_copy(&self, key: K) -> Option<V>
    where
        V: Copy,
    {
        self.get(key).copied()
    }

    /// Returns whether the map holds an entry for `key`.
    #[inline]
    pub(crate) fn contains(&self, key: K) -> bool {
        self.get(key).is_some()
    }

    /// Doubles the slot count and re-inserts every retained entry at its new
    /// home.
    #[cold]
    #[inline(never)]
    fn grow(&mut self) {
        let new_capacity = (self.slots.len().max(MIN_CAPACITY / 2)) * 2;
        let old_slots = std::mem::take(&mut self.slots);

        self.slots = (0..new_capacity).map(|_| None).collect();
        self.mask = new_capacity - 1;
        self.shift = 64 - new_capacity.trailing_zeros();
        self.len = 0;

        for (key, value) in old_slots.into_iter().flatten() {
            self.insert(key, value);
        }
    }

    /// Inserts `value` under `key`, growing the table first when the load
    /// factor would pass 7/8.
    pub(crate) fn insert(&mut self, key: K, value: V) {
        if self.len + 1 > self.slots.len() * 7 / 8 {
            self.grow();
        }

        let mut index = self.home(key.fold());
        loop {
            let slot = &mut self.slots[index];
            match slot {
                None => {
                    *slot = Some((key, value));
                    self.len += 1;
                    return;
                }
                Some((occupied, existing)) if *occupied == key => {
                    debug_assert!(
                        false,
                        "flat map keys are unique; identities register exactly once"
                    );
                    *existing = value;
                    return;
                }
                Some(_) => index = (index + 1) & self.mask,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq)]
    struct RowKey(u32);

    impl FlatKey for RowKey {
        fn fold(self) -> u64 {
            self.0 as u64
        }
    }

    #[test]
    fn lookup_finds_inserted_keys_and_rejects_missing_ones() {
        let mut map = FlatMap::new();
        for key in 0..64u32 {
            map.insert(RowKey(key), key * 3);
        }

        for key in 0..64u32 {
            assert_eq!(map.get_copy(RowKey(key)), Some(key * 3));
        }

        assert_eq!(map.get_copy(RowKey(9999)), None);
        assert_eq!(map.len(), 64);
    }

    #[test]
    fn growth_keeps_every_entry() {
        let mut map = FlatMap::new();
        for key in 0..1000u32 {
            map.insert(RowKey(key), key);
        }

        for key in 0..1000u32 {
            assert_eq!(map.get_copy(RowKey(key)), Some(key));
        }
    }

    #[test]
    fn distinct_type_ids_fold_to_distinct_values() {
        assert_ne!(TypeId::of::<u8>().fold(), TypeId::of::<u16>().fold());
        assert_eq!(TypeId::of::<u8>().fold(), TypeId::of::<u8>().fold());
    }
}
