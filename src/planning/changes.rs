//! Small-accumulation container for keyed changes.
//!
//! Most world operations touch at most two archetypes, so the first two
//! changes are stored inline. Larger batches graduate to a heap-backed dense
//! index without any hashing.

use crate::planning::ArchetypeRowsChange;

/// Dense-index sentinel for a key with no accumulated change.
const MISSING_POSITION: usize = usize::MAX;

/// Change type that identifies itself by a dense integer key.
pub(crate) trait KeyedChange: Copy {
    /// Returns the dense key this change accumulates under.
    fn key(self) -> usize;
}

impl KeyedChange for ArchetypeRowsChange {
    #[inline]
    fn key(self) -> usize {
        self.id.index()
    }
}

/// Heap-backed storage for more than two changes.
#[derive(Debug)]
pub(crate) struct ManyChanges<T> {
    /// Changes in insertion order.
    values: Vec<T>,
    /// Dense key-to-value index. Component and archetype identifiers are
    /// assigned densely, so direct indexing is cheaper than hashing here.
    positions: Vec<usize>,
}

/// Inline-first accumulation of keyed changes.
#[derive(Debug)]
pub(crate) enum Changes<T> {
    /// No changes accumulated.
    None,
    /// A single change, stored inline.
    One(T),
    /// Two changes, stored inline.
    Two(T, T),
    /// More than two changes, moved to the heap.
    Many(Box<ManyChanges<T>>),
}

impl<T> Default for Changes<T>
where
    T: KeyedChange,
{
    #[inline]
    fn default() -> Self {
        Self::None
    }
}

impl<T> Changes<T>
where
    T: KeyedChange,
{
    /// Returns whether no change is accumulated.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Replaces an already-accumulated change with the same key.
    ///
    /// Returns `false` when the key is new and nothing was replaced.
    #[inline]
    pub(crate) fn replace_existing(&mut self, value: T) -> bool {
        let key = value.key();

        match self {
            Changes::None => false,
            Changes::One(first) => {
                if (*first).key() == key {
                    *first = value;
                    true
                } else {
                    false
                }
            }
            Changes::Two(first, second) => {
                if (*first).key() == key {
                    *first = value;
                    true
                } else if (*second).key() == key {
                    *second = value;
                    true
                } else {
                    false
                }
            }
            Changes::Many(boxed_changes) => {
                let ManyChanges { values, positions } = boxed_changes.as_mut();

                let Some(&index) = positions.get(key) else {
                    return false;
                };

                if index == MISSING_POSITION {
                    return false;
                }

                values[index] = value;
                true
            }
        }
    }

    /// Inserts a new change or replaces one with the same key.
    #[inline]
    pub(crate) fn upsert(&mut self, value: T) {
        if self.replace_existing(value) {
            return;
        }

        match self {
            Changes::None => {
                *self = Self::One(value);
            }
            Changes::One(first) => {
                let first = *first;
                *self = Self::Two(first, value);
            }
            Changes::Two(first, second) => {
                let first = *first;
                let second = *second;

                let values = vec![first, second, value];
                let max_key = values
                    .iter()
                    .copied()
                    .map(KeyedChange::key)
                    .max()
                    .expect("three changes must have a maximum key");
                let mut positions = vec![MISSING_POSITION; max_key + 1];

                for (index, change) in values.iter().copied().enumerate() {
                    let position = &mut positions[change.key()];
                    debug_assert_eq!(*position, MISSING_POSITION);
                    *position = index;
                }

                *self = Changes::Many(Box::new(ManyChanges { values, positions }));
            }
            Changes::Many(boxed_changes) => {
                let ManyChanges { values, positions } = boxed_changes.as_mut();

                let index = values.len();
                let key = value.key();

                if key >= positions.len() {
                    positions.resize(key + 1, MISSING_POSITION);
                }

                debug_assert_eq!(
                    positions[key], MISSING_POSITION,
                    "upsert tried to insert an existing key"
                );

                positions[key] = index;
                values.push(value);
            }
        }
    }

    /// Consumes the accumulator, invoking `f` for every change in insertion
    /// order.
    pub(crate) fn for_each<F>(self, mut f: F)
    where
        F: FnMut(T),
    {
        match self {
            Changes::None => {}
            Changes::One(first) => f(first),
            Changes::Two(first, second) => {
                f(first);
                f(second)
            }
            Changes::Many(boxed_changes) => boxed_changes.values.into_iter().for_each(f),
        }
    }
}
