//! Process-local world identifiers.
//!
//! A world id lets query state and system state detect whether cached metadata
//! belongs to the world currently being executed.

use std::{
    fmt,
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

/// Unique identifier assigned to a world.
///
/// Identifiers are unique within the current process and are never reused.
/// The non-zero inner representation exploits the `NonZeroU64` niche, so
/// `Option<WorldId>` stays the same size as `WorldId`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct WorldId(NonZeroU64);

/// Counter handing out identifiers; starts at 1 so zero stays reserved for
/// the `NonZeroU64` niche.
static NEXT_WORLD_ID: AtomicU64 = AtomicU64::new(1);

/// Renders the numeric identifier directly.
impl fmt::Display for WorldId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.get().fmt(formatter)
    }
}

impl WorldId {
    /// Allocates a new process-unique world identifier.
    ///
    /// Returns `None` only when the identifier space has been exhausted.
    pub(crate) fn allocate() -> Option<Self> {
        NEXT_WORLD_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .ok()
            .and_then(NonZeroU64::new)
            .map(Self)
    }

    /// Returns the numeric representation of this identifier.
    #[inline]
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[cfg(test)]
mod tests {
    use super::WorldId;

    #[test]
    fn allocated_world_ids_are_unique_and_non_zero() {
        let first = WorldId::allocate().unwrap();
        let second = WorldId::allocate().unwrap();

        assert_ne!(first, second);
        assert_ne!(first.get(), 0);
        assert_ne!(second.get(), 0);
    }
}
