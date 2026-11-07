//! Allocation-free duplicate detection for repeated entity batches.

use std::collections::HashSet;

use crate::entity::Entity;

use super::{World, WorldError};
use crate::entity::EntityError;

/// One 64-slot bitset word, tagged with the epoch that last wrote it.
///
/// A word whose stored epoch differs from the current one is logically empty;
/// that lazily invalidated state is what removes the need for bulk clears
/// between batches.
#[derive(Clone, Copy, Default)]
struct ValidationWord {
    bits: u64,
    epoch: u32,
}

/// Reusable dense set keyed by entity index.
///
/// One epoch-tagged word represents 64 entity slots, keeping retained memory
/// small while avoiding per-batch allocation, hashing and whole-buffer clears.
/// Inputs outside materialized entity metadata, including the invalid sentinel,
/// use an exact fallback so hostile indices cannot force an oversized resize.
#[derive(Default)]
pub(super) struct BatchEntityValidator {
    words: Vec<ValidationWord>,
    epoch: u32,
}

impl BatchEntityValidator {
    /// Starts a new validation epoch and returns its tag.
    ///
    /// Stale words are reset lazily on first touch. When the counter wraps to
    /// zero, every word is cleared physically, because the subsequent epoch
    /// values would otherwise collide with those still stored in stale words
    /// and report false duplicates.
    fn begin(&mut self) -> u32 {
        self.epoch = self.epoch.wrapping_add(1);

        if self.epoch == 0 {
            for word in &mut self.words {
                word.epoch = 0;
                word.bits = 0;
            }
            self.epoch = 1;
        }

        self.epoch
    }

    /// Returns the first duplicate entity in the batch, if any.
    ///
    /// Indices are marked in epoch-tagged words. Invalid handles, indices
    /// outside the materialized slot range, and repeated indices all defer to
    /// `exact_duplicate`, which compares full handles and cannot be confused
    /// by stale generations.
    pub(super) fn find_duplicate(
        &mut self,
        entities: &[Entity],
        materialized_slots: usize,
    ) -> Option<Entity> {
        let epoch = self.begin();

        for &entity in entities {
            let index = entity.index() as usize;

            if !entity.is_valid() || index >= materialized_slots {
                return exact_duplicate(entities);
            }

            let word_index = index >> 6;
            let bit = 1_u64 << (index & 63);

            if word_index >= self.words.len() {
                self.words.resize(word_index + 1, ValidationWord::default());
            }

            let word = &mut self.words[word_index];
            if word.epoch != epoch {
                word.epoch = epoch;
                word.bits = 0;
            }

            if word.bits & bit != 0 {
                // Equal indices normally imply an exact duplicate for a live
                // batch. Fall back to the complete handle so stale handles
                // with different generations preserve the previous semantics.
                return exact_duplicate(entities);
            }

            word.bits |= bit;
        }

        None
    }
}

/// Exact duplicate search by full handle, used when the fast epoch path
/// cannot prove uniqueness safely.
fn exact_duplicate(entities: &[Entity]) -> Option<Entity> {
    let mut seen = HashSet::with_capacity(entities.len());

    entities
        .iter()
        .copied()
        .find(|entity| !seen.insert((entity.index(), entity.generation())))
}

impl World {
    /// Validates one entity batch without allocating on the normal live path.
    pub(crate) fn validate_unique_batch_entities(
        &mut self,
        entities: &[Entity],
    ) -> Result<(), WorldError> {
        let materialized_slots = self.entities.materialized_slot_capacity();

        if let Some(entity) = self
            .batch_entity_validator
            .find_duplicate(entities, materialized_slots)
        {
            return Err(EntityError::Duplicate(entity).into());
        }

        Ok(())
    }
}
