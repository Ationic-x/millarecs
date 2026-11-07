//! Sorted archetype signatures.
//!
//! A signature is the canonical table-component set for an archetype. Keeping
//! component ids sorted allows fast membership checks, deterministic column
//! order, and cheap set comparisons during query planning and migration.

use core::cmp::Ordering::*;

use crate::component::ComponentId;

use super::ArchetypeError;

/// Sorted set of table component ids defining an archetype layout.
#[derive(PartialEq, Eq, Clone, Hash, Default)]
pub struct ArchetypeSignature {
    components: Vec<ComponentId>,
}

impl ArchetypeSignature {
    /// Returns whether the signature contains `component_id`.
    pub fn contains(&self, component_id: ComponentId) -> bool {
        self.components.binary_search(&component_id).is_ok()
    }

    /// Returns the sorted component ids in this signature.
    pub fn components(&self) -> &[ComponentId] {
        &self.components
    }

    /// Inserts one component while preserving sorted order and uniqueness.
    pub fn insert(&mut self, component_id: ComponentId) {
        if let Result::Err(index) = self.components.binary_search(&component_id) {
            self.components.insert(index, component_id);
        }
    }

    /// Removes `component_id` if it is present.
    pub fn remove(&mut self, component_id: ComponentId) {
        if let Result::Ok(index) = self.components.binary_search(&component_id) {
            self.components.remove(index);
        }
    }

    /// Keeps only component ids also present in the sorted `component_ids` slice.
    pub fn retain(&mut self, component_ids: &[ComponentId]) {
        self.components
            .retain(|component_id| component_ids.binary_search(component_id).is_ok());
    }

    /// Returns whether all ids in the sorted `need` slice are present.
    pub fn is_superset_of(&self, need: &[ComponentId]) -> bool {
        let components = &self.components;

        if need.len() > components.len() {
            return false;
        }

        let mut component_i = 0usize;
        let mut need_i = 0usize;

        while component_i < components.len() && need_i < need.len() {
            match components[component_i].cmp(&need[need_i]) {
                Less => component_i += 1,
                Equal => {
                    component_i += 1;
                    need_i += 1;
                }
                Greater => return false,
            }
        }

        need_i == need.len()
    }

    /// Returns whether the signature has no ids in common with the sorted `other` slice.
    pub fn is_disjoint_with(&self, other: &[ComponentId]) -> bool {
        let t = &self.components;

        let mut component_i = 0usize;
        let mut other_i = 0usize;

        while component_i < t.len() && other_i < other.len() {
            match t[component_i].cmp(&other[other_i]) {
                Less => component_i += 1,
                Equal => return false,
                Greater => other_i += 1,
            }
        }

        true
    }

    /// Returns the column position assigned to `component_id`.
    pub fn column_index(&self, component_id: ComponentId) -> Result<usize, ArchetypeError> {
        match self.components.binary_search(&component_id) {
            Ok(i) => Ok(i),
            Err(_) => Err(ArchetypeError::SignatureMismatch),
        }
    }
}
