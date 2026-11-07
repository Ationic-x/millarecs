//! Cached archetype operations compiled from stable world metadata.

use crate::{
    component::{bundle_layout::BundleLayoutId, ArchetypeId},
    utils::flat_map::FlatMap,
};

use self::{
    spawn::{SpawnPlan, SpawnPlanId},
    transition::{
        StructuralPlanId, StructuralTransitionKey, TableStructuralPlan, TransitionPlan,
        TransitionPlanId,
    },
};

pub(crate) mod batch_transition;
pub(crate) mod spawn;
pub(crate) mod transition;

/// Owns every precompiled archetype plan and its lookup indices.
#[derive(Default)]
pub(crate) struct ArchetypePlanCache {
    /// Transition edges indexed by source archetype, keyed by bundle layout.
    transition_edges: Vec<FlatMap<BundleLayoutId, TransitionPlanId>>,
    /// Transition plans appended in creation order.
    transition_plans: Vec<TransitionPlan>,
    /// Structural edges indexed by source archetype, keyed by structural operation.
    structural_edges: Vec<FlatMap<StructuralTransitionKey, StructuralPlanId>>,
    /// Structural plans appended in creation order.
    structural_plans: Vec<TableStructuralPlan>,
    /// Spawn plan per bundle layout, indexed by layout index.
    spawn_edges: Vec<Option<SpawnPlanId>>,
    /// Spawn plans appended in creation order.
    spawn_plans: Vec<SpawnPlan>,
}

impl ArchetypePlanCache {
    /// Adds empty edge maps for a newly registered archetype.
    pub(crate) fn add_archetype(&mut self) {
        self.transition_edges.push(FlatMap::new());
        self.structural_edges.push(FlatMap::new());
    }

    /// Returns the cached transition edge for `(source, layout)`.
    pub(crate) fn transition_plan(
        &self,
        source: ArchetypeId,
        layout: BundleLayoutId,
    ) -> Option<TransitionPlanId> {
        self.transition_edges.get(source.index())?.get_copy(layout)
    }

    /// Caches a transition plan under `(source, layout)`.
    pub(crate) fn insert_transition(
        &mut self,
        source: ArchetypeId,
        layout: BundleLayoutId,
        plan: TransitionPlan,
    ) -> TransitionPlanId {
        let id = TransitionPlanId::new(self.transition_plans.len());
        self.transition_plans.push(plan);
        self.transition_edges[source.index()].insert(layout, id);
        id
    }

    /// Returns the transition plan stored under `id`.
    pub(crate) fn transition(&self, id: TransitionPlanId) -> &TransitionPlan {
        self.transition_plans
            .get(id.index())
            .expect("transition plan id must belong to this cache")
    }

    /// Returns the cached structural edge for `(source, key)`.
    pub(crate) fn structural_plan(
        &self,
        source: ArchetypeId,
        key: StructuralTransitionKey,
    ) -> Option<StructuralPlanId> {
        self.structural_edges.get(source.index())?.get_copy(key)
    }

    /// Caches a structural plan under `(source, key)`.
    pub(crate) fn insert_structural(
        &mut self,
        source: ArchetypeId,
        key: StructuralTransitionKey,
        plan: TableStructuralPlan,
    ) -> StructuralPlanId {
        let id = StructuralPlanId::new(self.structural_plans.len());
        self.structural_plans.push(plan);
        self.structural_edges[source.index()].insert(key, id);
        id
    }

    /// Returns the structural plan stored under `id`.
    pub(crate) fn structural(&self, id: StructuralPlanId) -> &TableStructuralPlan {
        self.structural_plans
            .get(id.index())
            .expect("structural plan id must belong to this cache")
    }

    /// Returns the cached spawn plan for `layout`, if compiled.
    pub(crate) fn spawn_plan(&self, layout: BundleLayoutId) -> Option<SpawnPlanId> {
        self.spawn_edges.get(layout.index()).copied().flatten()
    }

    /// Caches a spawn plan under `layout`.
    pub(crate) fn insert_spawn(&mut self, layout: BundleLayoutId, plan: SpawnPlan) -> SpawnPlanId {
        debug_assert!(self.spawn_plan(layout).is_none());

        let id = SpawnPlanId::new(self.spawn_plans.len());
        self.spawn_plans.push(plan);

        let layout_index = layout.index();
        if layout_index >= self.spawn_edges.len() {
            self.spawn_edges.resize(layout_index + 1, None);
        }
        self.spawn_edges[layout_index] = Some(id);
        id
    }

    /// Returns the spawn plan stored under `id`.
    pub(crate) fn spawn(&self, id: SpawnPlanId) -> &SpawnPlan {
        self.spawn_plans
            .get(id.index())
            .expect("spawn plan id must belong to this cache")
    }

    #[cfg(test)]
    pub(crate) fn transition_count(&self) -> usize {
        self.transition_plans.len() + self.structural_plans.len()
    }

    #[cfg(test)]
    pub(crate) fn spawn_count(&self) -> usize {
        self.spawn_plans.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_plan(destination: usize) -> SpawnPlan {
        SpawnPlan {
            destination: ArchetypeId::new(destination),
            bundle_items_by_column: Box::new([]),
        }
    }

    #[test]
    fn inserting_lower_spawn_layout_does_not_discard_higher_layout() {
        let mut cache = ArchetypePlanCache::default();
        let higher_layout = BundleLayoutId::new(4);
        let lower_layout = BundleLayoutId::new(1);

        let higher_plan = cache.insert_spawn(higher_layout, spawn_plan(4));
        let lower_plan = cache.insert_spawn(lower_layout, spawn_plan(1));

        assert_eq!(cache.spawn_plan(higher_layout), Some(higher_plan));
        assert_eq!(cache.spawn_plan(lower_layout), Some(lower_plan));
    }
}
