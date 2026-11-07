//! Registry, topology cache, and owner of all archetypes in a world.

use std::collections::HashMap;

use crate::{
    change_detection::ChangeTick,
    component::{
        archetype::plan::batch_transition::{
            execute_migration_batch, execute_structural_migration_batch, write_bundle_batch,
            BatchTransitionInput, BatchTransitionResult,
        },
        bundle::{BatchBundleStorage, ErasedBundleItem},
        bundle_layout::{BundleLayoutId, BundleSlot},
        ArchetypeError, Bundle, BundleError, MovedEntity,
    },
    entity::{ArchetypeStorageKind, Entity, EntityLocation},
};

use super::{
    plan::{
        spawn::{compile_spawn_plan, execute_spawn_plan, execute_spawn_plan_batch, SpawnPlanId},
        transition::{
            compile_structural_plan, compile_transition_plan, execute_in_place, execute_migration,
            execute_structural_migration, StructuralPlanId, StructuralTransitionKey,
            TableStructuralPlan, TransitionPlan, TransitionPlanId, TransitionResult,
        },
        ArchetypePlanCache,
    },
    Archetype, ArchetypeColumn, ArchetypeId, ArchetypeManagerError, ArchetypeSignature,
    BatchTransitionRow, CompactionPolicy, FragmentedCompactionReport,
};

/// Registry and storage owner for all created archetypes.
#[derive(Default)]
pub(crate) struct ArchetypeManager {
    archetypes: Vec<Archetype>,
    signatures: HashMap<ArchetypeSignature, ArchetypeId>,
    plans: ArchetypePlanCache,
}

impl ArchetypeManager {
    /// Creates an empty manager.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Returns the archetype registered for `signature`, if any.
    pub(crate) fn get_id(&self, signature: &ArchetypeSignature) -> Option<ArchetypeId> {
        self.signatures.get(signature).copied()
    }

    /// Registers a new archetype with its columns and signature.
    pub(crate) fn create_archetype(
        &mut self,
        contiguous_columns: Vec<ArchetypeColumn>,
        fragmented_columns: Vec<ArchetypeColumn>,
        signature: ArchetypeSignature,
    ) -> Result<ArchetypeId, ArchetypeManagerError> {
        let id = ArchetypeId::new(self.archetypes.len());
        let archetype = Archetype::new(
            id,
            signature.clone(),
            contiguous_columns,
            fragmented_columns,
        )?;

        self.archetypes.push(archetype);
        self.plans.add_archetype();
        self.signatures.insert(signature, id);

        Ok(id)
    }

    #[inline]
    /// Returns the archetype registered under `id`, if any.
    pub(crate) fn get_archetype(&self, id: ArchetypeId) -> Option<&Archetype> {
        self.archetypes.get(id.index())
    }

    #[inline]
    /// Returns mutable access to the archetype registered under `id`, if any.
    pub(crate) fn get_archetype_mut(&mut self, id: ArchetypeId) -> Option<&mut Archetype> {
        self.archetypes.get_mut(id.index())
    }

    #[inline]
    /// Returns the cached transition plan for `(source, layout)`, if compiled.
    pub(crate) fn transition_plan(
        &self,
        source: ArchetypeId,
        layout: BundleLayoutId,
    ) -> Option<TransitionPlanId> {
        self.plans.transition_plan(source, layout)
    }

    /// Returns the cached spawn plan for `layout`, if compiled.
    pub(crate) fn spawn_plan(&self, layout: BundleLayoutId) -> Option<SpawnPlanId> {
        self.plans.spawn_plan(layout)
    }

    /// Compiles and caches the table transition for `(source, layout)`.
    pub(crate) fn compile_transition_plan(
        &mut self,
        source: ArchetypeId,
        destination: ArchetypeId,
        layout: BundleLayoutId,
        table: &[BundleSlot],
    ) -> Result<TransitionPlanId, BundleError> {
        if let Some(id) = self.transition_plan(source, layout) {
            return Ok(id);
        }

        let source_components = self
            .get_archetype(source)
            .ok_or(BundleError::InvalidArchetype(source))?
            .signature()
            .components();
        let destination_components = self
            .get_archetype(destination)
            .ok_or(BundleError::InvalidArchetype(destination))?
            .signature()
            .components();

        let plan = compile_transition_plan(
            source,
            destination,
            source_components,
            destination_components,
            table,
        )?;
        Ok(self.plans.insert_transition(source, layout, plan))
    }

    /// Compiles and caches the spawn plan for `layout` into `destination`.
    pub(crate) fn compile_spawn_plan(
        &mut self,
        layout: BundleLayoutId,
        destination: ArchetypeId,
        table: &[BundleSlot],
    ) -> Result<SpawnPlanId, BundleError> {
        if let Some(id) = self.spawn_plan(layout) {
            return Ok(id);
        }

        let destination_components = self
            .get_archetype(destination)
            .ok_or(BundleError::InvalidArchetype(destination))?
            .signature()
            .components();

        let plan = compile_spawn_plan(destination, destination_components, table)?;

        Ok(self.plans.insert_spawn(layout, plan))
    }

    /// Executes one entity's transition: in-place writes or a full migration.
    pub(crate) fn execute_transition_plan(
        &mut self,
        plan_id: TransitionPlanId,
        storage: ArchetypeStorageKind,
        source_row: usize,
        items: &mut [ErasedBundleItem],
        change_tick: ChangeTick,
    ) -> Result<TransitionResult, BundleError> {
        let (plans, archetypes) = (&self.plans, &mut self.archetypes);
        let plan = plans.transition(plan_id);

        match plan {
            TransitionPlan::InPlace { source, writes } => execute_in_place(
                &mut archetypes[source.index()],
                storage,
                source_row,
                writes,
                items,
                change_tick,
            ),
            TransitionPlan::Migrate {
                source,
                destination,
                operations,
            } => {
                let (source, destination) = get_two_mut(archetypes, *source, *destination)
                    .ok_or(BundleError::InvalidArchetype(*source))?;
                execute_migration(
                    source,
                    destination,
                    storage,
                    source_row,
                    operations,
                    items,
                    change_tick,
                )
            }
        }
    }

    /// Returns the moved-column count of a migration plan, or `None` for
    /// in-place plans.
    pub(crate) fn transition_migration_column_count(
        &self,
        plan_id: TransitionPlanId,
    ) -> Option<usize> {
        match self.plans.transition(plan_id) {
            TransitionPlan::Migrate { operations, .. } => Some(operations.len()),
            TransitionPlan::InPlace { .. } => None,
        }
    }

    /// Migrates a batch of entities between two archetypes column-major.
    pub(crate) fn execute_transition_batch(
        &mut self,
        plan_id: TransitionPlanId,
        storage: ArchetypeStorageKind,
        inputs: &[BatchTransitionInput],
    ) -> Result<BatchTransitionResult, BundleError> {
        let (plans, archetypes) = (&self.plans, &mut self.archetypes);

        let TransitionPlan::Migrate {
            source,
            destination,
            operations,
        } = plans.transition(plan_id)
        else {
            return Err(BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch));
        };

        let (source, destination) = get_two_mut(archetypes, *source, *destination)
            .ok_or(BundleError::InvalidArchetype(*source))?;

        execute_migration_batch(source, destination, storage, inputs, operations)
    }

    /// Writes bundle-owned values into the migrated destination rows.
    pub(crate) fn write_transition_batch<B: Bundle>(
        &mut self,
        plan_id: TransitionPlanId,
        storage: ArchetypeStorageKind,
        rows: &[BatchTransitionRow],
        bundles: &mut BatchBundleStorage<B>,
        tick: ChangeTick,
    ) -> Result<(), BundleError> {
        let (plans, archetypes) = (&self.plans, &mut self.archetypes);

        let TransitionPlan::Migrate {
            destination,
            operations,
            ..
        } = plans.transition(plan_id)
        else {
            return Err(BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch));
        };

        let destination = archetypes
            .get_mut(destination.index())
            .ok_or(BundleError::InvalidArchetype(*destination))?;

        write_bundle_batch(destination, storage, rows, operations, bundles, tick);
        Ok(())
    }

    /// Allocates one row in the destination archetype and executes the plan.
    ///
    /// Returns the destination archetype and the new row index.
    pub(crate) fn execute_spawn_plan(
        &mut self,
        plan_id: SpawnPlanId,
        storage: ArchetypeStorageKind,
        entity: Entity,
        items: &mut [ErasedBundleItem],
        change_tick: ChangeTick,
    ) -> Result<(ArchetypeId, usize), BundleError> {
        let (plans, archetypes) = (&self.plans, &mut self.archetypes);
        let plan = plans.spawn(plan_id);
        let destination = plan.destination;
        let archetype = archetypes
            .get_mut(destination.index())
            .ok_or(BundleError::InvalidArchetype(destination))?;
        let row = execute_spawn_plan(archetype, plan, storage, entity, items, change_tick);

        Ok((destination, row))
    }

    /// Executes a homogeneous spawn batch into one destination archetype.
    pub(crate) fn execute_spawn_plan_batch<B: Bundle>(
        &mut self,
        plan_id: SpawnPlanId,
        storage: ArchetypeStorageKind,
        entities: &[Entity],
        bundles: &mut BatchBundleStorage<B>,
        change_tick: ChangeTick,
        activations: &mut Vec<(Entity, EntityLocation)>,
    ) -> Result<ArchetypeId, BundleError> {
        let (plans, archetypes) = (&self.plans, &mut self.archetypes);
        let plan = plans.spawn(plan_id);
        let destination = plan.destination;
        let archetype = archetypes
            .get_mut(destination.index())
            .ok_or(BundleError::InvalidArchetype(destination))?;

        execute_spawn_plan_batch(
            archetype,
            plan,
            storage,
            entities,
            bundles,
            change_tick,
            activations,
        );

        Ok(destination)
    }

    /// Returns the cached structural plan for `(source, key)`, if compiled.
    #[inline]
    pub(crate) fn structural_plan(
        &self,
        source: ArchetypeId,
        key: StructuralTransitionKey,
    ) -> Option<StructuralPlanId> {
        self.plans.structural_plan(source, key)
    }

    /// Compiles and caches the structural plan for `(source, key)`.
    pub(crate) fn compile_structural_plan(
        &mut self,
        source: ArchetypeId,
        destination: ArchetypeId,
        key: StructuralTransitionKey,
    ) -> Result<StructuralPlanId, BundleError> {
        if let Some(id) = self.structural_plan(source, key) {
            return Ok(id);
        }

        let source_components = self
            .get_archetype(source)
            .ok_or(BundleError::InvalidArchetype(source))?
            .signature()
            .components();
        let destination_components = self
            .get_archetype(destination)
            .ok_or(BundleError::InvalidArchetype(destination))?
            .signature()
            .components();

        let plan = compile_structural_plan(
            source,
            destination,
            source_components,
            destination_components,
        )?;
        Ok(self.plans.insert_structural(source, key, plan))
    }

    /// Executes one row's structural plan: a no-op when unchanged, otherwise
    /// a migration to the destination archetype.
    pub(crate) fn execute_structural_plan(
        &mut self,
        plan_id: StructuralPlanId,
        storage: ArchetypeStorageKind,
        source_row: usize,
    ) -> Result<TransitionResult, BundleError> {
        let (plans, archetypes) = (&self.plans, &mut self.archetypes);
        let plan = plans.structural(plan_id);

        match plan {
            TableStructuralPlan::Unchanged { source } => {
                Ok(TransitionResult::unchanged(*source, source_row))
            }
            TableStructuralPlan::Migrate {
                source,
                destination,
                operations,
                ..
            } => {
                let (source, destination) = get_two_mut(archetypes, *source, *destination)
                    .ok_or(BundleError::InvalidArchetype(*source))?;
                execute_structural_migration(source, destination, storage, source_row, operations)
            }
        }
    }

    /// Returns whether the plan migrates the entity to another archetype.
    pub(crate) fn structural_plan_is_migration(&self, plan_id: StructuralPlanId) -> bool {
        matches!(
            self.plans.structural(plan_id),
            TableStructuralPlan::Migrate { .. }
        )
    }

    /// Returns the shared-column width copied by a structural migration.
    #[inline]
    pub(crate) fn structural_plan_moved_columns(&self, plan_id: StructuralPlanId) -> usize {
        match self.plans.structural(plan_id) {
            TableStructuralPlan::Unchanged { .. } => 0,
            TableStructuralPlan::Migrate { moved_columns, .. } => *moved_columns,
        }
    }

    /// Migrates a batch of rows between two archetypes column-major.
    pub(crate) fn execute_structural_transition_batch(
        &mut self,
        plan_id: StructuralPlanId,
        storage: ArchetypeStorageKind,
        source_rows: &[usize],
    ) -> Result<BatchTransitionResult, BundleError> {
        let (plans, archetypes) = (&self.plans, &mut self.archetypes);

        let TableStructuralPlan::Migrate {
            source,
            destination,
            operations,
            ..
        } = plans.structural(plan_id)
        else {
            return Err(BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch));
        };

        let (source, destination) = get_two_mut(archetypes, *source, *destination)
            .ok_or(BundleError::InvalidArchetype(*source))?;

        execute_structural_migration_batch(source, destination, storage, source_rows, operations)
    }

    /// Compacts fragmented lanes of every archetype under `policy`.
    ///
    /// Returns the displaced entities for location repair plus a summary.
    pub(crate) fn compact_fragmented_with_policy(
        &mut self,
        policy: CompactionPolicy,
    ) -> (Vec<MovedEntity>, FragmentedCompactionReport) {
        let mut moved = Vec::new();
        let mut report = FragmentedCompactionReport::default();

        for archetype in &mut self.archetypes {
            let Some(mut archetype_moved) = archetype.compact_fragmented_with_policy(policy) else {
                continue;
            };

            report.compacted_archetypes += 1;
            report.moved_rows += archetype_moved.len();
            moved.append(&mut archetype_moved);
        }

        (moved, report)
    }

    #[inline]
    /// Returns the number of registered archetypes.
    pub(crate) fn len(&self) -> usize {
        self.archetypes.len()
    }

    #[cfg(test)]
    pub(crate) fn transition_plan_count(&self) -> usize {
        self.plans.transition_count()
    }

    #[cfg(test)]
    pub(crate) fn spawn_plan_count(&self) -> usize {
        self.plans.spawn_count()
    }
}

/// Borrows two distinct archetypes mutably, or `None` on equal or unknown ids.
///
/// The slice is split once at the higher index so each borrow targets a
/// disjoint half, and the halves are reordered back to argument order.
fn get_two_mut(
    archetypes: &mut [Archetype],
    first: ArchetypeId,
    second: ArchetypeId,
) -> Option<(&mut Archetype, &mut Archetype)> {
    let first_index = first.index();
    let second_index = second.index();

    if first_index == second_index
        || first_index >= archetypes.len()
        || second_index >= archetypes.len()
    {
        return None;
    }

    let (lower, higher, swapped) = if first_index < second_index {
        (first_index, second_index, false)
    } else {
        (second_index, first_index, true)
    };

    let (left, right) = archetypes.split_at_mut(higher);
    let lower = &mut left[lower];
    let higher = &mut right[0];

    Some(if swapped {
        (higher, lower)
    } else {
        (lower, higher)
    })
}
