//! Batch-oriented world operations.
//!
//! Batch paths prepare and validate all shared metadata before mutating storage.
//! Keeping them separate from `world::mod` makes their ordering, rollback, and
//! partial-failure contracts easier to audit.

use std::{any::TypeId, cmp::Reverse, collections::HashSet};

use crate::{
    __private::{Bundle, BundleError},
    component::{
        archetype::plan::{
            batch_transition::BatchTransitionInput,
            transition::{PreparedTransitionPlan, StructuralPlanId},
        },
        bundle::BatchBundleStorage,
        Component, SparseInsertBatchEntry, SparseLocationUpdateGroups, SparseMembershipLinks,
        SparseRemovalGroups, SparseSpawnBatchEntry, StorageKind,
    },
    entity::{
        metadata::{ArchetypeStorageKind, EntityLocation},
        Entity, EntityError,
    },
    planning::PlanningChanges,
};

use super::{World, WorldError};

/// One queued bundle insert with the entity's pre-batch location.
struct BatchInsertEntry<B> {
    entity: Entity,
    initial: EntityLocation,
    bundle: B,
}

/// One homogeneous table-insert lane: entities sharing source archetype.
struct BatchInsertRun {
    len: usize,
    storage: ArchetypeStorageKind,
    prepared: PreparedTransitionPlan,
}

/// One homogeneous structural-removal lane.
struct BatchRemoveRun {
    len: usize,
    storage: ArchetypeStorageKind,
    plan: StructuralPlanId,
}

/// One queued entity with its location captured before the batch mutates
/// storage.
#[derive(Clone, Copy)]
struct BatchDespawnEntry {
    entity: Entity,
    initial: EntityLocation,
}

/// Destruction strategy selected for a despawn batch.
#[derive(Clone, Copy)]
enum BatchDespawnRoute {
    /// Component-major pass: table rows are removed per archetype lane and
    /// sparse memberships are transposed into grouped updates.
    Grouped,
    /// Entity-major pass that skips sparse-group preparation. `remove_sparse`
    /// says whether per-entity sparse cleanup is still required; narrow
    /// table-only batches can prove it unnecessary.
    Sequential { remove_sparse: bool },
}

/// Ownership state of the entity handles fed into a batch spawn.
#[derive(Clone, Copy)]
enum BatchSpawnEntityState {
    /// Handles come from allocator reservations: they are validated with
    /// `ensure_reserved_many` and activated once storage mutation succeeds.
    Reserved,
    /// Handles are allocated but not yet live: they are published with
    /// `commit_spawn_many`, or aborted wholesale when a spawn fails.
    Unpublished,
}

/// Minimum despawn batch size for which the grouped component-major pass pays
/// for itself.
///
/// Below this size, sorting plus component-major transposition costs more than
/// the locality it recovers on the common narrow-batch path. The dedicated
/// scaling benchmark keeps this crossover visible and easy to retune.
const MIN_GROUPED_DESPAWN_ENTITIES: usize = 256;

/// Maximum number of table columns for a contiguous-only despawn batch to
/// stay on the sequential route.
///
/// Contiguous tables this narrow do not amortize sparse-group preparation or
/// the displaced-row buffer when no sparse membership is present.
const MAX_SEQUENTIAL_TABLE_COLUMNS: usize = 2;

/// Packs an entity's index and generation into one `u64` duplicate-detection key.
#[inline]
fn entity_key(entity: Entity) -> u64 {
    (u64::from(entity.index()) << 32) | u64::from(entity.generation())
}

/// Ranks storage kinds so contiguous lanes sort before fragmented ones.
#[inline]
fn storage_rank(storage: ArchetypeStorageKind) -> u8 {
    match storage {
        ArchetypeStorageKind::Contiguous => 0,
        ArchetypeStorageKind::Fragmented => 1,
    }
}

/// Sorts insert entries by source archetype, then storage rank, then
/// descending row; descending rows are a correctness requirement because
/// contiguous swap-removal can displace rows still pending in the same lane.
#[inline]
fn batch_entry_key<B>(entry: &BatchInsertEntry<B>) -> (usize, u8, Reverse<u32>) {
    (
        entry.initial.archetype_id.index(),
        storage_rank(entry.initial.storage),
        Reverse(entry.initial.row),
    )
}

/// Sorts structural-removal entries with the same key shape as
/// `batch_entry_key`: source archetype, storage rank, then descending row.
#[inline]
fn structural_entry_key(entry: &BatchDespawnEntry) -> (usize, u8, Reverse<u32>) {
    (
        entry.initial.archetype_id.index(),
        storage_rank(entry.initial.storage),
        Reverse(entry.initial.row),
    )
}

/// Sorts despawn entries by source archetype, storage rank, and a per-storage
/// row key that yields descending contiguous rows but ascending fragmented
/// rows under one ordinary ascending tuple sort.
#[inline]
fn despawn_entry_key(entry: &BatchDespawnEntry) -> (usize, u8, u32) {
    let row_key = match entry.initial.storage {
        // Transforming the row produces descending row order while preserving
        // an ordinary ascending tuple sort.
        ArchetypeStorageKind::Contiguous => u32::MAX - entry.initial.row,

        // Ascending fragmented rows produce ascending page groups and improve
        // sequential access to component pages.
        ArchetypeStorageKind::Fragmented => entry.initial.row,
    };

    (
        entry.initial.archetype_id.index(),
        storage_rank(entry.initial.storage),
        row_key,
    )
}

/// Returns whether two entries share the same archetype and storage lane.
#[inline]
fn same_despawn_lane(left: &BatchDespawnEntry, right: &BatchDespawnEntry) -> bool {
    left.initial.archetype_id == right.initial.archetype_id
        && left.initial.storage == right.initial.storage
}

/// Validates uniqueness of a growing insert batch, including the entry being
/// built and the caller-supplied remainder.
fn validate_unique_insert_parts<B>(
    prepared: &[BatchInsertEntry<B>],
    current: Entity,
    remaining: &[(Entity, B)],
) -> Result<(), WorldError> {
    let mut seen = HashSet::with_capacity(prepared.len() + 1 + remaining.len());

    for entry in prepared {
        if !seen.insert(entity_key(entry.entity)) {
            return Err(EntityError::Duplicate(entry.entity).into());
        }
    }

    if !seen.insert(entity_key(current)) {
        return Err(EntityError::Duplicate(current).into());
    }

    for (entity, _) in remaining {
        if !seen.insert(entity_key(*entity)) {
            return Err(EntityError::Duplicate(*entity).into());
        }
    }

    Ok(())
}

impl World {
    /// Despawns several live entities through one prevalidated structural pass.
    ///
    /// Small and narrow batches use a low-overhead entity-major pass. Larger
    /// batches group table rows by archetype lane and transpose sparse
    /// memberships into component-major groups. Input already in spawn order
    /// (one contiguous lane, ascending rows) skips the sort through a linear
    /// reversal. Both routes validate the full input before mutating storage
    /// and commit planning once.
    pub fn despawn_batch<I>(&mut self, entities: I) -> Result<usize, WorldError>
    where
        I: IntoIterator<Item = Entity>,
    {
        let entities = entities.into_iter().collect::<Vec<_>>();

        if entities.is_empty() {
            return Ok(0);
        }

        let mut entries = Vec::with_capacity(entities.len());

        for &entity in &entities {
            let Some(location) = self.entities.location(entity) else {
                // Preserve duplicate-before-invalid error precedence without
                // paying for a hash set on the successful path.
                let mut seen = HashSet::with_capacity(entities.len());
                for &candidate in &entities {
                    if !seen.insert(entity_key(candidate)) {
                        return Err(EntityError::Duplicate(candidate).into());
                    }
                }

                return Err(WorldError::EntityNotFound(entity));
            };

            entries.push(BatchDespawnEntry {
                entity,
                initial: location,
            });
        }

        let single_lane = entries
            .windows(2)
            .all(|pair| same_despawn_lane(&pair[0], &pair[1]));

        let ascending_rows = entries
            .windows(2)
            .all(|pair| pair[0].initial.row < pair[1].initial.row);

        let contiguous_spawn_order = single_lane
            && entries[0].initial.storage == ArchetypeStorageKind::Contiguous
            && ascending_rows;

        if contiguous_spawn_order {
            entries.reverse();
        } else if !entries
            .windows(2)
            .all(|pair| despawn_entry_key(&pair[0]) <= despawn_entry_key(&pair[1]))
        {
            entries.sort_unstable_by_key(despawn_entry_key);
        }

        for pair in entries.windows(2) {
            if pair[0].entity == pair[1].entity {
                return Err(EntityError::Duplicate(pair[0].entity).into());
            }
        }

        if let BatchDespawnRoute::Sequential { remove_sparse } =
            self.select_despawn_batch_route(&entries)
        {
            return self.despawn_batch_sequential_prevalidated(&entries, &entities, remove_sparse);
        }

        let mut sparse_groups = SparseRemovalGroups::default();
        self.components
            .prepare_sparse_despawn_batch(&entities, &mut sparse_groups);
        let sparse_changed = !sparse_groups.is_empty();

        let mut planning = PlanningChanges::default();
        let mut displaced = Vec::with_capacity(entries.len());
        let mut fragmented_removals = Vec::new();
        let mut start = 0;

        while start < entries.len() {
            let source = entries[start];
            let mut end = start + 1;

            while end < entries.len() && same_despawn_lane(&source, &entries[end]) {
                end += 1;
            }

            let rows = {
                let archetype = self
                    .components
                    .get_archetype_mut(source.initial.archetype_id)
                    .expect("prevalidated despawn archetype must exist");

                match source.initial.storage {
                    ArchetypeStorageKind::Contiguous => {
                        // Descending order remains mandatory because removing a row can
                        // move the live tail into that position.
                        for entry in &entries[start..end] {
                            let moved = archetype
                                .remove_row(
                                    ArchetypeStorageKind::Contiguous,
                                    entry.initial.row as usize,
                                )
                                .expect("descending prepared row must remain live");

                            displaced.extend(moved);
                        }
                    }

                    ArchetypeStorageKind::Fragmented => {
                        // Fragmented rows never move. The complete lane run can therefore
                        // be removed using page masks and column-major traversal.
                        let removed = archetype.remove_fragmented_rows_prepared(
                            entries[start..end]
                                .iter()
                                .map(|entry| entry.initial.row as usize),
                            &mut fragmented_removals,
                        );

                        debug_assert_eq!(removed, end - start);
                    }
                }

                archetype.len()
            };

            planning.mark_archetype_rows(source.initial.archetype_id, rows);
            start = end;
        }

        self.apply_moved_entities(displaced);
        self.components
            .execute_sparse_despawn_batch(&entities, &mut sparse_groups);

        if sparse_changed {
            planning.mark_query_work_changed();
        }

        self.entities.free_alive_many_prevalidated(&entities);
        self.world_planning.apply(planning);

        Ok(entities.len())
    }

    /// Chooses the low-overhead destruction route for batches whose fixed
    /// preparation cost cannot be recovered through component-major work.
    ///
    /// Batches below `MIN_GROUPED_DESPAWN_ENTITIES` always stay sequential.
    /// Contiguous-only batches whose archetypes hold at most
    /// `MAX_SEQUENTIAL_TABLE_COLUMNS` components also stay sequential and skip
    /// sparse cleanup entirely.
    #[inline]
    fn select_despawn_batch_route(&self, entries: &[BatchDespawnEntry]) -> BatchDespawnRoute {
        if entries.len() < MIN_GROUPED_DESPAWN_ENTITIES {
            return BatchDespawnRoute::Sequential {
                remove_sparse: true,
            };
        }

        let narrow_table_only = entries.iter().all(|entry| {
            if entry.initial.storage != ArchetypeStorageKind::Contiguous
                || self.components.entity_has_sparse_components(entry.entity)
            {
                return false;
            }

            self.components
                .get_archetype(entry.initial.archetype_id)
                .is_some_and(|archetype| {
                    archetype.signature().components().len() <= MAX_SEQUENTIAL_TABLE_COLUMNS
                })
        });

        if narrow_table_only {
            BatchDespawnRoute::Sequential {
                remove_sparse: false,
            }
        } else {
            BatchDespawnRoute::Grouped
        }
    }

    /// Despawns an already validated and lane-sorted batch without building
    /// component-major sparse groups. Planning and entity recycling are still
    /// committed once for the complete batch.
    fn despawn_batch_sequential_prevalidated(
        &mut self,
        entries: &[BatchDespawnEntry],
        entities: &[Entity],
        remove_sparse: bool,
    ) -> Result<usize, WorldError> {
        let mut planning = PlanningChanges::default();
        let mut start = 0;

        while start < entries.len() {
            let source = entries[start];
            let mut end = start + 1;

            while end < entries.len() && same_despawn_lane(&source, &entries[end]) {
                end += 1;
            }

            for entry in &entries[start..end] {
                let moved = {
                    let archetype = self
                        .components
                        .get_archetype_mut(entry.initial.archetype_id)
                        .expect("prevalidated despawn archetype must exist");

                    archetype.remove_row(entry.initial.storage, entry.initial.row as usize)?
                };

                self.apply_moved_entity(moved);
                if remove_sparse {
                    self.components
                        .remove_sparse_components(entry.entity, &mut planning);
                }
            }

            let rows = self
                .components
                .get_archetype(source.initial.archetype_id)
                .expect("prevalidated despawn archetype must exist")
                .len();

            planning.mark_archetype_rows(source.initial.archetype_id, rows);
            start = end;
        }

        self.entities.free_alive_many_prevalidated(entities);
        self.world_planning.apply(planning);

        Ok(entities.len())
    }

    /// Spawns reserved entities with no components into the empty archetype.
    pub(crate) fn spawn_reserved_empty_batch(
        &mut self,
        entities: &[Entity],
    ) -> Result<(), WorldError> {
        self.entities.ensure_reserved_many(entities)?;
        let mut planning = PlanningChanges::default();

        let mut activations = Vec::with_capacity(entities.len());
        let rows = {
            let archetype = self
                .components
                .get_archetype_mut(self.empty_archetype)
                .expect("empty archetype must exist");

            for &entity in entities {
                let row = archetype.push_empty_row(ArchetypeStorageKind::Contiguous, entity)?;

                activations.push((
                    entity,
                    EntityLocation {
                        archetype_id: self.empty_archetype,
                        storage: ArchetypeStorageKind::Contiguous,
                        row: row as u32,
                    },
                ));
            }

            archetype.len()
        };

        self.entities.activate_many(activations)?;

        planning.mark_archetype_rows(self.empty_archetype, rows);
        self.world_planning.apply(planning);

        Ok(())
    }

    /// Removes one component type from several live entities.
    ///
    /// Table-backed components are grouped by source lane. Large migrating
    /// runs execute one component column at a time, while small runs retain the
    /// lower-overhead entity-major path. Sparse components are prepared and
    /// compacted as one component-major group.
    pub fn remove_batch<I, C>(&mut self, entities: I) -> Result<usize, WorldError>
    where
        I: IntoIterator<Item = Entity>,
        C: Component,
    {
        let entities = entities.into_iter().collect::<Vec<_>>();
        if entities.is_empty() {
            return Ok(0);
        }

        self.validate_unique_batch_entities(&entities)?;

        let component = self
            .components
            .id_of::<C>()
            .ok_or(BundleError::ComponentNotRegistered(TypeId::of::<C>()))?;
        let storage = self
            .components
            .storage_of(component)
            .ok_or(BundleError::StorageUnknown(component))?;

        if storage == StorageKind::Sparse {
            for &entity in &entities {
                self.entities
                    .location(entity)
                    .ok_or(WorldError::EntityNotFound(entity))?;
            }

            self.advance_tick();
            let mut planning = PlanningChanges::default();
            self.components
                .remove_sparse_component_batch(component, &entities, &mut planning);
            self.world_planning.apply(planning);
            return Ok(entities.len());
        }

        let mut entries = Vec::with_capacity(entities.len());
        for &entity in &entities {
            let location = self
                .entities
                .location(entity)
                .ok_or(WorldError::EntityNotFound(entity))?;
            entries.push(BatchDespawnEntry {
                entity,
                initial: location,
            });
        }

        self.advance_tick();
        let mut planning = PlanningChanges::default();

        let single_lane = entries
            .windows(2)
            .all(|pair| same_despawn_lane(&pair[0], &pair[1]));
        let ascending_rows = entries
            .windows(2)
            .all(|pair| pair[0].initial.row < pair[1].initial.row);

        if single_lane
            && entries[0].initial.storage == ArchetypeStorageKind::Contiguous
            && ascending_rows
        {
            // Spawn-order batches only need a linear reversal to satisfy the
            // swap-removal contract; an O(N log N) sort would be redundant.
            entries.reverse();
        } else if !entries
            .windows(2)
            .all(|pair| structural_entry_key(&pair[0]) <= structural_entry_key(&pair[1]))
        {
            entries.sort_unstable_by_key(structural_entry_key);
        }

        let mut runs = Vec::new();
        let mut start = 0;

        // Prepare every archetype edge before the first structural mutation.
        while start < entries.len() {
            let source = entries[start].initial.archetype_id;
            let storage = entries[start].initial.storage;
            let mut end = start + 1;

            while end < entries.len()
                && entries[end].initial.archetype_id == source
                && entries[end].initial.storage == storage
            {
                end += 1;
            }

            let plan = self.components.prepare_remove_component_transition(
                source,
                component,
                &mut planning,
            )?;

            runs.push(BatchRemoveRun {
                len: end - start,
                storage,
                plan,
            });
            start = end;
        }

        let mut displaced = Vec::with_capacity(entries.len());
        let mut entries = entries.into_iter();
        let mut location_updates = SparseLocationUpdateGroups::default();

        let execution = (|| {
            for run in runs {
                let run_entries = entries.by_ref().take(run.len).collect::<Vec<_>>();

                if self
                    .components
                    .can_batch_structural_transition(run.plan, run_entries.len())
                {
                    let source_rows = run_entries
                        .iter()
                        .map(|entry| entry.initial.row as usize)
                        .collect::<Vec<_>>();
                    let result = self.components.execute_prepared_structural_batch(
                        run.plan,
                        run.storage,
                        &source_rows,
                        &mut planning,
                    )?;

                    let destination = result.destination;
                    displaced.extend(result.moved_in_source);

                    debug_assert!(run_entries
                        .iter()
                        .zip(&result.rows)
                        .all(|(entry, transitioned)| entry.entity == transitioned.entity));

                    self.apply_entity_locations_with_groups(
                        result.rows.iter().map(|transitioned| {
                            (
                                transitioned.entity,
                                EntityLocation {
                                    archetype_id: destination,
                                    storage: run.storage,
                                    row: transitioned.destination_row as u32,
                                },
                            )
                        }),
                        &mut location_updates,
                    );

                    continue;
                }

                for entry in run_entries {
                    let moved = {
                        let components = &mut self.components;
                        let entity_manager = &mut self.entities;
                        let location = entity_manager
                            .location_mut(entry.entity)
                            .expect("batch entity was prevalidated");

                        components.execute_prepared_structural_transition(
                            entry.entity,
                            run.plan,
                            location,
                            &mut planning,
                        )?
                    };

                    displaced.extend(moved);
                }
            }

            debug_assert!(entries.next().is_none());
            Ok::<_, WorldError>(())
        })();

        self.apply_moved_entities_with_groups(displaced, &mut location_updates);
        self.world_planning.apply(planning);

        execution.map(|()| entities.len())
    }

    /// Applies one bundle layout to several live entities.
    ///
    /// Duplicate handles and initial locations are validated before any storage
    /// mutation. Entries are then sorted by source archetype, storage rank, and
    /// descending row; that ordering is a correctness requirement of contiguous
    /// swap-removal, not an optimization, because removing a row can move the
    /// live tail into a pending row's position.
    ///
    /// On partial failure, migrations completed before the error remain
    /// applied and their metadata repairs are still published.
    pub fn insert_batch<B: Bundle>(&mut self, entries: Vec<(Entity, B)>) -> Result<(), WorldError> {
        if entries.is_empty() {
            return Ok(());
        }

        let mut source_entries = entries.into_iter();
        let mut entries = Vec::with_capacity(source_entries.len());

        while let Some((entity, bundle)) = source_entries.next() {
            let Some(initial) = self.entities.location(entity) else {
                // Duplicate detection normally piggybacks on the structural
                // sort below. Only pay for a hash set on this cold error path
                // to retain the previous duplicate-before-invalid precedence.
                validate_unique_insert_parts(&entries, entity, source_entries.as_slice())?;
                return Err(WorldError::EntityNotFound(entity));
            };

            entries.push(BatchInsertEntry {
                entity,
                initial,
                bundle,
            });
        }

        // This is the only remaining sort. Reverse row order is a correctness
        // requirement for contiguous swap-removal within each source lane.
        entries.sort_unstable_by_key(batch_entry_key);

        // Equal entities necessarily have the same initial location and are
        // therefore adjacent after the structural sort.
        for pair in entries.windows(2) {
            if pair[0].entity == pair[1].entity {
                return Err(EntityError::Duplicate(pair[0].entity).into());
            }
        }

        let tick = self.advance_tick();
        let mut planning = PlanningChanges::default();
        let layout = self.components.ensure_bundle_layout::<B>()?;

        // Prepare every fallible transition before mutating component storage,
        // while retaining one compact plan per source-archetype run.
        let mut runs = Vec::new();
        let mut start = 0;

        while start < entries.len() {
            let source = entries[start].initial.archetype_id;
            let storage = entries[start].initial.storage;
            let mut end = start + 1;

            while end < entries.len()
                && entries[end].initial.archetype_id == source
                && entries[end].initial.storage == storage
            {
                end += 1;
            }

            let prepared =
                self.components
                    .prepare_transition_plan(source, layout, &mut planning)?;

            runs.push(BatchInsertRun {
                len: end - start,
                storage,
                prepared,
            });

            start = end;
        }

        let mut displaced = Vec::with_capacity(entries.len());
        let mut entries = entries.into_iter();
        let mut transition_inputs = Vec::new();
        let mut bundle_scratch = Vec::new();
        let mut sparse_entries = Vec::<SparseInsertBatchEntry>::new();
        let mut location_updates = SparseLocationUpdateGroups::default();

        let execution = (|| {
            for run in runs {
                if self
                    .components
                    .can_batch_table_transition(run.prepared, run.len)
                {
                    transition_inputs.clear();
                    transition_inputs.reserve(run.len);
                    bundle_scratch.clear();
                    bundle_scratch.reserve(run.len);

                    for entry in entries.by_ref().take(run.len) {
                        transition_inputs.push(BatchTransitionInput {
                            entity: entry.entity,
                            source_row: entry.initial.row as usize,
                        });
                        bundle_scratch.push(entry.bundle);
                    }

                    debug_assert_eq!(transition_inputs.len(), run.len);
                    debug_assert_eq!(bundle_scratch.len(), run.len);

                    self.components
                        .prepare_sparse_transition_batch(run.prepared, &transition_inputs)?;

                    let item_count = self.components.prepared_bundle_item_count(run.prepared);
                    let mut bundle_storage =
                        BatchBundleStorage::new(core::mem::take(&mut bundle_scratch), item_count);

                    let result = self.components.begin_prepared_transition_batch(
                        run.prepared,
                        run.storage,
                        &transition_inputs,
                        &mut planning,
                    )?;

                    let destination = result.destination;
                    displaced.extend(result.moved_in_source);

                    self.components.write_prepared_transition_table_batch(
                        run.prepared,
                        run.storage,
                        &result.rows,
                        &mut bundle_storage,
                        tick,
                    )?;

                    self.apply_entity_locations_with_groups(
                        result.rows.iter().map(|transitioned| {
                            (
                                transitioned.entity,
                                EntityLocation {
                                    archetype_id: destination,
                                    storage: run.storage,
                                    row: transitioned.destination_row as u32,
                                },
                            )
                        }),
                        &mut location_updates,
                    );

                    self.components.insert_prepared_transition_sparse_batch(
                        run.prepared,
                        run.storage,
                        destination,
                        &result.rows,
                        &mut bundle_storage,
                        tick,
                        &mut planning,
                        &mut sparse_entries,
                    );

                    bundle_scratch = bundle_storage.finish_reuse();

                    continue;
                }

                for entry in entries.by_ref().take(run.len) {
                    let BatchInsertEntry { entity, bundle, .. } = entry;

                    let moved = {
                        let components = &mut self.components;
                        let entity_manager = &mut self.entities;

                        bundle.with_items(components, |components, items| {
                            let location = entity_manager
                                .location_mut(entity)
                                .ok_or(BundleError::EntityInvalid)?;

                            components.apply_bundle_prepared(
                                entity,
                                location,
                                run.prepared,
                                items,
                                tick,
                                &mut planning,
                            )
                        })?
                    };

                    displaced.extend(moved);
                }
            }

            debug_assert!(entries.next().is_none());
            Ok::<_, WorldError>(())
        })();

        // Preserve the current partial-failure contract: all migrations that
        // completed before an error must publish their metadata repairs.
        self.apply_moved_entities_with_groups(displaced, &mut location_updates);
        self.world_planning.apply(planning);

        execution
    }

    /// Removes one row already inserted by a failed spawn and releases its
    /// sparse components.
    fn rollback_reserved_spawn(&mut self, entity: Entity, location: EntityLocation) {
        let mut planning = PlanningChanges::default();

        let (moved, rows) = {
            let archetype = self
                .components
                .get_archetype_mut(location.archetype_id)
                .expect("rollback references a missing archetype");
            let moved = archetype
                .remove_row(location.storage, location.row as usize)
                .expect("rollback references an invalid archetype row");

            (moved, archetype.len())
        };

        debug_assert!(
            moved.is_none(),
            "reserved spawns must be rolled back in reverse insertion order"
        );

        self.apply_moved_entity(moved);
        planning.mark_archetype_rows(location.archetype_id, rows);
        self.components
            .remove_sparse_components(entity, &mut planning);
        self.world_planning.apply(planning);
    }

    /// Releases the entity handles owned by a failed batch operation.
    fn discard_spawn_batch_entities(
        &mut self,
        entities: &[Entity],
        entity_state: BatchSpawnEntityState,
    ) {
        match entity_state {
            BatchSpawnEntityState::Reserved => {
                for &entity in entities {
                    if self.entities.is_reserved(entity) {
                        let discarded = self.entities.discard_reserved(entity);
                        debug_assert!(discarded.is_ok());
                    }
                }
            }

            BatchSpawnEntityState::Unpublished => self.entities.abort_spawn_many(entities),
        }
    }

    /// Removes every row already allocated by a failed spawn and releases all
    /// reservations. Rows are removed in reverse insertion order so contiguous
    /// storage cannot displace another unactivated row in the same batch.
    fn rollback_reserved_spawn_batch(
        &mut self,
        entities: &[Entity],
        activations: Vec<(Entity, EntityLocation)>,
        entity_state: BatchSpawnEntityState,
    ) {
        for (entity, location) in activations.into_iter().rev() {
            self.rollback_reserved_spawn(entity, location);
        }

        self.discard_spawn_batch_entities(entities, entity_state);
    }

    /// Spawns reserved entities with their bundles in contiguous storage.
    pub(crate) fn spawn_reserved_batch<B: Bundle>(
        &mut self,
        entities: &[Entity],
        bundles: Vec<B>,
    ) -> Result<(), WorldError> {
        self.spawn_reserved_batch_with_storage(entities, bundles, ArchetypeStorageKind::Contiguous)
    }

    /// Spawns reserved entities with their bundles in the requested storage.
    pub(crate) fn spawn_reserved_batch_with_storage<B: Bundle>(
        &mut self,
        entities: &[Entity],
        bundles: Vec<B>,
        storage: ArchetypeStorageKind,
    ) -> Result<(), WorldError> {
        self.spawn_batch_with_storage_impl(
            entities,
            bundles,
            storage,
            BatchSpawnEntityState::Reserved,
        )
    }

    /// Spawns allocated but unpublished entities with their bundles in the
    /// requested storage.
    pub(crate) fn spawn_allocated_batch_with_storage<B: Bundle>(
        &mut self,
        entities: &[Entity],
        bundles: Vec<B>,
        storage: ArchetypeStorageKind,
    ) -> Result<(), WorldError> {
        self.spawn_batch_with_storage_impl(
            entities,
            bundles,
            storage,
            BatchSpawnEntityState::Unpublished,
        )
    }

    /// Shared implementation for every batch spawn entry point.
    ///
    /// Dispatches to the table-only fast path, the sparse-only path, or the
    /// generic per-entity path. On failure, already-allocated rows are rolled
    /// back, in reverse insertion order on the sparse and generic paths, and
    /// the entity handles are released or aborted according to `entity_state`.
    fn spawn_batch_with_storage_impl<B: Bundle>(
        &mut self,
        entities: &[Entity],
        bundles: Vec<B>,
        storage: ArchetypeStorageKind,
        entity_state: BatchSpawnEntityState,
    ) -> Result<(), WorldError> {
        if entities.len() != bundles.len() {
            return Err(WorldError::BatchLengthMismatch {
                entities: entities.len(),
                bundles: bundles.len(),
            });
        }

        let mut planning = PlanningChanges::default();
        let tick = self.advance_tick();

        if matches!(entity_state, BatchSpawnEntityState::Reserved) {
            self.entities.ensure_reserved_many(entities)?;
        }

        let prepared = match self.components.prepare_spawn::<B>(&mut planning) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.discard_spawn_batch_entities(entities, entity_state);
                return Err(error.into());
            }
        };

        if let Some(item_count) = self.components.table_only_spawn_item_count(prepared) {
            let mut bundle_storage = BatchBundleStorage::new(bundles, item_count);
            let activations = match self.components.spawn_table_batch(
                entities,
                storage,
                prepared,
                &mut bundle_storage,
                tick,
                &mut planning,
            ) {
                Ok(activations) => activations,
                Err(error) => {
                    self.discard_spawn_batch_entities(entities, entity_state);
                    return Err(error.into());
                }
            };

            bundle_storage.finish();

            match entity_state {
                BatchSpawnEntityState::Reserved => {
                    self.entities.activate_many_prevalidated(&activations);
                }

                BatchSpawnEntityState::Unpublished => {
                    self.entities.commit_spawn_many(&activations);
                }
            }

            self.world_planning.apply(planning);
            return Ok(());
        }

        if let Some(slots) = self.components.sparse_only_spawn_slots(prepared) {
            if let Err(error) = self.components.prepare_sparse_spawn_batch(&slots, entities) {
                self.discard_spawn_batch_entities(entities, entity_state);
                return Err(error.into());
            }

            let mut activations = Vec::with_capacity(entities.len());

            for &entity in entities {
                match self
                    .components
                    .allocate_sparse_spawn_row(entity, storage, prepared, tick)
                {
                    Ok(location) => activations.push((entity, location)),
                    Err(error) => {
                        self.rollback_reserved_spawn_batch(entities, activations, entity_state);
                        return Err(error.into());
                    }
                }
            }

            let mut bundle_storage = BatchBundleStorage::new(bundles, slots.len());
            debug_assert_eq!(bundle_storage.len(), entities.len());
            let mut entries = Vec::with_capacity(entities.len());

            for (slot_index, &(component, item_index)) in slots.iter().enumerate() {
                let previous = slots
                    .get(slot_index + 1)
                    .map_or(SparseMembershipLinks::UNLINKED.previous, |slot| {
                        slot.0.index() as u32
                    });
                let next = slot_index
                    .checked_sub(1)
                    .map_or(SparseMembershipLinks::UNLINKED.next, |index| {
                        slots[index].0.index() as u32
                    });
                let links = SparseMembershipLinks { previous, next };

                entries.clear();

                for row in 0..entities.len() {
                    let item = unsafe { bundle_storage.item(row, item_index) };
                    entries.push(SparseSpawnBatchEntry {
                        entity: entities[row],
                        location: activations[row].1,
                        item,
                        links,
                    });
                }

                unsafe {
                    self.components
                        .insert_sparse_spawn_component_batch(component, &entries, tick);
                }
            }

            let archetype_id = activations
                .first()
                .expect("non-empty sparse batch must allocate an archetype row")
                .1
                .archetype_id;
            let head = slots
                .last()
                .expect("sparse-only spawn layout must contain a component")
                .0;

            if let Err(error) = self.components.finish_sparse_spawn_batch(
                entities,
                head,
                archetype_id,
                &mut planning,
            ) {
                self.rollback_reserved_spawn_batch(entities, activations, entity_state);
                return Err(error.into());
            }

            bundle_storage.finish();

            match entity_state {
                BatchSpawnEntityState::Reserved => {
                    // Command batches come from one allocator reservation and
                    // were validated by `ensure_reserved_many` before storage
                    // mutation, so sorting them again cannot discover new state.
                    self.entities.activate_many_prevalidated(&activations);
                }

                BatchSpawnEntityState::Unpublished => {
                    self.entities.commit_spawn_many(&activations);
                }
            }

            self.world_planning.apply(planning);
            return Ok(());
        }

        let mut activations = Vec::with_capacity(entities.len());

        for (&entity, bundle) in entities.iter().zip(bundles) {
            match bundle.with_items(&mut self.components, |components, items| {
                components.spawn_with_plan(entity, storage, prepared, items, tick, &mut planning)
            }) {
                Ok(location) => activations.push((entity, location)),
                Err(error) => {
                    self.rollback_reserved_spawn_batch(entities, activations, entity_state);
                    return Err(error.into());
                }
            }
        }

        match entity_state {
            BatchSpawnEntityState::Reserved => {
                self.entities.activate_many_prevalidated(&activations);
            }

            BatchSpawnEntityState::Unpublished => {
                self.entities.commit_spawn_many(&activations);
            }
        }

        self.world_planning.apply(planning);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use millarecs_macros::Component;

    use super::*;

    #[derive(Component, Debug, PartialEq, Eq)]
    struct Base(u32);

    #[derive(Component, Debug, PartialEq, Eq)]
    struct Marker;

    #[derive(Component, Debug, PartialEq, Eq)]
    struct Added(u32);

    fn batch_world() -> World {
        let mut world = World::new();
        world.register_table::<Base>().unwrap();
        world.register_table::<Marker>().unwrap();
        world.register_table::<Added>().unwrap();
        world
    }

    #[test]
    fn insert_batch_keeps_pending_rows_valid_after_swap_removal() {
        let mut world = batch_world();
        let entities = (0..5)
            .map(|value| world.spawn((Base(value),)).unwrap())
            .collect::<Vec<_>>();

        world
            .insert_batch(vec![
                (entities[1], (Added(10),)),
                (entities[3], (Added(30),)),
            ])
            .unwrap();

        for (value, entity) in entities.iter().copied().enumerate() {
            assert_eq!(world.get::<Base>(entity), Some(&Base(value as u32)));
        }

        assert_eq!(world.get::<Added>(entities[1]), Some(&Added(10)));
        assert_eq!(world.get::<Added>(entities[3]), Some(&Added(30)));
        assert_eq!(world.get::<Added>(entities[0]), None);
        assert_eq!(world.get::<Added>(entities[2]), None);
        assert_eq!(world.get::<Added>(entities[4]), None);
    }

    #[test]
    fn insert_batch_handles_multiple_sources_and_storage_lanes() {
        let mut world = batch_world();
        let contiguous_base = world.spawn((Base(0),)).unwrap();
        let fragmented_base = world
            .spawn_with_storage(ArchetypeStorageKind::Fragmented, (Base(1),))
            .unwrap();
        let contiguous_marked = world.spawn((Base(2), Marker)).unwrap();
        let fragmented_marked = world
            .spawn_with_storage(ArchetypeStorageKind::Fragmented, (Base(3), Marker))
            .unwrap();

        world
            .insert_batch(vec![
                (fragmented_marked, (Added(3),)),
                (contiguous_base, (Added(0),)),
                (fragmented_base, (Added(1),)),
                (contiguous_marked, (Added(2),)),
            ])
            .unwrap();

        for (value, entity) in [
            contiguous_base,
            fragmented_base,
            contiguous_marked,
            fragmented_marked,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(world.get::<Base>(entity), Some(&Base(value as u32)));
            assert_eq!(world.get::<Added>(entity), Some(&Added(value as u32)));
        }

        assert!(world.get::<Marker>(contiguous_base).is_none());
        assert!(world.get::<Marker>(fragmented_base).is_none());
        assert_eq!(world.get::<Marker>(contiguous_marked), Some(&Marker));
        assert_eq!(world.get::<Marker>(fragmented_marked), Some(&Marker));
        assert_eq!(
            world.entities.location(contiguous_base).unwrap().storage,
            ArchetypeStorageKind::Contiguous
        );
        assert_eq!(
            world.entities.location(fragmented_base).unwrap().storage,
            ArchetypeStorageKind::Fragmented
        );
        assert_eq!(
            world.entities.location(contiguous_marked).unwrap().storage,
            ArchetypeStorageKind::Contiguous
        );
        assert_eq!(
            world.entities.location(fragmented_marked).unwrap().storage,
            ArchetypeStorageKind::Fragmented
        );
    }

    #[test]
    fn insert_batch_reports_duplicates_before_invalid_handles() {
        let mut world = batch_world();

        let result = world.insert_batch(vec![
            (Entity::INVALID, (Added(1),)),
            (Entity::INVALID, (Added(2),)),
        ]);

        assert!(matches!(
            result,
            Err(WorldError::Entity(EntityError::Duplicate(entity)))
                if entity == Entity::INVALID
        ));
    }
}
