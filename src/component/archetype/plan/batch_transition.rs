//! Column-major execution for homogeneous structural transition batches.
use crate::{
    change_detection::ChangeTick,
    component::{
        bundle::BatchBundleStorage, ArchetypeError, ArchetypeId, Bundle, BundleError, MovedEntity,
    },
    entity::{metadata::ArchetypeStorageKind, Entity},
};

use super::{
    super::{Archetype, BatchTransitionRow},
    transition::{StructuralTransitionOp, TableTransitionOp},
};

/// Compact source metadata shared by table migration and sparse preparation.
#[derive(Clone, Copy)]
pub(crate) struct BatchTransitionInput {
    /// Entity expected to live at `source_row`.
    pub(crate) entity: Entity,
    /// Live row the entity occupies in the source archetype.
    pub(crate) source_row: usize,
}

/// Outcome of one homogeneous transition batch.
pub(crate) struct BatchTransitionResult {
    /// Archetype the batch moved from.
    pub(crate) source: ArchetypeId,
    /// Archetype the batch moved into.
    pub(crate) destination: ArchetypeId,
    /// One row description per input, in input order.
    pub(crate) rows: Vec<BatchTransitionRow>,
    /// Entities displaced by contiguous backfills, for metadata repair.
    pub(crate) moved_in_source: Vec<MovedEntity>,
}

/// Row access shared by bare row indices and row-plus-entity inputs.
trait BatchRowInputs {
    /// Returns the number of rows in the batch.
    fn len(&self) -> usize;
    /// Returns the source row stored at `index`.
    fn source_row(&self, index: usize) -> usize;
    /// Returns the entity expected at `index`, when the caller knows one.
    fn expected_entity(&self, index: usize) -> Option<Entity>;
}

impl BatchRowInputs for [usize] {
    #[inline]
    fn len(&self) -> usize {
        <[usize]>::len(self)
    }

    #[inline]
    fn source_row(&self, index: usize) -> usize {
        self[index]
    }

    #[inline]
    fn expected_entity(&self, _index: usize) -> Option<Entity> {
        None
    }
}

impl BatchRowInputs for [BatchTransitionInput] {
    #[inline]
    fn len(&self) -> usize {
        <[BatchTransitionInput]>::len(self)
    }

    #[inline]
    fn source_row(&self, index: usize) -> usize {
        self[index].source_row
    }

    #[inline]
    fn expected_entity(&self, index: usize) -> Option<Entity> {
        Some(self[index].entity)
    }
}

/// Shorthand for the signature-mismatch error.
fn signature_mismatch() -> BundleError {
    BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch)
}

/// Builds the out-of-bounds error for a rejected row.
fn row_out_of_bounds(row: usize, len: usize) -> BundleError {
    BundleError::ArchetypeOp(ArchetypeError::RowOutOfBounds { row, len })
}

/// Validates the source sequence and reserves its destination rows.
///
/// Rows must arrive in strictly descending order. Contiguous storage requires
/// that order so every precomputed backfill remains valid while the logical
/// tail shrinks; fragmented storage uses the same ordering contract so callers
/// can share one structural grouping pass.
fn prepare_batch_rows<I: BatchRowInputs + ?Sized>(
    source: &Archetype,
    destination: &mut Archetype,
    storage: ArchetypeStorageKind,
    inputs: &I,
) -> Result<Vec<BatchTransitionRow>, BundleError> {
    if inputs.len() == 0 {
        return Ok(Vec::new());
    }

    let ordered =
        (1..inputs.len()).all(|index| inputs.source_row(index - 1) > inputs.source_row(index));

    if !ordered {
        return Err(signature_mismatch());
    }

    let source_data = source.data(storage);
    // Models the contiguous tail as if each input had already been removed:
    // descending order shrinks `virtual_len` one row per input, so every
    // backfill names the live tail at its own step and never a row that a
    // later input still has to release.
    let mut virtual_len = source_data.len();
    let mut rows = Vec::with_capacity(inputs.len());

    // Validate the complete input before mutating destination storage.
    for index in 0..inputs.len() {
        let source_row = inputs.source_row(index);

        if !source_data.is_live(source_row) {
            return Err(row_out_of_bounds(source_row, source_data.len()));
        }

        let entity = source_data
            .entity(source_row)
            .ok_or_else(|| row_out_of_bounds(source_row, source_data.len()))?;
        if let Some(expected) = inputs.expected_entity(index) {
            debug_assert_eq!(entity, expected);
        }

        let backfill_row = match storage {
            ArchetypeStorageKind::Contiguous => {
                if source_row >= virtual_len {
                    return Err(row_out_of_bounds(source_row, virtual_len));
                }

                let last_row = virtual_len - 1;
                virtual_len -= 1;

                (source_row != last_row).then_some(last_row)
            }

            ArchetypeStorageKind::Fragmented => None,
        };

        rows.push(BatchTransitionRow {
            entity,
            source_row,
            destination_row: usize::MAX,
            backfill_row,
        });
    }

    let destination_data = destination.data_mut(storage);

    for row in &mut rows {
        let destination_row = destination_data.allocate_row();
        destination_data.write_entity(destination_row, row.entity);
        row.destination_row = destination_row;
    }

    Ok(rows)
}

/// Commits lane metadata for every migrated source row.
///
/// `rows` must be finished in the same order the component columns processed
/// them, so each precomputed backfill still targets a live tail row. Returns
/// the entities displaced by contiguous backfills for repair.
fn finish_source_rows(
    source: &mut Archetype,
    storage: ArchetypeStorageKind,
    rows: &[BatchTransitionRow],
) -> Vec<MovedEntity> {
    let source_id = source.id;
    let source_data = source.data_mut(storage);
    let mut moved = Vec::new();

    // This order must match the order used by every component column.
    for movement in rows {
        if let Some(entity) =
            source_data.finish_migrated_row(source_id, movement.source_row, movement.backfill_row)
        {
            moved.push(entity);
        }
    }

    moved
}

/// Moves all existing source columns before bundle values are consumed.
///
/// Replace and Insert operations only reserve their destination slots here.
/// Their bundle values are written later by `write_bundle_batch`.
pub(crate) fn execute_migration_batch(
    source: &mut Archetype,
    destination: &mut Archetype,
    storage: ArchetypeStorageKind,
    inputs: &[BatchTransitionInput],
    operations: &[TableTransitionOp],
) -> Result<BatchTransitionResult, BundleError> {
    let source_id = source.id;
    let destination_id = destination.id;

    let rows = prepare_batch_rows(source, destination, storage, inputs)?;

    {
        let source_data = source.data_mut(storage);
        let destination_data = destination.data_mut(storage);

        for operation in operations {
            match *operation {
                TableTransitionOp::Move {
                    src_column,
                    dst_column,
                } => unsafe {
                    source_data
                        .column_mut(src_column as usize)
                        .data_mut()
                        .move_rows_to_and_backfill(
                            destination_data.column_mut(dst_column as usize).data_mut(),
                            &rows,
                        );
                },

                TableTransitionOp::Replace { src_column, .. } => unsafe {
                    source_data
                        .column_mut(src_column as usize)
                        .data_mut()
                        .drop_rows_and_backfill(&rows);
                },

                TableTransitionOp::Insert { .. } => {
                    // The source has no corresponding column.
                }
            }
        }
    }

    let moved_in_source = finish_source_rows(source, storage, &rows);

    debug_assert!(source
        .data(storage)
        .columns()
        .iter()
        .all(|column| column.len() == source.data(storage).len()));

    Ok(BatchTransitionResult {
        source: source_id,
        destination: destination_id,
        rows,
        moved_in_source,
    })
}

/// Writes bundle-owned values one complete destination column at a time.
///
/// The erased column implementation is resolved once per operation and bundle
/// ownership is tracked by `BatchBundleStorage`, so unwind cleanup only drops
/// fields that have not reached ECS storage.
pub(crate) fn write_bundle_batch<B: Bundle>(
    destination: &mut Archetype,
    storage: ArchetypeStorageKind,
    rows: &[BatchTransitionRow],
    operations: &[TableTransitionOp],
    bundles: &mut BatchBundleStorage<B>,
    tick: ChangeTick,
) {
    debug_assert_eq!(rows.len(), bundles.len());
    let destination_data = destination.data_mut(storage);

    for operation in operations {
        let (dst_column, bundle_item) = match *operation {
            TableTransitionOp::Replace {
                dst_column,
                bundle_item,
                ..
            }
            | TableTransitionOp::Insert {
                dst_column,
                bundle_item,
            } => (dst_column, bundle_item),

            TableTransitionOp::Move { .. } => continue,
        };

        let column = destination_data.column_mut(dst_column as usize).data_mut();

        for (bundle_row, transitioned) in rows.iter().enumerate() {
            let item = unsafe { bundles.item(bundle_row, bundle_item) };

            unsafe {
                column.push_move_from_ptr_at(transitioned.destination_row, item.ptr, tick);
                item.mark_consumed();
            }
        }
    }
}

/// Moves a structural batch of source rows in column-major order.
///
/// Shared columns are moved and removed columns dropped for the whole batch
/// before source lane metadata is committed. Inputs must satisfy the strictly
/// descending order required by `prepare_batch_rows`; no bundle values are
/// consumed.
pub(crate) fn execute_structural_migration_batch(
    source: &mut Archetype,
    destination: &mut Archetype,
    storage: ArchetypeStorageKind,
    source_rows: &[usize],
    operations: &[StructuralTransitionOp],
) -> Result<BatchTransitionResult, BundleError> {
    let source_id = source.id;
    let destination_id = destination.id;

    let rows = prepare_batch_rows(source, destination, storage, source_rows)?;

    {
        let source_data = source.data_mut(storage);
        let destination_data = destination.data_mut(storage);

        for operation in operations {
            match *operation {
                StructuralTransitionOp::Move {
                    src_column,
                    dst_column,
                } => unsafe {
                    source_data
                        .column_mut(src_column as usize)
                        .data_mut()
                        .move_rows_to_and_backfill(
                            destination_data.column_mut(dst_column as usize).data_mut(),
                            &rows,
                        );
                },

                StructuralTransitionOp::Drop { src_column } => unsafe {
                    source_data
                        .column_mut(src_column as usize)
                        .data_mut()
                        .drop_rows_and_backfill(&rows);
                },
            }
        }
    }

    let moved_in_source = finish_source_rows(source, storage, &rows);

    debug_assert!(source
        .data(storage)
        .columns()
        .iter()
        .all(|column| column.len() == source.data(storage).len()));

    debug_assert!(destination
        .data(storage)
        .columns()
        .iter()
        .all(|column| column.len() == destination.data(storage).len()));

    Ok(BatchTransitionResult {
        source: source_id,
        destination: destination_id,
        rows,
        moved_in_source,
    })
}
