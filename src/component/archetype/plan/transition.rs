//! Precompiled table operations for applying bundles to existing entities.

use crate::{
    change_detection::ChangeTick,
    component::{
        bundle_layout::{BundleLayoutId, BundleSlot},
        ArchetypeError, ArchetypeId, BundleError, ComponentId, ErasedBundleItem, MovedEntity,
    },
    entity::metadata::ArchetypeStorageKind,
    utils::flat_map::FlatKey,
};

use super::super::Archetype;

/// Stable identifier for a transition plan owned by an archetype manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TransitionPlanId(u32);

impl TransitionPlanId {
    /// Creates an id from a transition-plan vector index.
    #[inline]
    pub(crate) fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("transition plan count exceeded u32 capacity"))
    }

    /// Returns the transition-plan vector index encoded by this id.
    #[inline]
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Stable identifier for a cached transition that only removes table components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StructuralPlanId(u32);

impl StructuralPlanId {
    /// Creates an id from a structural-plan vector index.
    #[inline]
    pub(crate) fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("structural plan count exceeded u32 capacity"))
    }

    /// Returns the structural-plan vector index encoded by this id.
    #[inline]
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Operation-specific key for an outgoing structural edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum StructuralTransitionKey {
    /// Removes a single component from the source signature.
    RemoveComponent(ComponentId),
    /// Removes every component of a cached bundle layout.
    RemoveBundle(BundleLayoutId),
    /// Keeps only the components of a cached bundle layout.
    RetainBundle(BundleLayoutId),
}

impl FlatKey for StructuralTransitionKey {
    /// Folds the key into a collision-free `u64`.
    ///
    /// Component and layout indices both occupy the low 32 bits, so the bundle
    /// variants xor in distinct high discriminants, `1 << 32` and `2 << 32`,
    /// to keep the three variants in disjoint numeric ranges.
    #[inline]
    fn fold(self) -> u64 {
        match self {
            Self::RemoveComponent(component) => component.index() as u64,
            Self::RemoveBundle(layout) => layout.index() as u64 ^ (1 << 32),
            Self::RetainBundle(layout) => layout.index() as u64 ^ (2 << 32),
        }
    }
}

/// Validated pair required to execute a cached transition.
#[derive(Clone, Copy)]
pub(crate) struct PreparedTransitionPlan {
    /// Bundle layout the cached edge was compiled for.
    pub(crate) layout: BundleLayoutId,
    /// Archetype the transition starts from.
    pub(crate) source: ArchetypeId,
    /// Cached transition for `(source, layout)`, or `None` when the bundle has
    /// no table components.
    pub(crate) transition: Option<TransitionPlanId>,
}

/// Compiled table work for one source-archetype and bundle-layout pair.
pub(crate) enum TransitionPlan {
    /// Every table component already exists in the source archetype.
    InPlace {
        source: ArchetypeId,
        writes: Box<[InPlaceWrite]>,
    },
    /// Applying the bundle adds table components and changes archetype.
    Migrate {
        source: ArchetypeId,
        destination: ArchetypeId,
        operations: Box<[TableTransitionOp]>,
    },
}

/// One bundle value written into an existing source column.
pub(crate) struct InPlaceWrite {
    /// Source column index replaced by the write.
    pub(crate) column: u16,
    /// Original bundle item position providing the value.
    pub(crate) bundle_item: u16,
}

/// Per-column operation compiled from one source and destination signature pair.
pub(crate) enum TableTransitionOp {
    /// Copies an existing source column into the destination archetype.
    Move {
        src_column: u16,
        dst_column: u16,
    },
    /// Drops the source value and writes the bundle value into the destination column.
    Replace {
        src_column: u16,
        dst_column: u16,
        bundle_item: u16,
    },
    /// Writes the bundle value into a column the source signature lacks.
    Insert {
        dst_column: u16,
        bundle_item: u16,
    },
}

/// Compiled table work for removal and retention operations.
pub(crate) enum TableStructuralPlan {
    /// The requested operation does not alter the table signature.
    Unchanged { source: ArchetypeId },
    /// The entity moves while shared columns are moved and removed columns are dropped.
    Migrate {
        source: ArchetypeId,
        destination: ArchetypeId,
        /// Number of shared columns copied into the destination archetype.
        moved_columns: usize,
        operations: Box<[StructuralTransitionOp]>,
    },
}

/// Per-column operation compiled from a structural remove or retain transition.
pub(crate) enum StructuralTransitionOp {
    /// Copies a shared source column into the destination archetype.
    Move { src_column: u16, dst_column: u16 },
    /// Drops a source column absent from the destination signature.
    Drop { src_column: u16 },
}

/// Result needed to repair entity metadata after applying table components.
pub(crate) struct TransitionResult {
    /// Archetype holding the entity after the transition.
    pub(crate) destination: ArchetypeId,
    /// Row the entity occupies in `destination`.
    pub(crate) row: usize,
    /// Entity displaced by a contiguous backfill in the source lane, if any.
    pub(crate) moved_in_source: Option<MovedEntity>,
}

impl TransitionResult {
    /// Builds a result for a transition that kept the entity at `row` in `source`.
    #[inline]
    pub(crate) fn unchanged(source: ArchetypeId, row: usize) -> Self {
        Self {
            destination: source,
            row,
            moved_in_source: None,
        }
    }
}

/// Compiles an insertion plan from immutable, sorted metadata.
pub(crate) fn compile_transition_plan(
    source: ArchetypeId,
    destination: ArchetypeId,
    source_components: &[ComponentId],
    destination_components: &[ComponentId],
    table: &[BundleSlot],
) -> Result<TransitionPlan, BundleError> {
    debug_assert!(!table.is_empty());

    if source == destination {
        let mut writes = Vec::with_capacity(table.len());

        for slot in table {
            let column = source_components
                .binary_search(&slot.component)
                .map_err(|_| signature_mismatch())?;
            writes.push(InPlaceWrite {
                column: column_index(column)?,
                bundle_item: slot.item_index,
            });
        }

        return Ok(TransitionPlan::InPlace {
            source,
            writes: writes.into_boxed_slice(),
        });
    }

    let mut operations = Vec::with_capacity(destination_components.len());
    let mut src_index = 0usize;
    let mut bundle_index = 0usize;

    for (dst_index, &component) in destination_components.iter().enumerate() {
        let in_source = source_components.get(src_index).copied() == Some(component);
        let in_bundle = table.get(bundle_index).map(|slot| slot.component) == Some(component);

        match (in_source, in_bundle) {
            (true, true) => {
                operations.push(TableTransitionOp::Replace {
                    src_column: column_index(src_index)?,
                    dst_column: column_index(dst_index)?,
                    bundle_item: table[bundle_index].item_index,
                });
                src_index += 1;
                bundle_index += 1;
            }
            (true, false) => {
                operations.push(TableTransitionOp::Move {
                    src_column: column_index(src_index)?,
                    dst_column: column_index(dst_index)?,
                });
                src_index += 1;
            }
            (false, true) => {
                operations.push(TableTransitionOp::Insert {
                    dst_column: column_index(dst_index)?,
                    bundle_item: table[bundle_index].item_index,
                });
                bundle_index += 1;
            }
            (false, false) => return Err(signature_mismatch()),
        }
    }

    if src_index != source_components.len() || bundle_index != table.len() {
        return Err(signature_mismatch());
    }

    Ok(TransitionPlan::Migrate {
        source,
        destination,
        operations: operations.into_boxed_slice(),
    })
}

/// Compiles a migration whose destination signature is a subset of the source.
pub(crate) fn compile_structural_plan(
    source: ArchetypeId,
    destination: ArchetypeId,
    source_components: &[ComponentId],
    destination_components: &[ComponentId],
) -> Result<TableStructuralPlan, BundleError> {
    if source == destination {
        if source_components != destination_components {
            return Err(signature_mismatch());
        }
        return Ok(TableStructuralPlan::Unchanged { source });
    }

    let mut operations = Vec::with_capacity(source_components.len());
    let mut src_index = 0usize;
    let mut dst_index = 0usize;

    while src_index < source_components.len() {
        let source_component = source_components[src_index];

        match destination_components.get(dst_index).copied() {
            Some(destination_component) if source_component == destination_component => {
                operations.push(StructuralTransitionOp::Move {
                    src_column: column_index(src_index)?,
                    dst_column: column_index(dst_index)?,
                });
                src_index += 1;
                dst_index += 1;
            }
            Some(destination_component) if source_component > destination_component => {
                return Err(signature_mismatch());
            }
            _ => {
                operations.push(StructuralTransitionOp::Drop {
                    src_column: column_index(src_index)?,
                });
                src_index += 1;
            }
        }
    }

    if dst_index != destination_components.len() {
        return Err(signature_mismatch());
    }

    Ok(TableStructuralPlan::Migrate {
        source,
        destination,
        moved_columns: destination_components.len(),
        operations: operations.into_boxed_slice(),
    })
}

/// Applies compiled in-place writes to one live source row.
///
/// Fails when `row` is not live in the selected lane.
pub(crate) fn execute_in_place(
    archetype: &mut Archetype,
    storage: ArchetypeStorageKind,
    row: usize,
    writes: &[InPlaceWrite],
    items: &mut [ErasedBundleItem],
    change_tick: ChangeTick,
) -> Result<TransitionResult, BundleError> {
    validate_live_row(archetype, storage, row)?;
    let source = archetype.id;
    let data = archetype.data_mut(storage);

    for write in writes {
        let item = &mut items[write.bundle_item as usize];
        // Consumed before the destructive write: if the replaced value's
        // `Drop` panics, the bundle guard must not drop the already-moved
        // bits a second time.
        item.consumed = true;
        unsafe {
            data.column_mut(write.column as usize)
                .data_mut()
                .write_move_from_ptr(row, item.ptr, change_tick);
        }
    }

    Ok(TransitionResult::unchanged(source, row))
}

/// Moves one live row to the destination archetype and consumes bundle items.
///
/// Fails when `source_row` is not live in the selected lane. The destination
/// row is allocated and registered first, then the compiled per-column
/// operations run, and only afterwards does `finish_migrated_row` commit the
/// source lane metadata. Returns the new location plus any entity displaced by
/// a contiguous backfill.
pub(crate) fn execute_migration(
    source: &mut Archetype,
    destination: &mut Archetype,
    storage: ArchetypeStorageKind,
    source_row: usize,
    operations: &[TableTransitionOp],
    items: &mut [ErasedBundleItem],
    change_tick: ChangeTick,
) -> Result<TransitionResult, BundleError> {
    validate_live_row(source, storage, source_row)?;

    let entity = source
        .data(storage)
        .entity(source_row)
        .ok_or_else(|| row_out_of_bounds(source, storage, source_row))?;
    let source_id = source.id;
    let destination_id = destination.id;

    let destination_row = {
        let data = destination.data_mut(storage);
        let row = data.allocate_row();
        data.write_entity(row, entity);
        row
    };

    let source_data = source.data_mut(storage);
    let destination_data = destination.data_mut(storage);
    let backfill_row = source_data.migration_backfill_row(source_row);

    for operation in operations {
        match *operation {
            TableTransitionOp::Move {
                src_column,
                dst_column,
            } => unsafe {
                source_data
                    .column_mut(src_column as usize)
                    .data_mut()
                    .move_row_to_and_backfill(
                        source_row,
                        destination_data.column_mut(dst_column as usize).data_mut(),
                        destination_row,
                        backfill_row,
                    );
            },
            TableTransitionOp::Replace {
                src_column,
                dst_column,
                bundle_item,
            } => {
                let item = &mut items[bundle_item as usize];
                unsafe {
                    source_data
                        .column_mut(src_column as usize)
                        .data_mut()
                        .drop_row_and_backfill(source_row, backfill_row);
                    destination_data
                        .column_mut(dst_column as usize)
                        .data_mut()
                        .push_move_from_ptr_at(destination_row, item.ptr, change_tick);
                }
                item.consumed = true;
            }
            TableTransitionOp::Insert {
                dst_column,
                bundle_item,
            } => {
                let item = &mut items[bundle_item as usize];
                unsafe {
                    destination_data
                        .column_mut(dst_column as usize)
                        .data_mut()
                        .push_move_from_ptr_at(destination_row, item.ptr, change_tick);
                }
                item.consumed = true;
            }
        }
    }

    let moved_in_source = source_data.finish_migrated_row(source_id, source_row, backfill_row);

    debug_assert!(source_data
        .columns()
        .iter()
        .all(|column| column.len() == source_data.len()));
    debug_assert!(destination_data
        .columns()
        .iter()
        .all(|column| column.len() == destination_data.len()));

    Ok(TransitionResult {
        destination: destination_id,
        row: destination_row,
        moved_in_source,
    })
}

/// Moves one live row to the destination archetype without bundle values.
///
/// Behaves like `execute_migration` except every operation is bundle-free:
/// shared columns are moved into the destination and removed columns are
/// dropped, then source lane metadata is committed.
pub(crate) fn execute_structural_migration(
    source: &mut Archetype,
    destination: &mut Archetype,
    storage: ArchetypeStorageKind,
    source_row: usize,
    operations: &[StructuralTransitionOp],
) -> Result<TransitionResult, BundleError> {
    validate_live_row(source, storage, source_row)?;

    let entity = source
        .data(storage)
        .entity(source_row)
        .ok_or_else(|| row_out_of_bounds(source, storage, source_row))?;
    let source_id = source.id;
    let destination_id = destination.id;

    let destination_row = {
        let data = destination.data_mut(storage);
        let row = data.allocate_row();
        data.write_entity(row, entity);
        row
    };

    let source_data = source.data_mut(storage);
    let destination_data = destination.data_mut(storage);
    let backfill_row = source_data.migration_backfill_row(source_row);

    for operation in operations {
        match *operation {
            StructuralTransitionOp::Move {
                src_column,
                dst_column,
            } => unsafe {
                source_data
                    .column_mut(src_column as usize)
                    .data_mut()
                    .move_row_to_and_backfill(
                        source_row,
                        destination_data.column_mut(dst_column as usize).data_mut(),
                        destination_row,
                        backfill_row,
                    );
            },
            StructuralTransitionOp::Drop { src_column } => unsafe {
                source_data
                    .column_mut(src_column as usize)
                    .data_mut()
                    .drop_row_and_backfill(source_row, backfill_row);
            },
        }
    }

    let moved_in_source = source_data.finish_migrated_row(source_id, source_row, backfill_row);

    debug_assert!(source_data
        .columns()
        .iter()
        .all(|column| column.len() == source_data.len()));
    debug_assert!(destination_data
        .columns()
        .iter()
        .all(|column| column.len() == destination_data.len()));

    Ok(TransitionResult {
        destination: destination_id,
        row: destination_row,
        moved_in_source,
    })
}

/// Rejects a row that is not currently live in the lane.
fn validate_live_row(
    archetype: &Archetype,
    storage: ArchetypeStorageKind,
    row: usize,
) -> Result<(), BundleError> {
    if archetype.is_row_live(storage, row) {
        Ok(())
    } else {
        Err(row_out_of_bounds(archetype, storage, row))
    }
}

/// Builds the out-of-bounds error with the lane's live length.
fn row_out_of_bounds(
    archetype: &Archetype,
    storage: ArchetypeStorageKind,
    row: usize,
) -> BundleError {
    BundleError::ArchetypeOp(ArchetypeError::RowOutOfBounds {
        row,
        len: archetype.data(storage).len(),
    })
}

/// Narrows a signature column index to the u16 used by compiled plans.
fn column_index(index: usize) -> Result<u16, BundleError> {
    u16::try_from(index).map_err(|_| BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch))
}

/// Shorthand for the signature-mismatch error.
fn signature_mismatch() -> BundleError {
    BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch)
}
