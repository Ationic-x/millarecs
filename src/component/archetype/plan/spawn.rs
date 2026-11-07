//! Precompiled writes for placing new entities directly in their archetype.

use crate::{
    change_detection::ChangeTick,
    component::{
        bundle::{BatchBundleStorage, ErasedBundleItem},
        bundle_layout::{BundleLayoutId, BundleSlot},
        ArchetypeId, Bundle, BundleError, ComponentId,
    },
    entity::{ArchetypeStorageKind, Entity, EntityLocation},
};

use super::super::Archetype;

/// Stable identifier for a spawn plan owned by an archetype manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SpawnPlanId(u32);

impl SpawnPlanId {
    /// Creates an id from a spawn-plan vector index.
    #[inline]
    pub(crate) fn new(index: usize) -> Self {
        Self(u32::try_from(index).expect("spawn plan count exceeded u32 capacity"))
    }

    /// Returns the spawn-plan vector index encoded by this id.
    #[inline]
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

/// Validated pair required to execute a cached spawn operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PreparedSpawn {
    /// Bundle layout the plan was compiled for.
    pub(crate) layout: BundleLayoutId,
    /// Cached plan stored under `layout`.
    pub(crate) plan: SpawnPlanId,
}

/// Destination and original bundle position for every table column.
pub(crate) struct SpawnPlan {
    /// Archetype the bundle spawns into.
    pub(crate) destination: ArchetypeId,
    /// Maps each destination column index to the original bundle item position.
    ///
    /// Indexed by destination signature order; every destination column
    /// receives exactly one bundle value, so the slice length equals the
    /// destination column count.
    pub(crate) bundle_items_by_column: Box<[u16]>,
}

/// Compiles a spawn plan from immutable, sorted destination metadata.
///
/// `destination_components` and `table` must describe the same components in
/// the same order; otherwise the prepared layout is rejected as stale.
pub(crate) fn compile_spawn_plan(
    destination: ArchetypeId,
    destination_components: &[ComponentId],
    table: &[BundleSlot],
) -> Result<SpawnPlan, BundleError> {
    if destination_components.len() != table.len() {
        return Err(BundleError::PreparedLayoutMismatch);
    }

    let mut bundle_items_by_column = Vec::with_capacity(table.len());

    for (&component, slot) in destination_components.iter().zip(table) {
        if component != slot.component {
            return Err(BundleError::PreparedLayoutMismatch);
        }

        bundle_items_by_column.push(slot.item_index);
    }

    Ok(SpawnPlan {
        destination,
        bundle_items_by_column: bundle_items_by_column.into_boxed_slice(),
    })
}

/// Executes a fully validated plan. No fallible work remains after row allocation.
pub(crate) fn execute_spawn_plan(
    archetype: &mut Archetype,
    plan: &SpawnPlan,
    storage: ArchetypeStorageKind,
    entity: Entity,
    items: &mut [ErasedBundleItem],
    change_tick: ChangeTick,
) -> usize {
    debug_assert_eq!(archetype.id, plan.destination);

    let data = archetype.data_mut(storage);
    debug_assert_eq!(data.columns().len(), plan.bundle_items_by_column.len());

    let row = data.allocate_row();
    data.write_entity(row, entity);

    for (column_index, &item_index) in plan.bundle_items_by_column.iter().enumerate() {
        let item = &mut items[item_index as usize];

        // SAFETY:
        // - `SpawnPlan` was compiled from the cached `BundleLayout`.
        // - `bundle_items_by_column` maps each destination column to the
        //   corresponding original bundle position.
        // - the unsafe `Bundle` contract guarantees that each item position
        //   contains the concrete component type declared by `BundleInfo`.
        unsafe {
            data.column_mut(column_index)
                .data_mut()
                .push_move_from_ptr_at(row, item.ptr, change_tick);
        }

        item.consumed = true;
    }

    debug_assert!(data
        .columns()
        .iter()
        .all(|column| column.len() == data.len()));

    row
}

/// Executes one homogeneous table spawn component-major.
///
/// Rows and entity handles are allocated first. Each erased destination column
/// is then resolved once and remains active while all bundle values for that
/// column are moved. `BatchBundleStorage` retains ownership of every field
/// until its matching column write succeeds.
pub(crate) fn execute_spawn_plan_batch<B: Bundle>(
    archetype: &mut Archetype,
    plan: &SpawnPlan,
    storage: ArchetypeStorageKind,
    entities: &[Entity],
    bundles: &mut BatchBundleStorage<B>,
    change_tick: ChangeTick,
    activations: &mut Vec<(Entity, EntityLocation)>,
) {
    debug_assert_eq!(archetype.id, plan.destination);
    debug_assert_eq!(entities.len(), bundles.len());

    let data = archetype.data_mut(storage);
    debug_assert_eq!(data.columns().len(), plan.bundle_items_by_column.len());

    activations.clear();
    activations.reserve(entities.len());

    for &entity in entities {
        let row = data.allocate_row();
        data.write_entity(row, entity);
        activations.push((
            entity,
            EntityLocation {
                archetype_id: plan.destination,
                storage,
                row: row as u32,
            },
        ));
    }

    // A complete column pass works well for narrow bundles, but wide bundles
    // are stored entity-major at the source. Keeping one small source block
    // active avoids reloading the full bundle array for every destination
    // column while preserving sequential writes inside each column page.
    /// Target size, in bytes, of one source bundle block before row clamping.
    const SOURCE_BLOCK_BYTES: usize = 16 * 1024;
    /// Lower bound on rows per source block.
    const MIN_BLOCK_ROWS: usize = 32;
    /// Upper bound on rows per source block.
    const MAX_BLOCK_ROWS: usize = 256;

    let block_rows = if plan.bundle_items_by_column.len() < 8 {
        // Narrow bundles fit one full column pass per bundle row, so blocking
        // would only add overhead; the whole batch becomes a single block.
        activations.len().max(1)
    } else {
        SOURCE_BLOCK_BYTES
            .checked_div(core::mem::size_of::<B>().max(1))
            .unwrap_or(MIN_BLOCK_ROWS)
            .clamp(MIN_BLOCK_ROWS, MAX_BLOCK_ROWS)
    };

    for block_start in (0..activations.len()).step_by(block_rows) {
        let block_end = (block_start + block_rows).min(activations.len());

        for (column_index, &item_index) in plan.bundle_items_by_column.iter().enumerate() {
            let column = data.column_mut(column_index).data_mut();

            for (bundle_row, (_, location)) in
                activations[block_start..block_end].iter().enumerate()
            {
                let bundle_row = block_start + bundle_row;
                let item = unsafe { bundles.item(bundle_row, item_index) };

                unsafe {
                    column.push_move_from_ptr_at(location.row as usize, item.ptr, change_tick);
                    item.mark_consumed();
                }
            }
        }
    }

    debug_assert!(data
        .columns()
        .iter()
        .all(|column| column.len() == data.len()));
}
