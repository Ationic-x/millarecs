//! Component registration and storage orchestration.
//!
//! `ComponentManager` is the bridge between high-level world operations and the
//! concrete storage backends. It assigns component ids, creates archetypes for
//! table components, owns sparse sets, and applies bundles to entity locations.

/// Archetype tables and type-erased column storage.
#[path = "archetype/mod.rs"]
pub(crate) mod archetype;
/// Bundle erasure and cleanup helpers.
pub(crate) mod bundle;
/// Cached, world-local descriptions of bundle item layouts.
mod bundle_layout;
/// Sparse-set storage for non-table components.
pub(crate) mod sparse_set;

use crate::{
    change_detection::ChangeTick,
    component::{
        archetype::{
            plan::{
                batch_transition::{BatchTransitionInput, BatchTransitionResult},
                spawn::{PreparedSpawn, SpawnPlanId},
                transition::{
                    PreparedTransitionPlan, StructuralPlanId, StructuralTransitionKey,
                    TransitionPlanId,
                },
            },
            ArchetypeColumn, ArchetypeManager, BatchTransitionRow, Column, ColumnFactory,
        },
        bundle::{BatchBundleStorage, ErasedBundleItem},
        bundle_layout::{BundleLayout, BundleSlot, CachedBundleLayouts},
    },
    entity::{
        metadata::{ArchetypeStorageKind, EntityLocation},
        Entity,
    },
    planning::PlanningChanges,
    utils::flat_map::FlatMap,
};
use std::any::TypeId;

/// Minimum homogeneous run length before a table transition is executed
/// column-major.
///
/// Column-major migration only amortizes its preparation cost on sufficiently
/// large homogeneous runs. The width benchmark should be used to retune this
/// crossover when the transition executor changes.
const MIN_COLUMN_MAJOR_TRANSITION_ROWS: usize = 256;
/// Minimum run length for a column-major mixed batch whose table migration
/// writes at most three destination columns.
///
/// Mixed batches pay for sparse preflight, ownership tracking, and a second
/// component-major pass. Narrow table migrations need a larger run before
/// those fixed costs beat the compact entity-major path.
const MIN_NARROW_MIXED_TRANSITION_ROWS: usize = 32 * 1024;
/// Minimum run length for a column-major mixed batch whose table migration
/// writes four or five destination columns.
///
/// Wider migrations amortize the fixed mixed-batch costs over more copied
/// data, so they cross over sooner than narrow ones yet still later than a
/// pure table batch.
const MIN_MEDIUM_MIXED_TRANSITION_ROWS: usize = 8 * 1024;

/// Minimum number of copied columns before a structural removal run is
/// executed column-major.
///
/// Structural removal from a two-column table leaves only one column to copy.
/// The width benchmark shows that sorting and transposing that work costs more
/// than the recovered locality. Three copied columns are the first measured
/// width where column-major removal wins consistently in both storage lanes.
const MIN_COLUMN_MAJOR_MOVED_COLUMNS: usize = 3;

pub use archetype::{
    Archetype, ArchetypeError, ArchetypeId, ArchetypeManagerError, ArchetypeSignature,
    ArchetypeStorageStats, ColumnFactoryError, CompactionPolicy, FragmentedCompactionReport,
};
pub use bundle::{Bundle, BundleError, BundleInfo};
pub(crate) use bundle_layout::BundleLayoutId;
pub(crate) use sparse_set::{
    ComponentSparseSet, SparseDriverBucket, SparseDriverLane, SparseDriverRow,
    SparseInsertBatchEntry, SparseLaneLocality, SparseLocationUpdateGroups, SparseMembershipLinks,
    SparseRemovalGroups, SparseSet, SparseSetManager, SparseSpawnBatchEntry,
};

/// Marker trait implemented by types that can be stored as ECS components.
///
/// Components are moved into storage and do not need to implement `Clone` or
/// `Default`. Thread-safety bounds allow disjoint component access to be
/// executed by worker threads.
pub trait Component: 'static + Send + Sync {}

/// Dense identifier assigned to a registered component type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ComponentId(u32);

impl ComponentId {
    /// Returns the dense index used by component storage managers.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Returns the id for a dense `index`, asserting the `u32` range is not exhausted.
    #[inline]
    pub(crate) const fn from_index(index: usize) -> Self {
        assert!(
            index < u32::MAX as usize,
            "component id exhausted its valid range"
        );
        Self(index as u32)
    }
}

/// Storage backend selected for a registered component.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum StorageKind {
    /// Component is stored inside archetype columns.
    Table,
    /// Component is stored separately in a sparse set.
    Sparse,
}

/// Entity that changed rows because another row was removed or migrated.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct MovedEntity {
    /// Entity whose `EntityLocation` must be updated.
    pub(crate) entity: Entity,
    /// New location of the moved entity.
    pub(crate) location: EntityLocation,
}

/// Errors raised while registering component types.
#[derive(Debug)]
pub enum ComponentManagerError {
    /// The same Rust type was registered more than once.
    AlreadyRegistered(TypeId),
}

/// Owns component ids and the storage backends associated with them.
pub struct ComponentManager {
    /// Registered component types; each id equals the type's insertion index.
    components: FlatMap<TypeId, ComponentId>,
    sparse_set_manager: SparseSetManager,
    column_factory: ColumnFactory,
    archetype_manager: ArchetypeManager,
    bundle_layouts: CachedBundleLayouts,
}

impl Default for ComponentManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentManager {
    /// Creates an empty manager without registered components or archetypes.
    pub fn new() -> Self {
        Self {
            components: FlatMap::new(),
            sparse_set_manager: SparseSetManager::new(),
            column_factory: ColumnFactory::new(),
            archetype_manager: ArchetypeManager::new(),
            bundle_layouts: CachedBundleLayouts::new(),
        }
    }

    /// Returns the amount of archetypes currently stored.
    #[inline]
    pub fn archetype_count(&self) -> usize {
        self.archetype_manager.len()
    }

    #[cfg(test)]
    pub(crate) fn transition_plan_count(&self) -> usize {
        self.archetype_manager.transition_plan_count()
    }

    #[cfg(test)]
    pub(crate) fn spawn_plan_count(&self) -> usize {
        self.archetype_manager.spawn_plan_count()
    }

    /// Returns raw storage pointers used by query internals.
    #[inline]
    pub(crate) fn storage_ptrs(&self) -> (*const ArchetypeManager, *const SparseSetManager) {
        (&self.archetype_manager, &self.sparse_set_manager)
    }

    /// Returns the archetype manager that owns this world's archetype tables.
    #[inline]
    pub(crate) fn archetypes(&self) -> &ArchetypeManager {
        &self.archetype_manager
    }

    /// Returns the id assigned to component type `T`.
    pub fn id_of<C: Component>(&self) -> Option<ComponentId> {
        self.components.get_copy(TypeId::of::<C>())
    }

    /// Returns the id assigned to a runtime `TypeId`.
    pub fn id_by_type_id(&self, type_id: &TypeId) -> Option<ComponentId> {
        self.components.get_copy(*type_id)
    }

    /// Reserves a new id for a component type.
    fn register<C: Component>(&mut self) -> Result<ComponentId, ComponentManagerError> {
        let type_id = TypeId::of::<C>();

        // Component ids are unique per Rust type, regardless of storage backend.
        if self.id_of::<C>().is_some() {
            return Err(ComponentManagerError::AlreadyRegistered(type_id));
        }

        let id = ComponentId::from_index(self.components.len());
        self.components.insert(type_id, id);
        Ok(id)
    }

    /// Registers `T` as a table-backed component.
    pub(crate) fn register_column<C: Component>(
        &mut self,
    ) -> Result<ComponentId, ComponentManagerError> {
        let id = self.register::<C>()?;
        let index = id.0 as usize;
        self.column_factory.register::<C>(index);
        Ok(id)
    }

    /// Creates an erased table column for a registered component id.
    pub(crate) fn make_column(
        &self,
        component_id: ComponentId,
    ) -> Result<Box<dyn Column>, ColumnFactoryError> {
        let index = component_id.0 as usize;
        if let Some(column) = self.column_factory.make(index) {
            Ok(column)
        } else {
            Err(ColumnFactoryError::NotRegistered(component_id))
        }
    }

    /// Returns an existing archetype for `signature` or creates a new one.
    pub(crate) fn ensure_archetype(
        &mut self,
        signature: ArchetypeSignature,
        planning: &mut PlanningChanges,
    ) -> Result<ArchetypeId, ArchetypeManagerError> {
        if let Some(id) = self.archetype_manager.get_id(&signature) {
            return Ok(id);
        }

        // Column order must match the sorted signature order.
        let mut contiguous_columns: Vec<ArchetypeColumn> =
            Vec::with_capacity(signature.components().len());
        let mut fragmented_columns: Vec<ArchetypeColumn> =
            Vec::with_capacity(signature.components().len());

        for &ci in signature.components() {
            contiguous_columns.push(ArchetypeColumn::new(ci, self.make_column(ci)?));
            fragmented_columns.push(ArchetypeColumn::new(ci, self.make_column(ci)?));
        }

        let id = self.archetype_manager.create_archetype(
            contiguous_columns,
            fragmented_columns,
            signature,
        )?;
        planning.mark_archetype_created(id, 0);
        Ok(id)
    }

    /// Borrows an archetype by id.
    #[inline]
    pub fn get_archetype(&self, id: ArchetypeId) -> Option<&Archetype> {
        self.archetype_manager.get_archetype(id)
    }

    /// Borrows an archetype mutably by id.
    #[inline]
    pub(crate) fn get_archetype_mut(&mut self, id: ArchetypeId) -> Option<&mut Archetype> {
        self.archetype_manager.get_archetype_mut(id)
    }

    /// Compacts fragmented lanes of every archetype under `policy`.
    ///
    /// Returns the entities displaced by compaction for location repair.
    pub(crate) fn compact_fragmented_with_policy(
        &mut self,
        policy: CompactionPolicy,
    ) -> (Vec<MovedEntity>, FragmentedCompactionReport) {
        let (moved, report) = self
            .archetype_manager
            .compact_fragmented_with_policy(policy);

        (moved, report)
    }

    /// Registers `T` as a sparse-set-backed component.
    pub(crate) fn register_sparse_set<C: Component>(
        &mut self,
        planning: &mut PlanningChanges,
    ) -> Result<ComponentId, ComponentManagerError> {
        let id = self.register::<C>()?;
        let index = id.0 as usize;
        self.sparse_set_manager.register::<C>(index);
        planning.mark_sparse_set_registered();
        Ok(id)
    }

    /// Typed borrow of a sparse set behind a resolved id.
    #[inline]
    pub(crate) fn sparse_set_by_id<C: Component>(
        &self,
        id: ComponentId,
    ) -> Option<&ComponentSparseSet<C>> {
        self.sparse_set_manager.get_sparse_set::<C>(id.index())
    }

    /// Mutable typed borrow of a sparse set behind a resolved id.
    #[inline]
    pub(crate) fn sparse_set_mut_by_id<C: Component>(
        &mut self,
        id: ComponentId,
    ) -> Option<&mut ComponentSparseSet<C>> {
        self.sparse_set_manager.get_sparse_set_mut::<C>(id.index())
    }

    /// Live entry count of the sparse set for `component`.
    pub(crate) fn sparse_len(&self, component: ComponentId) -> Option<usize> {
        self.sparse_set_manager
            .get_sparse_set_dyn(component.index())
            .map(|set| set.len())
    }

    /// Returns the query-driver buckets of the component's sparse set.
    pub(crate) fn sparse_driver_buckets(
        &self,
        component: ComponentId,
    ) -> Option<&[SparseDriverBucket]> {
        self.sparse_set_manager.driver_buckets(component.index())
    }

    /// Returns the current driver revision of the component's sparse set, if registered.
    #[inline]
    pub(crate) fn sparse_driver_revision(&self, component: ComponentId) -> Option<u64> {
        self.sparse_set_manager.driver_revision(component.index())
    }

    /// Propagates a table migration to every sparse set owned by `entity`.
    pub(crate) fn update_sparse_entity_location(
        &mut self,
        entity: Entity,
        location: EntityLocation,
    ) {
        self.sparse_set_manager
            .update_entity_location(entity, location);
    }

    /// Captures sparse location repairs for component-major execution.
    pub(crate) fn prepare_sparse_entity_location_update(
        &self,
        entity: Entity,
        location: EntityLocation,
        groups: &mut SparseLocationUpdateGroups,
    ) {
        self.sparse_set_manager
            .prepare_entity_location_update(entity, location, groups);
    }

    /// Applies all prepared sparse location repairs component-major.
    pub(crate) fn execute_sparse_entity_location_updates(
        &mut self,
        groups: &mut SparseLocationUpdateGroups,
    ) {
        self.sparse_set_manager
            .execute_entity_location_updates(groups);
    }

    /// Returns the storage backend registered for a component id.
    pub fn storage_of(&self, id: ComponentId) -> Option<StorageKind> {
        let index = id.0 as usize;
        if self.column_factory.is_registered(index) {
            Some(StorageKind::Table)
        } else if self.sparse_set_manager.is_registered(index) {
            Some(StorageKind::Sparse)
        } else {
            None
        }
    }

    /// Removes an entity from every sparse-set component storage.
    pub(crate) fn remove_sparse_components(
        &mut self,
        entity: Entity,
        planning: &mut PlanningChanges,
    ) {
        self.sparse_set_manager
            .remove_entity_from_all(entity, |_component_id| planning.mark_query_work_changed());
    }

    /// Removes one resolved sparse component from several live entities.
    ///
    /// The sparse manager prepares the complete set mutation before applying
    /// it, so type resolution, driver repair and query invalidation are paid at
    /// most once for the whole batch.
    pub(crate) fn remove_sparse_component_batch(
        &mut self,
        component: ComponentId,
        entities: &[Entity],
        planning: &mut PlanningChanges,
    ) -> usize {
        debug_assert_eq!(self.storage_of(component), Some(StorageKind::Sparse));

        let removed = self
            .sparse_set_manager
            .remove_component_batch(component, entities);

        if removed != 0 {
            planning.mark_query_work_changed();
        }

        removed
    }

    /// Removes one sparse component through already resolved metadata.
    #[inline]
    pub(crate) fn remove_sparse_component_prepared(
        &mut self,
        component: ComponentId,
        entity: Entity,
        planning: &mut PlanningChanges,
    ) -> bool {
        debug_assert_eq!(self.storage_of(component), Some(StorageKind::Sparse));

        let removed = self.sparse_set_manager.remove_component(component, entity);

        if removed {
            planning.mark_query_work_changed();
        }

        removed
    }

    /// Whether `entity` owns at least one sparse component.
    #[inline]
    pub(crate) fn entity_has_sparse_components(&self, entity: Entity) -> bool {
        self.sparse_set_manager.entity_has_components(entity)
    }

    /// Groups the sparse memberships owned by a prevalidated destruction batch.
    pub(crate) fn prepare_sparse_despawn_batch(
        &self,
        entities: &[Entity],
        groups: &mut SparseRemovalGroups,
    ) {
        self.sparse_set_manager
            .prepare_despawn_batch(entities, groups);
    }

    /// Removes sparse memberships component by component.
    pub(crate) fn execute_sparse_despawn_batch(
        &mut self,
        entities: &[Entity],
        groups: &mut SparseRemovalGroups,
    ) {
        self.sparse_set_manager
            .execute_despawn_batch(entities, groups);
    }

    /// Resolves or compiles the layout for bundle type `B`.
    ///
    /// Layouts cache the table and sparse slots of the bundle, sorted by
    /// component id, and are reused for the lifetime of the world.
    pub(crate) fn ensure_bundle_layout<B: BundleInfo>(
        &mut self,
    ) -> Result<BundleLayoutId, BundleError> {
        let bundle_type = TypeId::of::<B>();

        if let Some(id) = self.bundle_layouts.id_of(bundle_type) {
            return Ok(id);
        }

        let item_ids = B::component_ids(self)?;
        u16::try_from(item_ids.len()).map_err(|_| BundleError::BundleTooLarge)?;

        let mut table = Vec::new();
        let mut sparse = Vec::new();

        for (item_index, &component) in item_ids.iter().enumerate() {
            let item_index = item_index as u16;
            let slot = BundleSlot {
                component,
                item_index,
            };

            match self
                .storage_of(component)
                .ok_or(BundleError::StorageUnknown(component))?
            {
                StorageKind::Table => table.push(slot),
                StorageKind::Sparse => sparse.push(slot),
            }
        }

        table.sort_unstable_by_key(|slot| slot.component);
        sparse.sort_unstable_by_key(|slot| slot.component);

        let duplicate = table
            .windows(2)
            .chain(sparse.windows(2))
            .find_map(|pair| (pair[0].component == pair[1].component).then_some(pair[0].component));

        if let Some(component) = duplicate {
            return Err(BundleError::DuplicateComponent(component));
        }

        Ok(self.bundle_layouts.insert(
            bundle_type,
            BundleLayout {
                table: table.into_boxed_slice(),
                sparse: sparse.into_boxed_slice(),
            },
        ))
    }

    
    /// Creates the destination archetype for `destination_signature` and caches
    /// the structural plan for `(source, key)`.
    fn compile_structural_plan(
        &mut self,
        source: ArchetypeId,
        key: StructuralTransitionKey,
        destination_signature: ArchetypeSignature,
        planning: &mut PlanningChanges,
    ) -> Result<StructuralPlanId, BundleError> {
        let destination = self.ensure_archetype(destination_signature, planning)?;
        self.archetype_manager
            .compile_structural_plan(source, destination, key)
    }

    /// Resolves or compiles the structural plan that removes the table
    /// components of `layout_id` from `source`.
    fn ensure_remove_bundle_plan(
        &mut self,
        source: ArchetypeId,
        layout_id: BundleLayoutId,
        planning: &mut PlanningChanges,
    ) -> Result<StructuralPlanId, BundleError> {
        let key = StructuralTransitionKey::RemoveBundle(layout_id);
        if let Some(id) = self.archetype_manager.structural_plan(source, key) {
            return Ok(id);
        }

        let mut destination_signature = self
            .get_archetype(source)
            .ok_or(BundleError::InvalidArchetype(source))?
            .signature()
            .clone();

        for slot in &self.bundle_layouts.get(layout_id).table {
            destination_signature.remove(slot.component);
        }

        self.compile_structural_plan(source, key, destination_signature, planning)
    }

    /// Moves one entity row through a structural plan, updating the location
    /// and the archetype row counters.
    fn execute_structural_plan(
        &mut self,
        plan_id: StructuralPlanId,
        location: &mut EntityLocation,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        let source = location.archetype_id;
        let result = self.archetype_manager.execute_structural_plan(
            plan_id,
            location.storage,
            location.row as usize,
        )?;

        location.archetype_id = result.destination;
        location.row = result.row as u32;

        if result.destination != source {
            let source_len = self
                .get_archetype(source)
                .ok_or(BundleError::InvalidArchetype(source))?
                .len();
            let destination_len = self
                .get_archetype(result.destination)
                .ok_or(BundleError::InvalidArchetype(result.destination))?
                .len();
            planning.mark_archetype_rows(source, source_len);
            planning.mark_archetype_rows(result.destination, destination_len);
        }

        Ok(result.moved_in_source)
    }

    /// Removes every component of bundle `B` from `entity`, migrating the
    /// entity down and dropping matching sparse values.
    pub(crate) fn remove_bundle<B: BundleInfo>(
        &mut self,
        entity: Entity,
        location: &mut EntityLocation,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        if !entity.is_valid() {
            return Err(BundleError::EntityInvalid);
        }

        let layout_id = self.ensure_bundle_layout::<B>()?;
        let previous_location = *location;
        let plan_id = self.ensure_remove_bundle_plan(location.archetype_id, layout_id, planning)?;
        let moved_in_source = self.execute_structural_plan(plan_id, location, planning)?;

        let sparse_slots = &self.bundle_layouts.get(layout_id).sparse;
        for slot in sparse_slots {
            let component_id = slot.component;
            if self
                .sparse_set_manager
                .remove_component(component_id, entity)
            {
                planning.mark_query_work_changed();
            }
        }

        if *location != previous_location {
            self.update_sparse_entity_location(entity, *location);
        }

        Ok(moved_in_source)
    }

    /// Keeps only the sparse components of bundle `B`, removing every other
    /// one, and migrates the entity to the matching archetype.
    pub(crate) fn retain_bundle<B: BundleInfo>(
        &mut self,
        entity: Entity,
        location: &mut EntityLocation,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        if !entity.is_valid() {
            return Err(BundleError::EntityInvalid);
        }

        let layout_id = self.ensure_bundle_layout::<B>()?;
        let previous_location = *location;

        let plan_id = self.ensure_retain_bundle_plan(location.archetype_id, layout_id, planning)?;
        let moved_in_source = self.execute_structural_plan(plan_id, location, planning)?;

        let (bundle_layouts, sparse_sets) = (&self.bundle_layouts, &mut self.sparse_set_manager);
        let keep = &bundle_layouts.get(layout_id).sparse;
        sparse_sets.retain_entity_components(
            entity,
            |component_id| {
                keep.binary_search_by_key(&component_id, |slot| slot.component)
                    .is_ok()
            },
            |_component_id| planning.mark_query_work_changed(),
        );

        if *location != previous_location {
            self.update_sparse_entity_location(entity, *location);
        }

        Ok(moved_in_source)
    }

    /// Resolves or compiles the cached transition edge from `source` to the
    /// archetype that additionally stores `layout_id`'s table components.
    fn ensure_transition_plan(
        &mut self,
        source: ArchetypeId,
        layout_id: BundleLayoutId,
        planning: &mut PlanningChanges,
    ) -> Result<TransitionPlanId, BundleError> {
        if let Some(id) = self.archetype_manager.transition_plan(source, layout_id) {
            return Ok(id);
        }

        let mut destination_signature = self
            .get_archetype(source)
            .ok_or(BundleError::InvalidArchetype(source))?
            .signature()
            .clone();

        for slot in &self.bundle_layouts.get(layout_id).table {
            destination_signature.insert(slot.component);
        }

        let destination = self.ensure_archetype(destination_signature, planning)?;

        let (archetype_manager, bundle_layouts) =
            (&mut self.archetype_manager, &self.bundle_layouts);
        archetype_manager.compile_transition_plan(
            source,
            destination,
            layout_id,
            &bundle_layouts.get(layout_id).table,
        )
    }

    /// Resolves or compiles the table transition for `(source, layout)`.
    ///
    /// Returns no transition when the bundle has no table components.
    pub(crate) fn prepare_transition_plan(
        &mut self,
        source: ArchetypeId,
        layout: BundleLayoutId,
        planning: &mut PlanningChanges,
    ) -> Result<PreparedTransitionPlan, BundleError> {
        let transition = if self.bundle_layouts.get(layout).table.is_empty() {
            None
        } else {
            Some(self.ensure_transition_plan(source, layout, planning)?)
        };

        Ok(PreparedTransitionPlan {
            layout,
            source,
            transition,
        })
    }

    /// Whether a prepared bundle can use the column-major table path.
    ///
    /// Sparse storage is prepared before migration, so mixed bundles can share
    /// the same table migration and then publish their sparse values component-
    /// major. In-place writes remain entity-major because they do not pay the
    /// cost of migrating existing columns.
    pub(crate) fn can_batch_table_transition(
        &self,
        prepared: PreparedTransitionPlan,
        row_count: usize,
    ) -> bool {
        if row_count < MIN_COLUMN_MAJOR_TRANSITION_ROWS {
            return false;
        }

        let Some(destination_columns) = prepared.transition.and_then(|plan| {
            self.archetype_manager
                .transition_migration_column_count(plan)
        }) else {
            return false;
        };

        if self.bundle_layouts.get(prepared.layout).sparse.is_empty() {
            return true;
        }

        // Mixed batches pay for sparse preflight, ownership tracking, and a
        // second component-major pass. Narrow table migrations need a larger
        // run before those fixed costs beat the compact entity-major path.
        let minimum_rows = match destination_columns {
            0..=3 => MIN_NARROW_MIXED_TRANSITION_ROWS,
            4..=5 => MIN_MEDIUM_MIXED_TRANSITION_ROWS,
            _ => MIN_COLUMN_MAJOR_TRANSITION_ROWS,
        };

        row_count >= minimum_rows
    }

    /// Returns the total number of table and sparse items the prepared bundle exposes.
    #[inline]
    pub(crate) fn prepared_bundle_item_count(&self, prepared: PreparedTransitionPlan) -> usize {
        let layout = self.bundle_layouts.get(prepared.layout);
        layout.table.len() + layout.sparse.len()
    }

    /// Validates and reserves every sparse set before table migration starts.
    ///
    /// Once this succeeds, the mixed batch has no fallible sparse lookup or
    /// capacity preparation left after destination rows are allocated.
    pub(crate) fn prepare_sparse_transition_batch(
        &mut self,
        prepared: PreparedTransitionPlan,
        inputs: &[BatchTransitionInput],
    ) -> Result<(), BundleError> {
        let sparse_len = self.bundle_layouts.get(prepared.layout).sparse.len();

        for index in 0..sparse_len {
            let component = self.bundle_layouts.get(prepared.layout).sparse[index].component;

            if !self
                .sparse_set_manager
                .prepare_insert_component_batch(component, inputs)
            {
                return Err(BundleError::SparseSetMissing(component));
            }
        }

        Ok(())
    }

    /// Moves the existing table columns for one homogeneous source lane.
    ///
    /// Bundle-provided values are intentionally left unwritten. The caller
    /// must immediately complete the returned rows through the prepared batch
    /// table and sparse writers before exposing the world again.
    pub(crate) fn begin_prepared_transition_batch(
        &mut self,
        prepared: PreparedTransitionPlan,
        storage: ArchetypeStorageKind,
        inputs: &[BatchTransitionInput],
        planning: &mut PlanningChanges,
    ) -> Result<BatchTransitionResult, BundleError> {
        let plan = prepared
            .transition
            .ok_or(BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch))?;

        let result = self
            .archetype_manager
            .execute_transition_batch(plan, storage, inputs)?;

        debug_assert_eq!(result.source, prepared.source);

        let source_rows = self
            .get_archetype(result.source)
            .ok_or(BundleError::InvalidArchetype(result.source))?
            .len();
        let destination_rows = self
            .get_archetype(result.destination)
            .ok_or(BundleError::InvalidArchetype(result.destination))?
            .len();

        planning.mark_archetype_rows(result.source, source_rows);
        planning.mark_archetype_rows(result.destination, destination_rows);

        Ok(result)
    }

    /// Writes every bundle-owned table column component-major.
    pub(crate) fn write_prepared_transition_table_batch<B: Bundle>(
        &mut self,
        prepared: PreparedTransitionPlan,
        storage: ArchetypeStorageKind,
        rows: &[BatchTransitionRow],
        bundles: &mut BatchBundleStorage<B>,
        tick: ChangeTick,
    ) -> Result<(), BundleError> {
        let plan = prepared
            .transition
            .ok_or(BundleError::ArchetypeOp(ArchetypeError::SignatureMismatch))?;

        self.archetype_manager
            .write_transition_batch(plan, storage, rows, bundles, tick)
    }

    /// Inserts mixed-bundle sparse values component-major after every entity
    /// has received its final table location.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn insert_prepared_transition_sparse_batch<B: Bundle>(
        &mut self,
        prepared: PreparedTransitionPlan,
        storage: ArchetypeStorageKind,
        destination: ArchetypeId,
        rows: &[BatchTransitionRow],
        bundles: &mut BatchBundleStorage<B>,
        tick: ChangeTick,
        planning: &mut PlanningChanges,
        entries: &mut Vec<SparseInsertBatchEntry>,
    ) {
        entries.clear();
        let sparse_len = self.bundle_layouts.get(prepared.layout).sparse.len();
        if sparse_len == 0 {
            return;
        }

        let mut inserted = 0usize;
        entries.reserve(rows.len());

        for slot_index in 0..sparse_len {
            let slot = self.bundle_layouts.get(prepared.layout).sparse[slot_index];
            entries.clear();

            for (row_index, row) in rows.iter().enumerate() {
                let item = unsafe { bundles.item(row_index, slot.item_index) };
                entries.push(SparseInsertBatchEntry {
                    entity: row.entity,
                    location: EntityLocation {
                        archetype_id: destination,
                        storage,
                        row: row.destination_row as u32,
                    },
                    item,
                    inserted_dense_pos: None,
                });
            }

            inserted += unsafe {
                self.sparse_set_manager
                    .insert_component_batch(slot.component, entries, tick)
            };
        }

        entries.clear();

        if inserted != 0 {
            planning.mark_query_work_changed();
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_bundle_prepared_inner(
        &mut self,
        entity: Entity,
        location: &mut EntityLocation,
        prepared: PreparedTransitionPlan,
        items: &mut [ErasedBundleItem],
        tick: ChangeTick,
        planning: &mut PlanningChanges,
        repair_sparse_location: bool,
    ) -> Result<Option<MovedEntity>, BundleError> {
        debug_assert_eq!(location.archetype_id, prepared.source);
        self.debug_validate_bundle_items_for_layout(prepared.layout, items);

        let previous = *location;
        let mut moved = None;

        if let Some(plan) = prepared.transition {
            let result = self.archetype_manager.execute_transition_plan(
                plan,
                location.storage,
                location.row as usize,
                items,
                tick,
            )?;

            location.archetype_id = result.destination;
            location.row = result.row as u32;
            moved = result.moved_in_source;

            if result.destination != prepared.source {
                let source_rows = self.get_archetype(prepared.source).unwrap().len();
                let destination_rows = self.get_archetype(result.destination).unwrap().len();

                planning.mark_archetype_rows(prepared.source, source_rows);
                planning.mark_archetype_rows(result.destination, destination_rows);
            }
        }

        if repair_sparse_location && *location != previous {
            self.update_sparse_entity_location(entity, *location);
        }

        self.insert_sparse_items_prepared(
            entity,
            *location,
            prepared.layout,
            items,
            tick,
            planning,
        )?;

        Ok(moved)
    }


    /// Applies a prepared bundle, repairing sparse locations immediately.
    pub(crate) fn apply_bundle_prepared(
        &mut self,
        entity: Entity,
        location: &mut EntityLocation,
        prepared: PreparedTransitionPlan,
        items: &mut [ErasedBundleItem],
        tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        self.apply_bundle_prepared_inner(entity, location, prepared, items, tick, planning, true)
    }

    /// Applies a prepared bundle while deferring sparse location repair.
    ///
    /// The caller must publish the final entity and displaced-row locations
    /// before the world becomes observable again.
    pub(crate) fn apply_bundle_prepared_deferred_location(
        &mut self,
        entity: Entity,
        location: &mut EntityLocation,
        prepared: PreparedTransitionPlan,
        items: &mut [ErasedBundleItem],
        tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        self.apply_bundle_prepared_inner(entity, location, prepared, items, tick, planning, false)
    }

    /// Resolves or compiles the cached spawn plan for `layout_id`, creating the
    /// empty destination archetype on first use.
    fn ensure_spawn_plan(
        &mut self,
        layout_id: BundleLayoutId,
        planning: &mut PlanningChanges,
    ) -> Result<SpawnPlanId, BundleError> {
        if let Some(id) = self.archetype_manager.spawn_plan(layout_id) {
            return Ok(id);
        }

        let mut signature = ArchetypeSignature::default();

        for slot in &self.bundle_layouts.get(layout_id).table {
            signature.insert(slot.component);
        }

        let destination = self.ensure_archetype(signature, planning)?;

        let (archetype_manager, bundle_layouts) =
            (&mut self.archetype_manager, &self.bundle_layouts);
        archetype_manager.compile_spawn_plan(
            layout_id,
            destination,
            &bundle_layouts.get(layout_id).table,
        )
    }

    /// Resolves or compiles the spawn plan for bundle type `B`.
    pub(crate) fn prepare_spawn<B: BundleInfo>(
        &mut self,
        planning: &mut PlanningChanges,
    ) -> Result<PreparedSpawn, BundleError> {
        let layout = self.ensure_bundle_layout::<B>()?;
        let plan = self.ensure_spawn_plan(layout, planning)?;

        Ok(PreparedSpawn { layout, plan })
    }

    /// Returns the component ids and original bundle positions for a sparse-only
    /// spawn layout. Mixed and table-only layouts retain the general executor.
    pub(crate) fn sparse_only_spawn_slots(
        &self,
        prepared: PreparedSpawn,
    ) -> Option<Vec<(ComponentId, u16)>> {
        let layout = self.bundle_layouts.get(prepared.layout);

        if !layout.table.is_empty() || layout.sparse.is_empty() {
            return None;
        }

        Some(
            layout
                .sparse
                .iter()
                .map(|slot| (slot.component, slot.item_index))
                .collect(),
        )
    }

    /// Returns the number of erased fields consumed by the table-only batch
    /// executor. Sparse and mixed layouts retain their specialised or general
    /// paths respectively.
    pub(crate) fn table_only_spawn_item_count(&self, prepared: PreparedSpawn) -> Option<usize> {
        let layout = self.bundle_layouts.get(prepared.layout);

        layout.sparse.is_empty().then_some(layout.table.len())
    }

    /// Moves a homogeneous table bundle batch into one prepared archetype.
    ///
    /// The spawn plan and destination archetype are resolved once. Component
    /// values are then written column-major while ownership remains guarded by
    /// `BatchBundleStorage`.
    pub(crate) fn spawn_table_batch<B: Bundle>(
        &mut self,
        entities: &[Entity],
        storage: ArchetypeStorageKind,
        prepared: PreparedSpawn,
        bundles: &mut BatchBundleStorage<B>,
        tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<Vec<(Entity, EntityLocation)>, BundleError> {
        if entities.iter().any(|entity| !entity.is_valid()) {
            return Err(BundleError::EntityInvalid);
        }

        debug_assert!(self.bundle_layouts.get(prepared.layout).sparse.is_empty());
        debug_assert_eq!(entities.len(), bundles.len());

        let mut activations = Vec::with_capacity(entities.len());
        let archetype_id = self.archetype_manager.execute_spawn_plan_batch(
            prepared.plan,
            storage,
            entities,
            bundles,
            tick,
            &mut activations,
        )?;

        let row_count = self
            .get_archetype(archetype_id)
            .ok_or(BundleError::InvalidArchetype(archetype_id))?
            .len();
        planning.mark_archetype_rows(archetype_id, row_count);

        Ok(activations)
    }

    /// Validates every sparse set and reserves the complete batch before any
    /// component value is moved out of its bundle.
    pub(crate) fn prepare_sparse_spawn_batch(
        &mut self,
        slots: &[(ComponentId, u16)],
        entities: &[Entity],
    ) -> Result<(), BundleError> {
        for &(component, _) in slots {
            if !self.sparse_set_manager.is_registered(component.index()) {
                return Err(BundleError::SparseSetMissing(component));
            }

            if !self
                .sparse_set_manager
                .prepare_spawn_component_batch(component, entities)
            {
                return Err(BundleError::SparseInsertFailed(component));
            }
        }

        Ok(())
    }

    /// Allocates the empty table row that owns one sparse-only entity.
    pub(crate) fn allocate_sparse_spawn_row(
        &mut self,
        entity: Entity,
        storage: ArchetypeStorageKind,
        prepared: PreparedSpawn,
        tick: ChangeTick,
    ) -> Result<EntityLocation, BundleError> {
        debug_assert!(self.bundle_layouts.get(prepared.layout).table.is_empty());
        let mut no_items = [];
        let (archetype_id, row) = self.archetype_manager.execute_spawn_plan(
            prepared.plan,
            storage,
            entity,
            &mut no_items,
            tick,
        )?;

        Ok(EntityLocation {
            archetype_id,
            storage,
            row: row as u32,
        })
    }

    /// Moves one prepared sparse component column into its typed set.
    ///
    /// # Safety
    ///
    /// The component set and every entry must satisfy
    /// `SparseSet::insert_spawn_batch_prepared`.
    pub(crate) unsafe fn insert_sparse_spawn_component_batch(
        &mut self,
        component: ComponentId,
        entries: &[SparseSpawnBatchEntry],
        tick: ChangeTick,
    ) {
        unsafe {
            self.sparse_set_manager
                .insert_spawn_component_batch(component, entries, tick);
        }
    }

    /// Publishes sparse membership heads and one final archetype row count.
    pub(crate) fn finish_sparse_spawn_batch(
        &mut self,
        entities: &[Entity],
        head: ComponentId,
        archetype_id: ArchetypeId,
        planning: &mut PlanningChanges,
    ) -> Result<(), BundleError> {
        self.sparse_set_manager
            .commit_spawn_batch_memberships(entities, head);

        let rows = self
            .get_archetype(archetype_id)
            .ok_or(BundleError::InvalidArchetype(archetype_id))?
            .len();
        planning.mark_archetype_rows(archetype_id, rows);
        planning.mark_query_work_changed();
        Ok(())
    }

    /// Resolves or compiles the structural plan that removes `component` from
    /// `source`.
    fn ensure_remove_component_plan(
        &mut self,
        source: ArchetypeId,
        component: ComponentId,
        planning: &mut PlanningChanges,
    ) -> Result<StructuralPlanId, BundleError> {
        let key = StructuralTransitionKey::RemoveComponent(component);
        if let Some(id) = self.archetype_manager.structural_plan(source, key) {
            return Ok(id);
        }

        let mut destination_signature = self
            .get_archetype(source)
            .ok_or(BundleError::InvalidArchetype(source))?
            .signature()
            .clone();
        destination_signature.remove(component);

        self.compile_structural_plan(source, key, destination_signature, planning)
    }

    /// Prepares the cached table transition used to remove one component from
    /// every entity in a homogeneous source-archetype run.
    pub(crate) fn prepare_remove_component_transition(
        &mut self,
        source: ArchetypeId,
        component: ComponentId,
        planning: &mut PlanningChanges,
    ) -> Result<StructuralPlanId, BundleError> {
        self.ensure_remove_component_plan(source, component, planning)
    }

    /// Whether a structural removal run can amortize column-major
    /// execution and actually changes its source archetype.
    pub(crate) fn can_batch_structural_transition(
        &self,
        plan: StructuralPlanId,
        row_count: usize,
    ) -> bool {
        row_count >= MIN_COLUMN_MAJOR_TRANSITION_ROWS
            && self.archetype_manager.structural_plan_is_migration(plan)
            && self.archetype_manager.structural_plan_moved_columns(plan)
                >= MIN_COLUMN_MAJOR_MOVED_COLUMNS
    }

    /// Executes one row through an already resolved structural edge.
    ///
    /// This is the low-overhead counterpart to the column-major executor for
    /// narrow or small runs. It avoids repeating component lookup and plan
    /// resolution while preserving sparse-driver location repair.
    pub(crate) fn execute_prepared_structural_transition(
        &mut self,
        entity: Entity,
        plan: StructuralPlanId,
        location: &mut EntityLocation,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        self.execute_prepared_structural_transition_inner(entity, plan, location, planning, true)
    }

    /// Executes a prepared structural transition while deferring sparse repair.
    pub(crate) fn execute_prepared_structural_transition_deferred_location(
        &mut self,
        entity: Entity,
        plan: StructuralPlanId,
        location: &mut EntityLocation,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        self.execute_prepared_structural_transition_inner(entity, plan, location, planning, false)
    }

    /// Shared body of the prepared structural executors; skips the sparse
    /// location repair when the caller defers it through its batch.
    fn execute_prepared_structural_transition_inner(
        &mut self,
        entity: Entity,
        plan: StructuralPlanId,
        location: &mut EntityLocation,
        planning: &mut PlanningChanges,
        repair_sparse_location: bool,
    ) -> Result<Option<MovedEntity>, BundleError> {
        let previous = *location;
        let moved = self.execute_structural_plan(plan, location, planning)?;

        if repair_sparse_location && *location != previous {
            self.update_sparse_entity_location(entity, *location);
        }

        Ok(moved)
    }

    /// Executes one prepared structural transition over an ordered source lane.
    pub(crate) fn execute_prepared_structural_batch(
        &mut self,
        plan: StructuralPlanId,
        storage: ArchetypeStorageKind,
        source_rows: &[usize],
        planning: &mut PlanningChanges,
    ) -> Result<BatchTransitionResult, BundleError> {
        let result = self.archetype_manager.execute_structural_transition_batch(
            plan,
            storage,
            source_rows,
        )?;

        let source_rows = self
            .get_archetype(result.source)
            .ok_or(BundleError::InvalidArchetype(result.source))?
            .len();
        let destination_rows = self
            .get_archetype(result.destination)
            .ok_or(BundleError::InvalidArchetype(result.destination))?
            .len();

        planning.mark_archetype_rows(result.source, source_rows);
        planning.mark_archetype_rows(result.destination, destination_rows);

        Ok(result)
    }

    /// Resolves or compiles the structural plan that keeps only the table
    /// components of `layout_id` from `source`.
    fn ensure_retain_bundle_plan(
        &mut self,
        source: ArchetypeId,
        layout_id: BundleLayoutId,
        planning: &mut PlanningChanges,
    ) -> Result<StructuralPlanId, BundleError> {
        let key = StructuralTransitionKey::RetainBundle(layout_id);
        if let Some(id) = self.archetype_manager.structural_plan(source, key) {
            return Ok(id);
        }

        let source_components = self
            .get_archetype(source)
            .ok_or(BundleError::InvalidArchetype(source))?
            .signature()
            .components();
        let table = &self.bundle_layouts.get(layout_id).table;
        let mut destination_signature = ArchetypeSignature::default();

        for &component in source_components {
            if table
                .binary_search_by_key(&component, |slot| slot.component)
                .is_ok()
            {
                destination_signature.insert(component);
            }
        }

        self.compile_structural_plan(source, key, destination_signature, planning)
    }

    /// Asserts that erased bundle items match the layout and were not yet consumed.
    #[inline(always)]
    fn debug_validate_bundle_items_for_layout(
        &self,
        layout_id: BundleLayoutId,
        items: &[ErasedBundleItem],
    ) {
        let layout = self.bundle_layouts.get(layout_id);

        debug_assert_eq!(
            layout.table.len() + layout.sparse.len(),
            items.len(),
            "bundle implementation produced an unexpected number of items"
        );

        debug_assert!(
            items.iter().all(|item| !item.consumed),
            "bundle item was already consumed before plan execution"
        );
    }

    /// Inserts the sparse items of a cached bundle layout using their original
    /// item positions.
    ///
    /// Dispatches on the sparse slot count: zero slots returns immediately,
    /// a single slot takes the dedicated immediate path, and several slots are
    /// inserted through the deferred-membership chain.
    fn insert_sparse_items_prepared(
        &mut self,
        entity: Entity,
        location: EntityLocation,
        layout_id: BundleLayoutId,
        items: &mut [ErasedBundleItem],
        change_tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<(), BundleError> {
        match self.bundle_layouts.get(layout_id).sparse.len() {
            0 => Ok(()),
            1 => {
                let slot = self.bundle_layouts.get(layout_id).sparse[0];
                self.insert_single_sparse_item_prepared(
                    entity,
                    location,
                    slot,
                    items,
                    change_tick,
                    planning,
                )
            }
            _ => self.insert_many_sparse_items_prepared(
                entity,
                location,
                layout_id,
                items,
                change_tick,
                planning,
            ),
        }
    }

    /// Inserts one sparse item without deferred-membership bookkeeping.
    ///
    /// Keeping this path out of the multi-item implementation lets the
    /// compiler optimize both structural cases independently.
    #[inline(always)]
    fn insert_single_sparse_item_prepared(
        &mut self,
        entity: Entity,
        location: EntityLocation,
        slot: BundleSlot,
        items: &mut [ErasedBundleItem],
        change_tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<(), BundleError> {
        let item = &mut items[slot.item_index as usize];

        let result = unsafe {
            self.sparse_set_manager.insert_move_immediate(
                slot.component,
                entity,
                location,
                item.ptr,
                change_tick,
            )
        }
        .ok_or(BundleError::SparseInsertFailed(slot.component))?;

        item.consumed = true;

        if result.inserted {
            planning.mark_query_work_changed();
        }

        Ok(())
    }

    /// Inserts several sparse items and commits their memberships as one chain.
    ///
    /// Successfully moved items are linked before an error is returned so the
    /// normal rollback path can still discover and remove them.
    #[inline(never)]
    fn insert_many_sparse_items_prepared(
        &mut self,
        entity: Entity,
        location: EntityLocation,
        layout_id: BundleLayoutId,
        items: &mut [ErasedBundleItem],
        change_tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<(), BundleError> {
        let (layouts, sparse_sets) = (&self.bundle_layouts, &mut self.sparse_set_manager);
        let sparse_slots = layouts.get(layout_id).slots(StorageKind::Sparse);

        debug_assert!(sparse_slots.len() > 1);
        sparse_sets.begin_deferred_insertions(sparse_slots.len());

        for slot in sparse_slots {
            let item = &mut items[slot.item_index as usize];

            let Some(result) = (unsafe {
                sparse_sets.insert_move_deferred(
                    slot.component,
                    entity,
                    location,
                    item.ptr,
                    change_tick,
                )
            }) else {
                sparse_sets.finish_deferred_insertions(entity);
                return Err(BundleError::SparseInsertFailed(slot.component));
            };

            item.consumed = true;

            if result.inserted {
                planning.mark_query_work_changed();
            }
        }

        sparse_sets.finish_deferred_insertions(entity);
        Ok(())
    }

    /// Applies table and sparse bundle items to an entity location.
    ///
    /// Table items may migrate the entity into another archetype. Sparse items are
    /// inserted afterwards and do not affect the table location.
    pub(crate) fn apply_bundle(
        &mut self,
        entity: Entity,
        location: &mut EntityLocation,
        layout_id: BundleLayoutId,
        bundle: &mut [ErasedBundleItem],
        change_tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        if !entity.is_valid() {
            return Err(BundleError::EntityInvalid);
        }

        self.debug_validate_bundle_items_for_layout(layout_id, bundle);

        let has_table_components = !self.bundle_layouts.get(layout_id).table.is_empty();

        if !has_table_components {
            self.insert_sparse_items_prepared(
                entity,
                *location,
                layout_id,
                bundle,
                change_tick,
                planning,
            )?;

            return Ok(None);
        }

        let previous_location = *location;
        let source = location.archetype_id;

        let plan_id = self.ensure_transition_plan(source, layout_id, planning)?;

        let result = self.archetype_manager.execute_transition_plan(
            plan_id,
            location.storage,
            location.row as usize,
            bundle,
            change_tick,
        )?;

        location.archetype_id = result.destination;
        location.row = result.row as u32;

        if result.destination != source {
            let source_len = self
                .get_archetype(source)
                .ok_or(BundleError::InvalidArchetype(source))?
                .len();
            let destination_len = self
                .get_archetype(result.destination)
                .ok_or(BundleError::InvalidArchetype(result.destination))?
                .len();
            planning.mark_archetype_rows(source, source_len);
            planning.mark_archetype_rows(result.destination, destination_len);
        }

        if *location != previous_location {
            self.update_sparse_entity_location(entity, *location);
        }

        self.insert_sparse_items_prepared(
            entity,
            *location,
            layout_id,
            bundle,
            change_tick,
            planning,
        )?;

        Ok(result.moved_in_source)
    }

    /// Executes a prepared spawn: allocates the final row, writes every bundle
    /// value, and publishes the sparse items.
    pub(crate) fn spawn_with_plan(
        &mut self,
        entity: Entity,
        storage: ArchetypeStorageKind,
        prepared: PreparedSpawn,
        items: &mut [ErasedBundleItem],
        tick: ChangeTick,
        planning: &mut PlanningChanges,
    ) -> Result<EntityLocation, BundleError> {
        if !entity.is_valid() {
            return Err(BundleError::EntityInvalid);
        }

        self.debug_validate_bundle_items_for_layout(prepared.layout, items);

        let (archetype_id, row) = self.archetype_manager.execute_spawn_plan(
            prepared.plan,
            storage,
            entity,
            items,
            tick,
        )?;

        let location = EntityLocation {
            archetype_id,
            storage,
            row: row as u32,
        };

        self.insert_sparse_items_prepared(
            entity,
            location,
            prepared.layout,
            items,
            tick,
            planning,
        )?;

        let rows = self
            .get_archetype(archetype_id)
            .ok_or(BundleError::InvalidArchetype(archetype_id))?
            .len();
        planning.mark_archetype_rows(archetype_id, rows);

        Ok(location)
    }

    /// Removes one component from `entity`, migrating table rows or unlinking
    /// sparse memberships as the storage kind requires.
    pub(crate) fn remove<C: Component>(
        &mut self,
        entity: Entity,
        location: &mut EntityLocation,
        planning: &mut PlanningChanges,
    ) -> Result<Option<MovedEntity>, BundleError> {
        let component_id = self
            .id_of::<C>()
            .ok_or(BundleError::ComponentNotRegistered(TypeId::of::<C>()))?;

        match self
            .storage_of(component_id)
            .ok_or(BundleError::StorageUnknown(component_id))?
        {
            StorageKind::Sparse => {
                if self
                    .sparse_set_manager
                    .remove_component(component_id, entity)
                {
                    planning.mark_query_work_changed();
                }

                Ok(None)
            }

            StorageKind::Table => {
                let previous_location = *location;
                let plan_id = self.ensure_remove_component_plan(
                    location.archetype_id,
                    component_id,
                    planning,
                )?;
                let moved = self.execute_structural_plan(plan_id, location, planning)?;
                if *location != previous_location {
                    self.update_sparse_entity_location(entity, *location);
                }
                Ok(moved)
            }
        }
    }
}
