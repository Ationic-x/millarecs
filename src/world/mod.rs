//! World storage and high-level ECS operations.
//!
//! A [`World`] owns every live entity, every registered component storage, and the
//! archetype graph used by table-backed components. It is intentionally a thin
//! orchestration layer: entity allocation is delegated to `EntityManager`, while
//! component layout and migrations are delegated to [`ComponentManager`].
//!
//! This module holds the struct and the core lifecycle (spawn, insert, remove,
//! despawn, access views). The query construction API lives in `query_api`,
//! resource management in `resources`, event observation in `observation`, and
//! batch operations in `batch`.

mod batch;
mod batch_validation;
pub(crate) mod execution_world;
mod identifier;
mod observation;
mod prepared;
mod query_api;
mod resources;

use std::marker::PhantomData;
use std::sync::Arc;
use std::{any::TypeId, collections::VecDeque};

use crate::hierarchy::{Children, Parent};
use crate::{
    __private::{Bundle, BundleError, BundleInfo},
    change_detection::{ChangeTick, Mut},
    component::{
        archetype::{ArchetypeError, ArchetypeId, ArchetypeManagerError, ArchetypeSignature},
        CompactionPolicy, Component, ComponentId, ComponentManager, ComponentManagerError,
        FragmentedCompactionReport, MovedEntity, SparseLocationUpdateGroups, StorageKind,
    },
    entity::{
        metadata::{ArchetypeStorageKind, EntityLocation},
        Entity, EntityError, EntityManager,
    },
    event::{Despawned, Events, Mutated, Spawned},
    planning::{PlanningChanges, WorldPlanning},
    query::{filter::Changed, state::QueryState, QueryError},
    resource::{ResourceError, ResourceManager},
    system::{access::SystemAccessError, commands::CommandError},
    task::TaskExecutorHandle,
};

pub use execution_world::ExecutionWorld;
pub use identifier::WorldId;
pub use prepared::{
    PreparedInsert, PreparedInsertSession, PreparedRemove, PreparedRemoveSession, PreparedSpawn,
    PreparedSpawnSession,
};

/// One registered event cycle: retires the buffers of a single
/// `Events<E>` resource when the world advances a cycle.
pub(crate) type EventCycle = Arc<dyn Fn(&mut World) + Send + Sync>;

/// Registered interest in entity lifecycle events.
///
/// Spawn and despawn sites branch on these flags before touching event
/// storage, so a world without observers pays nothing for lifecycle events.
#[derive(Default)]
pub(crate) struct LifecycleWatch {
    pub(crate) spawned: bool,
    pub(crate) despawned: bool,
}

/// One registered change observer: a retained `Changed<C>` query whose
/// matches are published as a single `Mutated<C>` batch event each cycle.
trait ChangeDispatch: Send {
    /// Publishes one batch covering the changes since the previous call.
    fn dispatch(&mut self, world: &mut World) -> Result<(), WorldError>;
}

/// Object-safe change dispatcher for one component type `C`.
///
/// Retaining the query state across cycles is what makes each dispatch
/// report only the entities changed since the previous one.
pub(crate) struct ChangeDispatchFor<C: Component> {
    /// Retained `Entity`-only `Changed<C>` state; each dispatch run advances
    /// its change window.
    pub(crate) state: QueryState<Entity, Changed<C>>,
}

impl<C: Component> ChangeDispatch for ChangeDispatchFor<C> {
    fn dispatch(&mut self, world: &mut World) -> Result<(), WorldError> {
        let mut entities = Vec::new();

        {
            let mut query = world
                .query_with(&mut self.state)
                .expect("change observer query must remain valid for its world");
            query.for_each(|entity| entities.push(entity));
        }

        if !entities.is_empty() {
            if let Some(mut events) = world.get_resource_mut::<Events<Mutated<C>>>() {
                events.send(Mutated {
                    entities: entities.into_boxed_slice(),
                    _marker: PhantomData,
                });
            }
        }

        Ok(())
    }
}

/// Central container for entities, component storages, and archetypes.
///
/// Table components are stored in archetypes and may move an entity when they are
/// added. Sparse components are stored separately and do not affect the entity's
/// archetype signature.
pub struct World {
    id: WorldId,
    /// Query-planning metadata; structural entry points stage their changes
    /// in a local `PlanningChanges` and publish them here only at the end,
    /// after storage mutations have settled.
    world_planning: WorldPlanning,
    entities: EntityManager,
    batch_entity_validator: batch_validation::BatchEntityValidator,
    components: ComponentManager,
    resources: ResourceManager,
    /// Archetype shared by every entity without table components; it exists
    /// from world creation and is never removed.
    empty_archetype: ArchetypeId,
    /// Retirement callbacks for registered `Events<E>` resources, keyed by
    /// their type id.
    event_cycles: Vec<(TypeId, EventCycle)>,
    lifecycle_watch: LifecycleWatch,
    /// Registered change observers, keyed by their component type id.
    change_watch: Vec<(TypeId, Box<dyn ChangeDispatch>)>,
    /// World-wide change-detection tick; every mutating entry point advances
    /// it so its change lands in a fresh window.
    change_tick: ChangeTick,
}

/// Component metadata resolved for repeated access in one world.
///
/// A handle avoids resolving `TypeId`, `ComponentId`, and the storage backend
/// again for every entity. Handles are tied to the world that created them.
pub struct ComponentHandle<C: Component> {
    world_id: WorldId,
    component: ComponentId,
    storage: StorageKind,
    /// Ties the handle to `C` without holding a value or a borrow, so the
    /// handle stays `Copy` regardless of `C`.
    _marker: PhantomData<fn() -> C>,
}

impl<C: Component> Copy for ComponentHandle<C> {}

impl<C: Component> Clone for ComponentHandle<C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<C: Component> std::fmt::Debug for ComponentHandle<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ComponentHandle")
            .field("world_id", &self.world_id)
            .field("component", &self.component)
            .field("storage", &self.storage)
            .finish()
    }
}

impl<C: Component> ComponentHandle<C> {
    /// Returns the world this handle belongs to.
    #[inline]
    pub const fn world_id(self) -> WorldId {
        self.world_id
    }

    /// Returns the resolved component id.
    #[inline]
    pub const fn component_id(self) -> ComponentId {
        self.component
    }

    /// Returns the resolved component storage backend.
    #[inline]
    pub const fn storage(self) -> StorageKind {
        self.storage
    }
}

/// Errors produced by [`World`] operations.
#[derive(Debug)]
pub enum WorldError {
    /// A component registry or storage operation failed.
    Component(ComponentManagerError),
    /// A bundle could not be validated or applied.
    Bundle(BundleError),
    /// Query state creation failed.
    Query(QueryError),
    /// Archetype lookup or creation failed at the manager level.
    ArchetypeManager(ArchetypeManagerError),
    /// A concrete archetype operation failed.
    Archetype(ArchetypeError),
    /// A resource access or validation failed.
    Resource(ResourceError),
    /// The requested entity is not alive in this world.
    EntityNotFound(Entity),
    /// A system parameter access validation failed.
    SystemAccess(SystemAccessError),
    /// A deferred command failed while being applied.
    Command(CommandError),
    /// An entity lifecycle operation failed.
    Entity(EntityError),
    /// A batch received a different entity and bundle count.
    BatchLengthMismatch {
        /// Number of entities supplied.
        entities: usize,
        /// Number of bundles supplied.
        bundles: usize,
    },
    /// An item bound to one world was used against another.
    WorldMismatch {
        /// Identifier of the world the item was created with.
        expected: WorldId,
        /// Identifier of the world it was used against.
        actual: WorldId,
    },
    /// No more unique world identifiers can be allocated.
    WorldIdExhausted,
}

impl std::fmt::Display for WorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Component(err) => write!(f, "component error: {err:?}"),
            Self::Bundle(err) => write!(f, "bundle error: {err:?}"),
            Self::Query(err) => write!(f, "query error: {err:?}"),
            Self::ArchetypeManager(err) => write!(f, "archetype manager error: {err:?}"),
            Self::Archetype(err) => write!(f, "archetype error: {err:?}"),
            Self::Resource(err) => write!(f, "resource error: {err}"),
            Self::EntityNotFound(entity) => write!(f, "entity not found: {entity:?}"),
            Self::SystemAccess(err) => write!(f, "system access error: {err}"),
            Self::Command(err) => write!(f, "command error: {err}"),
            Self::Entity(err) => write!(f, "entity error: {err}"),
            Self::BatchLengthMismatch { entities, bundles } => write!(
                f,
                "batch length mismatch. Entities length: {entities}. Bundle length: {bundles}"
            ),
            Self::WorldMismatch { expected, actual } => write!(
                f,
                "world id mismatch. Expected world id: {expected} Received world id: {actual}"
            ),
            Self::WorldIdExhausted => f.write_str("world identifier space is exhausted"),
        }
    }
}

impl std::error::Error for WorldError {}

impl From<EntityError> for WorldError {
    fn from(value: EntityError) -> Self {
        match value {
            EntityError::NotFound(entity)
            | EntityError::NotAlive(entity)
            | EntityError::Stale(entity)
            | EntityError::Invalid(entity) => Self::EntityNotFound(entity),

            other => Self::Entity(other),
        }
    }
}

impl From<ComponentManagerError> for WorldError {
    fn from(value: ComponentManagerError) -> Self {
        Self::Component(value)
    }
}

impl From<BundleError> for WorldError {
    fn from(value: BundleError) -> Self {
        Self::Bundle(value)
    }
}

impl From<QueryError> for WorldError {
    fn from(value: QueryError) -> Self {
        Self::Query(value)
    }
}

impl From<ArchetypeManagerError> for WorldError {
    fn from(value: ArchetypeManagerError) -> Self {
        Self::ArchetypeManager(value)
    }
}

impl From<ArchetypeError> for WorldError {
    fn from(value: ArchetypeError) -> Self {
        Self::Archetype(value)
    }
}

impl From<ResourceError> for WorldError {
    fn from(value: ResourceError) -> Self {
        Self::Resource(value)
    }
}

impl From<SystemAccessError> for WorldError {
    fn from(value: SystemAccessError) -> Self {
        Self::SystemAccess(value)
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
impl World {
    /// Returns the entity allocator backing this world.
    #[inline]
    #[cfg(test)]
    pub(crate) fn entities(&self) -> &EntityManager {
        &self.entities
    }

    /// Returns the component registry and storage manager.
    #[inline]
    pub(crate) fn components(&self) -> &ComponentManager {
        &self.components
    }

    /// Returns planning metadata used to refresh reusable query plans.
    #[inline]
    pub(crate) fn planning(&self) -> &WorldPlanning {
        &self.world_planning
    }

    /// Creates a new empty world.
    ///
    /// # Panics
    ///
    /// Panics if the world cannot be initialized. Use [`World::try_new`] when the
    /// failure must be handled explicitly.
    pub fn new() -> Self {
        Self::try_new().expect("failed to create world")
    }

    /// Borrows the world regions that remain stable while systems execute.
    ///
    /// Structural changes require `&mut World`, so they cannot occur until the
    /// returned view and all values derived from it have been dropped.
    pub(crate) fn execution_world(&self) -> ExecutionWorld<'_> {
        ExecutionWorld::new(
            self.id,
            &self.components,
            &self.world_planning,
            &self.resources,
            &self.entities,
            None,
            self.change_tick,
        )
    }

    /// Borrows the world regions that remain stable while systems execute,
    /// exposing the schedule's task executor to parallel systems.
    pub(crate) fn execution_world_with_tasks<'w>(
        &'w self,
        tasks: Option<&'w TaskExecutorHandle>,
    ) -> ExecutionWorld<'w> {
        ExecutionWorld::new(
            self.id,
            &self.components,
            &self.world_planning,
            &self.resources,
            &self.entities,
            tasks,
            self.change_tick,
        )
    }

    /// Tries to create a new empty world.
    pub fn try_new() -> Result<Self, WorldError> {
        let mut components = ComponentManager::new();
        let mut world_planning = WorldPlanning::new();
        let mut planning = PlanningChanges::default();
        let empty_archetype =
            components.ensure_archetype(ArchetypeSignature::default(), &mut planning)?;
        let id = WorldId::allocate().ok_or(WorldError::WorldIdExhausted)?;

        world_planning.apply(planning);

        Ok(Self {
            id,
            entities: EntityManager::new(),
            batch_entity_validator: batch_validation::BatchEntityValidator::default(),
            resources: ResourceManager::new(),
            components,
            world_planning,
            empty_archetype,
            event_cycles: Vec::new(),
            lifecycle_watch: LifecycleWatch::default(),
            change_watch: Vec::new(),
            change_tick: ChangeTick::ZERO.next(),
        })
    }

    /// Registers `C` as a table/archetype component.
    ///
    /// Table components participate in archetype signatures. Adding one to an
    /// existing entity can migrate the entity to another archetype.
    pub fn register_table<C: Component>(&mut self) -> Result<ComponentId, WorldError> {
        Ok(self.components.register_column::<C>()?)
    }

    /// Registers `C` as a sparse-set component.
    ///
    /// Sparse components are looked up per entity and do not change archetype
    /// membership when inserted or removed.
    pub fn register_sparse<C: Component>(&mut self) -> Result<ComponentId, WorldError> {
        let mut planning = PlanningChanges::default();

        let component_id = self.components.register_sparse_set::<C>(&mut planning)?;

        self.world_planning.apply(planning);

        Ok(component_id)
    }

    /// Returns whether `entity` is currently alive in this world.
    #[inline]
    pub fn contains(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    /// Returns the number of live entities.
    #[inline]
    pub fn len(&self) -> usize {
        self.entities.alive_len()
    }

    /// Returns whether the world has no live entities.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    /// Spawns an entity without user components.
    pub fn spawn_empty(&mut self) -> Result<Entity, WorldError> {
        self.spawn_empty_with_storage(ArchetypeStorageKind::Contiguous)
    }

    /// Spawns an entity without user components in the requested archetype storage.
    pub fn spawn_empty_with_storage(
        &mut self,
        storage: ArchetypeStorageKind,
    ) -> Result<Entity, WorldError> {
        let entity = self.entities.reserve();
        let empty_archetype = self.empty_archetype;
        let mut planning = PlanningChanges::default();

        let row = self
            .components
            .get_archetype_mut(empty_archetype)
            .expect("empty archetype must exist")
            .push_empty_row(storage, entity)?;
        let rows = self
            .components
            .get_archetype(empty_archetype)
            .expect("empty archetype must exist")
            .len();

        let location = EntityLocation {
            archetype_id: empty_archetype,
            storage,
            row: row as u32,
        };

        self.entities.activate(entity, location)?;

        planning.mark_archetype_rows(empty_archetype, rows);
        self.world_planning.apply(planning);

        Ok(entity)
    }

    /// Updates metadata for an entity moved by a swap-remove operation.
    #[inline]
    fn apply_moved_entity(&mut self, moved: Option<MovedEntity>) {
        if let Some(moved) = moved {
            self.components
                .update_sparse_entity_location(moved.entity, moved.location);
            self.entities
                .set_location(moved.entity, moved.location)
                .expect("moved entity must still be alive");
        }
    }

    /// Applies a batch of swap-remove moves through one component-major
    /// grouped sparse pass.
    fn apply_moved_entities<I>(&mut self, moved: I)
    where
        I: IntoIterator<Item = MovedEntity>,
    {
        let mut groups = SparseLocationUpdateGroups::default();
        self.apply_moved_entities_with_groups(moved, &mut groups);
    }

    /// Repairs moved entities through one component-major sparse pass.
    fn apply_moved_entities_with_groups<I>(
        &mut self,
        moved: I,
        groups: &mut SparseLocationUpdateGroups,
    ) where
        I: IntoIterator<Item = MovedEntity>,
    {
        self.apply_entity_locations_with_groups(
            moved
                .into_iter()
                .map(|moved| (moved.entity, moved.location)),
            groups,
        );
    }

    /// Publishes entity locations and groups sparse repairs by component.
    fn apply_entity_locations_with_groups<I>(
        &mut self,
        updates: I,
        groups: &mut SparseLocationUpdateGroups,
    ) where
        I: IntoIterator<Item = (Entity, EntityLocation)>,
    {
        debug_assert!(groups.is_empty());

        for (entity, location) in updates {
            self.components
                .prepare_sparse_entity_location_update(entity, location, groups);
            self.entities
                .set_location(entity, location)
                .expect("moved entity must still be alive");
        }

        self.components
            .execute_sparse_entity_location_updates(groups);
    }

    /// Repairs sparse location metadata after entity locations were already
    /// published during an exclusive structural session.
    fn repair_sparse_entity_locations<I>(&mut self, updates: I)
    where
        I: IntoIterator<Item = (Entity, EntityLocation)>,
    {
        let mut groups = SparseLocationUpdateGroups::default();

        for (entity, location) in updates {
            self.components
                .prepare_sparse_entity_location_update(entity, location, &mut groups);
        }

        self.components
            .execute_sparse_entity_location_updates(&mut groups);
    }

    /// Compacts fragmented archetype pages and updates entity locations.
    ///
    /// This is a structural operation. It must not run while query iterators or
    /// component references are alive.
    pub fn compact_archetypes(&mut self) {
        let report = self.compact_fragmented_archetypes(CompactionPolicy::aggressive());

        debug_assert!(
            report.compacted_archetypes == 0 || report.moved_rows <= self.entities.alive_len()
        );
    }

    /// Compacts fragmented archetype pages selected by `policy`.
    ///
    /// The returned report tells how much storage maintenance happened. This is
    /// a structural operation and must not run while query iterators or
    /// component references are alive.
    pub fn compact_fragmented_archetypes(
        &mut self,
        policy: CompactionPolicy,
    ) -> FragmentedCompactionReport {
        let (moved, report) = self.components.compact_fragmented_with_policy(policy);
        self.apply_moved_entities(moved);

        if report.compacted_archetypes != 0 {
            let mut planning = PlanningChanges::default();
            planning.mark_query_work_changed();
            self.world_planning.apply(planning);
        }

        report
    }

    /// Applies a bundle to an existing live entity.
    ///
    /// Table components may migrate the entity to another archetype. Sparse
    /// components are inserted into their sparse sets after bundle validation.
    pub fn insert<B: Bundle>(&mut self, entity: Entity, bundle: B) -> Result<(), WorldError> {
        let tick = self.advance_tick();
        self.insert_at_tick(entity, bundle, tick)
    }

    /// Applies a bundle stamped with a caller-supplied change tick.
    ///
    /// Batched paths share one tick so a whole batch falls inside a single
    /// change-filter window.
    pub(crate) fn insert_at_tick<B: Bundle>(
        &mut self,
        entity: Entity,
        bundle: B,
        tick: ChangeTick,
    ) -> Result<(), WorldError> {
        let mut planning = PlanningChanges::default();
        let layout_id = self.components.ensure_bundle_layout::<B>()?;

        let moved = {
            let components = &mut self.components;
            let entities = &mut self.entities;

            let location = entities
                .location_mut(entity)
                .ok_or(WorldError::EntityNotFound(entity))?;

            bundle.with_items(components, |components, items| {
                components.apply_bundle(entity, location, layout_id, items, tick, &mut planning)
            })?
        };

        self.apply_moved_entity(moved);
        self.world_planning.apply(planning);
        Ok(())
    }

    /// Spawns an entity and places `bundle` directly into its final storage.
    ///
    /// Table-backed components are written into the destination archetype in one
    /// pass. This avoids the older spawn-empty-then-insert path, which created a
    /// row in the empty archetype and immediately migrated it for non-empty
    /// table bundles.
    pub fn spawn<B: Bundle>(&mut self, bundle: B) -> Result<Entity, WorldError> {
        self.spawn_with_storage(ArchetypeStorageKind::Contiguous, bundle)
    }

    /// Spawns a homogeneous bundle batch using contiguous archetype storage.
    ///
    /// Sparse-only bundles use a component-major path that reserves each sparse
    /// set once and publishes structural metadata once for the complete batch.
    pub fn spawn_batch<I, B>(&mut self, bundles: I) -> Result<Vec<Entity>, WorldError>
    where
        I: IntoIterator<Item = B>,
        B: Bundle,
    {
        self.spawn_batch_with_storage(ArchetypeStorageKind::Contiguous, bundles)
    }

    /// Spawns a homogeneous bundle batch using the requested archetype storage.
    pub fn spawn_batch_with_storage<I, B>(
        &mut self,
        storage: ArchetypeStorageKind,
        bundles: I,
    ) -> Result<Vec<Entity>, WorldError>
    where
        I: IntoIterator<Item = B>,
        B: Bundle,
    {
        let bundles = bundles.into_iter().collect::<Vec<_>>();

        if bundles.is_empty() {
            return Ok(Vec::new());
        }

        let amount = u32::try_from(bundles.len())
            .expect("spawn batch exceeds the maximum supported entity count");
        let entities = self.entities.allocate_many_for_spawn(amount);

        self.spawn_allocated_batch_with_storage(&entities, bundles, storage)?;
        Ok(entities)
    }

    /// Spawns an entity using the requested archetype storage policy.
    pub fn spawn_with_storage<B: Bundle>(
        &mut self,
        storage: ArchetypeStorageKind,
        bundle: B,
    ) -> Result<Entity, WorldError> {
        let tick = self.advance_tick();
        let mut planning = PlanningChanges::default();
        let prepared = self.components.prepare_spawn::<B>(&mut planning)?;
        let entity = self.entities.allocate_for_spawn();

        let location = match bundle.with_items(&mut self.components, |components, items| {
            components.spawn_with_plan(entity, storage, prepared, items, tick, &mut planning)
        }) {
            Ok(location) => location,
            Err(err) => {
                self.entities.abort_spawn(entity);
                return Err(WorldError::Bundle(err));
            }
        };

        self.entities.commit_spawn(entity, location);

        if self.lifecycle_watch.spawned {
            self.emit_lifecycle(Spawned { entity });
        }

        self.world_planning.apply(planning);

        Ok(entity)
    }

    /// Spawns `bundle` into a previously reserved entity handle.
    ///
    /// On failure the reservation is discarded, so no half-spawned handle
    /// survives.
    pub(crate) fn spawn_reserved<B: Bundle>(
        &mut self,
        entity: Entity,
        bundle: B,
    ) -> Result<(), WorldError> {
        self.entities.ensure_reserved(entity)?;

        let tick = self.advance_tick();
        let mut planning = PlanningChanges::default();
        let prepared = match self.components.prepare_spawn::<B>(&mut planning) {
            Ok(prepared) => prepared,
            Err(error) => {
                let _ = self.entities.discard_reserved(entity);
                return Err(error.into());
            }
        };

        let location = match bundle.with_items(&mut self.components, |components, items| {
            components.spawn_with_plan(
                entity,
                ArchetypeStorageKind::Contiguous,
                prepared,
                items,
                tick,
                &mut planning,
            )
        }) {
            Ok(location) => location,
            Err(err) => {
                let _ = self.entities.discard_reserved(entity);
                return Err(WorldError::Bundle(err));
            }
        };

        self.entities.activate(entity, location)?;

        if self.lifecycle_watch.spawned {
            self.emit_lifecycle(Spawned { entity });
        }

        self.world_planning.apply(planning);

        Ok(())
    }

    /// Releases `entity`: discards a pending reservation, or despawns the
    /// entity when it is already live.
    pub(crate) fn discard_reserved_or_despawn(&mut self, entity: Entity) {
        if self.entities.is_reserved(entity) {
            let _ = self.entities.discard_reserved(entity);
        } else {
            let _ = self.despawn(entity);
        }
    }

    /// Removes one component bundle from `entity`, migrating it down.
    pub fn remove_bundle<B: BundleInfo>(&mut self, entity: Entity) -> Result<(), WorldError> {
        self.advance_tick();
        let mut planning = PlanningChanges::default();

        let moved = {
            let components = &mut self.components;
            let entities = &mut self.entities;

            let location = entities
                .location_mut(entity)
                .ok_or(WorldError::EntityNotFound(entity))?;

            components.remove_bundle::<B>(entity, location, &mut planning)?
        };

        self.apply_moved_entity(moved);
        self.world_planning.apply(planning);

        Ok(())
    }

    /// Keeps only the components in `B`, removing every other one.
    pub fn retain_bundle<B: BundleInfo>(&mut self, entity: Entity) -> Result<(), WorldError> {
        self.advance_tick();
        let mut planning = PlanningChanges::default();

        let moved = {
            let components = &mut self.components;
            let entities = &mut self.entities;

            let location = entities
                .location_mut(entity)
                .ok_or(WorldError::EntityNotFound(entity))?;

            components.retain_bundle::<B>(entity, location, &mut planning)?
        };

        self.apply_moved_entity(moved);
        self.world_planning.apply(planning);

        Ok(())
    }

    /// Removes every component from `entity` without despawning it.
    pub fn clear_entity(&mut self, entity: Entity) -> Result<(), WorldError> {
        self.retain_bundle::<()>(entity)
    }

    /// Registers the hierarchy components (`Parent` and `Children`).
    ///
    /// Call once before using [`Self::add_child`]. Scalar despawn cleanup
    /// works in any world where these components are registered.
    pub fn init_hierarchy(&mut self) -> Result<(), WorldError> {
        self.register_table::<Parent>()?;
        self.register_table::<Children>()?;
        Ok(())
    }

    /// Returns the parent of `entity`, if any.
    pub fn parent(&self, entity: Entity) -> Option<Entity> {
        self.get::<Parent>(entity).map(|parent| parent.0)
    }

    /// Returns the live children of `entity`, if any.
    pub fn children(&self, entity: Entity) -> Option<&[Entity]> {
        self.get::<Children>(entity).map(|children| children.get())
    }

    /// Adds `child` to `parent`, reparenting it if it already had one.
    ///
    /// Rejects unknown handles, self-parenting, and cycles.
    pub fn add_child(&mut self, parent: Entity, child: Entity) -> Result<(), WorldError> {
        if parent == child || !self.contains(parent) || !self.contains(child) {
            return Err(WorldError::EntityNotFound(child));
        }

        // Adding `child` under `parent` creates a cycle when `parent`
        // already hangs below `child`: walk the ancestor chain to reject it.
        let mut ancestor = parent;
        while let Some(current) = self.get::<Parent>(ancestor).map(|parent| parent.0) {
            if current == child {
                return Err(WorldError::EntityNotFound(child));
            }

            ancestor = current;
        }

        // Detach from the previous parent, if any.
        if let Some(previous) = self.get::<Parent>(child).map(|parent| parent.0) {
            if previous != parent {
                if let Some(mut siblings) = self.get_mut::<Children>(previous) {
                    siblings.0.retain(|&sibling| sibling != child);
                }
            }
        }

        self.insert(child, (Parent(parent),))?;

        if let Some(mut children) = self.get_mut::<Children>(parent) {
            children.0.push(child);
        } else {
            self.insert(parent, (Children(vec![child]),))?;
        }

        Ok(())
    }

    /// Despawns every descendant of `entity` without despawning `entity`.
    pub fn despawn_descendants(&mut self, entity: Entity) -> Result<(), WorldError> {
        let mut descendants = Vec::new();
        let mut queue = VecDeque::from([entity]);

        while let Some(current) = queue.pop_front() {
            // Batch despawns do not perform hierarchy cleanup, so parent
            // child lists can name already-dead handles: skip them.
            if !self.contains(current) {
                continue;
            }

            let children = self
                .get::<Children>(current)
                .map(|children| children.0.clone())
                .unwrap_or_default();

            for child in children {
                if self.contains(child) {
                    queue.push_back(child);
                    descendants.push(child);
                }
            }
        }

        if descendants.is_empty() {
            return Ok(());
        }

        self.despawn_batch(descendants)?;
        Ok(())
    }

    /// Despawns `entity` together with every descendant.
    pub fn despawn_with_descendants(&mut self, entity: Entity) -> Result<(), WorldError> {
        self.despawn_descendants(entity)?;
        self.despawn(entity)
    }

    /// Despawns `entity` and removes all of its components.
    pub fn despawn(&mut self, entity: Entity) -> Result<(), WorldError> {
        // Hierarchy cleanup runs first, while `entity` and its relatives are
        // fully readable. Removing a child's `Parent` component can migrate
        // rows, so the storage location is resolved after this point.
        if let Some(parent) = self.get::<Parent>(entity).map(|parent| parent.0) {
            if let Some(mut children) = self.get_mut::<Children>(parent) {
                children.0.retain(|&child| child != entity);
            }
        }

        if let Some(children) = self.get::<Children>(entity) {
            let orphans = children.0.clone();
            for orphan in orphans {
                let _ = self.remove::<Parent>(orphan);
            }
        }

        let location = self
            .entities
            .location(entity)
            .ok_or(WorldError::EntityNotFound(entity))?;

        let mut planning = PlanningChanges::default();

        if location.archetype_id.is_valid() {
            let (moved, rows) = {
                let archetype = self
                    .components
                    .get_archetype_mut(location.archetype_id)
                    .ok_or(WorldError::EntityNotFound(entity))?;
                let moved = archetype.remove_row(location.storage, location.row as usize)?;

                (moved, archetype.len())
            };

            self.apply_moved_entity(moved);

            planning.mark_archetype_rows(location.archetype_id, rows);
        }

        self.components
            .remove_sparse_components(entity, &mut planning);
        if self.lifecycle_watch.despawned {
            self.emit_lifecycle(Despawned { entity });
        }

        self.entities.free_alive(entity)?;

        self.world_planning.apply(planning);

        Ok(())
    }

    /// Resolves component metadata for repeated access in this world.
    #[inline]
    pub fn component_handle<C: Component>(&self) -> Option<ComponentHandle<C>> {
        let component = self.components.id_of::<C>()?;
        let storage = self.components.storage_of(component)?;

        Some(ComponentHandle {
            world_id: self.id,
            component,
            storage,
            _marker: PhantomData,
        })
    }

    /// Reads `C` from the storage backend named by the resolved ids,
    /// skipping entities whose table row is not live.
    #[inline]
    fn get_resolved<C: Component>(
        &self,
        entity: Entity,
        component: ComponentId,
        storage: StorageKind,
    ) -> Option<&C> {
        match storage {
            StorageKind::Sparse => self
                .components
                .sparse_set_by_id::<C>(component)?
                .get_ref(entity),
            StorageKind::Table => {
                let location = self.entities.location(entity)?;
                let archetype = self.components.get_archetype(location.archetype_id)?;
                let row = location.row as usize;

                if !archetype.is_row_live(location.storage, row) {
                    return None;
                }

                let (ptr, len) = unsafe {
                    archetype
                        .get_column_read_ptrs::<C>(location.storage, component)
                        .ok()?
                };

                if row >= len {
                    return None;
                }

                Some(unsafe { &*ptr.add(row) })
            }
        }
    }

    /// Returns an immutable reference to `entity`'s `C` component.
    pub fn get<C: Component>(&self, entity: Entity) -> Option<&C> {
        let component = self.components.id_of::<C>()?;
        let storage = self.components.storage_of(component)?;
        self.get_resolved(entity, component, storage)
    }

    /// Returns a component using metadata resolved by [`World::component_handle`].
    #[inline]
    pub fn get_with<C: Component>(&self, entity: Entity, handle: ComponentHandle<C>) -> Option<&C> {
        if handle.world_id != self.id {
            return None;
        }

        self.get_resolved(entity, handle.component, handle.storage)
    }

    /// Returns mutable access to `C` through the resolved storage ids,
    /// advancing the world tick and latching the column for change
    /// detection.
    #[inline]
    fn get_mut_resolved<C: Component>(
        &mut self,
        entity: Entity,
        component: ComponentId,
        storage: StorageKind,
    ) -> Option<Mut<'_, C>> {
        let change_tick = self.advance_tick();

        match storage {
            StorageKind::Sparse => {
                let (value, ticks) = self
                    .components
                    .sparse_set_mut_by_id::<C>(component)?
                    .get_mut(entity)?;
                Some(unsafe { Mut::new(value, ticks, change_tick) })
            }
            StorageKind::Table => {
                let location = self.entities.location(entity)?;
                let archetype = self.components.get_archetype_mut(location.archetype_id)?;
                let row = location.row as usize;

                if !archetype.is_row_live(location.storage, row) {
                    return None;
                }

                // Handing out write access may mark ticks through `Mut`,
                // so the column's change-detection latch must observe it.
                unsafe {
                    archetype.note_column_write_access(location.storage, component, change_tick)
                };

                let (value, ticks) = unsafe {
                    archetype
                        .get_component_mut_ptrs::<C>(location.storage, component, row)
                        .ok()?
                };

                Some(unsafe { Mut::new(value, ticks, change_tick) })
            }
        }
    }

    /// Returns a mutable reference to `entity`'s `C` component.
    pub fn get_mut<C: Component>(&mut self, entity: Entity) -> Option<Mut<'_, C>> {
        let component = self.components.id_of::<C>()?;
        let storage = self.components.storage_of(component)?;
        self.get_mut_resolved(entity, component, storage)
    }

    /// Returns mutable component access using pre-resolved metadata.
    #[inline]
    pub fn get_mut_with<C: Component>(
        &mut self,
        entity: Entity,
        handle: ComponentHandle<C>,
    ) -> Option<Mut<'_, C>> {
        if handle.world_id != self.id {
            return None;
        }

        self.get_mut_resolved(entity, handle.component, handle.storage)
    }

    /// Removes one component from `entity`.
    pub fn remove<C: Component>(&mut self, entity: Entity) -> Result<(), WorldError> {
        self.advance_tick();
        let mut planning = PlanningChanges::default();

        let moved = {
            let components = &mut self.components;
            let entities = &mut self.entities;

            let location = entities
                .location_mut(entity)
                .ok_or(WorldError::EntityNotFound(entity))?;

            components.remove::<C>(entity, location, &mut planning)?
        };

        self.apply_moved_entity(moved);
        self.world_planning.apply(planning);

        Ok(())
    }
}

impl World {
    /// Returns an immutable entity view when `entity` is alive.
    pub fn entity(&self, entity: Entity) -> Option<EntityRef<'_>> {
        if self.contains(entity) {
            Some(EntityRef {
                world: self,
                entity,
            })
        } else {
            None
        }
    }

    /// Returns a mutable entity view when `entity` is alive.
    pub fn entity_mut(&mut self, entity: Entity) -> Option<EntityMut<'_>> {
        if self.contains(entity) {
            Some(EntityMut {
                world: self,
                entity,
            })
        } else {
            None
        }
    }
}
/// Immutable view over a single live entity.
pub struct EntityRef<'w> {
    world: &'w World,
    entity: Entity,
}

impl<'w> EntityRef<'w> {
    /// Returns the entity identifier represented by this view.
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Returns an immutable component reference from this entity.
    pub fn get<C: Component>(&self) -> Option<&'w C> {
        self.world.get::<C>(self.entity)
    }
}

/// Mutable view over a single live entity.
pub struct EntityMut<'w> {
    world: &'w mut World,
    entity: Entity,
}

impl<'w> EntityMut<'w> {
    /// Returns the entity identifier represented by this view.
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Returns an immutable component reference from this entity.
    pub fn get<C: Component>(&self) -> Option<&C> {
        self.world.get::<C>(self.entity)
    }

    /// Returns a mutable component reference from this entity.
    pub fn get_mut<C: Component>(&mut self) -> Option<Mut<'_, C>> {
        self.world.get_mut::<C>(self.entity)
    }

    /// Applies `bundle` to this entity.
    pub fn insert<B: Bundle>(&mut self, bundle: B) -> Result<(), WorldError> {
        self.world.insert(self.entity, bundle)
    }

    /// Despawns this entity.
    pub fn despawn(self) -> Result<(), WorldError> {
        self.world.despawn(self.entity)
    }

    /// Removes one component from the entity this view wraps.
    pub fn remove<C: Component>(&mut self) -> Result<(), WorldError> {
        self.world.remove::<C>(self.entity)
    }
}

#[cfg(test)]
mod tests;
