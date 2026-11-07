//! Reusable metadata for repeated scalar structural operations.

use std::{any::TypeId, fmt, marker::PhantomData};

use crate::{
    change_detection::ChangeTick,
    component::{
        archetype::{
            plan::{
                spawn::PreparedSpawn as ComponentPreparedSpawn,
                transition::{PreparedTransitionPlan, StructuralPlanId},
            },
            ArchetypeId,
        },
        Bundle, BundleError, BundleInfo, BundleLayoutId, Component, ComponentId, MovedEntity,
        StorageKind,
    },
    entity::{metadata::EntityLocation, ArchetypeStorageKind, Entity},
    planning::PlanningChanges,
};

use super::{World, WorldError, WorldId};

/// Prepared metadata for repeatedly spawning the same bundle type.
///
/// The handle resolves the bundle layout, destination archetype, and column
/// mapping once. It is tied to the [`World`] that created it because every
/// cached identifier is world-local.
#[must_use = "a prepared spawn has no effect until World::spawn_prepared uses it"]
pub struct PreparedSpawn<B: BundleInfo> {
    world_id: WorldId,
    prepared: ComponentPreparedSpawn,
    storage: ArchetypeStorageKind,
    _marker: PhantomData<fn() -> B>,
}

impl<B: BundleInfo> Copy for PreparedSpawn<B> {}

impl<B: BundleInfo> Clone for PreparedSpawn<B> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<B: BundleInfo> PreparedSpawn<B> {
    /// Return this prepared spawn belongs to.
    #[inline]
    pub const fn world_id(&self) -> WorldId {
        self.world_id
    }

    /// Returns the archetype storage selected for new entities.
    #[inline]
    pub const fn storage(&self) -> ArchetypeStorageKind {
        self.storage
    }
}

impl<B: BundleInfo> fmt::Debug for PreparedSpawn<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedSpawn")
            .field("world_id", &self.world_id)
            .field("storage", &self.storage)
            .finish_non_exhaustive()
    }
}

/// Prepared metadata for repeatedly inserting the same bundle type.
///
/// The handle resolves the bundle layout once and lazily retains one compiled
/// transition per source archetype. It is tied to the [`World`] that created it
/// because component and archetype identifiers are world-local.
#[must_use = "a prepared insertion has no effect until World::insert_prepared uses it"]
pub struct PreparedInsert<B: BundleInfo> {
    world_id: WorldId,
    layout: BundleLayoutId,
    /// Lazily compiled transitions, indexed densely by `ArchetypeId::index`;
    /// unvisited archetypes remain `None`.
    transitions: Vec<Option<PreparedTransitionPlan>>,
    _marker: PhantomData<fn() -> B>,
}

impl<B: BundleInfo> PreparedInsert<B> {
    /// Return the world this handle was created against.
    #[inline]
    pub const fn world_id(&self) -> WorldId {
        self.world_id
    }

    /// Return the number of archetypes that have a cached transition.
    #[inline]
    pub fn cached_archetype_count(&self) -> usize {
        self.transitions.iter().flatten().count()
    }

    /// Returns the cached transition for `source`, if compiled.
    #[inline]
    fn transition(&self, source: ArchetypeId) -> Option<PreparedTransitionPlan> {
        self.transitions.get(source.index()).copied().flatten()
    }

    /// Caches the transition compiled for `source`.
    #[inline]
    fn cache_transition(&mut self, source: ArchetypeId, plan: PreparedTransitionPlan) {
        let index = source.index();
        if self.transitions.len() <= index {
            self.transitions.resize(index + 1, None);
        }
        self.transitions[index] = Some(plan);
    }
}

impl<B: BundleInfo> fmt::Debug for PreparedInsert<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedInsert")
            .field("world_id", &self.world_id)
            .field("cached_archetypes", &self.cached_archetype_count())
            .finish_non_exhaustive()
    }
}

/// Prepared metadata for repeatedly removing one component type.
///
/// Sparse components use their already resolved component id directly. Table
/// components additionally retain one compiled structural edge per source
/// archetype.
#[must_use = "a prepared removal has no effect until World::remove_prepared uses it"]
pub struct PreparedRemove<C: Component> {
    world_id: WorldId,
    component: ComponentId,
    storage: StorageKind,
    /// Lazily compiled structural edges, indexed densely by
    /// `ArchetypeId::index`; unvisited archetypes remain `None`.
    transitions: Vec<Option<StructuralPlanId>>,
    _marker: PhantomData<fn() -> C>,
}

impl<C: Component> PreparedRemove<C> {
    /// Returns the resolved component id belongs to.
    #[inline]
    pub const fn world_id(&self) -> WorldId {
        self.world_id
    }

    /// Returns the resolved component id.
    #[inline]
    pub const fn component_id(&self) -> ComponentId {
        self.component
    }

    /// Returns the resolved component storage backend.
    #[inline]
    pub const fn storage(&self) -> StorageKind {
        self.storage
    }

    /// Returns the archetypes with a compiled structural edge so far.
    #[inline]
    pub fn cached_archetype_count(&self) -> usize {
        self.transitions.iter().flatten().count()
    }

    /// Returns the cached structural transition for `source`, if compiled.
    #[inline]
    fn transition(&self, source: ArchetypeId) -> Option<StructuralPlanId> {
        self.transitions.get(source.index()).copied().flatten()
    }

    /// Caches the structural transition compiled for `source`.
    #[inline]
    fn cache_transition(&mut self, source: ArchetypeId, plan: StructuralPlanId) {
        let index = source.index();
        if self.transitions.len() <= index {
            self.transitions.resize(index + 1, None);
        }
        self.transitions[index] = Some(plan);
    }
}

impl<C: Component> fmt::Debug for PreparedRemove<C> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PreparedRemove")
            .field("world_id", &self.world_id)
            .field("component", &self.component)
            .field("storage", &self.storage)
            .field("cached_archetypes", &self.cached_archetype_count())
            .finish()
    }
}

/// Exclusive session for repeated prepared spawns.
///
/// Each spawned entity becomes live immediately. Query-planning changes are
/// accumulated and published once when the session is finished or dropped,
/// and every spawn in the session shares one change tick.
#[must_use = "the session keeps the world exclusively borrowed until it is finished or dropped"]
pub struct PreparedSpawnSession<'a, B: Bundle> {
    world: &'a mut World,
    prepared: ComponentPreparedSpawn,
    storage: ArchetypeStorageKind,
    planning: PlanningChanges,
    /// Change tick shared by every spawn in the session, advanced on first use.
    tick: Option<ChangeTick>,
    operations: usize,
    /// Guards against publishing planning twice from `finish` plus `drop`.
    committed: bool,
    _marker: PhantomData<fn() -> B>,
}

impl<B: Bundle> PreparedSpawnSession<'_, B> {
    /// Spawns one bundle while retaining shared bookkeeping in the session.
    pub fn spawn(&mut self, bundle: B) -> Result<Entity, WorldError> {
        let tick = match self.tick {
            Some(tick) => tick,
            None => {
                let tick = self.world.advance_tick();
                self.tick = Some(tick);
                tick
            }
        };
        let entity = self.world.entities.allocate_for_spawn();

        let location = match bundle.with_items(&mut self.world.components, |components, items| {
            components.spawn_with_plan(
                entity,
                self.storage,
                self.prepared,
                items,
                tick,
                &mut self.planning,
            )
        }) {
            Ok(location) => location,
            Err(error) => {
                self.world.entities.abort_spawn(entity);
                return Err(WorldError::Bundle(error));
            }
        };

        self.world.entities.commit_spawn(entity, location);
        self.operations += 1;
        Ok(entity)
    }

    /// Returns the number of completed spawns in this session.
    #[inline]
    pub const fn len(&self) -> usize {
        self.operations
    }

    /// Returns whether the session has completed no spawns.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.operations == 0
    }

    /// Publishes deferred planning metadata and returns the number of spawns.
    pub fn finish(mut self) -> usize {
        self.commit();
        self.operations
    }

    /// Applies accumulated planning changes once; idempotent, so `finish` and
    /// `drop` cannot double-publish.
    fn commit(&mut self) {
        if self.committed {
            return;
        }

        self.world
            .world_planning
            .apply(std::mem::take(&mut self.planning));
        self.committed = true;
    }
}

impl<B: Bundle> Drop for PreparedSpawnSession<'_, B> {
    fn drop(&mut self) {
        self.commit();
    }
}

/// Exclusive session for repeated prepared insertions.
///
/// Entity locations remain valid after every call, while sparse-driver repairs
/// and query-planning changes are published once when the session is finished
/// or dropped. All insertions in the session share one change tick.
#[must_use = "the session keeps the world exclusively borrowed until it is finished or dropped"]
pub struct PreparedInsertSession<'a, B: Bundle> {
    world: &'a mut World,
    prepared: &'a mut PreparedInsert<B>,
    planning: PlanningChanges,
    /// Sparse-driver location repairs for transitioned or displaced entities,
    /// deferred until commit so intermediate locations never need repairing.
    location_repairs: Vec<(Entity, EntityLocation)>,
    /// Change tick shared by every insertion in the session, advanced on first
    /// use.
    tick: Option<ChangeTick>,
    operations: usize,
    /// Guards against publishing repairs and planning twice from `finish`
    /// plus `drop`.
    committed: bool,
}

impl<B: Bundle> PreparedInsertSession<'_, B> {
    /// Inserts one bundle while retaining structural bookkeeping in the session.
    pub fn insert(&mut self, entity: Entity, bundle: B) -> Result<(), WorldError> {
        let previous = self
            .world
            .entities
            .location(entity)
            .ok_or(WorldError::EntityNotFound(entity))?;
        let source = previous.archetype_id;
        let tick = match self.tick {
            Some(tick) => tick,
            None => {
                let tick = self.world.advance_tick();
                self.tick = Some(tick);
                tick
            }
        };

        let transition = match self.prepared.transition(source) {
            Some(transition) => transition,
            None => {
                let transition = self.world.components.prepare_transition_plan(
                    source,
                    self.prepared.layout,
                    &mut self.planning,
                )?;
                self.prepared.cache_transition(source, transition);
                transition
            }
        };

        let mut current = previous;

        let moved = {
            let components = &mut self.world.components;
            let entities = &mut self.world.entities;

            bundle.with_items(components, |components, items| {
                let location = entities
                    .location_mut(entity)
                    .expect("session insertion entity was prevalidated");

                let result = components.apply_bundle_prepared_deferred_location(
                    entity,
                    location,
                    transition,
                    items,
                    tick,
                    &mut self.planning,
                );

                current = *location;
                result
            })?
        };

        if current != previous && self.world.components.entity_has_sparse_components(entity) {
            self.location_repairs.push((entity, current));
        }

        self.publish_moved_location(moved);
        self.operations += 1;
        Ok(())
    }

    /// Returns the number of completed insertions in this session.
    #[inline]
    pub const fn len(&self) -> usize {
        self.operations
    }

    /// Returns whether the session has completed no insertions.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.operations == 0
    }

    /// Publishes deferred metadata and returns the number of insertions.
    pub fn finish(mut self) -> usize {
        self.commit();
        self.operations
    }

    /// Publishes a displaced entity's location, queueing sparse repairs.
    #[inline]
    fn publish_moved_location(&mut self, moved: Option<MovedEntity>) {
        if let Some(moved) = moved {
            self.world
                .entities
                .set_location(moved.entity, moved.location)
                .expect("moved entity must still be alive");
            if self
                .world
                .components
                .entity_has_sparse_components(moved.entity)
            {
                self.location_repairs.push((moved.entity, moved.location));
            }
        }
    }

    /// Publishes sparse repairs and accumulated planning changes once;
    /// idempotent, so `finish` and `drop` cannot double-publish.
    fn commit(&mut self) {
        if self.committed {
            return;
        }

        self.world
            .repair_sparse_entity_locations(self.location_repairs.drain(..));
        self.world
            .world_planning
            .apply(std::mem::take(&mut self.planning));
        self.committed = true;
    }
}

impl<B: Bundle> Drop for PreparedInsertSession<'_, B> {
    fn drop(&mut self) {
        self.commit();
    }
}

/// Exclusive session for repeated prepared component removals.
///
/// Table migrations share deferred sparse-driver repair and one planning
/// publication. Sparse removals reuse the resolved component storage directly.
#[must_use = "the session keeps the world exclusively borrowed until it is finished or dropped"]
pub struct PreparedRemoveSession<'a, C: Component> {
    world: &'a mut World,
    prepared: &'a mut PreparedRemove<C>,
    planning: PlanningChanges,
    /// Sparse-driver location repairs for migrated entities, deferred until
    /// commit.
    location_repairs: Vec<(Entity, EntityLocation)>,
    /// Whether the world tick was already advanced for this session. Removals
    /// write no component data, so only the fact that the tick moved is shared.
    tick_advanced: bool,
    operations: usize,
    /// Guards against publishing repairs and planning twice from `finish`
    /// plus `drop`.
    committed: bool,
}

impl<C: Component> PreparedRemoveSession<'_, C> {
    /// Removes one component while retaining structural bookkeeping in the session.
    pub fn remove(&mut self, entity: Entity) -> Result<(), WorldError> {
        let previous = self
            .world
            .entities
            .location(entity)
            .ok_or(WorldError::EntityNotFound(entity))?;

        if !self.tick_advanced {
            self.world.advance_tick();
            self.tick_advanced = true;
        }

        let mut current = previous;

        let moved = match self.prepared.storage {
            StorageKind::Sparse => {
                self.world.components.remove_sparse_component_prepared(
                    self.prepared.component,
                    entity,
                    &mut self.planning,
                );
                None
            }
            StorageKind::Table => {
                let source = previous.archetype_id;
                let transition = match self.prepared.transition(source) {
                    Some(transition) => transition,
                    None => {
                        let transition =
                            self.world.components.prepare_remove_component_transition(
                                source,
                                self.prepared.component,
                                &mut self.planning,
                            )?;
                        self.prepared.cache_transition(source, transition);
                        transition
                    }
                };

                let components = &mut self.world.components;
                let entities = &mut self.world.entities;
                let location = entities
                    .location_mut(entity)
                    .expect("session removal entity was prevalidated");

                let result = components.execute_prepared_structural_transition_deferred_location(
                    entity,
                    transition,
                    location,
                    &mut self.planning,
                );

                current = *location;
                result?
            }
        };

        if current != previous && self.world.components.entity_has_sparse_components(entity) {
            self.location_repairs.push((entity, current));
        }

        self.publish_moved_location(moved);
        self.operations += 1;
        Ok(())
    }

    /// Returns the number of completed removals in this session.
    #[inline]
    pub const fn len(&self) -> usize {
        self.operations
    }

    /// Returns whether the session has completed no removals.
    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.operations == 0
    }

    /// Publishes deferred metadata and returns the number of removals.
    pub fn finish(mut self) -> usize {
        self.commit();
        self.operations
    }

    /// Publishes a displaced entity's location, queueing sparse repairs.
    #[inline]
    fn publish_moved_location(&mut self, moved: Option<MovedEntity>) {
        if let Some(moved) = moved {
            self.world
                .entities
                .set_location(moved.entity, moved.location)
                .expect("moved entity must still be alive");
            if self
                .world
                .components
                .entity_has_sparse_components(moved.entity)
            {
                self.location_repairs.push((moved.entity, moved.location));
            }
        }
    }

    /// Publishes sparse repairs and accumulated planning changes once;
    /// idempotent, so `finish` and `drop` cannot double-publish.
    fn commit(&mut self) {
        if self.committed {
            return;
        }

        self.world
            .repair_sparse_entity_locations(self.location_repairs.drain(..));
        self.world
            .world_planning
            .apply(std::mem::take(&mut self.planning));
        self.committed = true;
    }
}

impl<C: Component> Drop for PreparedRemoveSession<'_, C> {
    fn drop(&mut self) {
        self.commit();
    }
}

impl World {
    /// Resolves a contiguous spawn plan for repeated use.
    pub fn prepare_spawn<B: BundleInfo>(&mut self) -> Result<PreparedSpawn<B>, WorldError> {
        self.prepare_spawn_with_storage(ArchetypeStorageKind::Contiguous)
    }

    /// Resolves a spawn plan and fixes its archetype storage for repeated use.
    pub fn prepare_spawn_with_storage<B: BundleInfo>(
        &mut self,
        storage: ArchetypeStorageKind,
    ) -> Result<PreparedSpawn<B>, WorldError> {
        let mut planning = PlanningChanges::default();
        let prepared = self.components.prepare_spawn::<B>(&mut planning)?;
        self.world_planning.apply(planning);

        Ok(PreparedSpawn {
            world_id: self.id,
            prepared,
            storage,
            _marker: PhantomData,
        })
    }

    /// Spawns one bundle using a previously resolved plan.
    pub fn spawn_prepared<B: Bundle>(
        &mut self,
        bundle: B,
        prepared: &PreparedSpawn<B>,
    ) -> Result<Entity, WorldError> {
        self.validate_world_id(prepared.world_id)?;

        let tick = self.advance_tick();
        let mut planning = PlanningChanges::default();
        let entity = self.entities.allocate_for_spawn();

        let location = match bundle.with_items(&mut self.components, |components, items| {
            components.spawn_with_plan(
                entity,
                prepared.storage,
                prepared.prepared,
                items,
                tick,
                &mut planning,
            )
        }) {
            Ok(location) => location,
            Err(error) => {
                self.entities.abort_spawn(entity);
                return Err(WorldError::Bundle(error));
            }
        };

        self.entities.commit_spawn(entity, location);
        self.world_planning.apply(planning);
        Ok(entity)
    }

    /// Starts an exclusive session for repeated prepared spawns.
    pub fn spawn_prepared_session<B: Bundle>(
        &mut self,
        prepared: &PreparedSpawn<B>,
    ) -> Result<PreparedSpawnSession<'_, B>, WorldError> {
        self.validate_world_id(prepared.world_id)?;

        Ok(PreparedSpawnSession {
            world: self,
            prepared: prepared.prepared,
            storage: prepared.storage,
            planning: PlanningChanges::default(),
            tick: None,
            operations: 0,
            committed: false,
            _marker: PhantomData,
        })
    }

    /// Resolves a bundle layout for repeated scalar insertion.
    pub fn prepare_insert<B: BundleInfo>(&mut self) -> Result<PreparedInsert<B>, WorldError> {
        let layout = self.components.ensure_bundle_layout::<B>()?;

        Ok(PreparedInsert {
            world_id: self.id,
            layout,
            transitions: Vec::new(),
            _marker: PhantomData,
        })
    }

    /// Inserts a bundle using reusable layout and source-archetype metadata.
    ///
    /// The first entity observed in each source archetype prepares its outgoing
    /// transition. Later entities from that archetype use a dense cache lookup.
    pub fn insert_prepared<B: Bundle>(
        &mut self,
        entity: Entity,
        bundle: B,
        prepared: &mut PreparedInsert<B>,
    ) -> Result<(), WorldError> {
        self.validate_world_id(prepared.world_id)?;

        let source = self
            .entities
            .location(entity)
            .ok_or(WorldError::EntityNotFound(entity))?
            .archetype_id;
        let tick = self.advance_tick();
        let mut planning = PlanningChanges::default();

        let transition = match prepared.transition(source) {
            Some(transition) => transition,
            None => {
                let transition = self.components.prepare_transition_plan(
                    source,
                    prepared.layout,
                    &mut planning,
                )?;
                prepared.cache_transition(source, transition);
                transition
            }
        };

        let moved = {
            let components = &mut self.components;
            let entities = &mut self.entities;

            bundle.with_items(components, |components, items| {
                let location = entities
                    .location_mut(entity)
                    .expect("prepared insertion entity was prevalidated");

                components.apply_bundle_prepared(
                    entity,
                    location,
                    transition,
                    items,
                    tick,
                    &mut planning,
                )
            })?
        };

        self.apply_moved_entity(moved);
        self.world_planning.apply(planning);
        Ok(())
    }

    /// Starts an exclusive session for repeated prepared insertions.
    pub fn insert_prepared_session<'a, B: Bundle>(
        &'a mut self,
        prepared: &'a mut PreparedInsert<B>,
    ) -> Result<PreparedInsertSession<'a, B>, WorldError> {
        self.validate_world_id(prepared.world_id)?;

        Ok(PreparedInsertSession {
            world: self,
            prepared,
            planning: PlanningChanges::default(),
            location_repairs: Vec::new(),
            tick: None,
            operations: 0,
            committed: false,
        })
    }

    /// Resolves component metadata for repeated scalar removal.
    pub fn prepare_remove<C: Component>(&self) -> Result<PreparedRemove<C>, WorldError> {
        let component = self
            .components
            .id_of::<C>()
            .ok_or(BundleError::ComponentNotRegistered(TypeId::of::<C>()))?;
        let storage = self
            .components
            .storage_of(component)
            .ok_or(BundleError::StorageUnknown(component))?;

        Ok(PreparedRemove {
            world_id: self.id,
            component,
            storage,
            transitions: Vec::new(),
            _marker: PhantomData,
        })
    }

    /// Removes a component using reusable component and transition metadata.
    ///
    /// Table transitions are cached independently for every source archetype;
    /// sparse removals bypass type and storage lookup entirely.
    pub fn remove_prepared<C: Component>(
        &mut self,
        entity: Entity,
        prepared: &mut PreparedRemove<C>,
    ) -> Result<(), WorldError> {
        self.validate_world_id(prepared.world_id)?;

        let source = self
            .entities
            .location(entity)
            .ok_or(WorldError::EntityNotFound(entity))?
            .archetype_id;
        self.advance_tick();
        let mut planning = PlanningChanges::default();

        let moved = match prepared.storage {
            StorageKind::Sparse => {
                self.components.remove_sparse_component_prepared(
                    prepared.component,
                    entity,
                    &mut planning,
                );
                None
            }
            StorageKind::Table => {
                let transition = match prepared.transition(source) {
                    Some(transition) => transition,
                    None => {
                        let transition = self.components.prepare_remove_component_transition(
                            source,
                            prepared.component,
                            &mut planning,
                        )?;
                        prepared.cache_transition(source, transition);
                        transition
                    }
                };

                let components = &mut self.components;
                let entities = &mut self.entities;
                let location = entities
                    .location_mut(entity)
                    .ok_or(WorldError::EntityNotFound(entity))?;

                components.execute_prepared_structural_transition(
                    entity,
                    transition,
                    location,
                    &mut planning,
                )?
            }
        };

        self.apply_moved_entity(moved);
        self.world_planning.apply(planning);
        Ok(())
    }

    /// Starts an exclusive session for repeated prepared removals.
    pub fn remove_prepared_session<'a, C: Component>(
        &'a mut self,
        prepared: &'a mut PreparedRemove<C>,
    ) -> Result<PreparedRemoveSession<'a, C>, WorldError> {
        self.validate_world_id(prepared.world_id)?;

        Ok(PreparedRemoveSession {
            world: self,
            prepared,
            planning: PlanningChanges::default(),
            location_repairs: Vec::new(),
            tick_advanced: false,
            operations: 0,
            committed: false,
        })
    }
}
