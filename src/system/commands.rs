//! Deferred operations over a [`World`].
//!
//! Commands are recorded while a system executes its queries and applied at an
//! exclusive schedule barrier. Entities created during recording remain reserved
//! until their creation command is applied or discarded.

use std::{
    any::{Any, TypeId},
    fmt,
};

use crate::{
    __private::{Bundle, BundleInfo},
    component::Component,
    entity::{Entity, EntityAllocator, EntityError},
    resource::{Resource, ResourceError, ResourceId},
    system::{param::SystemParam, SystemAccess},
    world::{execution_world::ExecutionWorld, World, WorldError},
};

/// Owned command closure applied once against the world.
type BoxedCommand = Box<dyn FnOnce(&mut World) -> Result<(), WorldError> + Send + 'static>;

/// Owned closure that receives the world plus the entity batch it reserved.
type BoxedReservedCommand =
    Box<dyn FnOnce(&mut World, &[Entity]) -> Result<(), WorldError> + Send + 'static>;

/// Minimum adjacent despawns a fused run needs before using `despawn_batch`.
///
/// The value matches the point at which the world's despawn batch starts
/// transposing sparse membership by component instead of executing an
/// entity-major pass; shorter runs stay on the low-overhead scalar path.
const MIN_FUSED_DESPAWN_COMMANDS: usize = 256;

/// Minimum adjacent homogeneous spawns a fused run needs before batching.
///
/// Runs below this size remain entity-major; larger ones amortize command
/// dispatch and structural planning through the world's batch spawn path.
const MIN_FUSED_SPAWN_COMMANDS: usize = 32;

// TODO: the fused batch paths skip Spawned/Despawned publication; so far only
// the scalar paths below these thresholds publish lifecycle events.

/// Category of deferred command used when reporting errors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CommandKind {
    /// A user-supplied closure.
    Custom,
    /// `init_resource`.
    InitResource,
    /// `get_resource_or_insert_with`.
    GetResourceOrInsert,
    /// `modify_resource` and its variants.
    ModifyResource,
    /// `insert_resource`.
    InsertResource,
    /// `remove_resource`.
    RemoveResource,
    /// Entity reservation for a later command.
    ReserveEntity,
    /// Batch entity reservation.
    ReserveEntities,
    /// Single entity spawn.
    Spawn,
    /// Homogeneous batch spawn.
    SpawnBatch,
    /// `insert` on a single entity.
    Insert,
    /// The same insert applied to a run of entities.
    InsertBatch,
    /// `remove` on a single entity.
    Remove,
    /// `remove` across a batch.
    RemoveBatch,
    /// Bundle removal on one entity.
    RemoveBundle,
    /// Bundle removal across a batch.
    RemoveBundleBatch,
    /// Keeps only the bundle's components on one entity.
    RetainBundle,
    /// Same retention over a batch.
    RetainBundleBatch,
    /// Single entity despawn.
    Despawn,
    /// Despawn of a whole run of entities.
    DespawnBatch,
    /// Component clear on one entity.
    Clear,
    /// Component clear across a batch.
    ClearBatch,
}

impl fmt::Display for CommandKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Custom => "custom",
            Self::InitResource => "init_resource",
            Self::GetResourceOrInsert => "get_resource_or_insert",
            Self::ModifyResource => "modify_resource",
            Self::InsertResource => "insert_resource",
            Self::RemoveResource => "remove_resource",
            Self::ReserveEntity => "reserve_entity",
            Self::ReserveEntities => "reserve_entities",
            Self::Spawn => "spawn",
            Self::SpawnBatch => "spawn_batch",
            Self::Insert => "insert",
            Self::InsertBatch => "insert_batch",
            Self::Remove => "remove",
            Self::RemoveBatch => "remove_batch",
            Self::RemoveBundle => "remove_bundle",
            Self::RemoveBundleBatch => "remove_bundle_batch",
            Self::RetainBundle => "retain_bundle",
            Self::RetainBundleBatch => "retain_bundle_batch",
            Self::Despawn => "despawn",
            Self::DespawnBatch => "despawn_batch",
            Self::Clear => "clear",
            Self::ClearBatch => "clear_batch",
        };

        f.write_str(name)
    }
}

/// Contextual error produced while applying a deferred command.
#[derive(Debug)]
pub struct CommandError {
    index: usize,
    kind: CommandKind,
    entity: Option<Entity>,
    /// Boxed to keep `CommandError` cheap to move through command results.
    source: Box<WorldError>,
}

impl CommandError {
    /// Bundles a world failure with its queue position, kind, and entity.
    fn new(index: usize, kind: CommandKind, entity: Option<Entity>, source: WorldError) -> Self {
        Self {
            index,
            kind,
            entity,
            source: Box::new(source),
        }
    }

    /// Position of the command inside the flushed queue.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Category of the command that failed.
    pub fn kind(&self) -> CommandKind {
        self.kind
    }

    /// Entity the command targeted, when it had one.
    pub fn entity(&self) -> Option<Entity> {
        self.entity
    }

    /// Underlying error reported by the world operation.
    pub fn source(&self) -> &WorldError {
        &self.source
    }
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.entity {
            Some(entity) => write!(
                f,
                "command #{} (`{}`) failed for {entity:?}: {}",
                self.index, self.kind, self.source
            ),
            None => write!(
                f,
                "command #{} (`{}`) failed: {}",
                self.index, self.kind, self.source
            ),
        }
    }
}

impl std::error::Error for CommandError {}

/// Releases reservations that have not yet been transferred to the queue.
///
/// This guard covers immediate bundle construction that depends on a reserved
/// entity. An error or panic during construction does not leave the slot reserved.
struct ReservationGuard<'w> {
    allocator: &'w EntityAllocator,
    /// Reserved entities; `None` once ownership has been taken from the guard.
    entities: Option<Vec<Entity>>,
}

impl<'w> ReservationGuard<'w> {
    /// Arms the guard over `entities` reserved from `allocator`.
    fn new(allocator: &'w EntityAllocator, entities: Vec<Entity>) -> Self {
        Self {
            allocator,
            entities: Some(entities),
        }
    }

    /// Returns the guarded entities; panics if ownership was already taken.
    fn entities(&self) -> &[Entity] {
        self.entities.as_deref().expect("guard already consumed")
    }

    /// Moves the entities out, disarming the guard so drop does not cancel them.
    fn take(mut self) -> Vec<Entity> {
        self.entities.take().expect("guard already consumed")
    }
}

impl Drop for ReservationGuard<'_> {
    /// Cancels every reservation that was never taken.
    fn drop(&mut self) {
        if let Some(entities) = self.entities.take() {
            self.allocator
                .cancel_many_unique(&entities)
                .expect("reservation guard only owns pending entities");
        }
    }
}

/// Type-erased bundle retained as data until a deferred spawn is applied.
///
/// Unlike an opaque command closure, this exposes the concrete bundle type to
/// the queue so adjacent homogeneous spawns can share one batch execution.
trait DeferredSpawnBundle: Send {
    /// Returns the type id of the concrete bundle, used to match runs.
    fn bundle_type_id(&self) -> TypeId;

    /// Erases the box so a homogeneous run can recover the concrete bundle.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send>;

    /// Spawns `entity` in the world carrying this bundle.
    fn apply_one(self: Box<Self>, world: &mut World, entity: Entity) -> Result<(), WorldError>;

    /// Spawns `entities` through one batch call, consuming this bundle plus
    /// the rest of the run; `remaining` always holds the same bundle type.
    fn apply_batch(
        self: Box<Self>,
        world: &mut World,
        entities: &[Entity],
        remaining: Vec<Box<dyn DeferredSpawnBundle>>,
    ) -> Result<(), WorldError>;
}

impl<B> DeferredSpawnBundle for B
where
    B: Bundle + Send + 'static,
{
    #[inline]
    /// Returns the type id of `B`.
    fn bundle_type_id(&self) -> TypeId {
        TypeId::of::<B>()
    }

    #[inline]
    /// Returns the boxed bundle unchanged.
    fn into_any(self: Box<Self>) -> Box<dyn Any + Send> {
        self
    }

    #[inline]
    /// Consumes the box and spawns the reserved entity with the bundle.
    fn apply_one(self: Box<Self>, world: &mut World, entity: Entity) -> Result<(), WorldError> {
        world.spawn_reserved(entity, *self)
    }

    /// Downcasts the run's bundles back to `B` and spawns them as one batch.
    fn apply_batch(
        self: Box<Self>,
        world: &mut World,
        entities: &[Entity],
        remaining: Vec<Box<dyn DeferredSpawnBundle>>,
    ) -> Result<(), WorldError> {
        let mut bundles = Vec::with_capacity(remaining.len() + 1);
        bundles.push(*self);

        for bundle in remaining {
            debug_assert_eq!(bundle.bundle_type_id(), TypeId::of::<B>());

            let bundle = match bundle.into_any().downcast::<B>() {
                Ok(bundle) => bundle,
                Err(_) => unreachable!("a homogeneous spawn run changed bundle type"),
            };
            bundles.push(*bundle);
        }

        world.spawn_reserved_batch(entities, bundles)
    }
}

/// Command stored until application.
///
/// Strict spawns and despawns are stored as plain data instead of opaque
/// closures so the apply phase can recognize adjacent commands of the same
/// shape and fuse them into one batch execution. Everything else is recorded
/// as a type-erased closure with the metadata error reporting needs.
enum QueuedCommand {
    /// A type-erased closure plus its error-reporting metadata.
    Normal {
        kind: CommandKind,
        entity: Option<Entity>,
        command: BoxedCommand,
    },
    /// A strict despawn kept as data so adjacent despawns can fuse.
    Despawn {
        entity: Entity,
    },
    /// A reserved spawn kept as data so homogeneous runs can share one batch.
    Spawn {
        kind: CommandKind,
        entity: Entity,
        bundle: Box<dyn DeferredSpawnBundle>,
    },
    /// A closure that owns its reserved entity batch until it is applied.
    Reserved {
        kind: CommandKind,
        entities: Box<[Entity]>,
        command: BoxedReservedCommand,
    },
}

/// One queued spawn pending its deferred application.
struct DeferredSpawnEntry {
    /// Position of the command inside the flushed queue.
    index: usize,
    /// Reserved handle the bundle will be applied to.
    entity: Entity,
    /// Type-erased bundle moved out of the recording closure.
    bundle: Box<dyn DeferredSpawnBundle>,
}

/// Policy used when an entity is no longer alive before its command runs.
#[derive(Clone, Copy)]
enum MissingEntityPolicy {
    /// The command fails with `EntityNotFound`.
    Error,
    /// The command is skipped silently.
    Ignore,
}

/// Zero-sized marker granting `Commands` its (empty) access declaration.
///
/// Declaring the parameter unique through this marker is what rejects a
/// system that takes two `Commands` parameters, which would alias one queue.
struct CommandsAccessMarker;

/// Owned command buffer drained by the deferred apply phase.
#[derive(Default)]
struct CommandQueue {
    commands: Vec<QueuedCommand>,
}

/// Applying context that drains a queue against exclusive world access,
/// with one-command lookahead for spawn and despawn fusion.
///
/// Dropping the drainer releases the reservations of every command that was
/// not applied, covering both an earlier failure and a panic unwind.
struct PendingCommands<'w> {
    world: &'w mut World,
    /// Remaining commands paired with their queue positions; peekable so the
    /// fusion checks can look one command ahead without consuming it.
    commands: std::iter::Peekable<std::iter::Enumerate<std::vec::IntoIter<QueuedCommand>>>,
}

impl<'w> PendingCommands<'w> {
    /// Takes ownership of the queue for draining.
    fn new(world: &'w mut World, commands: Vec<QueuedCommand>) -> Self {
        Self {
            world,
            commands: commands.into_iter().enumerate().peekable(),
        }
    }

    /// Returns the next command with its original queue position.
    fn next(&mut self) -> Option<(usize, QueuedCommand)> {
        self.commands.next()
    }

    /// Returns whether the following command is a strict despawn.
    #[inline]
    fn next_is_despawn(&mut self) -> bool {
        matches!(
            self.commands.peek(),
            Some((_, QueuedCommand::Despawn { .. }))
        )
    }

    /// Consumes the following despawn command when it is one.
    fn next_despawn(&mut self) -> Option<(usize, Entity)> {
        if !self.next_is_despawn() {
            return None;
        }

        let (index, QueuedCommand::Despawn { entity }) = self.commands.next()? else {
            unreachable!("the next queued command was checked as a despawn")
        };

        Some((index, entity))
    }

    /// Returns whether the following command spawns `bundle_type` of the
    /// same `kind`.
    #[inline]
    fn next_is_spawn(&mut self, kind: CommandKind, bundle_type: TypeId) -> bool {
        matches!(
            self.commands.peek(),
            Some((_, QueuedCommand::Spawn {
                kind: next_kind,
                bundle,
                ..
            })) if *next_kind == kind && bundle.bundle_type_id() == bundle_type
        )
    }

    /// Consumes the following homogeneous spawn command when it is one.
    fn next_spawn(&mut self, kind: CommandKind, bundle_type: TypeId) -> Option<DeferredSpawnEntry> {
        if !self.next_is_spawn(kind, bundle_type) {
            return None;
        }

        let (index, QueuedCommand::Spawn { entity, bundle, .. }) = self.commands.next()? else {
            unreachable!("the next queued command was checked as a homogeneous spawn")
        };

        Some(DeferredSpawnEntry {
            index,
            entity,
            bundle,
        })
    }

    /// Returns the world the queue is applied against.
    fn world(&mut self) -> &mut World {
        self.world
    }
}

impl Drop for PendingCommands<'_> {
    /// Releases the reservations of the commands that never ran.
    fn drop(&mut self) {
        for (_, command) in self.commands.by_ref() {
            match command {
                QueuedCommand::Spawn { entity, .. } => {
                    CommandQueue::discard_entities(self.world, [entity]);
                }
                QueuedCommand::Reserved { entities, .. } => {
                    CommandQueue::discard_entities(self.world, entities);
                }
                QueuedCommand::Normal { .. } | QueuedCommand::Despawn { .. } => {}
            }
        }
    }
}

/// Keeps one deferred entity reservation recoverable until its spawn succeeds.
struct PendingSingleReservation<'w> {
    world: &'w mut World,
    /// Reserved entity; `None` once a spawn has committed it.
    entity: Option<Entity>,
}

impl<'w> PendingSingleReservation<'w> {
    /// Arms the guard over `entity` until its spawn is applied.
    fn new(world: &'w mut World, entity: Entity) -> Self {
        Self {
            world,
            entity: Some(entity),
        }
    }

    /// Applies `bundle` to the reserved entity, disarming the guard on success.
    ///
    /// A failed spawn leaves the guard armed so drop still releases the entity.
    fn apply(&mut self, bundle: Box<dyn DeferredSpawnBundle>) -> Result<(), WorldError> {
        let entity = self
            .entity
            .expect("pending single reservation already consumed");
        let result = bundle.apply_one(self.world, entity);

        if result.is_ok() {
            self.entity = None;
        }

        result
    }
}

impl Drop for PendingSingleReservation<'_> {
    /// Releases the reservation when its spawn never committed.
    fn drop(&mut self) {
        if let Some(entity) = self.entity.take() {
            CommandQueue::discard_entities(self.world, [entity]);
        }
    }
}

/// Owns the reservations in a fused spawn run until each entity is committed.
struct PendingSpawnRun<'w> {
    world: &'w mut World,
    entities: Vec<Entity>,
    /// Leading entities already committed; drop releases the uncommitted tail.
    committed: usize,
}

impl<'w> PendingSpawnRun<'w> {
    /// Arms the guard over the run's reserved entities, none committed yet.
    fn new(world: &'w mut World, entities: Vec<Entity>) -> Self {
        Self {
            world,
            entities,
            committed: 0,
        }
    }

    /// Commits each entry through its own `spawn_reserved` call.
    ///
    /// Returns the queue index, entity, and error of the first failure;
    /// entries applied before it stay alive and the tail is released on drop.
    fn apply_scalar(
        &mut self,
        entries: Vec<DeferredSpawnEntry>,
    ) -> Result<(), (usize, Entity, WorldError)> {
        for entry in entries {
            if let Err(source) = entry.bundle.apply_one(self.world, entry.entity) {
                return Err((entry.index, entry.entity, source));
            }

            self.committed += 1;
        }

        Ok(())
    }

    /// Commits the whole run through one `spawn_reserved_batch` call.
    ///
    /// A failure locates the offending entity inside the run so the reported
    /// queue index still points at the command that broke the batch.
    fn apply_batch(
        &mut self,
        entries: Vec<DeferredSpawnEntry>,
    ) -> Result<(), (usize, Entity, WorldError)> {
        let mut entries = entries.into_iter();
        let first = entries
            .next()
            .expect("a deferred spawn run must contain at least one entry");
        let first_index = first.index;
        let first_entity = first.entity;
        let remaining = entries.map(|entry| entry.bundle).collect();

        match first
            .bundle
            .apply_batch(self.world, &self.entities, remaining)
        {
            Ok(()) => {
                self.committed = self.entities.len();
                Ok(())
            }
            Err(source) => {
                let failed_entity =
                    command_error_entity(&source, &self.entities).unwrap_or(first_entity);
                let offset = self
                    .entities
                    .iter()
                    .position(|&entity| entity == failed_entity)
                    .unwrap_or(0);

                Err((first_index + offset, failed_entity, source))
            }
        }
    }
}

impl Drop for PendingSpawnRun<'_> {
    /// Releases the entities that were reserved but never committed.
    fn drop(&mut self) {
        CommandQueue::discard_entities(self.world, self.entities[self.committed..].iter().copied());
    }
}

/// Unwind guard over reserved entities awaiting a batch application.
///
/// Drops release any entities that were reserved but not yet committed.
struct PendingReservation<'w> {
    world: &'w mut World,
    /// Reserved batch; `None` once the command has committed it.
    entities: Option<Box<[Entity]>>,
}

impl<'w> PendingReservation<'w> {
    /// Arms the guard over the command's reserved entity batch.
    fn new(world: &'w mut World, entities: Box<[Entity]>) -> Self {
        Self {
            world,
            entities: Some(entities),
        }
    }

    /// Runs `command` with the reserved batch, disarming the guard on success.
    ///
    /// A failure leaves the guard armed so drop releases the reservations.
    fn apply(&mut self, command: BoxedReservedCommand) -> Result<(), WorldError> {
        let entities = self
            .entities
            .as_deref()
            .expect("pending reservation already consumed");

        let result = command(self.world, entities);

        if result.is_ok() {
            self.entities = None;
        }

        result
    }

    /// Returns the guarded entities; valid only while the guard is armed.
    fn entities(&self) -> &[Entity] {
        self.entities
            .as_deref()
            .expect("failed reservation must remain armed")
    }
}

impl Drop for PendingReservation<'_> {
    /// Releases the reservations the command never committed.
    fn drop(&mut self) {
        if let Some(entities) = self.entities.take() {
            CommandQueue::discard_entities(self.world, entities);
        }
    }
}

impl CommandQueue {
    /// Consecutive strict despawns can be executed as one structural batch.
    ///
    /// Keeping despawns as data instead of opaque closures lets the apply phase
    /// recover component-major sparse removal without reordering commands across
    /// other structural operations.
    fn push_despawn(&mut self, entity: Entity) {
        self.commands.push(QueuedCommand::Despawn { entity });
    }

    /// Queues a reserved spawn as data carrying its bundle, the shape the
    /// apply phase matches against to fuse homogeneous runs.
    fn push_spawn<B>(&mut self, kind: CommandKind, entity: Entity, bundle: B)
    where
        B: Bundle + Send + 'static,
    {
        self.commands.push(QueuedCommand::Spawn {
            kind,
            entity,
            bundle: Box::new(bundle),
        });
    }

    /// Queues a custom closure reported as `CommandKind::Custom`.
    fn push<F>(&mut self, command: F)
        where
            F: FnOnce(&mut World) -> Result<(), WorldError> + Send + 'static,
    {
        self.push_with_kind(CommandKind::Custom, None, command);
    }

    /// Queues a closure tagged with the kind and entity error reporting uses.
    fn push_with_kind<F>(&mut self, kind: CommandKind, entity: Option<Entity>, command: F)
    where
        F: FnOnce(&mut World) -> Result<(), WorldError> + Send + 'static,
    {
        self.commands.push(QueuedCommand::Normal {
            kind,
            entity,
            command: Box::new(command),
        });
    }

    #[cfg(test)]
    fn push_reserved<F>(&mut self, kind: CommandKind, entity: Entity, command: F)
    where
        F: FnOnce(&mut World, Entity) -> Result<(), WorldError> + Send + 'static,
    {
        self.push_reserved_batch(kind, vec![entity], move |world, entities| {
            let entity = entities[0];
            command(world, entity)
        });
    }

    /// Queues a closure that owns its reserved entity batch until it runs.
    ///
    /// An empty batch records nothing.
    fn push_reserved_batch<F>(&mut self, kind: CommandKind, entities: Vec<Entity>, command: F)
    where
        F: FnOnce(&mut World, &[Entity]) -> Result<(), WorldError> + Send + 'static,
    {
        if entities.is_empty() {
            return;
        }

        self.commands.push(QueuedCommand::Reserved {
            kind,
            entities: entities.into_boxed_slice(),
            command: Box::new(command),
        });
    }

    /// Drains the queue against `world`, fusing spawn and despawn runs as the
    /// one-command lookahead finds them.
    ///
    /// The first failing command aborts the flush; the drainer then releases
    /// every reservation belonging to commands that never ran.
    fn apply(&mut self, world: &mut World) -> Result<(), WorldError> {
        let commands = std::mem::take(&mut self.commands);
        let mut pending = PendingCommands::new(world, commands);

        while let Some((index, queued)) = pending.next() {
            match queued {
                QueuedCommand::Normal {
                    kind,
                    entity,
                    command,
                } => {
                    if let Err(err) = command(pending.world()) {
                        let entity = entity.or_else(|| command_error_entity(&err, &[]));

                        return Err(WorldError::Command(CommandError::new(
                            index, kind, entity, err,
                        )));
                    }
                }

                QueuedCommand::Spawn {
                    kind,
                    entity,
                    bundle,
                } => {
                    let bundle_type = bundle.bundle_type_id();

                    if !pending.next_is_spawn(kind, bundle_type) {
                        let mut reservation =
                            PendingSingleReservation::new(pending.world(), entity);

                        if let Err(source) = reservation.apply(bundle) {
                            return Err(WorldError::Command(CommandError::new(
                                index,
                                kind,
                                Some(entity),
                                source,
                            )));
                        }

                        continue;
                    }

                    let mut entries = Vec::with_capacity(MIN_FUSED_SPAWN_COMMANDS);
                    entries.push(DeferredSpawnEntry {
                        index,
                        entity,
                        bundle,
                    });

                    while let Some(entry) = pending.next_spawn(kind, bundle_type) {
                        debug_assert_eq!(entry.index, index + entries.len());
                        entries.push(entry);
                    }

                    if let Err((index, entity, source)) =
                        Self::apply_spawn_run(pending.world(), entries)
                    {
                        return Err(WorldError::Command(CommandError::new(
                            index,
                            kind,
                            Some(entity),
                            source,
                        )));
                    }
                }

                QueuedCommand::Despawn { entity } => {
                    if !pending.next_is_despawn() {
                        if let Err(source) = pending.world().despawn(entity) {
                            return Err(WorldError::Command(CommandError::new(
                                index,
                                CommandKind::Despawn,
                                Some(entity),
                                source,
                            )));
                        }

                        continue;
                    }

                    let first_index = index;
                    let mut entities = Vec::with_capacity(32);
                    entities.push(entity);

                    while let Some((index, entity)) = pending.next_despawn() {
                        debug_assert_eq!(index, first_index + entities.len());
                        entities.push(entity);
                    }

                    if let Err((index, entity, source)) =
                        Self::apply_despawn_run(pending.world(), first_index, &entities)
                    {
                        return Err(WorldError::Command(CommandError::new(
                            index,
                            CommandKind::Despawn,
                            Some(entity),
                            source,
                        )));
                    }
                }

                QueuedCommand::Reserved {
                    kind,
                    entities,
                    command,
                } => {
                    let mut reservation = PendingReservation::new(pending.world(), entities);

                    if let Err(err) = reservation.apply(command) {
                        let entity = command_error_entity(&err, reservation.entities());

                        return Err(WorldError::Command(CommandError::new(
                            index, kind, entity, err,
                        )));
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies one fused spawn run through the guard that owns its entities,
    /// scalar below `MIN_FUSED_SPAWN_COMMANDS` and batched above it.
    fn apply_spawn_run(
        world: &mut World,
        entries: Vec<DeferredSpawnEntry>,
    ) -> Result<(), (usize, Entity, WorldError)> {
        let entities = entries.iter().map(|entry| entry.entity).collect();
        let mut reservation = PendingSpawnRun::new(world, entities);

        if entries.len() < MIN_FUSED_SPAWN_COMMANDS {
            reservation.apply_scalar(entries)
        } else {
            reservation.apply_batch(entries)
        }
    }

    /// Applies adjacent strict despawns through the world batch path.
    ///
    /// `World::despawn_batch` validates the complete input before mutation. If
    /// validation fails, replaying the run entity by entity preserves the
    /// original command index, error kind, and partial-application contract.
    fn apply_despawn_run(
        world: &mut World,
        first_index: usize,
        entities: &[Entity],
    ) -> Result<(), (usize, Entity, WorldError)> {
        if entities.len() < MIN_FUSED_DESPAWN_COMMANDS {
            for (offset, &entity) in entities.iter().enumerate() {
                if let Err(source) = world.despawn(entity) {
                    return Err((first_index + offset, entity, source));
                }
            }

            return Ok(());
        }

        if world.despawn_batch(entities.iter().copied()).is_ok() {
            return Ok(());
        }

        for (offset, &entity) in entities.iter().enumerate() {
            if let Err(source) = world.despawn(entity) {
                return Err((first_index + offset, entity, source));
            }
        }

        unreachable!("a failed prevalidated despawn batch must fail during scalar replay")
    }

    /// Releases the reservations carried by commands that never applied.
    fn discard_remaining_reservations<I>(world: &mut World, commands: I)
    where
        I: IntoIterator<Item = QueuedCommand>,
    {
        for command in commands {
            match command {
                QueuedCommand::Spawn { entity, .. } => {
                    Self::discard_entities(world, [entity]);
                }
                QueuedCommand::Reserved { entities, .. } => {
                    Self::discard_entities(world, entities);
                }
                QueuedCommand::Normal { .. } | QueuedCommand::Despawn { .. } => {}
            }
        }
    }

    /// Drops every queued command, releasing the reservations they hold.
    fn discard(&mut self, world: &mut World) {
        let commands = std::mem::take(&mut self.commands);

        Self::discard_remaining_reservations(world, commands);
    }

    /// Discards each entity, whether still reserved or already alive.
    fn discard_entities(world: &mut World, entities: impl IntoIterator<Item = Entity>) {
        for entity in entities {
            world.discard_reserved_or_despawn(entity);
        }
    }

    /// Returns whether no command is queued.
    fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Returns the number of queued commands.
    fn len(&self) -> usize {
        self.commands.len()
    }

    /// Preallocates capacity for `additional` more commands.
    fn reserve(&mut self, additional: usize) {
        self.commands.reserve(additional);
    }
}

/// Extracts the offending entity from an error, falling back to the batch
/// entry that was being applied when it failed.
fn command_error_entity(error: &WorldError, fallback_entities: &[Entity]) -> Option<Entity> {
    match error {
        WorldError::EntityNotFound(entity) => Some(*entity),

        WorldError::Entity(
            EntityError::Duplicate(entity)
            | EntityError::NotReserved(entity)
            | EntityError::AlreadyAlive(entity)
            | EntityError::NotAlive(entity)
            | EntityError::Stale(entity)
            | EntityError::Invalid(entity),
        ) => Some(*entity),

        WorldError::Command(error) => error.entity(),
        _ if fallback_entities.len() == 1 => Some(fallback_entities[0]),
        _ => None,
    }
}

/// Persistent [`Commands`] state associated with one system.
pub struct CommandsState {
    /// Unique-param access derived from `CommandsAccessMarker`.
    access: SystemAccess,
    queue: CommandQueue,
}

/// Fluent sequence of commands directed at one entity.
pub struct EntityCommands<'s> {
    entity: Entity,
    queue: &'s mut CommandQueue,
    /// Policy applied when the entity is dead at flush time.
    missing_entity: MissingEntityPolicy,
}

/// Applies the missing-entity policy to a world result.
///
/// Under `Ignore` an `EntityNotFound` failure becomes a success; every other
/// combination passes through unchanged.
fn handle_entity_result(
    policy: MissingEntityPolicy,
    result: Result<(), WorldError>,
) -> Result<(), WorldError> {
    match (policy, result) {
        (MissingEntityPolicy::Ignore, Err(WorldError::EntityNotFound(_))) => Ok(()),

        (_, result) => result,
    }
}

/// Checks that every entity is still alive before mutating a strict batch.
fn validate_entities_alive(world: &World, entities: &[Entity]) -> Result<(), WorldError> {
    if let Some(&entity) = entities.iter().find(|entity| !world.contains(**entity)) {
        return Err(WorldError::EntityNotFound(entity));
    }

    Ok(())
}

impl<'s> EntityCommands<'s> {
    /// Returns the entity these commands target.
    #[inline]
    pub fn id(&self) -> Entity {
        self.entity
    }

    /// Queues a bundle insert on the entity.
    pub fn insert<B>(self, bundle: B) -> Self
    where
        B: Bundle + Send + 'static,
    {
        let entity = self.entity;
        let policy = self.missing_entity;

        self.queue
            .push_with_kind(CommandKind::Insert, Some(entity), move |world| {
                handle_entity_result(policy, world.insert(entity, bundle))
            });

        self
    }

    /// Queues the despawn of the entity and all of its components.
    pub fn despawn(self) {
        let entity = self.entity;
        let policy = self.missing_entity;

        match policy {
            MissingEntityPolicy::Error => self.queue.push_despawn(entity),
            MissingEntityPolicy::Ignore => {
                self.queue
                    .push_with_kind(CommandKind::Despawn, Some(entity), move |world| {
                        handle_entity_result(policy, world.despawn(entity))
                    });
            }
        }
    }

    /// Queues the removal of one component from the entity.
    pub fn remove<C>(self) -> Self
    where
        C: Component,
    {
        let entity = self.entity;
        let policy = self.missing_entity;

        self.queue
            .push_with_kind(CommandKind::Remove, Some(entity), move |world| {
                handle_entity_result(policy, world.remove::<C>(entity))
            });

        self
    }

    /// Queues the removal of one bundle's components from the entity.
    pub fn remove_bundle<B>(self) -> Self
    where
        B: BundleInfo + 'static,
    {
        let entity = self.entity;
        let policy = self.missing_entity;

        self.queue
            .push_with_kind(CommandKind::RemoveBundle, Some(entity), move |world| {
                handle_entity_result(policy, world.remove_bundle::<B>(entity))
            });

        self
    }

    /// Queues a retain: every component outside `B` is removed from the entity.
    pub fn retain_bundle<B>(self) -> Self
    where
        B: BundleInfo + 'static,
    {
        let entity = self.entity;
        let policy = self.missing_entity;

        self.queue
            .push_with_kind(CommandKind::RetainBundle, Some(entity), move |world| {
                handle_entity_result(policy, world.retain_bundle::<B>(entity))
            });

        self
    }

    /// Queues the removal of every component from the entity, despawning
    /// nothing.
    pub fn clear(self) -> Self {
        let entity = self.entity;
        let policy = self.missing_entity;

        self.queue
            .push_with_kind(CommandKind::Clear, Some(entity), move |world| {
                handle_entity_result(policy, world.clear_entity(entity))
            });

        self
    }
}

/// Deferred structural and resource operations for a system.
///
/// Commands do not mutate the [`World`] immediately. This permits queueing them
/// while iterating a query without invalidating archetype rows or component borrows.
///
/// The schedule applies queued operations only after every system in the current
/// parallel execution phase has stopped borrowing the world.
///
/// Commands apply in registration order. If one fails, earlier changes remain,
/// later commands are discarded, and their reservations are released. Panics
/// also release the current and remaining reservations before unwinding.
pub struct Commands<'w, 's> {
    /// Allocator borrowed so entities are reserved at record time.
    allocator: &'w EntityAllocator,
    queue: &'s mut CommandQueue,
}

impl<'w, 's> Commands<'w, 's> {
    /// Queues the insertion of `R::default()` when the resource is missing.
    pub fn init_resource<R>(&mut self)
    where
        R: Resource + Default,
    {
        self.queue
            .push_with_kind(CommandKind::InitResource, None, |world| {
                world.init_resource::<R>();
                Ok(())
            });
    }

    /// Returns commands that fail if the entity is not alive when applied.
    pub fn entity(&mut self, entity: Entity) -> EntityCommands<'_> {
        EntityCommands {
            entity,
            queue: self.queue,
            missing_entity: MissingEntityPolicy::Error,
        }
    }

    /// Returns commands that are skipped if the entity is no longer alive.
    pub fn entity_if_alive(&mut self, entity: Entity) -> EntityCommands<'_> {
        EntityCommands {
            entity,
            queue: self.queue,
            missing_entity: MissingEntityPolicy::Ignore,
        }
    }

    /// Queues a per-entity operation over a batch as one command.
    ///
    /// The command rejects duplicate entities before mutating anything and,
    /// under `MissingEntityPolicy::Error`, checks that every entity is alive
    /// up front; `Ignore` then skips dead entities one by one.
    fn push_entity_batch<I, F>(
        &mut self,
        kind: CommandKind,
        entities: I,
        policy: MissingEntityPolicy,
        mut operation: F,
    ) where
        I: IntoIterator<Item = Entity>,
        F: FnMut(&mut World, Entity) -> Result<(), WorldError> + Send + 'static,
    {
        let entities: Vec<Entity> = entities.into_iter().collect();

        if entities.is_empty() {
            return;
        }

        self.queue.push_with_kind(kind, None, move |world| {
            world.validate_unique_batch_entities(&entities)?;

            if matches!(policy, MissingEntityPolicy::Error) {
                validate_entities_alive(world, &entities)?;
            }

            for entity in entities {
                handle_entity_result(policy, operation(world, entity))?;
            }

            Ok(())
        });
    }

    /// Queues the insertion of `f`'s result when the resource is missing.
    pub fn get_resource_or_insert_with<R, F>(&mut self, f: F)
    where
        R: Resource,
        F: FnOnce() -> R + Send + 'static,
    {
        self.queue
            .push_with_kind(CommandKind::GetResourceOrInsert, None, move |world| {
                world.get_resource_or_insert_with::<R, _>(f);
                Ok(())
            });
    }

    /// Queues `f` against resource `R`, failing when the resource is missing.
    pub fn modify_resource<R, F>(&mut self, f: F)
    where
        R: Resource,
        F: FnOnce(&mut R) + Send + 'static,
    {
        self.queue
            .push_with_kind(CommandKind::ModifyResource, None, move |world| {
                let mut resource = world
                    .get_resource_mut::<R>()
                    .ok_or_else(|| ResourceError::Missing(ResourceId::of::<R>()))?;

                f(&mut *resource);

                Ok(())
            });
    }

    /// Queues `f` against resource `R` when it exists, doing nothing otherwise.
    pub fn modify_resource_if_exists<R, F>(&mut self, f: F)
    where
        R: Resource,
        F: FnOnce(&mut R) + Send + 'static,
    {
        self.queue
            .push_with_kind(CommandKind::ModifyResource, None, move |world| {
                if let Some(mut resource) = world.get_resource_mut::<R>() {
                    f(&mut *resource);
                }
                Ok(())
            });
    }

    /// Queues a modify of resource `R`, inserting `default` when missing.
    pub fn modify_or_insert_resource<R, F>(&mut self, default: R, f: F)
    where
        R: Resource,
        F: FnOnce(&mut R) + Send + 'static,
    {
        self.queue
            .push_with_kind(CommandKind::ModifyResource, None, move |world| {
                let mut resource = world.get_resource_or_insert_with(|| default);

                f(&mut *resource);

                Ok(())
            });
    }

    /// Queues a modify of resource `R`, inserting `default`'s result when
    /// missing.
    pub fn modify_or_insert_resource_with<R, D, F>(&mut self, default: D, modify: F)
    where
        R: Resource,
        D: FnOnce() -> R + Send + 'static,
        F: FnOnce(&mut R) + Send + 'static,
    {
        self.queue
            .push_with_kind(CommandKind::ModifyResource, None, move |world| {
                let mut resource = world.get_resource_or_insert_with(default);

                modify(&mut *resource);

                Ok(())
            });
    }

    /// Reserves an entity and schedules its spawn with `bundle`, returning
    /// chained commands for it.
    pub fn spawn<B>(&mut self, bundle: B) -> EntityCommands<'_>
    where
        B: Bundle + Send + 'static,
    {
        let entity = self.allocator.reserve();
        let missing_entity = MissingEntityPolicy::Error;

        self.queue.push_spawn(CommandKind::Spawn, entity, bundle);

        EntityCommands {
            entity,
            queue: self.queue,
            missing_entity,
        }
    }

    /// Reserves an entity and schedules its empty spawn, returning chained
    /// commands for it.
    pub fn spawn_empty(&mut self) -> EntityCommands<'_> {
        let entity = self.allocator.reserve();
        let missing_entity = MissingEntityPolicy::Error;

        self.queue.push_spawn(CommandKind::Spawn, entity, ());

        EntityCommands {
            entity,
            queue: self.queue,
            missing_entity,
        }
    }

    /// Reserves an entity and schedules its empty creation.
    pub fn reserve_entity(&mut self) -> Entity {
        let entity = self.allocator.reserve();

        self.queue
            .push_spawn(CommandKind::ReserveEntity, entity, ());

        entity
    }

    /// Reserves several entities and schedules their empty creation.
    pub fn reserve_entities(&mut self, amount: u32) -> Vec<Entity> {
        if amount == 0 {
            return Vec::new();
        }

        let entities = self.allocator.reserve_many(amount);
        let result = entities.clone();

        self.queue.push_reserved_batch(
            CommandKind::ReserveEntities,
            entities,
            |world, entities| world.spawn_reserved_empty_batch(entities),
        );

        result
    }

    /// Reserves one entity per bundle and schedules their homogeneous batch
    /// spawn, returning the reserved handles.
    pub fn spawn_batch<I, B>(&mut self, bundles: I) -> Vec<Entity>
    where
        I: IntoIterator<Item = B>,
        B: Bundle + Send + 'static,
    {
        let bundles: Vec<B> = bundles.into_iter().collect();

        if bundles.is_empty() {
            return Vec::new();
        }

        let amount = u32::try_from(bundles.len())
            .expect("spawn batch exceeds the maximum supported entity count");

        let entities = self.allocator.reserve_many(amount);
        let result = entities.clone();

        self.queue.push_reserved_batch(
            CommandKind::SpawnBatch,
            entities,
            move |world, entities| world.spawn_reserved_batch(entities, bundles),
        );

        result
    }

    /// Reserves entities and builds their bundles. If `build` fails, every
    /// reservation made by this call is released.
    pub fn try_spawn_batch_with<B, E, F>(
        &mut self,
        amount: u32,
        mut build: F,
    ) -> Result<Vec<Entity>, E>
    where
        B: Bundle + Send + 'static,
        F: FnMut(Entity) -> Result<B, E>,
    {
        if amount == 0 {
            return Ok(Vec::new());
        }

        let entities = self.allocator.reserve_many(amount);
        let reservation = ReservationGuard::new(self.allocator, entities);
        let mut bundles = Vec::with_capacity(amount as usize);

        for entity in reservation.entities().iter().copied() {
            bundles.push(build(entity)?);
        }

        let entities = reservation.take();
        let result = entities.clone();

        self.queue.push_reserved_batch(
            CommandKind::SpawnBatch,
            entities,
            move |world, entities| world.spawn_reserved_batch(entities, bundles),
        );

        Ok(result)
    }

    /// Reserves `amount` entities and schedules their empty batch spawn.
    pub fn spawn_empty_batch(&mut self, amount: u32) -> Vec<Entity> {
        if amount == 0 {
            return Vec::new();
        }

        let entities = self.allocator.reserve_many(amount);

        let result = entities.clone();

        self.queue
            .push_reserved_batch(CommandKind::SpawnBatch, entities, |world, entities| {
                world.spawn_reserved_empty_batch(entities)
            });

        result
    }

    /// Reserves entities and builds each bundle from its identifier.
    pub fn spawn_batch_with<B, F>(&mut self, amount: u32, mut build: F) -> Vec<Entity>
    where
        B: Bundle + Send + 'static,
        F: FnMut(Entity) -> B,
    {
        if amount == 0 {
            return Vec::new();
        }

        let entities = self.allocator.reserve_many(amount);
        let reservation = ReservationGuard::new(self.allocator, entities);
        let bundles: Vec<B> = reservation
            .entities()
            .iter()
            .copied()
            .map(&mut build)
            .collect();
        let entities = reservation.take();
        let result = entities.clone();

        self.queue.push_reserved_batch(
            CommandKind::SpawnBatch,
            entities,
            move |world, entities| world.spawn_reserved_batch(entities, bundles),
        );

        result
    }

    /// Queues the replacement of resource `R`.
    pub fn insert_resource<R>(&mut self, resource: R)
    where
        R: Resource,
    {
        self.queue
            .push_with_kind(CommandKind::InsertResource, None, move |world| {
                world.insert_resource(resource);
                Ok(())
            });
    }

    /// Queues the removal of resource `R`.
    pub fn remove_resource<R>(&mut self)
    where
        R: Resource,
    {
        self.queue
            .push_with_kind(CommandKind::RemoveResource, None, |world| {
                world.remove_resource::<R>();
                Ok(())
            });
    }

    /// Queues a bundle insert on `entity`; fails at flush time when dead.
    pub fn insert<B>(&mut self, entity: Entity, bundle: B)
    where
        B: Bundle + Send + 'static,
    {
        self.entity(entity).insert(bundle);
    }

    /// Queues a heterogeneous bundle insert over `(entity, bundle)` pairs.
    pub fn insert_batch<I, B>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (Entity, B)>,
        B: Bundle + Send + 'static,
    {
        let (entities, bundles): (Vec<Entity>, Vec<B>) = entries.into_iter().unzip();

        if entities.is_empty() {
            return;
        }

        self.queue
            .push_with_kind(CommandKind::InsertBatch, None, move |world| {
                world.insert_batch(entities.into_iter().zip(bundles).collect())
            });
    }

    /// Queues a bundle insert over pairs, skipping entities dead at flush.
    pub fn insert_batch_if_alive<I, B>(&mut self, entries: I)
    where
        I: IntoIterator<Item = (Entity, B)>,
        B: Bundle + Send + 'static,
    {
        let (entities, bundles): (Vec<Entity>, Vec<B>) = entries.into_iter().unzip();

        if entities.is_empty() {
            return;
        }

        self.queue
            .push_with_kind(CommandKind::InsertBatch, None, move |world| {
                world.validate_unique_batch_entities(&entities)?;

                let tick = world.advance_tick();

                for (entity, bundle) in entities.into_iter().zip(bundles) {
                    handle_entity_result(
                        MissingEntityPolicy::Ignore,
                        world.insert_at_tick(entity, bundle, tick),
                    )?;
                }

                Ok(())
            });
    }

    /// Queues the removal of component `C` from `entity`.
    pub fn remove<C: Component>(&mut self, entity: Entity) {
        self.entity(entity).remove::<C>();
    }

    /// Keeps only the components in `B` on `entity`; the rest go at flush time.
    pub fn retain_bundle<B>(&mut self, entity: Entity)
    where
        B: BundleInfo + 'static,
    {
        self.entity(entity).retain_bundle::<B>();
    }

    /// Queues bundle retention over a batch; fails at flush when dead.
    pub fn retain_bundle_batch<I, B>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
        B: BundleInfo + 'static,
    {
        self.push_entity_batch(
            CommandKind::RetainBundleBatch,
            entities,
            MissingEntityPolicy::Error,
            |world, entity| world.retain_bundle::<B>(entity),
        );
    }

    /// Queues bundle retention over a batch, skipped when dead.
    pub fn retain_bundle_batch_if_alive<I, B>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
        B: BundleInfo + 'static,
    {
        self.push_entity_batch(
            CommandKind::RetainBundleBatch,
            entities,
            MissingEntityPolicy::Ignore,
            |world, entity| world.retain_bundle::<B>(entity),
        );
    }

    /// Queues the removal of component `C` over a batch; fails at flush when
    /// dead.
    pub fn remove_batch<I, C>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
        C: Component,
    {
        let entities = entities.into_iter().collect::<Vec<_>>();

        if entities.is_empty() {
            return;
        }

        self.queue
            .push_with_kind(CommandKind::RemoveBatch, None, move |world| {
                world.remove_batch::<_, C>(entities).map(|_| ())
            });
    }

    /// Queues the removal of component `C` over a batch, skipped when dead.
    pub fn remove_batch_if_alive<I, C>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
        C: Component,
    {
        self.push_entity_batch(
            CommandKind::RemoveBatch,
            entities,
            MissingEntityPolicy::Ignore,
            |world, entity| world.remove::<C>(entity),
        );
    }

    /// Queues the removal of every bundle component from `entity`.
    pub fn remove_bundle<B>(&mut self, entity: Entity)
    where
        B: BundleInfo + 'static,
    {
        self.entity(entity).remove_bundle::<B>();
    }

    /// Queues bundle removal over a batch; fails at flush when dead.
    pub fn remove_bundle_batch<I, B>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
        B: BundleInfo + 'static,
    {
        self.push_entity_batch(
            CommandKind::RemoveBundleBatch,
            entities,
            MissingEntityPolicy::Error,
            |world, entity| world.remove_bundle::<B>(entity),
        );
    }

    /// Queues bundle removal over a batch, skipped when dead.
    pub fn remove_bundle_batch_if_alive<I, B>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
        B: BundleInfo + 'static,
    {
        self.push_entity_batch(
            CommandKind::RemoveBundleBatch,
            entities,
            MissingEntityPolicy::Ignore,
            |world, entity| world.remove_bundle::<B>(entity),
        );
    }

    /// Queues the despawn of `entity` and all of its components.
    pub fn despawn(&mut self, entity: Entity) {
        self.entity(entity).despawn();
    }

    /// Queues a batch despawn; fails at flush when any entity is dead.
    pub fn despawn_batch<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
    {
        let entities = entities.into_iter().collect::<Vec<_>>();

        if entities.is_empty() {
            return;
        }

        self.queue
            .push_with_kind(CommandKind::DespawnBatch, None, move |world| {
                world.despawn_batch(entities).map(|_| ())
            });
    }

    /// Queues a batch despawn of the entities still alive at flush time.
    pub fn despawn_batch_if_alive<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
    {
        let entities = entities.into_iter().collect::<Vec<_>>();

        if entities.is_empty() {
            return;
        }

        self.queue
            .push_with_kind(CommandKind::DespawnBatch, None, move |world| {
                world.validate_unique_batch_entities(&entities)?;

                let live = entities
                    .into_iter()
                    .filter(|entity| world.contains(*entity))
                    .collect::<Vec<_>>();

                world.despawn_batch(live).map(|_| ())
            });
    }

    /// Queues an arbitrary world operation, failing at flush on error.
    pub fn add<F>(&mut self, command: F)
    where
        F: FnOnce(&mut World) -> Result<(), WorldError> + Send + 'static,
    {
        self.queue.push(command);
    }

    /// Queues an infallible world operation.
    pub fn add_infallible<F>(&mut self, command: F)
    where
        F: FnOnce(&mut World) + Send + 'static,
    {
        self.queue.push(move |world| {
            command(world);
            Ok(())
        });
    }

    /// Queues the removal of every component from `entity`.
    pub fn clear(&mut self, entity: Entity) {
        self.entity(entity).clear();
    }

    /// Queues a component clear over a batch; fails at flush when dead.
    pub fn clear_batch<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
    {
        self.push_entity_batch(
            CommandKind::ClearBatch,
            entities,
            MissingEntityPolicy::Error,
            |world, entity| world.clear_entity(entity),
        );
    }

    /// Queues a component clear over a batch, skipped when dead.
    pub fn clear_batch_if_alive<I>(&mut self, entities: I)
    where
        I: IntoIterator<Item = Entity>,
    {
        self.push_entity_batch(
            CommandKind::ClearBatch,
            entities,
            MissingEntityPolicy::Ignore,
            |world, entity| world.clear_entity(entity),
        );
    }

    /// Returns whether no command is queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the number of queued commands.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Preallocates capacity for `additional` more commands.
    pub fn reserve_commands(&mut self, additional: usize) {
        self.queue.reserve(additional);
    }
}

unsafe impl SystemParam for Commands<'_, '_> {
    type State = CommandsState;
    type Item<'w, 's> = Commands<'w, 's>;

    /// Creates the state with its unique-param access and an empty queue.
    fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
        Ok(CommandsState {
            access: SystemAccess::from_unique_param::<CommandsAccessMarker>(),
            queue: CommandQueue::default(),
        })
    }

    /// Returns the unique-param access captured during initialization.
    fn access(state: &Self::State) -> &SystemAccess {
        &state.access
    }

    /// Borrows the entity allocator and the system's queue for one run.
    ///
    /// The unique access declared through `CommandsAccessMarker` guarantees
    /// the queue is never handed to two parameters at once.
    unsafe fn get_param<'w, 's>(
        world: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        Ok(Commands {
            allocator: world.entities(),
            queue: &mut state.queue,
        })
    }

    /// Returns whether any command is queued.
    fn has_deferred_work(state: &Self::State) -> bool {
        !state.queue.is_empty()
    }

    /// Applies the queued commands against the world.
    fn apply(world: &mut World, state: &mut Self::State) -> Result<(), WorldError> {
        state.queue.apply(world)
    }

    /// Drops the queued commands, releasing their reservations.
    fn discard(world: &mut World, state: &mut Self::State) {
        state.queue.discard(world);
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use millarecs_macros::Component;

    use super::{CommandKind, CommandQueue, Commands};
    use crate::{
        entity::{Entity, EntityError},
        system::{
            param::SystemParam,
            query::Query as SystemQuery,
            schedule::{Schedule, ScheduleError, ScheduleExecutor},
        },
        world::{World, WorldError},
    };

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Counter(i32);

    #[derive(Default, Component, Debug, PartialEq, Eq)]
    struct Marker;

    #[derive(Component, Debug, PartialEq, Eq)]
    struct Owner(Entity);

    impl Default for Owner {
        fn default() -> Self {
            Self(Entity::INVALID)
        }
    }

    #[derive(Component, Default)]
    struct Unregistered;

    #[derive(Default, Debug, PartialEq, Eq)]
    struct Config(i32);

    #[derive(Debug, PartialEq, Eq)]
    struct BuildError;

    fn setup_world() -> World {
        let mut world = World::new();
        world.register_table::<Counter>().unwrap();
        world.register_table::<Marker>().unwrap();
        world.register_table::<Owner>().unwrap();
        world
    }

    fn counter_values(world: &World) -> Vec<i32> {
        let mut query = world.query_all::<&Counter>().unwrap();
        let mut values: Vec<_> = query.iter().map(|counter| counter.0).collect();
        values.sort_unstable();
        values
    }

    fn marker_count(world: &World) -> usize {
        let mut query = world.query_all::<&Marker>().unwrap();
        query.iter().count()
    }

    #[test]
    fn commands_are_deferred_and_visible_to_later_systems() {
        struct SpawnFinished;

        fn spawn(mut commands: Commands, mut query: SystemQuery<&Counter>) {
            commands.spawn((Counter(1),));
            assert_eq!(query.iter().count(), 0);
        }

        fn observe(mut query: SystemQuery<&Counter>) {
            assert_eq!(query.iter().count(), 1);
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(spawn).label::<SpawnFinished>();
        schedule.add_system(observe).after::<SpawnFinished>();

        schedule.run(&mut world, &mut executor).unwrap();
        assert_eq!(counter_values(&world), vec![1]);
    }

    #[test]
    fn fluent_entity_commands_run_after_the_reserved_spawn() {
        fn spawn(mut commands: Commands) {
            let entity = commands.spawn_empty().insert((Counter(4), Marker)).id();

            commands.add(move |world| {
                assert!(world.contains(entity));
                Ok(())
            });
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(spawn);

        schedule.run(&mut world, &mut executor).unwrap();
        assert_eq!(counter_values(&world), vec![4]);
        assert_eq!(marker_count(&world), 1);
    }

    #[test]
    fn reserved_entities_can_be_completed_by_later_commands() {
        fn reserve(mut commands: Commands) {
            let entities = commands.reserve_entities(3);

            commands.insert_batch(
                entities
                    .into_iter()
                    .enumerate()
                    .map(|(index, entity)| (entity, (Counter(index as i32),))),
            );
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(reserve);

        schedule.run(&mut world, &mut executor).unwrap();
        assert_eq!(world.len(), 3);
        assert_eq!(counter_values(&world), vec![0, 1, 2]);
    }

    #[test]
    fn spawn_batches_create_the_expected_entities() {
        fn spawn(mut commands: Commands) {
            commands.spawn_batch([(Counter(2),), (Counter(5),)]);
            commands.spawn_empty_batch(2);
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(spawn);

        schedule.run(&mut world, &mut executor).unwrap();
        assert_eq!(world.len(), 4);
        assert_eq!(counter_values(&world), vec![2, 5]);
    }

    #[test]
    fn spawn_batch_with_receives_each_reserved_entity() {
        fn spawn(mut commands: Commands) {
            commands.spawn_batch_with(3, |entity| (Counter(entity.index() as i32), Owner(entity)));
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(spawn);

        schedule.run(&mut world, &mut executor).unwrap();

        let mut query = world.query_all::<(&Counter, &Owner)>().unwrap();
        let pairs: Vec<_> = query
            .iter()
            .map(|(counter, owner)| (counter.0, owner.0.index() as i32))
            .collect();

        assert_eq!(pairs.len(), 3);
        assert!(pairs.iter().all(|(counter, index)| counter == index));
    }

    #[test]
    fn failed_batch_builder_releases_its_reservations() {
        fn spawn(mut commands: Commands) {
            let mut calls = 0;
            let result: Result<Vec<Entity>, BuildError> = commands.try_spawn_batch_with(4, |_| {
                calls += 1;

                if calls == 3 {
                    Err(BuildError)
                } else {
                    Ok((Counter(calls),))
                }
            });

            assert_eq!(result, Err(BuildError));
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(spawn);

        schedule.run(&mut world, &mut executor).unwrap();
        assert_eq!(world.len(), 0);
        assert_eq!(world.entities().reserved_len(), 0);
    }

    #[test]
    fn panicking_batch_builder_releases_its_reservations() {
        let mut world = setup_world();
        let mut state = <Commands<'_, '_> as SystemParam>::init_state(&mut world).unwrap();

        let result = catch_unwind(AssertUnwindSafe(|| {
            let cell = world.execution_world();
            let mut commands =
                unsafe { <Commands<'_, '_> as SystemParam>::get_param(cell, &mut state).unwrap() };

            commands.spawn_batch_with::<(Counter,), _>(3, |_| panic!("builder panic"));
        }));

        assert!(result.is_err());
        assert_eq!(world.len(), 0);
        assert_eq!(world.entities().reserved_len(), 0);
    }

    #[test]
    fn panicking_reserved_command_rolls_back_its_entities() {
        let mut world = setup_world();
        let entity = world.execution_world().entities().reserve();
        let mut queue = CommandQueue::default();

        queue.push_reserved(CommandKind::Spawn, entity, |world, entity| {
            world.spawn_reserved(entity, (Counter(7),))?;
            panic!("reserved command panic");
        });

        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = queue.apply(&mut world);
        }));

        assert!(result.is_err());
        assert_eq!(world.len(), 0);
        assert_eq!(world.entities().reserved_len(), 0);
    }

    #[test]
    fn deferred_resource_operations_preserve_their_order() {
        fn mutate(mut commands: Commands) {
            commands.init_resource::<Config>();
            commands.modify_resource::<Config, _>(|config| config.0 += 2);
            commands.modify_or_insert_resource(Config(10), |config| config.0 += 3);
            commands.remove_resource::<Config>();
            commands.modify_resource_if_exists::<Config, _>(|config| config.0 += 100);
            commands.get_resource_or_insert_with(|| Config(7));
            commands.modify_or_insert_resource_with(|| Config(3), |config| config.0 *= 2);
            commands.insert_resource(Config(20));
            commands.modify_resource::<Config, _>(|config| config.0 += 1);
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(mutate);

        schedule.run(&mut world, &mut executor).unwrap();
        assert_eq!(world.get_resource::<Config>(), Some(&Config(21)));
    }

    #[test]
    fn fluent_structural_operations_apply_in_order() {
        fn mutate(mut commands: Commands) {
            commands.spawn((Counter(1), Marker)).remove::<Marker>();
            commands
                .spawn((Counter(2), Marker))
                .retain_bundle::<(Counter,)>();
            commands.spawn((Counter(3),)).clear();
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(mutate);

        schedule.run(&mut world, &mut executor).unwrap();

        assert_eq!(world.len(), 3);
        assert_eq!(counter_values(&world), vec![1, 2]);
        assert_eq!(marker_count(&world), 0);
    }

    #[test]
    fn strict_batch_validates_every_entity_before_mutation() {
        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let entity = world.spawn((Marker,)).unwrap();
        let mut schedule = Schedule::new();

        schedule.add_system(move |mut commands: Commands| {
            commands.despawn_batch([entity, Entity::INVALID]);
        });

        let err = schedule.run(&mut world, &mut executor).unwrap_err();

        let ScheduleError::SystemApplyDeferred { source, .. } = err else {
            panic!("expected deferred apply error");
        };

        let WorldError::Command(error) = source else {
            panic!("expected command error, got {source:?}");
        };

        assert_eq!(error.index(), 0);
        assert_eq!(error.kind(), CommandKind::DespawnBatch);
        assert_eq!(error.entity(), Some(Entity::INVALID));
        assert!(matches!(error.source(), WorldError::EntityNotFound(_)));
        assert!(world.contains(entity));
    }

    #[test]
    fn duplicate_batch_entities_are_reported_without_mutation() {
        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let entity = world.spawn((Marker,)).unwrap();
        let mut schedule = Schedule::new();

        schedule.add_system(move |mut commands: Commands| {
            commands.remove_batch::<_, Marker>([entity, entity]);
        });

        let err = schedule.run(&mut world, &mut executor).unwrap_err();
        let ScheduleError::SystemApplyDeferred { source, .. } = err else {
            panic!("expected deferred command error");
        };
        let WorldError::Command(error) = source else {
            panic!("expected command error");
        };

        assert_eq!(error.kind(), CommandKind::RemoveBatch);
        assert_eq!(error.entity(), Some(entity));
        assert!(matches!(
            error.source(),
            WorldError::Entity(EntityError::Duplicate(duplicate)) if *duplicate == entity
        ));
        assert!(world.get::<Marker>(entity).is_some());
    }

    #[test]
    fn permissive_batches_ignore_missing_entities() {
        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let entity = world.spawn((Marker,)).unwrap();
        let mut schedule = Schedule::new();

        schedule.add_system(move |mut commands: Commands| {
            commands.despawn_batch_if_alive([Entity::INVALID, entity]);
        });

        schedule.run(&mut world, &mut executor).unwrap();
        assert!(!world.contains(entity));
    }

    #[test]
    fn strict_clear_batch_applies_nothing_when_an_entity_is_dead() {
        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let entity = world.spawn((Counter(1), Marker)).unwrap();
        let mut schedule = Schedule::new();

        schedule.add_system(move |mut commands: Commands| {
            commands.clear_batch([entity, Entity::INVALID]);
        });

        let err = schedule.run(&mut world, &mut executor).unwrap_err();
        let ScheduleError::SystemApplyDeferred { source, .. } = err else {
            panic!("expected deferred command error");
        };
        let WorldError::Command(error) = source else {
            panic!("expected command error, got {source:?}");
        };

        assert_eq!(error.kind(), CommandKind::ClearBatch);
        assert_eq!(error.entity(), Some(Entity::INVALID));
        assert!(matches!(error.source(), WorldError::EntityNotFound(_)));
        // The live entity was never touched: a strict batch is all-or-nothing.
        assert_eq!(world.get::<Counter>(entity), Some(&Counter(1)));
        assert!(world.get::<Marker>(entity).is_some());
    }

    #[test]
    fn permissive_retain_bundle_batch_skips_dead_entities() {
        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let first = world.spawn((Counter(1), Marker)).unwrap();
        let second = world.spawn((Counter(2), Marker)).unwrap();
        let mut schedule = Schedule::new();

        schedule.add_system(move |mut commands: Commands| {
            commands
                .retain_bundle_batch_if_alive::<_, (Counter,)>([first, Entity::INVALID, second]);
        });

        schedule.run(&mut world, &mut executor).unwrap();

        assert_eq!(counter_values(&world), vec![1, 2]);
        assert_eq!(marker_count(&world), 0);
    }

    #[test]
    fn remove_bundle_batch_rejects_duplicates_before_mutating() {
        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let entity = world.spawn((Counter(1), Marker)).unwrap();
        let mut schedule = Schedule::new();

        schedule.add_system(move |mut commands: Commands| {
            commands.remove_bundle_batch::<_, (Marker,)>([entity, entity]);
        });

        let err = schedule.run(&mut world, &mut executor).unwrap_err();
        let ScheduleError::SystemApplyDeferred { source, .. } = err else {
            panic!("expected deferred command error");
        };
        let WorldError::Command(error) = source else {
            panic!("expected command error, got {source:?}");
        };

        assert_eq!(error.kind(), CommandKind::RemoveBundleBatch);
        assert!(matches!(
            error.source(),
            WorldError::Entity(EntityError::Duplicate(duplicate)) if *duplicate == entity
        ));
        assert!(world.get::<Marker>(entity).is_some());
    }

    #[test]
    fn permissive_insert_batch_rejects_duplicates() {
        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let entity = world.spawn_empty().unwrap();
        let mut schedule = Schedule::new();

        schedule.add_system(move |mut commands: Commands| {
            commands.insert_batch_if_alive([(entity, (Counter(1),)), (entity, (Counter(2),))]);
        });

        let err = schedule.run(&mut world, &mut executor).unwrap_err();
        let ScheduleError::SystemApplyDeferred { source, .. } = err else {
            panic!("expected deferred command error");
        };
        let WorldError::Command(error) = source else {
            panic!("expected command error");
        };

        assert_eq!(error.kind(), CommandKind::InsertBatch);
        assert_eq!(error.entity(), Some(entity));
        assert!(world.get::<Counter>(entity).is_none());
    }

    #[test]
    fn custom_commands_are_applied_in_order() {
        #[derive(Debug, PartialEq, Eq)]
        struct Phase(i32);

        fn apply(mut commands: Commands) {
            assert!(commands.is_empty());
            commands.reserve_commands(2);
            commands.add_infallible(|world| {
                world.insert_resource(Phase(1));
            });
            commands.add(|world| {
                world.get_resource_mut::<Phase>().unwrap().0 += 1;
                Ok(())
            });
            assert_eq!(commands.len(), 2);
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(apply);

        schedule.run(&mut world, &mut executor).unwrap();
        assert_eq!(world.get_resource::<Phase>(), Some(&Phase(2)));
    }

    #[test]
    fn failing_command_discards_later_reservations() {
        fn fail_before_spawn(mut commands: Commands) {
            commands.add(|_| Err(WorldError::EntityNotFound(Entity::INVALID)));
            commands.spawn((Counter(1),));
        }

        let mut world = setup_world();
        let mut executor = ScheduleExecutor::default();

        let mut schedule = Schedule::new();
        schedule.add_system(fail_before_spawn);

        assert!(matches!(
            schedule.run(&mut world, &mut executor),
            Err(ScheduleError::SystemApplyDeferred { .. })
        ));

        assert_eq!(world.len(), 0);
        assert_eq!(world.entities().reserved_len(), 0);
    }

    #[test]
    fn failed_reserved_command_discards_its_own_reservation() {
        fn spawn(mut commands: Commands) {
            commands.spawn((Unregistered,));
        }

        let mut world = World::new();
        let mut executor = ScheduleExecutor::default();
        let mut schedule = Schedule::new();
        schedule.add_system(spawn);

        let err = schedule.run(&mut world, &mut executor).unwrap_err();
        let ScheduleError::SystemApplyDeferred { source, .. } = err else {
            panic!("expected deferred command error");
        };
        let WorldError::Command(error) = source else {
            panic!("expected command error");
        };

        assert_eq!(error.kind(), CommandKind::Spawn);
        assert_eq!(world.len(), 0);
        assert_eq!(world.entities().reserved_len(), 0);
    }

    #[test]
    fn discarding_commands_state_releases_pending_reservations() {
        let mut world = setup_world();
        let mut state = <Commands<'_, '_> as SystemParam>::init_state(&mut world).unwrap();

        {
            let cell = world.execution_world();
            let mut commands =
                unsafe { <Commands<'_, '_> as SystemParam>::get_param(cell, &mut state).unwrap() };

            commands.reserve_entities(2);
        }

        assert_eq!(world.entities().reserved_len(), 2);
        <Commands<'_, '_> as SystemParam>::discard(&mut world, &mut state);

        assert_eq!(world.len(), 0);
        assert_eq!(world.entities().reserved_len(), 0);
    }

    #[test]
    fn system_rejects_multiple_commands_parameters() {
        fn invalid(_a: Commands, _b: Commands) {}

        let mut world = setup_world();

        let mut schedule = Schedule::new();
        schedule.add_system(invalid);

        let err = schedule.initialize(&mut world).unwrap_err();

        assert!(matches!(err, ScheduleError::SystemInitialize { .. }));
    }
}
