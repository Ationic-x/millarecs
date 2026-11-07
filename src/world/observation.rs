//! Event registration and observation APIs on [`World`].
use super::{ChangeDispatchFor, World, WorldId};
use crate::{
    change_detection::{ChangeTick, Mut},
    component::Component,
    entity::Entity,
    event::{Despawned, Event, Events, Mutated, Spawned},
    query::filter::Changed,
    world::WorldError,
};
use std::{any::TypeId, sync::Arc};

impl World {
    /// Registers the `Events<E>` resource and its cycle retirement.
    pub fn add_event<E: Event>(&mut self) -> Mut<'_, Events<E>> {
        self.register_event_cycle::<E>();
        self.init_resource::<Events<E>>()
    }

    /// Observes entity spawns, publishing `Spawned` events.
    ///
    /// Flips the watch flag and registers the `Events<Spawned>` resource;
    /// from then on every spawn site publishes one event.
    pub fn observe_spawned(&mut self) {
        self.lifecycle_watch.spawned = true;
        self.add_event::<Spawned>();
    }

    /// Observes entity despawns, publishing `Despawned` events.
    ///
    /// Flips the watch flag and registers the `Events<Despawned>` resource;
    /// from then on every despawn site publishes one event.
    pub fn observe_despawned(&mut self) {
        self.lifecycle_watch.despawned = true;
        self.add_event::<Despawned>();
    }

    /// Registers change observation for component `C`.
    ///
    /// Every cycle, the entities whose `C` changed since the previous
    /// dispatch are published as one `Mutated<C>` batch event. Registration
    /// performs one empty query run immediately, which advances the retained
    /// state's change window past every pre-existing mark: the first
    /// dispatch reports only changes that happen after it.
    pub fn observe_changed<C: Component>(&mut self) -> Result<(), WorldError> {
        self.add_event::<Mutated<C>>();

        let id = TypeId::of::<C>();
        if self
            .change_watch
            .iter()
            .any(|(registered, _)| *registered == id)
        {
            return Ok(());
        }

        let mut state = self.query_state::<Entity, Changed<C>>()?;
        state.begin_run(self.id, self.advance_tick())?;
        state.finish_run();

        self.change_watch
            .push((id, Box::new(ChangeDispatchFor { state })));

        Ok(())
    }

    /// Sends `event` when its `Events<E>` resource exists, dropping it
    /// silently otherwise.
    #[inline]
    pub(crate) fn emit_lifecycle<E: Event>(&mut self, event: E) {
        if let Some(mut events) = self.get_resource_mut::<Events<E>>() {
            events.send(event);
        }
    }

    /// Dispatches change observation: for every observed component, the
    /// entities whose component changed since the previous dispatch are
    /// published as one `Mutated<C>` batch event.
    ///
    /// The watcher list is temporarily taken out of the world (dispatch
    /// needs `&mut World` while iterating it) and restored afterwards even
    /// when a dispatch fails.
    ///
    /// Called automatically at the start of every schedule run, after
    /// [`Self::event_update`]; cycles without changes publish nothing.
    pub fn dispatch_changed(&mut self) -> Result<(), WorldError> {
        if self.change_watch.is_empty() {
            return Ok(());
        }

        let _tick = self.advance_tick();

        let mut watchers = std::mem::take(&mut self.change_watch);
        let result = (|| {
            for (_, watcher) in &mut watchers {
                watcher.dispatch(self)?;
            }

            Ok(())
        })();
        self.change_watch = watchers;

        result
    }

    /// Registers the retirement callback for `Events<E>`, deduplicated by
    /// type.
    ///
    /// The callback retires that buffer inside [`Self::event_update`]; it is
    /// stored behind an `Arc` because the update loop must clone it out to
    /// free the world borrow while the callback runs.
    fn register_event_cycle<E: Event>(&mut self) {
        let id = TypeId::of::<Events<E>>();

        if self
            .event_cycles
            .iter()
            .any(|(registered, _)| *registered == id)
        {
            return;
        }

        self.event_cycles.push((
            id,
            Arc::new(|world: &mut World| {
                if let Some(mut events) = world.get_resource_mut::<Events<E>>() {
                    events.update();
                }
            }),
        ));
    }

    /// Advances the event cycle: readable events move one step closer to
    /// retirement and everything older is dropped.
    ///
    /// Called automatically at the start of every schedule run; call it
    /// manually when driving the world without a schedule.
    pub fn event_update(&mut self) {
        for index in 0..self.event_cycles.len() {
            // Cloning the Arc releases the borrow on `event_cycles` while
            // the callback mutates the world.
            let cycle = Arc::clone(&self.event_cycles[index].1);
            cycle(self);
        }
    }

    /// Returns the process-unique identifier assigned to this world.
    #[inline]
    pub const fn id(&self) -> WorldId {
        self.id
    }

    /// Returns `Ok` when `expected` names this world, and a mismatch error
    /// naming both worlds otherwise.
    #[inline]
    pub(crate) fn validate_world_id(&self, expected: WorldId) -> Result<(), WorldError> {
        if self.id == expected {
            Ok(())
        } else {
            Err(WorldError::WorldMismatch {
                expected,
                actual: self.id,
            })
        }
    }

    /// Advances the world change tick and returns the new value.
    ///
    /// Mutating entry points call this first so the change they are about to
    /// make lands in a fresh tick window.
    pub(crate) fn advance_tick(&mut self) -> ChangeTick {
        self.change_tick = self.change_tick.next();
        self.change_tick
    }
}
