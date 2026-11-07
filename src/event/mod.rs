//! Typed event storage used by event reader and writer system parameters.
//!
//! Events live for exactly two cycles: the cycle they were sent in and the
//! one after. Each [`Events::update`] retires the older buffer, so memory
//! stays bounded by the write rate instead of growing forever. Identifiers
//! increase monotonically across updates; readers persist a cursor and
//! silently skip anything older than the retired window.

use std::marker::PhantomData;
use std::slice::Iter;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::component::Component;
use crate::entity::Entity;

/// Marker trait for values that can be sent through the ECS event system.
pub trait Event: Send + Sync + 'static {}

impl<T> Event for T where T: Send + Sync + 'static {}

/// Global counter assigning every `Events` resource a unique identity.
///
/// Reader cursors persist an epoch so a resource that is removed and
/// re-created is never confused with its predecessor, however the
/// identifiers line up. Starts at 1: a fresh cursor (epoch 0) never matches.
static NEXT_EVENT_EPOCH: AtomicU64 = AtomicU64::new(1);

/// One retained event paired with its monotonic identifier.
pub(crate) struct EventRecord<E> {
    /// Monotonic identifier compared against reader cursors.
    pub id: u64,
    pub value: E,
}

/// Ordered storage for events of one concrete type.
///
/// Writers append during the exclusive deferred phase into the current
/// buffer. [`Events::update`] moves that buffer to previous and drops the
/// records it replaces; readers use independent cursors to consume each
/// record once while it remains available.
pub struct Events<E: Event> {
    previous: Vec<EventRecord<E>>,
    current: Vec<EventRecord<E>>,
    epoch: u64,
    next_event_id: u64,
}

impl<E: Event> Default for Events<E> {
    fn default() -> Self {
        Self {
            previous: Vec::new(),
            current: Vec::new(),
            epoch: NEXT_EVENT_EPOCH.fetch_add(1, Ordering::Relaxed),
            next_event_id: 0,
        }
    }
}

impl<E: Event> Events<E> {
    /// Reserves space for at least `additional` more events.
    pub fn reserve(&mut self, additional: usize) {
        self.current.reserve(additional);
    }

    /// Retires the current buffer and starts a new event cycle.
    ///
    /// Records sent in the finished cycle remain readable for one more
    /// cycle; everything older is dropped. Called once per cycle by
    /// [`crate::world::World::event_update`].
    pub fn update(&mut self) {
        self.previous = std::mem::take(&mut self.current);
    }

    /// Removes every retained event.
    ///
    /// Identifiers keep increasing, so reader cursors clamp forward and
    /// simply observe an empty window until new events arrive.
    pub fn clear(&mut self) {
        self.previous.clear();
        self.current.clear();
    }

    /// Appends every event yielded by `events` in iteration order.
    pub fn send_batch<I>(&mut self, events: I)
    where
        I: IntoIterator<Item = E>,
    {
        let iter = events.into_iter();
        let (lower_bound, _) = iter.size_hint();

        self.current.reserve(lower_bound);

        for event in iter {
            self.send(event);
        }
    }

    /// Appends one event and assigns its identifier.
    pub fn send(&mut self, event: E) {
        let id = self.next_event_id;

        self.next_event_id = self
            .next_event_id
            .checked_add(1)
            .expect("event identifier overflow");

        self.current.push(EventRecord { id, value: event });
    }

    /// Returns the identifier of the oldest readable event, or the next
    /// identifier when nothing is readable.
    pub(crate) fn window_start(&self) -> u64 {
        self.previous
            .first()
            .or_else(|| self.current.first())
            .map(|record| record.id)
            .unwrap_or(self.next_event_id)
    }

    /// Returns the number of retained events (previous + current).
    pub fn len(&self) -> usize {
        self.previous.len() + self.current.len()
    }

    /// Returns whether no events are currently retained.
    pub fn is_empty(&self) -> bool {
        self.previous.is_empty() && self.current.is_empty()
    }

    /// Returns the identity epoch of this resource.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch
    }

    /// Returns the identifier the next sent event will receive.
    pub(crate) fn next_event_id(&self) -> u64 {
        self.next_event_id
    }

    /// Iterator over every readable record with `id >= event_id`,
    /// previous buffer first.
    pub(crate) fn records_from(&self, event_id: u64) -> RecordsFrom<'_, E> {
        let previous_offset = self.previous.partition_point(|record| record.id < event_id);
        let current_offset = self.current.partition_point(|record| record.id < event_id);

        RecordsFrom {
            previous: self.previous[previous_offset..].iter(),
            current: self.current[current_offset..].iter(),
        }
    }
}

/// Chained iterator over the readable records of both buffers.
///
/// Every identifier in `previous` is lower than every identifier in
/// `current`, so plain concatenation preserves order.
pub(crate) struct RecordsFrom<'a, E: Event> {
    previous: Iter<'a, EventRecord<E>>,
    current: Iter<'a, EventRecord<E>>,
}

impl<'a, E: Event> Iterator for RecordsFrom<'a, E> {
    type Item = &'a EventRecord<E>;

    fn next(&mut self) -> Option<Self::Item> {
        self.previous.next().or_else(|| self.current.next())
    }
}

/// Emitted for every spawned entity when observed through
/// [`crate::world::World::observe_spawned`].
#[derive(Clone, Copy, Debug)]
pub struct Spawned {
    /// The handle of the entity that was spawned.
    pub entity: Entity,
}

/// Emitted for every despawned entity when observed through
/// [`crate::world::World::observe_despawned`].
#[derive(Clone, Copy, Debug)]
pub struct Despawned {
    /// The handle of the entity that was despawned.
    pub entity: Entity,
}

/// Batch of entities whose component `C` changed during a cycle, emitted
/// when observed through [`crate::world::World::observe_changed`].
///
/// One event is published per observing cycle with at least one change:
/// reactions see the whole change set instead of one notification per
/// entity, and cycles without changes publish nothing.
pub struct Mutated<C: Component> {
    /// Every live entity whose `C` component changed during the reported
    /// cycle, in no guaranteed order.
    pub entities: Box<[Entity]>,
    pub(crate) _marker: PhantomData<fn() -> C>,
}

// TODO: per-component Inserted<C>/Removed<C> batches alongside
// Mutated<C>; component adds and drops are still silent changes.
