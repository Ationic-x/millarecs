//! Deferred event writers and cursor-based event readers.
//!
//! Each writer records events in system-local state while systems run. The
//! scheduler merges that state into [`Events`] during an exclusive deferred
//! phase, allowing producers of the same event type to execute concurrently.

use std::marker::PhantomData;

use crate::{
    event::{Event, Events, RecordsFrom},
    resource::{ResourceError, ResourceId},
    system::{param::SystemParam, SystemAccess},
    world::{execution_world::ExecutionWorld, World, WorldError},
};

/// System parameter that buffers events until the deferred phase.
///
/// The writer declares no resource access: events accumulate in system-local
/// state and reach the [`Events`] resource only when the scheduler applies the
/// buffer in its exclusive deferred phase. Keeping the run itself access-free
/// is what lets several writers of the same event type run concurrently.
pub struct EventWriter<'s, E: Event> {
    pending: &'s mut Vec<E>,
}

impl<E: Event> EventWriter<'_, E> {
    /// Queues one event for deferred publication.
    pub fn send(&mut self, event: E) {
        self.pending.push(event);
    }

    /// Queues a batch of events while preserving iteration order.
    pub fn send_batch<I>(&mut self, events: I)
    where
        I: IntoIterator<Item = E>,
    {
        self.pending.extend(events);
    }
}

/// Cached local buffer and access metadata for an [`EventWriter`].
pub struct EventWriterState<E: Event> {
    /// Always empty; the writer touches no resource during the run.
    access: SystemAccess,
    /// Events queued by the current run, drained by `apply`.
    pending: Vec<E>,
}

unsafe impl<E: Event> SystemParam for EventWriter<'_, E> {
    type State = EventWriterState<E>;
    type Item<'w, 's> = EventWriter<'s, E>;

    /// Creates an empty buffer with empty access.
    fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
        Ok(EventWriterState {
            access: SystemAccess::default(),
            pending: Vec::new(),
        })
    }

    /// Returns the empty access set.
    fn access(state: &Self::State) -> &SystemAccess {
        &state.access
    }

    /// Lends the local buffer after checking that the [`Events`] resource
    /// exists, so buffered events always have a destination.
    unsafe fn get_param<'w, 's>(
        world: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        if !world.resources().contains::<Events<E>>() {
            return Err(ResourceError::Missing(ResourceId::of::<Events<E>>()).into());
        }

        Ok(EventWriter {
            pending: &mut state.pending,
        })
    }

    /// Returns whether any event is still buffered.
    fn has_deferred_work(state: &Self::State) -> bool {
        !state.pending.is_empty()
    }

    /// Drains the buffer into the world's [`Events`] resource during the
    /// deferred phase.
    fn apply(world: &mut World, state: &mut Self::State) -> Result<(), WorldError> {
        let mut events = world
            .get_resource_mut::<Events<E>>()
            .ok_or(ResourceError::Missing(ResourceId::of::<Events<E>>()))?;

        events.send_batch(state.pending.drain(..));
        Ok(())
    }

    /// Drops buffered events after a failed run.
    fn discard(_: &mut World, state: &mut Self::State) {
        state.pending.clear();
    }
}

/// System parameter that reads unseen events using a persistent cursor.
///
/// Unlike the writer, the reader borrows the [`Events`] resource during the
/// run and therefore declares a read of it.
pub struct EventReader<'w, 's, E: Event> {
    events: &'w Events<E>,
    cursor: &'s mut EventCursor,
}

impl<E: Event> EventReader<'_, '_, E> {
    /// Clamps the cursor into the currently readable window.
    ///
    /// A cursor from a different epoch belongs to an earlier incarnation of
    /// the resource, so its position is meaningless and restarts at the window
    /// start. Otherwise the position is clamped to the window: retired events
    /// below the start are skipped and positions past the newest event are
    /// pulled back.
    fn sync_cursor(&mut self) {
        let window_start = self.events.window_start();

        if self.cursor.epoch != self.events.epoch() {
            // A different resource epoch means every persisted position is
            // meaningless: start at the beginning of the readable window.
            self.cursor.epoch = self.events.epoch();
            self.cursor.next_event_id = window_start;
            return;
        }

        self.cursor.next_event_id = self
            .cursor
            .next_event_id
            .min(self.events.next_event_id())
            .max(window_start);
    }

    /// Returns an iterator over events not yet consumed by this reader.
    pub fn read(&mut self) -> EventIter<'_, E> {
        self.sync_cursor();

        EventIter {
            records: self.events.records_from(self.cursor.next_event_id),
            cursor: self.cursor,
        }
    }

    /// Returns whether this reader has no unseen events.
    pub fn is_empty(&mut self) -> bool {
        self.sync_cursor();
        self.cursor.next_event_id >= self.events.next_event_id()
    }
}

/// Reader position within one event resource.
///
/// The epoch ties the cursor to a specific resource incarnation; the
/// identifier marks the next record the reader has not consumed.
struct EventCursor {
    epoch: u64,
    next_event_id: u64,
}
/// Cached cursor and access metadata for an [`EventReader`].
pub struct EventReaderState<E: Event> {
    /// Declares the read of the `Events<E>` resource the reader borrows.
    access: SystemAccess,
    /// Read position persisted across runs of the system.
    cursor: EventCursor,
    _marker: PhantomData<E>,
}

unsafe impl<'a, E: Event> SystemParam for EventReader<'a, '_, E> {
    type State = EventReaderState<E>;
    type Item<'w, 's> = EventReader<'w, 's, E>;

    /// Declares the read of the [`Events`] resource and resets the cursor.
    fn init_state(_: &mut World) -> Result<Self::State, WorldError> {
        let cursor = EventCursor {
            epoch: 0,
            next_event_id: 0,
        };

        Ok(EventReaderState {
            access: SystemAccess::from_resource_read(ResourceId::of::<Events<E>>())?,
            cursor,
            _marker: PhantomData,
        })
    }

    /// Returns the declared read access.
    fn access(state: &Self::State) -> &SystemAccess {
        &state.access
    }

    /// Borrows the [`Events`] resource, failing with `Missing` when absent.
    unsafe fn get_param<'w, 's>(
        world: ExecutionWorld<'w>,
        state: &'s mut Self::State,
    ) -> Result<Self::Item<'w, 's>, WorldError> {
        let resources = world.resources();

        let events = resources
            .get::<Events<E>>()
            .ok_or_else(|| ResourceError::Missing(ResourceId::of::<Events<E>>()))?;

        Ok(EventReader {
            events,
            cursor: &mut state.cursor,
        })
    }
}

/// Iterator that advances an event reader cursor as records are consumed.
pub struct EventIter<'a, E: Event> {
    records: RecordsFrom<'a, E>,
    cursor: &'a mut EventCursor,
}

impl<'a, E: Event> Iterator for EventIter<'a, E> {
    type Item = &'a E;

    fn next(&mut self) -> Option<Self::Item> {
        let record = self.records.next()?;

        self.cursor.next_event_id = record.id + 1;

        Some(&record.value)
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        event::Events,
        system::{
            event::{EventCursor, EventReader, EventWriter},
            param::SystemParam,
            resource::ResMut,
            schedule::{Schedule, ScheduleExecutor},
        },
        world::World,
    };

    #[test]
    fn reader_delivers_each_event_once() {
        let mut events = Events::<u32>::default();

        events.send(10);
        events.send(20);

        let mut cursor = EventCursor {
            epoch: 0,
            next_event_id: 0,
        };

        let first: Vec<_> = {
            let mut reader = EventReader {
                events: &events,
                cursor: &mut cursor,
            };

            reader.read().copied().collect()
        };

        assert_eq!(first, vec![10, 20]);

        let second: Vec<_> = {
            let mut reader = EventReader {
                events: &events,
                cursor: &mut cursor,
            };

            reader.read().copied().collect()
        };

        assert!(second.is_empty());
    }

    #[test]
    fn writer_buffers_events_until_deferred_application() {
        type Writer = EventWriter<'static, u32>;

        let mut world = World::new();
        world.add_event::<u32>();
        let mut state = <Writer as SystemParam>::init_state(&mut world).unwrap();

        {
            let mut writer =
                unsafe { <Writer as SystemParam>::get_param(world.execution_world(), &mut state) }
                    .unwrap();
            writer.send_batch([10, 20]);
        }

        assert!(world.get_resource::<Events<u32>>().unwrap().is_empty());
        assert!(<Writer as SystemParam>::has_deferred_work(&state));

        <Writer as SystemParam>::apply(&mut world, &mut state).unwrap();

        assert_eq!(world.get_resource::<Events<u32>>().unwrap().len(), 2);
        assert!(!<Writer as SystemParam>::has_deferred_work(&state));
    }

    #[test]
    fn dependency_observes_events_after_the_writer_deferred_phase() {
        struct EventsWritten;

        fn write(mut writer: EventWriter<u32>) {
            writer.send(7);
        }

        fn read(mut reader: EventReader<u32>, mut observed: ResMut<Vec<u32>>) {
            observed.extend(reader.read().copied());
        }

        let mut world = World::new();
        world.add_event::<u32>();
        world.insert_resource(Vec::<u32>::new());

        let mut schedule = Schedule::new();
        schedule.add_system(write).label::<EventsWritten>();
        schedule.add_system(read).after::<EventsWritten>();

        let mut executor = ScheduleExecutor::new(2);
        schedule.run(&mut world, &mut executor).unwrap();

        assert_eq!(world.get_resource::<Vec<u32>>().unwrap(), &[7]);
    }

    #[test]
    fn update_retires_events_after_two_windows() {
        let mut events = Events::<u32>::default();
        let mut cursor = EventCursor {
            epoch: 0,
            next_event_id: 0,
        };

        let read = |events: &Events<u32>, cursor: &mut EventCursor| {
            let mut reader = EventReader { events, cursor };
            reader.read().copied().collect::<Vec<_>>()
        };

        events.send(1);
        assert_eq!(read(&events, &mut cursor), vec![1]);

        // Cycle boundary: the event stays readable from the previous buffer.
        events.update();
        assert!(read(&events, &mut cursor).is_empty());

        events.send(2);
        events.update(); // retires event 1; event 2 moves to previous
        assert_eq!(read(&events, &mut cursor), vec![2]);

        events.update(); // retires event 2
        assert!(read(&events, &mut cursor).is_empty());
    }

    #[test]
    fn recreated_event_resource_is_not_confused_with_its_predecessor() {
        let mut events = Events::<u32>::default();
        let mut cursor = EventCursor {
            epoch: 0,
            next_event_id: 0,
        };

        let read = |events: &Events<u32>, cursor: &mut EventCursor| {
            let mut reader = EventReader { events, cursor };
            reader.read().copied().collect::<Vec<_>>()
        };

        events.send(10);
        assert_eq!(read(&events, &mut cursor), vec![10]);

        // A removed-and-recreated resource has a fresh epoch and restarts
        // its identifiers; the stale cursor must reset instead of clamping.
        let mut events = Events::<u32>::default();
        events.send(30);
        assert_eq!(read(&events, &mut cursor), vec![30]);
    }
}
