//! Read-only execution view of a world.
//!
//! Systems receive this value while they run. It exposes component/resource
//! storage and entity reservation, but leaves out the structural mutation
//! APIs so those changes queue up and apply in a later exclusive phase.

use crate::{
    change_detection::ChangeTick,
    component::ComponentManager,
    entity::{EntityAllocator, EntityManager},
    planning::WorldPlanning,
    resource::ResourceManager,
    task::TaskExecutorHandle,
    world::WorldId,
};

/// Restricted view used while systems are executing.
///
/// It exposes value storage but no structural mutation. The view is `Copy`,
/// so a parallel schedule can hand it to every worker task unchanged.
#[derive(Clone, Copy)]
pub struct ExecutionWorld<'w> {
    id: WorldId,
    components: &'w ComponentManager,
    planning: &'w WorldPlanning,
    resources: &'w ResourceManager,
    entities: &'w EntityManager,
    tasks: Option<&'w TaskExecutorHandle>,
    change_tick: ChangeTick,
}

impl<'w> ExecutionWorld<'w> {
    /// Creates an execution view from world internals.
    pub(crate) fn new(
        id: WorldId,
        components: &'w ComponentManager,
        planning: &'w WorldPlanning,
        resources: &'w ResourceManager,
        entities: &'w EntityManager,
        tasks: Option<&'w TaskExecutorHandle>,
        change_tick: ChangeTick,
    ) -> Self {
        Self {
            id,
            components,
            planning,
            resources,
            entities,
            tasks,
            change_tick,
        }
    }

    #[inline]
    /// Returns component storage available during system execution.
    pub(crate) fn components(self) -> &'w ComponentManager {
        self.components
    }

    #[inline]
    /// Returns planning metadata visible during system execution.
    pub(crate) fn planning(self) -> &'w WorldPlanning {
        self.planning
    }

    #[inline]
    /// Returns resource storage available during system execution.
    pub(crate) fn resources(self) -> &'w ResourceManager {
        self.resources
    }

    #[inline]
    /// Returns the shared entity reservation allocator.
    ///
    /// Systems use this to reserve entity ids without touching location
    /// metadata; the full manager, including live storage locations, lives
    /// behind [`Self::entity_locations`].
    pub(crate) fn entities(self) -> &'w EntityAllocator {
        self.entities.allocator()
    }

    #[inline]
    /// Returns entity metadata, including live storage locations. This is
    /// the read side of [`Self::entities`]: reservation allocates handles,
    /// activation publishes their locations here.
    pub(crate) fn entity_locations(self) -> &'w EntityManager {
        self.entities
    }

    #[inline]
    /// Returns the world tick for the current execution.
    pub(crate) fn change_tick(self) -> ChangeTick {
        self.change_tick
    }

    /// Returns the unique identifier assigned to this world.
    #[inline]
    pub const fn id(self) -> WorldId {
        self.id
    }

    /// Returns the shared task executor handle, if the schedule is parallel.
    pub(crate) fn task_executor(self) -> Option<&'w TaskExecutorHandle> {
        self.tasks
    }
}

#[cfg(test)]
mod tests {
    use super::ExecutionWorld;

    fn assert_copy_send_sync<T: Copy + Send + Sync>() {}

    #[test]
    fn execution_world_is_copy_send_and_sync() {
        assert_copy_send_sync::<ExecutionWorld<'static>>();
    }
}
