//! Reusable world and schedule setup.

use crate::{system::schedule::Schedule, world::World};

/// Reusable setup for a world and a schedule.
///
/// Plugins group registration into named pieces an application composes,
/// instead of one long build function. Types and closures alike implement
/// it:
///
/// ```
/// use millarecs::{plugin::Plugin, system::schedule::Schedule, world::World};
///
/// struct TimePlugin;
///
/// impl Plugin for TimePlugin {
///     fn build(&mut self, world: &mut World, schedule: &mut Schedule) {
///         world.insert_resource(Time { delta: 0.016 });
///         # struct Time { delta: f32 }
///     }
/// }
/// ```
pub trait Plugin {
    /// Configures `world` and `schedule` with this plugin's resources,
    /// components, and systems.
    fn build(&mut self, world: &mut World, schedule: &mut Schedule);
}

impl<F> Plugin for F
where
    F: FnMut(&mut World, &mut Schedule),
{
    /// Delegates to the closure body.
    fn build(&mut self, world: &mut World, schedule: &mut Schedule) {
        self(world, schedule);
    }
}

#[cfg(test)]
mod tests {
    use super::Plugin;
    use crate::{
        system::{
            resource::ResMut,
            schedule::{Schedule, ScheduleExecutor},
        },
        world::World,
    };

    #[test]
    fn plugins_configure_world_and_schedule() {
        struct CountPlugin;

        struct Count(u32);

        impl Plugin for CountPlugin {
            fn build(&mut self, world: &mut World, schedule: &mut Schedule) {
                world.insert_resource(Count(10));
                schedule.add_system(|mut count: ResMut<Count>| count.0 += 1);
            }
        }

        let mut world = World::new();
        let mut schedule = Schedule::new();
        CountPlugin.build(&mut world, &mut schedule);

        let mut executor = ScheduleExecutor::new(2);
        schedule.run(&mut world, &mut executor).unwrap();

        assert_eq!(world.get_resource::<Count>().unwrap().0, 11);
    }

    #[test]
    fn closures_implement_plugin() {
        struct Setup(bool);

        let mut plugin = |world: &mut World, schedule: &mut Schedule| {
            world.insert_resource(Setup(true));
            schedule.add_system(|| {});
        };

        let mut world = World::new();
        let mut schedule = Schedule::new();
        plugin.build(&mut world, &mut schedule);

        assert!(world.get_resource::<Setup>().unwrap().0);
    }
}
