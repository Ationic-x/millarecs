//! Documents the lifecycle-event semantics of millarecs' spawn/despawn
//! command paths.
//!
//! millarecs fuses adjacent spawn commands into one batch above
//! `MIN_FUSED_SPAWN_COMMANDS` (32) and adjacent despawns above
//! `MIN_FUSED_DESPAWN_COMMANDS` (256). The scalar paths publish one
//! `Spawned`/`Despawned` event per entity; these tests pin down what the
//! batch paths do, so demo counters are never misread as a bug.

use millarecs::entity::Entity;
use millarecs::event::{Despawned, Events, Spawned};
use millarecs::system::commands::Commands;
use millarecs::system::query::Query;
use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::World;

use std::collections::HashMap;

use millarecs_playground::app::{Args, Demo, FrameCtx};
use millarecs_playground::demos::particles::Particles;
use millarecs_playground::panel::PanelValues;
use millarecs_playground::time::{SimBounds, Time, FIXED_DT};

fn retained<E: millarecs::event::Event>(world: &World) -> usize {
    world
        .get_resource::<Events<E>>()
        .map(|events| events.len())
        .unwrap_or(0)
}

#[test]
fn scalar_burst_publishes_per_entity_lifecycle_events() {
    let mut world = World::new();
    let mut schedule = Schedule::new();
    let mut executor = ScheduleExecutor::single_threaded();
    world.insert_resource(Time { dt: FIXED_DT });
    world.insert_resource(SimBounds {
        width: 800.0,
        height: 600.0,
    });
    let mut demo = Particles::new(&mut world, &mut schedule, &mut executor, &Args::default());
    world.insert_resource(PanelValues {
        values: HashMap::from([("rate", 60.0), ("burst", 26.0)]),
    });

    demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
    // Events are readable in the cycle they were sent plus the next one.
    assert_eq!(
        retained::<Spawned>(&world),
        26,
        "26 spawns below the fused threshold publish 26 Spawned events"
    );
}

#[test]
fn fused_burst_is_silent_by_design() {
    let mut world = World::new();
    let mut schedule = Schedule::new();
    let mut executor = ScheduleExecutor::single_threaded();
    world.insert_resource(Time { dt: FIXED_DT });
    world.insert_resource(SimBounds {
        width: 800.0,
        height: 600.0,
    });
    let mut demo = Particles::new(&mut world, &mut schedule, &mut executor, &Args::default());
    world.insert_resource(PanelValues {
        values: HashMap::from([("rate", 100.0), ("burst", 7030.0)]),
    });

    demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
    // The emitter's while-loop pays off the timer debt, so the first tick can
    // fire more than one burst at high rates; every burst is `burst` large.
    assert!(
        world.len() >= 7030 && world.len() % 7030 == 0,
        "whole bursts were applied, got {}",
        world.len()
    );
    assert_eq!(
        retained::<Spawned>(&world),
        0,
        "the fused spawn batch publishes no per-entity Spawned events"
    );
}

/// Marker component for the despawn-path probes.
#[derive(millarecs_macros::Component)]
struct Probe;

fn despawn_everything(mut commands: Commands, mut q: Query<Entity>) {
    q.for_each(|entity| commands.despawn(entity));
}

fn despawn_session(count: usize) -> World {
    let mut world = World::new();
    world.register_table::<Probe>().expect("register Probe");
    world.observe_despawned();
    world
        .spawn_batch((0..count).map(|_| (Probe,)))
        .expect("spawn probes");

    let mut schedule = Schedule::new();
    schedule.add_system(despawn_everything);
    schedule.add_deferred_barrier();
    let mut executor = ScheduleExecutor::single_threaded();
    schedule.run(&mut world, &mut executor).expect("run");
    world
}

#[test]
fn despawn_events_follow_the_same_fusion_threshold() {
    // Below 256 adjacent despawns: the scalar path publishes per entity.
    let scalar = despawn_session(100);
    assert_eq!(scalar.len(), 0);
    assert_eq!(
        retained::<Despawned>(&scalar),
        100,
        "100 adjacent despawns stay scalar and publish 100 Despawned events"
    );

    // At or above 256: the commands fuse into one despawn batch.
    let fused = despawn_session(1_000);
    assert_eq!(fused.len(), 0, "everyone is gone either way");
    println!(
        "despawned retained after a 1000-entity batch: {}",
        retained::<Despawned>(&fused)
    );
    assert_eq!(
        retained::<Despawned>(&fused),
        0,
        "the fused despawn batch publishes no per-entity Despawned events"
    );
}
