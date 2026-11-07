//! Solar systems, the hierarchy demo. Suns, planets and moons linked
//! with `Parent`/`Children` + `add_child`, propagated by an exclusive system,
//! removed wholesale with `despawn_with_descendants`.
//!
//! The population follows the `systems` panel parameter live (Space/D step
//! it too, one per press): an exclusive reconcile system compares the number
//! of suns against the target and spawns or despawns whole systems every
//! tick. `planets` caps the planets of systems spawned from then on.

use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::{World, WorldError};

use crate::prelude::*;

/// World-space position, owned by every body.
#[derive(Component, Clone, Copy)]
pub struct Pos(pub Vec2);

/// Position relative to the parent, refreshed from `Orbit` every tick.
#[derive(Component, Clone, Copy)]
pub struct Offset(pub Vec2);

/// Circular orbit parameters around the parent.
#[derive(Component)]
pub struct Orbit {
    /// Distance to the parent in pixels.
    pub dist: f32,
    /// Current angle in radians.
    pub angle: f32,
    /// Angular speed in radians per second (sign = direction).
    pub speed: f32,
}

/// Visual radius and color of a body.
#[derive(Component, Clone, Copy)]
pub struct Body {
    /// Draw radius in pixels.
    pub radius: f32,
    /// Fill color.
    pub color: Rgba,
}

/// Root marker: bodies without a [`Parent`].
#[derive(Component)]
pub struct Sun;

/// Cumulative count of `Despawned` lifecycle events, shown in the HUD.
#[derive(Default)]
struct DespawnStats {
    total: u64,
}

fn advance(mut q: Query<&mut Orbit>, time: Res<Time>) {
    let start = std::time::Instant::now();
    let dt = time.dt;
    q.for_each_mut(|mut orbit| {
        orbit.angle += orbit.speed * dt;
    });
    perf::record("advance", start.elapsed().as_nanos() as u64);
}

fn local_offsets(mut q: Query<(&Orbit, &mut Offset)>) {
    let start = std::time::Instant::now();
    q.for_each_mut(|(orbit, mut offset)| {
        offset.0 = Vec2::from_angle(orbit.angle) * orbit.dist;
    });
    perf::record("offsets", start.elapsed().as_nanos() as u64);
}

/// Exclusive system: walks every hierarchy from its root and writes world
/// positions as parent position + child offset. Runs alone in the schedule.
fn propagate(world: &mut World) -> Result<(), WorldError> {
    let start = std::time::Instant::now();
    let roots: Vec<Entity> = {
        let mut q = world.query::<Entity, Without<Parent>>()?;
        let mut roots = Vec::new();
        q.for_each(|e| roots.push(e));
        roots
    };
    for root in roots {
        walk(world, root)?;
    }
    perf::record("propagate", start.elapsed().as_nanos() as u64);
    Ok(())
}

fn walk(world: &mut World, entity: Entity) -> Result<(), WorldError> {
    let Some(children) = world.children(entity).map(|c| c.to_vec()) else {
        return Ok(());
    };
    let here = *world
        .get::<Pos>(entity)
        .expect("every orbiting body carries a Pos");
    for child in children {
        if let Some(offset) = world.get::<Offset>(child) {
            let target = here.0 + offset.0;
            if let Some(mut pos) = world.get_mut::<Pos>(child) {
                pos.0 = target;
            }
        }
        walk(world, child)?;
    }
    Ok(())
}

fn count_despawned(mut reader: EventReader<Despawned>, mut stats: ResMut<DespawnStats>) {
    for _ in reader.read() {
        stats.total += 1;
    }
}

/// Runs `f` with the world's `Rng` taken out of the world and restored after
/// (an exclusive system cannot borrow the world and its resource at once).
fn with_world_rng<F>(world: &mut World, f: F) -> Result<(), WorldError>
where
    F: FnOnce(&mut World, &mut Rng) -> Result<(), WorldError>,
{
    let mut rng = {
        let mut slot = world
            .get_resource_mut::<Rng>()
            .expect("orbits inserts the Rng resource");
        std::mem::take(&mut *slot)
    };
    let result = f(world, &mut rng);
    if let Some(mut slot) = world.get_resource_mut::<Rng>() {
        *slot = rng;
    }
    result
}

fn spawn_solar_system(
    world: &mut World,
    rng: &mut Rng,
    at: Vec2,
    max_planets: usize,
) -> Result<Entity, WorldError> {
    let sun = world.spawn((
        Pos(at),
        Body {
            radius: 15.0,
            color: color::rgb8(255, 208, 84),
        },
        Sun,
    ))?;
    let planets = 2 + rng.usize((max_planets - 1).max(1));
    for _ in 0..planets {
        let planet = world.spawn((
            Pos(Vec2::ZERO),
            Offset(Vec2::ZERO),
            Orbit {
                dist: rng.range(55.0, 210.0),
                angle: rng.angle(),
                speed: rng.range(0.35, 1.1) * if rng.chance(0.5) { 1.0 } else { -1.0 },
            },
            Body {
                radius: rng.range(4.0, 8.0),
                color: color::hsl(rng.f32(), 0.55, 0.6),
            },
        ))?;
        world.add_child(sun, planet)?;
        let moons = rng.usize(3);
        for _ in 0..moons {
            let moon = world.spawn((
                Pos(Vec2::ZERO),
                Offset(Vec2::ZERO),
                Orbit {
                    dist: rng.range(12.0, 30.0),
                    angle: rng.angle(),
                    speed: rng.range(1.5, 3.5),
                },
                Body {
                    radius: rng.range(1.8, 3.2),
                    color: color::rgb8(198, 198, 212),
                },
            ))?;
            world.add_child(planet, moon)?;
        }
    }
    Ok(sun)
}

/// Exclusive system: matches the number of suns to the `systems` panel
/// target, spawning or despawning at most 64 whole solar systems per tick so
/// huge slider jumps ramp instead of hitching.
fn reconcile_systems(world: &mut World) -> Result<(), WorldError> {
    let start = std::time::Instant::now();
    let (target, max_planets) = match world.get_resource::<PanelValues>() {
        Some(values) => (
            values.get_or("systems", 2.0).clamp(0.0, 4_096.0).round() as usize,
            values.get_or("planets", 4.0).clamp(2.0, 64.0).round() as usize,
        ),
        None => (2, 4),
    };

    let mut suns = Vec::new();
    {
        let mut q = world.query::<Entity, With<Sun>>()?;
        q.for_each(|e| suns.push(e));
    }

    if suns.len() >= target {
        let excess = (suns.len() - target).min(64);
        for &sun in suns.iter().skip(target).take(excess) {
            world.despawn_with_descendants(sun)?;
        }
        perf::record("reconcile", start.elapsed().as_nanos() as u64);
        return Ok(());
    }

    let bounds = *world.get_resource::<SimBounds>().expect("SimBounds");
    let mut created = suns.len();
    with_world_rng(world, |world, rng| {
        while created < target && created - suns.len() < 64 {
            let angle = rng.f32() * std::f32::consts::TAU;
            let at = bounds.center() + Vec2::from_angle(angle) * bounds.width * 0.32;
            spawn_solar_system(world, rng, at, max_planets)?;
            created += 1;
        }
        Ok(())
    })?;
    perf::record("reconcile", start.elapsed().as_nanos() as u64);
    Ok(())
}

/// State for the hierarchy demo; `draw` reads the HUD counters back from here.
pub struct Orbits {
    suns: usize,
    despawn_events: u64,
}

impl Demo for Orbits {
    fn title(&self) -> &'static str {
        "millarecs - orbits - hierarchy - SPACE adds a system, D removes one"
    }

    fn usage() -> String {
        String::from(
            "orbits: Parent/Children hierarchy + despawn_with_descendants\n  --systems N  initial solar systems (default 2; adjust live in the panel, or with Space/D)\n  --planets N  max planets per system (default 4; applies to systems spawned afterwards)",
        )
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new("systems", "SYSTEMS", 2.0, 0.0, 4_096.0, 1.0).integer(),
            ParamSpec::new("planets", "PLANETS", 4.0, 2.0, 64.0, 1.0).integer(),
        ]
    }
    fn new(
        world: &mut World,
        schedule: &mut Schedule,
        _executor: &mut ScheduleExecutor,
        args: &Args,
    ) -> Self {
        world.register_table::<Pos>().expect("register Pos");
        world.register_table::<Offset>().expect("register Offset");
        world.register_table::<Orbit>().expect("register Orbit");
        world.register_table::<Body>().expect("register Body");
        world.register_table::<Sun>().expect("register Sun");
        world.init_hierarchy().expect("init hierarchy");
        world.observe_despawned();
        world.insert_resource(DespawnStats::default());
        world.insert_resource(Rng::new(99));
        world.insert_resource(PanelValues::default());

        schedule.add_system(advance).label::<Advance>();
        schedule.add_system(local_offsets).after::<Advance>();
        schedule
            .add_system(exclusive_system(propagate))
            .label::<Propagate>()
            .after::<Advance>();
        schedule.add_system(count_despawned).after::<Propagate>();
        schedule.add_system(exclusive_system(reconcile_systems));

        // Initial systems (the panel target takes over from the first tick).
        let systems_count = args.get_u32("systems", 2).min(4_096) as usize;
        let max_planets = args.get_u32("planets", 4).clamp(2, 16) as usize;
        let bounds = *world
            .get_resource::<SimBounds>()
            .expect("runner inserts SimBounds");
        let mut rng = Rng::default();
        for i in 0..systems_count {
            let angle = i as f32 * std::f32::consts::TAU / systems_count.max(1) as f32;
            let at = bounds.center() + Vec2::from_angle(angle) * bounds.width * 0.28;
            spawn_solar_system(world, &mut rng, at, max_planets).expect("spawn solar system");
        }

        Orbits {
            suns: systems_count,
            despawn_events: 0,
        }
    }

    fn frame(
        &mut self,
        _ctx: &FrameCtx,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
    ) {
        schedule.run(world, executor).expect("orbits schedule runs");
    }

    fn draw(&mut self, world: &World, _atlas: &FontAtlas, out: &mut DrawList) {
        self.despawn_events = world
            .get_resource::<DespawnStats>()
            .map(|s| s.total)
            .unwrap_or(0);
        let mut suns = 0usize;
        {
            let mut q = world.query::<Entity, With<Sun>>().expect("query suns");
            q.for_each(|_| suns += 1);
        }
        self.suns = suns;
        let mut q = world.query_all::<(&Pos, &Body)>().expect("query bodies");
        q.for_each(|(p, body)| {
            out.circle(p.0, body.radius, body.color);
        });
    }

    fn hud(&mut self) -> Vec<HudLine> {
        vec![
            HudLine::new(format!(
                "SISTEMAS {} - DESPAWN EVENTS {}",
                self.suns, self.despawn_events
            )),
            HudLine::colored(
                "ADD_CHILD / DESPAWN_WITH_DESCENDANTS - EXCLUSIVE PROPAGATE",
                color::rgb8(150, 150, 165),
            ),
        ]
    }

    fn on_key(&mut self, key: KeyCode, _world: &mut World, panel: &mut Panel) {
        match key {
            // The panel is the source of truth; the reconcile system follows
            // on the next tick.
            KeyCode::Space => panel.adjust("systems", 1.0),
            KeyCode::KeyD => panel.adjust("systems", -1.0),
            _ => {}
        }
    }
}

#[derive(Default)]
struct Advance;
#[derive(Default)]
struct Propagate;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{SimBounds, Time, FIXED_DT};
    use millarecs::system::schedule::ScheduleExecutor;

    fn test_session() -> (World, Schedule, ScheduleExecutor, Orbits, Panel) {
        test_session_with(&Args::default())
    }

    fn test_session_with(args: &Args) -> (World, Schedule, ScheduleExecutor, Orbits, Panel) {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        let mut executor = ScheduleExecutor::single_threaded();
        world.insert_resource(Time { dt: FIXED_DT });
        world.insert_resource(SimBounds {
            width: 960.0,
            height: 640.0,
        });
        let demo = Orbits::new(&mut world, &mut schedule, &mut executor, &Args::default());
        let panel = Panel::new(demo.params(), args, 640.0);
        world.insert_resource(PanelValues {
            values: panel.values().clone(),
        });
        (world, schedule, executor, demo, panel)
    }

    fn sun_count(world: &World) -> usize {
        let mut suns = 0;
        world
            .query::<Entity, With<Sun>>()
            .unwrap()
            .for_each(|_| suns += 1);
        suns
    }

    /// Applies a demo key like the runner does: on_key, then publish dirty
    /// panel values, then one tick.
    fn press_key(
        key: KeyCode,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
        demo: &mut Orbits,
        panel: &mut Panel,
        frame: u64,
    ) {
        demo.on_key(key, world, panel);
        if panel.take_dirty() {
            world.insert_resource(PanelValues {
                values: panel.values().clone(),
            });
        }
        demo.frame(&FrameCtx { frame }, world, schedule, executor);
    }

    #[test]
    fn hierarchy_propagates_and_despawns_descendants() {
        let (mut world, mut schedule, mut executor, mut demo, mut panel) = test_session();
        let bodies_at_start = world.len();
        assert_eq!(sun_count(&world), 2, "two systems were spawned");
        assert!(bodies_at_start > 4);

        // Planets start at Pos::ZERO; after one tick the exclusive propagate
        // system must have placed them around their sun.
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        let mut moved = false;
        world
            .query_all::<&Pos>()
            .unwrap()
            .for_each(|p| {
                if p.0 != Vec2::ZERO {
                    moved = true;
                }
            });
        assert!(moved, "propagation moved children away from the origin");

        // D steps the systems parameter down: the reconcile system despawns
        // one whole system and the Despawned counter grows.
        press_key(
            KeyCode::KeyD,
            &mut world,
            &mut schedule,
            &mut executor,
            &mut demo,
            &mut panel,
            1,
        );
        assert_eq!(sun_count(&world), 1);
        assert!(world.len() < bodies_at_start, "descendants were removed");

        demo.frame(&FrameCtx { frame: 2 }, &mut world, &mut schedule, &mut executor);
        let stats = world.get_resource::<DespawnStats>().unwrap();
        assert!(stats.total > 0, "Despawned events were published");
    }

    #[test]
    fn panel_target_controls_the_system_count() {
        let (mut world, mut schedule, mut executor, mut demo, mut _panel) = test_session();

        world.insert_resource(PanelValues {
            values: std::collections::HashMap::from([("systems", 5.0)]),
        });
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(sun_count(&world), 5, "reconcile spawned systems up to target");

        world.insert_resource(PanelValues {
            values: std::collections::HashMap::from([("systems", 1.0)]),
        });
        demo.frame(&FrameCtx { frame: 1 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(sun_count(&world), 1, "reconcile despawned down to target");
    }

    #[test]
    fn space_key_grows_through_the_panel() {
        let (mut world, mut schedule, mut executor, mut demo, mut panel) = test_session();
        press_key(
            KeyCode::Space,
            &mut world,
            &mut schedule,
            &mut executor,
            &mut demo,
            &mut panel,
            0,
        );
        press_key(
            KeyCode::Space,
            &mut world,
            &mut schedule,
            &mut executor,
            &mut demo,
            &mut panel,
            1,
        );
        assert_eq!(sun_count(&world), 4, "two Space presses add two systems");
    }
}
