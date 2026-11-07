//! Field: the canonical "move N entities" benchmark that every ECS project
//! publishes (EnTT, flecs and bevy_ecs all ship this exact workload). One
//! query iterates N entities and integrates `Pos += Vel * dt` per tick -
//! nothing else. It isolates raw query throughput from any neighbor search,
//! hierarchy or churn, which makes it the cleanest cross-ECS comparison
//! point.
//!
//! The HUD's `SYS MOVE` line, divided by the entity count, gives the
//! ns-per-entity figure to quote against other engines.

use std::time::Instant;

use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::World;

use crate::prelude::*;

/// Entity position in pixels.
#[derive(Component, Clone, Copy)]
pub struct Pos(pub Vec2);

/// Constant velocity in pixels per second.
#[derive(Component, Clone, Copy)]
pub struct Vel(pub Vec2);

/// The canonical workload: one mutable query, one linear integration.
fn move_sys(
    mut q: Query<(&mut Pos, &mut Vel)>,
    time: Res<Time>,
    bounds: Res<SimBounds>,
) {
    let start = Instant::now();
    let dt = time.dt;
    q.par_for_each_mut(|(mut p, v)| {
        p.0 = bounds.wrap(p.0 + v.0 * dt);
    });
    perf::record("move", start.elapsed().as_nanos() as u64);
}

/// Known population for the lazy reconcile: it only scans the world when the
/// panel target differs from the last known count.
#[derive(Default)]
struct ReconcileState {
    known: usize,
}

/// Grows or shrinks the population towards the `n` panel target in bounded
/// chunks per tick. When the target is already met it does nothing - no
/// scan, no allocation - so `SYS RECONCILE` only costs when it actually
/// resizes.
fn reconcile_population(
    mut commands: Commands,
    values: Res<PanelValues>,
    mut rng: ResMut<Rng>,
    bounds: Res<SimBounds>,
    mut q: Query<Entity>,
    mut state: Local<ReconcileState>,
) {
    let target = values
        .get_or("n", 1_000_000.0)
        .clamp(10_000.0, 5_000_000.0)
        .round() as usize;
    if state.known == target {
        return;
    }

    let start = Instant::now();
    let mut count = 0usize;
    q.for_each(|_| count += 1);
    if count < target {
        let missing = (target - count).min(200_000);
        let bundles = (0..missing)
            .map(|_| {
                (
                    Pos(Vec2::new(rng.range(0.0, bounds.width), rng.range(0.0, bounds.height))),
                    Vel(Vec2::from_angle(rng.angle()) * rng.range(20.0, 160.0)),
                )
            })
            .collect::<Vec<_>>();
        commands.spawn_batch(bundles);
        state.known = count + missing;
    } else if count > target {
        let mut ids = Vec::with_capacity((count - target).min(200_000));
        q.for_each(|entity| ids.push(entity));
        let excess = (count - target).min(200_000);
        commands.despawn_batch(ids.into_iter().take(excess));
        state.known = count - excess;
    } else {
        state.known = count;
    }
    perf::record("reconcile", start.elapsed().as_nanos() as u64);
}

/// Field demo state. Caches the population for the HUD and the
/// precomputed direction palette (no per-sprite trigonometry in the
/// extract).
pub struct Field {
    alive: usize,
    drawn: usize,
    palette: Vec<Rgba>,
}

/// Render cap: above this, dots are sampled so the frame measures simulation
/// throughput instead of instance upload bandwidth.
const MAX_DRAWN: usize = 100_000;

impl Demo for Field {
    fn title(&self) -> &'static str {
        "millarecs - field - move N entities (canonical query benchmark)"
    }

    fn usage() -> String {
        String::from(
            "field: the canonical move-N-entities benchmark\n  --n N  entity count (default 1000000; slider 10000..5000000)",
        )
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::new("n", "N", 1_000_000.0, 10_000.0, 5_000_000.0, 1.0).integer()]
    }

    fn new(
        world: &mut World,
        schedule: &mut Schedule,
        _executor: &mut ScheduleExecutor,
        args: &Args,
    ) -> Self {
        world.register_table::<Pos>().expect("register Pos");
        world.register_table::<Vel>().expect("register Vel");
        world.insert_resource(Rng::default());
        world.insert_resource(PanelValues::default());

        schedule.add_system(move_sys).label::<MoveSys>();
        schedule.add_system(reconcile_population);
        schedule.add_deferred_barrier();

        let bounds = *world
            .get_resource::<SimBounds>()
            .expect("runner inserts SimBounds");
        let n = args.get_u32("n", 1_000_000).clamp(10_000, 5_000_000) as usize;
        let mut rng = Rng::default();
        let bundles = (0..n)
            .map(|_| {
                (
                    Pos(Vec2::new(rng.range(0.0, bounds.width), rng.range(0.0, bounds.height))),
                    Vel(Vec2::from_angle(rng.angle()) * rng.range(20.0, 160.0)),
                )
            })
            .collect::<Vec<_>>();
        world.spawn_batch(bundles).expect("spawn field");

        Field {
            alive: 0,
            drawn: 0,
            palette: (0..8)
                .map(|i| color::hsl(i as f32 / 8.0, 0.6, 0.65))
                .collect(),
        }
    }

    fn frame(
        &mut self,
        _ctx: &FrameCtx,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
    ) {
        schedule.run(world, executor).expect("field schedule runs");
    }

    fn draw(&mut self, world: &World, _atlas: &FontAtlas, out: &mut DrawList) {
        self.alive = world.len();
        // Extract cost stays O(MAX_DRAWN): iteration order is by storage
        // rows, so a `take` draws the first slice of the population (spawned
        // with random positions, so it is spread across the screen). Color
        // comes from an 8-direction octant lookup - no per-sprite trig.
        let palette = &self.palette;
        let mut q = world.query_all::<(&Pos, &Vel)>().expect("query field");
        q.iter().take(MAX_DRAWN).for_each(|(p, v)| {
            let v = v.0;
            let octant = if v.y >= 0.0 {
                if v.x >= 0.0 {
                    if v.x >= v.y { 0 } else { 1 }
                } else if -v.x <= v.y {
                    2
                } else {
                    3
                }
            } else if v.x <= v.y {
                4
            } else if v.x >= -v.y {
                7
            } else {
                6
            };
            out.rect(p.0, Vec2::splat(1.5), palette[octant]);
        });
        self.drawn = out.sprites.len();
    }

    fn hud(&mut self) -> Vec<HudLine> {
        vec![
            HudLine::new(format!(
                "N {} - DIBUJADAS {} (PRIMERAS)",
                fmt_count(self.alive),
                fmt_count(self.drawn)
            )),
            HudLine::colored(
                "MOVE NS/ENT = SYS MOVE(MS) * 1E6 / N - cifra para comparar con otros ECS",
                color::rgb8(150, 150, 165),
            ),
        ]
    }
}

/// Compact count formatting (1500000 -> "1.5M").
fn fmt_count(value: usize) -> String {
    if value >= 1_000_000 {
        format!("{:.2}M", value as f32 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f32 / 1_000.0)
    } else {
        format!("{value}")
    }
}

#[derive(Default)]
struct MoveSys;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{SimBounds, Time, FIXED_DT};
    use millarecs::system::schedule::ScheduleExecutor;

    fn test_session() -> (World, Schedule, ScheduleExecutor, Field) {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        let mut executor = ScheduleExecutor::single_threaded();
        world.insert_resource(Time { dt: FIXED_DT });
        world.insert_resource(SimBounds {
            width: 1280.0,
            height: 720.0,
        });
        world.insert_resource(PanelValues::default());
        let demo = Field::new(&mut world, &mut schedule, &mut executor, &Args::default());
        (world, schedule, executor, demo)
    }

    #[test]
    fn spawns_one_million_and_moves_them() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        assert_eq!(world.len(), 1_000_000, "initial population matches the panel");

        let mut before = Vec::new();
        world
            .query_all::<&Pos>()
            .unwrap()
            .for_each(|p| before.push(p.0));

        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);

        let mut moved = 0usize;
        let mut index = 0usize;
        world
            .query_all::<&Pos>()
            .unwrap()
            .for_each(|p| {
                if before[index] != p.0 {
                    moved += 1;
                }
                index += 1;
            });
        assert!(moved > 990_000, "nearly every entity moved, got {moved}");

        // Sampling keeps the draw list at or under the render cap.
        let atlas = FontAtlas::with_font_chars();
        let mut out = DrawList::new();
        demo.draw(&world, &atlas, &mut out);
        assert!(out.sprites.len() <= MAX_DRAWN);
        assert_eq!(demo.drawn, out.sprites.len());
    }

    #[test]
    fn panel_target_resizes_the_population() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();

        // Shrinking 1M -> 50K takes several 200K-per-tick chunks.
        world.insert_resource(PanelValues {
            values: std::collections::HashMap::from([("n", 50_000.0)]),
        });
        for frame in 0..12 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        assert_eq!(world.len(), 50_000, "reconcile converged down");

        world.insert_resource(PanelValues {
            values: std::collections::HashMap::from([("n", 400_000.0)]),
        });
        for frame in 12..24 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        assert_eq!(world.len(), 400_000, "reconcile converged up");
    }
}
