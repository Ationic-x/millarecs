//! Archetype lanes, the storage-routing demo. Entities drift while a
//! churn system inserts/removes two marker components (`TagA` registered as
//! a table column, `TagB` as a sparse set), forcing archetype migrations that
//! are visualized as horizontal lanes keyed by the tag combination.

use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::World;

use crate::prelude::*;

/// Horizontal drift position (the lane fixes the vertical placement).
#[derive(Component)]
pub struct Pos(pub Vec2);

/// Drift speed in pixels per second.
#[derive(Component)]
pub struct Vel(pub f32);

/// Table-storage tag; joining it moves the entity to a table archetype.
#[derive(Component)]
pub struct TagA;

/// Sparse-storage tag; joining it routes the entity through a sparse driver.
#[derive(Component)]
pub struct TagB;

#[derive(Default)]
struct ChurnTimer {
    t: f32,
}

fn drift(mut q: Query<(&mut Pos, &Vel)>, time: Res<Time>, bounds: Res<SimBounds>) {
    let start = std::time::Instant::now();
    let margin = 48.0;
    let hi = (bounds.width - margin).max(margin + 1.0);
    let dt = time.dt;
    q.for_each_mut(|(mut p, v)| {
        p.0.x += v.0 * dt;
        if p.0.x > hi {
            p.0.x = margin;
        } else if p.0.x < margin {
            p.0.x = hi;
        }
    });
    perf::record("drift", start.elapsed().as_nanos() as u64);
}

fn churn(
    mut commands: Commands,
    mut timer: Local<ChurnTimer>,
    mut rng: ResMut<Rng>,
    values: Res<PanelValues>,
    time: Res<Time>,
    mut q: Query<(Entity, &Pos)>,
) {
    timer.t -= time.dt;
    if timer.t > 0.0 {
        return;
    }
    timer.t = 0.3;
    let start = std::time::Instant::now();
    let per_tick = values.get_or("churn", 60.0).clamp(0.0, 100_000.0).round() as u32;

    let mut picks = Vec::new();
    q.for_each(|(e, _)| picks.push(e));
    for _ in 0..per_tick.min(picks.len() as u32) {
        let Some(e) = rng.choose(&picks).copied() else {
            break;
        };
        match rng.usize(4) {
            0 => commands.insert(e, (TagA,)),
            1 => commands.remove::<TagA>(e),
            2 => commands.insert(e, (TagB,)),
            _ => commands.remove::<TagB>(e),
        }
    }
    perf::record("churn", start.elapsed().as_nanos() as u64);
}

/// Known population for the lazy reconcile (see [`reconcile_population`]).
#[derive(Default)]
struct ReconcileState {
    known: usize,
}

/// Grows or shrinks the drifting population towards the `count` panel
/// target, in bounded chunks per tick; does nothing when the target is
/// already met.
fn reconcile_population(
    mut commands: Commands,
    values: Res<PanelValues>,
    mut rng: ResMut<Rng>,
    bounds: Res<SimBounds>,
    mut q: Query<Entity>,
    mut state: Local<ReconcileState>,
) {
    let target = values
        .get_or("count", 600.0)
        .clamp(100.0, 2_000_000.0)
        .round() as usize;
    if state.known == target {
        return;
    }

    let start = std::time::Instant::now();
    let mut count = 0usize;
    q.for_each(|_| count += 1);
    if count < target {
        let missing = (target - count).min(100_000);
        let bundles = (0..missing)
            .map(|_| {
                (
                    Pos(Vec2::new(
                        rng.range(bounds.width * 0.08, bounds.width * 0.92),
                        0.0,
                    )),
                    Vel(rng.range(30.0, 130.0) * if rng.chance(0.5) { 1.0 } else { -1.0 }),
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

/// Lane colors: none / A (table) / B (sparse) / A+B.
const LANE_COLORS: [Rgba; 4] = [
    [0.42, 0.44, 0.50, 1.0],
    [0.20, 0.75, 0.85, 1.0],
    [0.95, 0.60, 0.25, 1.0],
    [0.85, 0.35, 0.80, 1.0],
];

const LANE_LABELS: [&str; 4] = [
    "NO TAGS",
    "TAG A (TABLE)",
    "TAG B (SPARSE)",
    "TAG A + TAG B",
];

/// Demo state for archetype migration; lane counts are cached for the HUD.
pub struct Lanes {
    counts: [usize; 4],
}

impl Lanes {
    fn band_center(&self, lane: usize, bounds: SimBounds) -> f32 {
        let top = bounds.height * 0.14;
        let usable = bounds.height * 0.78;
        top + (lane as f32 + 0.5) * (usable / 4.0)
    }
}

impl Demo for Lanes {
    fn title(&self) -> &'static str {
        "millarecs - lanes - table vs sparse archetype migration"
    }

    fn usage() -> String {
        String::from(
            "lanes: table vs sparse registration + archetype migration\n  --count N  initial entities across the four lanes (default 600; adjust live in the panel)\n  --churn N  tag insert/remove commands per churn tick (default 60)",
        )
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new("count", "ENTITIES", 600.0, 100.0, 2_000_000.0, 1.0).integer(),
            ParamSpec::new("churn", "CHURN", 60.0, 0.0, 100_000.0, 1.0).integer(),
        ]
    }

    fn new(
        world: &mut World,
        schedule: &mut Schedule,
        _executor: &mut ScheduleExecutor,
        args: &Args,
    ) -> Self {
        world.register_table::<Pos>().expect("register Pos");
        world.register_table::<Vel>().expect("register Vel");
        world.register_table::<TagA>().expect("register TagA (table)");
        world.register_sparse::<TagB>().expect("register TagB (sparse)");
        world.insert_resource(Rng::default());
        world.insert_resource(PanelValues::default());

        schedule.add_system(drift).label::<Drift>();
        schedule.add_system(churn).after::<Drift>();
        schedule.add_system(reconcile_population);
        schedule.add_deferred_barrier();

        let bounds = *world
            .get_resource::<SimBounds>()
            .expect("runner inserts SimBounds");
        let count = args.get_u32("count", 600).clamp(1, 100_000) as usize;
        let mut rng = Rng::default();
        let entities = (0..count)            .map(|_| {
                (
                    Pos(Vec2::new(
                        rng.range(bounds.width * 0.08, bounds.width * 0.92),
                        0.0,
                    )),
                    Vel(rng.range(30.0, 130.0) * if rng.chance(0.5) { 1.0 } else { -1.0 }),
                )
            })
            .collect::<Vec<_>>();
        world.spawn_batch(entities).expect("spawn lanes entities");

        Lanes { counts: [0; 4] }
    }

    fn frame(
        &mut self,
        _ctx: &FrameCtx,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
    ) {
        schedule.run(world, executor).expect("lanes schedule runs");
    }

    fn draw(&mut self, world: &World, atlas: &FontAtlas, out: &mut DrawList) {
        let bounds = *world.get_resource::<SimBounds>().expect("SimBounds");
        let margin = 48.0;

        // Band backgrounds + labels.
        for lane in 0..4 {
            let y = self.band_center(lane, bounds);
            let half_h = bounds.height * 0.78 / 4.0 * 0.5 - 6.0;
            out.rect(
                Vec2::new(bounds.width * 0.5, y),
                Vec2::new(bounds.width * 0.5 - margin, half_h),
                [LANE_COLORS[lane][0] * 0.16, LANE_COLORS[lane][1] * 0.16, LANE_COLORS[lane][2] * 0.16, 1.0],
            );
            out.text(
                atlas,
                Vec2::new(margin, y - half_h + 6.0),
                LANE_LABELS[lane],
                2.0,
                color::scale_rgb(LANE_COLORS[lane], 0.9),
            );
        }

        // Count and draw each tag combination as its own lane. Each query is
        // a different archetype route - exactly what the bands visualize.
        let mut counts = [0usize; 4];
        {
            let mut q = world
                .query::<(&Pos,), (Not<With<TagA>>, Not<With<TagB>>)>()
                .expect("query lane 0");
            q.for_each(|(p,)| {
                counts[0] += 1;
                out.rect(Vec2::new(p.0.x, self.band_center(0, bounds)), Vec2::splat(3.0), LANE_COLORS[0]);
            });
        }
        {
            let mut q = world
                .query::<(&Pos,), (With<TagA>, Not<With<TagB>>)>()
                .expect("query lane 1");
            q.for_each(|(p,)| {
                counts[1] += 1;
                out.rect(Vec2::new(p.0.x, self.band_center(1, bounds)), Vec2::splat(3.0), LANE_COLORS[1]);
            });
        }
        {
            let mut q = world
                .query::<(&Pos,), (Not<With<TagA>>, With<TagB>)>()
                .expect("query lane 2");
            q.for_each(|(p,)| {
                counts[2] += 1;
                out.rect(Vec2::new(p.0.x, self.band_center(2, bounds)), Vec2::splat(3.0), LANE_COLORS[2]);
            });
        }
        {
            let mut q = world
                .query::<(&Pos,), (With<TagA>, With<TagB>)>()
                .expect("query lane 3");
            q.for_each(|(p,)| {
                counts[3] += 1;
                out.rect(Vec2::new(p.0.x, self.band_center(3, bounds)), Vec2::splat(3.0), LANE_COLORS[3]);
            });
        }
        self.counts = counts;
    }

    fn hud(&mut self) -> Vec<HudLine> {
        let c = self.counts;
        vec![
            HudLine::colored(format!("L0 NO TAGS  {}", c[0]), LANE_COLORS[0]),
            HudLine::colored(format!("L1 TAG A    {}  (TABLE)", c[1]), LANE_COLORS[1]),
            HudLine::colored(format!("L2 TAG B    {}  (SPARSE)", c[2]), LANE_COLORS[2]),
            HudLine::colored(format!("L3 A + B    {}", c[3]), LANE_COLORS[3]),
        ]
    }
}

#[derive(Default)]
struct Drift;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{SimBounds, Time, FIXED_DT};
    use millarecs::system::schedule::ScheduleExecutor;

    fn test_session() -> (World, Schedule, ScheduleExecutor, Lanes) {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        let mut executor = ScheduleExecutor::single_threaded();
        world.insert_resource(Time { dt: FIXED_DT });
        world.insert_resource(SimBounds {
            width: 800.0,
            height: 600.0,
        });
        let demo = Lanes::new(&mut world, &mut schedule, &mut executor, &Args::default());
        (world, schedule, executor, demo)
    }

    #[test]
    fn churn_migrates_entities_between_lanes() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();

        // First tick: churn timer starts at zero, so migrations happen and
        // are applied at the deferred barrier.
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);

        let mut out = DrawList::new();
        let atlas = FontAtlas::with_font_chars();
        demo.draw(&world, &atlas, &mut out);

        let total: usize = demo.counts.iter().sum();
        assert_eq!(total, 600, "every entity lands in exactly one lane");
        assert!(demo.counts[1] + demo.counts[2] + demo.counts[3] > 0,
            "churn moved entities into tagged lanes: {:?}", demo.counts);
        assert_eq!(out.sprites.len(), 600 + 4, "600 dots + 4 band backgrounds");
    }

    #[test]
    fn panel_target_resizes_the_population() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();

        world.insert_resource(PanelValues {
            values: std::collections::HashMap::from([("count", 150.0)]),
        });
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(world.len(), 150);

        world.insert_resource(PanelValues {
            values: std::collections::HashMap::from([("count", 1200.0)]),
        });
        demo.frame(&FrameCtx { frame: 1 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(world.len(), 1200);
    }
}
