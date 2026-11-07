//! Flocking (boids), the classic parallel-queries workload: ~1400 boids with
//! `Position`/`Velocity` components steer using a uniform grid resource and
//! integrate through `Query::par_for_each_mut` on a multi-thread executor.

use std::f32::consts::TAU;

use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::World;

use crate::prelude::*;

/// Boid position in pixels.
#[derive(Component)]
pub struct Pos(pub Vec2);

/// Boid velocity in pixels per second.
#[derive(Component)]
pub struct Vel(pub Vec2);

/// One neighbor snapshot captured by the grid rebuild pass.
#[derive(Clone, Copy)]
struct Neighbor {
    pos: Vec2,
    vel: Vec2,
}

/// Uniform spatial grid over [`SimBounds`], rebuilt every tick as a resource.
/// Cell size equals the view radius so neighbors live in the 3x3 buckets
/// around a boid.
struct Grid {
    cell: f32,
    cols: u32,
    rows: u32,
    buckets: Vec<Vec<Neighbor>>,
}

impl Grid {
    fn new(cell: f32, bounds: SimBounds) -> Self {
        let cols = ((bounds.width / cell).ceil() as u32).max(1);
        let rows = ((bounds.height / cell).ceil() as u32).max(1);
        Self {
            cell,
            cols,
            rows,
            buckets: vec![Vec::new(); (cols * rows) as usize],
        }
    }

    fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
    }

    fn insert(&mut self, pos: Vec2, vel: Vec2) {
        let (col, row) = self.cell_of(pos);
        self.buckets[(row * self.cols + col) as usize].push(Neighbor { pos, vel });
    }

    fn cell_of(&self, pos: Vec2) -> (u32, u32) {
        let col = ((pos.x / self.cell).floor().max(0.0) as u32).min(self.cols - 1);
        let row = ((pos.y / self.cell).floor().max(0.0) as u32).min(self.rows - 1);
        (col, row)
    }

    /// Visits every recorded neighbor in the 3x3 buckets around `pos`,
    /// skipping the boid itself (exact position match).
    fn for_each_neighbor<F: FnMut(&Neighbor)>(&self, pos: Vec2, mut visit: F) {
        let (col, row) = self.cell_of(pos);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                let c = col as i32 + dx;
                let r = row as i32 + dy;
                if c < 0 || r < 0 || c >= self.cols as i32 || r >= self.rows as i32 {
                    continue;
                }
                for neighbor in &self.buckets[(r as u32 * self.cols + c as u32) as usize] {
                    if neighbor.pos != pos {
                        visit(neighbor);
                    }
                }
            }
        }
    }
}

/// Tuning constants of the flocking behavior.
struct Params {
    view: f32,
    separation: f32,
    alignment: f32,
    cohesion: f32,
    max_speed: f32,
}

#[derive(Default)]
struct RebuildGrid;

fn rebuild_grid(mut grid: ResMut<Grid>, mut q: Query<(&Pos, &Vel)>) {
    let start = std::time::Instant::now();
    grid.clear();
    q.for_each(|(p, v)| grid.insert(p.0, v.0));
    perf::record("grid", start.elapsed().as_nanos() as u64);
}

fn flock(
    mut q: Query<(&mut Pos, &mut Vel)>,
    grid: Res<Grid>,
    params: Res<Params>,
    time: Res<Time>,
    bounds: Res<SimBounds>,
) {
    let start = std::time::Instant::now();
    let dt = time.dt;
    q.par_for_each_mut(|(mut pos, mut vel)| {
        let here = pos.0;
        let mut ali = Vec2::ZERO;
        let mut coh = Vec2::ZERO;
        let mut sep = Vec2::ZERO;
        let mut count = 0u32;
        grid.for_each_neighbor(here, |nb| {
            let d = here.distance(nb.pos);
            if d < params.view {
                count += 1;
                ali += nb.vel;
                coh += nb.pos;
                if d > 1.0 {
                    sep += (here - nb.pos) * (1.0 / (d * d));
                }
            }
        });
        if count > 0 {
            let inv = 1.0 / count as f32;
            ali = ali * inv - vel.0;
            coh = coh * inv - here;
        }
        let steer = (ali * params.alignment + coh * params.cohesion + sep * params.separation)
            .clamp_length(600.0);
        vel.0 = (vel.0 + steer * dt).clamp_length(params.max_speed);
        pos.0 = bounds.wrap(pos.0 + vel.0 * dt);
    });
    perf::record("flock", start.elapsed().as_nanos() as u64);
}

/// Known population for the lazy reconcile (see [`reconcile_population`]).
#[derive(Default)]
struct ReconcileState {
    known: usize,
}

/// Grows or shrinks the flock towards the `count` panel target in bounded
/// chunks per tick; does nothing when the target is already met.
fn reconcile_population(
    mut commands: Commands,
    values: Res<PanelValues>,
    mut rng: ResMut<Rng>,
    bounds: Res<SimBounds>,
    mut q: Query<Entity>,
    mut state: Local<ReconcileState>,
) {
    let target = values
        .get_or("count", 1400.0)
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
                    Pos(Vec2::new(rng.range(0.0, bounds.width), rng.range(0.0, bounds.height))),
                    Vel(Vec2::from_angle(rng.angle()) * rng.range(40.0, 120.0)),
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

/// Boids state, with the live count cached for the HUD.
pub struct Boids {
    alive: usize,
}

impl Demo for Boids {
    fn title(&self) -> &'static str {
        "millarecs - boids - parallel queries"
    }

    fn usage() -> String {
        String::from(
            "boids: flocking through Query::par_for_each_mut\n  --count N  initial boid population (default 1400; adjust live in the panel)",
        )
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::new("count", "BOIDS", 1400.0, 100.0, 2_000_000.0, 1.0).integer()]
    }

    fn new(
        world: &mut World,
        schedule: &mut Schedule,
        _executor: &mut ScheduleExecutor,
        args: &Args,
    ) -> Self {
        world.register_table::<Pos>().expect("register Pos");
        world.register_table::<Vel>().expect("register Vel");
        let bounds = *world
            .get_resource::<SimBounds>()
            .expect("runner inserts SimBounds");
        world.insert_resource(Grid::new(36.0, bounds));
        world.insert_resource(Params {
            view: 36.0,
            separation: 95.0,
            alignment: 1.7,
            cohesion: 0.9,
            max_speed: 135.0,
        });
        world.insert_resource(Rng::default());

        let count = args.get_u32("count", 1400).clamp(1, 200_000) as usize;
        let mut rng = Rng::default();
        let boids = (0..count)
            .map(|_| {
                (
                    Pos(Vec2::new(rng.range(0.0, bounds.width), rng.range(0.0, bounds.height))),
                    Vel(Vec2::from_angle(rng.angle()) * rng.range(40.0, 120.0)),
                )
            })
            .collect::<Vec<_>>();
        world.spawn_batch(boids).expect("spawn boids");

        schedule.add_system(rebuild_grid).label::<RebuildGrid>();
        schedule.add_system(flock).after::<RebuildGrid>();
        schedule.add_system(reconcile_population);
        schedule.add_deferred_barrier();
        Boids { alive: 0 }
    }

    fn frame(
        &mut self,
        _ctx: &FrameCtx,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
    ) {
        schedule.run(world, executor).expect("boids schedule runs");
    }

    fn draw(&mut self, world: &World, _atlas: &FontAtlas, out: &mut DrawList) {
        self.alive = world.len();
        let mut q = world.query_all::<(&Pos, &Vel)>().expect("query boids");
        q.for_each(|(p, v)| {
            let heading = v.0.angle();
            let hue = heading.rem_euclid(TAU) / TAU;
            out.triangle(p.0, Vec2::new(7.0, 4.5), heading, color::hsl(hue, 0.65, 0.62));
        });
    }

    fn hud(&mut self) -> Vec<HudLine> {
        vec![
            HudLine::new(format!("BOIDS {}", self.alive)),
            HudLine::colored(
                "QUERY::PAR_FOR_EACH_MUT - MULTI-THREADED EXECUTOR",
                color::rgb8(150, 150, 165),
            ),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::{SimBounds, Time, FIXED_DT};
    use millarecs::system::schedule::ScheduleExecutor;
    use std::collections::HashMap;

    fn test_session() -> (World, Schedule, ScheduleExecutor, Boids) {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        let mut executor = ScheduleExecutor::single_threaded();
        world.insert_resource(Time { dt: FIXED_DT });
        world.insert_resource(SimBounds {
            width: 640.0,
            height: 480.0,
        });
        world.insert_resource(PanelValues::default());
        let demo = Boids::new(&mut world, &mut schedule, &mut executor, &Args::default());
        (world, schedule, executor, demo)
    }

    #[test]
    fn boids_move_and_render_one_triangle_each() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();

        let mut before = Vec::new();
        world
            .query_all::<&Pos>()
            .unwrap()
            .for_each(|p| before.push(p.0));

        for frame in 0..5 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }

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
        assert!(moved > 1200, "expected most boids to move, got {moved}/1400");

        let mut out = DrawList::new();
        let atlas = FontAtlas::with_font_chars();
        demo.draw(&world, &atlas, &mut out);
        assert_eq!(out.sprites.len(), 1400);
    }

    #[test]
    fn grid_excludes_self_and_stays_in_bounds() {
        let bounds = SimBounds {
            width: 200.0,
            height: 200.0,
        };
        let mut grid = Grid::new(36.0, bounds);
        grid.insert(Vec2::new(10.0, 10.0), Vec2::ONE);
        grid.insert(Vec2::new(20.0, 12.0), Vec2::ONE * -1.0);
        grid.insert(Vec2::new(195.0, 195.0), Vec2::ONE);
        grid.insert(Vec2::new(188.0, 190.0), Vec2::ONE * -1.0);
        let mut seen = 0;
        grid.for_each_neighbor(Vec2::new(10.0, 10.0), |_| seen += 1);
        assert_eq!(seen, 1, "self is excluded, far buckets are not visited");
        grid.for_each_neighbor(Vec2::new(195.0, 195.0), |_| seen += 1);
        assert_eq!(seen, 2);
    }

    #[test]
    fn panel_target_resizes_the_flock() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();

        world.insert_resource(PanelValues {
            values: HashMap::from([("count", 300.0)]),
        });
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(world.len(), 300, "reconcile shrinks to the panel target");

        world.insert_resource(PanelValues {
            values: HashMap::from([("count", 2000.0)]),
        });
        demo.frame(&FrameCtx { frame: 1 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(world.len(), 2000, "reconcile grows to the panel target");
    }
}
