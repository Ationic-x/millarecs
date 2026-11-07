//! Fireworks: mass spawn/despawn churn through `Commands`, entity recycling,
//! `Spawned`/`Despawned` event counting and per-system `Local<T>` timers.
//!
//! Bursts above the 32-command fused-spawn threshold stop publishing
//! per-entity `Spawned` events (scalar path only) - the HUD counters freeze
//! by design in that regime. Stress scaling goes through `--rate` (more
//! bursts per second) and `--burst` (bigger bursts, up to 262144).

use std::time::Instant;

use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::World;

use crate::prelude::*;

/// Particle position in pixels.
#[derive(Component)]
pub struct Pos(pub Vec2);

/// Particle velocity in pixels per second.
#[derive(Component)]
pub struct Vel(pub Vec2);

/// Remaining lifetime in seconds; the particle despawns at zero.
#[derive(Component)]
pub struct Life(pub f32);

/// Lifetime at spawn, used to normalize the fade.
#[derive(Component)]
pub struct MaxLife(pub f32);

/// Base particle color; alpha comes from remaining life.
#[derive(Component)]
pub struct Tint(pub Rgba);

/// Per-emitter `Local<T>` state.
#[derive(Default)]
struct EmitTimer {
    t: f32,
}

/// Cumulative event counters shown in the HUD.
#[derive(Default, Clone, Copy)]
struct Stats {
    spawned: u64,
    despawned: u64,
}

#[derive(Default)]
struct CountEvents;

#[derive(Default)]
struct Emit;

#[derive(Default)]
struct Physics;

fn count_events(
    mut spawned: EventReader<Spawned>,
    mut despawned: EventReader<Despawned>,
    mut stats: ResMut<Stats>,
) {
    let start = Instant::now();
    for _ in spawned.read() {
        stats.spawned += 1;
    }
    for _ in despawned.read() {
        stats.despawned += 1;
    }
    perf::record("events", start.elapsed().as_nanos() as u64);
}

fn emit(
    mut commands: Commands,
    mut timer: Local<EmitTimer>,
    mut rng: ResMut<Rng>,
    values: Res<PanelValues>,
    time: Res<Time>,
    bounds: Res<SimBounds>,
) {
    timer.t -= time.dt;
    if timer.t > 0.0 {
        return;
    }
    let start = Instant::now();
    // Knobs come from the panel resource (CLI flags seed it): rate in bursts
    // per second, burst up to 262144. Above 31 spawns per command run the
    // fused batch path skips the per-entity `Spawned` publication, so the
    // HUD event counters freeze - expected, not a bug.
    let rate = values.get_or("rate", 12.0).clamp(1.0, 1000.0);
    let burst = values
        .get_or("burst", 26.0)
        .clamp(1.0, 262_144.0)
        .round() as u32;
    //timer.t = (1.0 / rate) * rng.range(0.6, 1.4);
    while timer.t <= 0.0 {
        timer.t += (1.0 / rate) * rng.range(0.6, 1.4);

        let origin = Vec2::new(
            rng.range(60.0, bounds.width - 60.0),
            rng.range(bounds.height * 0.15, bounds.height * 0.55),
        );
        let hue = rng.f32();
        let tint = color::hsl(hue, 0.85, 0.62);
        for _ in 0..burst {
            let dir = Vec2::from_angle(rng.angle());
            let speed = rng.range(30.0, 340.0);
            let life = rng.range(0.8, 1.8);
            commands.spawn((
                Pos(origin),
                Vel(dir * speed),
                Life(life),
                MaxLife(life),
                Tint(tint),
            ));
        }
    }
    perf::record("emit", start.elapsed().as_nanos() as u64);
}

fn physics(mut q: Query<(&mut Pos, &mut Vel)>, time: Res<Time>) {
    let start = Instant::now();
    let dt = time.dt;
    q.for_each_mut(|(mut p, mut v)| {
        v.0 += Vec2::new(0.0, 150.0) * dt;
        v.0 *= 0.99;
        p.0 += v.0 * dt;
    });
    perf::record("physics", start.elapsed().as_nanos() as u64);
}

fn aging(mut commands: Commands, mut q: Query<(Entity, &mut Life)>, time: Res<Time>) {
    let start = Instant::now();
    let dt = time.dt;
    q.for_each_mut(|(e, mut life)| {
        life.0 -= dt;
        if life.0 <= 0.0 {
            commands.despawn(e);
        }
    });
    perf::record("aging", start.elapsed().as_nanos() as u64);
}

/// Fireworks state; the HUD reads its stats.
pub struct Particles {
    alive: usize,
    stats: Stats,
    /// (instant, spawned, despawned) of the previous `draw`, for the churn
    /// rate line.
    last: Option<(Instant, u64, u64)>,
    /// Smoothed spawn/despawn rates, in events per second.
    spawn_rate: f32,
    despawn_rate: f32,
}

/// Formats a count compactly (1234 -> "1.2K", 5400000 -> "5.4M").
fn fmt_k(value: f32) -> String {
    let v = value.abs();
    let sign = if value < 0.0 { "-" } else { "" };
    if v >= 100_000.0 {
        format!("{sign}{:.1}M", v / 100_000.0)
    } else if v >= 1_000.0 {
        format!("{sign}{:.1}K", v / 1_000.0)
    } else {
        format!("{sign}{v:.0}")
    }
}

impl Demo for Particles {
    fn title(&self) -> &'static str {
        "millarecs - fireworks - commands churn + lifecycle events"
    }

    fn usage() -> String {
        String::from(
            "fireworks: Commands churn + Spawned/Despawned events\n  --rate N   initial bursts per second (default 12; slider goes up to 1000)\n  --burst N  particles per burst (default 26; slider up to 262144 - above 31 the fused batch path stops publishing Spawned)",
        )
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![
            ParamSpec::new("rate", "RATE", 12.0, 1.0, 1000.0, 1.0).integer(),
            ParamSpec::new("burst", "BURST", 26.0, 1.0, 100000.0, 1.0).integer(),
        ]
    }

    fn new(
        world: &mut World,
        schedule: &mut Schedule,
        _executor: &mut ScheduleExecutor,
        _args: &Args,
    ) -> Self {
        world.register_table::<Pos>().expect("register Pos");
        world.register_table::<Vel>().expect("register Vel");
        world.register_table::<Life>().expect("register Life");
        world.register_table::<MaxLife>().expect("register MaxLife");
        world.register_table::<Tint>().expect("register Tint");
        world.observe_spawned();
        world.observe_despawned();
        world.insert_resource(Rng::default());
        world.insert_resource(Stats::default());
        world.insert_resource(PanelValues::default());

        schedule.add_system(count_events).label::<CountEvents>();
        schedule.add_system(emit).label::<Emit>().after::<CountEvents>();
        schedule.add_deferred_barrier();
        schedule.add_system(physics).label::<Physics>().after::<Emit>();
        schedule.add_system(aging).after::<Physics>();
        schedule.add_deferred_barrier();

        Particles {
            alive: 0,
            stats: Stats::default(),
            last: None,
            spawn_rate: 0.0,
            despawn_rate: 0.0,
        }
    }

    fn frame(
        &mut self,
        _ctx: &FrameCtx,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
    ) {
        schedule.run(world, executor).expect("particles schedule runs");
    }

    fn draw(&mut self, world: &World, _atlas: &FontAtlas, out: &mut DrawList) {
        self.alive = world.len();
        self.stats = world.get_resource::<Stats>().copied().unwrap_or_default();

        // Churn rate: events/second since the previous frame, EMA smoothed.
        let now = Instant::now();
        if let Some((then, spawned, despawned)) = self.last {
            let dt = now.duration_since(then).as_secs_f32();
            if dt > 1e-3 {
                let sr = (self.stats.spawned - spawned) as f32 / dt;
                let dr = (self.stats.despawned - despawned) as f32 / dt;
                self.spawn_rate += (sr - self.spawn_rate) * 0.25;
                self.despawn_rate += (dr - self.despawn_rate) * 0.25;
            }
        }
        self.last = Some((now, self.stats.spawned, self.stats.despawned));

        let mut q = world
            .query_all::<(&Pos, &Life, &MaxLife, &Tint)>()
            .expect("query particles");
        q.for_each(|(p, life, max, tint)| {
            let frac = (life.0 / max.0).clamp(0.0, 1.0);
            out.circle(p.0, 1.6 + 1.8 * frac, color::with_alpha(tint.0, 0.15 + 0.85 * frac));
        });
    }

    fn hud(&mut self) -> Vec<HudLine> {
        vec![
            HudLine::new(format!(
                "ALIVE {} - CHURN +{}/S -{}/S",
                fmt_k(self.alive as f32),
                fmt_k(self.spawn_rate),
                fmt_k(self.despawn_rate)
            )),
            HudLine::new(format!(
                "SPAWNED {} - DESPAWNED {} (TOTAL)",
                fmt_k(self.stats.spawned as f32),
                fmt_k(self.stats.despawned as f32)
            )),
            HudLine::colored(
                "COMMANDS SPAWN/DESPAWN + EVENTREADER + LOCAL<T>",
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

    fn test_session() -> (World, Schedule, ScheduleExecutor, Particles) {
        test_session_with(&Args::default())
    }

    fn test_session_with(args: &Args) -> (World, Schedule, ScheduleExecutor, Particles) {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        let mut executor = ScheduleExecutor::single_threaded();
        world.insert_resource(Time { dt: FIXED_DT });
        world.insert_resource(SimBounds {
            width: 800.0,
            height: 600.0,
        });
        let demo = Particles::new(&mut world, &mut schedule, &mut executor, &Args::default());
        // Seed the panel resource exactly like the runner would from `args`.
        world.insert_resource(PanelValues::init(&demo.params(), args));
        (world, schedule, executor, demo)
    }

    #[test]
    fn bursts_spawn_then_age_out() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();

        // The Local<EmitTimer> starts at zero, so the first tick emits one
        // 26-particle burst (applied at the deferred barrier).
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(world.len(), 26, "one scalar burst of 26 particles");

        demo.frame(&FrameCtx { frame: 1 }, &mut world, &mut schedule, &mut executor);
        let early = world.get_resource::<Stats>().unwrap();
        assert!(
            early.spawned >= 26,
            "Spawned events must be readable one cycle after the barrier (spawned={}, despawned={}, alive={})",
            early.spawned,
            early.despawned,
            world.len()
        );

        for frame in 2..300 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        let stats = world.get_resource::<Stats>().unwrap();
        assert!(stats.spawned >= 26);
        assert!(stats.despawned > 0, "particles age out and despawn");
        assert!(world.len() < 1500, "population stays bounded");

        let mut out = DrawList::new();
        let atlas = FontAtlas::with_font_chars();
        demo.draw(&world, &atlas, &mut out);
        assert_eq!(out.sprites.len(), world.len());
    }

    #[test]
    fn rate_flag_scales_the_steady_population() {
        // A higher rate must translate into more simultaneous particles
        // without touching the burst size (which is capped for event
        // publication).
        let args = Args::parse_iter(
            ["--rate", "60"].iter().map(|s| s.to_string()),
        );
        let (mut world, mut schedule, mut executor, mut demo) = test_session_with(&args);

        // Let the population reach steady state (lives last up to 1.8 s).
        for frame in 0..240 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        assert!(
            world.len() > 1_000,
            "rate 60 should hold a four-digit population, got {}",
            world.len()
        );
    }
}
