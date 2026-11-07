<p align="center">
  <img src="docs/img/logo_t3.svg" alt="millarecs" width="700">
</p>

<h1 align="center">Millarecs</h1>

## What is Millarecs?

Millarecs is a personal research project focused on learning and experimenting with the development of a game engine built around the **Entity Component System (ECS)** paradigm.

## Goals

Experimentation has always been, and will remain, the primary goal of this project.

This means Millarecs is not necessarily intended to compete with other ECS implementations or to reach 1.0 release. While I draw inspiration and knowledge from projects such as [Bevy](https://bevy.org/), [Flecs](https://www.flecs.dev/), [EnTT](https://github.com/skypjack/entt)... Whose communities I greatly appreciate, Millarecs is ultimately driven by researching and experimenting.

This approach may occasionally lead to setbacks, changes in direction, or improvements.

## The Name

The name **Millarecs** comes from combining *Millares* and *ECS*.

On one hand, it reflects the large number of entities and components an ECS can handle. On the other, it is a personal tribute to my homeland, **Almería**.

## Roadmap

Although there are clear plans to improve the ECS and introduce additional features in the future, my main priority is to create comfortable, approachable, and user-friendly documentation for anyone interested in using the project.

Many of the features originally considered for the road to **0.11** will be put on hold for an extended period. These include hierarchical systems, serialization, command merging, further development of fragmented archetypes, and several other improvements.

I have been actively working on this project for an entire year, and I feel it is time to take a break. Millarecs will remain a personal project, and I hope to return to its development in the future, when I feel ready to pick it up again.

## Development Environment

| Crate                  | Role                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `millarecs`            | Core ECS implementation: world management, storage, queries, scheduling, and a `std`-only foundation. |
| `millarecs_macros`     | Component and bundle implementations.                                                                 |
| `millarecs_playground` | `wgpu` demos and visual benchmarks.                                                                   |


## First steps

Add both crates to `Cargo.toml`:

```toml
[dependencies]
millarecs = "0.1"          # path or git dependency until published
millarecs_macros = "0.1"
```

Components are plain types with a derive. Resources need no derive at all, any
`Send + Sync` type qualifies:

```rust
use millarecs::system::query::Query;
use millarecs::system::resource::Res;
use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::World;
use millarecs_macros::Component;

#[derive(Component, Clone, Copy)]
struct Pos(f32, f32);

#[derive(Component, Clone, Copy)]
struct Vel(f32, f32);

struct Dt(f32);

fn integrate(mut q: Query<(&mut Pos, &Vel)>, dt: Res<Dt>) {
    q.for_each_mut(|(mut pos, vel)| {
        pos.0 += vel.0 * dt.0;
        pos.1 += vel.1 * dt.0;
    });
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut world = World::new();

    world.register_table::<Pos>()?;
    world.register_table::<Vel>()?;
    world.insert_resource(Dt(1.0 / 60.0));

    for row in 0..100 {
        world.spawn((Pos(0.0, row as f32), Vel(1.0, 0.0)))?;
    }

    let mut schedule = Schedule::new();
    schedule.add_system(integrate);

    let mut executor = ScheduleExecutor::default(); // one worker per core

    for _ in 0..60 {
        schedule.run(&mut world, &mut executor)?;
    }

    Ok(())
}
```

`register_table` places a component in archetype columns; `register_sparse`
keeps it in a sparse set instead, which suits big or rarely iterated components.
The planner picks the execution lane, so both registrations serve the same
queries.

Reading from outside any system works without a schedule:

```rust
let mut q = world.query::<&Pos, ()>()?;
q.for_each(|pos| println!("row {}", pos.1));
```

Inside a system, spawn through deferred commands so iteration stays valid. The
queue applies after the parallel phase ends:

```rust
use millarecs::system::commands::Commands;

fn spawn_wave(mut commands: Commands) {
    for row in 0..50 {
        commands.spawn((Pos(0.0, row as f32), Vel(1.0, 0.0)));
    }
}
```

When a run is wide enough to pay for the dispatch, `par_for_each_mut` on a
`Query` spreads its rows over the executor workers.

## Demos

```
cargo run --release -p millarecs_playground --bin boids        # parallel queries, ~1400 agents
cargo run --release -p millarecs_playground --bin particles    # command churn, lifecycle events
cargo run --release -p millarecs_playground --bin orbits       # parent/child hierarchy
cargo run --release -p millarecs_playground --bin lanes        # table vs sparse migration
cargo run --release -p millarecs_playground --bin field        # move N entities, up to 5M
cargo run --release -p millarecs_playground --bin bullet_hell  # scripted shooter: waves, spells
```

All of them open a window; append `-- --frames 240` for a headless smoke run.

## Benchmarks

```
cargo run --release --example ecs_bench_suite -- --rows 100000
```

The suite covers structural operations, point gets, query families, route
selection, deferred commands and scaling by archetype width. Compare runs only
against runs taken on the same machine at rest. The module header of
`examples/ecs_bench_suite.rs` doubles as a cookbook of the common invocations.

## License

MIT, see [LICENSE](LICENSE).
