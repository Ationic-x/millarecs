//! Visual examples for the [millarecs](https://crates.io/crates/millarecs) ECS.
//!
//! The crate hosts a tiny wgpu renderer plus six self-contained demos that
//! exercise different millarecs subsystems from a 60 Hz windowed loop (or an
//! offscreen `--frames N` smoke run):
//!
//! | Binary       | Focus                                                          |
//! |--------------|----------------------------------------------------------------|
//! | `boids`      | parallel `Query::par_for_each_mut` over a multi-thread executor |
//! | `particles`  | `Commands` churn, `Spawned`/`Despawned` events, `Local<T>`      |
//! | `orbits`     | `Parent`/`Children` hierarchy, exclusive systems, despawn       |
//! | `lanes`      | table vs sparse registration and archetype migration            |
//! | `field`      | the classic "move N entities" throughput benchmark             |
//! | `bullet_hell`| scripted waves and spell cards over churn and collisions        |
//!
//! The demos never render through the ECS: each frame runs the simulation
//! schedule, then reads the world from the outside to build one sprite list
//! for the GPU. That keeps millarecs free of any rendering concern.
//!
//! # Running
//!
//! ```text
//! cargo run -p millarecs_playground --bin boids            # windowed
//! cargo run -p millarecs_playground --bin boids -- --frames 240   # offscreen smoke run
//! ```

#![warn(missing_docs)]

pub mod app;
pub mod atlas;
pub mod color;
pub mod demos;
pub mod draw;
pub mod font;
pub mod gfx;
pub mod hud;
pub mod math;
pub mod panel;
pub mod perf;
pub mod rng;
pub mod time;

/// Entry point shared by every demo binary: [`crate::app::run`].
pub use app::run;

/// Common imports shared by every demo binary.
pub mod prelude {
    pub use crate::app::{run, Args, Demo, FrameCtx, InputState};
    pub use crate::atlas::FontAtlas;
    pub use crate::color::{self, Rgba};
    pub use crate::draw::{DrawList, Shape};
    pub use crate::hud::HudLine;
    pub use crate::math::Vec2;
    pub use crate::panel::{Panel, PanelValues, ParamSpec};
    pub use crate::perf;
    pub use crate::rng::Rng;
    pub use crate::time::{SimBounds, Time, FIXED_DT};

    pub use winit::keyboard::KeyCode;

    pub use millarecs::change_detection::Mut;
    pub use millarecs::entity::Entity;
    pub use millarecs::event::{Despawned, Mutated, Spawned};
    pub use millarecs::hierarchy::{Children, Parent};
    pub use millarecs::query::filter::{Added, AnyOf, Changed, Not, Or, With, Without};
    pub use millarecs::system::commands::Commands;
    pub use millarecs::system::event::{EventIter, EventReader, EventWriter};
    pub use millarecs::system::exclusive_system;
    pub use millarecs::system::local::Local;
    pub use millarecs::system::query::Query;
    pub use millarecs::system::resource::{Res, ResMut};
    pub use millarecs::system::schedule::{Schedule, ScheduleExecutor};
    pub use millarecs::world::{World, WorldError};
    pub use millarecs_macros::{Bundle, Component};
}
