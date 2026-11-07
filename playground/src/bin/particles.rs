//! Fireworks demo: `Commands` spawn/despawn churn, `Spawned`/`Despawned`
//! lifecycle events and `Local<T>` per-system timers.
//!
//! ```text
//! cargo run -p millarecs_playground --bin particles
//! cargo run -p millarecs_playground --bin particles -- --frames 240
//! ```

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    millarecs_playground::run::<millarecs_playground::demos::particles::Particles>()
}
