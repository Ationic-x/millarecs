//! Bullet-hell demo: a scripted shooter with waves, bullet patterns,
//! collisions and explosion sparks.
//!
//! ```text
//! cargo run -p millarecs_playground --bin bullet_hell
//! cargo run -p millarecs_playground --bin bullet_hell -- --frames 240
//! ```

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    millarecs_playground::run::<millarecs_playground::demos::bullet_hell::BulletHell>()
}
