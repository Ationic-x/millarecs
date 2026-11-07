//! Boids demo: parallel queries (`Query::par_for_each_mut`) over a
//! multi-thread executor.
//!
//! ```text
//! cargo run -p millarecs_playground --bin boids
//! cargo run -p millarecs_playground --bin boids -- --frames 240
//! ```

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    millarecs_playground::run::<millarecs_playground::demos::boids::Boids>()
}
