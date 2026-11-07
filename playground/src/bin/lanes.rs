//! Archetype-lanes demo: table vs sparse component registration and
//! archetype migration driven by insert/remove churn.
//!
//! ```text
//! cargo run -p millarecs_playground --bin lanes
//! cargo run -p millarecs_playground --bin lanes -- --frames 240
//! ```

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    millarecs_playground::run::<millarecs_playground::demos::lanes::Lanes>()
}
