//! Field demo: the canonical "move N entities" query benchmark (up to 5M),
//! for cross-ECS comparisons.
//!
//! ```text
//! cargo run -p millarecs_playground --bin field
//! cargo run -p millarecs_playground --bin field -- --frames 240
//! ```

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    millarecs_playground::run::<millarecs_playground::demos::field::Field>()
}
