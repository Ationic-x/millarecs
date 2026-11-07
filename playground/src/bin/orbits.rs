//! Solar-systems demo: `Parent`/`Children` hierarchy with `add_child`,
//! exclusive-system transform propagation and `despawn_with_descendants`
//! (Space adds a system, D removes one, Escape closes).
//!
//! ```text
//! cargo run -p millarecs_playground --bin orbits
//! cargo run -p millarecs_playground --bin orbits -- --frames 240
//! ```

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    millarecs_playground::run::<millarecs_playground::demos::orbits::Orbits>()
}
