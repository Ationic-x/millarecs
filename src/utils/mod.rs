//! Utility modules shared by the ECS internals.
//!
//! These helpers are intentionally small and low-level. They are kept outside
//! the entity, component and query modules so those systems can reuse common
//! building blocks without depending on each other.

pub(crate) mod atomic;
pub(crate) mod ecs_vec;
pub(crate) mod flat_map;
pub(crate) mod prefetch;
pub(crate) mod type_size;
