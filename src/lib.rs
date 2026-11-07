//! Core ECS runtime.
//!
//! The crate provides entity allocation, component storage, resources, queries,
//! events, and a dependency-aware schedule executor. Storage internals are
//! intentionally split between archetype tables, sparse sets, and query
//! execution planning so each part can evolve without leaking implementation
//! details into user-facing APIs.
//!
//! # Panic safety
//!
//! If a panic escapes a structural operation (spawn, insert, remove,
//! despawn, or any batch variant), the world must be discarded: column
//! metadata is committed after per-column work completes and cannot be
//! rolled back during unwinding. Catching such a panic and continuing to
//! use the world is undefined behavior.
//!
//! # Iteration order
//!
//! Query iteration order is not guaranteed. Table routes visit archetypes in
//! internal order and sparse-driver routes may reorder rows; systems that
//! need a specific processing order must impose it through explicit
//! scheduling dependencies, not through entity or component layout.
//!
//! # Events
//!
//! An event is readable during the cycle it was sent in and the next one;
//! after that it is retired. Applications that need longer retention must
//! copy the data out before retirement. Removing and re-adding an event
//! resource starts a fresh identity, so existing readers reset cleanly.
//!
//! # Parallel execution
//!
//! [`System`](system::System) bodies that stay under roughly 50 µs
//! parallelize well; below that, per-system dispatch overhead erodes
//! scaling. Exclusive systems always run alone, inlined between
//! parallel phases.
//!
//! # Handle recycling
//!
//! Entity generations wrap after 2³² recycles of a single index. A handle
//! stored that long may alias a live entity; keep handles for less than the
//! lifetime of that many resurrections.

#![warn(missing_docs)]

pub mod change_detection;
pub mod component;
pub mod entity;
pub mod event;
pub mod hierarchy;
pub mod plugin;
pub mod query;
pub mod resource;
pub mod system;
pub mod world;

mod planning;
mod task;
mod utils;

#[doc(hidden)]
pub mod __private {
    pub use crate::component::bundle::{
        __bundle_erased_item_from_ref, __bundle_erased_item_of, __bundle_run, ErasedBundleItem,
    };
    pub use crate::component::{Bundle, BundleError, BundleInfo};
    pub use crate::component::{Component, ComponentId, ComponentManager};
}
