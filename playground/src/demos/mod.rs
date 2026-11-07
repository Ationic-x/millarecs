//! Demo registry: each submodule is a standalone visual example, exposed as
//! a [`crate::app::Demo`] implementation so its simulation can also be unit
//! tested headlessly.

pub mod boids;
pub mod bullet_hell;
pub mod field;
pub mod lanes;
pub mod orbits;
pub mod particles;
