//! Standard resources and constants for the demo loop: the per-tick delta
//! time and the simulation viewport, both injected by the app runner.

/// Fixed simulation timestep: the schedule runs exactly once per 1/60 s tick.
pub const FIXED_DT: f32 = 1.0 / 60.0;

/// Resource with the delta time of the current simulation tick. The runner
/// refreshes it before every `schedule.run`; systems read it via `Res<Time>`.
#[derive(Debug, Clone, Copy)]
pub struct Time {
    /// Delta time in seconds for this tick (always [`FIXED_DT`] today).
    pub dt: f32,
}

/// Resource with the simulation viewport in pixels (matches the render
/// target). Systems use it for wrapping, centering and spawning.
#[derive(Debug, Clone, Copy)]
pub struct SimBounds {
    /// Viewport width in pixels.
    pub width: f32,
    /// Viewport height in pixels.
    pub height: f32,
}

impl SimBounds {
    /// Viewport center.
    pub fn center(&self) -> crate::math::Vec2 {
        crate::math::Vec2::new(self.width * 0.5, self.height * 0.5)
    }

    /// Wraps `v` around the viewport as if the edges were joined.
    pub fn wrap(&self, v: crate::math::Vec2) -> crate::math::Vec2 {
        crate::math::Vec2::new(v.x.rem_euclid(self.width), v.y.rem_euclid(self.height))
    }
}
