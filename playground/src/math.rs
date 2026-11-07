//! Minimal 2D vector math shared by the demos and the renderer.

use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A 2D vector of `f32` components, in pixels for positions and sizes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    /// Horizontal component; grows to the right.
    pub x: f32,
    /// Vertical component; grows downwards (screen convention).
    pub y: f32,
}

impl Vec2 {
    /// Builds a vector from its components.
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Builds a vector with both components set to `v`.
    pub const fn splat(v: f32) -> Self {
        Self { x: v, y: v }
    }

    /// The zero vector.
    pub const ZERO: Self = Self { x: 0.0, y: 0.0 };

    /// The `(1, 1)` vector.
    pub const ONE: Self = Self { x: 1.0, y: 1.0 };

    /// Dot product; also the squared length when both operands are equal.
    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// Squared euclidean length.
    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    /// Euclidean length.
    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    /// Distance between two points.
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    /// Unit vector with the same direction, or the zero vector when `self`
    /// is degenerate.
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len > f32::EPSILON {
            self * (1.0 / len)
        } else {
            Self::ZERO
        }
    }

    /// Unit vector pointing at `angle` radians (0 points along +X).
    pub fn from_angle(angle: f32) -> Self {
        Self {
            x: angle.cos(),
            y: angle.sin(),
        }
    }

    /// Heading of the vector in radians, measured from +X towards +Y.
    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }

    /// Linear interpolation between `self` and `other` by `t` in `[0, 1]`.
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }

    /// Component-wise minimum.
    pub fn min(self, other: Self) -> Self {
        Self {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }

    /// Component-wise maximum.
    pub fn max(self, other: Self) -> Self {
        Self {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }

    /// Clamps the length to `max_len`, leaving direction untouched.
    pub fn clamp_length(self, max_len: f32) -> Self {
        let len = self.length();
        if len > max_len && len > f32::EPSILON {
            self * (max_len / len)
        } else {
            self
        }
    }
}

impl Add for Vec2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl SubAssign for Vec2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul<f32> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl MulAssign<f32> for Vec2 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

#[cfg(test)]
mod tests {
    use super::Vec2;

    #[test]
    fn dot_and_length_are_consistent() {
        let v = Vec2::new(3.0, 4.0);
        assert!((v.length() - 5.0).abs() < 1e-5);
        assert!((v.dot(v) - v.length_squared()).abs() < 1e-5);
    }

    #[test]
    fn from_angle_and_angle_roundtrip() {
        let v = Vec2::from_angle(1.25);
        assert!((v.angle() - 1.25).abs() < 1e-5);
        assert!((v.length() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn normalized_zero_is_zero() {
        assert_eq!(Vec2::ZERO.normalized(), Vec2::ZERO);
    }

    #[test]
    fn clamp_length_keeps_short_vectors() {
        let v = Vec2::new(1.0, 0.0).clamp_length(2.0);
        assert!((v.x - 1.0).abs() < 1e-6);
    }
}
