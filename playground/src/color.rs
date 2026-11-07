//! Color helpers: linear RGBA tuples ready for the GPU.

/// A color as non-premultiplied linear RGBA, each channel in `[0.0, 1.0]`.
pub type Rgba = [f32; 4];

/// Opaque color from 8-bit components.
pub const fn rgb8(r: u8, g: u8, b: u8) -> Rgba {
    [
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        1.0,
    ]
}

/// Opaque color from float components.
pub const fn rgb(r: f32, g: f32, b: f32) -> Rgba {
    [r, g, b, 1.0]
}

/// Color from float components with alpha.
pub const fn rgba(r: f32, g: f32, b: f32, a: f32) -> Rgba {
    [r, g, b, a]
}

/// Converts HSL to RGBA. `h` is in turns (`0.0..1.0` wrapping), `s` and `l`
/// in `[0.0, 1.0]`; alpha is 1.0.
pub fn hsl(h: f32, s: f32, l: f32) -> Rgba {
    let h = h.rem_euclid(1.0);
    if s <= 0.0 {
        return [l, l, l, 1.0];
    }

    fn channel(p: f32, q: f32, mut t: f32) -> f32 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            p + (q - p) * 6.0 * t
        } else if t < 0.5 {
            q
        } else if t < 2.0 / 3.0 {
            p + (q - p) * (2.0 / 3.0 - t) * 6.0
        } else {
            p
        }
    }

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    [
        channel(p, q, h + 1.0 / 3.0),
        channel(p, q, h),
        channel(p, q, h - 1.0 / 3.0),
        1.0,
    ]
}

/// Linear interpolation between two colors, component-wise including alpha.
pub fn lerp(a: Rgba, b: Rgba, t: f32) -> Rgba {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
        a[3] + (b[3] - a[3]) * t,
    ]
}

/// Returns the color with its alpha multiplied by `alpha`.
pub fn with_alpha(c: Rgba, alpha: f32) -> Rgba {
    [c[0], c[1], c[2], c[3] * alpha]
}

/// Scales the RGB channels by `k`, leaving alpha untouched.
pub fn scale_rgb(c: Rgba, k: f32) -> Rgba {
    [c[0] * k, c[1] * k, c[2] * k, c[3]]
}

#[cfg(test)]
mod tests {
    use super::{hsl, lerp, rgb8, with_alpha};

    #[test]
    fn hsl_primaries() {
        let red = hsl(0.0, 1.0, 0.5);
        assert!((red[0] - 1.0).abs() < 1e-5 && red[1] < 1e-5 && red[2] < 1e-5);
        let green = hsl(1.0 / 3.0, 1.0, 0.5);
        assert!(green[1] > 0.99 && green[0] < 1e-5);
    }

    #[test]
    fn rgb8_normalizes() {
        let white = rgb8(255, 255, 255);
        assert!((white[0] - 1.0).abs() < 1e-5);
        assert!((white[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn lerp_and_alpha() {
        let c = lerp([0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], 0.5);
        assert!((c[0] - 0.5).abs() < 1e-6);
        let faded = with_alpha(c, 0.5);
        assert!((faded[3] - 0.5).abs() < 1e-6, "alpha 1.0 scaled by 0.5");
    }
}
