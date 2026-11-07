//! Frame-scoped draw lists: the demos record sprites and glyphs, and the
//! renderer uploads them as two instance buffers.

use crate::atlas::FontAtlas;
use crate::color::Rgba;
use crate::math::Vec2;

/// GPU instance data for one sprite quad. Layout must stay in sync with the
/// WGSL vertex inputs (`gfx::SPRITE_WGSL`).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteInstance {
    /// Center in pixels.
    pub pos: [f32; 2],
    /// Half extents in pixels (rotation is applied around the center).
    pub half: [f32; 2],
    /// Rotation in radians.
    pub rot: f32,
    /// Non-premultiplied RGBA color.
    pub color: [f32; 4],
    /// Fill rule: a [`Shape`] discriminant.
    pub shape: u32,
    /// Explicit padding to keep the struct 4-byte aligned at 44 bytes.
    pub _pad: f32,
}

/// GPU instance data for one glyph quad (position is the top-left corner in
/// pixels). Layout must stay in sync with the WGSL text pipeline.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GlyphInstance {
    /// Top-left corner in pixels.
    pub pos: [f32; 2],
    /// Glyph quad size in pixels (width x height).
    pub size: [f32; 2],
    /// Top-left UV of the glyph inside the atlas.
    pub uv0: [f32; 2],
    /// Bottom-right UV of the glyph inside the atlas.
    pub uv1: [f32; 2],
    /// Text color (alpha is multiplied by the glyph coverage).
    pub color: [f32; 4],
}

/// Fill rules supported by the sprite fragment shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    /// Solid rectangle covering the quad.
    Rect,
    /// Anti-aliased inscribed ellipse (a circle for square half-extents).
    Circle,
    /// Anti-aliased triangle pointing towards local +X (boid-style).
    Triangle,
}

impl Shape {
    /// Discriminant uploaded in [`SpriteInstance::shape`].
    pub const fn id(self) -> u32 {
        match self {
            Shape::Rect => 0,
            Shape::Circle => 1,
            Shape::Triangle => 2,
        }
    }
}

/// Everything one frame wants on screen. Built by the demo, consumed by the
/// renderer, and cleared between frames.
#[derive(Default)]
pub struct DrawList {
    /// Sprite instances to draw.
    pub sprites: Vec<SpriteInstance>,
    /// Glyph instances to draw (HUD text).
    pub glyphs: Vec<GlyphInstance>,
}

impl DrawList {
    /// Creates an empty list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Drops all recorded instances, keeping capacity for the next frame.
    pub fn clear(&mut self) {
        self.sprites.clear();
        self.glyphs.clear();
    }

    /// Records a filled axis-aligned rectangle centered at `center`.
    pub fn rect(&mut self, center: Vec2, half: Vec2, color: Rgba) {
        self.sprites.push(SpriteInstance {
            pos: [center.x, center.y],
            half: [half.x, half.y],
            rot: 0.0,
            color,
            shape: Shape::Rect.id(),
            _pad: 0.0,
        });
    }

    /// Records a filled rectangle rotated by `rot`, centered at `center`.
    pub fn rot_rect(&mut self, center: Vec2, half: Vec2, rot: f32, color: Rgba) {
        self.sprites.push(SpriteInstance {
            pos: [center.x, center.y],
            half: [half.x, half.y],
            rot,
            color,
            shape: Shape::Rect.id(),
            _pad: 0.0,
        });
    }

    /// Records an anti-aliased circle centered at `center`.
    pub fn circle(&mut self, center: Vec2, radius: f32, color: Rgba) {
        self.sprites.push(SpriteInstance {
            pos: [center.x, center.y],
            half: [radius, radius],
            rot: 0.0,
            color,
            shape: Shape::Circle.id(),
            _pad: 0.0,
        });
    }

    /// Records a triangle pointing along `rot`, centered at `center`.
    pub fn triangle(&mut self, center: Vec2, half: Vec2, rot: f32, color: Rgba) {
        self.sprites.push(SpriteInstance {
            pos: [center.x, center.y],
            half: [half.x, half.y],
            rot,
            color,
            shape: Shape::Triangle.id(),
            _pad: 0.0,
        });
    }

    /// Records `text` with the bitmap font; unknown characters are skipped.
    /// `top_left` is the top-left corner of the first glyph cell.
    pub fn text(
        &mut self,
        atlas: &FontAtlas,
        top_left: Vec2,
        text: &str,
        scale: f32,
        color: Rgba,
    ) {
        let glyph_size = Vec2::new(5.0 * scale, 7.0 * scale);
        let advance = Vec2::new(6.0 * scale, 0.0);
        let mut pen = top_left;
        for c in text.chars() {
            if let Some((uv0, uv1)) = atlas.uv(c) {
                self.glyphs.push(GlyphInstance {
                    pos: [pen.x, pen.y],
                    size: [glyph_size.x, glyph_size.y],
                    uv0,
                    uv1,
                    color,
                });
            }
            pen += advance;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DrawList, GlyphInstance, Shape, SpriteInstance};

    #[test]
    fn helpers_record_expected_instances() {
        let mut out = DrawList::new();
        out.rect(crate::math::Vec2::new(10.0, 10.0), crate::math::Vec2::splat(4.0), crate::color::rgb(1.0, 0.0, 0.0));
        out.circle(crate::math::Vec2::ZERO, 3.0, crate::color::rgb(0.0, 1.0, 0.0));
        out.triangle(crate::math::Vec2::ZERO, crate::math::Vec2::new(2.0, 1.5), 0.5, crate::color::rgb(0.0, 0.0, 1.0));
        assert_eq!(out.sprites.len(), 3);
        assert_eq!(out.sprites[0].shape, Shape::Rect.id());
        assert_eq!(out.sprites[1].shape, Shape::Circle.id());
        assert_eq!(out.sprites[2].shape, Shape::Triangle.id());
        out.clear();
        assert_eq!(out.sprites.len(), 0);
    }

    #[test]
    fn text_skips_unknown_chars() {
        let atlas = crate::atlas::FontAtlas::with_font_chars();
        let mut out = DrawList::new();
        out.text(&atlas, crate::math::Vec2::ZERO, "FPS 60 ñ", 2.0, crate::color::rgb(1.0, 1.0, 1.0));
        assert_eq!(out.glyphs.len(), "FPS 60 ".chars().count());
        assert_eq!(out.glyphs[0].color, [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn sprite_instances_are_pod_at_expected_stride() {
        assert_eq!(
            std::mem::size_of::<SpriteInstance>(),
            44,
            "WGSL sprite instance layout expects 44-byte stride"
        );
        assert_eq!(
            std::mem::size_of::<GlyphInstance>(),
            48,
            "WGSL glyph instance layout expects 48-byte stride"
        );
    }
}
