//! CPU-side glyph atlas: packs the [`crate::font`] bitmaps into one R8
//! texture that the text pipeline samples, plus UV lookup per character.

use std::collections::HashMap;

use crate::font::{glyph, FONT_CHARS};

/// Cell size of one glyph slot in the atlas, in texels.
pub const CELL: (u32, u32) = (8, 9);

/// Number of glyph columns per atlas row.
pub const ATLAS_COLS: u32 = 16;

/// A packed atlas texture (single channel, 255 = opaque glyph pixel) and the
/// normalized UV rectangle of every character it contains.
pub struct FontAtlas {
    size: (u32, u32),
    pixels: Vec<u8>,
    cells: HashMap<char, ([f32; 2], [f32; 2])>,
}

impl FontAtlas {
    /// Builds an atlas containing every character in [`crate::font::FONT_CHARS`].
    pub fn with_font_chars() -> Self {
        Self::build(FONT_CHARS.chars())
    }

    /// Builds an atlas for an arbitrary character set (later duplicates are
    /// ignored).
    pub fn build(chars: impl Iterator<Item = char>) -> Self {
        let chars: Vec<char> = {
            let mut seen = Vec::new();
            for c in chars {
                if !seen.contains(&c) && glyph(c).is_some() {
                    seen.push(c);
                }
            }
            seen
        };
        let rows = ((chars.len() as u32) + ATLAS_COLS - 1) / ATLAS_COLS;
        let size = (ATLAS_COLS * CELL.0, rows.max(1) * CELL.1);
        let mut pixels = vec![0u8; (size.0 * size.1) as usize];

        let mut cells = HashMap::new();
        for (i, c) in chars.iter().enumerate() {
            let col = (i as u32) % ATLAS_COLS;
            let row = (i as u32) / ATLAS_COLS;
            let base_x = col * CELL.0;
            let base_y = row * CELL.1;
            let bitmap = glyph(*c).unwrap();
            for (y, bits) in bitmap.iter().enumerate() {
                for x in 0..5 {
                    if bits & (1 << x) != 0 {
                        // One texel of padding: glyph occupies (1..6, 1..8).
                        let px = base_x + 1 + x as u32;
                        let py = base_y + 1 + y as u32;
                        pixels[(py * size.0 + px) as usize] = 255;
                    }
                }
            }
            let u0 = base_x as f32 / size.0 as f32;
            let v0 = base_y as f32 / size.1 as f32;
            let u1 = (base_x + 6) as f32 / size.0 as f32;
            let v1 = (base_y + 8) as f32 / size.1 as f32;
            cells.insert(*c, ([u0, v0], [u1, v1]));
        }

        Self {
            size,
            pixels,
            cells,
        }
    }

    /// Texture dimensions in texels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// The packed single-channel pixel data, row-major.
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    /// Normalized `(uv0, uv1)` corners of the glyph rectangle for `c`.
    pub fn uv(&self, c: char) -> Option<([f32; 2], [f32; 2])> {
        self.cells.get(&c).copied()
    }

    /// How many distinct glyphs the atlas holds.
    pub fn glyph_count(&self) -> usize {
        self.cells.len()
    }
}

#[cfg(test)]
mod tests {
    use super::{FontAtlas, CELL};

    #[test]
    fn atlas_contains_full_charset() {
        let atlas = FontAtlas::with_font_chars();
        assert_eq!(atlas.glyph_count(), super::super::font::FONT_CHARS.chars().count());
        assert_eq!(atlas.size().0, super::ATLAS_COLS * CELL.0);
    }

    #[test]
    fn uvs_are_normalized_and_ordered() {
        let atlas = FontAtlas::with_font_chars();
        let (uv0, uv1) = atlas.uv('A').unwrap();
        let (u0, v0) = (uv0[0], uv0[1]);
        let (u1, v1) = (uv1[0], uv1[1]);
        assert!((0.0..1.0).contains(&u0) && (u0..1.0).contains(&u1));
        assert!((0.0..1.0).contains(&v0) && (v0..1.0).contains(&v1));
    }

    #[test]
    fn glyph_pixels_are_packed() {
        let atlas = FontAtlas::with_font_chars();
        assert!(atlas.pixels().iter().any(|&p| p == 255));
        assert_eq!(atlas.uv('ñ'), None);
    }
}
