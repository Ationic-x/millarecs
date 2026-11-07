//! HUD overlay: stacks colored text lines in the top-left corner using the
//! bitmap-font atlas.

use crate::atlas::FontAtlas;
use crate::color::Rgba;
use crate::draw::DrawList;
use crate::math::Vec2;

/// Glyph scale used by the HUD (each bitmap pixel becomes `scale` screen
/// pixels).
pub const HUD_SCALE: f32 = 2.0;

/// Padding from the window edges, in pixels.
pub const HUD_MARGIN: f32 = 10.0;

/// Background panel color: dark and semi-transparent so the HUD reads over
/// any scene.
const PANEL: Rgba = [0.02, 0.03, 0.05, 0.78];

/// Panel padding around the text block, in pixels.
const PANEL_PAD: f32 = 8.0;

/// One line of HUD text with its color.
#[derive(Debug, Clone)]
pub struct HudLine {
    /// Text content; unknown characters are skipped by the atlas.
    pub text: String,
    /// Line color.
    pub color: Rgba,
}

impl HudLine {
    /// Convenience constructor for a plain white line.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            color: crate::color::rgb(1.0, 1.0, 1.0),
        }
    }

    /// Convenience constructor with an explicit color.
    pub fn colored(text: impl Into<String>, color: Rgba) -> Self {
        Self {
            text: text.into(),
            color,
        }
    }
}

/// Records `lines` stacked vertically starting at the HUD margin, on top of
/// a dark translucent panel sized to the text block.
pub fn draw_hud(out: &mut DrawList, atlas: &FontAtlas, lines: &[HudLine]) {
    if lines.is_empty() {
        return;
    }
    let line_height = 7.0 * HUD_SCALE + 4.0;

    // The panel spans the widest line (every glyph cell is 6xscale px wide;
    // unknown characters only make the box slightly generous).
    let widest = lines
        .iter()
        .map(|line| line.text.chars().count() as f32 * 6.0 * HUD_SCALE)
        .fold(0.0_f32, f32::max);
    let left = HUD_MARGIN - PANEL_PAD;
    let top = HUD_MARGIN - PANEL_PAD;
    let width = widest + 2.0 * PANEL_PAD;
    let height = lines.len() as f32 * line_height - 4.0 + 2.0 * PANEL_PAD;
    out.rect(
        Vec2::new(left + width * 0.5, top + height * 0.5),
        Vec2::new(width * 0.5, height * 0.5),
        PANEL,
    );

    let mut pen = Vec2::new(HUD_MARGIN, HUD_MARGIN);
    for line in lines {
        out.text(&atlas, pen, &line.text, HUD_SCALE, line.color);
        pen.y += line_height;
    }
}

#[cfg(test)]
mod tests {
    use super::{draw_hud, HudLine, HUD_MARGIN};

    #[test]
    fn hud_stacks_lines_over_a_panel() {
        let atlas = crate::atlas::FontAtlas::with_font_chars();
        let mut out = crate::draw::DrawList::new();
        let lines = [HudLine::new("FPS 60"), HudLine::new("E 100")];
        draw_hud(&mut out, &atlas, &lines);
        assert_eq!(out.glyphs.len(), "FPS 60".len() + "E 100".len());
        assert_eq!(out.sprites.len(), 1, "one translucent panel behind the text");
        assert!(out.sprites[0].color[3] > 0.5, "panel is dark, not opaque");
        let first_y = out.glyphs[0].pos[1];
        let second_y = out.glyphs["FPS 60".len()].pos[1];
        assert!((first_y - HUD_MARGIN).abs() < 1e-4);
        assert!(second_y > first_y);
    }
}
