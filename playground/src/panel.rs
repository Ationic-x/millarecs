//! Runtime parameter panel.
//!
//! Every demo exposes numeric knobs ([`ParamSpec`]) that the runner shows as
//! an on-screen panel anchored bottom-left: drag the slider or click the
//! `-`/`+` buttons with the mouse, or select a row with `Up`/`Down` and step
//! it with `Left`/`Right`. Changed values are published to the world as a
//! [`PanelValues`] resource, so demo systems pick them up on the next tick:
//! tweak parameters live without rebuilding. CLI flags with the same key
//! only set the initial value.

use std::collections::HashMap;

use winit::keyboard::KeyCode;

use crate::app::Args;
use crate::atlas::FontAtlas;
use crate::color::Rgba;
use crate::draw::DrawList;
use crate::math::Vec2;

/// One adjustable numeric parameter shown as a panel row.
pub struct ParamSpec {
    /// Lookup key shared with CLI flags and `PanelValues`.
    pub key: &'static str,
    /// Short label drawn in the panel.
    pub label: &'static str,
    /// Value used when no flag overrides it.
    pub default: f32,
    /// Lowest allowed value.
    pub min: f32,
    /// Highest allowed value.
    pub max: f32,
    /// Value delta per step (button click, arrow press, or repeat tick).
    pub step: f32,
    /// Whether the value snaps to whole numbers.
    pub integer: bool,
}

impl ParamSpec {
    /// Creates a parameter; chain [`ParamSpec::integer`] for whole-number
    /// knobs.
    pub const fn new(
        key: &'static str,
        label: &'static str,
        default: f32,
        min: f32,
        max: f32,
        step: f32,
    ) -> Self {
        Self {
            key,
            label,
            default,
            min,
            max,
            step,
            integer: false,
        }
    }

    /// Makes the parameter snap to whole numbers.
    pub const fn integer(mut self) -> Self {
        self.integer = true;
        self
    }

    /// Clamps `value` into `[min, max]`, rounding it when integer.
    pub fn clamp(&self, value: f32) -> f32 {
        let value = value.clamp(self.min, self.max);
        if self.integer {
            value.round()
        } else {
            value
        }
    }

    /// Slider position in `[0, 1]` for `value`.
    pub fn to_slider(&self, value: f32) -> f32 {
        ((value - self.min) / (self.max - self.min)).clamp(0.0, 1.0)
    }

    /// Value at slider position `t` in `[0, 1]`.
    pub fn from_slider(&self, t: f32) -> f32 {
        self.clamp(self.min + t.clamp(0.0, 1.0) * (self.max - self.min))
    }
}

/// Snapshot of the panel values, inserted as a world resource every time a
/// value changes; systems read their knobs with [`PanelValues::get_or`].
#[derive(Clone, Default)]
pub struct PanelValues {
    /// Current value per parameter key.
    pub values: HashMap<&'static str, f32>,
}

impl PanelValues {
    /// Computes the initial snapshot for `specs`, letting matching CLI flags
    /// in `args` override each default.
    pub fn init(specs: &[ParamSpec], args: &Args) -> Self {
        let mut values = HashMap::new();
        for spec in specs {
            let raw = if spec.integer {
                args.get_u32(spec.key, spec.default as u32) as f32
            } else {
                args.get_f32(spec.key, spec.default)
            };
            values.insert(spec.key, spec.clamp(raw));
        }
        Self { values }
    }

    /// Reads `key`, falling back to `fallback` when it is absent.
    pub fn get_or(&self, key: &str, fallback: f32) -> f32 {
        self.values.get(key).copied().unwrap_or(fallback)
    }
}

// Panel chrome.
const PANEL_BG: Rgba = [0.02, 0.03, 0.05, 0.88];
const TRACK: Rgba = [0.16, 0.18, 0.22, 1.0];
const FILL: Rgba = [0.20, 0.75, 0.85, 1.0];
const KNOB: Rgba = [0.95, 0.95, 0.98, 1.0];
const TEXT: Rgba = [0.85, 0.86, 0.9, 1.0];
const TEXT_DIM: Rgba = [0.5, 0.52, 0.58, 1.0];
const SEL: Rgba = [1.0, 0.82, 0.25, 1.0];
const HOVER: Rgba = [0.1, 0.12, 0.16, 0.6];

// Panel geometry, in physical pixels.
const MARGIN: f32 = 12.0;
const PAD: f32 = 8.0;
const WIDTH: f32 = 330.0;
const HEADER_H: f32 = 22.0;
const ROW_H: f32 = 42.0;
const BUTTON: f32 = 16.0;
const TEXT_SCALE: f32 = 1.5;
const TRACK_H: f32 = 8.0;

/// Which panel element a screen position landed on.
enum Hit {
    /// The slider track of row `i`.
    Track(usize),
    /// The `-` button of row `i`.
    Minus(usize),
    /// The `+` button of row `i`.
    Plus(usize),
    /// The row background of `i` (selection only).
    Row(usize),
}

/// Outcome of a press, so the runner knows what to auto-repeat while the
/// button stays held.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PressHit {
    /// Nothing actionable was under the cursor.
    None,
    /// The slider track: dragging adjusts it continuously, no repeat needed.
    Track,
    /// A step button; the payload is its direction (`-1.0` or `1.0`).
    Step(f32),
    /// A row background (selection only).
    Row,
}

/// Interactive parameter panel: owns the specs, current values, selection,
/// and mouse-drag state. Lives in the app thread; only its [`PanelValues`]
/// snapshots enter the world.
pub struct Panel {
    specs: Vec<ParamSpec>,
    values: HashMap<&'static str, f32>,
    selected: usize,
    dirty: bool,
    hover: Option<usize>,
    drag: Option<usize>,
    height: f32,
}

impl Panel {
    /// Builds the panel for `specs`; `args` provide initial overrides and
    /// `height` anchors the bottom-left layout.
    pub fn new(specs: Vec<ParamSpec>, args: &Args, height: f32) -> Self {
        let values = PanelValues::init(&specs, args).values;
        Self {
            specs,
            values,
            selected: 0,
            dirty: false,
            hover: None,
            drag: None,
            height,
        }
    }

    /// Current values (the source of truth for resource syncing).
    pub fn values(&self) -> &HashMap<&'static str, f32> {
        &self.values
    }

    /// Returns whether any value changed since the last call, clearing the
    /// flag.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// Steps the parameter named `key` by `steps` x its step size (used by
    /// demo keys such as orbits' Space/D).
    pub fn adjust(&mut self, key: &str, steps: f32) {
        if let Some(index) = self.specs.iter().position(|spec| spec.key == key) {
            self.selected = index;
            self.step(index, steps);
        }
    }

    /// Handles arrow keys: `Up`/`Down` move the selection, `Left`/`Right`
    /// step the selected value. Returns whether the key was consumed (demo
    /// key handlers are skipped then).
    pub fn on_key(&mut self, code: KeyCode) -> bool {
        if self.specs.is_empty() {
            return false;
        }
        let len = self.specs.len();
        match code {
            KeyCode::ArrowUp => {
                self.selected = (self.selected + len - 1) % len;
                true
            }
            KeyCode::ArrowDown => {
                self.selected = (self.selected + 1) % len;
                true
            }
            KeyCode::ArrowLeft => {
                self.step(self.selected, -1.0);
                true
            }
            KeyCode::ArrowRight => {
                self.step(self.selected, 1.0);
                true
            }
            _ => false,
        }
    }

    /// Updates the hovered row from a cursor move (physical pixels).
    pub fn cursor(&mut self, pos: Vec2) {
        self.hover = match self.hit_test(pos) {
            Some(Hit::Track(i)) | Some(Hit::Minus(i)) | Some(Hit::Plus(i)) | Some(Hit::Row(i)) => {
                Some(i)
            }
            None => None,
        };
    }

    /// Handles a left-button press: tracks start a drag, buttons step and
    /// report their direction for hold-to-repeat, rows select.
    pub fn press(&mut self, pos: Vec2) -> PressHit {
        match self.hit_test(pos) {
            Some(Hit::Track(i)) => {
                self.selected = i;
                self.drag = Some(i);
                let value = self.value_from_x(i, pos.x);
                self.set(i, value);
                PressHit::Track
            }
            Some(Hit::Minus(i)) => {
                self.selected = i;
                self.step(i, -1.0);
                PressHit::Step(-1.0)
            }
            Some(Hit::Plus(i)) => {
                self.selected = i;
                self.step(i, 1.0);
                PressHit::Step(1.0)
            }
            Some(Hit::Row(i)) => {
                self.selected = i;
                PressHit::Row
            }
            None => PressHit::None,
        }
    }

    /// Updates the dragged slider from a cursor move while the button holds.
    pub fn drag_to(&mut self, pos: Vec2) {
        if let Some(i) = self.drag {
            let value = self.value_from_x(i, pos.x);
            self.set(i, value);
        }
    }

    /// Ends an in-progress slider drag.
    pub fn release(&mut self) {
        self.drag = None;
    }

    /// Steps the selected row (the auto-repeat entry point for held buttons
    /// and arrows).
    pub fn step_selected(&mut self, steps: f32) {
        self.step(self.selected, steps);
    }

    /// Records the panel into `out`, anchored at the bottom-left of a
    /// viewport `viewport_height` pixels tall.
    pub fn draw(&mut self, out: &mut DrawList, atlas: &FontAtlas, viewport_height: f32) {
        self.height = viewport_height;
        if self.specs.is_empty() {
            return;
        }

        let (_, panel_h) = self.panel_size();
        let top = self.panel_top();
        out.rect(
            Vec2::new(MARGIN + WIDTH * 0.5, top + panel_h * 0.5),
            Vec2::new(WIDTH * 0.5, panel_h * 0.5),
            PANEL_BG,
        );
        out.text(
            atlas,
            Vec2::new(MARGIN + PAD, top + PAD + 2.0),
            "PARAMS - ARROWS OR DRAG",
            TEXT_SCALE,
            TEXT_DIM,
        );

        for (i, spec) in self.specs.iter().enumerate() {
            let row_top = self.row_top(i);
            if self.hover == Some(i) {
                out.rect(
                    Vec2::new(MARGIN + WIDTH * 0.5, row_top + ROW_H * 0.5),
                    Vec2::new(WIDTH * 0.5 - 2.0, ROW_H * 0.5 - 1.0),
                    HOVER,
                );
            }
            if self.selected == i {
                out.rect(
                    Vec2::new(MARGIN + 1.5, row_top + ROW_H * 0.5),
                    Vec2::new(1.5, ROW_H * 0.5 - 3.0),
                    SEL,
                );
            }

            let value = self.values.get(spec.key).copied().unwrap_or(spec.default);
            let label_color = if self.selected == i { SEL } else { TEXT };
            out.text(
                atlas,
                Vec2::new(MARGIN + PAD + 4.0, row_top + 3.0),
                spec.label,
                TEXT_SCALE,
                label_color,
            );
            let value_text = format_value(value, spec.integer);
            let value_w = value_text.chars().count() as f32 * 6.0 * TEXT_SCALE;
            out.text(
                atlas,
                Vec2::new(MARGIN + WIDTH - PAD - value_w, row_top + 3.0),
                &value_text,
                TEXT_SCALE,
                TEXT,
            );

            // Step buttons at the track ends.
            let ((mx, my), (px, py)) = self.button_centers(i);
            out.rect(Vec2::new(mx, my), Vec2::splat(BUTTON * 0.5), TRACK);
            out.rect(Vec2::new(px, py), Vec2::splat(BUTTON * 0.5), TRACK);
            let gs = 1.2;
            out.text(atlas, Vec2::new(mx - 3.0 * gs, my - 3.5 * gs), "-", gs, TEXT);
            out.text(atlas, Vec2::new(px - 3.0 * gs, py - 3.5 * gs), "+", gs, TEXT);

            // Slider track, fill, and knob.
            let (x0, x1) = self.track_x();
            let ty = self.track_y(i);
            out.rect(
                Vec2::new((x0 + x1) * 0.5, ty),
                Vec2::new((x1 - x0) * 0.5, TRACK_H * 0.5),
                TRACK,
            );
            let t = spec.to_slider(value);
            let fx = x0 + (x1 - x0) * t;
            if fx > x0 {
                out.rect(
                    Vec2::new((x0 + fx) * 0.5, ty),
                    Vec2::new((fx - x0) * 0.5, TRACK_H * 0.5),
                    FILL,
                );
            }
            out.circle(Vec2::new(fx, ty), 5.0, KNOB);
        }
    }

    fn step(&mut self, index: usize, steps: f32) {
        let key = self.specs[index].key;
        let current = self
            .values
            .get(&key)
            .copied()
            .unwrap_or(self.specs[index].default);
        let target = self.specs[index].clamp(current + self.specs[index].step * steps);
        self.set(index, target);
    }

    fn set(&mut self, index: usize, value: f32) {
        let spec = &self.specs[index];
        let clamped = spec.clamp(value);
        let changed = (self
            .values
            .get(&spec.key)
            .copied()
            .unwrap_or(spec.default)
            - clamped)
            .abs()
            > f32::EPSILON;
        if changed {
            self.values.insert(spec.key, clamped);
            self.dirty = true;
        }
    }

    fn value_from_x(&self, index: usize, x: f32) -> f32 {
        let (x0, x1) = self.track_x();
        let t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0);
        self.specs[index].from_slider(t)
    }

    fn hit_test(&self, pos: Vec2) -> Option<Hit> {
        for i in 0..self.specs.len() {
            let ((mx, my), (px, py)) = self.button_centers(i);
            if near(pos, mx, my, BUTTON * 0.5 + 3.0) {
                return Some(Hit::Minus(i));
            }
            if near(pos, px, py, BUTTON * 0.5 + 3.0) {
                return Some(Hit::Plus(i));
            }
            let (x0, x1) = self.track_x();
            let ty = self.track_y(i);
            if pos.x >= x0 - 4.0 && pos.x <= x1 + 4.0 && (pos.y - ty).abs() <= 10.0 {
                return Some(Hit::Track(i));
            }
        }
        for i in 0..self.specs.len() {
            let top = self.row_top(i);
            if pos.x >= MARGIN && pos.x <= MARGIN + WIDTH && pos.y >= top && pos.y <= top + ROW_H
            {
                return Some(Hit::Row(i));
            }
        }
        None
    }

    fn panel_size(&self) -> (f32, f32) {
        (
            WIDTH,
            PAD * 2.0 + HEADER_H + self.specs.len() as f32 * ROW_H,
        )
    }

    fn panel_top(&self) -> f32 {
        let (_, h) = self.panel_size();
        self.height - MARGIN - h
    }

    fn row_top(&self, index: usize) -> f32 {
        self.panel_top() + PAD + HEADER_H + index as f32 * ROW_H
    }

    fn track_x(&self) -> (f32, f32) {
        (
            MARGIN + PAD + BUTTON + 8.0,
            MARGIN + WIDTH - PAD - BUTTON - 8.0,
        )
    }

    fn track_y(&self, index: usize) -> f32 {
        self.row_top(index) + ROW_H - 14.0
    }

    fn button_centers(&self, index: usize) -> ((f32, f32), (f32, f32)) {
        let y = self.track_y(index);
        (
            (MARGIN + PAD + BUTTON * 0.5, y),
            (MARGIN + WIDTH - PAD - BUTTON * 0.5, y),
        )
    }
}

fn near(pos: Vec2, cx: f32, cy: f32, r: f32) -> bool {
    (pos.x - cx).abs() <= r && (pos.y - cy).abs() <= r
}

fn format_value(value: f32, integer: bool) -> String {
    if integer {
        format!("{}", value.round() as i64)
    } else {
        format!("{value:.1}")
    }
}

#[cfg(test)]
mod tests {
    use super::{format_value, ParamSpec, Panel, PanelValues};
    use crate::app::Args;
    use crate::math::Vec2;
    use std::collections::HashMap;

    fn specs() -> Vec<ParamSpec> {
        vec![
            ParamSpec::new("count", "COUNT", 1400.0, 100.0, 200_000.0, 1.0).integer(),
            ParamSpec::new("rate", "RATE", 12.0, 1.0, 240.0, 1.0).integer(),
        ]
    }

    fn args(list: &[&str]) -> Args {
        Args::parse_iter(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn init_overrides_defaults_from_flags_and_clamps() {
        let values = PanelValues::init(&specs(), &args(&["--count", "5000"]));
        assert_eq!(values.get_or("count", 0.0), 5000.0);
        assert_eq!(values.get_or("rate", 0.0), 12.0);

        let clamped = PanelValues::init(&specs(), &args(&["--count", "999999"]));
        assert_eq!(clamped.get_or("count", 0.0), 200_000.0, "clamped to max");
    }

    #[test]
    fn arrow_keys_select_and_step() {
        let mut panel = Panel::new(specs(), &Args::default(), 720.0);
        assert!(panel.on_key(winit::keyboard::KeyCode::ArrowRight));
        assert_eq!(panel.values()[&"count"], 1401.0, "step +1 from default");

        assert!(panel.on_key(winit::keyboard::KeyCode::ArrowDown));
        assert!(panel.on_key(winit::keyboard::KeyCode::ArrowRight));
        assert_eq!(panel.values()[&"rate"], 13.0);
        assert!(panel.take_dirty(), "stepping marks the panel dirty");

        assert!(!panel.on_key(winit::keyboard::KeyCode::Space),
            "non-arrow keys are left for the demo");
    }

    #[test]
    fn press_and_drag_set_slider_values() {
        let mut panel = Panel::new(specs(), &Args::default(), 720.0);
        // One row of specs: 100..=200_000, integer. Press the track middle.
        let (x0, x1) = (44.0, 310.0); // track_x with the constants above
        let ty = panel.row_top(0) + 42.0 - 14.0;
        panel.press(Vec2::new((x0 + x1) * 0.5, ty));
        assert_eq!(panel.values()[&"count"], 100_050.0, "midpoint of the span");

        panel.drag_to(Vec2::new(x1, ty));
        assert_eq!(panel.values()[&"count"], 200_000.0);
        panel.drag_to(Vec2::new(x0 - 500.0, ty));
        assert_eq!(panel.values()[&"count"], 100.0, "dragging past the ends clamps");
        panel.release();
    }

    #[test]
    fn buttons_step_by_spec_step() {
        let mut panel = Panel::new(specs(), &Args::default(), 720.0);
        let ((mx, my), _) = panel.button_centers(0);
        assert!(matches!(
            panel.press(Vec2::new(mx, my)),
            super::PressHit::Step(dir) if dir < 0.0
        ));
        assert_eq!(panel.values()[&"count"], 1399.0, "- button steps down by 1");

        let (_, (px, py)) = panel.button_centers(0);
        assert!(matches!(
            panel.press(Vec2::new(px, py)),
            super::PressHit::Step(dir) if dir > 0.0
        ));
        assert_eq!(panel.values()[&"count"], 1400.0);
        panel.step_selected(1.0);
        assert_eq!(panel.values()[&"count"], 1401.0, "step_selected repeats the last row");
        assert!(matches!(panel.press(Vec2::new(5000.0, 5.0)), super::PressHit::None));
    }

    #[test]
    fn adjust_moves_named_parameters() {
        let mut panel = Panel::new(specs(), &Args::default(), 720.0);
        panel.adjust("rate", 3.0);
        assert_eq!(panel.values()[&"rate"], 15.0);
        panel.adjust("missing", 1.0); // unknown keys are ignored
        assert!(panel.take_dirty());
    }

    #[test]
    fn float_formatting() {
        assert_eq!(format_value(1234.0, true), "1234");
        assert_eq!(format_value(12.34, false), "12.3");
    }

    #[test]
    fn linear_slider_maps_by_proportion() {
        let specs = vec![ParamSpec::new("n", "N", 1_000_000.0, 10_000.0, 5_000_000.0, 1.0)
            .integer()];

        assert_eq!(specs[0].from_slider(0.0), 10_000.0);
        assert_eq!(specs[0].from_slider(1.0), 5_000_000.0);
        assert_eq!(specs[0].from_slider(0.5), 2_505_000.0, "midpoint of the span");
        assert!((specs[0].to_slider(2_505_000.0) - 0.5).abs() < 1e-6, "roundtrip");
    }

    #[test]
    fn empty_values_map_falls_back() {
        let values = PanelValues::default();
        assert_eq!(values.get_or("count", 42.0), 42.0);
        let manual = PanelValues {
            values: HashMap::from([("count", 7.0)]),
        };
        assert_eq!(manual.get_or("count", 42.0), 7.0);
    }
}
