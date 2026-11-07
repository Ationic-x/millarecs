//! App runner shared by all demos: fixed-timestep simulation (60 Hz), winit
//! windowed loop, and an offscreen `--frames N` mode for smoke runs.
//!
//! The runner owns the [`World`]/[`Schedule`]/executor and the window; demos
//! only implement [`Demo`]. Rendering never goes through the ECS: after the
//! simulation ticks, the demo reads the world from outside and records one
//! [`DrawList`] for the GPU.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::World;

use crate::atlas::FontAtlas;
use crate::color::{rgb, Rgba};
use crate::draw::DrawList;
use crate::gfx::Gfx;
use crate::hud::{draw_hud, HudLine};
use crate::math::Vec2;
use crate::panel::{Panel, PanelValues, ParamSpec};
use crate::time::{SimBounds, Time, FIXED_DT};

/// Default window width in logical pixels.
pub const DEFAULT_WIDTH: u32 = 1280;

/// Default window height in logical pixels.
pub const DEFAULT_HEIGHT: u32 = 720;

/// Background clear color.
const CLEAR: Rgba = rgb(0.04, 0.05, 0.07);

/// Hold-to-repeat timing for panel controls: after this initial delay, one
/// step fires per interval until the button or key is released. The interval
/// shrinks the longer the control stays held, so values accelerate without
/// ever taking steps bigger than the spec's.
const REPEAT_DELAY: Duration = Duration::from_millis(320);
const REPEAT_START_MS: f32 = 150.0;
const REPEAT_ACCEL: f32 = 0.85;
const REPEAT_MIN_MS: f32 = 12.0;

/// What is being held for auto-repeat.
enum HeldStep {
    /// An arrow key, re-fed to the panel on every repeat.
    Key(KeyCode),
    /// A panel step button direction (`-1.0` / `1.0`).
    Mouse(f32),
}

/// A panel control held down: repeats its step while the source stays
/// pressed, firing faster the longer it is held.
struct HeldRepeat {
    step: HeldStep,
    started: Instant,
    last: Instant,
    fires: u32,
}

impl HeldRepeat {
    fn key(code: KeyCode) -> Self {
        Self {
            step: HeldStep::Key(code),
            started: Instant::now(),
            last: Instant::now(),
            fires: 0,
        }
    }

    fn mouse(dir: f32) -> Self {
        Self {
            step: HeldStep::Mouse(dir),
            started: Instant::now(),
            last: Instant::now(),
            fires: 0,
        }
    }

    fn is_key(&self, code: KeyCode) -> bool {
        matches!(self.step, HeldStep::Key(k) if k == code)
    }

    fn is_mouse(&self) -> bool {
        matches!(self.step, HeldStep::Mouse(_))
    }

    /// Whether a repeat step is due, updating the step clock. The interval
    /// decays geometrically with the number of fires so far.
    fn due(&mut self) -> bool {
        let now = Instant::now();
        if now.duration_since(self.started) < REPEAT_DELAY {
            return false;
        }
        let interval_ms = (REPEAT_START_MS * REPEAT_ACCEL.powi(self.fires as i32))
            .max(REPEAT_MIN_MS);
        if now.duration_since(self.last).as_secs_f32() * 1000.0 >= interval_ms {
            self.last = now;
            self.fires += 1;
            true
        } else {
            false
        }
    }
}

/// Parsed command-line arguments. Common flags drive the run mode; every
/// other `--key value` pair is forwarded to the demo, which reads it with
/// [`Args::get_u32`] / [`Args::get_f32`] and applies its own defaults and
/// clamps. This is what lets stress parameters change without recompiling.
#[derive(Debug, Default, Clone)]
pub struct Args {
    frames: Option<u64>,
    window_frames: Option<u64>,
    help: bool,
    values: HashMap<String, String>,
}

impl Args {
    /// Parses the process arguments (the executable path is skipped).
    pub fn parse() -> Self {
        Self::parse_iter(std::env::args().skip(1))
    }

    /// Parses an explicit flag iterator; [`Args::parse`] is the process-level
    /// wrapper. Exposed for tests.
    pub fn parse_iter(items: impl Iterator<Item = String>) -> Self {
        let mut args = Args::default();
        let items: Vec<String> = items.collect();
        let mut i = 0;
        while i < items.len() {
            let item = items[i].clone();
            i += 1;
            if item == "--help" || item == "-h" {
                args.help = true;
                continue;
            }
            if !item.starts_with("--") {
                // Positional noise is ignored; demos only read named flags.
                continue;
            }
            let key = item[2..].to_string();
            let Some(value) = items.get(i) else {
                eprintln!("millarecs_playground: flag `--{key}` needs a value");
                args.help = true;
                break;
            };
            let value = value.clone();
            i += 1;
            match key.as_str() {
                "frames" => match value.parse() {
                    Ok(n) => args.frames = Some(n),
                    Err(_) => eprintln!("millarecs_playground: --frames wants a number, got {value:?}"),
                },
                "window-frames" => match value.parse() {
                    Ok(n) => args.window_frames = Some(n),
                    Err(_) => {
                        eprintln!("millarecs_playground: --window-frames wants a number, got {value:?}")
                    }
                },
                _ => {
                    args.values.insert(key, value);
                }
            }
        }
        args
    }

    /// Number of offscreen frames requested by `--frames N` (no window).
    pub fn frames(&self) -> Option<u64> {
        self.frames
    }

    /// Frame limit for a windowed run requested by `--window-frames N`.
    pub fn window_frames(&self) -> Option<u64> {
        self.window_frames
    }

    /// Whether `--help` or an invalid flag asked for the usage text.
    pub fn help_requested(&self) -> bool {
        self.help
    }

    /// Reads `--name` as `u32`, falling back to `default` when absent or
    /// unparsable.
    pub fn get_u32(&self, name: &str, default: u32) -> u32 {
        self.values
            .get(name)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    /// Reads `--name` as `f32`, falling back to `default` when absent or
    /// unparsable.
    pub fn get_f32(&self, name: &str, default: f32) -> f32 {
        self.values
            .get(name)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

/// Snapshot of the keys currently held down, inserted as a world resource
/// every tick so gameplay demos can read continuous input (`Res<InputState>`).
/// Just-pressed keys additionally reach [`Demo::on_key`].
#[derive(Debug, Default, Clone)]
pub struct InputState {
    /// Keys held at the moment the tick started.
    pub held: HashSet<KeyCode>,
}

/// Snapshot passed to [`Demo::frame`] once per simulation tick.
pub struct FrameCtx {
    /// Monotonic tick counter, starting at 0.
    pub frame: u64,
}

/// Keys that toggle the debug HUD overlay (FPS/telemetry/panel) at runtime.
fn is_hud_toggle(code: KeyCode) -> bool {
    code == KeyCode::F1 || code == KeyCode::KeyH
}

/// The playground contract: demos build their own millarecs setup and read
/// the world back for rendering. The runner drives everything else.
pub trait Demo: 'static {
    /// Window title (also shown in headless smoke-run output).
    fn title(&self) -> &'static str;

    /// Usage text printed by `--help`: one line per demo-specific flag,
    /// including its default. Common flags are appended by the runner.
    fn usage() -> String
    where
        Self: Sized,
    {
        String::from("no demo-specific flags")
    }

    /// Runtime-adjustable knobs shown in the on-screen panel (see
    /// [`crate::panel`]); CLI flags with the same key set the initial value.
    fn params(&self) -> Vec<ParamSpec> {
        Vec::new()
    }

    /// When `true` (queried per event), every key reaches [`Demo::on_key`]
    /// and the panel does not intercept the arrows.
    fn owns_keys(&self, _world: &World) -> bool {
        false
    }

    /// When `false`, the parameter panel is not drawn (e.g. behind a menu).
    fn panel_visible(&self, _world: &World) -> bool {
        true
    }

    /// Registers components/resources/systems and spawns the initial
    /// entities. `Time` and `SimBounds` resources are already inserted;
    /// `args` carries the user's stress flags (see [`Args`]).
    fn new(
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
        args: &Args,
    ) -> Self
    where
        Self: Sized;

    /// Runs one fixed 1/60 s tick of simulation (normally one `schedule.run`).
    fn frame(
        &mut self,
        ctx: &FrameCtx,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
    );

    /// Reads the world and records the sprites for this frame. `atlas` is
    /// available for demos that draw labels directly on the canvas.
    fn draw(&mut self, world: &World, atlas: &FontAtlas, out: &mut DrawList);

    /// HUD lines shown under the FPS counter; recomputed each frame.
    fn hud(&mut self) -> Vec<HudLine> {
        Vec::new()
    }

    /// Called for each key pressed since the last tick, before [`Demo::frame`].
    /// Escape closes the window and never reaches here; arrow keys belong to
    /// the parameter panel. `panel` lets demos step their own knobs (orbits
    /// maps Space/D to the systems parameter).
    fn on_key(&mut self, _key: KeyCode, _world: &mut World, _panel: &mut Panel) {}
}

/// Everything one demo instance needs, created once.
struct Session<D: Demo> {
    demo: D,
    world: World,
    schedule: Schedule,
    executor: ScheduleExecutor,
}

impl<D: Demo> Session<D> {
    fn build(args: &Args) -> Self {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .max(1);
        let mut executor = ScheduleExecutor::multi_threaded(workers);
        world.insert_resource(Time { dt: FIXED_DT });
        world.insert_resource(SimBounds {
            width: DEFAULT_WIDTH as f32,
            height: DEFAULT_HEIGHT as f32,
        });
        let demo = D::new(&mut world, &mut schedule, &mut executor, args);
        Self {
            demo,
            world,
            schedule,
            executor,
        }
    }
}

/// Applies queued keys, syncs panel values into the world, and runs one
/// simulation tick.
fn tick_session<D: Demo>(
    session: &mut Session<D>,
    frame: u64,
    keys: &[KeyCode],
    panel: &mut Panel,
    held: &HashSet<KeyCode>,
) {
    for &key in keys {
        session.demo.on_key(key, &mut session.world, panel);
    }
    if panel.take_dirty() {
        let values = panel.values().clone();
        session.world.insert_resource(PanelValues { values });
    }
    session.world.insert_resource(InputState {
        held: held.clone(),
    });
    if let Some(mut time) = session.world.get_resource_mut::<Time>() {
        time.dt = FIXED_DT;
    }
    let ctx = FrameCtx { frame };
    session
        .demo
        .frame(&ctx, &mut session.world, &mut session.schedule, &mut session.executor);
}

/// Entry point of every demo binary. `--frames N` renders N frames offscreen
/// (no window, exits 0); `--window-frames N` opens the window and closes it
/// after N presented frames; without either, the window runs until closed.
/// Any other `--key value` pair is forwarded to the demo as a stress
/// parameter; `--help` prints the demo's usage.
pub fn run<D: Demo>() -> Result<(), Box<dyn Error + Send + Sync>> {
    let args = Args::parse();
    if args.help_requested() {
        println!("{}", D::usage());
        println!(
            "\ncommon flags:\n  --frames N         render N frames offscreen (no window), then exit\n  --window-frames N  open the window and close it after N frames\n  --help             show this help"
        );
        return Ok(());
    }
    if let Some(frames) = args.frames() {
        run_headless::<D>(frames, &args)
    } else {
        run_windowed::<D>(args.window_frames(), &args)
    }
}

fn run_headless<D: Demo>(frames: u64, args: &Args) -> Result<(), Box<dyn Error + Send + Sync>> {
    let mut session = Session::<D>::build(args);
    let mut panel = Panel::new(session.demo.params(), args, DEFAULT_HEIGHT as f32);
    session.world.insert_resource(PanelValues {
        values: panel.values().clone(),
    });
    let mut gfx = pollster::block_on(Gfx::new_headless((DEFAULT_WIDTH, DEFAULT_HEIGHT)))?;
    let atlas = FontAtlas::with_font_chars();
    let mut out = DrawList::new();

    for frame in 0..frames {
        let held = HashSet::new();
        tick_session(&mut session, frame, &[], &mut panel, &held);
        out.clear();
        session.demo.draw(&session.world, &atlas, &mut out);
        let mut lines = vec![HudLine::new(format!("FPS --  FRAME {frame}"))];
        lines.extend(session.demo.hud());
        draw_hud(&mut out, &atlas, &lines);
        panel.draw(&mut out, &atlas, DEFAULT_HEIGHT as f32);
        gfx.render(&out, CLEAR);
    }

    println!(
        "millarecs_playground: demo '{}' rendered {} frames headless ({}x{})",
        session.demo.title(),
        frames,
        gfx.size().0,
        gfx.size().1
    );
    Ok(())
}

fn run_windowed<D: Demo>(
    window_frames: Option<u64>,
    args: &Args,
) -> Result<(), Box<dyn Error + Send + Sync>> {
    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = WindowedApp::<D> {
        session: None,
        gfx: None,
        window: None,
        atlas: FontAtlas::with_font_chars(),
        out: DrawList::new(),
        pending_keys: Vec::new(),
        last: None,
        accumulator: 0.0,
        frame: 0,
        fps: FpsMeter::new(),
        window_frames,
        presented: 0,
        args: args.clone(),
        panel: None,
        cursor: None,
        held: None,
        held_keys: HashSet::new(),
        sim_ms: 0.0,
        draw_ms: 0.0,
        gpu_ms: 0.0,
        render_ms: 0.0,
        sim_peak_ms: 0.0,
        sys_ms: HashMap::new(),
        sys_order: Vec::new(),
        hud_visible: true,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

/// Rolling FPS meter; reports the rate every half second and keeps the worst
/// frame of each window for the HUD. `tick` returns `true` when the window
/// rolls, so callers can reset their own per-window peaks.
struct FpsMeter {
    frames: u64,
    last_report: Instant,
    fps: f32,
    peak_frame_ms: f32,
}

impl FpsMeter {
    fn new() -> Self {
        Self {
            frames: 0,
            last_report: Instant::now(),
            fps: 0.0,
            peak_frame_ms: 0.0,
        }
    }

    fn tick(&mut self, frame_ms: f32) -> bool {
        if frame_ms > self.peak_frame_ms {
            self.peak_frame_ms = frame_ms;
        }
        self.frames += 1;
        let elapsed = self.last_report.elapsed().as_secs_f32();
        if elapsed >= 0.5 {
            self.fps = self.frames as f32 / elapsed;
            self.frames = 0;
            self.peak_frame_ms = 0.0;
            self.last_report = Instant::now();
            true
        } else {
            false
        }
    }

    fn fps(&self) -> f32 {
        self.fps
    }

    /// Slowest complete frame since the last reported window.
    fn peak_frame_ms(&self) -> f32 {
        self.peak_frame_ms
    }
}

struct WindowedApp<D: Demo> {
    session: Option<Session<D>>,
    gfx: Option<Gfx>,
    window: Option<Arc<Window>>,
    atlas: FontAtlas,
    out: DrawList,
    pending_keys: Vec<KeyCode>,
    last: Option<Instant>,
    accumulator: f32,
    frame: u64,
    fps: FpsMeter,
    /// Frame limit for bounded windowed runs (`--window-frames N`).
    window_frames: Option<u64>,
    /// Redraws presented so far; only compared when `window_frames` is set.
    presented: u64,
    /// User flags; the session is built with them on first resume.
    args: Args,
    /// Runtime parameter panel (created once the demo exists).
    panel: Option<Panel>,
    /// Last cursor position in physical pixels, for mouse interactions.
    cursor: Option<Vec2>,
    /// Panel control held down for auto-repeat (arrow key or step button).
    held: Option<HeldRepeat>,
    /// Keys currently held, published to the world as [`InputState`] each
    /// tick.
    held_keys: HashSet<KeyCode>,
    /// Exponential average of simulation cost per tick, in milliseconds.
    sim_ms: f32,
    /// Exponential average of the demo's extract pass (`Demo::draw`), in
    /// milliseconds.
    draw_ms: f32,
    /// Exponential average of the GPU submission (`gfx.render` + HUD glyph
    /// recording), in milliseconds.
    gpu_ms: f32,
    /// Exponential average of the whole render section, in milliseconds.
    render_ms: f32,
    /// Slowest single simulation tick since the last redraw, in milliseconds.
    sim_peak_ms: f32,
    /// Smoothed per-system cost (ms per run) from [`crate::perf`].
    sys_ms: HashMap<&'static str, f32>,
    /// System labels in first-seen order, so the SYS lines stay put.
    sys_order: Vec<&'static str>,
    /// Whether the HUD overlay (lines + panel) is drawn; F1/H toggles it.
    hud_visible: bool,
}

impl<D: Demo> ApplicationHandler for WindowedApp<D> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let mut session = Session::<D>::build(&self.args.clone());
        let attrs = Window::default_attributes()
            .with_title(session.demo.title())
            .with_inner_size(LogicalSize::new(DEFAULT_WIDTH, DEFAULT_HEIGHT));
        let window = match event_loop.create_window(attrs) {
            Ok(window) => Arc::new(window),
            Err(e) => {
                eprintln!("millarecs_playground: window creation failed: {e}");
                event_loop.exit();
                return;
            }
        };
        let physical = window.inner_size();
        let display = event_loop.owned_display_handle();
        match pollster::block_on(Gfx::new_windowed(
            &window,
            display,
            (physical.width, physical.height),
        )) {
            Ok(gfx) => self.gfx = Some(gfx),
            Err(e) => {
                eprintln!("millarecs_playground: GPU initialization failed: {e}");
                event_loop.exit();
                return;
            }
        }
        if let Some(mut bounds) = session.world.get_resource_mut::<SimBounds>() {
            *bounds = SimBounds {
                width: physical.width as f32,
                height: physical.height as f32,
            };
        }
        let panel = Panel::new(
            session.demo.params(),
            &self.args.clone(),
            physical.height as f32,
        );
        session.world.insert_resource(PanelValues {
            values: panel.values().clone(),
        });
        self.panel = Some(panel);
        self.last = Some(Instant::now());
        self.session = Some(session);
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(gfx) = &mut self.gfx {
                    gfx.resize((size.width, size.height));
                }
                if let Some(session) = &mut self.session {
                    if let Some(mut bounds) = session.world.get_resource_mut::<SimBounds>() {
                        *bounds = SimBounds {
                            width: size.width as f32,
                            height: size.height as f32,
                        };
                    }
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed if !event.repeat => {
                            if code == KeyCode::Escape {
                                event_loop.exit();
                            } else if is_hud_toggle(code) {
                                // F1/H: hide or show the debug HUD overlay.
                                self.hud_visible = !self.hud_visible;
                            } else {
                                let owns = self
                                    .session
                                    .as_ref()
                                    .map(|s| s.demo.owns_keys(&s.world))
                                    .unwrap_or(false);
                                if !owns
                                    && self
                                        .panel
                                        .as_mut()
                                        .is_some_and(|p| p.on_key(code))
                                {
                                    // Panel arrows start a hold-to-repeat.
                                    self.held = Some(HeldRepeat::key(code));
                                } else {
                                    self.pending_keys.push(code);
                                    self.held_keys.insert(code);
                                }
                            }
                        }
                        ElementState::Released => {
                            self.held_keys.remove(&code);
                            if self.held.as_ref().is_some_and(|held| held.is_key(code)) {
                                self.held = None;
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let pos = Vec2::new(position.x as f32, position.y as f32);
                self.cursor = Some(pos);
                if let Some(panel) = self.panel.as_mut() {
                    panel.cursor(pos);
                    panel.drag_to(pos);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if button == winit::event::MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            if let (Some(panel), Some(pos)) = (self.panel.as_mut(), self.cursor) {
                                if let crate::panel::PressHit::Step(dir) = panel.press(pos) {
                                    // Step buttons start a hold-to-repeat.
                                    self.held = Some(HeldRepeat::mouse(dir));
                                }
                            }
                        }
                        ElementState::Released => {
                            if let Some(panel) = self.panel.as_mut() {
                                panel.release();
                            }
                            if self.held.as_ref().is_some_and(|held| held.is_mouse()) {
                                self.held = None;
                            }
                        }
                    }
                }
            }
            WindowEvent::Focused(false) => {
                // Alt-tab and friends can swallow key/button releases.
                self.held = None;
                self.held_keys.clear();
                if let Some(panel) = self.panel.as_mut() {
                    panel.release();
                }
            }
            WindowEvent::RedrawRequested => self.redraw(event_loop),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

impl<D: Demo> WindowedApp<D> {
    fn redraw(&mut self, event_loop: &ActiveEventLoop) {
        let Some(session) = &mut self.session else {
            return;
        };
        let Some(gfx) = &mut self.gfx else {
            return;
        };
        let Some(panel) = &mut self.panel else {
            return;
        };

        // Hold-to-repeat: fire one panel step per interval while a button or
        // arrow key stays pressed.
        if let Some(held) = &mut self.held {
            if held.due() {
                match held.step {
                    HeldStep::Key(code) => {
                        panel.on_key(code);
                    }
                    HeldStep::Mouse(dir) => {
                        panel.step_selected(dir);
                    }
                }
            }
        }

        // Fixed-step simulation: consume real time in 1/60 s chunks, at most
        // 15 ticks per redraw to avoid spiral-of-death catch-up.
        let now = Instant::now();
        let delta = self
            .last
            .map(|t| now.duration_since(t).as_secs_f32())
            .unwrap_or(FIXED_DT);
        self.last = Some(now);
        self.accumulator = (self.accumulator + delta).min(0.25);
        let sim_start = Instant::now();
        let mut ticks = 0u32;
        while self.accumulator >= FIXED_DT {
            let tick_start = Instant::now();
            let keys = std::mem::take(&mut self.pending_keys);
            tick_session(session, self.frame, &keys, panel, &self.held_keys);
            let tick_ms = tick_start.elapsed().as_secs_f32() * 1000.0;
            if tick_ms > self.sim_peak_ms {
                self.sim_peak_ms = tick_ms;
            }
            self.accumulator -= FIXED_DT;
            self.frame += 1;
            ticks += 1;
        }
        if ticks > 0 {
            let per_tick = sim_start.elapsed().as_secs_f32() * 1000.0 / ticks as f32;
            self.sim_ms = self.sim_ms + (per_tick - self.sim_ms) * 0.15;
        }

        let render_start = Instant::now();
        self.out.clear();
        let draw_start = Instant::now();
        session.demo.draw(&session.world, &self.atlas, &mut self.out);
        let draw_ms = draw_start.elapsed().as_secs_f32() * 1000.0;
        self.draw_ms += (draw_ms - self.draw_ms) * 0.15;

        // The HUD overlay (lines + panel) can be toggled with F1/H; the
        // telemetry registry still drains every frame so nothing smears
        // while the overlay is hidden.
        let hud_on = self.hud_visible;
        let mut lines = Vec::new();
        if hud_on {
            lines.push(HudLine::colored(
                format!(
                    "FPS {:>6.1}  FRAME {}  SIM {:>4.1}  DRAW {:>4.1}  GPU {:>4.1} MS",
                    self.fps.fps(),
                    self.frame,
                    self.sim_ms,
                    self.draw_ms,
                    self.gpu_ms
                ),
                rgb(0.85, 0.85, 0.85),
            ));
            lines.extend(session.demo.hud());
        }

        // Per-system telemetry: drain the perf registry, smooth it, and show
        // one HUD line per handful of systems. Labels persist in first-seen
        // order and keep their last smoothed value between ticks (the sim
        // runs at 60 Hz but the render is uncapped, so many redraws carry no
        // new samples).
        let samples = crate::perf::snapshot_and_reset();
        for (label, sample) in samples {
            let instant = sample.ms_per_run();
            let ema = self.sys_ms.entry(label).or_insert(instant);
            *ema += (instant - *ema) * 0.2;
            if !self.sys_order.contains(&label) {
                self.sys_order.push(label);
            }
        }
        let telemetry_visible = hud_on && session.demo.panel_visible(&session.world);
        if telemetry_visible && !self.sys_order.is_empty() {
            let parts: Vec<String> = self
                .sys_order
                .iter()
                .filter_map(|label| {
                    self.sys_ms
                        .get(label)
                        .map(|value| format!("{label} {value:.2}"))
                })
                .collect();
            for chunk in parts.chunks(4) {
                lines.push(HudLine::colored(
                    format!("SYS {}", chunk.join(" - ")),
                    rgb(0.55, 0.78, 0.92),
                ));
            }
            lines.push(HudLine::colored(
                format!(
                    "PICO TICK {:.1}MS - PICO FRAME {:.1}MS",
                    self.sim_peak_ms,
                    self.fps.peak_frame_ms()
                ),
                rgb(0.75, 0.6, 0.85),
            ));
        }

        if hud_on {
            draw_hud(&mut self.out, &self.atlas, &lines);
        }
        if hud_on && session.demo.panel_visible(&session.world) {
            panel.draw(&mut self.out, &self.atlas, gfx.size().1 as f32);
        }
        let gpu_start = Instant::now();
        gfx.render(&self.out, CLEAR);
        let gpu_ms = gpu_start.elapsed().as_secs_f32() * 1000.0;
        self.gpu_ms += (gpu_ms - self.gpu_ms) * 0.15;
        let frame_ms = render_start.elapsed().as_secs_f32() * 1000.0;
        self.render_ms += (frame_ms - self.render_ms) * 0.15;
        if self.fps.tick(frame_ms) {
            // Window rolled: start a fresh peak for the next one.
            self.sim_peak_ms = 0.0;
        }

        self.presented += 1;
        if let Some(limit) = self.window_frames {
            if self.presented >= limit {
                event_loop.exit();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Args, FpsMeter};

    fn args(list: &[&str]) -> Args {
        Args::parse_iter(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn parses_modes_and_demo_flags() {
        let a = args(&["--frames", "240", "--count", "5000"]);
        assert_eq!(a.frames(), Some(240));
        assert_eq!(a.get_u32("count", 1400), 5000);
        assert!(!a.help_requested());

        let b = args(&["--window-frames", "600", "--rate", "48.5"]);
        assert_eq!(b.window_frames(), Some(600));
        assert_eq!(b.frames(), None);
        assert!((b.get_f32("rate", 12.0) - 48.5).abs() < 1e-4);
    }

    #[test]
    fn missing_or_bad_values_fall_back_to_defaults() {
        let a = args(&["--count", "abc"]);
        assert_eq!(a.get_u32("count", 1400), 1400, "unparsable value -> default");
        let b = Args::default();
        assert_eq!(b.frames(), None);
        assert_eq!(b.get_u32("systems", 2), 2);
    }

    #[test]
    fn valueless_flag_and_help_request_usage() {
        let a = args(&["--help"]);
        assert!(a.help_requested());
        let b = args(&["--count"]);
        assert!(b.help_requested(), "a dangling flag requests the usage text");
    }

    #[test]
    fn fps_meter_reports_window_rolls() {
        let mut meter = FpsMeter::new();
        // Offsets accumulate on the same window anchor: keep the first two
        // ticks under 0.5 s of elapsed time, then push past it.
        meter.last_report -= std::time::Duration::from_millis(400);
        assert!(!meter.tick(5.0), "first tick stays inside the window");
        meter.last_report -= std::time::Duration::from_millis(60);
        assert!(!meter.tick(9.0));
        assert!((meter.peak_frame_ms() - 9.0).abs() < 1e-5);

        // Force the window to roll; the peak resets with it (the frame that
        // triggered the roll belonged to the old window).
        meter.last_report -= std::time::Duration::from_millis(200);
        assert!(meter.tick(2.0), "window rolled");
        assert!((meter.peak_frame_ms() - 0.0).abs() < 1e-5, "fresh window");
        assert!(meter.fps() > 0.0);
    }

    #[test]
    fn hud_toggle_keys() {
        use super::is_hud_toggle;
        use winit::keyboard::KeyCode;

        assert!(is_hud_toggle(KeyCode::F1), "F1 toggles the HUD");
        assert!(is_hud_toggle(KeyCode::KeyH), "H toggles the HUD");
        assert!(!is_hud_toggle(KeyCode::Space), "game keys do not");
        assert!(!is_hud_toggle(KeyCode::ArrowUp), "panel keys do not");
        assert!(!is_hud_toggle(KeyCode::Escape), "escape stays reserved");
    }
}
