//! Classic bullet-hell shooter: a scripted level of enemy waves and a boss
//! with four named spell cards. Slow, readable bullets draw shapes (rings,
//! double spirals, sweeping fans, accelerating stars), plus focus mode,
//! bombs, grazing, lives and a level-select menu.
//!
//! Controls: **WASD** moves (arrows stay with the parameter panel), hold
//! **Shift/Space** to focus (slow + visible hitbox), **X** bombs (screen
//! clear + damage), **R** returns to the menu, **Escape** closes.
//!
//! ECS features on display: an exclusive level *director* that owns the
//! timeline (waves, boss, spells, level clear), per-level enemy pools, a
//! data-driven spell table dispatching to pattern functions, one fused
//! movement system with optional fetches for pattern correction and
//! accelerating bullets, an exclusive collision system with graze scoring,
//! `Commands` churn everywhere and the shared per-system telemetry.

use std::time::Instant;

use millarecs::system::schedule::{Schedule, ScheduleExecutor};
use millarecs::world::{World, WorldError};

use crate::prelude::*;

/// Entity position in pixels.
#[derive(Component, Clone, Copy)]
pub struct Pos(pub Vec2);

/// Linear velocity in pixels per second (bullets and sparks; fairies and the
/// boss move through their own systems).
#[derive(Component, Clone, Copy)]
pub struct Vel(pub Vec2);

/// Velocity multiplier per second, for accelerating bullets.
#[derive(Component, Clone, Copy)]
pub struct Accel(pub f32);

/// Visual and collision size, color, and graze bookkeeping of a bullet.
#[derive(Component, Clone, Copy)]
pub struct Bullet {
    /// Collision and draw radius.
    pub radius: f32,
    /// Bullet color.
    pub tint: Rgba,
    /// Whether this bullet was already grazed (scored once).
    pub grazed: bool,
}

/// Fired by enemies or the boss.
#[derive(Component)]
pub struct EnemyShot;

/// Fired by the player.
#[derive(Component)]
pub struct PlayerShot;

/// Small enemy archetype. Each stage mixes a different pool of kinds.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FairyKind {
    /// Crosses the arena on a straight line, firing aimed singles.
    Sweeper,
    /// Weaves horizontally while descending, firing slow rings.
    Weaver,
    /// Dives fast toward the player, firing one quick aimed shot.
    Diver,
    /// Drifts in, stops mid-screen for fast aimed spreads, then dives.
    Sniper,
    /// Orbits a slowly descending anchor, firing aimed pairs.
    Orbiter,
}

/// Enemy state.
#[derive(Component, Clone, Copy)]
pub struct Fairy {
    /// Remaining hits.
    pub hp: f32,
    /// Archetype.
    pub kind: FairyKind,
    /// Pattern clock in seconds.
    pub t: f32,
    /// Horizontal anchor for weavers/orbiters.
    pub x0: f32,
    /// Vertical anchor for orbiters.
    pub y0: f32,
    /// Weaver/orbiter angular speed (rad/s).
    pub omega: f32,
    /// Weaver/orbiter amplitude (px).
    pub amp: f32,
    /// Straight-line velocity for sweepers/divers/snipers.
    pub dir: Vec2,
    /// Seconds until the next volley.
    pub fire_cd: f32,
    /// Seconds between volleys.
    pub fire_rate: f32,
    /// White hit-flash remaining (seconds).
    pub flash: f32,
}

/// Collectible dropped by defeated enemies or converted from bullets.
#[derive(Component, Clone, Copy)]
pub struct Pickup {
    /// Seconds alive (drives the spin/pulse).
    pub t: f32,
    /// Kind of reward.
    pub kind: PickupKind,
    /// Score granted on collection.
    pub value: u32,
}

/// What a pickup grants when collected.
#[derive(Clone, Copy, PartialEq)]
pub enum PickupKind {
    /// Shot power (+1 power point).
    Power,
    /// Score bonus.
    Score,
}

/// The player's last position, published as a resource by `player_control`
/// so aiming systems can read it without touching `Pos` (which they do not
/// write, keeping every `Pos` writer conflict-free).
#[derive(Clone, Copy, Default)]
pub struct PlayerPos(pub Option<Vec2>);

/// The player's single entity.
#[derive(Component)]
pub struct Player;

/// Invulnerability window in seconds.
#[derive(Component, Clone, Copy)]
pub struct Iframes(pub f32);

/// Explosion spark: drifts, fades and dies.
#[derive(Component, Clone, Copy)]
pub struct Spark {
    /// Remaining life in seconds.
    pub life: f32,
    /// Life at spawn, for the fade fraction.
    pub max: f32,
    /// Spark color.
    pub tint: Rgba,
}

/// The boss: one entity alive per spell card.
#[derive(Component, Clone, Copy)]
pub struct Boss {
    /// Current spell index (into the stage's [`STAGE_SPELLS`] row).
    pub spell: usize,
    /// Remaining spell health.
    pub hp: f32,
    /// Health at spell start (draws the bar).
    pub max_hp: f32,
    /// Seconds inside this spell.
    pub t: f32,
    /// Position the boss drifts around.
    pub target: Vec2,
    /// Pattern sub-timer.
    pub volley: f32,
    /// Secondary pattern accumulator.
    pub sub: f32,
    /// Spiral angle.
    pub angle: f32,
    /// Spiral direction (+1/-1).
    pub dir: f32,
    /// White hit-flash remaining (seconds).
    pub flash: f32,
}

/// Stage flow. The exclusive director owns every transition.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Phase {
    /// Stage picker; the demo owns the keys while here.
    Menu,
    /// Short breather before the first wave.
    Intro,
    /// Fairy waves.
    Waves,
    /// Pause that also clears leftover bullets before the boss.
    ClearPause,
    /// The boss is on screen running spell `boss.spell`.
    BossSpell,
    /// Stage defeated: bonus, then next stage.
    StageClear,
}

/// Display names of the six levels, easiest first. Generic by design: the
/// menu and announcements only ever say "LEVEL N".
pub const STAGE_NAMES: [&str; 6] = [
    "LEVEL 1",
    "LEVEL 2",
    "LEVEL 3",
    "LEVEL 4",
    "LEVEL 5",
    "LEVEL 6",
];

/// Number of selectable levels.
pub const STAGE_COUNT: u32 = STAGE_NAMES.len() as u32;

/// Round state, mutated by the director and the collision system.
#[derive(Clone, Copy)]
pub struct GameState {
    /// Current stage (1-based).
    pub stage: u32,
    /// Current phase.
    pub phase: Phase,
    /// Seconds left in the current phase timer.
    pub phase_timer: f32,
    /// Fairies waves spawned this stage.
    pub wave_index: u32,
    /// Seconds spent inside the current spell.
    pub spell_timer: f32,
    /// Seconds of gameplay since the stage started.
    pub game_time: f32,
    /// Points scored this round.
    pub score: u32,
    /// Lives left.
    pub lives: u32,
    /// Bombs left.
    pub bombs: u32,
    /// Raw power collected; shot level grows every 5 points.
    pub power: u32,
    /// Near-misses scored.
    pub graze: u32,
    /// Whether the round is over (R returns to the menu).
    pub game_over: bool,
    /// Selected row in the stage menu.
    pub menu_index: u32,
    /// Red hit-flash overlay strength (seconds remaining).
    pub hit_flash: f32,
    /// Screen shake magnitude in pixels.
    pub shake: f32,
    /// Shake oscillator phase.
    pub shake_phase: f32,
}

impl Default for GameState {
    fn default() -> Self {
        Self {
            stage: 1,
            phase: Phase::Menu,
            phase_timer: 0.0,
            wave_index: 0,
            spell_timer: 0.0,
            game_time: 0.0,
            score: 0,
            lives: 3,
            bombs: 2,
            power: 0,
            graze: 0,
            game_over: false,
            menu_index: 0,
            hit_flash: 0.0,
            shake: 0.0,
            shake_phase: 0.0,
        }
    }
}

/// Fresh round state for `stage`, as if picked from the menu.
pub fn start_state(stage: u32) -> GameState {
    GameState {
        stage: stage.clamp(1, STAGE_COUNT),
        phase: Phase::Intro,
        phase_timer: 1.0,
        ..GameState::default()
    }
}

/// Seconds before a spell auto-resolves (survive it for the bonus).
const SPELL_DURATION: f32 = 40.0;
/// Fairy waves per stage.
const WAVES_PER_STAGE: u32 = 6;
/// Spell cards per stage boss.
const SPELLS_PER_STAGE: usize = 4;

/// Boss title per level, shown on the spell announcement. Generic by
/// design: the game carries no setting, only levels and bosses.
pub const BOSS_NAMES: [&str; 6] = ["BOSS 1", "BOSS 2", "BOSS 3", "BOSS 4", "BOSS 5", "BOSS 6"];

/// Level color identity: boss body, enemies and patterns tint from here.
const STAGE_HUES: [f32; 6] = [0.42, 0.58, 0.03, 0.62, 0.6, 0.98];

/// A spell card: display name plus its pattern runner.
pub struct SpellDef {
    /// Spell name shown in the announcement and HUD.
    pub name: &'static str,
    /// Per-tick pattern behavior.
    pub run: PatternFn,
}

/// Pattern runner: receives the mutable boss timers and firing context.
pub type PatternFn = for<'a> fn(&mut SpellCtx<'a>);

const fn spell(name: &'static str, run: PatternFn) -> SpellDef {
    SpellDef { name, run }
}

/// The four spell cards of each level boss. Every level runs a different
/// sequence over a shared library of eight patterns, named after the shape
/// the bullets draw.
pub const STAGE_SPELLS: [[SpellDef; SPELLS_PER_STAGE]; 6] = [
    [
        spell("Twin Rings", pattern_rings),
        spell("Sweep Fan", pattern_fan),
        spell("Twin Spiral", pattern_spiral),
        spell("Star Burst", pattern_stars),
    ],
    [
        spell("Bubbles", pattern_bubbles),
        spell("Twin Spiral", pattern_spiral),
        spell("Twin Rings", pattern_rings),
        spell("Sweep Fan", pattern_fan),
    ],
    [
        spell("Bullet Rain", pattern_rain),
        spell("Sweep Fan", pattern_fan),
        spell("Rotating Cross", pattern_cross),
        spell("Twin Rings", pattern_rings),
    ],
    [
        spell("Star Burst", pattern_stars),
        spell("Rotating Cross", pattern_cross),
        spell("Bloom", pattern_flower),
        spell("Twin Spiral", pattern_spiral),
    ],
    [
        spell("Bullet Rain", pattern_rain),
        spell("Bloom", pattern_flower),
        spell("Twin Spiral", pattern_spiral),
        spell("Twin Rings", pattern_rings),
    ],
    [
        spell("Rotating Cross", pattern_cross),
        spell("Star Burst", pattern_stars),
        spell("Bloom", pattern_flower),
        spell("Twin Spiral", pattern_spiral),
    ],
];

/// Spell card `spell` of level `stage`.
fn spell_def(stage: u32, spell: usize) -> &'static SpellDef {
    &STAGE_SPELLS[(stage.max(1) - 1) as usize % STAGE_SPELLS.len()][spell % SPELLS_PER_STAGE]
}

/// Enemy archetypes available per level; every level mixes a different set.
fn stage_enemy_pool(stage: u32) -> &'static [FairyKind] {
    match stage.max(1).min(6) {
        2 => &[FairyKind::Weaver, FairyKind::Orbiter, FairyKind::Sweeper],
        3 => &[FairyKind::Sweeper, FairyKind::Sniper, FairyKind::Weaver],
        4 => &[FairyKind::Diver, FairyKind::Orbiter, FairyKind::Weaver],
        5 => &[FairyKind::Sniper, FairyKind::Orbiter, FairyKind::Diver],
        6 => &[FairyKind::Diver, FairyKind::Sniper, FairyKind::Orbiter],
        _ => &[FairyKind::Sweeper, FairyKind::Weaver],
    }
}
/// Player constants.
const PLAYER_SPEED: f32 = 390.0;
const FOCUS_MULT: f32 = 0.45;
const FIRE_CD: f32 = 0.09;
const SHOT_SPEED: f32 = 700.0;
/// Player hitbox radius (tiny, genre standard) and graze band.
const HITBOX: f32 = 3.0;
const GRAZE_BAND: f32 = 12.0;
/// Arena margin the player cannot leave.
const PLAYER_MARGIN: f32 = 16.0;
/// Radius added to both circles for a hit.
const HIT_PAD: f32 = 2.0;
/// Seconds of invulnerability after being hit.
const HIT_IFRAMES: f32 = 2.5;

#[derive(Default)]
struct FireTimer {
    t: f32,
}

fn difficulty(values: &PanelValues) -> f32 {
    values.get_or("difficulty", 2.0).clamp(1.0, 5.0)
}

/// Bullet speed multiplier for the current difficulty and stage.
fn speed_mult(diff: f32, stage: u32) -> f32 {
    0.85 + 0.1 * diff + 0.03 * (stage - 1) as f32
}

/// Spell health at `stage` (break it before the timer, or survive it).
fn spell_hp(spell: usize, stage: u32) -> f32 {
    460.0 + 60.0 * spell as f32 + 140.0 * (stage - 1) as f32
}

/// Shot streams fired per volley for a given raw power total. Cheap on
/// purpose: one power pickup per level, so the ship maxes out within the
/// first stage (L2 at 2, L3 at 4, L4 at 6, L5 at 8).
pub fn shot_level(power: u32) -> u32 {
    (1 + power / 2).min(5)
}

/// Removes every enemy bullet currently alive without reward (player-hit
/// screen clear, bombs).
fn clear_enemy_shots(world: &mut World) -> Result<(), WorldError> {
    let mut doomed = Vec::new();
    {
        let mut q = world.query::<Entity, With<EnemyShot>>()?;
        q.for_each(|e| doomed.push(e));
    }
    if !doomed.is_empty() {
        world.despawn_batch(doomed)?;
    }
    Ok(())
}

/// Classic bullet-hell harvest: converts every enemy bullet on screen into a
/// score pickup at its position, then despawns the bullets. Used when a
/// phase completes (waves end, spell breaks, stage clears), letting the
/// player absorb the pattern they just survived.
fn harvest_enemy_shots(world: &mut World, stage: u32) -> Result<u32, WorldError> {
    let mut doomed = Vec::new();
    let mut positions = Vec::new();
    {
        let mut q = world.query::<(Entity, &Pos), With<EnemyShot>>()?;
        q.for_each(|(e, p)| {
            doomed.push(e);
            positions.push(p.0);
        });
    }
    let count = doomed.len() as u32;
    if !doomed.is_empty() {
        with_world_rng(world, |world, rng| {
            let value = 40 + 10 * stage;
            for at in &positions {
                let _ = world.spawn((
                    Pos(*at),
                    Vel(Vec2::from_angle(rng.angle()) * rng.range(40.0, 130.0)),
                    Pickup {
                        t: rng.range(0.0, 6.0),
                        kind: PickupKind::Score,
                        value,
                    },
                ));
            }
            Ok(())
        })?;
        world.despawn_batch(doomed)?;
    }
    Ok(count)
}

/// Removes every fairy still on screen, with a small pop at each one. Used
/// when the boss arrives: the stage ends, the fairies do not linger.
fn clear_fairies(world: &mut World) -> Result<(), WorldError> {
    let mut doomed = Vec::new();
    let mut centers = Vec::new();
    {
        let mut q = world.query::<(Entity, &Pos), With<Fairy>>()?;
        q.for_each(|(e, p)| {
            doomed.push(e);
            centers.push(p.0);
        });
    }
    if !doomed.is_empty() {
        world.despawn_batch(doomed)?;
        for center in centers {
            burst_sparks(world, center, 6, color::hsl(0.12, 0.7, 0.72))?;
        }
    }
    Ok(())
}

/// Spawns the boss for `spell` at a comfortable position for that pattern.
fn spawn_boss(world: &mut World, spell: usize, stage: u32) -> Result<(), WorldError> {
    let bounds = *world.get_resource::<SimBounds>().expect("SimBounds");
    let target = match spell {
        0 => Vec2::new(bounds.width * 0.5, bounds.height * 0.22),
        1 => Vec2::new(bounds.width * 0.5, bounds.height * 0.18),
        2 => Vec2::new(bounds.width * 0.5, bounds.height * 0.2),
        _ => Vec2::new(bounds.width * 0.5, bounds.height * 0.22),
    };
    let hp = spell_hp(spell, stage);
    // Every spell anchors the boss on the horizontal center; only the height
    // and drift radius change per pattern.
    let _ = world.spawn((
        Pos(target),
        Boss {
            spell,
            hp,
            max_hp: hp,
            t: 0.0,
            target,
            volley: 0.6,
            sub: 0.4,
            angle: 0.0,
            dir: 1.0,
            flash: 0.0,
        },
    ));
    Ok(())
}

/// The stage director: an exclusive system owning the timeline. Spawns fairy
/// waves, clears the screen between acts, summons the boss per spell, detects
/// breaks (hp) and timeouts, pays bonuses and advances the stage.
fn director(world: &mut World) -> Result<(), WorldError> {
    let start = Instant::now();
    if game_over_flag(world) {
        perf::record("director", start.elapsed().as_nanos() as u64);
        return Ok(());
    }
    let dt = world.get_resource::<Time>().map(|t| t.dt).unwrap_or(0.0);
    let bounds = *world.get_resource::<SimBounds>().expect("SimBounds");

    let (phase, timer, wave_index, stage, spell_timer) = {
        let state = world.get_resource::<GameState>().expect("GameState");
        (
            state.phase,
            state.phase_timer,
            state.wave_index,
            state.stage,
            state.spell_timer,
        )
    };

    // Feedback decay + gameplay clock, shared by every phase.
    if let Some(mut state) = world.get_resource_mut::<GameState>() {
        state.game_time += dt;
        state.hit_flash = (state.hit_flash - dt).max(0.0);
        state.shake = (state.shake - 16.0 * dt).max(0.0);
        state.shake_phase += dt * 34.0;
    }

    match phase {
        Phase::Menu => {
            // The stage picker is pure UI; nothing advances here.
        }
        Phase::Intro => {
            set_phase(world, Phase::Waves, if timer - dt <= 0.0 { 0.4 } else { timer - dt });
        }
        Phase::Waves => {
            if timer - dt <= 0.0 {
                if wave_index < WAVES_PER_STAGE {
                    spawn_wave(world, wave_index, stage, bounds)?;
                    if let Some(mut state) = world.get_resource_mut::<GameState>() {
                        state.wave_index += 1;
                    }
                    set_phase(world, Phase::Waves, 3.0);
                } else {
                    // Screen sweep before the boss arrives: leftover bullets
                    // become absorbable score pickups.
                    harvest_enemy_shots(world, stage)?;
                    set_phase(world, Phase::ClearPause, 1.4);
                }
            } else {
                set_phase(world, Phase::Waves, timer - dt);
            }
        }
        Phase::ClearPause => {
            // Boss incoming: sweep bullets (as pickups) AND fairies off the
            // screen.
            harvest_enemy_shots(world, stage)?;
            clear_fairies(world)?;
            if timer - dt <= 0.0 {
                spawn_boss(world, 0, stage)?;
                set_phase(world, Phase::BossSpell, 0.0);
                if let Some(mut state) = world.get_resource_mut::<GameState>() {
                    state.spell_timer = 0.0;
                }
            } else {
                set_phase(world, Phase::ClearPause, timer - dt);
            }
        }
        Phase::BossSpell => {
            // Snapshot the boss first, then mutate through handles.
            let mut boss: Option<(Entity, Vec2, f32, f32)> = None;
            {
                let mut q = world.query_all::<(Entity, &Pos, &Boss)>()?;
                q.for_each(|(e, _p, b)| boss = Some((e, b.target, b.hp, b.t)));
            }

            let mut broken = false;
            if let Some((boss_e, target, hp, t)) = boss {
                // Gentle drift around the spell anchor.
                let wobble = Vec2::new((t * 0.7).sin() * 46.0, (t * 0.53).sin() * 22.0);
                if let Some(mut p) = world.get_mut::<Pos>(boss_e) {
                    p.0 = target + wobble;
                }
                if hp <= 0.0 {
                    broken = true;
                }
            }

            let timed_out = spell_timer + dt >= SPELL_DURATION;
            if broken || timed_out {
                let survived = !broken;
                harvest_enemy_shots(world, stage)?;
                if let Some((boss_e, _pos, _hp, _t)) = boss {
                    let at = world.get::<Pos>(boss_e).map(|p| p.0).unwrap_or_default();
                    let sidx = world.get::<Boss>(boss_e).map(|b| b.spell).unwrap_or(0);
                    world.despawn_batch(vec![boss_e])?;
                    burst_sparks(world, at, 40, color::hsl(0.12, 0.85, 0.7))?;
                    // A broken spell showers the arena with pickups.
                    drop_pickups(world, at, stage, !survived)?;
                    if let Some(mut state) = world.get_resource_mut::<GameState>() {
                        state.score += if survived { 3000 } else { 1000 };
                        state.shake = 6.0;
                    }
                    if sidx + 1 < SPELLS_PER_STAGE {
                        spawn_boss(world, sidx + 1, stage)?;
                        set_phase(world, Phase::BossSpell, 0.0);
                        if let Some(mut state) = world.get_resource_mut::<GameState>() {
                            state.spell_timer = 0.0;
                        }
                    } else {
                        set_phase(world, Phase::StageClear, 2.0);
                    }
                }
            } else {
                set_phase(world, Phase::BossSpell, 0.0);
                if let Some(mut state) = world.get_resource_mut::<GameState>() {
                    state.spell_timer += dt;
                }
            }
        }
        Phase::StageClear => {
            harvest_enemy_shots(world, stage)?;
            if timer - dt <= 0.0 {
                if let Some(mut state) = world.get_resource_mut::<GameState>() {
                    state.stage += 1;
                    state.wave_index = 0;
                }
                set_phase(world, Phase::Intro, 1.0);
            } else {
                set_phase(world, Phase::StageClear, timer - dt);
            }
        }
    }

    perf::record("director", start.elapsed().as_nanos() as u64);
    Ok(())
}

fn set_phase(world: &mut World, phase: Phase, timer: f32) {
    if let Some(mut state) = world.get_resource_mut::<GameState>() {
        state.phase = phase;
        state.phase_timer = timer;
    }
}

fn game_over_flag(world: &World) -> bool {
    world
        .get_resource::<GameState>()
        .map(|s| s.game_over)
        .unwrap_or(false)
}

/// Snapshot of the round state (defaults when absent).
fn state_of(world: &World) -> GameState {
    world
        .get_resource::<GameState>()
        .copied()
        .unwrap_or_default()
}

/// Spawns one fairy wave; the index picks the formation.
fn spawn_wave(world: &mut World, index: u32, stage: u32, bounds: SimBounds) -> Result<(), WorldError> {
    let mut rng = {
        let mut slot = world
            .get_resource_mut::<Rng>()
            .expect("bullet hell inserts the Rng resource");
        std::mem::take(&mut *slot)
    };
    let result = (|world: &mut World, rng: &mut Rng| {
        let count = 6 + rng.usize(3) + (stage / 3) as usize;
        let y = 90.0 + rng.range(0.0, 40.0);
        // Later stages: fairies die slower, fire faster and aim harder.
        let hp_scale = 2.0 + 0.4 * (stage - 1) as f32;
        let rate_scale = 1.0 / (1.0 + 0.12 * (stage - 1) as f32);
        let speed_scale = 1.0 + 0.07 * (stage - 1) as f32;
        // Every stage fields a different mix of fairy archetypes.
        let pool = stage_enemy_pool(stage);
        let player = world
            .get_resource::<PlayerPos>()
            .and_then(|p| p.0)
            .unwrap_or(Vec2::new(bounds.width * 0.5, bounds.height));
        let spawn_fairy = |world: &mut World, pos: Vec2, kind: FairyKind, rng: &mut Rng| {
            let (x0, y0, dir, omega, amp) = match kind {
                FairyKind::Sweeper => (
                    pos.x,
                    pos.y,
                    Vec2::new(120.0 * speed_scale, 26.0),
                    0.0,
                    0.0,
                ),
                FairyKind::Weaver => (
                    pos.x,
                    pos.y,
                    Vec2::new(0.0, 42.0),
                    rng.range(1.6, 2.4),
                    rng.range(50.0, 90.0),
                ),
                FairyKind::Diver => {
                    let dir = (player - pos).normalized() * (190.0 * speed_scale).max(190.0);
                    (pos.x, pos.y, dir, 0.0, 0.0)
                }
                FairyKind::Sniper => (
                    pos.x,
                    pos.y,
                    Vec2::new(rng.range(-40.0, 40.0), 70.0 * speed_scale),
                    0.0,
                    0.0,
                ),
                FairyKind::Orbiter => (
                    pos.x,
                    pos.y,
                    Vec2::new(0.0, 30.0),
                    rng.range(2.2, 3.2),
                    rng.range(40.0, 80.0),
                ),
            };
            // Kind base health: divers are glass, snipers are tough.
            let kind_hp = match kind {
                FairyKind::Diver => 1.0,
                FairyKind::Sniper => 3.0,
                _ => 2.0,
            };
            let kind_rate = match kind {
                FairyKind::Diver => 2.4,
                FairyKind::Sniper => 0.95,
                FairyKind::Orbiter => 1.3,
                FairyKind::Weaver => 1.6,
                FairyKind::Sweeper => 1.1,
            };
            let _ = world.spawn((
                Pos(pos),
                Fairy {
                    hp: hp_scale * kind_hp,
                    kind,
                    t: rng.angle(),
                    x0,
                    y0,
                    omega,
                    amp,
                    dir,
                    fire_cd: rng.range(0.3, 1.2),
                    fire_rate: kind_rate * rate_scale,
                    flash: 0.0,
                },
            ));
        };

        match index % 3 {
            0 => {
                // Line crossing the arena, kinds picked from the stage pool.
                for i in 0..count {
                    let t = i as f32 / (count - 1).max(1) as f32;
                    let x = 80.0 + t * (bounds.width - 160.0);
                    let kind = pool[i % pool.len()];
                    spawn_fairy(world, Vec2::new(x, y + 24.0 * (i % 2) as f32), kind, rng);
                }
            }
            1 => {
                // V formation descending, second pool kind.
                for i in 0..count {
                    let t = i as f32 / (count - 1).max(1) as f32;
                    let x = 140.0 + t * (bounds.width - 280.0);
                    let kind = pool[(i + 1) % pool.len()];
                    spawn_fairy(world, Vec2::new(x, y - 30.0), kind, rng);
                }
            }
            _ => {
                // Mixed column: a tanky core flanked by pool kinds.
                let mid = bounds.width * 0.5;
                spawn_fairy(world, Vec2::new(mid, y), pool[0], rng);
                for side in [-1.0f32, 1.0] {
                    for k in 1..=3usize {
                        let x = mid + side * 90.0 * k as f32;
                        let kind = pool[k % pool.len()];
                        spawn_fairy(world, Vec2::new(x, y - 20.0 * k as f32), kind, rng);
                    }
                }
            }
        }
        Ok(())
    })(world, &mut rng);
    if let Some(mut slot) = world.get_resource_mut::<Rng>() {
        *slot = rng;
    }
    result
}

/// Explosion sparks at `at`; `n` particles with a shared hue band.
fn burst_sparks(world: &mut World, at: Vec2, n: usize, hue_base: Rgba) -> Result<(), WorldError> {
    with_world_rng(world, |world, rng| {
        for _ in 0..n {
            let life = rng.range(0.4, 0.9);
            let _ = world.spawn((
                Pos(at),
                Vel(Vec2::from_angle(rng.angle()) * rng.range(60.0, 300.0)),
                Spark {
                    life,
                    max: life,
                    tint: color::scale_rgb(hue_base, rng.range(0.7, 1.15)),
                },
            ));
        }
        Ok(())
    })
}

/// Drops pickups at `at`: power items and score items fanned outward.
/// Big drops (boss breaks) carry more power.
fn drop_pickups(
    world: &mut World,
    at: Vec2,
    stage: u32,
    big: bool,
) -> Result<(), WorldError> {
    with_world_rng(world, |world, rng| {
        let powers = if big { 6 } else { 1 + rng.usize(2) };
        let scores = if big { 12 } else { 3 + rng.usize(3) };
        let score_value = 120 + 20 * stage;
        for _ in 0..powers {
            let _ = world.spawn((
                Pos(at + Vec2::new(rng.range(-14.0, 14.0), rng.range(-10.0, 10.0))),
                Vel(Vec2::from_angle(rng.angle()) * rng.range(70.0, 190.0)),
                Pickup {
                    t: rng.range(0.0, 6.0),
                    kind: PickupKind::Power,
                    value: 0,
                },
            ));
        }
        for _ in 0..scores {
            let _ = world.spawn((
                Pos(at + Vec2::new(rng.range(-18.0, 18.0), rng.range(-12.0, 12.0))),
                Vel(Vec2::from_angle(rng.angle()) * rng.range(60.0, 170.0)),
                Pickup {
                    t: rng.range(0.0, 6.0),
                    kind: PickupKind::Score,
                    value: score_value,
                },
            ));
        }
        Ok(())
    })
}

/// Pickup drift: falls gently, homes to the player when close, grants its
/// reward on contact.
fn pickup_act(
    mut q: Query<(&mut Pos, &mut Vel, &mut Pickup, Entity)>,
    player_pos: Res<PlayerPos>,
    mut state: ResMut<GameState>,
    time: Res<Time>,
    mut commands: Commands,
) {
    let start = Instant::now();
    let dt = time.dt;
    let Some(target) = player_pos.0 else {
        perf::record("pickups", start.elapsed().as_nanos() as u64);
        return;
    };
    q.for_each_mut(|(mut pos, mut vel, mut pickup, e)| {
        pickup.t += dt;
        let d = target - pos.0;
        let dist = d.length();
        if dist < 120.0 {
            // Magnet: stronger the closer it gets.
            vel.0 = d.normalized() * (150.0 + 340.0 * (1.0 - dist / 120.0));
        } else {
            vel.0.y += (45.0 - vel.0.y) * 1.4 * dt;
            vel.0.x *= 1.0 - 0.8 * dt;
        }
        pos.0 += vel.0 * dt;
        if dist < 15.0 {
            match pickup.kind {
                PickupKind::Power => {
                    state.power += 1;
                    state.score += 30;
                }
                PickupKind::Score => state.score += pickup.value,
            }
            commands.despawn(e);
        }
    });
    perf::record("pickups", start.elapsed().as_nanos() as u64);
}

/// Runs `f` with the world's `Rng` taken out and restored after (an exclusive
/// system cannot borrow the world and its resource at once).
fn with_world_rng<F>(world: &mut World, f: F) -> Result<(), WorldError>
where
    F: FnOnce(&mut World, &mut Rng) -> Result<(), WorldError>,
{
    let mut rng = {
        let mut slot = world
            .get_resource_mut::<Rng>()
            .expect("bullet hell inserts the Rng resource");
        std::mem::take(&mut *slot)
    };
    let result = f(world, &mut rng);
    if let Some(mut slot) = world.get_resource_mut::<Rng>() {
        *slot = rng;
    }
    result
}

fn game_over_check(world: &World) -> bool {
    world
        .get_resource::<GameState>()
        .map(|s| s.game_over)
        .unwrap_or(false)
}

/// Whether the player is holding a focus key (Space or Shift).
fn input_focus(world: &World) -> bool {
    world
        .get_resource::<InputState>()
        .map(|i| {
            i.held.contains(&KeyCode::Space)
                || i.held.contains(&KeyCode::ShiftLeft)
                || i.held.contains(&KeyCode::ShiftRight)
        })
        .unwrap_or(false)
}

/// Movement (WASD + focus), invulnerability window, and dual autofire.
fn player_control(
    mut q: Query<(&mut Pos, &mut Iframes), With<Player>>,
    state: Res<GameState>,
    input: Res<InputState>,
    mut pos_out: ResMut<PlayerPos>,
    mut commands: Commands,
    time: Res<Time>,
    bounds: Res<SimBounds>,
    mut fire: Local<FireTimer>,
) {
    let start = Instant::now();
    let dt = time.dt;
    // Focus: Space or Shift slows the ship for precise dodging.
    let focus = input.held.contains(&KeyCode::Space)
        || input.held.contains(&KeyCode::ShiftLeft)
        || input.held.contains(&KeyCode::ShiftRight);
    q.for_each_mut(|(mut pos, mut iframes)| {
        iframes.0 = (iframes.0 - dt).max(0.0);
        pos_out.0 = Some(pos.0);
        if state.game_over {
            return;
        }
        let mut dir = Vec2::ZERO;
        if input.held.contains(&KeyCode::KeyA) || input.held.contains(&KeyCode::ArrowLeft) {
            dir.x -= 1.0;
        }
        if input.held.contains(&KeyCode::KeyD) || input.held.contains(&KeyCode::ArrowRight) {
            dir.x += 1.0;
        }
        if input.held.contains(&KeyCode::KeyW) || input.held.contains(&KeyCode::ArrowUp) {
            dir.y -= 1.0;
        }
        if input.held.contains(&KeyCode::KeyS) || input.held.contains(&KeyCode::ArrowDown) {
            dir.y += 1.0;
        }
        if dir != Vec2::ZERO {
            let speed = PLAYER_SPEED * if focus { FOCUS_MULT } else { 1.0 };
            pos.0 += dir.normalized() * speed * dt;
            pos.0.x = pos.0.x.clamp(PLAYER_MARGIN, bounds.width - PLAYER_MARGIN);
            pos.0.y = pos.0.y.clamp(PLAYER_MARGIN, bounds.height - PLAYER_MARGIN);
        }
        // Autofire: the ship always shoots on its own cadence.
        fire.t -= dt;
        if fire.t <= 0.0 {
            fire.t = FIRE_CD;
            // Power levels widen the volley: 1..5 streams, outer ones fanned.
            let level = shot_level(state.power) as i32;
            for i in 0..level {
                let off = i as f32 - (level as f32 - 1.0) * 0.5;
                let angle = -std::f32::consts::FRAC_PI_2 + off * 0.05;
                let _ = commands.spawn((
                    Pos(pos.0 + Vec2::new(off * 9.0, -10.0)),
                    Vel(Vec2::from_angle(angle) * SHOT_SPEED),
                    Bullet {
                        radius: 2.5,
                        tint: color::rgb8(140, 235, 255),
                        grazed: false,
                    },
                    PlayerShot,
                ));
            }
        }
    });
    perf::record("player", start.elapsed().as_nanos() as u64);
}

/// Fairy movement and volleys for the five archetypes. Runs only during the
/// fairy act of the stage.
fn fairy_act(
    mut q: Query<(&mut Pos, &mut Fairy)>,
    player_pos: Res<PlayerPos>,
    state: Res<GameState>,
    values: Res<PanelValues>,
    time: Res<Time>,
    mut commands: Commands,
) {
    let start = Instant::now();
    // Fairy act: only during the fairy phase of the stage.
    if state.phase != Phase::Waves || state.game_over {
        return;
    }
    let dt = time.dt;
    let target: Option<Vec2> = player_pos.0;
    let diff = difficulty(&values);
    let sm = speed_mult(diff, state.stage);
    let hue = STAGE_HUES[(state.stage - 1) as usize % STAGE_HUES.len()];
    q.for_each_mut(|(mut pos, mut fairy)| {
        fairy.t += dt;
        fairy.flash = (fairy.flash - dt).max(0.0);
        // Per-kind movement.
        match fairy.kind {
            FairyKind::Sweeper | FairyKind::Diver => {
                pos.0 += fairy.dir * dt;
            }
            FairyKind::Weaver => {
                pos.0.x = fairy.x0 + (fairy.omega * fairy.t).sin() * fairy.amp;
                pos.0.y += fairy.dir.y * dt;
            }
            FairyKind::Sniper => {
                // Drift in, hold position to snipe, then dive away.
                if fairy.t < 1.2 {
                    pos.0 += fairy.dir * dt;
                } else if fairy.t > 4.0 {
                    pos.0.y += 170.0 * dt;
                }
            }
            FairyKind::Orbiter => {
                // Orbit a slowly descending anchor.
                fairy.y0 += fairy.dir.y * dt;
                pos.0.x = fairy.x0 + (fairy.omega * fairy.t).cos() * fairy.amp;
                pos.0.y = fairy.y0 + (fairy.omega * fairy.t).sin() * fairy.amp * 0.55;
            }
        }
        fairy.fire_cd -= dt;
        if fairy.fire_cd > 0.0 {
            return;
        }
        fairy.fire_cd = fairy.fire_rate;
        // Snipers only fire while holding position.
        if fairy.kind == FairyKind::Sniper && (fairy.t < 1.2 || fairy.t > 4.0) {
            return;
        }
        let tint = |shift: f32| color::hsl(hue + shift, 0.8, 0.64);
        match fairy.kind {
            FairyKind::Sweeper | FairyKind::Diver => {
                if let Some(t) = target {
                    let dir = (t - pos.0).normalized();
                    let speed = if fairy.kind == FairyKind::Diver { 175.0 } else { 130.0 };
                    let _ = commands.spawn((
                        Pos(pos.0),
                        Vel(dir * speed * sm),
                        Bullet {
                            radius: 3.0,
                            tint: tint(0.5),
                            grazed: false,
                        },
                        EnemyShot,
                    ));
                }
            }
            FairyKind::Weaver => {
                let ring = 8 + (count_extra(diff, state.stage) as u32);
                for i in 0..ring {
                    let angle =
                        fairy.t + (i as f32) * std::f32::consts::TAU / ring as f32;
                    let _ = commands.spawn((
                        Pos(pos.0),
                        Vel(Vec2::from_angle(angle) * 85.0 * sm),
                        Bullet {
                            radius: 3.0,
                            tint: tint(0.3),
                            grazed: false,
                        },
                        EnemyShot,
                    ));
                }
            }
            FairyKind::Sniper => {
                if let Some(t) = target {
                    let base = (t - pos.0).normalized().angle();
                    for spread in [-0.09f32, 0.0, 0.09] {
                        let _ = commands.spawn((
                            Pos(pos.0),
                            Vel(Vec2::from_angle(base + spread) * 205.0 * sm),
                            Bullet {
                                radius: 2.5,
                                tint: tint(0.45),
                                grazed: false,
                            },
                            EnemyShot,
                        ));
                    }
                }
            }
            FairyKind::Orbiter => {
                if let Some(t) = target {
                    let base = (t - pos.0).normalized().angle();
                    for spread in [-0.12f32, 0.12] {
                        let _ = commands.spawn((
                            Pos(pos.0),
                            Vel(Vec2::from_angle(base + spread) * 115.0 * sm),
                            Bullet {
                                radius: 3.0,
                                tint: tint(0.15),
                                grazed: false,
                            },
                            EnemyShot,
                        ));
                    }
                }
            }
        }
    });
    perf::record("fairies", start.elapsed().as_nanos() as u64);
}

/// Extra bullets above the base patterns, per difficulty and stage.
fn count_extra(diff: f32, stage: u32) -> f32 {
    (0.25 * (diff - 1.0) + 0.22 * (stage - 1) as f32).round()
}

/// One fused movement system for everything linear (bullets and sparks),
/// with an optional fetch for accelerating bullets.
fn move_all(
    mut q: Query<(&mut Pos, &mut Vel, Option<&mut Accel>)>,
    time: Res<Time>,
) {
    let start = Instant::now();
    let dt = time.dt;
    q.for_each_mut(|(mut pos, mut vel, accel)| {
        pos.0 += vel.0 * dt;
        if let Some(a) = accel {
            vel.0 *= 1.0 + a.0 * dt;
        }
    });
    perf::record("move", start.elapsed().as_nanos() as u64);
}

/// One enemy bullet requested by a spell pattern.
#[derive(Clone, Copy)]
pub struct ShotSpawn {
    /// Spawn position in pixels.
    pub pos: Vec2,
    /// Initial velocity in px/s.
    pub vel: Vec2,
    /// Optional velocity multiplier per second (accelerating bullets).
    pub accel: Option<f32>,
    /// Collision and draw radius.
    pub radius: f32,
    /// Bullet hue (stage-tinted).
    pub hue: f32,
}

/// Spell-card pattern context: everything a pattern needs to fire (mutable
/// boss timers, the firing anchor, the player aim, and the stage-scaled
/// multipliers. Patterns request shots through [`SpellCtx::shot`] into a
/// buffer the runner materializes after the pattern returns.
pub struct SpellCtx<'a> {
    shots: &'a mut Vec<ShotSpawn>,
    /// Boss anchor: patterns spawn their bullets relative to it.
    target: Vec2,
    /// Player position, if the ship is alive.
    player: Option<Vec2>,
    /// Speed multiplier (difficulty x stage).
    sm: f32,
    /// Extra bullets per pattern (difficulty x stage).
    extra: f32,
    /// Stage color identity for bullet tints.
    hue: f32,
    /// Arena width (rain walls span it).
    width: f32,
    t: &'a mut f32,
    volley: &'a mut f32,
    sub: &'a mut f32,
    angle: &'a mut f32,
    dir: &'a mut f32,
}

impl SpellCtx<'_> {
    /// Requests one enemy bullet tinted from the stage hue.
    fn shot(&mut self, pos: Vec2, vel: Vec2, radius: f32, hue: f32) {
        self.shots.push(ShotSpawn {
            pos,
            vel,
            accel: None,
            radius,
            hue: hue.rem_euclid(1.0),
        });
    }

    /// Requests one accelerating enemy bullet.
    fn accel_shot(&mut self, pos: Vec2, vel: Vec2, accel: f32, radius: f32, hue: f32) {
        self.shots.push(ShotSpawn {
            pos,
            vel,
            accel: Some(accel),
            radius,
            hue: hue.rem_euclid(1.0),
        });
    }

    /// Angle from the boss anchor toward the player (straight down if the
    /// ship is gone).
    fn aimed(&self) -> f32 {
        self.player
            .map(|p| (p - self.target).normalized().angle())
            .unwrap_or(std::f32::consts::FRAC_PI_2)
    }
}

// Pétalos: counter-rotating double rings.
fn pattern_rings(c: &mut SpellCtx) {
    if *c.volley <= 0.0 {
        *c.volley = 0.85;
        let n = ((18.0 * (1.0 + 0.2 * c.extra)).round() as u32).max(8);
        for i in 0..n {
            let a = *c.angle + i as f32 * std::f32::consts::TAU / n as f32;
            c.shot(c.target, Vec2::from_angle(a) * 95.0 * c.sm, 3.5, c.hue);
        }
        for i in 0..n {
            let a = -*c.angle * 0.7 + i as f32 * std::f32::consts::TAU / n as f32 + 0.13;
            c.shot(c.target, Vec2::from_angle(a) * 72.0 * c.sm, 3.0, c.hue + 0.5);
        }
        *c.angle += 0.31;
    }
}

// Abanico: aimed fan sweeping left and right, with a lazy backup ring.
fn pattern_fan(c: &mut SpellCtx) {
    if *c.volley <= 0.0 {
        *c.volley = 1.1;
        let base = c.aimed();
        let sweep = (*c.t * 1.1).sin() * 0.55;
        let arms = ((7.0 + 2.0 * c.extra).round() as u32).max(3);
        for i in 0..arms {
            let f = if arms == 1 {
                0.0
            } else {
                i as f32 / (arms - 1) as f32 - 0.5
            };
            let a = base + sweep + f * 1.1;
            c.shot(c.target, Vec2::from_angle(a) * 135.0 * c.sm, 3.5, c.hue + 0.21);
        }
    }
    if *c.sub <= 0.0 {
        *c.sub = 0.55;
        let n = 6;
        for i in 0..n {
            let a = *c.angle + i as f32 * std::f32::consts::TAU / n as f32;
            c.shot(c.target, Vec2::from_angle(a) * 60.0 * c.sm, 2.5, c.hue - 0.1);
        }
        *c.angle += 0.41;
    }
}

// Serpiente: continuous double spiral that reverses direction periodically.
fn pattern_spiral(c: &mut SpellCtx) {
    if *c.sub <= 0.0 {
        *c.sub = 0.05;
        *c.angle += 0.19 * *c.dir;
        for side in [0.0f32, std::f32::consts::PI] {
            let hue = if *c.dir > 0.0 {
                c.hue + 0.16
            } else {
                c.hue - 0.45
            };
            c.shot(c.target, Vec2::from_angle(*c.angle + side) * 112.0 * c.sm, 3.0, hue);
        }
    }
    if *c.volley <= 0.0 {
        *c.volley = 2.6;
        *c.dir = -*c.dir;
    }
}

// Estrellas: aimed accelerating star bursts over a slow rotating ring.
fn pattern_stars(c: &mut SpellCtx) {
    if *c.volley <= 0.0 {
        *c.volley = 1.6;
        let base = c.aimed();
        let arms = 5u32;
        for arm in 0..arms {
            let a = base + arm as f32 * std::f32::consts::TAU / arms as f32;
            for spread in [-0.14f32, 0.0, 0.14] {
                c.accel_shot(
                    c.target,
                    Vec2::from_angle(a + spread) * 70.0 * c.sm,
                    1.9,
                    3.5,
                    c.hue + 0.37,
                );
            }
        }
    }
    if *c.sub <= 0.0 {
        *c.sub = 0.8;
        let n = 12;
        for i in 0..n {
            let a = *c.angle + i as f32 * std::f32::consts::TAU / n as f32;
            c.shot(c.target, Vec2::from_angle(a) * 68.0 * c.sm, 2.5, c.hue - 0.7);
        }
        *c.angle += 0.23;
    }
}

// Burbujas: big slow bubbles drifting down plus an aimed shot.
fn pattern_bubbles(c: &mut SpellCtx) {
    if *c.volley <= 0.0 {
        *c.volley = 0.7;
        let wobble = (*c.t * 2.7).sin() * 55.0;
        c.shot(
            c.target + Vec2::new(wobble, 12.0),
            Vec2::new(wobble * 0.3, 46.0 * c.sm),
            6.0,
            c.hue,
        );
    }
    if *c.sub <= 0.0 {
        *c.sub = 1.1;
        let a = c.aimed();
        c.shot(c.target, Vec2::from_angle(a) * 150.0 * c.sm, 3.0, c.hue + 0.45);
    }
}

// Lluvia: horizontal walls of bullets with a shifting gap.
fn pattern_rain(c: &mut SpellCtx) {
    if *c.volley <= 0.0 {
        *c.volley = 1.0;
        let span = 700.0f32.min(c.width - 120.0).max(240.0);
        let left = c.width * 0.5 - span * 0.5;
        let gap = c.width * 0.5 + (*c.t * 0.8).sin() * span * 0.35;
        let step = 44.0f32.max(span / 14.0);
        let mut x = left;
        while x <= left + span {
            if (x - gap).abs() > 78.0 {
                c.shot(
                    Vec2::new(x, c.target.y + 26.0),
                    Vec2::new(0.0, 95.0 * c.sm),
                    3.0,
                    c.hue - 0.05,
                );
            }
            x += step;
        }
        *c.angle += 0.5;
    }
}

// Cruz: four rotating arms streaming bullets, with an aimed ring burst.
fn pattern_cross(c: &mut SpellCtx) {
    if *c.sub <= 0.0 {
        *c.sub = 0.09;
        *c.angle += 0.11;
        for arm in 0..4u32 {
            let a = *c.angle + arm as f32 * std::f32::consts::TAU / 4.0;
            c.shot(c.target, Vec2::from_angle(a) * 120.0 * c.sm, 3.0, c.hue);
        }
    }
    if *c.volley <= 0.0 {
        *c.volley = 2.2;
        let base = c.aimed();
        let n = 10;
        for i in 0..n {
            let a = base + i as f32 * std::f32::consts::TAU / n as f32;
            c.shot(c.target, Vec2::from_angle(a) * 100.0 * c.sm, 3.2, c.hue + 0.35);
        }
    }
}

// Flor: rings of petal bullets with layered speeds that bloom outward.
fn pattern_flower(c: &mut SpellCtx) {
    if *c.volley <= 0.0 {
        *c.volley = 1.5;
        let n = ((12.0 + 4.0 * c.extra).round() as u32).max(10);
        for i in 0..n {
            let a = *c.angle + i as f32 * std::f32::consts::TAU / n as f32;
            let speed = (62.0 + 26.0 * (i % 3) as f32) * c.sm;
            c.shot(
                c.target + Vec2::from_angle(a) * 18.0,
                Vec2::from_angle(a) * speed,
                3.2,
                c.hue + 0.02 * i as f32,
            );
        }
        *c.angle += 0.17;
    }
}

/// The boss pattern runner: looks up the current stage's spell table and
/// runs the active card's pattern against the shared context. Runs only
/// while a boss spell is active.
fn boss_pattern(
    mut q: Query<&mut Boss>,
    player_pos: Res<PlayerPos>,
    state: Res<GameState>,
    values: Res<PanelValues>,
    bounds: Res<SimBounds>,
    time: Res<Time>,
    mut commands: Commands,
) {
    let start = Instant::now();
    // Patterns only while a boss spell is running.
    if state.phase != Phase::BossSpell || state.game_over {
        return;
    }
    let dt = time.dt;
    let player = player_pos.0;
    let diff = difficulty(&values);
    let sm = speed_mult(diff, state.stage);
    let extra = count_extra(diff, state.stage);
    let hue = STAGE_HUES[(state.stage - 1) as usize % STAGE_HUES.len()];

    q.for_each_mut(|mut boss| {
        boss.t += dt;
        boss.volley -= dt;
        boss.sub -= dt;
        let target = boss.target;
        let def = spell_def(state.stage, boss.spell);
        // One DerefMut, five disjoint field borrows.
        let Boss {
            t,
            volley,
            sub,
            angle,
            dir,
            ..
        } = &mut *boss;
        let mut requested: Vec<ShotSpawn> = Vec::new();
        let mut ctx = SpellCtx {
            shots: &mut requested,
            target,
            player,
            sm,
            extra,
            hue,
            width: bounds.width,
            t,
            volley,
            sub,
            angle,
            dir,
        };
        (def.run)(&mut ctx);
        drop(ctx);
        // Materialize the pattern's bullets.
        for shot in requested {
            let tint = color::hsl(shot.hue, 0.8, 0.66);
            match shot.accel {
                Some(accel) => {
                    let _ = commands.spawn((
                        Pos(shot.pos),
                        Vel(shot.vel),
                        Accel(accel),
                        Bullet {
                            radius: shot.radius,
                            tint,
                            grazed: false,
                        },
                        EnemyShot,
                    ));
                }
                None => {
                    let _ = commands.spawn((
                        Pos(shot.pos),
                        Vel(shot.vel),
                        Bullet {
                            radius: shot.radius,
                            tint,
                            grazed: false,
                        },
                        EnemyShot,
                    ));
                }
            }
        }
    });
    perf::record("pattern", start.elapsed().as_nanos() as u64);
}

/// Despawns everything that left the arena in one batch.
fn cull(
    mut commands: Commands,
    bounds: Res<SimBounds>,
    mut bullets: Query<(Entity, &Pos, &Bullet)>,
    mut fairies: Query<(Entity, &Pos), With<Fairy>>,
    mut sparks: Query<(Entity, &Pos, &Spark)>,
    mut pickups: Query<(Entity, &Pos), With<Pickup>>,
) {
    let start = Instant::now();
    let m = 50.0;
    let outside = |p: &Pos| {
        p.0.x < -m || p.0.x > bounds.width + m || p.0.y < -m || p.0.y > bounds.height + m
    };
    let mut doomed = Vec::new();
    bullets.for_each(|(e, p, _)| {
        if outside(p) {
            doomed.push(e);
        }
    });
    fairies.for_each(|(e, p)| {
        if p.0.y > bounds.height + 80.0 || p.0.y < -140.0 {
            doomed.push(e);
        }
    });
    sparks.for_each(|(e, p, _)| {
        if outside(p) {
            doomed.push(e);
        }
    });
    pickups.for_each(|(e, p)| {
        if outside(p) {
            doomed.push(e);
        }
    });
    if !doomed.is_empty() {
        commands.despawn_batch(doomed);
    }
    perf::record("cull", start.elapsed().as_nanos() as u64);
}

/// Exclusive system owning the world: player shots vs fairies and boss,
/// enemy shots vs player (lives + screen clear), and graze scoring.
fn collisions(world: &mut World) -> Result<(), WorldError> {
    let start = Instant::now();
    let game_over = game_over_check(world);

    let mut player: Option<(Entity, Vec2)> = None;
    {
        let mut q = world.query::<(Entity, &Pos), With<Player>>()?;
        q.for_each(|(e, p)| player = Some((e, p.0)));
    }

    if let Some((player_e, player_pos)) = player {
        let iframes = world.get::<Iframes>(player_e).map(|i| i.0).unwrap_or(0.0);

        // Graze: near misses score once per bullet.
        {
            let mut grazed_ids: Vec<Entity> = Vec::new();
            {
                let mut q = world.query::<(Entity, &Pos, &Bullet), With<EnemyShot>>()?;
                q.for_each(|(e, p, b)| {
                    let d = p.0.distance(player_pos);
                    if !b.grazed && d <= b.radius + HITBOX + GRAZE_BAND && d > b.radius + HITBOX {
                        grazed_ids.push(e);
                    }
                });
            }
            for e in grazed_ids {
                if let Some(mut b) = world.get_mut::<Bullet>(e) {
                    if !b.grazed {
                        b.grazed = true;
                        if let Some(mut state) = world.get_resource_mut::<GameState>() {
                            state.graze += 1;
                            state.score += 12;
                        }
                    }
                }
            }
        }

        // Enemy shots vs player: a hit costs a life and clears the screen.
        if !game_over && iframes <= 0.0 {
            let mut hit_shots = Vec::new();
            {
                let mut q = world.query::<(Entity, &Pos, &Bullet), With<EnemyShot>>()?;
                q.for_each(|(e, p, b)| {
                    if p.0.distance(player_pos) <= b.radius + HITBOX + HIT_PAD {
                        hit_shots.push(e);
                    }
                });
            }
            if !hit_shots.is_empty() {
                clear_enemy_shots(world)?;
                burst_sparks(world, player_pos, 30, color::hsl(0.99, 0.9, 0.65))?;
                if let Some(mut state) = world.get_resource_mut::<GameState>() {
                    state.lives = state.lives.saturating_sub(1);
                    state.hit_flash = 0.4;
                    state.shake = 7.0;
                    // Dying costs power, like every self-respecting bullet hell.
                    state.power = state.power.saturating_sub(6);
                    if state.lives == 0 {
                        state.game_over = true;
                    }
                }
                if let Some(mut i) = world.get_mut::<Iframes>(player_e) {
                    i.0 = HIT_IFRAMES;
                }
            }
        }

        // Player shots vs fairies and the boss.
        if !game_over {
            let mut fairies: Vec<(Entity, Vec2, f32)> = Vec::new();
            {
                let mut q = world.query::<(Entity, &Pos, &Fairy), Without<Player>>()?;
                q.for_each(|(e, p, _f)| fairies.push((e, p.0, 10.0)));
            }
            let mut boss: Option<(Entity, Vec2)> = None;
            {
                let mut q = world.query::<(Entity, &Pos), With<Boss>>()?;
                q.for_each(|(e, p)| boss = Some((e, p.0)));
            }
            let mut shots: Vec<(Entity, Vec2)> = Vec::new();
            {
                let mut q = world.query::<(Entity, &Pos), With<PlayerShot>>()?;
                q.for_each(|(e, p)| shots.push((e, p.0)));
            }

            let mut used = vec![false; shots.len()];
            let mut hit_shots = Vec::new();
            let mut fairy_damage: Vec<(Entity, u32)> = Vec::new();
            let mut boss_hits = 0u32;

            for (fairy_e, fairy_pos, fairy_r) in &fairies {
                let mut dmg = 0u32;
                for (i, (shot_e, shot_pos)) in shots.iter().enumerate() {
                    if used[i] {
                        continue;
                    }
                    if shot_pos.distance(*fairy_pos) <= fairy_r + 2.5 + HIT_PAD {
                        used[i] = true;
                        hit_shots.push(*shot_e);
                        dmg += 1;
                    }
                }
                if dmg > 0 {
                    fairy_damage.push((*fairy_e, dmg));
                }
            }
            if let Some((boss_e, boss_pos)) = boss {
                for (i, (shot_e, shot_pos)) in shots.iter().enumerate() {
                    if !used[i] && shot_pos.distance(boss_pos) <= 26.0 + 2.5 + HIT_PAD {
                        used[i] = true;
                        hit_shots.push(*shot_e);
                        boss_hits += 1;
                    }
                }
                if boss_hits > 0 {
                    if let Some(mut b) = world.get_mut::<Boss>(boss_e) {
                        b.hp -= boss_hits as f32;
                        b.flash = 0.12;
                    }
                }
            }

            let mut dead = Vec::new();
            for (e, dmg) in fairy_damage {
                let mut died = false;
                if let Some(mut fairy) = world.get_mut::<Fairy>(e) {
                    fairy.hp -= dmg as f32;
                    fairy.flash = 0.15;
                    if fairy.hp <= 0.0 {
                        died = true;
                    }
                }
                if died {
                    dead.push(e);
                }
            }

            if !hit_shots.is_empty() {
                world.despawn_batch(hit_shots)?;
            }
            if !dead.is_empty() {
                let mut centers = Vec::new();
                for e in &dead {
                    if let Some(p) = world.get::<Pos>(*e) {
                        centers.push(p.0);
                    }
                }
                world.despawn_batch(dead)?;
                let stage = world
                    .get_resource::<GameState>()
                    .map(|s| s.stage)
                    .unwrap_or(1);
                for center in &centers {
                    burst_sparks(world, *center, 18, color::hsl(0.9, 0.8, 0.7))?;
                    drop_pickups(world, *center, stage, false)?;
                }
                if let Some(mut state) = world.get_resource_mut::<GameState>() {
                    state.score += 100 * centers.len() as u32;
                }
            }
        }
    }

    perf::record("collide", start.elapsed().as_nanos() as u64);
    Ok(())
}

/// Fades and despawns explosion sparks.
fn spark_aging(mut commands: Commands, mut q: Query<(Entity, &mut Spark)>, time: Res<Time>) {
    let start = Instant::now();
    let dt = time.dt;
    let mut doomed = Vec::new();
    q.for_each_mut(|(e, mut spark)| {
        spark.life -= dt;
        if spark.life <= 0.0 {
            doomed.push(e);
        }
    });
    if !doomed.is_empty() {
        commands.despawn_batch(doomed);
    }
    perf::record("sparks", start.elapsed().as_nanos() as u64);
}

/// State of the bullet-hell demo; the HUD counters are read back in `draw`.
pub struct BulletHell {
    score: u32,
    lives: u32,
    bombs: u32,
    power: u32,
    graze: u32,
    game_over: bool,
    stage: u32,
    phase: Phase,
    spell: usize,
    spell_name: &'static str,
    spell_left: f32,
    boss_hp: f32,
    boss_max: f32,
    bullets: usize,
    /// Static starfield: (position, size, drift speed).
    stars: Vec<(Vec2, f32, f32)>,
}

impl Demo for BulletHell {
    fn title(&self) -> &'static str {
        "millarecs - bullet hell - WASD move - SHIFT/SPACE focus - X bomb - R menu"
    }

    fn usage() -> String {
        String::from(
            "bullet hell: level select + 4 spell cards per boss\n  --difficulty N  1 easy .. 5 extreme (default 2; also live in the panel)",
        )
    }

    fn params(&self) -> Vec<ParamSpec> {
        vec![ParamSpec::new("difficulty", "DIFFICULTY", 2.0, 1.0, 5.0, 1.0).integer()]
    }

    fn new(
        world: &mut World,
        schedule: &mut Schedule,
        _executor: &mut ScheduleExecutor,
        args: &Args,
    ) -> Self {
        for register in register_fns() {
            register(world);
        }
        world.insert_resource(GameState::default());
        world.insert_resource(Rng::default());
        world.insert_resource(InputState::default());
        world.insert_resource(PlayerPos(None));
        world.insert_resource(PanelValues::default());

        schedule
            .add_system(exclusive_system(director))
            .label::<Director>();
        schedule
            .add_system(player_control)
            .label::<PlayerControl>()
            .after::<Director>();
        schedule
            .add_system(fairy_act)
            .label::<FairyAct>()
            .after::<Director>()
            .after::<PlayerControl>();
        schedule
            .add_system(boss_pattern)
            .label::<Pattern>()
            .after::<Director>()
            .after::<PlayerControl>()
            .after::<FairyAct>();
        schedule.add_deferred_barrier();
        schedule
            .add_system(move_all)
            .label::<MoveAll>()
            .after::<Director>()
            .after::<FairyAct>()
            .after::<Pattern>();
        schedule
            .add_system(cull)
            .label::<Cull>()
            .after::<Director>()
            .after::<MoveAll>();
        schedule.add_deferred_barrier();
        schedule
            .add_system(exclusive_system(collisions))
            .label::<Collide>()
            .after::<Director>()
            .after::<Cull>();
        schedule.add_deferred_barrier();
        schedule
            .add_system(spark_aging)
            .after::<Director>()
            .after::<Collide>();
        schedule
            .add_system(pickup_act)
            .after::<Director>()
            .after::<PlayerControl>()
            .after::<Collide>();

        // Initial panel values from flags (the panel takes over afterwards).
        if let Some(mut pv) = world.get_resource_mut::<PanelValues>() {
            *pv = PanelValues {
                values: PanelValues::init(&Self::params_static(), args).values,
            };
        }

        // No player yet: the ship only exists once a stage starts (spawned by
        // the menu confirm), so the picker shows a clean screen.

        // Static starfield: two depth layers drifting downward.
        let bounds = *world
            .get_resource::<SimBounds>()
            .expect("runner inserts SimBounds");
        let mut star_rng = Rng::new(0xC0FF_EE11);
        let stars = (0..150)
            .map(|_| {
                (
                    Vec2::new(star_rng.range(0.0, bounds.width), star_rng.range(0.0, bounds.height)),
                    star_rng.range(1.0, 2.6),
                    star_rng.range(9.0, 34.0),
                )
            })
            .collect();

        BulletHell {
            score: 0,
            lives: 3,
            bombs: 2,
            power: 0,
            graze: 0,
            game_over: false,
            stage: 1,
            phase: Phase::Menu,
            spell: 0,
            spell_name: STAGE_SPELLS[0][0].name,
            spell_left: SPELL_DURATION,
            boss_hp: 0.0,
            boss_max: 0.0,
            bullets: 0,
            stars,
        }
    }

    fn frame(
        &mut self,
        _ctx: &FrameCtx,
        world: &mut World,
        schedule: &mut Schedule,
        executor: &mut ScheduleExecutor,
    ) {
        schedule.run(world, executor).expect("bullet hell runs");
    }

    fn draw(&mut self, world: &World, _atlas: &FontAtlas, out: &mut DrawList) {
        let state = state_of(world);
        self.score = state.score;
        self.lives = state.lives;
        self.bombs = state.bombs;
        self.power = state.power;
        self.graze = state.graze;
        self.game_over = state.game_over;
        self.stage = state.stage;
        self.phase = state.phase;
        self.spell_left = (SPELL_DURATION - state.spell_timer).max(0.0);
        let bounds = world
            .get_resource::<SimBounds>()
            .copied()
            .unwrap_or(SimBounds {
                width: 1280.0,
                height: 720.0,
            });

        // Starfield: two visual layers drifting downward.
        let dt = world.get_resource::<Time>().map(|t| t.dt).unwrap_or(0.0);
        for (pos, size, speed) in &mut self.stars {
            pos.y += *speed * dt;
            if pos.y > bounds.height + 4.0 {
                pos.y = -4.0;
                pos.x = (pos.x + 173.0) % bounds.width;
            }
            let dim = 0.16 + 0.22 * (*speed / 34.0);
            out.rect(*pos, Vec2::splat(*size), [0.62, 0.68, 0.85, dim]);
        }

        // Boss + spell data.
        self.spell = 0;
        self.boss_hp = 0.0;
        self.boss_max = 0.0;
        let mut boss_data: Option<(Vec2, f32, f32, f32)> = None;
        {
            let mut q = world.query_all::<(&Pos, &Boss)>().expect("query boss");
            q.for_each(|(p, b)| {
                self.spell = b.spell;
                self.spell_name = spell_def(state.stage, b.spell).name;
                self.boss_hp = b.hp.max(0.0);
                self.boss_max = b.max_hp;
                boss_data = Some((p.0, b.t, b.spell as f32, b.flash));
            });
        }

        // Fairies, with hit-flash whitening.
        {
            let mut q = world.query_all::<(&Pos, &Fairy)>().expect("query fairies");
            q.for_each(|(p, f)| {
                let base = if f.kind == FairyKind::Sweeper {
                    color::hsl(0.9, 0.55, 0.68)
                } else {
                    color::hsl(0.45, 0.55, 0.68)
                };
                let tint = if f.flash > 0.0 {
                    color::lerp(base, [1.0, 1.0, 1.0, 1.0], (f.flash / 0.15).clamp(0.0, 1.0) * 0.9)
                } else {
                    base
                };
                out.circle(p.0, 10.0, tint);
                out.circle(p.0, 4.0, color::rgb8(250, 250, 250));
            });
        }

        // Boss: concentric body, orbiting ornaments, spell aura, health bar.
        if let Some((p, t, s, flash)) = boss_data {
            let aura_tint = color::hsl(0.78 - 0.06 * s, 0.6, 0.55);
            out.circle(p, 26.0 + (t * 2.2).sin() * 2.0, aura_tint);
            let body = if flash > 0.0 {
                color::lerp(
                    color::hsl(0.78, 0.35, 0.72),
                    [1.0, 1.0, 1.0, 1.0],
                    (flash / 0.12).clamp(0.0, 1.0) * 0.85,
                )
            } else {
                color::hsl(0.78, 0.35, 0.72)
            };
            out.circle(p, 17.0, body);
            out.circle(p, 8.0, color::rgb8(250, 250, 255));
            for i in 0..6 {
                let a = t * 1.4 + s + i as f32 * std::f32::consts::TAU / 6.0;
                out.circle(p + Vec2::from_angle(a) * 36.0, 4.0, color::hsl(0.78, 0.7, 0.78));
            }
            // Spell health bar: dark frame, gray drained portion, and a
            // bright current segment that shifts from green to red as it
            // drains. Each spell refills it (fresh boss per spell).
            let frac = if self.boss_max > 0.0 {
                (self.boss_hp / self.boss_max).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let w = (bounds.width * 0.5).min(460.0);
            let top = 32.0;
            let cx = bounds.width * 0.5;
            let half_h = 5.0;
            // Dark backing frame with a thin light outline.
            out.rect(
                Vec2::new(cx, top),
                Vec2::new(w * 0.5 + 4.0, half_h + 3.0),
                [0.03, 0.04, 0.07, 0.92],
            );
            out.rect(
                Vec2::new(cx, top),
                Vec2::new(w * 0.5 + 2.5, half_h + 2.0),
                [0.85, 0.86, 0.92, 0.3],
            );
            // Drained (gray) full-width track, anchored left.
            out.rect(
                Vec2::new(cx - w * 0.0, top),
                Vec2::new(w * 0.5, half_h),
                [0.16, 0.17, 0.2, 1.0],
            );
            // Current health: bright segment anchored left, hue green to red.
            if frac > 0.0 {
                let hue = 0.33 * frac;
                out.rect(
                    Vec2::new(cx - w * 0.5 + (w * frac) * 0.5, top),
                    Vec2::new(w * frac * 0.5, half_h),
                    color::hsl(hue, 0.9, 0.58),
                );
                // Gloss line on top of the filled part.
                out.rect(
                    Vec2::new(cx - w * 0.5 + (w * frac) * 0.5, top - 1.5),
                    Vec2::new(w * frac * 0.5, 1.0),
                    color::with_alpha(color::rgb8(255, 255, 255), 0.4),
                );
            }
            // Spell counter at the right end of the bar.
            let label = format!(
                "{}/{}",
                (self.spell + 1).min(SPELLS_PER_STAGE),
                SPELLS_PER_STAGE
            );
            out.text(
                _atlas,
                Vec2::new(cx + w * 0.5 + 12.0, top - 6.0),
                &label,
                2.0,
                color::rgb8(230, 200, 160),
            );
        }

        // Bullets: enemy shots get a soft glow halo, all get white cores.
        self.bullets = 0;
        {
            let mut q = world
                .query::<(&Pos, &Bullet), With<EnemyShot>>()
                .expect("query bullets");
            q.for_each(|(p, b)| {
                self.bullets += 1;
                out.circle(p.0, b.radius * 2.1, color::with_alpha(b.tint, 0.28));
                out.circle(p.0, b.radius, b.tint);
                out.circle(p.0, b.radius * 0.45, color::rgb8(255, 255, 255));
            });
        }
        {
            let mut q = world
                .query::<(&Pos, &Bullet), With<PlayerShot>>()
                .expect("query player shots");
            q.for_each(|(p, b)| {
                self.bullets += 1;
                out.rect(p.0, Vec2::new(1.6, 5.0), b.tint);
            });
        }

        // Sparks.
        {
            let mut q = world.query_all::<(&Pos, &Spark)>().expect("query sparks");
            q.for_each(|(p, s)| {
                let frac = (s.life / s.max).clamp(0.0, 1.0);
                out.circle(p.0, 1.5 + 3.0 * (1.0 - frac), color::with_alpha(s.tint, frac));
            });
        }

        // Pickups: spinning diamonds, power = pink, score = gold.
        {
            let mut q = world.query_all::<(&Pos, &Pickup)>().expect("query pickups");
            q.for_each(|(p, k)| {
                let spin = k.t * 2.6 + p.0.x * 0.01;
                let pulse = 5.0 + (k.t * 5.0).sin() * 0.8;
                let tint = if k.kind == PickupKind::Power {
                    color::hsl(0.97, 0.85, 0.66)
                } else {
                    color::hsl(0.14, 0.85, 0.64)
                };
                out.rot_rect(p.0, Vec2::splat(pulse), spin, tint);
                out.rot_rect(p.0, Vec2::splat(pulse * 0.45), spin, color::rgb8(255, 255, 245));
            });
        }

        // Player: body triangle (visual only) + the tiny damage core
        // (small circle), engine flame, hitbox ring when focused, blink on
        // invulnerability.
        let focus = input_focus(world);
        if state.phase != Phase::Menu {
            let mut q = world
                .query::<(&Pos, &Iframes), With<Player>>()
                .expect("query player");
            q.for_each(|(p, i)| {
                let blink = i.0 > 0.0 && ((state.game_time * 14.0) as i32) % 2 == 0;
                if blink {
                    return;
                }
                let flicker = 0.55 + 0.45 * (state.game_time * 31.0).sin().abs();
                out.triangle(
                    p.0 + Vec2::new(0.0, 13.0),
                    Vec2::new(3.0, 4.0 + 2.5 * flicker),
                    std::f32::consts::FRAC_PI_2,
                    color::with_alpha(color::hsl(0.09, 0.9, 0.62), flicker),
                );
                let tint = if i.0 > 0.0 {
                    color::with_alpha(color::rgb8(240, 240, 250), 0.75)
                } else {
                    color::rgb8(240, 240, 250)
                };
                out.triangle(p.0, Vec2::new(9.0, 14.0), -std::f32::consts::FRAC_PI_2, tint);
                // The damage core: subtle when unfocused, bright ring when
                // focused.
                out.circle(p.0, HITBOX, color::with_alpha(color::rgb8(255, 90, 120), 0.5));
                out.circle(p.0, HITBOX * 0.5, color::rgb8(255, 230, 240));
                if focus {
                    out.circle(p.0, HITBOX + 4.5, color::with_alpha(color::rgb8(255, 90, 120), 0.4));
                    out.circle(p.0, HITBOX + 2.0, color::rgb8(255, 120, 150));
                    out.circle(p.0, HITBOX, color::rgb8(255, 235, 245));
                }
            });
        }

        // Damage overlay: red vignette flash when the player is hit.
        if state.hit_flash > 0.0 {
            let a = (state.hit_flash / 0.4).clamp(0.0, 1.0) * 0.32;
            out.rect(
                bounds.center(),
                Vec2::new(bounds.width * 0.5, bounds.height * 0.5),
                [0.85, 0.08, 0.1, a],
            );
        }

        // Spell card announcement: big centered text fading in and out, with
        // the boss title above it.
        if state.phase == Phase::BossSpell {
            if let Some((_p, t, _s, _f)) = boss_data {
                if t < 2.5 {
                    let a = (t * 3.0).min(1.0) * ((2.5 - t) * 1.2).min(1.0);
                    let boss_name = BOSS_NAMES[(state.stage - 1) as usize % BOSS_NAMES.len()];
                    let spell_label = format!("SPELL: {}", self.spell_name.to_uppercase());
                    // Centered: scale 2 = 12 px per glyph cell, scale 3 = 18.
                    out.text(
                        _atlas,
                        Vec2::new(
                            bounds.width * 0.5 - (boss_name.len() as f32) * 12.0 * 0.5,
                            bounds.height * 0.3 - 34.0,
                        ),
                        boss_name,
                        2.0,
                        color::with_alpha(color::rgb8(230, 180, 250), a),
                    );
                    out.text(
                        _atlas,
                        Vec2::new(
                            bounds.width * 0.5 - (spell_label.len() as f32) * 18.0 * 0.5,
                            bounds.height * 0.3,
                        ),
                        &spell_label,
                        3.0,
                        color::with_alpha(color::rgb8(255, 205, 130), a),
                    );
                }
            }
        }

        // Level announcement during the intro breather.
        if state.phase == Phase::Intro && state.phase_timer > 0.25 {
            let a = ((state.phase_timer - 0.25) * 1.4).clamp(0.0, 1.0);
            let label = format!("LEVEL {}", state.stage);
            out.text(
                _atlas,
                Vec2::new(
                    bounds.width * 0.5 - (label.len() as f32) * 18.0 * 0.5,
                    bounds.height * 0.36,
                ),
                &label,
                3.0,
                color::with_alpha(color::rgb8(220, 225, 250), a),
            );
        }

        // Level menu: title, picker and hints. Panel and telemetry hide.
        if state.phase == Phase::Menu {
            let title = "MILLARECS BULLET HELL";
            out.text(
                _atlas,
                Vec2::new(bounds.width * 0.5 - (title.len() as f32) * 30.0 * 0.5, 90.0),
                title,
                5.0,
                color::rgb8(255, 120, 140),
            );
            out.text(
                _atlas,
                Vec2::new(bounds.width * 0.5 - 78.0, 150.0),
                "SELECT LEVEL",
                2.0,
                color::rgb8(200, 205, 230),
            );
            for i in 0..STAGE_COUNT as usize {
                let y = 210.0 + i as f32 * 52.0;
                let selected = state.menu_index as usize == i;
                let tint = if selected {
                    color::rgb8(255, 235, 160)
                } else {
                    color::rgb8(140, 145, 175)
                };
                if selected {
                    out.triangle(
                        Vec2::new(bounds.width * 0.5 - 210.0, y + 9.0),
                        Vec2::new(10.0, 7.0),
                        0.0,
                        color::rgb8(255, 210, 120),
                    );
                }
                out.text(
                    _atlas,
                    Vec2::new(bounds.width * 0.5 - 150.0, y),
                    &format!("LEVEL {}", i + 1),
                    2.0,
                    tint,
                );
                for d in 0..(i + 1) {
                    out.rot_rect(
                        Vec2::new(bounds.width * 0.5 + 190.0 + d as f32 * 16.0, y + 8.0),
                        Vec2::splat(4.0),
                        std::f32::consts::FRAC_PI_4,
                        color::hsl(0.98 - 0.02 * d as f32, 0.8, 0.62),
                    );
                }
            }
            out.text(
                _atlas,
                Vec2::new(bounds.width * 0.5 - 138.0, bounds.height - 70.0),
                "W/S OR ARROWS: SELECT",
                1.5,
                color::rgb8(160, 165, 195),
            );
            out.text(
                _atlas,
                Vec2::new(bounds.width * 0.5 - 108.0, bounds.height - 44.0),
                "SPACE / ENTER: PLAY",
                1.5,
                color::rgb8(220, 220, 240),
            );
        }

        // Screen shake: offset every world sprite (HUD glyphs are added by
        // the runner afterwards, so they stay put).
        if state.shake > 0.01 {
            let dx = state.shake * (state.shake_phase * 0.55).sin();
            let dy = state.shake * (state.shake_phase * 0.8).cos();
            for sprite in &mut out.sprites {
                sprite.pos[0] += dx;
                sprite.pos[1] += dy;
            }
        }
    }

    fn owns_keys(&self, world: &World) -> bool {
        state_of(world).phase == Phase::Menu
    }

    fn panel_visible(&self, world: &World) -> bool {
        state_of(world).phase != Phase::Menu
    }

    fn hud(&mut self) -> Vec<HudLine> {
        let mut lines = vec![HudLine::new(format!(
            "LEVEL {} - SCORE {} - LIVES {} - BOMBS {} - POWER L{} - GRAZE {} - BULLETS {}",
            self.stage,
            self.score,
            self.lives,
            self.bombs,
            shot_level(self.power),
            self.graze,
            self.bullets
        ))];
        if self.phase == Phase::BossSpell && self.boss_max > 0.0 {
            lines.push(HudLine::colored(
                format!("SPELL: {} - {}S", self.spell_name.to_uppercase(), self.spell_left),
                color::rgb8(255, 190, 120),
            ));
        }
        if self.game_over {
            lines.push(HudLine::colored(
                "GAME OVER - PRESS R FOR THE LEVEL MENU",
                color::rgb8(255, 90, 90),
            ));
        }
        lines.push(HudLine::colored(
            "WASD - SHIFT/SPACE FOCUS - AUTO FIRE - X BOMB - R MENU",
            color::rgb8(150, 150, 165),
        ));
        lines
    }

    fn on_key(&mut self, key: KeyCode, world: &mut World, _panel: &mut Panel) {
        let state = state_of(world);
        if state.phase == Phase::Menu {
            // Stage picker: navigate and launch.
            match key {
                KeyCode::ArrowUp | KeyCode::KeyW => {
                    if let Some(mut s) = world.get_resource_mut::<GameState>() {
                        s.menu_index = (s.menu_index + STAGE_COUNT - 1) % STAGE_COUNT;
                    }
                }
                KeyCode::ArrowDown | KeyCode::KeyS => {
                    if let Some(mut s) = world.get_resource_mut::<GameState>() {
                        s.menu_index = (s.menu_index + 1) % STAGE_COUNT;
                    }
                }
                KeyCode::Space | KeyCode::KeyZ | KeyCode::Enter => {
                    let stage = state.menu_index + 1;
                    world.insert_resource(start_state(stage));
                    // The ship is born here: the picker showed a clean screen.
                    let bounds = *world.get_resource::<SimBounds>().expect("SimBounds");
                    let _ = world.spawn((
                        Pos(bounds.center() + Vec2::new(0.0, bounds.height * 0.34)),
                        Iframes(0.0),
                        Player,
                    ));
                }
                _ => {}
            }
            return;
        }
        match key {
            KeyCode::KeyX => {
                if game_over_check(world) {
                    return;
                }
                let bombs = world
                    .get_resource::<GameState>()
                    .map(|s| s.bombs)
                    .unwrap_or(0);
                if bombs == 0 {
                    return;
                }
                let _ = clear_enemy_shots(world);
                // Damage everything on screen.
                let mut fairy_ids = Vec::new();
                {
                    let mut q = world.query::<Entity, With<Fairy>>().expect("fairies");
                    q.for_each(|e| fairy_ids.push(e));
                }
                for e in &fairy_ids {
                    if let Some(mut f) = world.get_mut::<Fairy>(*e) {
                        f.hp -= 5.0;
                        f.flash = 0.15;
                    }
                }
                let mut dead = Vec::new();
                for e in fairy_ids {
                    if world.get::<Fairy>(e).map(|f| f.hp <= 0.0).unwrap_or(false) {
                        dead.push(e);
                    }
                }
                if !dead.is_empty() {
                    world.despawn_batch(dead).expect("bomb fairies");
                }
                {
                    let mut q = world.query_mut::<&mut Boss, ()>().expect("boss");
                    q.for_each_mut(|mut b| {
                        b.hp -= 40.0;
                        b.flash = 0.15;
                    });
                }
                if let Some(mut s) = world.get_resource_mut::<GameState>() {
                    s.bombs -= 1;
                    s.shake = 5.0;
                }
                // Player burst visual + brief invulnerability.
                let player = world
                    .query::<(Entity, &Pos), With<Player>>()
                    .expect("player")
                    .iter()
                    .next()
                    .map(|(e, _)| e);
                if let Some(pe) = player {
                    if let Some(p) = world.get::<Pos>(pe).map(|p| p.0) {
                        let _ = burst_sparks(world, p, 46, color::hsl(0.55, 0.9, 0.75));
                    }
                    if let Some(mut i) = world.get_mut::<Iframes>(pe) {
                        i.0 = 1.2;
                    }
                }
            }
            KeyCode::KeyR => {
                // Back to the stage menu with a clean arena.
                let mut doomed = Vec::new();
                {
                    let mut q = world.query::<Entity, With<EnemyShot>>().expect("shots");
                    q.for_each(|e| doomed.push(e));
                }
                {
                    let mut q = world.query::<Entity, With<PlayerShot>>().expect("shots");
                    q.for_each(|e| doomed.push(e));
                }
                {
                    let mut q = world.query::<Entity, With<Fairy>>().expect("fairies");
                    q.for_each(|e| doomed.push(e));
                }
                {
                    let mut q = world.query::<Entity, With<Boss>>().expect("boss");
                    q.for_each(|e| doomed.push(e));
                }
                {
                    let mut q = world.query::<Entity, With<Spark>>().expect("sparks");
                    q.for_each(|e| doomed.push(e));
                }
                {
                    let mut q = world.query::<Entity, With<Pickup>>().expect("pickups");
                    q.for_each(|e| doomed.push(e));
                }
                {
                    let mut q = world.query::<Entity, With<Player>>().expect("player");
                    q.for_each(|e| doomed.push(e));
                }
                world.despawn_batch(doomed).expect("clear round");
                let back = GameState {
                    menu_index: state.stage.saturating_sub(1).min(STAGE_COUNT - 1),
                    ..GameState::default()
                };
                world.insert_resource(back);
                if let Some(mut pp) = world.get_resource_mut::<PlayerPos>() {
                    pp.0 = None;
                }
            }
            _ => {}
        }
    }
}

#[derive(Default)]
struct PlayerControl;
#[derive(Default)]
struct Director;
#[derive(Default)]
struct FairyAct;
#[derive(Default)]
struct Pattern;
#[derive(Default)]
struct MoveAll;
#[derive(Default)]
struct Cull;
#[derive(Default)]
struct Collide;

/// Component registration table (each closure registers one component).
fn register_fns() -> Vec<fn(&mut World)> {
    vec![
        |w| { w.register_table::<Pos>().expect("register Pos"); },
        |w| { w.register_table::<Vel>().expect("register Vel"); },
        |w| { w.register_table::<Accel>().expect("register Accel"); },
        |w| { w.register_table::<Bullet>().expect("register Bullet"); },
        |w| { w.register_table::<EnemyShot>().expect("register EnemyShot"); },
        |w| { w.register_table::<PlayerShot>().expect("register PlayerShot"); },
        |w| { w.register_table::<Fairy>().expect("register Fairy"); },
        |w| { w.register_table::<Boss>().expect("register Boss"); },
        |w| { w.register_table::<Player>().expect("register Player"); },
        |w| { w.register_table::<Iframes>().expect("register Iframes"); },
        |w| { w.register_table::<Spark>().expect("register Spark"); },
        |w| { w.register_table::<Pickup>().expect("register Pickup"); },
    ]
}

impl BulletHell {
    fn params_static() -> Vec<ParamSpec> {
        vec![ParamSpec::new("difficulty", "DIFFICULTY", 2.0, 1.0, 5.0, 1.0).integer()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_session() -> (World, Schedule, ScheduleExecutor, BulletHell) {
        let mut world = World::new();
        let mut schedule = Schedule::new();
        let mut executor = ScheduleExecutor::single_threaded();
        world.insert_resource(Time { dt: FIXED_DT });
        world.insert_resource(SimBounds {
            width: 800.0,
            height: 600.0,
        });
        world.insert_resource(InputState::default());
        let demo = BulletHell::new(&mut world, &mut schedule, &mut executor, &Args::default());
        // Start stage 1 directly (skipping the menu) and keep the scripted
        // player alive; it stands still and WILL get hit.
        world.insert_resource(start_state(1));
        let bounds = world.get_resource::<SimBounds>().copied().unwrap_or(SimBounds {
            width: 800.0,
            height: 600.0,
        });
        let _ = world.spawn((
            Pos(bounds.center() + Vec2::new(0.0, bounds.height * 0.34)),
            Iframes(0.0),
            Player,
        ));
        if let Some(mut state) = world.get_resource_mut::<GameState>() {
            state.lives = 1_000_000;
        }
        (world, schedule, executor, demo)
    }

    fn count_with<T: millarecs::component::Component>(world: &mut World) -> usize {
        let mut q = world.query::<Entity, With<T>>().expect("count query");
        let mut n = 0;
        q.for_each(|_| n += 1);
        n
    }

    #[test]
    fn stage_flows_from_waves_to_boss_spell() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();

        // Waves: 6 waves x ~3 s plus intro and clear pauses.
        for frame in 0..1500 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        let state = state_of(&world);
        assert_eq!(state.phase, Phase::BossSpell, "the boss arrived");
        assert_eq!(count_with::<Boss>(&mut world), 1, "one boss alive");
        assert_eq!(count_with::<Fairy>(&mut world), 0, "fairies cleared for the boss");
        assert!(count_with::<EnemyShot>(&mut world) > 0, "patterns fired");
    }

    #[test]
    fn spell_breaks_on_damage_and_advances() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        for frame in 0..1500 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        assert_eq!(state_of(&world).phase, Phase::BossSpell);

        // Point-blank damage: drop the spell to 1 hp and land a shot.
        let before = count_with::<EnemyShot>(&mut world);
        {
            let mut q = world.query_mut::<&mut Boss, ()>().expect("boss");
            q.for_each_mut(|mut b| b.hp = 0.5);
        }
        let boss_pos = {
            let mut q = world.query::<&Pos, With<Boss>>().expect("boss pos");
            let mut p = Vec2::ZERO;
            q.for_each(|pos| p = pos.0);
            p
        };
        let _ = world.spawn((
            Pos(boss_pos),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 2.5,
                tint: color::rgb8(140, 235, 255),
                grazed: false,
            },
            PlayerShot,
        ));

        // Collisions apply the damage after this tick's director pass, so the
        // break resolves on the following frame.
        demo.frame(&FrameCtx { frame: 1500 }, &mut world, &mut schedule, &mut executor);
        demo.frame(&FrameCtx { frame: 1501 }, &mut world, &mut schedule, &mut executor);
        let state = state_of(&world);
        assert_eq!(state.phase, Phase::BossSpell, "next spell, same stage");
        let spell: usize = {
            let mut q = world.query_all::<&Boss>().expect("boss");
            let mut s = 0;
            q.for_each(|b| s = b.spell);
            s
        };
        assert_eq!(spell, 1, "spell 1 replaced spell 0");
        // The break cleared the old volley; the new spell may have started
        // firing its own (few) bullets within the resolution tick.
        assert!(
            count_with::<EnemyShot>(&mut world) < before,
            "old pattern bullets were cleared"
        );
        assert!(state.score >= 1000, "break bonus paid");
    }

    #[test]
    fn final_spell_leads_to_next_stage() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        for frame in 0..1500 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        assert_eq!(state_of(&world).phase, Phase::BossSpell);

        // Skip straight to the last spell, nearly broken.
        let stage = state_of(&world).stage;
        {
            let mut q = world.query_mut::<&mut Boss, ()>().expect("boss");
            q.for_each_mut(|mut b| {
                b.spell = SPELLS_PER_STAGE - 1;
                b.max_hp = spell_hp(b.spell, stage);
                b.hp = 0.5;
            });
        }
        let boss_pos = {
            let mut q = world.query::<&Pos, With<Boss>>().expect("boss pos");
            let mut p = Vec2::ZERO;
            q.for_each(|pos| p = pos.0);
            p
        };
        let before = count_with::<EnemyShot>(&mut world);
        let _ = world.spawn((
            Pos(boss_pos),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 2.5,
                tint: color::rgb8(140, 235, 255),
                grazed: false,
            },
            PlayerShot,
        ));

        demo.frame(&FrameCtx { frame: 1500 }, &mut world, &mut schedule, &mut executor);
        demo.frame(&FrameCtx { frame: 1501 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(count_with::<Boss>(&mut world), 0, "boss defeated");
        assert!(count_with::<EnemyShot>(&mut world) < before, "old volley cleared");
        assert_eq!(state_of(&world).phase, Phase::StageClear, "stage clear pause");

        // The next stage starts once the clear-pause timer runs out (2 s).
        for frame in 1502..1650 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        assert_eq!(state_of(&world).stage, 2, "stage advanced");
        assert_eq!(state_of(&world).phase, Phase::Waves);
    }

    #[test]
    fn graze_scores_without_taking_damage() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        let player_pos = {
            let mut q = world.query::<&Pos, With<Player>>().expect("player pos");
            let mut p = Vec2::ZERO;
            q.for_each(|pos| p = pos.0);
            p
        };
        // 15 px away: outside the 8.5 px hit ring, inside the graze band.
        let _ = world.spawn((
            Pos(player_pos + Vec2::new(15.0, 0.0)),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 3.5,
                tint: color::rgb8(255, 90, 90),
                grazed: false,
            },
            EnemyShot,
        ));

        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        let state = state_of(&world);
        assert!(state.graze >= 1, "graze scored");
        assert!(!state.game_over, "no damage taken");
        assert_eq!(count_with::<EnemyShot>(&mut world), 1, "bullet survived");
    }

    #[test]
    fn bomb_clears_the_screen_and_spends_stock() {
        let (mut world, _schedule, _executor, mut demo) = test_session();
        let player_pos = {
            let mut q = world.query::<&Pos, With<Player>>().expect("player pos");
            let mut p = Vec2::ZERO;
            q.for_each(|pos| p = pos.0);
            p
        };
        for k in 0..5 {
            let _ = world.spawn((
                Pos(player_pos + Vec2::new(20.0 * k as f32, 0.0)),
                Vel(Vec2::ZERO),
                Bullet {
                    radius: 3.0,
                    tint: color::rgb8(255, 90, 90),
                    grazed: false,
                },
                EnemyShot,
            ));
        }
        let mut panel = Panel::new(BulletHell::params_static(), &Args::default(), 600.0);

        demo.on_key(KeyCode::KeyX, &mut world, &mut panel);
        assert_eq!(count_with::<EnemyShot>(&mut world), 0, "bomb cleared bullets");
        assert_eq!(state_of(&world).bombs, 1, "one bomb spent");

        // Second bomb with one left; third is a no-op.
        demo.on_key(KeyCode::KeyX, &mut world, &mut panel);
        assert_eq!(state_of(&world).bombs, 0);
        demo.on_key(KeyCode::KeyX, &mut world, &mut panel);
        assert_eq!(state_of(&world).bombs, 0, "no negative bombs");
    }

    #[test]
    fn restart_r_resets_the_stage() {
        let (mut world, _schedule, _executor, mut demo) = test_session();
        let mut state = start_state(3);
        state.phase = Phase::BossSpell;
        state.wave_index = 4;
        state.spell_timer = 5.0;
        state.score = 42_000;
        state.lives = 0;
        state.bombs = 0;
        state.graze = 99;
        state.game_over = true;
        world.insert_resource(state);
        let mut panel = Panel::new(BulletHell::params_static(), &Args::default(), 600.0);

        demo.on_key(KeyCode::KeyR, &mut world, &mut panel);
        let state = state_of(&world);
        assert_eq!(state.phase, Phase::Menu, "R returns to the stage menu");
        assert_eq!(state.menu_index, 2, "menu lands on the stage just played");
        assert!(!state.game_over);
        assert_eq!(count_with::<Fairy>(&mut world), 0, "stage cleared");
        assert_eq!(count_with::<Player>(&mut world), 0, "the ship waits in the menu");
    }

    #[test]
    fn menu_selects_and_starts_a_stage() {
        let (mut world, _schedule, _executor, mut demo) = test_session();
        // test_session starts stage 1; go back to the menu first.
        let mut panel = Panel::new(BulletHell::params_static(), &Args::default(), 600.0);
        demo.on_key(KeyCode::KeyR, &mut world, &mut panel);
        assert_eq!(state_of(&world).phase, Phase::Menu);
        assert_eq!(count_with::<Player>(&mut world), 0, "no ship in the menu");

        // Navigate down twice: FASE 3 highlighted.
        demo.on_key(KeyCode::ArrowDown, &mut world, &mut panel);
        demo.on_key(KeyCode::ArrowDown, &mut world, &mut panel);
        assert_eq!(state_of(&world).menu_index, 2);

        // Enter launches stage 3 and spawns the ship.
        demo.on_key(KeyCode::Enter, &mut world, &mut panel);
        let state = state_of(&world);
        assert_eq!(state.phase, Phase::Intro);
        assert_eq!(state.stage, 3);
        assert_eq!(state.lives, 3);
        assert!(!state.game_over);
        assert_eq!(count_with::<Player>(&mut world), 1, "ship spawned");
    }

    #[test]
    fn powerups_drop_and_upgrade_the_shot() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        // Park a weak fairy right on the player's nose so the magnet and the
        // collection both work at the same spot.
        let player_pos = {
            let mut q = world.query::<&Pos, With<Player>>().expect("player pos");
            let mut p = Vec2::ZERO;
            q.for_each(|pos| p = pos.0);
            p
        };
        let _ = world.spawn((
            Pos(player_pos),
            Vel(Vec2::ZERO),
            Fairy {
                hp: 1.0,
                kind: FairyKind::Sweeper,
                t: 0.0,
                x0: 0.0,
                y0: 0.0,
                omega: 0.0,
                amp: 0.0,
                dir: Vec2::ZERO,
                fire_cd: 99.0,
                fire_rate: 99.0,
                flash: 0.0,
            },
        ));
        let _ = world.spawn((
            Pos(player_pos),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 2.5,
                tint: color::rgb8(140, 235, 255),
                grazed: false,
            },
            PlayerShot,
        ));

        // The shot kills the fairy: a power pickup drops (and the magnet
        // collects it immediately, since it spawns on the player).
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        let pickups = count_with::<Pickup>(&mut world);
        let state = state_of(&world);
        assert!(
            pickups >= 1 || state.power >= 1,
            "the kill dropped a pickup or granted power directly"
        );

        // Let the magnet finish pulling everything in; power grows.
        demo.frame(&FrameCtx { frame: 1 }, &mut world, &mut schedule, &mut executor);
        for frame in 2..90 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        let state = state_of(&world);
        assert!(state.power >= 1, "power collected, got {}", state.power);
        assert!(shot_level(state.power) >= 1);

        // Higher power = wider volley: clear in-flight shots (autofire keeps
        // firing), run exactly one volley and count its streams.
        world.insert_resource(InputState::default());
        let mut flying = Vec::new();
        {
            let mut q = world.query::<Entity, With<PlayerShot>>().expect("shots");
            q.for_each(|e| flying.push(e));
        }
        world.despawn_batch(flying).expect("clear shots");
        let level = shot_level(state.power);
        demo.frame(&FrameCtx { frame: 90 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(
            count_with::<PlayerShot>(&mut world),
            level as usize,
            "one stream per shot level"
        );
    }

    #[test]
    fn stages_have_identity() {
        // Spell sequences differ per stage.
        for stage in 1..=5u32 {
            for spell in 0..SPELLS_PER_STAGE {
                let a = spell_def(stage, spell).name;
                let b = spell_def(stage + 1, spell).name;
                assert_ne!(a, b, "stage {stage} spell {spell} repeats next stage");
            }
        }
        // Enemy pools differ between stages.
        assert_ne!(stage_enemy_pool(1), stage_enemy_pool(2));
        assert_ne!(stage_enemy_pool(3), stage_enemy_pool(6));
        // Boss titles are unique.
        for a in 0..6usize {
            for b in (a + 1)..6usize {
                assert_ne!(BOSS_NAMES[a], BOSS_NAMES[b]);
            }
        }
    }

    #[test]
    fn shot_level_progression_is_cheap() {
        // One power pickup per level: L5 within 8 pickups.
        assert_eq!(shot_level(0), 1);
        assert_eq!(shot_level(1), 1);
        assert_eq!(shot_level(2), 2);
        assert_eq!(shot_level(4), 3);
        assert_eq!(shot_level(6), 4);
        assert_eq!(shot_level(8), 5);
        assert_eq!(shot_level(500), 5, "capped");
    }

    #[test]
    fn player_hit_costs_a_life_and_clears_the_screen() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        let mut state = start_state(1);
        state.lives = 3;
        world.insert_resource(state);
        let player_pos = {
            let mut q = world.query::<&Pos, With<Player>>().expect("player pos");
            let mut p = Vec2::ZERO;
            q.for_each(|pos| p = pos.0);
            p
        };
        let _ = world.spawn((
            Pos(player_pos),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 3.0,
                tint: color::rgb8(255, 90, 90),
                grazed: false,
            },
            EnemyShot,
        ));

        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        let state = state_of(&world);
        assert_eq!(state.lives, 2, "one hit taken");
        assert!(state.hit_flash > 0.0, "damage flash visible");
        assert!(state.shake > 0.0, "screen shook");
        assert_eq!(count_with::<EnemyShot>(&mut world), 0, "screen cleared");

        // During the invulnerability window nothing happens.
        let _ = world.spawn((
            Pos(player_pos),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 3.0,
                tint: color::rgb8(255, 90, 90),
                grazed: false,
            },
            EnemyShot,
        ));
        demo.frame(&FrameCtx { frame: 1 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(state_of(&world).lives, 2, "iframes protected the player");
    }

    #[test]
    fn phase_completion_harvests_bullets_into_pickups() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        // Force the pre-boss sweep: five enemy bullets on screen when the
        // wave act completes.
        let _bounds = world.get_resource::<SimBounds>().copied().unwrap_or(SimBounds {
            width: 800.0,
            height: 600.0,
        });
        for k in 0..5u32 {
            let _ = world.spawn((
                Pos(Vec2::new(100.0 + 80.0 * k as f32, 150.0)),
                Vel(Vec2::new(0.0, 90.0)),
                Bullet {
                    radius: 3.0,
                    tint: color::rgb8(255, 90, 90),
                    grazed: false,
                },
                EnemyShot,
            ));
        }
        if let Some(mut state) = world.get_resource_mut::<GameState>() {
            state.phase = Phase::ClearPause;
            state.phase_timer = 0.5;
        }
        // Park the player among the bullets so the magnet collects them.
        {
            let mut q = world.query_mut::<&mut Pos, With<Player>>().expect("player");
            q.for_each_mut(|mut p| p.0 = Vec2::new(260.0, 165.0));
        }

        // One tick: the director harvests, turning bullets into score
        // pickups. (One may be absorbed instantly if it spawned on the player.)
        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(count_with::<EnemyShot>(&mut world), 0, "bullets harvested");
        assert!(
            count_with::<Pickup>(&mut world) >= 4,
            "one pickup per bullet, minus instant collects"
        );

        // The player sits among them: the magnet absorbs the near ones, the
        // far ones fall and get culled at the bottom. Either way, every
        // pickup leaves the world.
        let score_before = state_of(&world).score;
        for frame in 1..900 {
            demo.frame(&FrameCtx { frame }, &mut world, &mut schedule, &mut executor);
        }
        let state = state_of(&world);
        assert!(state.score > score_before, "harvested pickups scored");
        assert_eq!(count_with::<Pickup>(&mut world), 0, "all absorbed or culled");
    }

    #[test]
    fn player_hit_screen_clear_gives_no_reward() {
        let (mut world, mut schedule, mut executor, mut demo) = test_session();
        let player_pos = {
            let mut q = world.query::<&Pos, With<Player>>().expect("player pos");
            let mut p = Vec2::ZERO;
            q.for_each(|pos| p = pos.0);
            p
        };
        let _ = world.spawn((
            Pos(player_pos + Vec2::new(40.0, 0.0)),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 3.0,
                tint: color::rgb8(255, 90, 90),
                grazed: false,
            },
            EnemyShot,
        ));
        let _ = world.spawn((
            Pos(player_pos),
            Vel(Vec2::ZERO),
            Bullet {
                radius: 3.0,
                tint: color::rgb8(255, 90, 90),
                grazed: false,
            },
            EnemyShot,
        ));

        demo.frame(&FrameCtx { frame: 0 }, &mut world, &mut schedule, &mut executor);
        assert_eq!(count_with::<EnemyShot>(&mut world), 0, "screen cleared");
        assert_eq!(count_with::<Pickup>(&mut world), 0, "no reward for being hit");
    }
}

