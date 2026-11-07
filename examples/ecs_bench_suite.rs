//! General-purpose ECS benchmark suite.
//!
//! Measures structural operations (spawn, despawn, insert, batch insert,
//! remove), point lookups (`get`), query families (driver, filter, changed,
//! oneshot, write), route selection (archetype vs sparse driver), deferred
//! commands, prepared operations, mixed transitions, parallel schedules, and
//! scaling by archetype width (2/4/8/16 components).
//!
//! Methodology: each case runs `warmup` untimed runs, then `samples` timed
//! samples of `runs` repetitions each over a fresh world per run. Reported
//! statistics are mean/median/p95/min/max per run, standard deviation and
//! relative standard deviation, plus normalized ns/item. Checksums keep the
//! optimizer from removing the measured work.
//!
//! Measurement conditions: compare only runs taken on the same machine at
//! rest (fixed power plan, no background load), and note the CPU alongside
//! any result worth keeping.
//!
//! CSV round-trip: `--csv [--output-name NAME]` saves a run under
//! `examples/results/ecs_bench_suite/`, and `--compare [FILE]` prints a
//! comparison against a stored run (latest when no file is given).
//!
//! Common commands:
//!
//! General measurement of 100K rows:
//! cargo run --release --example ecs_bench_suite -- --rows 100000
//!
//! High-precision general measurement:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --samples 30 --runs 10 --warmup 5
//!
//! Structural operations only:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite structural --samples 20 --runs 10 --warmup 5
//!
//! Single insertion, batch insertion, or removal only:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite insert --samples 20 --runs 10 --warmup 5
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite batch --samples 20 --runs 10 --warmup 5
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite remove --samples 20 --runs 10 --warmup 5
//!
//! Compare direct get, prepared get, and prepared get with permuted order:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite get --samples 30 --runs 20 --warmup 10
//!
//! Queries with reused state only:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite query --samples 30 --runs 20 --warmup 10
//!
//! Compare different selectivity of the sparse driver:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite query --sparse-stride 2
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite query --sparse-stride 32
//!
//! Compare automatic, archetype, and sparse routes over the same world:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite query-locality --sparse-density 75 --sparse-order both --samples 30 --runs 20 --warmup 10
//!
//! Compare dense and gapped fragmented storage:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite get --fragmented-gap-stride 0
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite get --fragmented-gap-stride 4
//!
//! Scaling by archetype width and CSV output:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite width
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --samples 30 --runs 10 --warmup 5 --csv
//!
//! Compare ordinary, prepared, and batch structural transitions:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite width --samples 20 --runs 5 --warmup 3 --csv --output-name scalar_structural_prepared
//!
//! Compare prepared scalar calls with one shared structural session:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite width --samples 20 --runs 5 --warmup 3 --csv --output-name prepared_structural_session
//!
//! Compare scalar, prepared, session, and batch spawning:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite width --samples 20 --runs 5 --warmup 3 --csv --output-name prepared_spawn_session
//!
//! Compare fused deferred spawning with explicit batch and direct spawning:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite spawn-commands --samples 15 --runs 5 --warmup 3 --csv --output-name deferred_spawn_fused
//!
//! Compare scalar and component-major batch spawning by storage and width:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite width --samples 10 --runs 3 --warmup 2 --csv --output-name sparse_spawn_component_major
//!
//! Despawn crossover by batch size and archetype width:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite despawn-scaling --samples 10 --runs 3 --warmup 2 --csv --output-name despawn_scaling
//!
//! Compare scalar, deferred fused, and explicit batch sparse despawn:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite despawn-commands --samples 15 --runs 5 --warmup 3 --csv --output-name sparse_despawn_commands
//!
//! Compare scalar and component-major mixed Table/Sparse insertion:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite mixed-transition --samples 10 --runs 3 --warmup 2 --csv --output-name mixed_transition
//!
//! Compare structural repair with dormant and active sparse drivers:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --suite sparse-driver-transition --samples 10 --runs 3 --warmup 2 --csv --output-name sparse_driver_transition
//!
//! Save with a custom name and compare with the latest or a specific result:
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --csv --output-name baseline
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --compare
//! cargo run --release --example ecs_bench_suite -- --rows 100000 --compare baseline.csv

use std::{
    collections::BTreeMap,
    env,
    error::Error,
    fs::{self, File},
    hint::black_box,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    process,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use millarecs::{
    component::{Bundle, Component},
    entity::{ArchetypeStorageKind, Entity},
    query::{
        filter::{Changed, QueryFilter, With},
        QueryData, QueryRoute, QueryRoutePolicy, QueryState,
    },
    system::{
        commands::Commands,
        query::Query,
        schedule::{Schedule, ScheduleExecutor},
    },
    world::{ComponentHandle, PreparedInsert, PreparedRemove, PreparedSpawn, World, WorldError},
};

#[doc(hidden)]
mod __private {
    pub use millarecs::__private::*;
}

type AppResult<T> = Result<T, Box<dyn Error>>;

const BENCHMARK_NAME: &str = "ecs_bench_suite";
const RESULTS_DIRECTORY: &str = "results";

#[derive(Clone, Copy)]
struct Position {
    x: u64,
    y: u64,
}

impl Component for Position {}

#[derive(Clone, Copy)]
struct Velocity {
    x: u64,
    y: u64,
}

impl Component for Velocity {}

#[derive(Clone, Copy)]
struct TableExtra {
    value: u64,
}

impl Component for TableExtra {}

#[derive(Clone, Copy)]
struct SparseExtra {
    value: u64,
}

impl Component for SparseExtra {}

#[derive(Clone, Copy)]
struct BenchColumn<const COLUMN: usize> {
    value: u64,
}

impl<const COLUMN: usize> Component for BenchColumn<COLUMN> {}

trait BenchValue {
    fn bench_value(&self) -> u64;
}

impl<const COLUMN: usize> BenchValue for BenchColumn<COLUMN> {
    #[inline]
    fn bench_value(&self) -> u64 {
        self.value
    }
}

trait WidthBundle: Bundle {
    type Last: Component + BenchValue;

    const COLUMNS: usize;

    fn new(row: usize) -> Self;
    fn register_table(world: &mut World) -> Result<(), WorldError>;
    fn register_sparse(world: &mut World) -> Result<(), WorldError>;
}

macro_rules! define_width_bundle {
    (
        $bundle:ident,
        $columns:literal,
        $last:literal,
        { $($field:ident: $column:literal),+ $(,)? }
    ) => {
        #[derive(millarecs_macros::Bundle)]
        struct $bundle {
            $($field: BenchColumn<$column>,)+
        }

        impl WidthBundle for $bundle {
            type Last = BenchColumn<$last>;

            const COLUMNS: usize = $columns;

            #[inline]
            fn new(row: usize) -> Self {
                Self {
                    $($field: bench_column::<$column>(row),)+
                }
            }

            fn register_table(world: &mut World) -> Result<(), WorldError> {
                $(world.register_table::<BenchColumn<$column>>()?;)+
                Ok(())
            }

            fn register_sparse(world: &mut World) -> Result<(), WorldError> {
                $(world.register_sparse::<BenchColumn<$column>>()?;)+
                Ok(())
            }
        }
    };
}

define_width_bundle!(Width2, 2, 1, { c0: 0, c1: 1 });
define_width_bundle!(Width4, 4, 3, { c0: 0, c1: 1, c2: 2, c3: 3 });
define_width_bundle!(Width8, 8, 7, {
    c0: 0,
    c1: 1,
    c2: 2,
    c3: 3,
    c4: 4,
    c5: 5,
    c6: 6,
    c7: 7,
});
define_width_bundle!(Width16, 16, 15, {
    c0: 0,
    c1: 1,
    c2: 2,
    c3: 3,
    c4: 4,
    c5: 5,
    c6: 6,
    c7: 7,
    c8: 8,
    c9: 9,
    c10: 10,
    c11: 11,
    c12: 12,
    c13: 13,
    c14: 14,
    c15: 15,
});
// Extra prefixes used by the one-shot query benchmark: prefix bundles 1..=9
// crossed with optional Velocity/TableExtra variants build a world with 36
// distinct archetypes, past the automatic promotion threshold.
define_width_bundle!(Width1, 1, 0, { c0: 0 });
define_width_bundle!(Width3, 3, 2, { c0: 0, c1: 1, c2: 2 });
define_width_bundle!(Width5, 5, 4, {
    c0: 0,
    c1: 1,
    c2: 2,
    c3: 3,
    c4: 4,
});
define_width_bundle!(Width6, 6, 5, {
    c0: 0,
    c1: 1,
    c2: 2,
    c3: 3,
    c4: 4,
    c5: 5,
});
define_width_bundle!(Width7, 7, 6, {
    c0: 0,
    c1: 1,
    c2: 2,
    c3: 3,
    c4: 4,
    c5: 5,
    c6: 6,
});
define_width_bundle!(Width9, 9, 8, {
    c0: 0,
    c1: 1,
    c2: 2,
    c3: 3,
    c4: 4,
    c5: 5,
    c6: 6,
    c7: 7,
    c8: 8,
});

macro_rules! define_query_locality_width_case {
    (
        $bench_fn:ident,
        $visit_fn:ident,
        $columns:literal,
        { $($value:ident: $column:literal),+ $(,)? }
    ) => {
        fn $visit_fn(
            world: &World,
            state: &mut QueryState<
                ($( &BenchColumn<$column>, )+),
                With<SparseExtra>,
            >,
        ) -> AppResult<BenchResult> {
            let mut items = 0usize;
            let mut checksum = 0_u64;
            let mut query = world.query_with(state)?;

            query.for_each(|($($value,)+)| {
                items += 1;

                $(
                    checksum = checksum.wrapping_add($value.value);
                )+
            });

            Ok(BenchResult {
                items,
                checksum: black_box(checksum),
            })
        }

        fn $bench_fn(
            config: Config,
            order: SparseInsertionOrder,
        ) -> AppResult<Vec<Measurement>> {
            let world = build_query_locality_width_world(
                config.rows,
                config.sparse_density,
                order,
            )?;

            let sparse_driver = world
                .component_handle::<SparseExtra>()
                .expect("SparseExtra must be registered")
                .component_id();

            let mut automatic_state = world.query_state_cached::<
                ($( &BenchColumn<$column>, )+),
                With<SparseExtra>,
            >()?;

            let mut archetype_state = world.query_state_cached::<
                ($( &BenchColumn<$column>, )+),
                With<SparseExtra>,
            >()?;

            let mut sparse_state = world.query_state_cached::<
                ($( &BenchColumn<$column>, )+),
                With<SparseExtra>,
            >()?;

            archetype_state
                .set_route_policy(QueryRoutePolicy::ForceArchetype)
                .map_err(WorldError::from)?;

            sparse_state
                .set_route_policy(
                    QueryRoutePolicy::ForceSparseDriver(sparse_driver),
                )
                .map_err(WorldError::from)?;

            let automatic_plan =
                query_plan_info(&world, &automatic_state)?
                    .with_columns($columns);

            let archetype_plan =
                query_plan_info(&world, &archetype_state)?
                    .with_columns($columns);

            let sparse_plan =
                query_plan_info(&world, &sparse_state)?
                    .with_columns($columns);

            let cases = match order {
                SparseInsertionOrder::Ordered => [
                    ("automatic_ordered", automatic_plan),
                    ("archetype_ordered", archetype_plan),
                    ("sparse_ordered", sparse_plan),
                ],
                SparseInsertionOrder::Permuted => [
                    ("automatic_permuted", automatic_plan),
                    ("archetype_permuted", archetype_plan),
                    ("sparse_permuted", sparse_plan),
                ],
            };

            measure_static_triplet(
                "query_locality_width",
                cases,
                config,
                || $visit_fn(&world, &mut automatic_state),
                || $visit_fn(&world, &mut archetype_state),
                || $visit_fn(&world, &mut sparse_state),
            )
        }
    };
}

define_query_locality_width_case!(
    bench_query_locality_width_2,
    visit_query_locality_width_2,
    2,
    { c0: 0, c1: 1 }
);

define_query_locality_width_case!(
    bench_query_locality_width_4,
    visit_query_locality_width_4,
    4,
    { c0: 0, c1: 1, c2: 2, c3: 3 }
);

define_query_locality_width_case!(
    bench_query_locality_width_8,
    visit_query_locality_width_8,
    8,
    {
        c0: 0, c1: 1, c2: 2, c3: 3,
        c4: 4, c5: 5, c6: 6, c7: 7
    }
);

define_query_locality_width_case!(
    bench_query_locality_width_12,
    visit_query_locality_width_12,
    12,
    {
        c0: 0, c1: 1, c2: 2, c3: 3,
        c4: 4, c5: 5, c6: 6, c7: 7,
        c8: 8, c9: 9, c10: 10, c11: 11
    }
);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Suite {
    /// Every general case; excludes the specialized suites listed below.
    All,
    /// Group alias for spawn, despawn, insert, batch, and remove.
    Structural,
    /// Spawning entities with a plain bundle.
    Spawn,
    /// Despawning entities one by one.
    Despawn,
    /// Group alias for insert, batch, and remove.
    Component,
    /// Inserting a bundle into existing entities (archetype migration).
    Insert,
    /// Batch insertion of a bundle into existing entities.
    Batch,
    /// Removing a component from existing entities (archetype migration).
    Remove,
    /// Point lookups through `World::get` on dense and gapped storage.
    Get,
    /// Query families: drivers, filters, changed, write, and oneshot.
    Query,
    /// Route comparison (automatic, archetype, sparse) over one world.
    QueryLocality,
    /// Route comparison repeated across archetype widths.
    QueryLocalityWidth,
    /// Despawn crossover by batch size and archetype width.
    DespawnScaling,
    /// Scalar, deferred fused, and batch despawn through commands.
    DespawnCommands,
    /// Scalar, deferred fused, and batch spawn through commands.
    SpawnCommands,
    /// Mixed table/sparse bundle insertion, scalar vs component-major.
    MixedTransition,
    /// Structural repair with dormant vs active sparse drivers.
    SparseDriverTransition,
    /// Schedule dispatch and parallel execution scaling.
    ParExecutor,
    /// Structural and query scaling across 2/4/8/16-column archetypes.
    Width,
}

impl Suite {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "all" => Ok(Self::All),
            "structural" => Ok(Self::Structural),
            "spawn" => Ok(Self::Spawn),
            "despawn" => Ok(Self::Despawn),
            "component" => Ok(Self::Component),
            "insert" => Ok(Self::Insert),
            "batch" => Ok(Self::Batch),
            "remove" => Ok(Self::Remove),
            "get" => Ok(Self::Get),
            "query" => Ok(Self::Query),
            "query-locality" => Ok(Self::QueryLocality),
            "query-locality-width" => Ok(Self::QueryLocalityWidth),
            "despawn-scaling" => Ok(Self::DespawnScaling),
            "despawn-commands" => Ok(Self::DespawnCommands),
            "spawn-commands" => Ok(Self::SpawnCommands),
            "mixed-transition" => Ok(Self::MixedTransition),
            "sparse-driver-transition" => Ok(Self::SparseDriverTransition),
            "par-executor" => Ok(Self::ParExecutor),
            "width" => Ok(Self::Width),
            other => Err(format!("invalid suite: {other}")),
        }
    }

    fn includes(self, suite: Self) -> bool {
        match self {
            Self::All => !matches!(
                suite,
                Self::QueryLocality
                    | Self::QueryLocalityWidth
                    | Self::DespawnScaling
                    | Self::DespawnCommands
                    | Self::SpawnCommands
                    | Self::MixedTransition
                    | Self::SparseDriverTransition
                    | Self::ParExecutor
                    | Self::Width
            ),
            Self::Structural => matches!(
                suite,
                Self::Spawn | Self::Despawn | Self::Insert | Self::Batch | Self::Remove
            ),
            Self::Component => matches!(suite, Self::Insert | Self::Batch | Self::Remove),
            _ => self == suite,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::Structural => "structural",
            Self::Spawn => "spawn",
            Self::Despawn => "despawn",
            Self::Component => "component",
            Self::Insert => "insert",
            Self::Batch => "batch",
            Self::Remove => "remove",
            Self::Get => "get",
            Self::Query => "query",
            Self::QueryLocality => "query-locality",
            Self::QueryLocalityWidth => "query-locality-width",
            Self::DespawnScaling => "despawn-scaling",
            Self::DespawnCommands => "despawn-commands",
            Self::SpawnCommands => "spawn-commands",
            Self::MixedTransition => "mixed-transition",
            Self::SparseDriverTransition => "sparse-driver-transition",
            Self::ParExecutor => "par-executor",
            Self::Width => "width",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SparseInsertionOrder {
    Ordered,
    Permuted,
}

#[derive(Clone, Copy)]
enum SparseOrderSelection {
    Ordered,
    Permuted,
    Both,
}

impl SparseOrderSelection {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "ordered" => Ok(Self::Ordered),
            "permuted" => Ok(Self::Permuted),
            "both" => Ok(Self::Both),
            other => Err(format!("invalid sparse insertion order: {other}")),
        }
    }

    #[inline]
    fn includes(self, order: SparseInsertionOrder) -> bool {
        matches!(self, Self::Both)
            || matches!(
                (self, order),
                (Self::Ordered, SparseInsertionOrder::Ordered)
                    | (Self::Permuted, SparseInsertionOrder::Permuted)
            )
    }

    const fn as_str(self) -> &'static str {
        match self {
            Self::Ordered => "ordered",
            Self::Permuted => "permuted",
            Self::Both => "both",
        }
    }
}

#[derive(Clone, Copy)]
struct Config {
    rows: usize,
    samples: usize,
    runs_per_sample: usize,
    warmup_runs: usize,
    sparse_stride: usize,
    sparse_density: usize,
    sparse_order: SparseOrderSelection,
    fragmented_gap_stride: usize,
    suite: Suite,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            rows: 20_000,
            samples: 8,
            runs_per_sample: 5,
            warmup_runs: 2,
            sparse_stride: 8,
            sparse_density: 75,
            sparse_order: SparseOrderSelection::Both,
            fragmented_gap_stride: 4,
            suite: Suite::All,
        }
    }
}

enum ComparisonSource {
    Latest,
    File(String),
}

#[derive(Default)]
struct OutputOptions {
    save_csv: bool,
    output_name: Option<String>,
    comparison: Option<ComparisonSource>,
}

impl Config {
    fn from_args() -> Result<Option<(Self, OutputOptions)>, String> {
        let mut config = Self::default();
        let mut output = OutputOptions::default();
        let mut args = env::args().skip(1).peekable();

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => return Ok(None),
                "--rows" => config.rows = parse_arg(&arg, args.next())?,
                "--samples" => config.samples = parse_arg(&arg, args.next())?,
                "--runs" => config.runs_per_sample = parse_arg(&arg, args.next())?,
                "--warmup" => config.warmup_runs = parse_arg(&arg, args.next())?,
                "--sparse-stride" => config.sparse_stride = parse_arg(&arg, args.next())?,
                "--sparse-density" => config.sparse_density = parse_arg(&arg, args.next())?,
                "--sparse-order" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --sparse-order".to_owned())?;
                    config.sparse_order = SparseOrderSelection::parse(&value)?;
                }
                "--fragmented-gap-stride" => {
                    config.fragmented_gap_stride = parse_arg(&arg, args.next())?
                }
                "--suite" => {
                    let value = args
                        .next()
                        .ok_or_else(|| "missing value for --suite".to_owned())?;
                    config.suite = Suite::parse(&value)?;
                }
                "--csv" => output.save_csv = true,
                "--output-name" | "--name" => {
                    output.output_name = Some(
                        args.next()
                            .ok_or_else(|| format!("missing value for {arg}"))?,
                    );
                    output.save_csv = true;
                }
                "--compare" => {
                    let file = args
                        .next_if(|value| !value.starts_with('-'))
                        .map(ComparisonSource::File)
                        .unwrap_or(ComparisonSource::Latest);
                    output.comparison = Some(file);
                }
                other => return Err(format!("unknown argument: {other}")),
            }
        }

        config.validate()?;
        Ok(Some((config, output)))
    }

    fn validate(self) -> Result<(), String> {
        if self.rows == 0 {
            return Err("--rows must be greater than 0".to_owned());
        }

        if self.samples == 0 {
            return Err("--samples must be greater than 0".to_owned());
        }

        if self.runs_per_sample == 0 {
            return Err("--runs must be greater than 0".to_owned());
        }

        if self.sparse_stride < 2 {
            return Err("--sparse-stride must be greater than or equal to 2".to_owned());
        }

        if !(1..=100).contains(&self.sparse_density) {
            return Err("--sparse-density must be between 1 and 100".to_owned());
        }

        if self.fragmented_gap_stride == 1 {
            return Err(
                "--fragmented-gap-stride must be 0 or greater than or equal to 2".to_owned(),
            );
        }

        Ok(())
    }
}

#[derive(Clone, Copy)]
struct PlanInfo {
    route: &'static str,
    world_rows: usize,
    archetype_rows: Option<usize>,
    sparse_candidates: Option<usize>,
    columns: Option<usize>,
}

impl PlanInfo {
    fn operation(world_rows: usize) -> Self {
        Self {
            route: "-",
            world_rows,
            archetype_rows: None,
            sparse_candidates: None,
            columns: None,
        }
    }

    fn with_columns(mut self, columns: usize) -> Self {
        self.columns = Some(columns);
        self
    }
}

struct BenchResult {
    items: usize,
    checksum: u64,
}

struct Measurement {
    operation: &'static str,
    case: &'static str,
    plan: PlanInfo,
    samples: Vec<Duration>,
    total_items: usize,
    checksum: u64,
    runs_per_sample: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct MeasurementKey {
    operation: String,
    case: String,
    columns: String,
}

#[derive(Clone, Copy)]
enum StoredStatistic {
    Mean,
    Median,
}

impl StoredStatistic {
    fn label(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Median => "median",
        }
    }
}

struct StoredMeasurement {
    route: String,
    items_per_run: f64,
    ns_per_item: f64,
}

struct StoredBenchmark {
    command: Option<String>,
    statistic: StoredStatistic,
    measurements: BTreeMap<MeasurementKey, StoredMeasurement>,
}

impl Measurement {
    fn key(&self) -> MeasurementKey {
        MeasurementKey {
            operation: self.operation.to_owned(),
            case: self.case.to_owned(),
            columns: optional_usize(self.plan.columns),
        }
    }

    fn measured_runs(&self) -> usize {
        self.samples.len() * self.runs_per_sample
    }

    fn total_ns(&self) -> f64 {
        self.samples
            .iter()
            .map(|duration| duration.as_nanos() as f64)
            .sum()
    }

    fn mean_run_ns(&self) -> f64 {
        self.total_ns() / self.measured_runs() as f64
    }

    fn sample_run_ns(&self) -> impl Iterator<Item = f64> + '_ {
        self.samples
            .iter()
            .map(|duration| duration.as_nanos() as f64 / self.runs_per_sample as f64)
    }

    fn percentile_run_ns(&self, percentile: f64) -> f64 {
        let mut values: Vec<_> = self.sample_run_ns().collect();
        values.sort_unstable_by(f64::total_cmp);

        let position = percentile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
        let lower = position.floor() as usize;
        let upper = position.ceil() as usize;

        if lower == upper {
            return values[lower];
        }

        let fraction = position - lower as f64;
        values[lower] + (values[upper] - values[lower]) * fraction
    }

    fn median_run_ns(&self) -> f64 {
        self.percentile_run_ns(0.5)
    }

    fn p95_run_ns(&self) -> f64 {
        self.percentile_run_ns(0.95)
    }

    fn min_run_ns(&self) -> f64 {
        self.sample_run_ns().fold(f64::INFINITY, f64::min)
    }

    fn max_run_ns(&self) -> f64 {
        self.sample_run_ns().fold(0.0, f64::max)
    }

    fn stddev_run_ns(&self) -> f64 {
        let mean = self.mean_run_ns();
        let variance = self
            .sample_run_ns()
            .map(|run_ns| {
                let delta = run_ns - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.samples.len() as f64;

        variance.sqrt()
    }

    fn items_per_run(&self) -> f64 {
        self.total_items as f64 / self.measured_runs() as f64
    }

    fn ns_per_item(&self) -> f64 {
        self.total_ns() / self.total_items.max(1) as f64
    }

    fn median_ns_per_item(&self) -> f64 {
        self.median_run_ns() / self.items_per_run().max(1.0)
    }

    fn p95_ns_per_item(&self) -> f64 {
        self.p95_run_ns() / self.items_per_run().max(1.0)
    }

    fn relative_stddev_percent(&self) -> f64 {
        let mean = self.mean_run_ns();

        if mean == 0.0 {
            0.0
        } else {
            self.stddev_run_ns() * 100.0 / mean
        }
    }
}

struct StaticMeasurementAccumulator {
    case: &'static str,
    plan: PlanInfo,
    samples: Vec<Duration>,
    total_items: usize,
    checksum: u64,
}

impl StaticMeasurementAccumulator {
    fn new(case: &'static str, plan: PlanInfo, sample_capacity: usize) -> Self {
        Self {
            case,
            plan,
            samples: Vec::with_capacity(sample_capacity),
            total_items: 0,
            checksum: 0,
        }
    }

    fn warm_up<F>(&mut self, run: &mut F) -> AppResult<()>
    where
        F: FnMut() -> AppResult<BenchResult>,
    {
        let result = run()?;
        self.checksum = self.checksum.wrapping_add(result.checksum);
        Ok(())
    }

    fn measure_sample<F>(&mut self, runs: usize, run: &mut F) -> AppResult<()>
    where
        F: FnMut() -> AppResult<BenchResult>,
    {
        let started = Instant::now();

        for _ in 0..runs {
            let result = run()?;
            self.total_items += result.items;
            self.checksum = self.checksum.wrapping_add(result.checksum);
        }

        self.samples.push(started.elapsed());
        Ok(())
    }

    fn finish(self, operation: &'static str, runs_per_sample: usize) -> Measurement {
        black_box(self.checksum);

        Measurement {
            operation,
            case: self.case,
            plan: self.plan,
            samples: self.samples,
            total_items: self.total_items,
            checksum: self.checksum,
            runs_per_sample,
        }
    }
}

fn measure_static_triplet<A, B, C>(
    operation: &'static str,
    cases: [(&'static str, PlanInfo); 3],
    config: Config,
    mut run_a: A,
    mut run_b: B,
    mut run_c: C,
) -> AppResult<Vec<Measurement>>
where
    A: FnMut() -> AppResult<BenchResult>,
    B: FnMut() -> AppResult<BenchResult>,
    C: FnMut() -> AppResult<BenchResult>,
{
    let mut a = StaticMeasurementAccumulator::new(cases[0].0, cases[0].1, config.samples);
    let mut b = StaticMeasurementAccumulator::new(cases[1].0, cases[1].1, config.samples);
    let mut c = StaticMeasurementAccumulator::new(cases[2].0, cases[2].1, config.samples);

    // Warm-up uses the same rotation as measured samples.
    for index in 0..config.warmup_runs {
        match index % 3 {
            0 => {
                a.warm_up(&mut run_a)?;
                b.warm_up(&mut run_b)?;
                c.warm_up(&mut run_c)?;
            }
            1 => {
                b.warm_up(&mut run_b)?;
                c.warm_up(&mut run_c)?;
                a.warm_up(&mut run_a)?;
            }
            _ => {
                c.warm_up(&mut run_c)?;
                a.warm_up(&mut run_a)?;
                b.warm_up(&mut run_b)?;
            }
        }
    }

    for sample in 0..config.samples {
        match sample % 3 {
            0 => {
                a.measure_sample(config.runs_per_sample, &mut run_a)?;
                b.measure_sample(config.runs_per_sample, &mut run_b)?;
                c.measure_sample(config.runs_per_sample, &mut run_c)?;
            }
            1 => {
                b.measure_sample(config.runs_per_sample, &mut run_b)?;
                c.measure_sample(config.runs_per_sample, &mut run_c)?;
                a.measure_sample(config.runs_per_sample, &mut run_a)?;
            }
            _ => {
                c.measure_sample(config.runs_per_sample, &mut run_c)?;
                a.measure_sample(config.runs_per_sample, &mut run_a)?;
                b.measure_sample(config.runs_per_sample, &mut run_b)?;
            }
        }
    }

    Ok(vec![
        a.finish(operation, config.runs_per_sample),
        b.finish(operation, config.runs_per_sample),
        c.finish(operation, config.runs_per_sample),
    ])
}

struct EntityWorld {
    world: World,
    entities: Vec<Entity>,
}

struct CommandDespawnWorld {
    world: World,
    schedule: Schedule,
    executor: ScheduleExecutor,
    items: usize,
    checksum: u64,
}

struct CommandSpawnWorld {
    world: World,
    schedule: Schedule,
    executor: ScheduleExecutor,
    items: usize,
}

#[derive(Clone, Copy)]
enum DeferredDespawnMode {
    Scalar,
    Fused,
}

#[derive(Clone, Copy)]
enum DeferredSpawnMode {
    Scalar,
    Batch,
}

struct BatchInsertWorld<B> {
    world: World,
    entries: Vec<(Entity, B)>,
    checksum: u64,
}

struct PreparedSpawnWorld<B: Bundle> {
    world: World,
    prepared: PreparedSpawn<B>,
}

struct PreparedInsertWorld<B: Bundle> {
    world: World,
    entities: Vec<Entity>,
    prepared: PreparedInsert<B>,
}

struct PreparedRemoveWorld<C: Component> {
    world: World,
    entities: Vec<Entity>,
    prepared: PreparedRemove<C>,
}

#[derive(Clone, Copy)]
enum WidthStorage {
    Table,
    Sparse,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        process::exit(1);
    }
}

fn run() -> AppResult<()> {
    let command = executed_command();
    let (config, output) = match Config::from_args().map_err(|err| {
        io::Error::new(io::ErrorKind::InvalidInput, format!("{err}\n\n{}", help()))
    })? {
        Some(options) => options,
        None => {
            println!("{}", help());
            return Ok(());
        }
    };

    println!("Preparing ECS benchmark suite...");
    println!(
        "rows={} samples={} runs={} warmup={} suite={} sparse_stride={} sparse_density={} sparse_order={} fragmented_gap_stride={}",
        config.rows,
        config.samples,
        config.runs_per_sample,
        config.warmup_runs,
        config.suite.as_str(),
        config.sparse_stride,
        config.sparse_density,
        config.sparse_order.as_str(),
        config.fragmented_gap_stride
    );
    println!();

    let mut measurements = Vec::new();

    if config.suite.includes(Suite::Spawn) {
        measurements.extend(bench_spawn(config)?);
    }

    if config.suite.includes(Suite::Despawn) {
        measurements.extend(bench_despawn(config)?);
    }

    if config.suite.includes(Suite::Insert) {
        measurements.extend(bench_insert_component(config)?);
    }

    if config.suite.includes(Suite::Batch) {
        measurements.extend(bench_insert_batch(config)?);
    }

    if config.suite.includes(Suite::Remove) {
        measurements.extend(bench_remove_component(config)?);
    }

    if config.suite.includes(Suite::Get) {
        measurements.extend(bench_get_component(config)?);
    }

    if config.suite.includes(Suite::Query) {
        measurements.extend(bench_query_entity_filter(config)?);
        measurements.extend(bench_query_read_driver(config)?);
        measurements.extend(bench_query_read_filter(config)?);
        measurements.extend(bench_query_write_filter(config)?);
        measurements.extend(bench_query_changed_filter(config)?);
        measurements.extend(bench_query_read_component(config)?);
        measurements.extend(bench_query_write_component(config)?);
        measurements.extend(bench_query_oneshot(config)?);
    }

    if config.suite.includes(Suite::QueryLocality) {
        measurements.extend(bench_query_locality(config)?);
    }

    if config.suite.includes(Suite::QueryLocalityWidth) {
        measurements.extend(bench_query_locality_width(config)?);
    }

    if config.suite.includes(Suite::DespawnScaling) {
        measurements.extend(bench_despawn_scaling::<Width2>(config)?);
        measurements.extend(bench_despawn_scaling::<Width4>(config)?);
        measurements.extend(bench_despawn_scaling::<Width8>(config)?);
        measurements.extend(bench_despawn_scaling::<Width16>(config)?);
    }

    if config.suite.includes(Suite::DespawnCommands) {
        measurements.extend(bench_despawn_commands::<Width2>(config)?);
        measurements.extend(bench_despawn_commands::<Width4>(config)?);
        measurements.extend(bench_despawn_commands::<Width8>(config)?);
        measurements.extend(bench_despawn_commands::<Width16>(config)?);
    }

    if config.suite.includes(Suite::SpawnCommands) {
        measurements.extend(bench_spawn_commands::<Width2>(config)?);
        measurements.extend(bench_spawn_commands::<Width4>(config)?);
        measurements.extend(bench_spawn_commands::<Width8>(config)?);
        measurements.extend(bench_spawn_commands::<Width16>(config)?);
    }

    if config.suite.includes(Suite::MixedTransition) {
        measurements.extend(bench_mixed_transition::<Width2>(config)?);
        measurements.extend(bench_mixed_transition::<Width4>(config)?);
        measurements.extend(bench_mixed_transition::<Width8>(config)?);
        measurements.extend(bench_mixed_transition::<Width16>(config)?);
    }

    if config.suite.includes(Suite::SparseDriverTransition) {
        measurements.extend(bench_sparse_driver_transition(config)?);
    }

    if config.suite.includes(Suite::ParExecutor) {
        measurements.extend(bench_par_executor(config)?);
    }

    if config.suite.includes(Suite::Width) {
        measurements.extend(bench_width::<Width2>(config)?);
        measurements.extend(bench_width::<Width4>(config)?);
        measurements.extend(bench_width::<Width8>(config)?);
        measurements.extend(bench_width::<Width16>(config)?);
    }

    print_results(&measurements);

    if let Some(source) = output.comparison.as_ref() {
        let reference = resolve_comparison_path(source)?;
        let stored = read_benchmark_csv(&reference)?;
        print_benchmark_comparison(&measurements, &stored, &reference);
    }

    if output.save_csv {
        let output_path =
            save_benchmark_csv(&measurements, output.output_name.as_deref(), &command)?;
        println!();
        println!("CSV saved to {}", output_path.display());
    }

    Ok(())
}

fn bench_spawn(config: Config) -> AppResult<Vec<Measurement>> {
    let rows = config.rows;

    Ok(vec![
        measure_mutating_case(
            "spawn_bundle",
            "contiguous_archetype",
            PlanInfo::operation(0),
            config,
            new_position_velocity_world,
            |world| spawn_position_velocity(world, rows, ArchetypeStorageKind::Contiguous),
        )?,
        measure_mutating_case(
            "spawn_bundle",
            "fragmented_archetype",
            PlanInfo::operation(0),
            config,
            new_position_velocity_world,
            |world| spawn_position_velocity(world, rows, ArchetypeStorageKind::Fragmented),
        )?,
        measure_mutating_case(
            "spawn_bundle",
            "sparse_set",
            PlanInfo::operation(0),
            config,
            new_sparse_extra_world,
            |world| spawn_sparse_extra(world, rows),
        )?,
    ])
}

fn bench_despawn(config: Config) -> AppResult<Vec<Measurement>> {
    Ok(vec![
        measure_mutating_case(
            "despawn",
            "contiguous_archetype",
            PlanInfo::operation(config.rows),
            config,
            || build_archetype_entities(config.rows, ArchetypeStorageKind::Contiguous),
            despawn_entities,
        )?,
        measure_mutating_case(
            "despawn",
            "fragmented_archetype",
            PlanInfo::operation(config.rows),
            config,
            || build_fragmented_entities(config.rows, config.fragmented_gap_stride),
            despawn_entities,
        )?,
        measure_mutating_case(
            "despawn",
            "sparse_set",
            PlanInfo::operation(sparse_driver_world_rows(config.rows, config.sparse_stride)),
            config,
            || build_sparse_driver_entities(config.rows, config.sparse_stride),
            despawn_entities,
        )?,
    ])
}

fn bench_insert_component(config: Config) -> AppResult<Vec<Measurement>> {
    Ok(vec![
        measure_mutating_case(
            "insert_component",
            "contiguous_archetype",
            PlanInfo::operation(config.rows),
            config,
            || build_table_insert_subject(config.rows, ArchetypeStorageKind::Contiguous, 0),
            insert_table_extra,
        )?,
        measure_mutating_case(
            "insert_component",
            "fragmented_archetype",
            PlanInfo::operation(config.rows),
            config,
            || {
                build_table_insert_subject(
                    config.rows,
                    ArchetypeStorageKind::Fragmented,
                    config.fragmented_gap_stride,
                )
            },
            insert_table_extra,
        )?,
        measure_mutating_case(
            "insert_component",
            "sparse_set",
            PlanInfo::operation(config.rows),
            config,
            || build_sparse_insert_subject(config.rows),
            insert_sparse_extra,
        )?,
    ])
}

fn bench_insert_batch(config: Config) -> AppResult<Vec<Measurement>> {
    Ok(vec![
        measure_mutating_case(
            "insert_batch",
            "contiguous_archetype",
            PlanInfo::operation(config.rows),
            config,
            || build_table_insert_batch_subject(config.rows, ArchetypeStorageKind::Contiguous, 0),
            apply_insert_batch,
        )?,
        measure_mutating_case(
            "insert_batch",
            "fragmented_archetype",
            PlanInfo::operation(config.rows),
            config,
            || {
                build_table_insert_batch_subject(
                    config.rows,
                    ArchetypeStorageKind::Fragmented,
                    config.fragmented_gap_stride,
                )
            },
            apply_insert_batch,
        )?,
        measure_mutating_case(
            "insert_batch",
            "sparse_set",
            PlanInfo::operation(config.rows),
            config,
            || build_sparse_insert_batch_subject(config.rows),
            apply_insert_batch,
        )?,
    ])
}

fn bench_remove_component(config: Config) -> AppResult<Vec<Measurement>> {
    Ok(vec![
        measure_mutating_case(
            "remove_component",
            "contiguous_archetype",
            PlanInfo::operation(config.rows),
            config,
            || build_table_extra_entities(config.rows, ArchetypeStorageKind::Contiguous, 0),
            remove_table_extra,
        )?,
        measure_mutating_case(
            "remove_component",
            "fragmented_archetype",
            PlanInfo::operation(config.rows),
            config,
            || {
                build_table_extra_entities(
                    config.rows,
                    ArchetypeStorageKind::Fragmented,
                    config.fragmented_gap_stride,
                )
            },
            remove_table_extra,
        )?,
        measure_mutating_case(
            "remove_component",
            "sparse_set",
            PlanInfo::operation(config.rows),
            config,
            || build_sparse_extra_entities(config.rows),
            remove_sparse_extra,
        )?,
    ])
}

fn bench_get_component(config: Config) -> AppResult<Vec<Measurement>> {
    let dense = build_table_extra_entities(config.rows, ArchetypeStorageKind::Contiguous, 0)?;
    let fragmented = build_table_extra_entities(
        config.rows,
        ArchetypeStorageKind::Fragmented,
        config.fragmented_gap_stride,
    )?;
    let sparse = build_sparse_extra_entities(config.rows)?;

    let dense_plan = PlanInfo::operation(dense.world.len());
    let fragmented_plan = PlanInfo::operation(fragmented.world.len());
    let sparse_plan = PlanInfo::operation(sparse.world.len());

    let dense_handle = dense
        .world
        .component_handle::<TableExtra>()
        .expect("the table component must be registered");
    let fragmented_handle = fragmented
        .world
        .component_handle::<TableExtra>()
        .expect("the table component must be registered");
    let sparse_handle = sparse
        .world
        .component_handle::<SparseExtra>()
        .expect("the sparse component must be registered");

    let dense_permuted = permute_entities(&dense.entities);
    let fragmented_permuted = permute_entities(&fragmented.entities);
    let sparse_permuted = permute_entities(&sparse.entities);

    Ok(vec![
        measure_static_case(
            "get_component",
            "contiguous_archetype",
            dense_plan,
            config,
            || get_table_extra(&dense.world, &dense.entities),
        )?,
        measure_static_case(
            "get_component",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || get_table_extra(&fragmented.world, &fragmented.entities),
        )?,
        measure_static_case("get_component", "sparse_set", sparse_plan, config, || {
            get_sparse_extra(&sparse.world, &sparse.entities)
        })?,
        measure_static_case(
            "get_prepared",
            "contiguous_archetype",
            dense_plan,
            config,
            || get_table_extra_prepared(&dense.world, &dense.entities, dense_handle),
        )?,
        measure_static_case(
            "get_prepared",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || get_table_extra_prepared(&fragmented.world, &fragmented.entities, fragmented_handle),
        )?,
        measure_static_case("get_prepared", "sparse_set", sparse_plan, config, || {
            get_sparse_extra_prepared(&sparse.world, &sparse.entities, sparse_handle)
        })?,
        measure_static_case(
            "get_prepared_permuted",
            "contiguous_archetype",
            dense_plan,
            config,
            || get_table_extra_prepared(&dense.world, &dense_permuted, dense_handle),
        )?,
        measure_static_case(
            "get_prepared_permuted",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || get_table_extra_prepared(&fragmented.world, &fragmented_permuted, fragmented_handle),
        )?,
        measure_static_case(
            "get_prepared_permuted",
            "sparse_set",
            sparse_plan,
            config,
            || get_sparse_extra_prepared(&sparse.world, &sparse_permuted, sparse_handle),
        )?,
    ])
}

fn bench_query_entity_filter(config: Config) -> AppResult<Vec<Measurement>> {
    let dense_world =
        build_archetype_entities(config.rows, ArchetypeStorageKind::Contiguous)?.world;
    let fragmented_world =
        build_fragmented_entities(config.rows, config.fragmented_gap_stride)?.world;
    let sparse_world = build_sparse_driver_entities(config.rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<Entity, ()>()?;
    let mut fragmented_state = fragmented_world.query_state_cached::<Entity, ()>()?;
    let mut sparse_state = sparse_world.query_state_cached::<Entity, With<SparseExtra>>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?;
    let fragmented_plan = query_plan_info(&fragmented_world, &fragmented_state)?;
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?;

    Ok(vec![
        measure_static_case(
            "query_entity_filter",
            "contiguous_archetype",
            dense_plan,
            config,
            || visit_entities(&dense_world, &mut dense_state),
        )?,
        measure_static_case(
            "query_entity_filter",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || visit_entities(&fragmented_world, &mut fragmented_state),
        )?,
        measure_static_case(
            "query_entity_filter",
            "sparse_driver",
            sparse_plan,
            config,
            || visit_entities(&sparse_world, &mut sparse_state),
        )?,
    ])
}

fn bench_query_read_driver(config: Config) -> AppResult<Vec<Measurement>> {
    let dense_world =
        build_table_extra_entities(config.rows, ArchetypeStorageKind::Contiguous, 0)?.world;
    let fragmented_world = build_table_extra_entities(
        config.rows,
        ArchetypeStorageKind::Fragmented,
        config.fragmented_gap_stride,
    )?
    .world;
    let sparse_world = build_sparse_driver_entities(config.rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<&TableExtra, ()>()?;
    let mut fragmented_state = fragmented_world.query_state_cached::<&TableExtra, ()>()?;
    let mut sparse_state = sparse_world.query_state_cached::<&SparseExtra, ()>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?;
    let fragmented_plan = query_plan_info(&fragmented_world, &fragmented_state)?;
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?;

    Ok(vec![
        measure_static_case(
            "query_read_driver",
            "contiguous_archetype",
            dense_plan,
            config,
            || visit_table_extra(&dense_world, &mut dense_state),
        )?,
        measure_static_case(
            "query_read_driver",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || visit_table_extra(&fragmented_world, &mut fragmented_state),
        )?,
        measure_static_case(
            "query_read_driver",
            "sparse_driver",
            sparse_plan,
            config,
            || visit_sparse_extra(&sparse_world, &mut sparse_state),
        )?,
    ])
}

fn bench_query_read_filter(config: Config) -> AppResult<Vec<Measurement>> {
    let dense_world =
        build_archetype_entities(config.rows, ArchetypeStorageKind::Contiguous)?.world;
    let fragmented_world =
        build_fragmented_entities(config.rows, config.fragmented_gap_stride)?.world;
    let sparse_world = build_sparse_driver_entities(config.rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<(&Position, &Velocity), ()>()?;
    let mut fragmented_state =
        fragmented_world.query_state_cached::<(&Position, &Velocity), ()>()?;
    let mut sparse_state =
        sparse_world.query_state_cached::<(&Position, &Velocity), With<SparseExtra>>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?;
    let fragmented_plan = query_plan_info(&fragmented_world, &fragmented_state)?;
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?;

    Ok(vec![
        measure_static_case(
            "query_read_filter",
            "contiguous_archetype",
            dense_plan,
            config,
            || visit_position_velocity(&dense_world, &mut dense_state),
        )?,
        measure_static_case(
            "query_read_filter",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || visit_position_velocity(&fragmented_world, &mut fragmented_state),
        )?,
        measure_static_case(
            "query_read_filter",
            "sparse_driver",
            sparse_plan,
            config,
            || visit_position_velocity(&sparse_world, &mut sparse_state),
        )?,
    ])
}

/// Measures a change-detection query whose window contains no marks.
///
/// All components were last ticked at world build time, so every measured run
/// matches zero rows. Before the write-access latch this scans the full
/// column; after it, cold units are skipped in O(1).
fn bench_query_changed_filter(config: Config) -> AppResult<Vec<Measurement>> {
    let dense_world =
        build_archetype_entities(config.rows, ArchetypeStorageKind::Contiguous)?.world;
    let fragmented_world =
        build_fragmented_entities(config.rows, config.fragmented_gap_stride)?.world;
    let sparse_world = build_sparse_driver_entities(config.rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<&Position, Changed<Position>>()?;
    let mut fragmented_state =
        fragmented_world.query_state_cached::<&Position, Changed<Position>>()?;
    // The sparse-driver case routes through the driver lane where change
    // filters still evaluate per row, so it doubles as the no-skip control.
    let mut sparse_state =
        sparse_world.query_state_cached::<&Position, (Changed<Position>, With<SparseExtra>)>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?;
    let fragmented_plan = query_plan_info(&fragmented_world, &fragmented_state)?;
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?;

    Ok(vec![
        measure_static_case(
            "query_changed_filter",
            "contiguous_archetype",
            dense_plan,
            config,
            || visit_position_changed(&dense_world, &mut dense_state),
        )?,
        measure_static_case(
            "query_changed_filter",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || visit_position_changed(&fragmented_world, &mut fragmented_state),
        )?,
        measure_static_case(
            "query_changed_filter",
            "sparse_driver",
            sparse_plan,
            config,
            || visit_position_changed(&sparse_world, &mut sparse_state),
        )?,
    ])
}

/// Measures the cost of repeated one-shot queries over an archetype-rich
/// world (36 archetypes, past the automatic promotion threshold).
///
/// The measured query filters for a column only the widest prefix contains,
/// so each call pays construction plus a small iteration instead of paying
/// for the rows it visits.
fn bench_query_oneshot(config: Config) -> AppResult<Vec<Measurement>> {
    let group_rows = (config.rows / 36).max(1);
    let mut world = World::new();

    world.register_table::<BenchColumn<0>>()?;
    world.register_table::<BenchColumn<1>>()?;
    world.register_table::<BenchColumn<2>>()?;
    world.register_table::<BenchColumn<3>>()?;
    world.register_table::<BenchColumn<4>>()?;
    world.register_table::<BenchColumn<5>>()?;
    world.register_table::<BenchColumn<6>>()?;
    world.register_table::<BenchColumn<7>>()?;
    world.register_table::<BenchColumn<8>>()?;
    register_position_velocity(&mut world)?;
    world.register_table::<TableExtra>()?;

    spawn_oneshot_variants::<Width1>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width2>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width3>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width4>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width5>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width6>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width7>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width8>(&mut world, group_rows)?;
    spawn_oneshot_variants::<Width9>(&mut world, group_rows)?;

    let plan = {
        let state = world.query_state_cached::<Entity, With<BenchColumn<8>>>()?;
        query_plan_info(&world, &state)?
    };

    Ok(vec![measure_static_case(
        "query_oneshot",
        "archetype_rich",
        plan,
        config,
        || oneshot_query_rich(&world),
    )?])
}

/// Spawns four archetype variants for one width prefix.
fn spawn_oneshot_variants<W: WidthBundle>(world: &mut World, rows: usize) -> AppResult<()> {
    for row in 0..rows {
        world.spawn(W::new(row))?;
    }
    for row in 0..rows {
        let entity = world.spawn(W::new(row))?;
        world.insert(entity, (Velocity { x: 1, y: 2 },))?;
    }
    for row in 0..rows {
        let entity = world.spawn(W::new(row))?;
        world.insert(entity, (TableExtra { value: 1 },))?;
    }
    for row in 0..rows {
        let entity = world.spawn(W::new(row))?;
        world.insert(entity, (Velocity { x: 1, y: 2 }, TableExtra { value: 1 }))?;
    }

    Ok(())
}

/// Runs one hundred one-shot queries, reporting the per-call cost.
fn oneshot_query_rich(world: &World) -> AppResult<BenchResult> {
    let mut items = 0usize;
    let mut checksum = 0_u64;

    for _ in 0..100 {
        let mut query = world.query::<Entity, With<BenchColumn<8>>>()?;
        query.for_each(|_| {
            checksum = checksum.wrapping_add(1);
        });
        items += 1;
    }

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

macro_rules! define_column_system {
    ($fn_name:ident, $column:literal) => {
        fn $fn_name(mut column: Query<&mut BenchColumn<$column>>) {
            for mut cell in column.iter_mut() {
                cell.value = cell.value.wrapping_add(1);
            }
        }
    };
}

define_column_system!(integrate_column_0, 0);
define_column_system!(integrate_column_1, 1);
define_column_system!(integrate_column_2, 2);
define_column_system!(integrate_column_3, 3);
define_column_system!(integrate_column_4, 4);
define_column_system!(integrate_column_5, 5);
define_column_system!(integrate_column_6, 6);
define_column_system!(integrate_column_7, 7);

/// Measures schedule dispatch under both executors.
///
/// Eight independent column systems (no shared access, no ordering) iterate
/// their own column. `heavy` gives each system real work, so the single vs
/// multi ratio shows parallel speedup; `light` shrinks the per-system body
/// until dispatch overhead (task boxing, queueing, worker handoff) becomes a
/// visible share of the frame.
fn bench_par_executor(config: Config) -> AppResult<Vec<Measurement>> {
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(4);

    let heavy_rows = config.rows;
    let light_rows = (config.rows / 50).max(16);

    Ok(vec![
        measure_schedule_case(
            "par_executor",
            "heavy_single_threaded",
            heavy_rows,
            false,
            workers,
            config,
        )?,
        measure_schedule_case(
            "par_executor",
            "heavy_multi_threaded",
            heavy_rows,
            true,
            workers,
            config,
        )?,
        measure_schedule_case(
            "par_executor",
            "light_single_threaded",
            light_rows,
            false,
            workers,
            config,
        )?,
        measure_schedule_case(
            "par_executor",
            "light_multi_threaded",
            light_rows,
            true,
            workers,
            config,
        )?,
    ])
}

/// Builds one world + schedule pair and measures a full schedule run.
fn measure_schedule_case(
    operation: &'static str,
    case: &'static str,
    rows: usize,
    multi_threaded: bool,
    workers: usize,
    config: Config,
) -> AppResult<Measurement> {
    let mut world = World::new();
    Width8::register_table(&mut world)?;

    for row in 0..rows {
        world.spawn(Width8::new(row))?;
    }

    let mut schedule = Schedule::new();
    schedule.add_system(integrate_column_0);
    schedule.add_system(integrate_column_1);
    schedule.add_system(integrate_column_2);
    schedule.add_system(integrate_column_3);
    schedule.add_system(integrate_column_4);
    schedule.add_system(integrate_column_5);
    schedule.add_system(integrate_column_6);
    schedule.add_system(integrate_column_7);

    // The executor owns persistent worker threads, so it is created once per
    // measurement instead of once per run.
    let mut executor = if multi_threaded {
        ScheduleExecutor::multi_threaded(workers)
    } else {
        ScheduleExecutor::single_threaded()
    };

    let items = rows * 8;

    measure_static_case(
        operation,
        case,
        PlanInfo::operation(items),
        config,
        move || {
            schedule.run(&mut world, &mut executor)?;

            // Column contents are engine-tested; this bench measures time only.
            Ok(BenchResult {
                items,
                checksum: black_box(1),
            })
        },
    )
}

fn bench_query_write_filter(config: Config) -> AppResult<Vec<Measurement>> {
    let mut dense_world =
        build_archetype_entities(config.rows, ArchetypeStorageKind::Contiguous)?.world;
    let mut fragmented_world =
        build_fragmented_entities(config.rows, config.fragmented_gap_stride)?.world;
    let mut sparse_world = build_sparse_driver_entities(config.rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<(&mut Position, &Velocity), ()>()?;
    let mut fragmented_state =
        fragmented_world.query_state_cached::<(&mut Position, &Velocity), ()>()?;
    let mut sparse_state =
        sparse_world.query_state_cached::<(&mut Position, &Velocity), With<SparseExtra>>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?;
    let fragmented_plan = query_plan_info(&fragmented_world, &fragmented_state)?;
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?;

    Ok(vec![
        measure_static_case(
            "query_write_filter",
            "contiguous_archetype",
            dense_plan,
            config,
            || visit_position_velocity_mut(&mut dense_world, &mut dense_state),
        )?,
        measure_static_case(
            "query_write_filter",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || visit_position_velocity_mut(&mut fragmented_world, &mut fragmented_state),
        )?,
        measure_static_case(
            "query_write_filter",
            "sparse_driver",
            sparse_plan,
            config,
            || visit_position_velocity_mut(&mut sparse_world, &mut sparse_state),
        )?,
    ])
}

fn bench_query_read_component(config: Config) -> AppResult<Vec<Measurement>> {
    let dense_world =
        build_table_extra_entities(config.rows, ArchetypeStorageKind::Contiguous, 0)?.world;
    let fragmented_world = build_table_extra_entities(
        config.rows,
        ArchetypeStorageKind::Fragmented,
        config.fragmented_gap_stride,
    )?
    .world;
    let sparse_world = build_sparse_driver_entities(config.rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<(&Position, &TableExtra), ()>()?;
    let mut fragmented_state =
        fragmented_world.query_state_cached::<(&Position, &TableExtra), ()>()?;
    let mut sparse_state = sparse_world.query_state_cached::<(&Position, &SparseExtra), ()>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?;
    let fragmented_plan = query_plan_info(&fragmented_world, &fragmented_state)?;
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?;

    Ok(vec![
        measure_static_case(
            "query_read_component",
            "contiguous_archetype",
            dense_plan,
            config,
            || visit_position_table_extra(&dense_world, &mut dense_state),
        )?,
        measure_static_case(
            "query_read_component",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || visit_position_table_extra(&fragmented_world, &mut fragmented_state),
        )?,
        measure_static_case(
            "query_read_component",
            "sparse_driver",
            sparse_plan,
            config,
            || visit_position_sparse_extra(&sparse_world, &mut sparse_state),
        )?,
    ])
}

fn bench_query_write_component(config: Config) -> AppResult<Vec<Measurement>> {
    let mut dense_world =
        build_table_extra_entities(config.rows, ArchetypeStorageKind::Contiguous, 0)?.world;
    let mut fragmented_world = build_table_extra_entities(
        config.rows,
        ArchetypeStorageKind::Fragmented,
        config.fragmented_gap_stride,
    )?
    .world;
    let mut sparse_world = build_sparse_driver_entities(config.rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<&mut TableExtra, ()>()?;
    let mut fragmented_state = fragmented_world.query_state_cached::<&mut TableExtra, ()>()?;
    let mut sparse_state = sparse_world.query_state_cached::<&mut SparseExtra, ()>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?;
    let fragmented_plan = query_plan_info(&fragmented_world, &fragmented_state)?;
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?;

    Ok(vec![
        measure_static_case(
            "query_write_component",
            "contiguous_archetype",
            dense_plan,
            config,
            || visit_table_extra_mut(&mut dense_world, &mut dense_state),
        )?,
        measure_static_case(
            "query_write_component",
            "fragmented_archetype",
            fragmented_plan,
            config,
            || visit_table_extra_mut(&mut fragmented_world, &mut fragmented_state),
        )?,
        measure_static_case(
            "query_write_component",
            "sparse_driver",
            sparse_plan,
            config,
            || visit_sparse_extra_mut(&mut sparse_world, &mut sparse_state),
        )?,
    ])
}

fn bench_query_locality(config: Config) -> AppResult<Vec<Measurement>> {
    let mut measurements = Vec::new();

    for order in [
        SparseInsertionOrder::Ordered,
        SparseInsertionOrder::Permuted,
    ] {
        if config.sparse_order.includes(order) {
            measurements.extend(bench_query_locality_order(config, order)?);
        }
    }

    Ok(measurements)
}

fn bench_query_locality_order(
    config: Config,
    order: SparseInsertionOrder,
) -> AppResult<Vec<Measurement>> {
    let world = build_sparse_locality_world(config.rows, config.sparse_density, order)?;
    let sparse_driver = world
        .component_handle::<SparseExtra>()
        .expect("SparseExtra must be registered")
        .component_id();

    let mut automatic_state =
        world.query_state_cached::<(&Position, &Velocity), With<SparseExtra>>()?;
    let mut archetype_state =
        world.query_state_cached::<(&Position, &Velocity), With<SparseExtra>>()?;
    let mut sparse_state =
        world.query_state_cached::<(&Position, &Velocity), With<SparseExtra>>()?;

    archetype_state
        .set_route_policy(QueryRoutePolicy::ForceArchetype)
        .map_err(WorldError::from)?;
    sparse_state
        .set_route_policy(QueryRoutePolicy::ForceSparseDriver(sparse_driver))
        .map_err(WorldError::from)?;

    let automatic_plan = query_plan_info(&world, &automatic_state)?;
    let archetype_plan = query_plan_info(&world, &archetype_state)?;
    let sparse_plan = query_plan_info(&world, &sparse_state)?;

    let (automatic_case, archetype_case, sparse_case) = match order {
        SparseInsertionOrder::Ordered => {
            ("automatic_ordered", "archetype_ordered", "sparse_ordered")
        }
        SparseInsertionOrder::Permuted => (
            "automatic_permuted",
            "archetype_permuted",
            "sparse_permuted",
        ),
    };

    measure_static_triplet(
        "query_locality",
        [
            (automatic_case, automatic_plan),
            (archetype_case, archetype_plan),
            (sparse_case, sparse_plan),
        ],
        config,
        || visit_position_velocity(&world, &mut automatic_state),
        || visit_position_velocity(&world, &mut archetype_state),
        || visit_position_velocity(&world, &mut sparse_state),
    )
}

fn bench_width<B>(config: Config) -> AppResult<Vec<Measurement>>
where
    B: WidthBundle,
{
    let columns = B::COLUMNS;
    let rows = config.rows;
    let mut measurements = Vec::with_capacity(45);

    measurements.push(measure_mutating_case(
        "width_spawn",
        "contiguous_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || new_width_spawn_world::<B>(WidthStorage::Table, ArchetypeStorageKind::Contiguous),
        |world| spawn_width::<B>(world, rows, ArchetypeStorageKind::Contiguous),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn",
        "fragmented_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || new_width_spawn_world::<B>(WidthStorage::Table, ArchetypeStorageKind::Fragmented),
        |world| spawn_width::<B>(world, rows, ArchetypeStorageKind::Fragmented),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn",
        "sparse_set",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || new_width_spawn_world::<B>(WidthStorage::Sparse, ArchetypeStorageKind::Contiguous),
        |world| spawn_width::<B>(world, rows, ArchetypeStorageKind::Contiguous),
    )?);

    measurements.push(measure_mutating_case(
        "width_spawn_prepared",
        "contiguous_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || {
            new_width_spawn_prepared_world::<B>(
                WidthStorage::Table,
                ArchetypeStorageKind::Contiguous,
            )
        },
        |subject| spawn_width_prepared::<B>(subject, rows),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn_prepared",
        "fragmented_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || {
            new_width_spawn_prepared_world::<B>(
                WidthStorage::Table,
                ArchetypeStorageKind::Fragmented,
            )
        },
        |subject| spawn_width_prepared::<B>(subject, rows),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn_prepared",
        "sparse_set",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || {
            new_width_spawn_prepared_world::<B>(
                WidthStorage::Sparse,
                ArchetypeStorageKind::Contiguous,
            )
        },
        |subject| spawn_width_prepared::<B>(subject, rows),
    )?);

    measurements.push(measure_mutating_case(
        "width_spawn_session",
        "contiguous_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || {
            new_width_spawn_prepared_world::<B>(
                WidthStorage::Table,
                ArchetypeStorageKind::Contiguous,
            )
        },
        |subject| spawn_width_session::<B>(subject, rows),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn_session",
        "fragmented_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || {
            new_width_spawn_prepared_world::<B>(
                WidthStorage::Table,
                ArchetypeStorageKind::Fragmented,
            )
        },
        |subject| spawn_width_session::<B>(subject, rows),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn_session",
        "sparse_set",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || {
            new_width_spawn_prepared_world::<B>(
                WidthStorage::Sparse,
                ArchetypeStorageKind::Contiguous,
            )
        },
        |subject| spawn_width_session::<B>(subject, rows),
    )?);

    measurements.push(measure_mutating_case(
        "width_spawn_batch",
        "contiguous_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || new_width_spawn_world::<B>(WidthStorage::Table, ArchetypeStorageKind::Contiguous),
        |world| spawn_width_batch::<B>(world, rows, ArchetypeStorageKind::Contiguous),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn_batch",
        "fragmented_archetype",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || new_width_spawn_world::<B>(WidthStorage::Table, ArchetypeStorageKind::Fragmented),
        |world| spawn_width_batch::<B>(world, rows, ArchetypeStorageKind::Fragmented),
    )?);
    measurements.push(measure_mutating_case(
        "width_spawn_batch",
        "sparse_set",
        PlanInfo::operation(0).with_columns(columns),
        config,
        || new_width_spawn_world::<B>(WidthStorage::Sparse, ArchetypeStorageKind::Contiguous),
        |world| spawn_width_batch::<B>(world, rows, ArchetypeStorageKind::Contiguous),
    )?);

    measurements.push(measure_mutating_case(
        "width_despawn",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || build_width_table_entities::<B>(rows, ArchetypeStorageKind::Contiguous, 0),
        despawn_entities,
    )?);
    measurements.push(measure_mutating_case(
        "width_despawn",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_table_entities::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
            )
        },
        despawn_entities,
    )?);
    measurements.push(measure_mutating_case(
        "width_despawn",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || build_width_sparse_entities::<B>(rows),
        despawn_entities,
    )?);

    measurements.push(measure_mutating_case(
        "width_despawn_batch",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || build_width_table_entities::<B>(rows, ArchetypeStorageKind::Contiguous, 0),
        despawn_entities_batch,
    )?);
    measurements.push(measure_mutating_case(
        "width_despawn_batch",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_table_entities::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
            )
        },
        despawn_entities_batch,
    )?);
    measurements.push(measure_mutating_case(
        "width_despawn_batch",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || build_width_sparse_entities::<B>(rows),
        despawn_entities_batch,
    )?);

    measurements.push(measure_mutating_case(
        "width_insert",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        insert_width_extra,
    )?);

    measurements.push(measure_mutating_case(
        "width_insert_batch",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_batch_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        apply_insert_batch,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert_batch",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_batch_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        apply_insert_batch,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert_batch",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_batch_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        apply_insert_batch,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        insert_width_extra,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        insert_width_extra,
    )?);

    measurements.push(measure_mutating_case(
        "width_insert_prepared",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        insert_width_extra_prepared,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert_prepared",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        insert_width_extra_prepared,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert_prepared",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        insert_width_extra_prepared,
    )?);

    measurements.push(measure_mutating_case(
        "width_insert_session",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        insert_width_extra_session,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert_session",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        insert_width_extra_session,
    )?);
    measurements.push(measure_mutating_case(
        "width_insert_session",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_insert_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        insert_width_extra_session,
    )?);

    measurements.push(measure_mutating_case(
        "width_remove",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        remove_width_last::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        remove_width_last::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        remove_width_last::<B>,
    )?);

    measurements.push(measure_mutating_case(
        "width_remove_prepared",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        remove_width_last_prepared::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove_prepared",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        remove_width_last_prepared::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove_prepared",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        remove_width_last_prepared::<B>,
    )?);

    measurements.push(measure_mutating_case(
        "width_remove_session",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        remove_width_last_session::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove_session",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        remove_width_last_session::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove_session",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_prepared_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        remove_width_last_session::<B>,
    )?);

    measurements.push(measure_mutating_case(
        "width_remove_batch",
        "contiguous_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Table,
            )
        },
        remove_width_last_batch::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove_batch",
        "fragmented_archetype",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_subject::<B>(
                rows,
                ArchetypeStorageKind::Fragmented,
                config.fragmented_gap_stride,
                WidthStorage::Table,
            )
        },
        remove_width_last_batch::<B>,
    )?);
    measurements.push(measure_mutating_case(
        "width_remove_batch",
        "sparse_set",
        PlanInfo::operation(rows).with_columns(columns),
        config,
        || {
            build_width_remove_subject::<B>(
                rows,
                ArchetypeStorageKind::Contiguous,
                0,
                WidthStorage::Sparse,
            )
        },
        remove_width_last_batch::<B>,
    )?);

    let dense_world =
        build_width_table_entities::<B>(rows, ArchetypeStorageKind::Contiguous, 0)?.world;
    let fragmented_world = build_width_table_entities::<B>(
        rows,
        ArchetypeStorageKind::Fragmented,
        config.fragmented_gap_stride,
    )?
    .world;
    let sparse_world = build_width_sparse_driver_entities::<B>(rows, config.sparse_stride)?.world;

    let mut dense_state = dense_world.query_state_cached::<(&BenchColumn<0>, &B::Last), ()>()?;
    let mut fragmented_state =
        fragmented_world.query_state_cached::<(&BenchColumn<0>, &B::Last), ()>()?;
    let mut sparse_state = sparse_world.query_state_cached::<(&BenchColumn<0>, &B::Last), ()>()?;

    let dense_plan = query_plan_info(&dense_world, &dense_state)?.with_columns(columns);
    let fragmented_plan =
        query_plan_info(&fragmented_world, &fragmented_state)?.with_columns(columns);
    let sparse_plan = query_plan_info(&sparse_world, &sparse_state)?.with_columns(columns);

    measurements.push(measure_static_case(
        "width_query_pair",
        "contiguous_archetype",
        dense_plan,
        config,
        || visit_width_pair(&dense_world, &mut dense_state),
    )?);
    measurements.push(measure_static_case(
        "width_query_pair",
        "fragmented_archetype",
        fragmented_plan,
        config,
        || visit_width_pair(&fragmented_world, &mut fragmented_state),
    )?);
    measurements.push(measure_static_case(
        "width_query_pair",
        "sparse_driver",
        sparse_plan,
        config,
        || visit_width_pair(&sparse_world, &mut sparse_state),
    )?);

    Ok(measurements)
}

const DESPAWN_SCALING_CASES: [(usize, &str, &str); 6] = [
    (32, "despawn_32", "despawn_batch_32"),
    (64, "despawn_64", "despawn_batch_64"),
    (256, "despawn_256", "despawn_batch_256"),
    (1_024, "despawn_1024", "despawn_batch_1024"),
    (10_000, "despawn_10000", "despawn_batch_10000"),
    (100_000, "despawn_100000", "despawn_batch_100000"),
];

/// Measures the destruction crossover without running the unrelated width
/// operations. `--rows` acts as the maximum batch size included in the sweep.
fn bench_despawn_scaling<B>(config: Config) -> AppResult<Vec<Measurement>>
where
    B: WidthBundle,
{
    let columns = B::COLUMNS;
    let case_count = DESPAWN_SCALING_CASES
        .iter()
        .filter(|(rows, _, _)| *rows <= config.rows)
        .count();
    let mut measurements = Vec::with_capacity(case_count * 6);

    for &(rows, scalar_operation, batch_operation) in &DESPAWN_SCALING_CASES {
        if rows > config.rows {
            continue;
        }

        let mut case_config = config;
        case_config.rows = rows;
        let plan = PlanInfo::operation(rows).with_columns(columns);

        measurements.push(measure_mutating_case(
            scalar_operation,
            "contiguous_archetype",
            plan,
            case_config,
            || build_width_table_entities::<B>(rows, ArchetypeStorageKind::Contiguous, 0),
            despawn_entities,
        )?);
        measurements.push(measure_mutating_case(
            batch_operation,
            "contiguous_archetype",
            plan,
            case_config,
            || build_width_table_entities::<B>(rows, ArchetypeStorageKind::Contiguous, 0),
            despawn_entities_batch,
        )?);

        measurements.push(measure_mutating_case(
            scalar_operation,
            "fragmented_archetype",
            plan,
            case_config,
            || {
                build_width_table_entities::<B>(
                    rows,
                    ArchetypeStorageKind::Fragmented,
                    config.fragmented_gap_stride,
                )
            },
            despawn_entities,
        )?);
        measurements.push(measure_mutating_case(
            batch_operation,
            "fragmented_archetype",
            plan,
            case_config,
            || {
                build_width_table_entities::<B>(
                    rows,
                    ArchetypeStorageKind::Fragmented,
                    config.fragmented_gap_stride,
                )
            },
            despawn_entities_batch,
        )?);

        measurements.push(measure_mutating_case(
            scalar_operation,
            "sparse_set",
            plan,
            case_config,
            || build_width_sparse_entities::<B>(rows),
            despawn_entities,
        )?);
        measurements.push(measure_mutating_case(
            batch_operation,
            "sparse_set",
            plan,
            case_config,
            || build_width_sparse_entities::<B>(rows),
            despawn_entities_batch,
        )?);
    }

    Ok(measurements)
}

/// Compares the public scalar and batch APIs with consecutive deferred
/// despawns, which the command queue can fuse without crossing command-order
/// boundaries.
fn bench_despawn_commands<B>(config: Config) -> AppResult<Vec<Measurement>>
where
    B: WidthBundle,
{
    let rows = config.rows;
    let plan = PlanInfo::operation(rows).with_columns(B::COLUMNS);

    Ok(vec![
        measure_mutating_case(
            "sparse_despawn_scalar",
            "sparse_set",
            plan,
            config,
            || build_width_sparse_entities::<B>(rows),
            despawn_entities,
        )?,
        measure_mutating_case(
            "sparse_commands_scalar",
            "sparse_set",
            plan,
            config,
            || build_width_sparse_command_despawn::<B>(rows, DeferredDespawnMode::Scalar),
            despawn_entities_commands,
        )?,
        measure_mutating_case(
            "sparse_commands_fused",
            "sparse_set",
            plan,
            config,
            || build_width_sparse_command_despawn::<B>(rows, DeferredDespawnMode::Fused),
            despawn_entities_commands,
        )?,
        measure_mutating_case(
            "sparse_despawn_batch",
            "sparse_set",
            plan,
            config,
            || build_width_sparse_entities::<B>(rows),
            despawn_entities_batch,
        )?,
    ])
}

/// Compares deferred scalar spawning, the explicit deferred batch API, and the
/// closest direct-world paths over the same component layouts.
fn bench_spawn_commands<B>(config: Config) -> AppResult<Vec<Measurement>>
where
    B: WidthBundle + Send + 'static,
{
    let rows = config.rows;
    let plan = PlanInfo::operation(0).with_columns(B::COLUMNS);
    let mut measurements = Vec::with_capacity(8);

    for (case, storage) in [
        ("table_bundle", WidthStorage::Table),
        ("sparse_bundle", WidthStorage::Sparse),
    ] {
        measurements.push(measure_mutating_case(
            "commands_spawn_scalar",
            case,
            plan,
            config,
            || build_width_command_spawn::<B>(rows, storage, DeferredSpawnMode::Scalar),
            spawn_entities_commands,
        )?);
        measurements.push(measure_mutating_case(
            "commands_spawn_batch",
            case,
            plan,
            config,
            || build_width_command_spawn::<B>(rows, storage, DeferredSpawnMode::Batch),
            spawn_entities_commands,
        )?);
        measurements.push(measure_mutating_case(
            "direct_spawn_session",
            case,
            plan,
            config,
            || new_width_spawn_prepared_world::<B>(storage, ArchetypeStorageKind::Contiguous),
            |subject| spawn_width_session::<B>(subject, rows),
        )?);
        measurements.push(measure_mutating_case(
            "direct_spawn_batch",
            case,
            plan,
            config,
            || new_width_spawn_world::<B>(storage, ArchetypeStorageKind::Contiguous),
            |world| spawn_width_batch::<B>(world, rows, ArchetypeStorageKind::Contiguous),
        )?);
    }

    Ok(measurements)
}

/// Measures insertion of one new table component and one new sparse component
/// into an existing table archetype of increasing width.
fn bench_mixed_transition<B>(config: Config) -> AppResult<Vec<Measurement>>
where
    B: WidthBundle,
{
    let rows = config.rows;
    let plan = PlanInfo::operation(rows).with_columns(B::COLUMNS);

    Ok(vec![
        measure_mutating_case(
            "mixed_insert_scalar",
            "contiguous_archetype",
            plan,
            config,
            || build_width_mixed_insert_subject::<B>(rows, ArchetypeStorageKind::Contiguous, 0),
            insert_mixed_extra,
        )?,
        measure_mutating_case(
            "mixed_insert_batch",
            "contiguous_archetype",
            plan,
            config,
            || {
                build_width_mixed_insert_batch_subject::<B>(
                    rows,
                    ArchetypeStorageKind::Contiguous,
                    0,
                )
            },
            apply_insert_batch,
        )?,
        measure_mutating_case(
            "mixed_insert_scalar",
            "fragmented_archetype",
            plan,
            config,
            || {
                build_width_mixed_insert_subject::<B>(
                    rows,
                    ArchetypeStorageKind::Fragmented,
                    config.fragmented_gap_stride,
                )
            },
            insert_mixed_extra,
        )?,
        measure_mutating_case(
            "mixed_insert_batch",
            "fragmented_archetype",
            plan,
            config,
            || {
                build_width_mixed_insert_batch_subject::<B>(
                    rows,
                    ArchetypeStorageKind::Fragmented,
                    config.fragmented_gap_stride,
                )
            },
            apply_insert_batch,
        )?,
    ])
}

/// Measures structural location repair with a dormant or active sparse driver.
fn bench_sparse_driver_transition(config: Config) -> AppResult<Vec<Measurement>> {
    let rows = config.rows;
    let plan = PlanInfo::operation(rows).with_columns(2);
    let cases = [
        (
            "contiguous_dormant",
            ArchetypeStorageKind::Contiguous,
            0,
            false,
        ),
        (
            "contiguous_active",
            ArchetypeStorageKind::Contiguous,
            0,
            true,
        ),
        (
            "fragmented_dormant",
            ArchetypeStorageKind::Fragmented,
            config.fragmented_gap_stride,
            false,
        ),
        (
            "fragmented_active",
            ArchetypeStorageKind::Fragmented,
            config.fragmented_gap_stride,
            true,
        ),
    ];
    let mut measurements = Vec::with_capacity(cases.len() * 2);

    for (case, storage, gap_stride, activate_driver) in cases {
        measurements.push(measure_mutating_case(
            "sparse_driver_transition_scalar",
            case,
            plan,
            config,
            || build_sparse_driver_transition_subject(rows, storage, gap_stride, activate_driver),
            insert_table_extra,
        )?);

        measurements.push(measure_mutating_case(
            "sparse_driver_transition_batch",
            case,
            plan,
            config,
            || {
                build_sparse_driver_transition_batch_subject(
                    rows,
                    storage,
                    gap_stride,
                    activate_driver,
                )
            },
            apply_insert_batch,
        )?);
    }

    Ok(measurements)
}

fn measure_static_case<F>(
    operation: &'static str,
    case: &'static str,
    plan: PlanInfo,
    config: Config,
    mut run: F,
) -> AppResult<Measurement>
where
    F: FnMut() -> AppResult<BenchResult>,
{
    let mut checksum = 0_u64;

    for _ in 0..config.warmup_runs {
        let result = run()?;
        checksum = checksum.wrapping_add(result.checksum);
    }

    let mut samples = Vec::with_capacity(config.samples);
    let mut total_items = 0usize;

    for _ in 0..config.samples {
        let started = Instant::now();

        for _ in 0..config.runs_per_sample {
            let result = run()?;
            total_items += result.items;
            checksum = checksum.wrapping_add(result.checksum);
        }

        samples.push(started.elapsed());
    }

    black_box(checksum);

    Ok(Measurement {
        operation,
        case,
        plan,
        samples,
        total_items,
        checksum,
        runs_per_sample: config.runs_per_sample,
    })
}

fn measure_mutating_case<S, Setup, Run>(
    operation: &'static str,
    case: &'static str,
    plan: PlanInfo,
    config: Config,
    mut setup: Setup,
    mut run: Run,
) -> AppResult<Measurement>
where
    Setup: FnMut() -> AppResult<S>,
    Run: FnMut(&mut S) -> AppResult<BenchResult>,
{
    let mut checksum = 0_u64;

    for _ in 0..config.warmup_runs {
        let mut subject = setup()?;
        let result = run(&mut subject)?;
        checksum = checksum.wrapping_add(result.checksum);
    }

    let mut samples = Vec::with_capacity(config.samples);
    let mut total_items = 0usize;

    for _ in 0..config.samples {
        let mut sample = Duration::ZERO;

        for _ in 0..config.runs_per_sample {
            let mut subject = setup()?;
            let started = Instant::now();
            let result = run(&mut subject)?;
            sample += started.elapsed();
            total_items += result.items;
            checksum = checksum.wrapping_add(result.checksum);
        }

        samples.push(sample);
    }

    black_box(checksum);

    Ok(Measurement {
        operation,
        case,
        plan,
        samples,
        total_items,
        checksum,
        runs_per_sample: config.runs_per_sample,
    })
}

fn new_position_velocity_world() -> AppResult<World> {
    let mut world = World::new();
    register_position_velocity(&mut world)?;
    Ok(world)
}

fn new_sparse_extra_world() -> AppResult<World> {
    let mut world = World::new();
    register_position_velocity(&mut world)?;
    world.register_sparse::<SparseExtra>()?;
    Ok(world)
}

fn new_width_table_world<B: WidthBundle>() -> AppResult<World> {
    new_width_world::<B>(WidthStorage::Table)
}

fn new_width_sparse_world<B: WidthBundle>() -> AppResult<World> {
    new_width_world::<B>(WidthStorage::Sparse)
}

fn new_width_world<B: WidthBundle>(storage: WidthStorage) -> AppResult<World> {
    let mut world = World::new();

    match storage {
        WidthStorage::Table => B::register_table(&mut world)?,
        WidthStorage::Sparse => B::register_sparse(&mut world)?,
    }

    Ok(world)
}

fn new_width_spawn_world<B: WidthBundle>(
    component_storage: WidthStorage,
    archetype_storage: ArchetypeStorageKind,
) -> AppResult<World> {
    let mut world = new_width_world::<B>(component_storage)?;
    let warm = world.spawn_with_storage(archetype_storage, B::new(0))?;
    world.despawn(warm)?;
    Ok(world)
}

fn new_width_spawn_prepared_world<B: WidthBundle>(
    component_storage: WidthStorage,
    archetype_storage: ArchetypeStorageKind,
) -> AppResult<PreparedSpawnWorld<B>> {
    let mut world = new_width_world::<B>(component_storage)?;
    let prepared = world.prepare_spawn_with_storage::<B>(archetype_storage)?;
    let warm = world.spawn_prepared(B::new(0), &prepared)?;
    world.despawn(warm)?;

    Ok(PreparedSpawnWorld { world, prepared })
}

fn register_position_velocity(world: &mut World) -> Result<(), WorldError> {
    world.register_table::<Position>()?;
    world.register_table::<Velocity>()?;
    Ok(())
}

fn spawn_position_velocity(
    world: &mut World,
    rows: usize,
    storage: ArchetypeStorageKind,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for index in 0..rows {
        let entity = world.spawn_with_storage(storage, row_bundle(index))?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    Ok(BenchResult {
        items: rows,
        checksum: black_box(checksum),
    })
}

fn spawn_sparse_extra(world: &mut World, rows: usize) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for index in 0..rows {
        let entity = world.spawn_with_storage(
            ArchetypeStorageKind::Contiguous,
            (
                row_position(index),
                row_velocity(index),
                sparse_extra(index),
            ),
        )?;
        checksum = checksum
            .wrapping_add(entity_checksum(entity))
            .wrapping_add(index as u64);
    }

    Ok(BenchResult {
        items: rows,
        checksum: black_box(checksum),
    })
}

fn spawn_width<B: WidthBundle>(
    world: &mut World,
    rows: usize,
    storage: ArchetypeStorageKind,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for row in 0..rows {
        let entity = world.spawn_with_storage(storage, B::new(row))?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    Ok(BenchResult {
        items: rows,
        checksum: black_box(checksum),
    })
}

fn spawn_width_prepared<B: WidthBundle>(
    subject: &mut PreparedSpawnWorld<B>,
    rows: usize,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for row in 0..rows {
        let entity = subject
            .world
            .spawn_prepared(B::new(row), &subject.prepared)?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    Ok(BenchResult {
        items: rows,
        checksum: black_box(checksum),
    })
}

fn spawn_width_session<B: WidthBundle>(
    subject: &mut PreparedSpawnWorld<B>,
    rows: usize,
) -> AppResult<BenchResult> {
    let prepared = subject.prepared;
    let mut session = subject.world.spawn_prepared_session(&prepared)?;
    let mut checksum = 0_u64;

    for row in 0..rows {
        let entity = session.spawn(B::new(row))?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    let items = session.finish();
    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn spawn_width_batch<B: WidthBundle>(
    world: &mut World,
    rows: usize,
    storage: ArchetypeStorageKind,
) -> AppResult<BenchResult> {
    let bundles = (0..rows).map(B::new);
    let entities = world.spawn_batch_with_storage(storage, bundles)?;
    let checksum = entities.iter().copied().fold(0_u64, |checksum, entity| {
        checksum.wrapping_add(entity_checksum(entity))
    });

    Ok(BenchResult {
        items: entities.len(),
        checksum: black_box(checksum),
    })
}

fn build_archetype_entities(rows: usize, storage: ArchetypeStorageKind) -> AppResult<EntityWorld> {
    let mut world = new_position_velocity_world()?;
    let entities = spawn_live_rows(&mut world, rows, storage, 0, row_bundle)?;

    Ok(EntityWorld { world, entities })
}

fn build_fragmented_entities(rows: usize, gap_stride: usize) -> AppResult<EntityWorld> {
    let mut world = new_position_velocity_world()?;
    let entities = spawn_live_rows(
        &mut world,
        rows,
        ArchetypeStorageKind::Fragmented,
        gap_stride,
        row_bundle,
    )?;

    Ok(EntityWorld { world, entities })
}

fn build_sparse_driver_entities(rows: usize, sparse_stride: usize) -> AppResult<EntityWorld> {
    let mut world = new_sparse_extra_world()?;
    let mut entities = Vec::with_capacity(rows);
    let mut sparse_rows = 0;
    let mut index = 0;

    while sparse_rows < rows {
        if index % sparse_stride == 0 {
            let entity = world.spawn((
                row_position(index),
                row_velocity(index),
                sparse_extra(index),
            ))?;
            entities.push(entity);
            sparse_rows += 1;
        } else {
            world.spawn(row_bundle(index))?;
        }

        index += 1;
    }

    Ok(EntityWorld { world, entities })
}

fn build_sparse_locality_world(
    rows: usize,
    density_percent: usize,
    order: SparseInsertionOrder,
) -> AppResult<World> {
    let mut world = new_sparse_extra_world()?;
    let entities = spawn_live_rows(
        &mut world,
        rows,
        ArchetypeStorageKind::Contiguous,
        0,
        row_bundle,
    )?;
    let sparse_rows = rows.saturating_mul(density_percent).div_ceil(100).min(rows);
    let mut selected = Vec::with_capacity(sparse_rows);

    for index in 0..sparse_rows {
        let row = index * rows / sparse_rows;
        selected.push((row, entities[row]));
    }

    if order == SparseInsertionOrder::Permuted {
        permute_in_place(&mut selected);
    }

    for (row, entity) in selected {
        world.insert(entity, (sparse_extra(row),))?;
    }

    Ok(world)
}

fn build_table_insert_subject(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
) -> AppResult<EntityWorld> {
    let mut world = new_position_velocity_world()?;
    world.register_table::<TableExtra>()?;
    let entities = spawn_live_rows(&mut world, rows, storage, gap_stride, row_bundle)?;

    Ok(EntityWorld { world, entities })
}

fn build_sparse_insert_subject(rows: usize) -> AppResult<EntityWorld> {
    let mut world = new_sparse_extra_world()?;
    let entities = spawn_live_rows(
        &mut world,
        rows,
        ArchetypeStorageKind::Contiguous,
        0,
        row_bundle,
    )?;

    Ok(EntityWorld { world, entities })
}

fn build_table_insert_batch_subject(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
) -> AppResult<BatchInsertWorld<(TableExtra,)>> {
    let subject = build_table_insert_subject(rows, storage, gap_stride)?;
    Ok(prepare_insert_batch(subject, |index| (table_extra(index),)))
}

fn build_sparse_insert_batch_subject(rows: usize) -> AppResult<BatchInsertWorld<(SparseExtra,)>> {
    let subject = build_sparse_insert_subject(rows)?;
    Ok(prepare_insert_batch(
        subject,
        |index| (sparse_extra(index),),
    ))
}

fn prepare_insert_batch<B, Make>(subject: EntityWorld, mut make_bundle: Make) -> BatchInsertWorld<B>
where
    B: Bundle,
    Make: FnMut(usize) -> B,
{
    let EntityWorld { world, entities } = subject;
    let mut checksum = 0_u64;
    let entries = entities
        .into_iter()
        .enumerate()
        .map(|(index, entity)| {
            checksum = checksum.wrapping_add(index as u64);
            (entity, make_bundle(index))
        })
        .collect();

    BatchInsertWorld {
        world,
        entries,
        checksum,
    }
}

fn build_table_extra_entities(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
) -> AppResult<EntityWorld> {
    let mut world = World::new();
    register_position_velocity(&mut world)?;
    world.register_table::<TableExtra>()?;
    let entities = spawn_live_rows(&mut world, rows, storage, gap_stride, |index| {
        (row_position(index), row_velocity(index), table_extra(index))
    })?;

    Ok(EntityWorld { world, entities })
}

fn build_sparse_extra_entities(rows: usize) -> AppResult<EntityWorld> {
    let mut world = new_sparse_extra_world()?;
    let entities = spawn_live_rows(
        &mut world,
        rows,
        ArchetypeStorageKind::Contiguous,
        0,
        |index| {
            (
                row_position(index),
                row_velocity(index),
                sparse_extra(index),
            )
        },
    )?;

    Ok(EntityWorld { world, entities })
}

fn build_width_table_entities<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
) -> AppResult<EntityWorld> {
    let mut world = new_width_table_world::<B>()?;
    let entities = spawn_live_rows(&mut world, rows, storage, gap_stride, B::new)?;

    Ok(EntityWorld { world, entities })
}

fn build_width_sparse_entities<B: WidthBundle>(rows: usize) -> AppResult<EntityWorld> {
    let mut world = new_width_sparse_world::<B>()?;
    let entities = spawn_live_rows(
        &mut world,
        rows,
        ArchetypeStorageKind::Contiguous,
        0,
        B::new,
    )?;

    Ok(EntityWorld { world, entities })
}

fn build_width_sparse_command_despawn<B: WidthBundle>(
    rows: usize,
    mode: DeferredDespawnMode,
) -> AppResult<CommandDespawnWorld> {
    let EntityWorld { world, entities } = build_width_sparse_entities::<B>(rows)?;
    let items = entities.len();
    let checksum = entities.iter().copied().fold(0_u64, |sum, entity| {
        sum.wrapping_add(entity_checksum(entity))
    });
    let mut schedule = Schedule::new();

    schedule.add_system(move |mut commands: Commands| {
        commands.reserve_commands(entities.len());

        match mode {
            DeferredDespawnMode::Scalar => {
                for entity in entities.iter().copied() {
                    commands.add(move |world| world.despawn(entity));
                }
            }
            DeferredDespawnMode::Fused => {
                for entity in entities.iter().copied() {
                    commands.despawn(entity);
                }
            }
        }
    });

    Ok(CommandDespawnWorld {
        world,
        schedule,
        executor: ScheduleExecutor::default(),
        items,
        checksum,
    })
}

fn build_width_command_spawn<B>(
    rows: usize,
    component_storage: WidthStorage,
    mode: DeferredSpawnMode,
) -> AppResult<CommandSpawnWorld>
where
    B: WidthBundle + Send + 'static,
{
    let world = new_width_spawn_world::<B>(component_storage, ArchetypeStorageKind::Contiguous)?;
    let mut schedule = Schedule::new();

    schedule.add_system(move |mut commands: Commands| match mode {
        DeferredSpawnMode::Scalar => {
            commands.reserve_commands(rows);

            for row in 0..rows {
                black_box(commands.spawn(B::new(row)).id());
            }
        }
        DeferredSpawnMode::Batch => {
            black_box(commands.spawn_batch((0..rows).map(B::new)));
        }
    });

    Ok(CommandSpawnWorld {
        world,
        schedule,
        executor: ScheduleExecutor::default(),
        items: rows,
    })
}

fn build_width_sparse_driver_entities<B: WidthBundle>(
    rows: usize,
    sparse_stride: usize,
) -> AppResult<EntityWorld> {
    let mut world = new_width_sparse_world::<B>()?;
    let mut entities = Vec::with_capacity(rows);
    let mut sparse_rows = 0;
    let mut row = 0;

    while sparse_rows < rows {
        if row % sparse_stride == 0 {
            entities.push(world.spawn(B::new(row))?);
            sparse_rows += 1;
        } else {
            world.spawn_empty()?;
        }

        row += 1;
    }

    Ok(EntityWorld { world, entities })
}

fn build_width_insert_subject<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    component_storage: WidthStorage,
) -> AppResult<EntityWorld> {
    let mut subject = match component_storage {
        WidthStorage::Table => build_width_table_entities::<B>(rows, storage, gap_stride)?,
        WidthStorage::Sparse => build_width_sparse_entities::<B>(rows)?,
    };

    match component_storage {
        WidthStorage::Table => subject.world.register_table::<BenchColumn<16>>()?,
        WidthStorage::Sparse => subject.world.register_sparse::<BenchColumn<16>>()?,
    };

    let warm = subject.world.spawn_with_storage(storage, B::new(rows))?;
    subject.world.insert(warm, (bench_column::<16>(rows),))?;
    subject.world.despawn(warm)?;

    Ok(subject)
}

fn build_width_insert_batch_subject<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    component_storage: WidthStorage,
) -> AppResult<BatchInsertWorld<(BenchColumn<16>,)>> {
    let subject = build_width_insert_subject::<B>(rows, storage, gap_stride, component_storage)?;
    Ok(prepare_insert_batch(subject, |row| {
        (bench_column::<16>(row),)
    }))
}

fn build_width_insert_prepared_subject<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    component_storage: WidthStorage,
) -> AppResult<PreparedInsertWorld<(BenchColumn<16>,)>> {
    let EntityWorld {
        mut world,
        entities,
    } = build_width_insert_subject::<B>(rows, storage, gap_stride, component_storage)?;
    let prepared = world.prepare_insert::<(BenchColumn<16>,)>()?;

    Ok(PreparedInsertWorld {
        world,
        entities,
        prepared,
    })
}

fn build_width_mixed_insert_subject<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
) -> AppResult<EntityWorld> {
    let mut subject = build_width_table_entities::<B>(rows, storage, gap_stride)?;
    subject.world.register_table::<TableExtra>()?;
    subject.world.register_sparse::<SparseExtra>()?;

    // Materialize and warm the exact mixed transition without changing the
    // entities that the measured operation will consume.
    let warm = subject.world.spawn_with_storage(storage, B::new(rows))?;
    subject
        .world
        .insert(warm, (table_extra(rows), sparse_extra(rows)))?;
    subject.world.despawn(warm)?;

    Ok(subject)
}

fn build_width_mixed_insert_batch_subject<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
) -> AppResult<BatchInsertWorld<(TableExtra, SparseExtra)>> {
    let subject = build_width_mixed_insert_subject::<B>(rows, storage, gap_stride)?;

    Ok(prepare_insert_batch(subject, |row| {
        (table_extra(row), sparse_extra(row))
    }))
}

fn build_sparse_driver_transition_subject(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    activate_driver: bool,
) -> AppResult<EntityWorld> {
    let mut world = new_sparse_extra_world()?;
    world.register_table::<TableExtra>()?;

    let entities = spawn_live_rows(&mut world, rows, storage, gap_stride, |row| {
        (row_position(row), row_velocity(row), sparse_extra(row))
    })?;

    // Materialize the table transition outside the measured interval.
    let warm = world.spawn_with_storage(
        storage,
        (row_position(rows), row_velocity(rows), sparse_extra(rows)),
    )?;
    world.insert(warm, (table_extra(rows),))?;
    world.despawn(warm)?;

    if activate_driver {
        // Query planning builds the lazy driver. Dropping the state keeps the
        // driver active while excluding query preparation from the timing.
        let _state = world.query_state_cached::<Entity, With<SparseExtra>>()?;
    }

    Ok(EntityWorld { world, entities })
}

fn build_sparse_driver_transition_batch_subject(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    activate_driver: bool,
) -> AppResult<BatchInsertWorld<(TableExtra,)>> {
    let subject =
        build_sparse_driver_transition_subject(rows, storage, gap_stride, activate_driver)?;

    Ok(prepare_insert_batch(subject, |row| (table_extra(row),)))
}

fn build_width_remove_subject<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    component_storage: WidthStorage,
) -> AppResult<EntityWorld> {
    let mut subject = match component_storage {
        WidthStorage::Table => build_width_table_entities::<B>(rows, storage, gap_stride)?,
        WidthStorage::Sparse => build_width_sparse_entities::<B>(rows)?,
    };

    let warm = subject.world.spawn_with_storage(storage, B::new(rows))?;
    subject.world.remove::<B::Last>(warm)?;
    subject.world.despawn(warm)?;

    Ok(subject)
}

fn build_width_remove_prepared_subject<B: WidthBundle>(
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    component_storage: WidthStorage,
) -> AppResult<PreparedRemoveWorld<B::Last>> {
    let EntityWorld { world, entities } =
        build_width_remove_subject::<B>(rows, storage, gap_stride, component_storage)?;
    let prepared = world.prepare_remove::<B::Last>()?;

    Ok(PreparedRemoveWorld {
        world,
        entities,
        prepared,
    })
}

fn build_query_locality_width_world(
    rows: usize,
    density_percent: usize,
    order: SparseInsertionOrder,
) -> AppResult<World> {
    let mut world = World::new();

    Width16::register_table(&mut world)?;
    world.register_sparse::<SparseExtra>()?;

    let mut entities = Vec::with_capacity(rows);

    for row in 0..rows {
        entities.push(world.spawn(Width16::new(row))?);
    }

    let sparse_rows = rows.saturating_mul(density_percent).div_ceil(100).min(rows);

    let mut selected = Vec::with_capacity(sparse_rows);

    for index in 0..sparse_rows {
        let row = index * rows / sparse_rows;
        selected.push((row, entities[row]));
    }

    if order == SparseInsertionOrder::Permuted {
        permute_in_place(&mut selected);
    }

    for (row, entity) in selected {
        world.insert(entity, (sparse_extra(row),))?;
    }

    Ok(world)
}

fn bench_query_locality_width(config: Config) -> AppResult<Vec<Measurement>> {
    let mut measurements = Vec::new();

    for order in [
        SparseInsertionOrder::Ordered,
        SparseInsertionOrder::Permuted,
    ] {
        if !config.sparse_order.includes(order) {
            continue;
        }

        measurements.extend(bench_query_locality_width_2(config, order)?);
        measurements.extend(bench_query_locality_width_4(config, order)?);
        measurements.extend(bench_query_locality_width_8(config, order)?);
        measurements.extend(bench_query_locality_width_12(config, order)?);
    }

    Ok(measurements)
}

fn spawn_live_rows<B, Make>(
    world: &mut World,
    rows: usize,
    storage: ArchetypeStorageKind,
    gap_stride: usize,
    mut make_bundle: Make,
) -> Result<Vec<Entity>, WorldError>
where
    B: Bundle,
    Make: FnMut(usize) -> B,
{
    let mut holes = Vec::new();
    let mut entities = Vec::with_capacity(rows);
    let mut live_rows = 0;
    let mut index = 0;

    while live_rows < rows {
        let entity = world.spawn_with_storage(storage, make_bundle(index))?;

        if gap_stride != 0 && index % gap_stride == 0 {
            holes.push(entity);
        } else {
            entities.push(entity);
            live_rows += 1;
        }

        index += 1;
    }

    for entity in holes {
        world.despawn(entity)?;
    }

    Ok(entities)
}

fn despawn_entities(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in subject.entities.iter().copied() {
        checksum = checksum.wrapping_add(entity_checksum(entity));
        subject.world.despawn(entity)?;
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn despawn_entities_batch(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let checksum = subject.entities.iter().copied().fold(0_u64, |sum, entity| {
        sum.wrapping_add(entity_checksum(entity))
    });

    let items = subject
        .world
        .despawn_batch(subject.entities.iter().copied())?;

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn despawn_entities_commands(subject: &mut CommandDespawnWorld) -> AppResult<BenchResult> {
    subject
        .schedule
        .run(&mut subject.world, &mut subject.executor)?;

    Ok(BenchResult {
        items: subject.items,
        checksum: black_box(subject.checksum),
    })
}

fn spawn_entities_commands(subject: &mut CommandSpawnWorld) -> AppResult<BenchResult> {
    subject
        .schedule
        .run(&mut subject.world, &mut subject.executor)?;

    debug_assert_eq!(subject.world.len(), subject.items);

    Ok(BenchResult {
        items: subject.items,
        checksum: black_box(subject.world.len() as u64),
    })
}

fn insert_table_extra(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for (index, entity) in subject.entities.iter().copied().enumerate() {
        subject.world.insert(entity, (table_extra(index),))?;
        checksum = checksum.wrapping_add(index as u64);
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn insert_sparse_extra(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for (index, entity) in subject.entities.iter().copied().enumerate() {
        subject.world.insert(entity, (sparse_extra(index),))?;
        checksum = checksum.wrapping_add(index as u64);
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn apply_insert_batch<B: Bundle>(subject: &mut BatchInsertWorld<B>) -> AppResult<BenchResult> {
    let entries = std::mem::take(&mut subject.entries);
    let items = entries.len();
    subject.world.insert_batch(entries)?;

    Ok(BenchResult {
        items,
        checksum: black_box(subject.checksum),
    })
}

fn remove_table_extra(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in subject.entities.iter().copied() {
        subject.world.remove::<TableExtra>(entity)?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn remove_sparse_extra(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in subject.entities.iter().copied() {
        subject.world.remove::<SparseExtra>(entity)?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn insert_width_extra(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for (row, entity) in subject.entities.iter().copied().enumerate() {
        subject.world.insert(entity, (bench_column::<16>(row),))?;
        checksum = checksum.wrapping_add(row as u64);
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn insert_width_extra_prepared(
    subject: &mut PreparedInsertWorld<(BenchColumn<16>,)>,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for (row, entity) in subject.entities.iter().copied().enumerate() {
        subject
            .world
            .insert_prepared(entity, (bench_column::<16>(row),), &mut subject.prepared)?;
        checksum = checksum.wrapping_add(row as u64);
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn insert_width_extra_session(
    subject: &mut PreparedInsertWorld<(BenchColumn<16>,)>,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;
    let mut session = subject
        .world
        .insert_prepared_session(&mut subject.prepared)?;

    for (row, entity) in subject.entities.iter().copied().enumerate() {
        session.insert(entity, (bench_column::<16>(row),))?;
        checksum = checksum.wrapping_add(row as u64);
    }

    let items = session.finish();
    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn insert_mixed_extra(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for (row, entity) in subject.entities.iter().copied().enumerate() {
        subject
            .world
            .insert(entity, (table_extra(row), sparse_extra(row)))?;
        checksum = checksum.wrapping_add(row as u64);
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn remove_width_last<B: WidthBundle>(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in subject.entities.iter().copied() {
        subject.world.remove::<B::Last>(entity)?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn remove_width_last_prepared<B: WidthBundle>(
    subject: &mut PreparedRemoveWorld<B::Last>,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in subject.entities.iter().copied() {
        subject
            .world
            .remove_prepared(entity, &mut subject.prepared)?;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    }

    Ok(BenchResult {
        items: subject.entities.len(),
        checksum: black_box(checksum),
    })
}

fn remove_width_last_session<B: WidthBundle>(
    subject: &mut PreparedRemoveWorld<B::Last>,
) -> AppResult<BenchResult> {
    let checksum = subject.entities.iter().copied().fold(0_u64, |sum, entity| {
        sum.wrapping_add(entity_checksum(entity))
    });
    let mut session = subject
        .world
        .remove_prepared_session(&mut subject.prepared)?;

    for entity in subject.entities.iter().copied() {
        session.remove(entity)?;
    }

    let items = session.finish();
    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn remove_width_last_batch<B: WidthBundle>(subject: &mut EntityWorld) -> AppResult<BenchResult> {
    let checksum = subject.entities.iter().copied().fold(0_u64, |sum, entity| {
        sum.wrapping_add(entity_checksum(entity))
    });
    let items = subject.entities.len();

    subject
        .world
        .remove_batch::<_, B::Last>(subject.entities.iter().copied())?;

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn get_table_extra(world: &World, entities: &[Entity]) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in entities.iter().copied() {
        let component = world
            .get::<TableExtra>(entity)
            .expect("the entity must contain the table component");
        checksum = checksum.wrapping_add(component.value);
    }

    Ok(BenchResult {
        items: entities.len(),
        checksum: black_box(checksum),
    })
}

fn get_table_extra_prepared(
    world: &World,
    entities: &[Entity],
    handle: ComponentHandle<TableExtra>,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in entities.iter().copied() {
        let component = world
            .get_with(entity, handle)
            .expect("the entity must contain the table component");
        checksum = checksum.wrapping_add(component.value);
    }

    Ok(BenchResult {
        items: entities.len(),
        checksum: black_box(checksum),
    })
}

fn get_sparse_extra(world: &World, entities: &[Entity]) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in entities.iter().copied() {
        let component = world
            .get::<SparseExtra>(entity)
            .expect("the entity must contain the sparse component");
        checksum = checksum.wrapping_add(component.value);
    }

    Ok(BenchResult {
        items: entities.len(),
        checksum: black_box(checksum),
    })
}

fn get_sparse_extra_prepared(
    world: &World,
    entities: &[Entity],
    handle: ComponentHandle<SparseExtra>,
) -> AppResult<BenchResult> {
    let mut checksum = 0_u64;

    for entity in entities.iter().copied() {
        let component = world
            .get_with(entity, handle)
            .expect("the entity must contain the sparse component");
        checksum = checksum.wrapping_add(component.value);
    }

    Ok(BenchResult {
        items: entities.len(),
        checksum: black_box(checksum),
    })
}

fn permute_entities(entities: &[Entity]) -> Vec<Entity> {
    let mut result = entities.to_vec();
    permute_in_place(&mut result);
    result
}

fn permute_in_place<T>(values: &mut [T]) {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;

    for index in (1..values.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        values.swap(index, state as usize % (index + 1));
    }
}

fn visit_entities<F>(world: &World, state: &mut QueryState<Entity, F>) -> AppResult<BenchResult>
where
    F: QueryFilter,
{
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|entity| {
        items += 1;
        checksum = checksum.wrapping_add(entity_checksum(entity));
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_table_extra(
    world: &World,
    state: &mut QueryState<&TableExtra, ()>,
) -> AppResult<BenchResult> {
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|extra| {
        items += 1;
        checksum = checksum.wrapping_add(extra.value);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_sparse_extra(
    world: &World,
    state: &mut QueryState<&SparseExtra, ()>,
) -> AppResult<BenchResult> {
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|extra| {
        items += 1;
        checksum = checksum.wrapping_add(extra.value);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_table_extra_mut(
    world: &mut World,
    state: &mut QueryState<&mut TableExtra, ()>,
) -> AppResult<BenchResult> {
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_mut_with(state)?;

    query.for_each_mut(|mut extra| {
        items += 1;
        extra.value = extra.value.wrapping_add(1);
        checksum = checksum.wrapping_add(extra.value);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_sparse_extra_mut(
    world: &mut World,
    state: &mut QueryState<&mut SparseExtra, ()>,
) -> AppResult<BenchResult> {
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_mut_with(state)?;

    query.for_each_mut(|mut extra| {
        items += 1;
        extra.value = extra.value.wrapping_add(1);
        checksum = checksum.wrapping_add(extra.value);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_width_pair<L>(
    world: &World,
    state: &mut QueryState<(&BenchColumn<0>, &L), ()>,
) -> AppResult<BenchResult>
where
    L: Component + BenchValue,
{
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|(first, last)| {
        items += 1;
        checksum = checksum
            .wrapping_add(first.bench_value())
            .wrapping_add(last.bench_value());
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_position_velocity<F>(
    world: &World,
    state: &mut QueryState<(&Position, &Velocity), F>,
) -> AppResult<BenchResult>
where
    F: QueryFilter,
{
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|(position, velocity)| {
        items += 1;
        checksum = checksum
            .wrapping_add(position.x)
            .wrapping_add(position.y)
            .wrapping_add(velocity.x)
            .wrapping_add(velocity.y);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_position_velocity_mut<F>(
    world: &mut World,
    state: &mut QueryState<(&mut Position, &Velocity), F>,
) -> AppResult<BenchResult>
where
    F: QueryFilter,
{
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_mut_with(state)?;

    query.for_each_mut(|(mut position, velocity)| {
        items += 1;
        position.x = position.x.wrapping_add(velocity.x);
        checksum = checksum
            .wrapping_add(position.x)
            .wrapping_add(position.y)
            .wrapping_add(velocity.x)
            .wrapping_add(velocity.y);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_position_changed<F>(
    world: &World,
    state: &mut QueryState<&Position, F>,
) -> AppResult<BenchResult>
where
    F: QueryFilter,
{
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|position| {
        items += 1;
        checksum = checksum.wrapping_add(position.x).wrapping_add(position.y);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_position_table_extra(
    world: &World,
    state: &mut QueryState<(&Position, &TableExtra), ()>,
) -> AppResult<BenchResult> {
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|(position, extra)| {
        items += 1;
        checksum = checksum
            .wrapping_add(position.x)
            .wrapping_add(position.y)
            .wrapping_add(extra.value);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn visit_position_sparse_extra(
    world: &World,
    state: &mut QueryState<(&Position, &SparseExtra), ()>,
) -> AppResult<BenchResult> {
    let mut items = 0usize;
    let mut checksum = 0_u64;
    let mut query = world.query_with(state)?;

    query.for_each(|(position, extra)| {
        items += 1;
        checksum = checksum
            .wrapping_add(position.x)
            .wrapping_add(position.y)
            .wrapping_add(extra.value);
    });

    Ok(BenchResult {
        items,
        checksum: black_box(checksum),
    })
}

fn query_plan_info<D, F>(world: &World, state: &QueryState<D, F>) -> AppResult<PlanInfo>
where
    D: QueryData,
    F: QueryFilter,
{
    let report = world.query_state_debug(state)?;

    Ok(PlanInfo {
        route: route_name(report.route),
        world_rows: world.len(),
        archetype_rows: Some(report.matched_archetype_rows),
        sparse_candidates: report.sparse_driver_candidates,
        columns: None,
    })
}

fn route_name(route: QueryRoute) -> &'static str {
    match route {
        QueryRoute::Archetype { .. } => "Archetype",
        QueryRoute::SparseDriver(_) => "SparseDriver",
    }
}

fn row_bundle(index: usize) -> (Position, Velocity) {
    (row_position(index), row_velocity(index))
}

fn row_position(index: usize) -> Position {
    Position {
        x: index as u64,
        y: index.wrapping_mul(3) as u64,
    }
}

fn row_velocity(index: usize) -> Velocity {
    Velocity {
        x: index.wrapping_mul(5) as u64,
        y: index.wrapping_mul(7) as u64,
    }
}

fn table_extra(index: usize) -> TableExtra {
    TableExtra {
        value: index.wrapping_mul(11) as u64,
    }
}

fn sparse_extra(index: usize) -> SparseExtra {
    SparseExtra {
        value: index.wrapping_mul(11) as u64,
    }
}

fn bench_column<const COLUMN: usize>(row: usize) -> BenchColumn<COLUMN> {
    BenchColumn {
        value: (row as u64)
            .wrapping_mul((COLUMN as u64).wrapping_add(1))
            .wrapping_add(COLUMN as u64),
    }
}

fn entity_checksum(entity: Entity) -> u64 {
    entity.index() as u64 ^ ((entity.generation() as u64) << 32)
}

fn sparse_driver_world_rows(rows: usize, sparse_stride: usize) -> usize {
    (rows - 1) * sparse_stride + 1
}

fn print_results(measurements: &[Measurement]) {
    println!(
        "{:<23} {:<22} {:>7} {:<13} {:>10} {:>11} {:>11} {:>12} {:>14} {:>12} {:>12} {:>12} {:>8} {:>12} {:>12} {:>12}",
        "operation",
        "case",
        "columns",
        "route",
        "world",
        "archetype",
        "sparse",
        "items/run",
        "mean/run",
        "ns/item",
        "median/item",
        "p95/item",
        "cv%",
        "min/run",
        "max/run",
        "stddev/run",
    );

    for measurement in measurements {
        println!(
            "{:<23} {:<22} {:>7} {:<13} {:>10} {:>11} {:>11} {:>12.1} {:>14} {:>12.2} {:>12.2} {:>12.2} {:>8.2} {:>12} {:>12} {:>12}",
            measurement.operation,
            measurement.case,
            optional_usize(measurement.plan.columns),
            measurement.plan.route,
            measurement.plan.world_rows,
            optional_usize(measurement.plan.archetype_rows),
            optional_usize(measurement.plan.sparse_candidates),
            measurement.items_per_run(),
            format_duration_ns(measurement.mean_run_ns()),
            measurement.ns_per_item(),
            measurement.median_ns_per_item(),
            measurement.p95_ns_per_item(),
            measurement.relative_stddev_percent(),
            format_duration_ns(measurement.min_run_ns()),
            format_duration_ns(measurement.max_run_ns()),
            format_duration_ns(measurement.stddev_run_ns()),
        );
    }

    print_ratios(measurements);
    print_get_comparison(measurements);
    print_spawn_batch_comparison(measurements);
    print_mixed_transition_comparison(measurements);
    print_width_scaling(measurements);
    print_noise_summary(measurements);

    println!();
    println!("checksum:");

    for measurement in measurements {
        println!(
            "  {:<22} {:<22} {:>7} {}",
            measurement.operation,
            measurement.case,
            optional_usize(measurement.plan.columns),
            measurement.checksum
        );
    }
}

fn print_spawn_batch_comparison(measurements: &[Measurement]) {
    let scalar: Vec<_> = measurements
        .iter()
        .filter(|measurement| measurement.operation == "width_spawn")
        .collect();

    if scalar.is_empty() {
        return;
    }

    println!();
    println!("width spawn: batch median relative to scalar spawning:");

    for baseline in scalar {
        let Some(batch) = measurements.iter().find(|measurement| {
            measurement.operation == "width_spawn_batch"
                && measurement.case == baseline.case
                && measurement.plan.columns == baseline.plan.columns
        }) else {
            continue;
        };

        let ratio = batch.median_ns_per_item() / baseline.median_ns_per_item();
        println!(
            "  {:<22} {:>2} columns {:>6.2}x ({:>+6.1}%)",
            baseline.case,
            baseline.plan.columns.unwrap_or_default(),
            ratio,
            (1.0 - ratio) * 100.0,
        );
    }
}

fn print_mixed_transition_comparison(measurements: &[Measurement]) {
    let scalar: Vec<_> = measurements
        .iter()
        .filter(|measurement| measurement.operation == "mixed_insert_scalar")
        .collect();

    if scalar.is_empty() {
        return;
    }

    println!();
    println!("mixed transition: batch median relative to scalar insertion:");

    for baseline in scalar {
        let Some(batch) = measurements.iter().find(|measurement| {
            measurement.operation == "mixed_insert_batch"
                && measurement.case == baseline.case
                && measurement.plan.columns == baseline.plan.columns
        }) else {
            continue;
        };

        let ratio = batch.median_ns_per_item() / baseline.median_ns_per_item();
        println!(
            "  {:<22} {:>2} columns {:>6.2}x ({:>+6.1}%)",
            baseline.case,
            baseline.plan.columns.unwrap_or_default(),
            ratio,
            (1.0 - ratio) * 100.0,
        );
    }
}

fn print_get_comparison(measurements: &[Measurement]) {
    let direct: Vec<_> = measurements
        .iter()
        .filter(|measurement| measurement.operation == "get_component")
        .collect();

    if direct.is_empty() {
        return;
    }

    println!();
    println!("get: median relative to direct sequential access:");

    for baseline in direct {
        let prepared = measurements.iter().find(|measurement| {
            measurement.operation == "get_prepared" && measurement.case == baseline.case
        });
        let permuted = measurements.iter().find(|measurement| {
            measurement.operation == "get_prepared_permuted" && measurement.case == baseline.case
        });

        let baseline_ns = baseline.median_ns_per_item();

        if let (Some(prepared), Some(permuted)) = (prepared, permuted) {
            let prepared_ratio = prepared.median_ns_per_item() / baseline_ns;
            let permuted_ratio = permuted.median_ns_per_item() / baseline_ns;

            println!(
                "  {:<22} prepared {:>6.2}x ({:>+6.1}%), permuted {:>6.2}x ({:>+6.1}%)",
                baseline.case,
                prepared_ratio,
                (prepared_ratio - 1.0) * 100.0,
                permuted_ratio,
                (permuted_ratio - 1.0) * 100.0,
            );
        }
    }
}

fn print_ratios(measurements: &[Measurement]) {
    let mut start = 0;

    while start < measurements.len() {
        let operation = measurements[start].operation;
        let columns = measurements[start].plan.columns;
        let mut end = start + 1;

        while end < measurements.len()
            && measurements[end].operation == operation
            && measurements[end].plan.columns == columns
        {
            end += 1;
        }

        let group = &measurements[start..end];
        let Some(base) = group
            .iter()
            .find(|measurement| measurement.case == "contiguous_archetype")
        else {
            start = end;
            continue;
        };

        println!();
        match columns {
            Some(columns) => println!(
                "{operation} ({columns} columns): median relative to contiguous_archetype:"
            ),
            None => println!("{operation}: median relative to contiguous_archetype:"),
        }

        for measurement in group {
            if measurement.case == "contiguous_archetype" {
                continue;
            }

            println!(
                "  {:<22} {:>6.2}x",
                measurement.case,
                measurement.median_ns_per_item() / base.median_ns_per_item()
            );
        }

        start = end;
    }
}

fn print_width_scaling(measurements: &[Measurement]) {
    let mut operations = Vec::new();

    for measurement in measurements {
        if measurement.plan.columns.is_some() && !operations.contains(&measurement.operation) {
            operations.push(measurement.operation);
        }
    }

    for operation in operations {
        let bases: Vec<_> = measurements
            .iter()
            .filter(|measurement| {
                measurement.operation == operation && measurement.plan.columns == Some(2)
            })
            .collect();

        if bases.is_empty() {
            continue;
        }

        println!();
        println!("{operation}: median scaling relative to 2 columns:");

        for measurement in measurements.iter().filter(|measurement| {
            measurement.operation == operation && measurement.plan.columns.is_some_and(|n| n > 2)
        }) {
            let Some(base) = bases.iter().find(|base| base.case == measurement.case) else {
                continue;
            };

            println!(
                "  {:<22} {:>2} columns {:>6.2}x",
                measurement.case,
                measurement.plan.columns.unwrap(),
                measurement.median_ns_per_item() / base.median_ns_per_item()
            );
        }
    }
}

fn print_noise_summary(measurements: &[Measurement]) {
    let noisy: Vec<_> = measurements
        .iter()
        .filter(|measurement| measurement.relative_stddev_percent() > 5.0)
        .collect();

    if noisy.is_empty() {
        return;
    }

    println!();
    println!("Warning: measurements with cv% above 5 should be repeated:");

    for measurement in noisy {
        println!(
            "  {:<23} {:<22} {:>6.2}%",
            measurement.operation,
            measurement.case,
            measurement.relative_stddev_percent(),
        );
    }
}

fn results_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(RESULTS_DIRECTORY)
        .join(BENCHMARK_NAME)
}

fn write_csv<W: Write>(
    writer: &mut W,
    command: &str,
    measurements: &[Measurement],
) -> io::Result<()> {
    writeln!(writer, "# command: {command}")?;
    writeln!(
        writer,
        "operation;case;columns;route;world_rows;archetype_rows;sparse_candidates;items_per_run;mean_ns_per_run;mean_ns_per_item;median_ns_per_run;median_ns_per_item;p95_ns_per_run;p95_ns_per_item;relative_stddev_percent;min_ns_per_run;max_ns_per_run;stddev_ns_per_run;checksum"
    )?;

    for measurement in measurements {
        writeln!(
            writer,
            "{};{};{};{};{};{};{};{:.3};{:.3};{:.3};{:.3};{:.3};{:.3};{:.3};{:.3};{:.3};{:.3};{:.3};{}",
            measurement.operation,
            measurement.case,
            optional_usize(measurement.plan.columns),
            measurement.plan.route,
            measurement.plan.world_rows,
            optional_usize(measurement.plan.archetype_rows),
            optional_usize(measurement.plan.sparse_candidates),
            measurement.items_per_run(),
            measurement.mean_run_ns(),
            measurement.ns_per_item(),
            measurement.median_run_ns(),
            measurement.median_ns_per_item(),
            measurement.p95_run_ns(),
            measurement.p95_ns_per_item(),
            measurement.relative_stddev_percent(),
            measurement.min_run_ns(),
            measurement.max_run_ns(),
            measurement.stddev_run_ns(),
            measurement.checksum,
        )?;
    }

    Ok(())
}

fn save_benchmark_csv(
    measurements: &[Measurement],
    requested_name: Option<&str>,
    command: &str,
) -> io::Result<PathBuf> {
    let directory = results_directory();
    fs::create_dir_all(&directory)?;

    let path = match requested_name {
        Some(name) => directory.join(normalize_output_name(name)?),
        None => next_dated_result_path(&directory),
    };

    let file = File::options().write(true).create_new(true).open(&path)?;
    let mut writer = BufWriter::new(file);
    write_csv(&mut writer, command, measurements)?;
    writer.flush()?;

    Ok(path)
}

fn normalize_output_name(name: &str) -> io::Result<String> {
    let name = name.trim();

    if name.is_empty()
        || name == "."
        || name == ".."
        || name.chars().any(|character| {
            matches!(
                character,
                '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*'
            )
        })
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the output name must be valid and must not contain a directory",
        ));
    }

    if name.to_ascii_lowercase().ends_with(".csv") {
        Ok(name.to_owned())
    } else {
        Ok(format!("{name}.csv"))
    }
}

fn next_dated_result_path(directory: &Path) -> PathBuf {
    let stem = utc_timestamp_file_stem();
    let initial = directory.join(format!("{stem}.csv"));

    if !initial.exists() {
        return initial;
    }

    for suffix in 2.. {
        let candidate = directory.join(format!("{stem}_{suffix}.csv"));
        if !candidate.exists() {
            return candidate;
        }
    }

    unreachable!("an available file suffix must always exist")
}

fn utc_timestamp_file_stem() -> String {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    let total_seconds = elapsed.as_secs();
    let days = (total_seconds / 86_400) as i64;
    let seconds_in_day = total_seconds % 86_400;
    let hour = seconds_in_day / 3_600;
    let minute = seconds_in_day % 3_600 / 60;
    let second = seconds_in_day % 60;
    let (year, month, day) = civil_date_from_unix_days(days);

    format!(
        "{year:04}-{month:02}-{day:02}_{hour:02}-{minute:02}-{second:02}-{:03}_UTC",
        elapsed.subsec_millis()
    )
}

fn civil_date_from_unix_days(days: i64) -> (i64, u32, u32) {
    let shifted = days + 719_468;
    let era = if shifted >= 0 {
        shifted
    } else {
        shifted - 146_096
    } / 146_097;
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_position = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_position + 2) / 5 + 1;
    let month = month_position + if month_position < 10 { 3 } else { -9 };
    year += i64::from(month <= 2);

    (year, month as u32, day as u32)
}

fn resolve_comparison_path(source: &ComparisonSource) -> io::Result<PathBuf> {
    match source {
        ComparisonSource::Latest => latest_result_path(&results_directory()),
        ComparisonSource::File(name) => {
            let supplied = PathBuf::from(name);
            let mut path = if supplied.is_absolute() || supplied.components().count() > 1 {
                supplied
            } else {
                results_directory().join(supplied)
            };

            if path.extension().is_none() {
                path.set_extension("csv");
            }

            if path.is_file() {
                Ok(path)
            } else {
                Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("reference benchmark not found: {}", path.display()),
                ))
            }
        }
    }
}

fn latest_result_path(directory: &Path) -> io::Result<PathBuf> {
    let entries = fs::read_dir(directory).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to read results directory {}: {error}",
                directory.display()
            ),
        )
    })?;

    let mut latest: Option<(SystemTime, PathBuf)> = None;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        if !path.is_file()
            || !path
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("csv"))
        {
            continue;
        }

        let modified = entry.metadata()?.modified().unwrap_or(UNIX_EPOCH);

        if latest
            .as_ref()
            .is_none_or(|(latest_date, _)| modified > *latest_date)
        {
            latest = Some((modified, path));
        }
    }

    latest.map(|(_, path)| path).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no CSV benchmarks found in directory {}",
                directory.display()
            ),
        )
    })
}

fn read_benchmark_csv(path: &Path) -> io::Result<StoredBenchmark> {
    let contents = fs::read_to_string(path)?;
    let command = contents.lines().find_map(|line| {
        line.trim_start()
            .strip_prefix("# command:")
            .map(str::trim)
            .filter(|command| !command.is_empty())
            .map(str::to_owned)
    });
    let mut lines = contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'));
    let (_, header_line) = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "the CSV file is empty"))?;
    let header_line = header_line.trim_start_matches('\u{feff}');
    let delimiter = if header_line.contains(';') {
        ';'
    } else if header_line.contains(',') {
        ','
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the CSV header does not contain a supported delimiter",
        ));
    };
    let headers: Vec<_> = header_line.split(delimiter).collect();

    let required_index = |name: &str| {
        headers
            .iter()
            .position(|header| *header == name)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing required column {name}"),
                )
            })
    };

    let operation_index = required_index("operation")?;
    let case_index = required_index("case")?;
    let columns_index = required_index("columns")?;
    let items_index = required_index("items_per_run")?;
    let route_index = headers.iter().position(|header| *header == "route");
    let (value_index, statistic) = if let Some(index) = headers
        .iter()
        .position(|header| *header == "median_ns_per_item")
    {
        (index, StoredStatistic::Median)
    } else if let Some(index) = headers
        .iter()
        .position(|header| *header == "mean_ns_per_item")
        .or_else(|| headers.iter().position(|header| *header == "ns_per_item"))
    {
        (index, StoredStatistic::Mean)
    } else {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing a per-item performance column",
        ));
    };

    let mut measurements = BTreeMap::new();

    for (line_index, line) in lines {
        let fields: Vec<_> = line.split(delimiter).collect();
        let field = |index: usize| {
            fields.get(index).copied().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("incomplete CSV row at line {}", line_index + 1),
                )
            })
        };
        let parse_number = |index: usize, name: &str| -> io::Result<f64> {
            let value = field(index)?;
            value.parse::<f64>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "invalid value for {name} at line {}: {value}",
                        line_index + 1
                    ),
                )
            })
        };

        let key = MeasurementKey {
            operation: field(operation_index)?.to_owned(),
            case: field(case_index)?.to_owned(),
            columns: field(columns_index)?.to_owned(),
        };
        let measurement = StoredMeasurement {
            route: route_index
                .map(field)
                .transpose()?
                .unwrap_or("-")
                .to_owned(),
            items_per_run: parse_number(items_index, "items_per_run")?,
            ns_per_item: parse_number(value_index, "per-item performance")?,
        };

        if measurements.insert(key, measurement).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("duplicate measurement at line {}", line_index + 1),
            ));
        }
    }

    Ok(StoredBenchmark {
        command,
        statistic,
        measurements,
    })
}

fn print_benchmark_comparison(current: &[Measurement], stored: &StoredBenchmark, reference: &Path) {
    println!();
    println!(
        "Comparison with {} using per-item {}:",
        reference.display(),
        stored.statistic.label()
    );
    if let Some(command) = stored.command.as_deref() {
        println!("Reference command: {command}");
    }
    println!(
        "{:<23} {:<22} {:>7} {:<25} {:>12} {:>12} {:>11}",
        "operation", "case", "columns", "route", "reference", "current", "improvement"
    );

    let mut compared = 0usize;
    let mut missing = 0usize;
    let mut incompatible = 0usize;

    for measurement in current {
        let Some(reference_measurement) = stored.measurements.get(&measurement.key()) else {
            missing += 1;
            continue;
        };

        let current_items = measurement.items_per_run();
        let item_tolerance = reference_measurement.items_per_run.abs().max(1.0) * 0.001;
        if (current_items - reference_measurement.items_per_run).abs() > item_tolerance {
            incompatible += 1;
            continue;
        }

        let current_ns = match stored.statistic {
            StoredStatistic::Mean => measurement.ns_per_item(),
            StoredStatistic::Median => measurement.median_ns_per_item(),
        };
        let improvement = if reference_measurement.ns_per_item == 0.0 {
            0.0
        } else {
            (reference_measurement.ns_per_item - current_ns) * 100.0
                / reference_measurement.ns_per_item
        };
        let route = if reference_measurement.route == measurement.plan.route {
            measurement.plan.route.to_owned()
        } else {
            format!(
                "{} -> {}",
                reference_measurement.route, measurement.plan.route
            )
        };

        println!(
            "{:<23} {:<22} {:>7} {:<25} {:>12.3} {:>12.3} {:>+10.2}%",
            measurement.operation,
            measurement.case,
            optional_usize(measurement.plan.columns),
            route,
            reference_measurement.ns_per_item,
            current_ns,
            improvement,
        );
        compared += 1;
    }

    println!(
        "Compared: {compared}; missing reference: {missing}; incompatible item count: {incompatible}."
    );
    println!(
        "A positive percentage indicates an improvement; a negative one indicates a regression."
    );
}

fn optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "-".to_owned())
}

fn executed_command() -> String {
    env::args()
        .map(|argument| quote_command_argument(&argument))
        .collect::<Vec<_>>()
        .join(" ")
}

fn quote_command_argument(argument: &str) -> String {
    if !argument.is_empty()
        && argument
            .chars()
            .all(|character| !character.is_whitespace() && character != '"')
    {
        return argument.to_owned();
    }

    let mut quoted = String::with_capacity(argument.len() + 2);
    quoted.push('"');

    for character in argument.chars() {
        match character {
            '"' => quoted.push_str("\\\""),
            '\n' => quoted.push_str("\\n"),
            '\r' => quoted.push_str("\\r"),
            '\t' => quoted.push_str("\\t"),
            _ => quoted.push(character),
        }
    }

    quoted.push('"');
    quoted
}

fn format_duration_ns(ns: f64) -> String {
    if ns >= 1_000_000.0 {
        format!("{:.3}ms", ns / 1_000_000.0)
    } else if ns >= 1_000.0 {
        format!("{:.3}us", ns / 1_000.0)
    } else {
        format!("{ns:.1}ns")
    }
}

fn parse_arg(name: &str, value: Option<String>) -> Result<usize, String> {
    let value = value.ok_or_else(|| format!("missing value for {name}"))?;

    value
        .parse::<usize>()
        .map_err(|_| format!("invalid value for {name}: {value}"))
}

fn help() -> &'static str {
    "Usage:
  cargo run --release --example ecs_bench_suite -- [options]

Options:
  --rows N                        Items touched by each measured run. Default: 20000
  --samples N                     Number of measured samples. Default: 8
  --runs N                        Measured runs per sample. Default: 5
  --warmup N                      Runs executed before measurement. Default: 2
  --suite NAME                    all, structural, spawn, despawn, component,
                                  insert, batch, remove, get, query,
                                  query-locality, query-locality-width,
                                  despawn-scaling, despawn-commands, spawn-commands,
                                  mixed-transition, sparse-driver-transition,
                                  width.
                                  Default: all
  --csv                           Save results to examples/results/ecs_bench_suite.
  --output-name NAME              Set the generated CSV file name; enables --csv.
  --name NAME                     Short alias for --output-name.
  --compare [FILE]                Compare with FILE, or the latest CSV if omitted.
  --sparse-stride N               One sparse-driver entity every N entities. Minimum: 2. Default: 8
  --sparse-density N              Sparse membership percentage for query-locality. Default: 75
  --sparse-order NAME             ordered, permuted, or both. Default: both
  --fragmented-gap-stride N       One gap every N fragmented rows. 0 disables gaps. Default: 4

Notes:
  - Mutating cases prepare their world outside the measured time.
  - insert_batch inputs are also built outside the measured time.
  - CSV files use a UTC date and are stored under the benchmark name by default.
  - CSV files store the executed process command in an initial # command comment.
  - --compare resolves simple file names inside the default results directory.
  - get measures direct, prepared sequential and prepared permuted access separately.
  - Query cases reuse cached QueryState, matching the usual steady-state ECS measurement.
  - sparse_driver cases create extra non-sparse entities so sparse queries return the same measured row count.
  - query-locality keeps the world size fixed and compares automatic, forced archetype,
    and forced sparse routes over exactly the same data.
  - query-locality is opt-in because it prepares separate ordered and permuted sparse layouts.
  - despawn-scaling sweeps 32, 64, 256, 1024, 10000, and 100000 rows up to --rows.
  - despawn-commands compares scalar, deferred fused, and explicit batch sparse destruction.
  - spawn-commands compares fused deferred scalar spawning with explicit deferred
    batch, direct prepared-session, and direct batch spawning.
  - mixed-transition compares scalar and component-major insertion of one Table
    and one Sparse component across source archetype widths.
  - sparse-driver-transition compares scalar and component-major table migration
    while an existing sparse component keeps dormant or active location metadata.
  - width is an opt-in diagnostic suite because it prepares many large worlds.
  - width compares ordinary, prepared, session, and batch spawn/insertion/removal paths.
  - width_query_pair always reads two columns; only the stored archetype width changes.
  - Ratios use the median. A cv% above 5 usually indicates a noisy measurement."
}
