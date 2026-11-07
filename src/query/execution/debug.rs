//! Diagnostics for query execution decisions.
//!
//! `QueryPlanDebug` describes the shape of a query. This module describes a
//! concrete run: which execution path was selected, how work was split, and why
//! parallel execution was or was not used.

use std::fmt;

use crate::query::execution::{
    parallel::{
        QueryParallelChoice, QueryParallelConfig, QueryParallelDecision, QueryParallelMode,
        QueryParallelReason,
    },
    work::{QueryWorkConfig, QueryWorkMode, ValidatedQueryWork},
};

/// User request that produced this execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryExecutionRequest {
    /// Ordinary sequential query execution.
    Sequential,
    /// Parallel-capable query execution. The final decision may still be
    /// sequential when the configured thresholds are not met.
    Parallel,
}

/// Specialized fetch path selected from the data/filter shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryExecutionPath {
    /// Table data with non-archetypal filters evaluated per entity.
    TableFiltered,
    /// Table data with filters that are fully decided by archetype metadata.
    TableArchetypal,
    /// Mixed table/sparse data with non-archetypal filters.
    MixedFiltered,
    /// Mixed table/sparse data with archetypal filters.
    MixedArchetypal,
}

/// Snapshot of the executable work generated for one query run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryWorkDebug {
    /// Work splitting mode used to build the units.
    pub mode: QueryWorkMode,
    /// Number of executable units produced.
    pub unit_count: usize,
    /// Sum of estimated live rows covered by all units.
    pub estimated_rows: usize,
    /// Requested row target per unit after normalization.
    pub target_rows_per_unit: usize,
    /// Requested page limit per unit after normalization.
    pub max_pages_per_unit: usize,
}

impl QueryWorkDebug {
    /// Builds the work snapshot from validated work and its splitting
    /// configuration, reporting the normalized limits.
    pub(crate) fn from_work(work: &ValidatedQueryWork, config: QueryWorkConfig) -> Self {
        let config = config.normalized();

        Self {
            mode: config.mode,
            unit_count: work.unit_count(),
            estimated_rows: work.estimated_rows(),
            target_rows_per_unit: config.target_rows_per_unit,
            max_pages_per_unit: config.max_pages_per_unit,
        }
    }
}

/// Normalized parallel policy used for one execution request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryParallelConfigDebug {
    /// Parallel selection mode.
    pub mode: QueryParallelMode,
    /// Minimum worker count required before parallel execution is allowed.
    pub min_workers: usize,
    /// Minimum work-unit count required before parallel execution is allowed.
    pub min_units: usize,
    /// Minimum estimated row count required by automatic mode.
    pub min_estimated_rows: usize,
    /// Row target used when building parallel-friendly work units.
    pub target_rows_per_unit: usize,
    /// Page limit used when building parallel-friendly work units.
    pub max_pages_per_unit: usize,
}

impl QueryParallelConfigDebug {
    /// Builds the policy snapshot from a normalized copy of the configuration.
    pub(crate) fn from_config(config: QueryParallelConfig) -> Self {
        let config = config.normalized();

        Self {
            mode: config.mode,
            min_workers: config.min_workers,
            min_units: config.min_units,
            min_estimated_rows: config.min_estimated_rows,
            target_rows_per_unit: config.target_rows_per_unit,
            max_pages_per_unit: config.max_pages_per_unit,
        }
    }
}

/// Diagnostics returned after a query execution request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryExecutionDebug {
    /// Whether the caller requested sequential or parallel-capable execution.
    pub request: QueryExecutionRequest,
    /// Concrete fetch path selected for this query.
    pub path: QueryExecutionPath,

    /// Final execution choice after evaluating the request and thresholds.
    pub selected: QueryParallelChoice,
    /// Reason for the final execution choice.
    pub reason: QueryParallelReason,

    /// Worker count visible to the query executor.
    pub worker_count: usize,
    /// Work generated for this concrete query run.
    pub work: QueryWorkDebug,
    /// Normalized parallel policy when this was a parallel-capable request.
    pub parallel_config: Option<QueryParallelConfigDebug>,
}

impl QueryExecutionDebug {
    /// Builds the report for a plain sequential request: forced-sequential
    /// choice, a single worker, and no parallel policy.
    pub(crate) fn sequential(
        path: QueryExecutionPath,
        work: &ValidatedQueryWork,
        work_config: QueryWorkConfig,
    ) -> Self {
        Self {
            request: QueryExecutionRequest::Sequential,
            path,
            selected: QueryParallelChoice::Sequential,
            reason: QueryParallelReason::ForcedSequential,
            worker_count: 1,
            work: QueryWorkDebug::from_work(work, work_config),
            parallel_config: None,
        }
    }

    /// Builds the report for a parallel request with its decision.
    pub(crate) fn parallel_request(
        path: QueryExecutionPath,
        work: &ValidatedQueryWork,
        work_config: QueryWorkConfig,
        parallel_config: QueryParallelConfig,
        decision: QueryParallelDecision,
        worker_count: usize,
    ) -> Self {
        Self {
            request: QueryExecutionRequest::Parallel,
            path,
            selected: decision.choice,
            reason: decision.reason,
            worker_count,
            work: QueryWorkDebug::from_work(work, work_config),
            parallel_config: Some(QueryParallelConfigDebug::from_config(parallel_config)),
        }
    }
}

impl fmt::Display for QueryExecutionDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QueryExecution debug")?;
        writeln!(f, "  Request:")?;
        writeln!(f, "          {:?}", self.request)?;
        writeln!(f, "  Selected:")?;
        writeln!(f, "          choice={:?}", self.selected)?;
        writeln!(f, "          reason={:?}", self.reason)?;
        writeln!(f, "          worker_count={}", self.worker_count)?;
        writeln!(f, "  Path:")?;
        writeln!(f, "          {:?}", self.path)?;
        writeln!(f, "  Work:")?;
        writeln!(f, "          mode={:?}", self.work.mode)?;
        writeln!(f, "          unit_count={}", self.work.unit_count)?;
        writeln!(f, "          estimated_rows={}", self.work.estimated_rows)?;
        writeln!(
            f,
            "          target_rows_per_unit={}",
            self.work.target_rows_per_unit
        )?;
        writeln!(
            f,
            "          max_pages_per_unit={}",
            self.work.max_pages_per_unit
        )?;

        if let Some(config) = self.parallel_config {
            writeln!(f, "  Parallel config:")?;
            writeln!(f, "          mode={:?}", config.mode)?;
            writeln!(f, "          min_workers={}", config.min_workers)?;
            writeln!(f, "          min_units={}", config.min_units)?;
            writeln!(
                f,
                "          min_estimated_rows={}",
                config.min_estimated_rows
            )?;
            writeln!(
                f,
                "          target_rows_per_unit={}",
                config.target_rows_per_unit
            )?;
            writeln!(
                f,
                "          max_pages_per_unit={}",
                config.max_pages_per_unit
            )?;
        } else {
            writeln!(f, "  Parallel config:")?;
            writeln!(f, "          none")?;
        }

        Ok(())
    }
}
