//! Parallel query execution policy: request modes, thresholds, and the
//! decision the executor takes for one validated work split.

use crate::{
    component::archetype::ARCHETYPE_PAGE_ROWS,
    query::execution::work::{QueryWorkConfig, ValidatedQueryWork},
};

/// Requested parallelism for a query execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryParallelMode {
    /// Parallelize when the work meets the configured thresholds.
    Auto,
    /// Parallelize whenever the executor has workers available.
    Always,
    /// Always execute sequentially.
    Never,
}

/// What the executor actually selected for one execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryParallelChoice {
    /// One unit, executed on the caller thread.
    Sequential,
    /// Work split into parallel tasks.
    Parallel,
}

/// Why the executor selected its [`QueryParallelChoice`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryParallelReason {
    /// The work split contains no units.
    EmptyWork,
    /// The request mode was [`QueryParallelMode::Never`].
    ForcedSequential,
    /// The request mode was [`QueryParallelMode::Always`].
    ForcedParallel,
    /// The executor has fewer workers than the threshold.
    NotEnoughWorkers,
    /// The work split has fewer units than the threshold.
    NotEnoughUnits,
    /// The work split has fewer estimated rows than the threshold.
    NotEnoughRows,
    /// Auto mode found every threshold satisfied.
    AutoThresholdsMet,
}

/// The parallelization decision for one execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryParallelDecision {
    /// What the executor selected.
    pub choice: QueryParallelChoice,
    /// Why it selected that.
    pub reason: QueryParallelReason,
}

/// Thresholds used to decide whether and how a query parallelizes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QueryParallelConfig {
    /// Requested parallelism mode.
    pub mode: QueryParallelMode,
    /// Minimum worker count required to parallelize.
    pub min_workers: usize,
    /// Minimum unit count the work split must produce.
    pub min_units: usize,
    /// Minimum estimated rows the work must cover.
    pub min_estimated_rows: usize,
    /// Target row count of each work unit in chunked mode.
    pub target_rows_per_unit: usize,
    /// Maximum archetype pages merged into one work unit.
    pub max_pages_per_unit: usize,
}

impl Default for QueryParallelConfig {
    /// Returns the balanced automatic configuration.
    fn default() -> Self {
        Self::auto()
    }
}

impl QueryParallelConfig {
    /// Balanced defaults: parallelize from two workers and two units over at
    /// least four archetype pages of rows.
    pub const fn auto() -> Self {
        Self {
            mode: QueryParallelMode::Auto,
            min_workers: 2,
            min_units: 2,
            min_estimated_rows: ARCHETYPE_PAGE_ROWS * 4,
            target_rows_per_unit: ARCHETYPE_PAGE_ROWS * 4,
            max_pages_per_unit: 4,
        }
    }

    /// Forces parallel execution while keeping the chunking thresholds.
    pub const fn always() -> Self {
        Self {
            mode: QueryParallelMode::Always,
            min_workers: 2,
            min_units: 2,
            min_estimated_rows: 1,
            target_rows_per_unit: ARCHETYPE_PAGE_ROWS * 4,
            max_pages_per_unit: 4,
        }
    }

    /// Disables parallel execution entirely.
    pub const fn never() -> Self {
        Self {
            mode: QueryParallelMode::Never,
            min_workers: 1,
            min_units: 1,
            min_estimated_rows: 0,
            target_rows_per_unit: usize::MAX,
            max_pages_per_unit: usize::MAX,
        }
    }

    /// Returns a configuration with internally valid thresholds.
    ///
    /// `Never` collapses to the canonical never configuration; the other modes
    /// clamp the worker and unit minimums and the chunking limits to at least
    /// one, leaving the row threshold untouched.
    pub(crate) fn normalized(self) -> Self {
        match self.mode {
            QueryParallelMode::Never => Self::never(),

            QueryParallelMode::Auto | QueryParallelMode::Always => Self {
                mode: self.mode,
                min_workers: self.min_workers.max(1),
                min_units: self.min_units.max(1),
                min_estimated_rows: self.min_estimated_rows,
                target_rows_per_unit: self.target_rows_per_unit.max(1),
                max_pages_per_unit: self.max_pages_per_unit.max(1),
            },
        }
    }

    /// Translates the normalized policy into the work-splitting configuration.
    ///
    /// `Never` keeps whole storage lanes as single units; the other modes chunk
    /// toward the configured row target and page cap.
    pub(crate) fn work_config(self) -> QueryWorkConfig {
        let config = self.normalized();

        match config.mode {
            QueryParallelMode::Never => QueryWorkConfig::sequential(),
            QueryParallelMode::Auto | QueryParallelMode::Always => {
                QueryWorkConfig::chunked(config.target_rows_per_unit, config.max_pages_per_unit)
            }
        }
    }

    /// Decides how one validated work split executes on the given worker
    /// count.
    ///
    /// Empty work is always sequential. `Always` requires only the worker and
    /// unit thresholds; `Auto` additionally requires the estimated-row
    /// threshold. The returned reason names the first failing check, or the
    /// satisfied one when the choice is parallel.
    pub(crate) fn decide(
        self,
        work: &ValidatedQueryWork,
        worker_count: usize,
    ) -> QueryParallelDecision {
        let config = self.normalized();

        if work.unit_count() == 0 {
            return QueryParallelDecision {
                choice: QueryParallelChoice::Sequential,
                reason: QueryParallelReason::EmptyWork,
            };
        }

        match config.mode {
            QueryParallelMode::Never => QueryParallelDecision {
                choice: QueryParallelChoice::Sequential,
                reason: QueryParallelReason::ForcedSequential,
            },

            QueryParallelMode::Always => {
                if worker_count < config.min_workers {
                    return QueryParallelDecision {
                        choice: QueryParallelChoice::Sequential,
                        reason: QueryParallelReason::NotEnoughWorkers,
                    };
                }

                if work.unit_count() < config.min_units {
                    return QueryParallelDecision {
                        choice: QueryParallelChoice::Sequential,
                        reason: QueryParallelReason::NotEnoughUnits,
                    };
                }

                QueryParallelDecision {
                    choice: QueryParallelChoice::Parallel,
                    reason: QueryParallelReason::ForcedParallel,
                }
            }

            QueryParallelMode::Auto => {
                if worker_count < config.min_workers {
                    return QueryParallelDecision {
                        choice: QueryParallelChoice::Sequential,
                        reason: QueryParallelReason::NotEnoughWorkers,
                    };
                }

                if work.unit_count() < config.min_units {
                    return QueryParallelDecision {
                        choice: QueryParallelChoice::Sequential,
                        reason: QueryParallelReason::NotEnoughUnits,
                    };
                }

                if work.estimated_rows() < config.min_estimated_rows {
                    return QueryParallelDecision {
                        choice: QueryParallelChoice::Sequential,
                        reason: QueryParallelReason::NotEnoughRows,
                    };
                }

                QueryParallelDecision {
                    choice: QueryParallelChoice::Parallel,
                    reason: QueryParallelReason::AutoThresholdsMet,
                }
            }
        }
    }
}
