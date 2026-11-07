//! Query work partitioning.
//!
//! A query plan chooses the archetypes and storage lanes to visit. This module
//! turns that source into work units that can be consumed sequentially and
//! distributed across worker threads by the query executor.

use std::borrow::Cow;

use crate::{
    component::{
        archetype::{ArchetypeId, ArchetypeManager, ARCHETYPE_PAGE_ROWS},
        ComponentId, ComponentManager, SparseDriverLane, SparseDriverRow,
    },
    entity::metadata::ArchetypeStorageKind,
    query::{
        fetch::QueryData, filter::QueryFilter, planning::QueryPlanning,
        state::QueryArchetypeSource, QueryRoute,
    },
};

/// Strategy used to split a query source into executable work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryWorkMode {
    /// Produce one large unit per non-empty storage lane.
    Sequential,
    /// Split storage lanes into bounded groups.
    Chunked,
}

/// Discriminant of the two work unit shapes.
///
/// A coherent work split contains units of exactly one kind; validation
/// rejects mixes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum QueryWorkUnitKind {
    /// Unit covering a page range of one archetype storage lane.
    ArchetypePages,
    /// Unit covering resolved rows of one sparse driver bucket lane.
    SparseRows,
}

/// Ways a work split violates the coherence contract of `QueryWork`.
///
/// Validation runs in release mode, so each variant records the offending
/// unit index (and, for overlaps, both ranges) to pinpoint builder bugs from
/// the panic message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum QueryWorkValidationError {
    /// An archetype unit covers no pages or estimates no live rows.
    EmptyArchetypeUnit {
        unit_index: usize,
    },
    /// A sparse unit covers no resolved rows.
    EmptySparseUnit {
        unit_index: usize,
    },
    /// The split mixes archetype-page and sparse-row units.
    MixedUnitKinds {
        first: QueryWorkUnitKind,
        found: QueryWorkUnitKind,
        unit_index: usize,
    },
    /// Sparse units name more than one driver component.
    MultipleSparseDrivers {
        first: ComponentId,
        found: ComponentId,
        unit_index: usize,
    },
    /// The stored row estimate differs from the sum over all units.
    EstimatedRowsMismatch {
        stored: usize,
        summed: usize,
    },
    /// Two archetype units cover overlapping page ranges of the same storage
    /// lane.
    OverlappingArchetypePages {
        archetype_id: ArchetypeId,
        storage: ArchetypeStorageKind,
        previous_unit: usize,
        current_unit: usize,
        previous_start: usize,
        previous_end: usize,
        current_start: usize,
        current_end: usize,
    },
    /// Two sparse units cover overlapping row ranges of the same driver bucket
    /// lane.
    OverlappingSparseSources {
        driver: ComponentId,
        bucket: usize,
        lane: SparseDriverLaneKind,
        previous_unit: usize,
        current_unit: usize,
        previous_start: usize,
        previous_end: usize,
        current_start: usize,
        current_end: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Tunables used while building query work.
///
/// The configuration is stored in `QueryPlan` and copied into every execution
/// context. Sequential mode normalizes limits to `usize::MAX`; chunked mode
/// clamps both limits to at least one so every produced unit is valid.
pub(crate) struct QueryWorkConfig {
    /// Selected splitting strategy.
    pub(crate) mode: QueryWorkMode,
    /// Soft upper bound for live rows in one unit.
    pub(crate) target_rows_per_unit: usize,
    /// Hard upper bound for pages in one unit.
    pub(crate) max_pages_per_unit: usize,
}

impl Default for QueryWorkConfig {
    /// Returns sequential mode; the finite limits are placeholders that
    /// normalization replaces with `usize::MAX`.
    fn default() -> Self {
        Self {
            mode: QueryWorkMode::Sequential,
            target_rows_per_unit: ARCHETYPE_PAGE_ROWS * 8,
            max_pages_per_unit: 8,
        }
    }
}

impl QueryWorkConfig {
    /// Builds a configuration that keeps each storage lane as one unit.
    pub(crate) const fn sequential() -> Self {
        Self {
            mode: QueryWorkMode::Sequential,
            target_rows_per_unit: usize::MAX,
            max_pages_per_unit: usize::MAX,
        }
    }

    /// Builds a grouped configuration for parallel-friendly execution.
    pub(crate) const fn chunked(target_rows_per_unit: usize, max_pages_per_unit: usize) -> Self {
        Self {
            mode: QueryWorkMode::Chunked,
            target_rows_per_unit,
            max_pages_per_unit,
        }
    }

    /// Returns a configuration with internally valid limits.
    pub(crate) fn normalized(self) -> Self {
        match self.mode {
            QueryWorkMode::Sequential => Self::sequential(),
            QueryWorkMode::Chunked => Self {
                mode: QueryWorkMode::Chunked,
                target_rows_per_unit: self.target_rows_per_unit.max(1),
                max_pages_per_unit: self.max_pages_per_unit.max(1),
            },
        }
    }
}

/// Page-range work over one archetype storage lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ArchetypePageWork {
    pub(crate) archetype_id: ArchetypeId,
    pub(crate) storage: ArchetypeStorageKind,
    /// First page of the range, inclusive.
    page_start: u32,
    /// Past-the-last page of the range.
    page_end: u32,
    /// Estimated live rows inside the range, not its full row capacity.
    estimated_rows: u32,
}

/// Storage lane of a sparse driver bucket that a unit iterates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SparseDriverLaneKind {
    /// Contiguous-storage lane of the bucket.
    Contiguous,
    /// Fragmented-storage lane of the bucket.
    Fragmented,
}

impl SparseDriverLaneKind {
    /// Returns the archetype storage kind this lane iterates.
    #[inline]
    pub(crate) const fn storage(self) -> ArchetypeStorageKind {
        match self {
            Self::Contiguous => ArchetypeStorageKind::Contiguous,
            Self::Fragmented => ArchetypeStorageKind::Fragmented,
        }
    }
}

/// Resolved row range within one sparse driver bucket lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseRowsSource {
    pub(crate) bucket: u32,
    pub(crate) lane: SparseDriverLaneKind,
    /// First resolved row index, inclusive.
    pub(crate) start: u32,
    /// Past-the-last resolved row index.
    pub(crate) end: u32,
}

/// Sparse-driver work already resolved to one archetype and bucket lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseRowsWork {
    pub(crate) driver: ComponentId,
    pub(crate) archetype_id: ArchetypeId,
    pub(crate) source: SparseRowsSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Smallest execution unit produced for a query.
///
/// Archetype units cover page ranges. Sparse-driver units cover rows that have
/// already been resolved to one archetype and one storage lane, so execution can
/// prepare fetch state once for the whole unit.
pub(crate) enum QueryWorkUnit {
    /// Page-range unit over one archetype storage lane.
    ArchetypePages(ArchetypePageWork),
    /// Resolved-row unit over one sparse driver bucket lane.
    SparseRows(SparseRowsWork),
}

impl QueryWorkUnit {
    #[inline]
    /// Returns the estimated amount of live candidate rows covered by this unit.
    pub(crate) const fn estimated_rows(self) -> usize {
        match self {
            Self::ArchetypePages(unit) => unit.estimated_rows(),
            Self::SparseRows(unit) => unit.row_count(),
        }
    }

    #[inline]
    pub(crate) const fn kind(self) -> QueryWorkUnitKind {
        match self {
            Self::ArchetypePages(_) => QueryWorkUnitKind::ArchetypePages,
            Self::SparseRows(_) => QueryWorkUnitKind::SparseRows,
        }
    }
}

impl ArchetypePageWork {
    #[inline]
    /// Returns the amount of pages covered by this archetype unit.
    pub(crate) const fn page_count(self) -> usize {
        (self.page_end - self.page_start) as usize
    }

    #[inline]
    pub(crate) const fn page_start(self) -> usize {
        self.page_start as usize
    }

    #[inline]
    pub(crate) const fn page_end(self) -> usize {
        self.page_end as usize
    }

    #[inline]
    pub(crate) const fn estimated_rows(self) -> usize {
        self.estimated_rows as usize
    }
}

impl SparseRowsWork {
    #[inline]
    /// Returns the amount of resolved rows covered by this sparse unit.
    pub(crate) const fn row_count(self) -> usize {
        (self.source.end - self.source.start) as usize
    }

    #[inline]
    pub(crate) const fn storage(self) -> ArchetypeStorageKind {
        self.source.lane.storage()
    }
}

/// Raw two-pointer cursor over a slice of resolved sparse driver rows.
///
/// The backing slice must outlive the cursor and stay unmodified while it is
/// drained.
#[derive(Clone, Copy)]
pub(crate) struct PreparedSparseRows {
    current: *const SparseDriverRow,
    end: *const SparseDriverRow,
}

impl PreparedSparseRows {
    /// Creates a cursor over the full row slice.
    #[inline]
    pub(crate) fn indexed(rows: &[SparseDriverRow]) -> Self {
        let range = rows.as_ptr_range();
        Self {
            current: range.start,
            end: range.end,
        }
    }

    /// Returns the next resolved row, or `None` when drained.
    ///
    /// # Safety
    ///
    /// The backing slice must still be alive and unmodified.
    #[inline]
    pub(crate) unsafe fn next(&mut self) -> Option<SparseDriverRow> {
        if self.current == self.end {
            return None;
        }
        let row = unsafe { *self.current };
        self.current = unsafe { self.current.add(1) };
        Some(row)
    }

    /// Returns the candidate `offset` positions ahead of the next one.
    ///
    /// The lookahead powers gather prefetching: execution warms the cache
    /// lines of a future candidate while it still processes earlier ones.
    #[inline]
    pub(crate) unsafe fn peek(&self, offset: usize) -> Option<SparseDriverRow> {
        let remaining = unsafe { self.end.offset_from(self.current) } as usize;
        if remaining <= offset {
            return None;
        }

        Some(unsafe { *self.current.add(offset) })
    }
}

#[derive(Clone, Debug, Default)]
/// Cursor that consumes prepared query work in unit order.
pub(crate) struct QueryWorkCursor<'a> {
    units: Cow<'a, [QueryWorkUnit]>,
    next_unit: usize,
}

impl<'a> QueryWorkCursor<'a> {
    /// Creates a cursor borrowing the units of a validated work split.
    #[inline]
    pub(crate) fn borrowed(units: &'a [QueryWorkUnit]) -> Self {
        Self {
            units: Cow::Borrowed(units),
            next_unit: 0,
        }
    }

    /// Creates a cursor owning its units, detached from the work that built
    /// them.
    #[inline]
    pub(crate) fn owned(units: Vec<QueryWorkUnit>) -> Self {
        Self {
            units: Cow::Owned(units),
            next_unit: 0,
        }
    }

    /// Returns the next unit and advances the cursor.
    pub(crate) fn next_unit(&mut self) -> Option<QueryWorkUnit> {
        let unit = self.units.get(self.next_unit).copied()?;
        self.next_unit += 1;
        Some(unit)
    }
}

/// Archetype page range tagged with its unit, sorted for overlap checks.
#[derive(Clone, Copy)]
struct ArchetypeRange {
    unit_index: usize,
    archetype_id: ArchetypeId,
    storage: ArchetypeStorageKind,
    start: usize,
    end: usize,
}

/// Sparse row range tagged with its unit, sorted for overlap checks.
#[derive(Clone, Copy)]
struct IndexedSparseRange {
    unit_index: usize,
    driver: ComponentId,
    bucket: usize,
    lane: SparseDriverLaneKind,
    start: usize,
    end: usize,
}

#[derive(Clone, Debug, Default)]
/// Prepared work for one query execution.
///
/// `QueryWork` is intentionally detached from the iterators. It gives the query
/// state one place to decide how much work exists, how it is divided, and how
/// that work can later be handed to either a sequential or parallel executor.
pub(crate) struct QueryWork {
    units: Vec<QueryWorkUnit>,
    estimated_rows: usize,
}

impl QueryWork {
    /// Validates internal work coherence.
    ///
    /// This checks the contract required by parallel mutable execution: units
    /// must belong to one route kind, must not be empty, must not describe
    /// overlapping mutable regions, and their row estimates must sum to the
    /// stored total.
    pub(crate) fn validate(&self) -> Result<(), QueryWorkValidationError> {
        let mut kind = None;
        let mut summed_rows = 0;

        for (unit_index, unit) in self.units.iter().copied().enumerate() {
            let unit_kind = unit.kind();

            match kind {
                None => kind = Some(unit_kind),
                Some(first) if first != unit_kind => {
                    return Err(QueryWorkValidationError::MixedUnitKinds {
                        first,
                        found: unit_kind,
                        unit_index,
                    });
                }
                Some(_) => {}
            }

            match unit {
                QueryWorkUnit::ArchetypePages(unit) => {
                    if unit.page_count() == 0 || unit.estimated_rows() == 0 {
                        return Err(QueryWorkValidationError::EmptyArchetypeUnit { unit_index });
                    }
                    summed_rows += unit.estimated_rows();
                }
                QueryWorkUnit::SparseRows(unit) => {
                    if unit.row_count() == 0 {
                        return Err(QueryWorkValidationError::EmptySparseUnit { unit_index });
                    }

                    summed_rows += unit.row_count();
                }
            }
        }

        if summed_rows != self.estimated_rows {
            return Err(QueryWorkValidationError::EstimatedRowsMismatch {
                stored: self.estimated_rows,
                summed: summed_rows,
            });
        }

        self.validate_archetype_overlaps()?;
        self.validate_sparse_rows()?;

        Ok(())
    }

    /// Rejects archetype units whose page ranges overlap within one storage
    /// lane.
    ///
    /// Ranges are sorted by lane and position, so one adjacent-pair sweep
    /// finds any overlap.
    fn validate_archetype_overlaps(&self) -> Result<(), QueryWorkValidationError> {
        let mut ranges = Vec::new();

        for (unit_index, unit) in self.units.iter().copied().enumerate() {
            let QueryWorkUnit::ArchetypePages(unit) = unit else {
                continue;
            };

            ranges.push(ArchetypeRange {
                unit_index,
                archetype_id: unit.archetype_id,
                storage: unit.storage,
                start: unit.page_start(),
                end: unit.page_end(),
            });
        }

        ranges.sort_by_key(|range| {
            (
                range.archetype_id.index(),
                storage_key(range.storage),
                range.start,
                range.end,
            )
        });

        for pair in ranges.windows(2) {
            let previous = pair[0];
            let current = pair[1];

            if previous.archetype_id != current.archetype_id || previous.storage != current.storage
            {
                continue;
            }

            if current.start < previous.end {
                return Err(QueryWorkValidationError::OverlappingArchetypePages {
                    archetype_id: current.archetype_id,
                    storage: current.storage,
                    previous_unit: previous.unit_index,
                    current_unit: current.unit_index,
                    previous_start: previous.start,
                    previous_end: previous.end,
                    current_start: current.start,
                    current_end: current.end,
                });
            }
        }

        Ok(())
    }

    /// Rejects sparse units that name a second driver and units whose row
    /// ranges overlap within one bucket lane.
    ///
    /// Ranges are sorted by bucket lane and position, so one adjacent-pair
    /// sweep finds any overlap.
    fn validate_sparse_rows(&self) -> Result<(), QueryWorkValidationError> {
        let mut indexed_ranges = Vec::new();
        let mut first_driver = None;

        for (unit_index, unit) in self.units.iter().copied().enumerate() {
            let QueryWorkUnit::SparseRows(unit) = unit else {
                continue;
            };

            match first_driver {
                None => first_driver = Some(unit.driver),
                Some(first) if first != unit.driver => {
                    return Err(QueryWorkValidationError::MultipleSparseDrivers {
                        first,
                        found: unit.driver,
                        unit_index,
                    });
                }
                Some(_) => {}
            }

            let SparseRowsSource {
                bucket,
                lane,
                start,
                end,
            } = unit.source;

            indexed_ranges.push(IndexedSparseRange {
                unit_index,
                driver: unit.driver,
                bucket: bucket as usize,
                lane,
                start: start as usize,
                end: end as usize,
            });
        }

        indexed_ranges.sort_by_key(|range| {
            (
                range.driver.index(),
                range.bucket,
                sparse_lane_key(range.lane),
                range.start,
                range.end,
            )
        });

        for pair in indexed_ranges.windows(2) {
            let previous = pair[0];
            let current = pair[1];

            if previous.driver != current.driver
                || previous.bucket != current.bucket
                || previous.lane != current.lane
            {
                continue;
            }

            if current.start < previous.end {
                return Err(QueryWorkValidationError::OverlappingSparseSources {
                    driver: current.driver,
                    bucket: current.bucket,
                    lane: current.lane,
                    previous_unit: previous.unit_index,
                    current_unit: current.unit_index,
                    previous_start: previous.start,
                    previous_end: previous.end,
                    current_start: current.start,
                    current_end: current.end,
                });
            }
        }

        Ok(())
    }

    /// Builds fresh work from a planning without validating it.
    fn build_unvalidated_from_planning<D, F>(
        cm: &ComponentManager,
        source: QueryArchetypeSource<'_, D, F>,
        planning: &QueryPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        config: QueryWorkConfig,
    ) -> Result<Self, QueryWorkValidationError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        let mut work = Self::default();
        work.rebuild_unvalidated_from_planning(
            cm,
            source,
            planning,
            data_state,
            filter_state,
            config,
        )?;
        Ok(work)
    }

    /// Rebuilds the units in place, keeping the existing allocation, without
    /// validating them.
    fn rebuild_unvalidated_from_planning<D, F>(
        &mut self,
        cm: &ComponentManager,
        source: QueryArchetypeSource<'_, D, F>,
        planning: &QueryPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        config: QueryWorkConfig,
    ) -> Result<(), QueryWorkValidationError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        let config = config.normalized();
        self.units.clear();
        self.estimated_rows = 0;
        let route = planning.route();

        match route {
            QueryRoute::Archetype { .. } => {
                self.push_archetype_source(cm.archetypes(), source, config);
            }
            QueryRoute::SparseDriver(_) => {
                let Some(driver) = route.driver() else {
                    return Ok(());
                };

                self.append_sparse_driver_work::<D, F>(
                    cm,
                    driver,
                    data_state,
                    filter_state,
                    config,
                );
            }
        }

        Ok(())
    }

    /// Returns all units without consuming the work.
    pub(crate) fn units(&self) -> &[QueryWorkUnit] {
        &self.units
    }

    /// Returns the number of units produced for this execution.
    pub(crate) fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Returns the summed estimated live rows for all units.
    pub(crate) fn estimated_rows(&self) -> usize {
        self.estimated_rows
    }

    /// Appends one unit and accumulates its estimated rows.
    fn push_unit(&mut self, unit: QueryWorkUnit) {
        self.estimated_rows += unit.estimated_rows();
        self.units.push(unit);
    }

    /// Appends units for every archetype the source yields, covering both
    /// storage lanes of each.
    fn push_archetype_source<D, F>(
        &mut self,
        archetypes: &ArchetypeManager,
        mut source: QueryArchetypeSource<'_, D, F>,
        config: QueryWorkConfig,
    ) where
        D: QueryData,
        F: QueryFilter,
    {
        while let Some(archetype_id) = source.next(archetypes) {
            if archetypes.get_archetype(archetype_id).is_none() {
                continue;
            }

            self.push_archetype_pages(
                archetypes,
                archetype_id,
                ArchetypeStorageKind::Contiguous,
                config,
            );

            self.push_archetype_pages(
                archetypes,
                archetype_id,
                ArchetypeStorageKind::Fragmented,
                config,
            );
        }
    }

    /// Splits one archetype storage lane into units by greedy page
    /// accumulation.
    ///
    /// A unit keeps absorbing consecutive non-empty pages while it stays
    /// within the soft row target `target_rows_per_unit` and the hard page cap
    /// `max_pages_per_unit`; the page that would cross either bound starts the
    /// next unit instead. A single page alone above the target still forms its
    /// own unit, and the trailing unit is flushed after the loop.
    fn push_archetype_pages(
        &mut self,
        archetypes: &ArchetypeManager,
        archetype_id: ArchetypeId,
        storage: ArchetypeStorageKind,
        config: QueryWorkConfig,
    ) {
        let archetype = archetypes
            .get_archetype(archetype_id)
            .expect("archetype id must be valid while building query work");

        let mut page_start = None;
        let mut page_end = 0;
        let mut estimated_rows = 0;

        for page in 0..archetype.page_count(storage) {
            let Some(batch) = archetype.archetype_page_batch(storage, page) else {
                continue;
            };

            let live_rows = live_rows_in_batch(batch.row_count, batch.live_mask);

            if live_rows == 0 {
                continue;
            }

            let start = page_start.unwrap_or(page);
            let would_pages = page + 1 - start;
            let would_rows = estimated_rows + live_rows;

            if page_start.is_some()
                && (would_pages > config.max_pages_per_unit
                    || would_rows > config.target_rows_per_unit)
            {
                self.push_unit(QueryWorkUnit::ArchetypePages(ArchetypePageWork {
                    archetype_id,
                    storage,
                    page_start: compact_work_index(start, "archetype page start"),
                    page_end: compact_work_index(page_end, "archetype page end"),
                    estimated_rows: compact_work_index(estimated_rows, "estimated rows"),
                }));
                page_start = Some(page);
                page_end = page + 1;
                estimated_rows = live_rows;
            } else {
                page_start = Some(start);
                page_end = page + 1;
                estimated_rows = would_rows;
            }
        }

        if let Some(start) = page_start {
            self.push_unit(QueryWorkUnit::ArchetypePages(ArchetypePageWork {
                archetype_id,
                storage,
                page_start: compact_work_index(start, "archetype page start"),
                page_end: compact_work_index(page_end, "archetype page end"),
                estimated_rows: compact_work_index(estimated_rows, "estimated rows"),
            }));
        }
    }

    /// Returns the driver of the first sparse unit, if the work holds any.
    fn sparse_driver(&self) -> Option<ComponentId> {
        self.units.iter().find_map(|unit| match *unit {
            QueryWorkUnit::SparseRows(unit) => Some(unit.driver),
            QueryWorkUnit::ArchetypePages(_) => None,
        })
    }

    /// Appends units for every driver bucket whose archetype matches both the
    /// data and the filter, covering both bucket lanes.
    fn append_sparse_driver_work<D, F>(
        &mut self,
        cm: &ComponentManager,
        driver: ComponentId,
        data_state: &D::State,
        filter_state: &F::State,
        config: QueryWorkConfig,
    ) where
        D: QueryData,
        F: QueryFilter,
    {
        let Some(buckets) = cm.sparse_driver_buckets(driver) else {
            return;
        };

        for (bucket_index, bucket) in buckets.iter().enumerate() {
            let bucket_index = compact_work_index(bucket_index, "sparse driver bucket");
            let Some(archetype) = cm.get_archetype(bucket.archetype_id) else {
                continue;
            };

            if !D::matches_archetype(data_state, archetype)
                || !F::matches_archetype(filter_state, archetype)
            {
                continue;
            }

            self.append_indexed_sparse_row_group(
                driver,
                bucket_index,
                bucket.archetype_id,
                SparseDriverLaneKind::Contiguous,
                &bucket.contiguous,
                config,
            );

            self.append_indexed_sparse_row_group(
                driver,
                bucket_index,
                bucket.archetype_id,
                SparseDriverLaneKind::Fragmented,
                &bucket.fragmented,
                config,
            );
        }
    }

    /// Splits one bucket lane into row-chunk units.
    ///
    /// Sequential mode keeps the whole lane as a single unit; chunked mode
    /// cuts it into consecutive slices of `target_rows_per_unit` rows.
    fn append_indexed_sparse_row_group(
        &mut self,
        driver: ComponentId,
        bucket: u32,
        archetype_id: ArchetypeId,
        lane: SparseDriverLaneKind,
        rows: &SparseDriverLane,
        config: QueryWorkConfig,
    ) {
        if rows.is_empty() {
            return;
        }

        let rows = rows.rows();
        let chunk_rows = match config.mode {
            QueryWorkMode::Sequential => rows.len().max(1),
            QueryWorkMode::Chunked => config.target_rows_per_unit.max(1),
        };

        let mut start = 0;

        while start < rows.len() {
            let end = (start + chunk_rows).min(rows.len());
            self.push_unit(QueryWorkUnit::SparseRows(SparseRowsWork {
                driver,
                archetype_id,
                source: SparseRowsSource {
                    bucket,
                    lane,
                    start: compact_work_index(start, "sparse row start"),
                    end: compact_work_index(end, "sparse row end"),
                },
            }));

            start = end;
        }
    }
}

/// Query work that has passed the release-mode safety validation.
///
/// Parallel mutable execution only accepts this wrapper, not raw `QueryWork`.
/// That keeps the non-overlap contract next to the work builder instead of
/// relying on debug-only assertions at the point where raw pointers are used.
#[derive(Clone, Debug)]
pub(crate) struct ValidatedQueryWork {
    inner: QueryWork,
}

impl ValidatedQueryWork {
    /// Validates the work and wraps it, or returns the first violation found.
    pub(crate) fn new(work: QueryWork) -> Result<Self, QueryWorkValidationError> {
        work.validate()?;
        Ok(Self { inner: work })
    }

    /// Builds and validates work units from the planning selected by a query plan.
    pub(crate) fn build_from_planning<D, F>(
        cm: &ComponentManager,
        source: QueryArchetypeSource<'_, D, F>,
        planning: &QueryPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        config: QueryWorkConfig,
    ) -> Self
    where
        D: QueryData,
        F: QueryFilter,
    {
        Self::try_build_from_planning(cm, source, planning, data_state, filter_state, config)
            .unwrap_or_else(|err| {
                panic!("invalid query work produced by QueryWork builder: {err:?}")
            })
    }

    /// Builds and validates work from a planning, returning the violation
    /// instead of panicking.
    pub(crate) fn try_build_from_planning<D, F>(
        cm: &ComponentManager,
        source: QueryArchetypeSource<'_, D, F>,
        planning: &QueryPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        config: QueryWorkConfig,
    ) -> Result<Self, QueryWorkValidationError>
    where
        D: QueryData,
        F: QueryFilter,
    {
        QueryWork::build_unvalidated_from_planning(
            cm,
            source,
            planning,
            data_state,
            filter_state,
            config,
        )
        .and_then(Self::new)
    }

    /// Rebuilds the prepared units while retaining their existing allocation.
    pub(crate) fn rebuild_from_planning<D, F>(
        &mut self,
        cm: &ComponentManager,
        source: QueryArchetypeSource<'_, D, F>,
        planning: &QueryPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        config: QueryWorkConfig,
    ) where
        D: QueryData,
        F: QueryFilter,
    {
        let result = self
            .inner
            .rebuild_unvalidated_from_planning(
                cm,
                source,
                planning,
                data_state,
                filter_state,
                config,
            )
            .and_then(|()| self.inner.validate());

        if let Err(err) = result {
            // Keep the validated wrapper in a harmless state if an internal
            // builder invariant is ever violated and the panic is caught.
            self.inner.units.clear();
            self.inner.estimated_rows = 0;
            panic!("invalid query work produced by QueryWork builder: {err:?}");
        }
    }

    /// Returns a cursor that borrows the units of this work.
    #[inline]
    pub(crate) fn cursor(&self) -> QueryWorkCursor<'_> {
        QueryWorkCursor::borrowed(self.inner.units())
    }

    /// Converts this validated work into a consuming cursor.
    #[inline]
    pub(crate) fn into_cursor<'a>(self) -> QueryWorkCursor<'a> {
        QueryWorkCursor::owned(self.inner.units)
    }

    /// Returns all units without consuming the work.
    pub(crate) fn units(&self) -> &[QueryWorkUnit] {
        self.inner.units()
    }

    /// Returns the number of units produced for this execution.
    pub(crate) fn unit_count(&self) -> usize {
        self.inner.unit_count()
    }

    /// Returns the summed estimated live rows for all units.
    pub(crate) fn estimated_rows(&self) -> usize {
        self.inner.estimated_rows()
    }

    /// Returns the driver of the first sparse unit, if the work holds any.
    #[inline]
    pub(crate) fn sparse_driver(&self) -> Option<ComponentId> {
        self.inner.sparse_driver()
    }
}

/// Narrows a work index to `u32`, panicking when it exceeds the compact
/// representation.
#[inline]
fn compact_work_index(value: usize, kind: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{kind} exceeded u32 capacity"))
}

/// Numeric ordering key for archetype lanes inside overlap validation.
fn storage_key(storage: ArchetypeStorageKind) -> u8 {
    match storage {
        ArchetypeStorageKind::Contiguous => 0,
        ArchetypeStorageKind::Fragmented => 1,
    }
}

/// Numeric ordering key for sparse driver lanes inside overlap validation.
fn sparse_lane_key(lane: SparseDriverLaneKind) -> u8 {
    match lane {
        SparseDriverLaneKind::Contiguous => 0,
        SparseDriverLaneKind::Fragmented => 1,
    }
}

/// Counts live rows of one page batch against its declared row count.
fn live_rows_in_batch(row_count: usize, live_mask: u64) -> usize {
    let mask = live_mask & row_count_mask(row_count);
    mask.count_ones() as usize
}

/// Builds the liveness bit mask covering `row_count` offsets of one page.
fn row_count_mask(row_count: usize) -> u64 {
    match row_count {
        0 => 0,
        ARCHETYPE_PAGE_ROWS => u64::MAX,
        rows => (1_u64 << rows) - 1,
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    #[test]
    fn query_work_units_keep_compact_indices() {
        assert_eq!(size_of::<ArchetypePageWork>(), 20);
        assert_eq!(size_of::<SparseRowsSource>(), 16);
        assert_eq!(size_of::<SparseRowsWork>(), 24);
        assert_eq!(size_of::<QueryWorkUnit>(), 24);
        assert_eq!(size_of::<PreparedSparseRows>(), size_of::<usize>() * 2);
    }

    #[test]
    fn validated_work_rejects_overlapping_archetype_units() {
        let archetype_id = ArchetypeId::new(1);
        let work = QueryWork {
            units: vec![
                QueryWorkUnit::ArchetypePages(ArchetypePageWork {
                    archetype_id,
                    storage: ArchetypeStorageKind::Contiguous,
                    page_start: 0,
                    page_end: 2,
                    estimated_rows: 1,
                }),
                QueryWorkUnit::ArchetypePages(ArchetypePageWork {
                    archetype_id,
                    storage: ArchetypeStorageKind::Contiguous,
                    page_start: 1,
                    page_end: 3,
                    estimated_rows: 1,
                }),
            ],
            estimated_rows: 2,
        };

        assert!(matches!(
            ValidatedQueryWork::new(work),
            Err(QueryWorkValidationError::OverlappingArchetypePages {
                archetype_id: found,
                ..
            }) if found == archetype_id
        ));
    }
}
