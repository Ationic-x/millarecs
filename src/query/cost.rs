//! Route cost model for query planning.
//!
//! [`QueryCostModel`] holds one measured constant per cost class (row scans,
//! column fetches, sparse lookups, page visits, locality jumps). The planner
//! combines them to estimate the total traversal cost of each candidate route.

use crate::{
    component::{
        ArchetypeId, ComponentManager, SparseDriverBucket, SparseLaneLocality, StorageKind,
    },
    query::{QueryAccess, RouteCandidateRole},
};

/// Measured per-class costs used to compare candidate query routes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct QueryCostModel {
    /// Base cost of visiting one row during a sequential scan.
    sequential_row: u64,
    /// Cost of gathering one table column for a matched row.
    table_component: u64,
    /// Cost of one entity-to-position lookup in a sparse set.
    sparse_lookup: u64,
    /// Base cost of visiting one sparse-driver row.
    driver_row: u64,
    /// Cost of fetching the driver's own value for a matched row.
    driver_value: u64,
    /// Cost of visiting one archetype page in the contiguous lane.
    contiguous_page_visit: u64,
    /// Cost of visiting one archetype page in the fragmented lane.
    fragmented_page_visit: u64,
    /// Linear cost of spreading additional table columns across unrelated pages.
    wide_table_stream: u64,
    /// Quadratic cost of increasing the number of additional table columns.
    wide_table_fanout: u64,
    /// Cost of a same-page discontinuity between consecutive rows.
    in_page_jump: u64,
    /// Extra cost of a fragmented-lane row versus a contiguous one.
    fragmented_row: u64,
    /// Fixed cost of setting up one sparse-driver lane.
    lane_setup: u64,
}

impl Default for QueryCostModel {
    /// Returns the model with its default measured constants.
    fn default() -> Self {
        Self {
            sequential_row: 256,
            table_component: 48,
            sparse_lookup: 144,
            driver_row: 320,
            driver_value: 64,
            contiguous_page_visit: 32,
            fragmented_page_visit: 176,
            wide_table_stream: 300,
            wide_table_fanout: 50,
            in_page_jump: 32,
            fragmented_row: 24,
            lane_setup: 192,
        }
    }
}

impl QueryCostModel {
    /// Estimates an archetype scan in two stages.
    ///
    /// Sparse predicates and entity iteration are paid for every scanned row,
    /// while table components are only fetched for rows expected to pass those
    /// predicates. This mirrors execution, where filters run before data fetches.
    pub(crate) fn archetype_cost(
        self,
        scanned_rows: usize,
        estimated_matching_rows: u64,
        shape: QueryShape,
    ) -> u64 {
        let scanned_rows = scanned_rows as u64;
        let matching_rows = estimated_matching_rows.min(scanned_rows);
        let scan_cost = self
            .sequential_row
            .saturating_add(shape.sparse_components.saturating_mul(self.sparse_lookup));
        let fetch_cost = shape.table_components.saturating_mul(self.table_component);

        scanned_rows
            .saturating_mul(scan_cost)
            .saturating_add(matching_rows.saturating_mul(fetch_cost))
    }

    /// Estimates the total cost of traversing a sparse-driver route.
    ///
    /// Per-row work combines the driver row, its access mode (`RequiredFilter`
    /// skips the value fetch while `ChangeFilter` adds a tick read), every
    /// table column, and the entity-to-position lookups the driver does not
    /// already resolve. Locality is charged on top: page visits are billed per
    /// lane and per table column, in-page jumps and fragmented rows add
    /// penalties, and each active lane pays a fixed setup. Queries fetching
    /// more table columns than the two-column baseline pay an additional
    /// wide-table term: locality pressure, `page_visits` squared over rows,
    /// times a linear stream cost plus a quadratic fanout cost in the number
    /// of extra columns. The squared, row-normalized pressure separates
    /// profiles with equal row counts: ordered membership stays near the row
    /// count while permuted membership approaches one locality penalty per
    /// row.
    pub(crate) fn sparse_cost(
        self,
        profile: SparseDriverProfile,
        shape: QueryShape,
        role: RouteCandidateRole,
    ) -> u64 {
        let driver_access = match role {
            RouteCandidateRole::RequiredFilter => 0,
            RouteCandidateRole::RequiredData => self.driver_value,
            RouteCandidateRole::ChangeFilter => self.driver_value + 32,
        };

        // The selected driver avoids one entity -> sparse-position lookup.
        let remaining_sparse = shape.sparse_components.saturating_sub(1);

        let extra_table_components = shape.table_components.saturating_sub(2);

        let per_row = self
            .driver_row
            .saturating_add(driver_access)
            .saturating_add(shape.table_components.saturating_mul(self.table_component))
            .saturating_add(remaining_sparse.saturating_mul(self.sparse_lookup));

        let rows = profile.rows().saturating_mul(per_row);

        let contiguous_pages = profile
            .contiguous
            .page_visits
            .saturating_mul(self.contiguous_page_visit)
            .saturating_mul(shape.table_components);

        let fragmented_pages = profile
            .fragmented
            .page_visits
            .saturating_mul(self.fragmented_page_visit)
            .saturating_mul(shape.table_components);

        let in_page_jumps = profile
            .contiguous
            .in_page_jumps
            .saturating_add(profile.fragmented.in_page_jumps)
            .saturating_mul(self.in_page_jump)
            .saturating_mul(shape.table_components);

        let fragmented_rows = profile
            .fragmented
            .rows
            .saturating_mul(self.fragmented_row)
            .saturating_mul(shape.table_components);

        let lane_setup = profile
            .contiguous
            .active_lanes
            .saturating_add(profile.fragmented.active_lanes)
            .saturating_mul(self.lane_setup);

        let page_visits = profile
            .contiguous
            .page_visits
            .saturating_add(profile.fragmented.page_visits);

        // Squaring and normalising the number of page transitions separates two
        // profiles with the same row count: ordered membership remains cheap,
        // while permuted membership approaches one locality penalty per row.
        let locality_pressure = page_visits
            .saturating_mul(page_visits)
            .div_ceil(profile.rows().max(1));
        let wide_table_factor = extra_table_components
            .saturating_mul(self.wide_table_stream)
            .saturating_add(
                extra_table_components
                    .saturating_mul(extra_table_components)
                    .saturating_mul(self.wide_table_fanout),
            );
        let wide_table_cost = locality_pressure.saturating_mul(wide_table_factor);

        rows.saturating_add(contiguous_pages)
            .saturating_add(fragmented_pages)
            .saturating_add(in_page_jumps)
            .saturating_add(fragmented_rows)
            .saturating_add(lane_setup)
            .saturating_add(wide_table_cost)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Aggregated locality metrics of one storage lane.
pub(crate) struct LaneCostProfile {
    /// Total member rows across buckets.
    pub(crate) rows: u64,
    /// Total archetype pages visited.
    pub(crate) page_visits: u64,
    /// Total same-page discontinuities.
    pub(crate) in_page_jumps: u64,
    /// Number of buckets contributing to this profile.
    pub(crate) active_lanes: u64,
}

impl LaneCostProfile {
    /// Merges one lane's locality into this profile; empty lanes are skipped
    /// so `active_lanes` only counts contributing buckets.
    fn add_lane(&mut self, lane: SparseLaneLocality) {
        if lane.rows == 0 {
            return;
        }

        self.rows += u64::from(lane.rows);
        self.page_visits += u64::from(lane.page_visits);
        self.in_page_jumps += u64::from(lane.in_page_jumps);
        self.active_lanes += 1;
    }
}

/// Combined lane profiles of one sparse component, for route estimation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SparseDriverProfile {
    /// Locality of the contiguous lane.
    pub(crate) contiguous: LaneCostProfile,
    /// Locality of the fragmented lane.
    pub(crate) fragmented: LaneCostProfile,
}

impl SparseDriverProfile {
    /// Builds the profile over buckets belonging to matched archetypes.
    pub(crate) fn build(
        buckets: &[SparseDriverBucket],
        matched_archetypes: &[ArchetypeId],
    ) -> Self {
        let mut profile = Self::default();

        for bucket in buckets {
            let matched = matched_archetypes
                .binary_search_by_key(&bucket.archetype_id.index(), |id| id.index())
                .is_ok();

            if !matched {
                continue;
            }

            profile.contiguous.add_lane(bucket.contiguous.locality());

            profile.fragmented.add_lane(bucket.fragmented.locality());
        }

        profile
    }

    /// Returns the total member rows across both lanes.
    #[inline]
    pub(crate) const fn rows(self) -> u64 {
        self.contiguous.rows + self.fragmented.rows
    }
}

/// Component-count shape of a query used by cost estimation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QueryShape {
    /// Components fetched from archetype table columns.
    table_components: u64,
    /// Components fetched through sparse-set lookups.
    sparse_components: u64,
}

impl QueryShape {
    /// Counts the table and sparse components declared by `access`.
    pub(crate) fn new(cm: &ComponentManager, access: &QueryAccess) -> Self {
        let mut shape = Self {
            table_components: 0,
            sparse_components: 0,
        };

        for item in access.regions().accesses() {
            match cm.storage_of(item.component()) {
                Some(StorageKind::Table) => shape.table_components += 1,
                Some(StorageKind::Sparse) => shape.sparse_components += 1,
                None => {}
            }
        }

        shape
    }
}
