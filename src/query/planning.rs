//! Query route planning: matching archetypes, cost estimation, and route
//! selection between archetype-table and sparse-driver execution.

use crate::{
    component::{ArchetypeId, ComponentId, ComponentManager, StorageKind},
    planning::{
        ArchetypePlanningSnapshot, PlanningCostVersion, TopologyPlanningSnapshot, WorldPlanning,
    },
    query::{
        cost::{QueryCostModel, QueryShape, SparseDriverProfile},
        filter::QueryFilter,
        QueryAccess, QueryData,
    },
};

/// Controls how a query chooses its execution source.
///
/// [`QueryRoutePolicy::Automatic`] is the production policy. Forced policies
/// are primarily useful for diagnostics and controlled benchmarks that compare
/// both valid routes over the same world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueryRoutePolicy {
    /// Lets the cost model choose between archetype and sparse-driver traversal.
    #[default]
    Automatic,
    /// Always scans the matching archetypes.
    ForceArchetype,
    /// Always traverses the specified sparse component.
    ForceSparseDriver(ComponentId),
}

/// Candidate component that may drive a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryRouteCandidate {
    /// Component the candidate refers to.
    pub(crate) component: ComponentId,
    /// Storage backend registered for the component.
    pub(crate) storage: StorageKind,
    /// Role the component plays in the query declaration.
    pub(crate) role: RouteCandidateRole,
    /// Whether the declaring data source was optional; such candidates never
    /// drive.
    pub(crate) optional: bool,
    /// Whether domain reasoning may rely on this declaration.
    pub(crate) exact: bool,
}

impl QueryRouteCandidate {
    /// Builds the candidate for a required component data source.
    #[inline]
    pub const fn required_data(component: ComponentId, storage: StorageKind) -> Self {
        Self {
            component,
            storage,
            role: RouteCandidateRole::RequiredData,
            optional: false,
            exact: true,
        }
    }

    /// Builds the candidate for an optional component data source.
    #[inline]
    pub const fn optional_data(component: ComponentId, storage: StorageKind) -> Self {
        Self {
            component,
            storage,
            role: RouteCandidateRole::RequiredData,
            optional: true,
            exact: false,
        }
    }

    /// Builds the candidate for a required `With<C>` filter.
    #[inline]
    pub const fn required_filter(component: ComponentId, storage: StorageKind) -> Self {
        Self {
            component,
            storage,
            role: RouteCandidateRole::RequiredFilter,
            optional: false,
            exact: true,
        }
    }

    /// Builds the candidate for an `Added<C>` / `Changed<C>` filter.
    #[inline]
    pub const fn change_filter(component: ComponentId, storage: StorageKind) -> Self {
        Self {
            component,
            storage,
            role: RouteCandidateRole::ChangeFilter,
            optional: false,
            exact: true,
        }
    }

    /// Returns the storage traversal order used by route tie-breaking.
    #[inline]
    pub fn storage_rank(&self) -> u8 {
        match self.storage {
            StorageKind::Table => 0,
            StorageKind::Sparse => 1,
        }
    }

    /// Returns the declaration-role order used by route tie-breaking.
    #[inline]
    pub fn role_rank(&self) -> u8 {
        match self.role {
            RouteCandidateRole::RequiredData => 0,
            RouteCandidateRole::RequiredFilter => 1,
            RouteCandidateRole::ChangeFilter => 2,
        }
    }
}

/// Role a component plays in the query declaration, as seen by the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteCandidateRole {
    /// Required query data (`&C` / `&mut C`).
    RequiredData,
    /// Required presence filter (`With<C>`).
    RequiredFilter,
    /// Change-detection filter (`Added<C>` / `Changed<C>`).
    ChangeFilter,
}

/// Selected execution source for one query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryRoute {
    /// Traverses the matching archetypes.
    Archetype {
        /// Estimated live rows the traversal covers.
        estimated_rows: usize,
    },
    /// Traverses the entries of one sparse component.
    SparseDriver(SparseRoute),
}

impl QueryRoute {
    /// Returns the driving sparse component, if any.
    #[inline]
    pub const fn driver(self) -> Option<ComponentId> {
        match self {
            Self::Archetype { .. } => None,
            Self::SparseDriver(route) => Some(route.driver),
        }
    }

    /// Returns why the sparse route was selected, if any.
    #[inline]
    pub const fn sparse_reason(self) -> Option<SparseRouteReason> {
        match self {
            Self::Archetype { .. } => None,
            Self::SparseDriver(route) => Some(route.reason),
        }
    }

    /// Returns the row count the selected route covers.
    #[inline]
    pub const fn estimated_rows(self) -> usize {
        match self {
            Self::Archetype { estimated_rows } => estimated_rows,
            Self::SparseDriver(route) => route.estimated_driver_rows,
        }
    }

    /// Returns the archetype rows the unselected route would have covered.
    #[inline]
    pub const fn estimated_archetype_rows(self) -> usize {
        match self {
            Self::Archetype { estimated_rows } => estimated_rows,
            Self::SparseDriver(route) => route.estimated_archetype_rows,
        }
    }
}

/// Details of a sparse-driver route selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseRoute {
    /// Sparse component driving the traversal.
    driver: ComponentId,
    /// Why the sparse route was selected.
    reason: SparseRouteReason,
    /// Driver entries covered by the traversal.
    estimated_driver_rows: usize,
    /// Rows the archetype route would have covered.
    estimated_archetype_rows: usize,
}

impl SparseRoute {
    /// Returns the driving component.
    #[inline]
    pub(crate) fn driver(self) -> ComponentId {
        self.driver
    }
}

/// Why the planner selected a sparse-driver route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseRouteReason {
    /// A required data source is sparse-stored.
    RequiredSparseData,
    /// A required `With<C>` filter is sparse-stored.
    RequiredSparseWith,
    /// A change-detection filter is sparse-stored.
    ChangedOrAddedSparse,
}

impl SparseRouteReason {
    /// Maps a declaration role to the sparse-route justification it produces.
    #[inline]
    const fn from_role(role: RouteCandidateRole) -> Self {
        match role {
            RouteCandidateRole::RequiredData => Self::RequiredSparseData,
            RouteCandidateRole::RequiredFilter => Self::RequiredSparseWith,
            RouteCandidateRole::ChangeFilter => Self::ChangedOrAddedSparse,
        }
    }
}

/// World versions and snapshots a plan watches for staleness.
struct QueryPlanningSnapshot {
    /// World route-cost version, watched for archetype row-count changes.
    global: PlanningCostVersion,
    /// World query-work version, watched for query topology changes.
    query_work_version: u64,
    /// Archetype topology observed at the last refresh.
    topology: TopologyPlanningSnapshot,
    /// Row snapshots of every matched archetype.
    watched_archetypes: Vec<ArchetypePlanningSnapshot>,
    /// Revision and locality snapshots of every watched sparse driver.
    watched_sparse_drivers: Vec<SparseDriverPlanningSnapshot>,
}

/// Result of the archetype-matching pass of a plan build.
struct ArchetypePlanningBuild {
    /// Archetypes matching the query.
    matched_archetypes: Vec<ArchetypeId>,
    /// Row snapshots kept for staleness tracking.
    watched_archetypes: Vec<ArchetypePlanningSnapshot>,
    /// Sum of live rows across matched archetypes.
    estimated_rows: usize,
    /// Topology snapshot at build time.
    topology: TopologyPlanningSnapshot,
}

/// Result of the sparse-driver pass of a plan build.
struct SparsePlanningBuild {
    /// Watch snapshots for every sparse-driver candidate.
    watched_drivers: Vec<SparseDriverPlanningSnapshot>,
}

/// Point-in-time planning state of one sparse driver.
#[derive(Clone, Copy, Debug)]
struct SparseDriverPlanningSnapshot {
    /// Sparse component this snapshot watches.
    component: ComponentId,
    /// Driver revision when the profile was built.
    revision: u64,
    /// Locality profile over matched archetypes.
    profile: SparseDriverProfile,
}

/// Route-selection state of one query.
///
/// Holds the normalized route candidates, the matched archetype set with its
/// row estimate, and the world versions watched for staleness. `refresh`
/// updates those inputs before route selection runs again.
pub(crate) struct QueryPlanning {
    /// Currently selected traversal source.
    route: QueryRoute,
    /// Policy governing route selection.
    policy: QueryRoutePolicy,
    /// Watched world versions and snapshots.
    snapshot: QueryPlanningSnapshot,
    /// Archetypes matching the query right now.
    matched_archetypes: Vec<ArchetypeId>,
    /// Candidates collected from data and filters, normalized once.
    route_candidates: Vec<QueryRouteCandidate>,
    /// Live rows summed over matched archetypes.
    estimated_archetype_rows: usize,
    /// Table and sparse component counts used by the cost model.
    query_shape: QueryShape,
}

impl QueryPlanning {
    /// Builds the planning state for one query.
    ///
    /// Collects route candidates from data and filters, matches every
    /// registered archetype, profiles the eligible sparse drivers, and selects
    /// the initial route under the automatic policy.
    pub(crate) fn build<D, F>(
        cm: &ComponentManager,
        planning: &WorldPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        access: &QueryAccess,
    ) -> Self
    where
        D: QueryData,
        F: QueryFilter,
    {
        let mut route_candidates = Vec::new();
        D::route_candidates(data_state, &mut route_candidates);
        F::route_candidates(filter_state, &mut route_candidates);

        normalize_route_candidates(&mut route_candidates);

        let archetypes =
            build_archetype_planning::<D, F>(cm, planning, data_state, filter_state, access);

        let estimated_archetype_rows = archetypes.estimated_rows;
        let query_shape = QueryShape::new(cm, access);
        let sparse = build_sparse_planning(cm, &route_candidates, &archetypes.matched_archetypes);

        let policy = QueryRoutePolicy::default();
        let route = select_route(
            estimated_archetype_rows,
            &route_candidates,
            &sparse.watched_drivers,
            query_shape,
            None,
            policy,
        );

        let snapshot = QueryPlanningSnapshot {
            global: planning.route_version(),
            query_work_version: planning.query_work_version(),
            topology: archetypes.topology,
            watched_archetypes: archetypes.watched_archetypes,
            watched_sparse_drivers: sparse.watched_drivers,
        };

        Self {
            route,
            policy,
            snapshot,
            matched_archetypes: archetypes.matched_archetypes,
            route_candidates,
            estimated_archetype_rows,
            query_shape,
        }
    }

    /// Returns the currently selected route.
    #[inline]
    pub(crate) const fn route(&self) -> QueryRoute {
        self.route
    }

    /// Returns the active route policy.
    #[inline]
    pub(crate) const fn route_policy(&self) -> QueryRoutePolicy {
        self.policy
    }

    /// Forces a route policy, validating sparse-driver requests.
    ///
    /// Returns whether the selected route changed.
    pub(crate) fn set_route_policy(
        &mut self,
        policy: QueryRoutePolicy,
    ) -> Result<bool, crate::query::QueryError> {
        if let QueryRoutePolicy::ForceSparseDriver(driver) = policy {
            let valid = self.route_candidates.iter().copied().any(|candidate| {
                candidate.component == driver && is_sparse_driver_candidate(candidate)
            });

            if !valid {
                return Err(crate::query::QueryError::InvalidSparseRouteDriver(driver));
            }
        }

        let previous_route = self.route;
        self.policy = policy;
        self.route = select_route(
            self.estimated_archetype_rows,
            &self.route_candidates,
            &self.snapshot.watched_sparse_drivers,
            self.query_shape,
            None,
            policy,
        );

        Ok(self.route != previous_route)
    }

    /// Returns the currently matched archetypes.
    #[inline]
    pub(crate) fn matched_archetypes(&self) -> &[ArchetypeId] {
        &self.matched_archetypes
    }

    /// Returns the archetype count observed at the last refresh.
    #[inline]
    pub(crate) const fn seen_archetypes(&self) -> usize {
        self.snapshot.topology.archetype_count
    }

    /// Refreshes route and matching set from current world versions.
    ///
    /// Returns whether the selected route changed.
    pub(crate) fn refresh<D, F>(
        &mut self,
        cm: &ComponentManager,
        planning: &WorldPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        access: &QueryAccess,
    ) -> bool
    where
        D: QueryData,
        F: QueryFilter,
    {
        let current = planning.route_version();
        let current_query_work_version = planning.query_work_version();

        let previous_route = self.route;
        let previous_seen_archetypes = self.snapshot.topology;
        let previous_matched_archetypes = self.matched_archetypes.len();
        let mut route_inputs_changed = false;

        let current_topology = planning.topology_snapshot();

        let matched_archetypes_changed =
            if current_topology.version != self.snapshot.topology.version {
                self.refresh_new_archetypes::<D, F>(cm, planning, data_state, filter_state, access)
            } else {
                false
            };

        route_inputs_changed |= matched_archetypes_changed;

        route_inputs_changed |= self.refresh_archetype_rows(planning);
        route_inputs_changed |= self.refresh_sparse_drivers(
            cm,
            matched_archetypes_changed,
            current_query_work_version != self.snapshot.query_work_version,
        );

        if route_inputs_changed {
            self.route = select_route(
                self.estimated_archetype_rows,
                &self.route_candidates,
                &self.snapshot.watched_sparse_drivers,
                self.query_shape,
                Some(previous_route),
                self.policy,
            );
        }

        self.snapshot.global = current;
        self.snapshot.query_work_version = current_query_work_version;

        self.route != previous_route
            || self.snapshot.topology != previous_seen_archetypes
            || self.matched_archetypes.len() != previous_matched_archetypes
    }

    /// Appends archetypes registered since the watched topology snapshot that
    /// match the query.
    ///
    /// Returns whether any archetype joined, which invalidates route costs.
    fn refresh_new_archetypes<D, F>(
        &mut self,
        cm: &ComponentManager,
        planning: &WorldPlanning,
        data_state: &D::State,
        filter_state: &F::State,
        access: &QueryAccess,
    ) -> bool
    where
        D: QueryData,
        F: QueryFilter,
    {
        if access.is_impossible() {
            self.snapshot.topology = planning.topology_snapshot();
            return false;
        }

        let mut changed_cost = false;

        let current_topology = planning.topology_snapshot();

        for index in self.snapshot.topology.archetype_count..current_topology.archetype_count {
            let id = ArchetypeId::new(index);
            let archetype = cm.get_archetype(id).unwrap();

            if D::matches_archetype(data_state, archetype)
                && F::matches_archetype(filter_state, archetype)
            {
                self.matched_archetypes.push(id);

                let snapshot = planning
                    .archetype_snapshot(id)
                    .expect("matched archetype must exist in WorldPlanning");

                self.estimated_archetype_rows += snapshot.rows;
                self.snapshot.watched_archetypes.push(snapshot);
                changed_cost = true;
            }
        }

        self.snapshot.topology = current_topology;
        changed_cost
    }

    /// Re-reads row snapshots of watched archetypes when the world reports
    /// row-count changes, adjusting the estimated total.
    ///
    /// Returns whether any matched archetype changed its row count.
    fn refresh_archetype_rows(&mut self, planning: &WorldPlanning) -> bool {
        if planning.route_version().archetype_rows() == self.snapshot.global.archetype_rows() {
            return false;
        }

        let mut changed = false;

        for snapshot in &mut self.snapshot.watched_archetypes {
            let current = planning
                .archetype_snapshot(snapshot.id)
                .expect("watched archetype must exist in WorldPlanning");

            if current.rows_version == snapshot.rows_version {
                continue;
            }

            self.estimated_archetype_rows = self
                .estimated_archetype_rows
                .checked_sub(snapshot.rows)
                .expect("query planning archetype row total must include every watched snapshot")
                + current.rows;

            *snapshot = current;
            changed = true;
        }

        changed
    }

    /// Rebuilds locality profiles of watched drivers after the matched set or
    /// the world's query-work version moved.
    ///
    /// A driver is rebuilt when the matched archetypes changed or the driver's
    /// own revision moved; returns whether any profile changed.
    fn refresh_sparse_drivers(
        &mut self,
        cm: &ComponentManager,
        matched_archetypes_changed: bool,
        query_work_changed: bool,
    ) -> bool {
        if !matched_archetypes_changed && !query_work_changed {
            return false;
        }

        let mut changed_cost = false;

        for snapshot in &mut self.snapshot.watched_sparse_drivers {
            let revision = cm
                .sparse_driver_revision(snapshot.component)
                .expect("watched sparse driver must exist in ComponentManager");

            if !matched_archetypes_changed && revision == snapshot.revision {
                continue;
            }

            let buckets = cm
                .sparse_driver_buckets(snapshot.component)
                .expect("watched sparse driver must expose its buckets");
            let profile = SparseDriverProfile::build(buckets, &self.matched_archetypes);

            changed_cost |= profile != snapshot.profile;
            snapshot.revision = revision;
            snapshot.profile = profile;
        }

        changed_cost
    }
}

/// Sorts candidates by component, storage, role, optionality and exactness,
/// then removes duplicates, making route selection and its tie-breaking
/// deterministic.
pub(crate) fn normalize_route_candidates(candidates: &mut Vec<QueryRouteCandidate>) {
    candidates.sort_unstable_by_key(|candidate| {
        (
            candidate.component.index(),
            candidate.storage_rank(),
            candidate.role_rank(),
            candidate.optional,
            candidate.exact,
        )
    });

    candidates.dedup_by_key(|candidate| {
        (
            candidate.component.index(),
            candidate.storage_rank(),
            candidate.role_rank(),
            candidate.optional,
            candidate.exact,
        )
    });
}

/// Matches every registered archetype against the query and snapshots the
/// results for staleness tracking.
fn build_archetype_planning<D, F>(
    cm: &ComponentManager,
    planning: &WorldPlanning,
    data_state: &D::State,
    filter_state: &F::State,
    access: &QueryAccess,
) -> ArchetypePlanningBuild
where
    D: QueryData,
    F: QueryFilter,
{
    let topology = planning.topology_snapshot();
    let mut matched_archetypes = Vec::new();
    let mut watched_archetypes = Vec::new();
    let mut estimated_rows = 0;

    if access.is_impossible() {
        return ArchetypePlanningBuild {
            matched_archetypes,
            watched_archetypes,
            estimated_rows,
            topology,
        };
    }

    for index in 0..topology.archetype_count {
        let id = ArchetypeId::new(index);
        let archetype = cm
            .get_archetype(id)
            .expect("planning references a missing archetype");

        if !D::matches_archetype(data_state, archetype)
            || !F::matches_archetype(filter_state, archetype)
        {
            continue;
        }

        let snapshot = planning
            .archetype_snapshot(id)
            .expect("matched archetype must exist in WorldPlanning");

        matched_archetypes.push(id);
        watched_archetypes.push(snapshot);
        estimated_rows += snapshot.rows;
    }

    ArchetypePlanningBuild {
        matched_archetypes,
        watched_archetypes,
        estimated_rows,
        topology,
    }
}

/// Builds sparse-driver watch snapshots for every eligible candidate.
fn build_sparse_planning(
    cm: &ComponentManager,
    route_candidates: &[QueryRouteCandidate],
    matched_archetypes: &[ArchetypeId],
) -> SparsePlanningBuild {
    let mut watched_drivers = Vec::new();

    for candidate in route_candidates
        .iter()
        .copied()
        .filter(|candidate| is_sparse_driver_candidate(*candidate))
    {
        if watched_drivers
            .iter()
            .any(|snapshot: &SparseDriverPlanningSnapshot| {
                snapshot.component == candidate.component
            })
        {
            continue;
        }

        let revision = cm
            .sparse_driver_revision(candidate.component)
            .expect("sparse route candidate must exist in ComponentManager");
        let buckets = cm
            .sparse_driver_buckets(candidate.component)
            .expect("sparse route candidate must expose its driver buckets");

        watched_drivers.push(SparseDriverPlanningSnapshot {
            component: candidate.component,
            revision,
            profile: SparseDriverProfile::build(buckets, matched_archetypes),
        });
    }

    SparsePlanningBuild { watched_drivers }
}

/// Applies the route policy over the cost model to pick the traversal source.
///
/// Forced policies short-circuit the comparison: `ForceArchetype` keeps the
/// archetype route, and `ForceSparseDriver` returns the requested driver when
/// it is a valid candidate. Under the automatic policy every eligible sparse
/// candidate is costed against the archetype baseline and the cheapest source
/// wins; the candidate matching the previously selected route is kept unless
/// the winner clears the hysteresis margin, so near-equal routes cannot flap
/// between refreshes.
fn select_route(
    archetype_rows: usize,
    route_candidates: &[QueryRouteCandidate],
    watched_sparse_drivers: &[SparseDriverPlanningSnapshot],
    query_shape: QueryShape,
    previous_route: Option<QueryRoute>,
    policy: QueryRoutePolicy,
) -> QueryRoute {
    let model = QueryCostModel::default();
    let estimated_matching_rows =
        estimate_matching_rows(archetype_rows, route_candidates, watched_sparse_drivers);
    let archetype_cost = model.archetype_cost(archetype_rows, estimated_matching_rows, query_shape);
    let archetype = RouteChoice::archetype(archetype_rows, archetype_cost);

    if policy == QueryRoutePolicy::ForceArchetype {
        return archetype.into_route(archetype_rows);
    }

    let forced_driver = match policy {
        QueryRoutePolicy::ForceSparseDriver(driver) => Some(driver),
        QueryRoutePolicy::Automatic | QueryRoutePolicy::ForceArchetype => None,
    };
    let previous_driver = previous_route.and_then(QueryRoute::driver);
    let mut best = archetype;
    let mut incumbent = match previous_route {
        Some(QueryRoute::Archetype { .. }) => Some(archetype),
        Some(QueryRoute::SparseDriver(_)) | None => None,
    };

    for candidate in route_candidates
        .iter()
        .copied()
        .filter(|candidate| is_sparse_driver_candidate(*candidate))
    {
        let Some(snapshot) = sparse_driver_snapshot(watched_sparse_drivers, candidate.component)
        else {
            continue;
        };

        let choice = RouteChoice::sparse(
            candidate,
            snapshot.profile.rows(),
            model.sparse_cost(snapshot.profile, query_shape, candidate.role),
        );

        if forced_driver == choice.driver() {
            return choice.into_route(archetype_rows);
        }

        if choice.sort_key() < best.sort_key() {
            best = choice;
        }

        if previous_driver == choice.driver() {
            incumbent = Some(match incumbent {
                Some(current) if current.sort_key() <= choice.sort_key() => current,
                _ => choice,
            });
        }
    }

    debug_assert!(
        forced_driver.is_none(),
        "forced sparse driver must remain a valid route candidate"
    );

    let selected = match incumbent {
        // Initial planning or an invalidated previous sparse source.
        None => best,

        // The best candidate is already the current source. Refresh its estimates.
        Some(current) if best.same_source(current) => best,

        // Switching an existing route requires a material improvement.
        Some(current) if materially_better(best.estimated_cost, current.estimated_cost) => best,

        // Keep the current route to avoid oscillation.
        Some(current) => current,
    };

    selected.into_route(archetype_rows)
}

/// Estimates how many archetype rows can survive all required sparse predicates.
///
/// Each exact, non-optional sparse candidate provides an upper bound for the
/// intersection. The smallest available membership count is therefore a safe,
/// conservative estimate. Queries without such a predicate keep the full scan
/// cardinality.
fn estimate_matching_rows(
    archetype_rows: usize,
    route_candidates: &[QueryRouteCandidate],
    watched_sparse_drivers: &[SparseDriverPlanningSnapshot],
) -> u64 {
    route_candidates
        .iter()
        .copied()
        .filter(|candidate| is_sparse_driver_candidate(*candidate))
        .filter_map(|candidate| {
            sparse_driver_snapshot(watched_sparse_drivers, candidate.component)
                .map(|snapshot| snapshot.profile.rows())
        })
        .fold(archetype_rows as u64, u64::min)
}

/// Returns whether a candidate satisfies the structural requirements to
/// drive a sparse route.
fn is_sparse_driver_candidate(candidate: QueryRouteCandidate) -> bool {
    candidate.storage == StorageKind::Sparse && !candidate.optional && candidate.exact
}

/// Finds the watch snapshot of one sparse component, if present.
fn sparse_driver_snapshot(
    snapshots: &[SparseDriverPlanningSnapshot],
    component: ComponentId,
) -> Option<SparseDriverPlanningSnapshot> {
    snapshots
        .iter()
        .copied()
        .find(|snapshot| snapshot.component == component)
}

/// Traversal source represented by one route choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RouteChoiceKind {
    /// Archetype-table traversal.
    Archetype,
    /// Sparse-set traversal driven by one component.
    Sparse {
        /// Sparse component driving the traversal.
        driver: ComponentId,
        /// Justification carried into the published route.
        reason: SparseRouteReason,
        /// Declaration-role rank, preserved for tie-breaking.
        role_rank: u8,
    },
}

/// One selectable traversal with the estimates used to compare it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RouteChoice {
    /// Source this choice represents.
    kind: RouteChoiceKind,
    /// Rows the choice's own traversal covers.
    estimated_rows: usize,
    /// Cost-model estimate driving the comparison.
    estimated_cost: u64,
}

impl RouteChoice {
    /// Builds the archetype-table choice.
    #[inline]
    const fn archetype(estimated_rows: usize, estimated_cost: u64) -> Self {
        Self {
            kind: RouteChoiceKind::Archetype,
            estimated_rows,
            estimated_cost,
        }
    }

    /// Builds a sparse-driver choice, saturating the row estimate into
    /// `usize`.
    #[inline]
    fn sparse(candidate: QueryRouteCandidate, estimated_rows: u64, estimated_cost: u64) -> Self {
        Self {
            kind: RouteChoiceKind::Sparse {
                driver: candidate.component,
                reason: SparseRouteReason::from_role(candidate.role),
                role_rank: candidate.role_rank(),
            },
            estimated_rows: usize::try_from(estimated_rows).unwrap_or(usize::MAX),
            estimated_cost,
        }
    }

    /// Returns the driving component of a sparse choice.
    #[inline]
    const fn driver(self) -> Option<ComponentId> {
        match self.kind {
            RouteChoiceKind::Archetype => None,
            RouteChoiceKind::Sparse { driver, .. } => Some(driver),
        }
    }

    /// Returns whether both choices traverse the same source.
    #[inline]
    const fn same_source(self, other: Self) -> bool {
        match (self.kind, other.kind) {
            (RouteChoiceKind::Archetype, RouteChoiceKind::Archetype) => true,
            (
                RouteChoiceKind::Sparse { driver: left, .. },
                RouteChoiceKind::Sparse { driver: right, .. },
            ) => left.index() == right.index(),
            _ => false,
        }
    }

    /// Returns the comparison key: cost first, then a preference for the
    /// archetype source over sparse, then the lowest driver index, then the
    /// lowest declaration-role rank.
    #[inline]
    fn sort_key(self) -> (u64, u8, usize, u8) {
        match self.kind {
            RouteChoiceKind::Archetype => (self.estimated_cost, 0, 0, 0),
            RouteChoiceKind::Sparse {
                driver, role_rank, ..
            } => (self.estimated_cost, 1, driver.index(), role_rank),
        }
    }

    /// Converts the choice into a route, pairing driver rows with the
    /// archetype rows of the alternative it beat.
    #[inline]
    const fn into_route(self, archetype_rows: usize) -> QueryRoute {
        match self.kind {
            RouteChoiceKind::Archetype => QueryRoute::Archetype {
                estimated_rows: archetype_rows,
            },
            RouteChoiceKind::Sparse { driver, reason, .. } => {
                QueryRoute::SparseDriver(SparseRoute {
                    driver,
                    reason,
                    estimated_driver_rows: self.estimated_rows,
                    estimated_archetype_rows: archetype_rows,
                })
            }
        }
    }
}

/// Returns whether `candidate` clears the hysteresis margin over `incumbent`.
///
/// Switching an established route requires the new cost to stay below 7/8 of
/// the current one (`candidate * 8 < incumbent * 7`, saturating), so repeated
/// refreshes cannot flap between near-equal routes.
#[inline]
fn materially_better(candidate: u64, incumbent: u64) -> bool {
    candidate.saturating_mul(8) < incumbent.saturating_mul(7)
}
