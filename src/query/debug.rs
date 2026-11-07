//! Structured diagnostics for query plans.
//!
//! This module is intentionally read-only: building a report must not refresh
//! cached archetype lists, advance ticks, or prepare iterator fetches.

use std::fmt;

use crate::{
    component::{archetype::ArchetypeId, ComponentId, ComponentManager, StorageKind},
    query::{
        filter::QueryFilter,
        planning::{QueryRoute, QueryRouteCandidate, RouteCandidateRole, SparseRouteReason},
        state::{QueryCacheDebug, QueryCacheKind, QueryPlan, QueryState},
        AccessMask, QueryData,
    },
};

/// Origin of a component entry in the query declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryDebugSource {
    /// The entry was declared by the query data.
    Data,
    /// The entry was declared by a query filter.
    Filter,
}

/// How a filter entry is composed with the rest of the filter expression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryDebugComposition {
    /// ANDed with the other constraints.
    Conjunction,
    /// One branch of an `Or`/`AnyOf` composition.
    Disjunction,
    /// Wrapped by a `Not`.
    Negation,
}

/// Component-level entry requested by query data or filters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryDebugItem {
    /// Whether the data or the filter declared this entry.
    pub source: QueryDebugSource,
    /// Human-readable expression text, e.g. `"&mut Position"`.
    pub expression: &'static str,
    /// Rust type name of the component.
    pub component_type: &'static str,
    /// Component identifier, absent for the `Entity` data source.
    pub component: Option<ComponentId>,
    /// Storage backend, absent for the `Entity` data source.
    pub storage: Option<StorageKind>,
    /// Access the entry declares.
    pub access: AccessMask,
    /// Whether the entry is optional (`Option<&C>` / `Option<&mut C>`).
    pub optional: bool,
    /// Whether the filter accepts presence rather than absence.
    pub positive: bool,
    /// Whether domain-based conflict elimination may rely on this entry.
    pub exact: bool,
    /// Composition role of the entry inside the filter expression.
    pub composition: QueryDebugComposition,
    /// Sparse-route candidacy, when this entry can drive one.
    pub route_role: Option<RouteCandidateRole>,
}

impl QueryDebugItem {
    /// Builds the entry describing the `Entity` data source.
    pub fn entity() -> Self {
        Self {
            source: QueryDebugSource::Data,
            expression: "Entity",
            component_type: "Entity",
            component: None,
            storage: None,
            access: AccessMask::NONE,
            optional: false,
            positive: true,
            exact: true,
            composition: QueryDebugComposition::Conjunction,
            route_role: None,
        }
    }

    /// Builds the entry for a required component data source.
    pub fn component(
        source: QueryDebugSource,
        expression: &'static str,
        component_type: &'static str,
        component: ComponentId,
        storage: StorageKind,
        access: AccessMask,
        route_role: Option<RouteCandidateRole>,
    ) -> Self {
        Self {
            source,
            expression,
            component_type,
            component: Some(component),
            storage: Some(storage),
            access,
            optional: false,
            positive: true,
            exact: true,
            composition: QueryDebugComposition::Conjunction,
            route_role,
        }
    }

    /// Builds the entry for an optional component data source.
    pub fn optional_data(
        expression: &'static str,
        component_type: &'static str,
        component: ComponentId,
        storage: StorageKind,
        access: AccessMask,
    ) -> Self {
        Self {
            source: QueryDebugSource::Data,
            expression,
            component_type,
            component: Some(component),
            storage: Some(storage),
            access,
            optional: true,
            positive: true,
            exact: false,
            composition: QueryDebugComposition::Conjunction,
            route_role: Some(RouteCandidateRole::RequiredData),
        }
    }

    /// Builds the entry for a `Without<C>` filter.
    pub fn excluded_filter(
        expression: &'static str,
        component_type: &'static str,
        component: ComponentId,
        storage: StorageKind,
        access: AccessMask,
    ) -> Self {
        Self {
            source: QueryDebugSource::Filter,
            expression,
            component_type,
            component: Some(component),
            storage: Some(storage),
            access,
            optional: false,
            positive: false,
            exact: true,
            composition: QueryDebugComposition::Conjunction,
            route_role: None,
        }
    }

    /// Marks the entry as a disjunction branch, removing it from exact
    /// domain reasoning.
    pub fn mark_disjunctive(&mut self) {
        self.composition = QueryDebugComposition::Disjunction;
        self.exact = false;
        self.route_role = None;
    }

    /// Marks the entry as negated, removing it from exact domain reasoning.
    pub fn mark_negated(&mut self) {
        self.composition = QueryDebugComposition::Negation;
        self.positive = false;
        self.exact = false;
        self.route_role = None;
    }

    /// Returns whether this entry satisfies every requirement to drive a
    /// sparse-driver route.
    pub fn can_drive_sparse_route(&self) -> bool {
        self.storage == Some(StorageKind::Sparse)
            && self.route_role.is_some()
            && !self.optional
            && self.positive
            && self.exact
            && self.composition == QueryDebugComposition::Conjunction
    }
}

/// One archetype accepted by the query plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryArchetypeDebug {
    /// Identifier of the matching archetype.
    pub id: ArchetypeId,
    /// Live rows of the archetype when the report was built.
    pub rows: usize,
}

/// Route-driver candidate as seen by `QueryPlanning`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryDriverCandidateDebug {
    /// Component the candidate refers to.
    pub component: ComponentId,
    /// Storage backend of the component.
    pub storage: StorageKind,
    /// Role the component plays in the query declaration.
    pub role: RouteCandidateRole,
    /// Whether the entry is optional.
    pub optional: bool,
    /// Whether domain reasoning may rely on the entry.
    pub exact: bool,
    /// Live sparse entries of the component, when sparse.
    pub sparse_candidates: Option<usize>,
    /// Whether the plan selected this candidate as the driver.
    pub selected: bool,
}

impl QueryDriverCandidateDebug {
    /// Returns whether the candidate satisfies the structural requirements
    /// to drive a sparse route, independent of cost.
    pub fn is_eligible(&self) -> bool {
        self.storage == StorageKind::Sparse && !self.optional && self.exact
    }
}

/// Combined access entry retained by the plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueryAccessDebug {
    /// Component whose accesses are summarized.
    pub component: ComponentId,
    /// Rust type name, when any debug item declared the component.
    pub component_type: Option<&'static str>,
    /// Storage backend, when the component is registered.
    pub storage: Option<StorageKind>,
    /// Union of every access declared for the component.
    pub mask: AccessMask,
}

/// Complete printable report for one `QueryPlan`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueryPlanDebug {
    /// Cache mode and retention counters of the plan's state.
    pub cache: QueryCacheDebug,
    /// Selected execution route.
    pub route: QueryRoute,
    /// Whether the data part resolves without sparse-set lookups.
    pub archetypal: bool,
    /// Whether the filter domain provably matches nothing.
    pub impossible: bool,
    /// Number of archetypes matched by the plan.
    pub matched_archetypes: usize,
    /// Sum of live rows across matching archetypes.
    pub matched_archetype_rows: usize,
    /// Per-archetype match details.
    pub matching_archetypes: Vec<QueryArchetypeDebug>,
    /// Live sparse entries of the selected driver, if any.
    pub sparse_driver_candidates: Option<usize>,
    /// Entries declared by the query data.
    pub data: Vec<QueryDebugItem>,
    /// Entries declared by the query filters.
    pub filters: Vec<QueryDebugItem>,
    /// Sparse-driver candidates considered by the planner.
    pub driver_candidates: Vec<QueryDriverCandidateDebug>,
    /// Combined component accesses of data and filters.
    pub accesses: Vec<QueryAccessDebug>,
}

impl QueryPlanDebug {
    /// Builds the report from a live query state.
    pub(crate) fn from_query_state<D, F>(cm: &ComponentManager, state: &QueryState<D, F>) -> Self
    where
        D: QueryData,
        F: QueryFilter,
    {
        Self::from_plan::<D, F>(cm, state.cache_debug(), state.plan())
    }

    /// Builds the report from explicit cache diagnostics and a plan.
    pub(crate) fn from_plan<D, F>(
        cm: &ComponentManager,
        cache: QueryCacheDebug,
        plan: &QueryPlan<D, F>,
    ) -> Self
    where
        D: QueryData,
        F: QueryFilter,
    {
        let mut data = Vec::new();
        D::debug_items(plan.data_state(), &mut data);

        let mut filters = Vec::new();
        F::debug_items(plan.filter_state(), &mut filters);

        let mut route_candidates = Vec::new();
        D::route_candidates(plan.data_state(), &mut route_candidates);
        F::route_candidates(plan.filter_state(), &mut route_candidates);

        let route = plan.route();
        let sparse_driver_candidates = route.driver().and_then(|driver| cm.sparse_len(driver));
        let matching_archetypes = matching_archetypes::<D, F>(cm, plan);
        let matched_archetype_rows = matching_archetypes
            .iter()
            .map(|archetype| archetype.rows)
            .sum();

        let driver_candidates = route_candidates
            .into_iter()
            .map(|candidate| driver_candidate_debug(cm, route, candidate))
            .collect();

        let accesses = plan
            .access()
            .regions()
            .accesses()
            .iter()
            .map(|access| {
                let component = access.component();
                QueryAccessDebug {
                    component,
                    component_type: component_type(component, &data, &filters),
                    storage: cm.storage_of(component),
                    mask: access.mask(),
                }
            })
            .collect();

        Self {
            cache,
            route,
            archetypal: plan.is_archetypal(),
            impossible: plan.is_impossible(),
            matched_archetypes: matching_archetypes.len(),
            matched_archetype_rows,
            matching_archetypes,
            sparse_driver_candidates,
            data,
            filters,
            driver_candidates,
            accesses,
        }
    }
}

impl fmt::Display for QueryPlanDebug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "QueryPlan debug")?;

        write_cache(f, &self.cache)?;
        write_route(f, self.route)?;
        write_domain(f, self)?;
        write_archetypes(f, self)?;
        write_sparse_driver_candidates(f, self.sparse_driver_candidates)?;
        write_items(f, "Data", &self.data, self.route.driver())?;
        write_items(f, "Filters", &self.filters, self.route.driver())?;
        write_driver_candidates(f, self)?;
        write_accesses(f, self)?;

        Ok(())
    }
}

/// Scans every archetype and collects the ones the plan matches.
fn matching_archetypes<D, F>(
    cm: &ComponentManager,
    plan: &QueryPlan<D, F>,
) -> Vec<QueryArchetypeDebug>
where
    D: QueryData,
    F: QueryFilter,
{
    if plan.is_impossible() {
        return Vec::new();
    }

    (0..cm.archetype_count())
        .filter_map(|index| {
            let id = ArchetypeId::new(index);
            let archetype = cm.get_archetype(id)?;

            plan.matches_archetype(archetype)
                .then_some(QueryArchetypeDebug {
                    id,
                    rows: archetype.len(),
                })
        })
        .collect()
}

/// Enriches one route candidate with live sparse counts and selection state.
fn driver_candidate_debug(
    cm: &ComponentManager,
    route: QueryRoute,
    candidate: QueryRouteCandidate,
) -> QueryDriverCandidateDebug {
    QueryDriverCandidateDebug {
        component: candidate.component,
        storage: candidate.storage,
        role: candidate.role,
        optional: candidate.optional,
        exact: candidate.exact,
        sparse_candidates: (candidate.storage == StorageKind::Sparse)
            .then(|| cm.sparse_len(candidate.component))
            .flatten(),
        selected: route.driver() == Some(candidate.component)
            && candidate.storage == StorageKind::Sparse
            && !candidate.optional
            && candidate.exact
            && route.sparse_reason() == Some(sparse_reason_from_role(candidate.role)),
    }
}

/// Maps a declaration role to its sparse-route justification.
fn sparse_reason_from_role(role: RouteCandidateRole) -> SparseRouteReason {
    match role {
        RouteCandidateRole::RequiredData => SparseRouteReason::RequiredSparseData,
        RouteCandidateRole::RequiredFilter => SparseRouteReason::RequiredSparseWith,
        RouteCandidateRole::ChangeFilter => SparseRouteReason::ChangedOrAddedSparse,
    }
}

/// Finds the Rust type name of a component among the debug items.
fn component_type(
    component: ComponentId,
    data: &[QueryDebugItem],
    filters: &[QueryDebugItem],
) -> Option<&'static str> {
    data.iter()
        .chain(filters.iter())
        .find(|item| item.component == Some(component))
        .map(|item| item.component_type)
}

/// Writes the cache section of the report.
fn write_cache(f: &mut fmt::Formatter<'_>, cache: &QueryCacheDebug) -> fmt::Result {
    writeln!(f, "  Cache:")?;
    writeln!(f, "    {}", format_cache_kind(cache.kind))?;

    if let Some(retained) = cache.retained_archetypes {
        writeln!(f, "      retained_archetypes={retained}")?;
    }

    if let Some(seen) = cache.seen_archetypes {
        writeln!(f, "      seen_archetypes={seen}")?;
    }

    if let Some(runs) = cache.runs {
        writeln!(f, "      runs={runs}")?;
    }

    if let Some(policy) = cache.policy {
        writeln!(f, "      policy:")?;
        writeln!(f, "        cache_after_runs={}", policy.cache_after_runs)?;
        writeln!(
            f,
            "        cache_after_archetypes={}",
            policy.cache_after_archetypes
        )?;
    }

    Ok(())
}

/// Renders a cache kind as its report label.
fn format_cache_kind(kind: QueryCacheKind) -> &'static str {
    match kind {
        QueryCacheKind::Cached => "cached",
        QueryCacheKind::Uncached => "uncached",
        QueryCacheKind::AutoCached => "auto(cached)",
        QueryCacheKind::AutoUncached => "auto(uncached)",
    }
}

/// Writes the route section of the report.
fn write_route(f: &mut fmt::Formatter<'_>, route: QueryRoute) -> fmt::Result {
    writeln!(f, "  Route:")?;

    match route {
        QueryRoute::Archetype { estimated_rows } => {
            writeln!(f, "    Archetype")?;
            writeln!(f, "      estimated_rows={estimated_rows}")?;
        }
        QueryRoute::SparseDriver(_) => {
            writeln!(f, "    SparseDriver")?;
            writeln!(f, "      driver={:?}", route.driver())?;
            writeln!(f, "      reason={:?}", route.sparse_reason())?;
            writeln!(f, "      driver_rows={}", route.estimated_rows())?;
            writeln!(
                f,
                "      archetype_rows={}",
                route.estimated_archetype_rows()
            )?;
        }
    }

    Ok(())
}

/// Writes the filter-domain section of the report.
fn write_domain(f: &mut fmt::Formatter<'_>, report: &QueryPlanDebug) -> fmt::Result {
    writeln!(f, "  Domain:")?;
    writeln!(f, "    archetypal={}", report.archetypal)?;
    writeln!(f, "    impossible={}", report.impossible)
}

/// Writes the matching-archetype section of the report.
fn write_archetypes(f: &mut fmt::Formatter<'_>, report: &QueryPlanDebug) -> fmt::Result {
    writeln!(f, "  Archetypes:")?;
    writeln!(f, "    matches={}", report.matched_archetypes)?;
    writeln!(f, "    rows={}", report.matched_archetype_rows)?;

    for archetype in &report.matching_archetypes {
        writeln!(f, "      - {:?}: rows={}", archetype.id, archetype.rows)?;
    }

    Ok(())
}

/// Writes the sparse-driver candidate count, when a driver is selected.
fn write_sparse_driver_candidates(
    f: &mut fmt::Formatter<'_>,
    candidates: Option<usize>,
) -> fmt::Result {
    if let Some(candidates) = candidates {
        writeln!(f, "  Sparse driver candidates: {candidates}")?;
    }

    Ok(())
}

/// Writes one titled section of debug items with driver-state annotations.
fn write_items(
    f: &mut fmt::Formatter<'_>,
    title: &str,
    items: &[QueryDebugItem],
    selected_driver: Option<ComponentId>,
) -> fmt::Result {
    writeln!(f, "  {title}:")?;

    if items.is_empty() {
        writeln!(f, "    - none")?;
        return Ok(());
    }

    for item in items {
        let component = item
            .component
            .map(|component| format!("{component:?}"))
            .unwrap_or_else(|| "none".to_owned());
        let storage = item
            .storage
            .map(|storage| format!("{storage:?}"))
            .unwrap_or_else(|| "none".to_owned());
        let selected = item.can_drive_sparse_route() && item.component == selected_driver;

        writeln!(f, "    - {} {}", item.expression, item.component_type,)?;
        writeln!(f, "      component={component}")?;
        writeln!(f, "      storage={storage}")?;
        writeln!(f, "      access={}", format_access(item.access))?;
        writeln!(f, "      optional={}", item.optional)?;
        writeln!(f, "      positive={}", item.positive)?;
        writeln!(f, "      exact={}", item.exact)?;
        writeln!(f, "      composition={:?}", item.composition)?;
        writeln!(
            f,
            "      driver={}",
            format_item_driver_state(item, selected)
        )?;
    }

    Ok(())
}

/// Writes the driver-candidate section of the report.
fn write_driver_candidates(f: &mut fmt::Formatter<'_>, report: &QueryPlanDebug) -> fmt::Result {
    writeln!(f, "  Driver candidates:")?;

    if report.driver_candidates.is_empty() {
        writeln!(f, "    - none")?;
        return Ok(());
    }

    for candidate in &report.driver_candidates {
        writeln!(f, "    - component={:?}", candidate.component)?;
        writeln!(
            f,
            "      type={}",
            component_type(candidate.component, &report.data, &report.filters).unwrap_or("unknown")
        )?;
        writeln!(f, "      storage={:?}", candidate.storage)?;
        writeln!(f, "      role={:?}", candidate.role)?;
        writeln!(f, "      optional={}", candidate.optional)?;
        writeln!(f, "      exact={}", candidate.exact)?;
        writeln!(
            f,
            "      sparse_candidates={}",
            format_optional_usize(candidate.sparse_candidates)
        )?;
        writeln!(f, "      eligible={}", candidate.is_eligible())?;
        writeln!(f, "      selected={}", candidate.selected)?;
    }

    Ok(())
}

/// Writes the combined-access section of the report.
fn write_accesses(f: &mut fmt::Formatter<'_>, report: &QueryPlanDebug) -> fmt::Result {
    writeln!(f, "  Combined access:")?;

    if report.accesses.is_empty() {
        writeln!(f, "    - none")?;
        return Ok(());
    }

    for access in &report.accesses {
        writeln!(f, "    - component={:?}", access.component)?;
        writeln!(
            f,
            "      type={}",
            access.component_type.unwrap_or("unknown")
        )?;
        writeln!(
            f,
            "      storage={}",
            access
                .storage
                .map(|storage| format!("{storage:?}"))
                .unwrap_or_else(|| "unknown".to_owned())
        )?;
        writeln!(f, "      mask={}", format_access(access.mask))?;
    }

    Ok(())
}

/// Renders an item's driver state: selected, eligible, or not a candidate.
fn format_item_driver_state(item: &QueryDebugItem, selected: bool) -> &'static str {
    if selected {
        "selected"
    } else if item.can_drive_sparse_route() {
        "eligible"
    } else {
        "no"
    }
}

/// Renders an optional count, or `none`.
fn format_optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "none".to_owned())
}

/// Renders an access mask as its set of category flags.
fn format_access(mask: AccessMask) -> String {
    if mask.is_empty() {
        return "none".to_owned();
    }

    let mut parts = Vec::new();

    if mask.contains(AccessMask::VALUE_READ) {
        parts.push("VALUE_READ");
    }
    if mask.contains(AccessMask::VALUE_WRITE) {
        parts.push("VALUE_WRITE");
    }
    if mask.contains(AccessMask::TICKS_READ) {
        parts.push("TICKS_READ");
    }
    if mask.contains(AccessMask::TICKS_WRITE) {
        parts.push("TICKS_WRITE");
    }
    if mask.contains(AccessMask::STRUCTURE_READ) {
        parts.push("STRUCTURE_READ");
    }
    if mask.contains(AccessMask::STRUCTURE_WRITE) {
        parts.push("STRUCTURE_WRITE");
    }

    parts.join("|")
}

#[cfg(test)]
mod tests {
    use crate::{
        component::Component,
        query::{
            filter::With, QueryCacheKind, QueryDebugComposition, QueryDebugSource, QueryRoute,
        },
        world::World,
    };

    struct Position;
    impl Component for Position {}

    struct Health;
    impl Component for Health {}

    #[test]
    fn query_plan_debug_reports_cache_route_items_and_counts() {
        let mut world = World::new();
        world.register_table::<Position>().unwrap();
        let health = world.register_sparse::<Health>().unwrap();

        let entity_a = world.spawn((Position,)).unwrap();
        let entity_b = world.spawn((Position,)).unwrap();
        world.insert(entity_b, (Health,)).unwrap();
        assert_ne!(entity_a, entity_b);

        let state = world
            .query_state_cached::<&Position, With<Health>>()
            .unwrap();
        let report = world.query_state_debug(&state).unwrap();

        assert_eq!(report.cache.kind, QueryCacheKind::Cached);
        assert!(matches!(report.route, QueryRoute::SparseDriver(_)));
        assert_eq!(report.route.driver(), Some(health));
        assert_eq!(report.matched_archetypes, 1);
        assert_eq!(report.matched_archetype_rows, 2);
        assert_eq!(report.sparse_driver_candidates, Some(1));
        assert_eq!(report.data.len(), 1);
        assert_eq!(report.filters.len(), 1);
        assert_eq!(report.filters[0].source, QueryDebugSource::Filter);
        assert_eq!(
            report.filters[0].composition,
            QueryDebugComposition::Conjunction
        );

        let text = report.to_string();
        assert!(text.contains("Cache:"));
        assert!(text.contains("    cached"));
        assert!(text.contains("SparseDriver"));
        assert!(text.contains("Driver candidates"));
        assert!(text.contains("Combined access"));
    }
}
