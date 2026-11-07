//! Filter-aware component access metadata.
//!
//! `AccessSet` describes which component regions a query touches, while the
//! types in this module retain the set of entities that the query may match.
//! Keeping both dimensions allows independent queries with mutually exclusive
//! filters to coexist without weakening ordinary alias checks.

use crate::{
    component::ComponentId,
    query::{AccessMask, AccessSet, ComponentAccess, QueryError},
};

/// Sorted set of component identifiers used by a filter clause.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ComponentSet {
    ids: Vec<ComponentId>,
}

impl ComponentSet {
    /// Whether both sets contain at least one common component.
    fn intersects(&self, other: &Self) -> bool {
        let mut left = 0;
        let mut right = 0;

        while left < self.ids.len() && right < other.ids.len() {
            match self.ids[left].cmp(&other.ids[right]) {
                std::cmp::Ordering::Less => left += 1,
                std::cmp::Ordering::Greater => right += 1,
                std::cmp::Ordering::Equal => return true,
            }
        }

        false
    }

    /// Inserts an identifier while preserving sorted, duplicate-free storage.
    fn insert(&mut self, component: ComponentId) {
        if let Err(index) = self.ids.binary_search(&component) {
            self.ids.insert(index, component);
        }
    }

    /// Inserts every identifier of `other`.
    fn extend_from(&mut self, other: &Self) {
        for component in other.iter() {
            self.insert(component);
        }
    }

    /// Iterates the stored identifiers in sorted order.
    fn iter(&self) -> impl Iterator<Item = ComponentId> + '_ {
        self.ids.iter().copied()
    }
}

/// One conjunction of component-presence requirements.
///
/// A clause matches entities containing every `required` component and none of
/// the `excluded` components.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FilterClause {
    required: ComponentSet,
    excluded: ComponentSet,
}

impl FilterClause {
    /// Whether the two clauses can match a common entity.
    ///
    /// A clause pair is provably disjoint when either side requires a
    /// component the other excludes; lacking such proof, this conservatively
    /// reports overlap.
    fn may_overlap(&self, other: &Self) -> bool {
        !self.required.intersects(&other.excluded) && !other.required.intersects(&self.excluded)
    }

    /// Whether the clause requires and excludes a same component, so it
    /// matches nothing.
    fn is_impossible(&self) -> bool {
        self.required.intersects(&self.excluded)
    }

    /// Builds the disjunction that complements this clause: one clause
    /// excluding each required component and one clause requiring each
    /// excluded component, per De Morgan over the clause conjunction.
    fn negated_expression(&self) -> FilterExpression {
        let mut expression = FilterExpression::empty();

        for component in self.required.iter() {
            let mut clause = Self::default();
            clause.excluded.insert(component);
            expression.clauses.push(clause);
        }

        for component in self.excluded.iter() {
            let mut clause = Self::default();
            clause.required.insert(component);
            expression.clauses.push(clause);
        }

        expression
    }
}

/// Disjunction of filter clauses.
///
/// The current built-in filters form a single conjunction. Representing the
/// domain as clauses also leaves room for an `Or` filter without changing the
/// scheduler-facing access model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct FilterExpression {
    clauses: Vec<FilterClause>,
}

impl Default for FilterExpression {
    /// The always-true expression: a single clause with no constraints.
    fn default() -> Self {
        Self {
            clauses: vec![FilterClause::default()],
        }
    }
}

impl FilterExpression {
    /// The always-false expression with no clauses.
    fn empty() -> Self {
        Self {
            clauses: Vec::new(),
        }
    }

    /// Requires `component` in every clause.
    fn require(&mut self, component: ComponentId) {
        for clause in &mut self.clauses {
            clause.required.insert(component);
        }
    }

    /// Excludes `component` from every clause.
    fn exclude(&mut self, component: ComponentId) {
        for clause in &mut self.clauses {
            clause.excluded.insert(component);
        }
    }

    /// Requires every component of `components` in all clauses.
    fn require_all(&mut self, components: &ComponentSet) {
        for component in components.iter() {
            self.require(component);
        }
    }

    /// Drops clauses that require and exclude a same component.
    fn remove_impossible_clauses(&mut self) {
        self.clauses.retain(|clause| !clause.is_impossible());
    }

    /// Extends the clause list with the surviving clauses of `other`, widening
    /// the domain to the union of both.
    fn disjoin(&mut self, mut other: Self) {
        other.remove_impossible_clauses();
        self.clauses.extend(other.clauses);
    }

    /// Cross-products the clauses of both expressions, merging their required
    /// and excluded sets. An empty result means the conjunction is
    /// unsatisfiable.
    fn conjoin(&mut self, mut other: Self) {
        self.remove_impossible_clauses();
        other.remove_impossible_clauses();

        if self.clauses.is_empty() || other.clauses.is_empty() {
            self.clauses.clear();
            return;
        }

        let left_clauses = std::mem::take(&mut self.clauses);
        let mut clauses = Vec::with_capacity(left_clauses.len() * other.clauses.len());

        for left in left_clauses {
            for right in &other.clauses {
                let mut clause = left.clone();
                clause.required.extend_from(&right.required);
                clause.excluded.extend_from(&right.excluded);

                if !clause.is_impossible() {
                    clauses.push(clause);
                }
            }
        }

        self.clauses = clauses;
    }

    /// Negates the whole expression by conjoining each clause's negated
    /// disjunction, per De Morgan. An already-impossible expression negates to
    /// the always-true one.
    fn negate(&mut self) {
        self.remove_impossible_clauses();

        if self.clauses.is_empty() {
            *self = Self::default();
            return;
        }

        let clauses = std::mem::take(&mut self.clauses);
        let mut expression = Self::default();

        for clause in clauses {
            expression.conjoin(clause.negated_expression());
        }

        *self = expression;
    }

    /// `true` when any clause pair can match a common entity.
    fn may_overlap(&self, other: &Self) -> bool {
        self.clauses
            .iter()
            .any(|left| other.clauses.iter().any(|right| left.may_overlap(right)))
    }

    /// No clause survived: the expression matches nothing.
    fn is_impossible(&self) -> bool {
        self.clauses.is_empty()
    }
}

/// Complete access description for one query.
#[derive(Clone, Debug)]
pub struct QueryAccess {
    regions: AccessSet,
    domain: FilterExpression,
}

impl QueryAccess {
    /// Combines data and filter access belonging to the same query.
    ///
    /// Region masks are unioned here without rejecting a data/filter
    /// conflict: the iterator evaluates its filter before yielding the data
    /// item. Independent queries are validated later by `FilteredAccessSet`.
    pub(crate) fn compose_query(data: DataAccess, mut filters: FilterAccess) -> Self {
        filters.expression.require_all(&data.required);
        filters.expression.remove_impossible_clauses();

        Self {
            regions: AccessSet::compose_query(data.regions, filters.regions),
            domain: filters.expression,
        }
    }

    /// The regions this query touches.
    pub fn regions(&self) -> &AccessSet {
        &self.regions
    }

    /// Returns whether the filter domain provably matches nothing.
    pub fn is_impossible(&self) -> bool {
        self.domain.is_impossible()
    }

    /// Whether this query conflicts with `other`.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.first_conflict(other).is_some()
    }

    /// Returns the first conflict with `other`, or `None` when the filter
    /// domains prove the two queries never observe the same entity.
    fn first_conflict(&self, other: &Self) -> Option<QueryError> {
        if !self.domain.may_overlap(&other.domain) {
            return None;
        }

        self.regions
            .first_conflict(&other.regions)
            .map(|(current, requested)| QueryError::AccessConflict {
                component: current.component(),
                current: current.mask(),
                requested: requested.mask(),
            })
    }
}

/// Builder used by `QueryData` implementations.
#[derive(Debug, Default)]
pub struct DataAccess {
    regions: AccessSet,
    required: ComponentSet,
}

impl DataAccess {
    /// Registers a required component access and records it as mandatory
    /// for the filter domain.
    pub fn add_required(
        &mut self,
        component: ComponentId,
        mask: AccessMask,
    ) -> Result<(), QueryError> {
        self.regions.add(component, mask)?;
        self.required.insert(component);
        Ok(())
    }

    /// Registers an optional component access without affecting the domain.
    pub fn add_optional(
        &mut self,
        component: ComponentId,
        mask: AccessMask,
    ) -> Result<(), QueryError> {
        self.regions.add(component, mask)
    }
}

/// Builder used by `QueryFilter` implementations.
#[derive(Debug)]
pub struct FilterAccess {
    regions: AccessSet,
    expression: FilterExpression,
    /// Whether the domain still mirrors the declared predicates exactly;
    /// change filters and lossy compositions clear it.
    exact: bool,
}

impl Default for FilterAccess {
    /// Returns an empty access whose domain matches every entity exactly.
    fn default() -> Self {
        Self {
            regions: AccessSet::default(),
            expression: FilterExpression::default(),
            exact: true,
        }
    }
}

impl FilterAccess {
    /// Registers a required component access and constrains the domain to
    /// entities that have it.
    pub fn add_required(
        &mut self,
        component: ComponentId,
        mask: AccessMask,
    ) -> Result<(), QueryError> {
        self.regions.add(component, mask)?;
        self.expression.require(component);
        Ok(())
    }

    /// Registers an excluded component access and constrains the domain to
    /// entities that do not have it.
    pub fn add_excluded(
        &mut self,
        component: ComponentId,
        mask: AccessMask,
    ) -> Result<(), QueryError> {
        self.regions.add(component, mask)?;
        self.expression.exclude(component);
        Ok(())
    }

    /// Demotes the filter domain to inexact, disabling domain-based conflict
    /// elimination for this query.
    pub fn mark_domain_inexact(&mut self) {
        self.exact = false;
    }

    /// Merges every branch of an `Or`/`AnyOf` composition into one domain.
    pub fn add_disjunction<I>(&mut self, branches: I)
    where
        I: IntoIterator<Item = Self>,
    {
        let mut expression = FilterExpression::empty();
        let mut exact = true;

        for branch in branches {
            self.regions.union_from(&branch.regions);
            expression.disjoin(branch.expression);
            exact &= branch.exact;
        }

        self.expression.conjoin(expression);
        self.exact &= exact;
    }

    /// Merges a `Not<...>` composition into the domain.
    ///
    /// The inner access regions still union into this filter; only the
    /// negated predicate is folded in when the inner domain was exact.
    pub fn add_negation(&mut self, mut inner: Self) {
        self.regions.union_from(&inner.regions);

        if inner.exact {
            inner.expression.negate();
            self.expression.conjoin(inner.expression);
        } else {
            self.mark_domain_inexact();
        }
    }
}

/// Query accesses retained independently for precise conflict checks.
#[derive(Clone, Debug, Default)]
pub struct FilteredAccessSet {
    /// Region union of every retained, non-impossible query.
    combined: AccessSet,
    /// Accesses retained individually so filter domains can veto conflicts.
    queries: Vec<QueryAccess>,
}

impl FilteredAccessSet {
    /// Wraps one already-validated query access.
    pub fn from_query(query: QueryAccess) -> Self {
        let mut accesses = Self::default();
        accesses
            .add(query)
            .expect("a single validated query cannot conflict with itself");
        accesses
    }

    /// Adds a query access, rejecting any conflict with a retained one.
    pub fn add(&mut self, query: QueryAccess) -> Result<(), QueryError> {
        for current in &self.queries {
            if let Some(error) = current.first_conflict(&query) {
                return Err(error);
            }
        }

        if !query.is_impossible() {
            self.combined.union_from(query.regions());
        }
        self.queries.push(query);
        Ok(())
    }

    /// Adds every query access retained by `other`, re-running the pairwise
    /// conflict checks against the queries held here.
    pub(crate) fn extend_from(&mut self, other: &Self) -> Result<(), QueryError> {
        for query in &other.queries {
            self.add(query.clone())?;
        }
        Ok(())
    }

    /// Whether any retained query pair conflicts.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        if !self.combined.conflicts_with(&other.combined) {
            return false;
        }

        self.queries
            .iter()
            .any(|left| other.queries.iter().any(|right| left.conflicts_with(right)))
    }

    /// Union of all retained query regions.
    pub fn combined(&self) -> &AccessSet {
        &self.combined
    }

    /// The retained per-query accesses.
    pub fn queries(&self) -> &[QueryAccess] {
        &self.queries
    }

    /// Union access mask for `component`, if any.
    pub fn get(&self, component: ComponentId) -> Option<AccessMask> {
        self.combined.get(component)
    }

    /// Sorted component access declarations of the union.
    pub fn accesses(&self) -> &[ComponentAccess] {
        self.combined.accesses()
    }

    /// Number of components with declared access.
    pub fn len(&self) -> usize {
        self.combined.len()
    }

    /// `true` when no component access is declared.
    pub fn is_empty(&self) -> bool {
        self.combined.is_empty()
    }
}
