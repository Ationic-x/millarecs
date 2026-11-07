//! Access metadata used by system initialization and scheduling.
//!
//! System parameters declare which component and resource types they read or
//! write. Tuple parameters merge those declarations into a single [`SystemAccess`]
//! value before any runtime references are materialized, allowing invalid aliases
//! such as `Res<T> + ResMut<T>` or `Query<&T> + Query<&mut T>` to fail early.

use std::any::TypeId;

use crate::{
    query::{FilteredAccessSet, QueryAccess},
    resource::{ResourceError, ResourceId},
    world::WorldError,
};

/// Errors raised while validating system parameter access.
#[derive(Debug)]
pub enum SystemAccessError {
    /// The same unique parameter appeared more than once in one system.
    DuplicateUniqueParam(SystemParamId),
}

impl std::fmt::Display for SystemAccessError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DuplicateUniqueParam(id) => {
                write!(f, "system parameter `{}` can only appear once", id.name(),)
            }
        }
    }
}

impl std::error::Error for SystemAccessError {}

/// Read/write access declared for resources by one parameter or system.
///
/// The internal vectors are kept sorted and deduplicated. This keeps membership
/// checks simple through `binary_search` and makes tests/debugging deterministic.
#[derive(Default, Clone)]
pub struct ResourceAccess {
    reads: Vec<ResourceId>,
    writes: Vec<ResourceId>,
}

impl ResourceAccess {
    /// Adds shared access to `id`.
    ///
    /// Multiple reads of the same resource are allowed and deduplicated. A read
    /// conflicts with an existing write to the same resource.
    pub fn add_read(&mut self, id: ResourceId) -> Result<(), ResourceError> {
        if self.contains_write(&id) {
            return Err(ResourceError::AccessConflict(id));
        }

        Self::insert_sorted(&mut self.reads, id);
        Ok(())
    }

    /// Adds exclusive access to `id`.
    ///
    /// A write conflicts with any existing read or write of the same resource.
    pub fn add_write(&mut self, id: ResourceId) -> Result<(), ResourceError> {
        if self.contains_read(&id) || self.contains_write(&id) {
            return Err(ResourceError::AccessConflict(id));
        }

        Self::insert_sorted(&mut self.writes, id);
        Ok(())
    }

    /// Whether `id` is declared as a read.
    fn contains_read(&self, id: &ResourceId) -> bool {
        self.reads.binary_search(id).is_ok()
    }

    /// Whether `id` is declared as a write.
    fn contains_write(&self, id: &ResourceId) -> bool {
        self.writes.binary_search(id).is_ok()
    }

    /// Inserts `id` keeping the vector sorted; an existing entry is kept as is.
    fn insert_sorted(vec: &mut Vec<ResourceId>, id: ResourceId) {
        match vec.binary_search(&id) {
            Ok(_) => {}
            Err(index) => vec.insert(index, id),
        }
    }

    /// Whether `self` and `other` cannot be active at the same time.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.writes
            .iter()
            .any(|id| other.contains_write(id) || other.contains_read(id))
            || self.reads.iter().any(|id| other.contains_write(id))
    }

    /// Merges another resource access set into this one.
    ///
    /// Used while building tuple system parameters: this is the step where two
    /// members aliasing the same resource, such as `Res<T>` next to `ResMut<T>`,
    /// are rejected. The merge mutates in place and stops at the first
    /// conflict, so entries added before the failure stay in the set.
    pub(crate) fn extend_from(&mut self, other: &Self) -> Result<(), ResourceError> {
        for &id in &other.reads {
            self.add_read(id)?;
        }

        for &id in &other.writes {
            self.add_write(id)?;
        }

        Ok(())
    }

    /// Returns the sorted set of resource reads.
    pub fn reads(&self) -> &[ResourceId] {
        &self.reads
    }

    /// Returns the sorted set of resource writes.
    pub fn writes(&self) -> &[ResourceId] {
        &self.writes
    }

    /// `true` when no resource access is declared.
    pub fn is_empty(&self) -> bool {
        self.reads.is_empty() && self.writes.is_empty()
    }
}

/// Identity of a unique system parameter.
///
/// Combines the concrete type with its full type name so diagnostics and
/// conflict reports stay readable.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SystemParamId {
    type_id: TypeId,
    name: &'static str,
}

impl SystemParamId {
    /// Creates the identifier for the parameter type `T`.
    pub fn of<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            name: std::any::type_name::<T>(),
        }
    }

    /// Returns the [`TypeId`] part of the identifier.
    pub fn type_id(self) -> TypeId {
        self.type_id
    }

    /// Returns the human-readable type name.
    pub fn name(self) -> &'static str {
        self.name
    }
}

/// Set of unique system parameters declared by one system.
///
/// Unique parameters (event writers, exclusive views) may appear at most once
/// per system; the schedule rejects tuples that repeat them.
#[derive(Default, Clone)]
pub struct UniqueParamAccess {
    /// Declared parameters, kept sorted for binary-search lookups.
    params: Vec<SystemParamId>,
}

impl UniqueParamAccess {
    /// Adds `id`, rejecting a duplicate.
    pub fn add(&mut self, id: SystemParamId) -> Result<(), SystemAccessError> {
        match self.params.binary_search(&id) {
            Ok(_) => Err(SystemAccessError::DuplicateUniqueParam(id)),

            Err(index) => {
                self.params.insert(index, id);
                Ok(())
            }
        }
    }

    /// Merges `other` into this set, failing on the first parameter repeated
    /// by both sides of a tuple.
    pub(crate) fn extend_from(&mut self, other: &Self) -> Result<(), SystemAccessError> {
        for &id in &other.params {
            self.add(id)?;
        }

        Ok(())
    }

    /// Whether `id` is declared.
    pub fn contains(&self, id: SystemParamId) -> bool {
        self.params.binary_search(&id).is_ok()
    }

    /// Whether the two sets share any unique parameter.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.params.iter().any(|id| other.contains(*id))
    }

    /// Returns the declared parameter identifiers.
    pub fn params(&self) -> &[SystemParamId] {
        &self.params
    }

    /// `true` when no unique parameter is declared.
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
}

/// Access declared by a whole system.
///
/// This combines component/query access with resource access so system
/// parameters can be validated as a single unit before the scheduler runs them.
#[derive(Default, Clone)]
pub struct SystemAccess {
    components: FilteredAccessSet,
    resources: ResourceAccess,
    unique_params: UniqueParamAccess,
    /// Flag for systems that need the whole world; when set, it short-circuits
    /// every conflict check.
    exclusive: bool,
}

impl SystemAccess {
    /// Creates an empty system access.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builds the access from a query declaration.
    pub fn from_query(query: QueryAccess) -> Self {
        Self {
            components: FilteredAccessSet::from_query(query),
            resources: ResourceAccess::default(),
            unique_params: UniqueParamAccess::default(),
            exclusive: false,
        }
    }

    /// Creates a system access that reads one resource type.
    pub fn from_resource_read(type_id: ResourceId) -> Result<Self, WorldError> {
        let mut resources = ResourceAccess::default();
        resources.add_read(type_id)?;

        Ok(Self {
            components: FilteredAccessSet::default(),
            resources,
            unique_params: UniqueParamAccess::default(),
            exclusive: false,
        })
    }

    /// Creates a system access that writes one resource type.
    pub fn from_resource_write(type_id: ResourceId) -> Result<Self, WorldError> {
        let mut resources = ResourceAccess::default();
        resources.add_write(type_id)?;

        Ok(Self {
            components: FilteredAccessSet::default(),
            resources,
            unique_params: UniqueParamAccess::default(),
            exclusive: false,
        })
    }

    /// Creates an access set conflicting with every other system.
    ///
    /// Used by exclusive systems, which receive `&mut World` and therefore
    /// cannot overlap with anything.
    pub fn exclusive() -> Self {
        Self {
            components: FilteredAccessSet::default(),
            resources: ResourceAccess::default(),
            unique_params: UniqueParamAccess::default(),
            exclusive: true,
        }
    }

    /// Builds the access from one unique parameter type `T`.
    pub fn from_unique_param<T: 'static>() -> Self {
        let mut unique_params = UniqueParamAccess::default();

        unique_params
            .add(SystemParamId::of::<T>())
            .expect("a fresh unique parameter set cannot contain duplicates");

        Self {
            components: FilteredAccessSet::default(),
            resources: ResourceAccess::default(),
            unique_params,
            exclusive: false,
        }
    }

    /// Returns the unique parameter access part.
    pub fn unique_params(&self) -> &UniqueParamAccess {
        &self.unique_params
    }

    /// Returns the component/query access part.
    pub fn components(&self) -> &FilteredAccessSet {
        &self.components
    }

    /// Returns the resource access part.
    pub fn resources(&self) -> &ResourceAccess {
        &self.resources
    }

    /// Whether this system conflicts with `other`.
    ///
    /// Sequential schedules may ignore this for ordering, but the same metadata
    /// is needed to reject invalid parameter tuples and to support
    /// parallel scheduling.
    pub fn conflicts_with(&self, other: &Self) -> bool {
        self.exclusive
            || other.exclusive
            || self.components.conflicts_with(&other.components)
            || self.resources.conflicts_with(&other.resources)
            || self.unique_params.conflicts_with(&other.unique_params)
    }

    /// Merges another system access into this one.
    ///
    /// This is used when building tuple system params:
    ///
    /// `(Query<&A>, Res<B>, ResMut<C>)`
    ///
    /// becomes one combined access set, and this merge is exactly where invalid
    /// aliases between tuple members fail. The merge runs on a clone, so a
    /// conflicting `other` leaves `self` untouched.
    pub(crate) fn extend_from(&mut self, other: &Self) -> Result<(), WorldError> {
        let mut merged = self.clone();

        merged.components.extend_from(&other.components)?;

        merged.resources.extend_from(&other.resources)?;

        merged.unique_params.extend_from(&other.unique_params)?;

        *self = merged;

        Ok(())
    }

    /// Whether this access set declares no component, resource, or
    /// unique-parameter access.
    pub fn is_empty(&self) -> bool {
        !self.exclusive
            && self.components.is_empty()
            && self.resources.is_empty()
            && self.unique_params.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Time;
    struct Counter;

    #[test]
    fn resource_access_starts_empty() {
        let access = ResourceAccess::default();

        assert!(access.is_empty());
        assert!(access.reads().is_empty());
        assert!(access.writes().is_empty());
    }

    #[test]
    fn resource_access_deduplicates_duplicate_reads() {
        let mut access = ResourceAccess::default();
        let id = ResourceId::of::<Time>();

        access.add_read(id).unwrap();
        access.add_read(id).unwrap();

        assert_eq!(access.reads(), &[id]);
        assert!(access.writes().is_empty());
    }

    #[test]
    fn resource_access_rejects_read_then_write_of_same_resource() {
        let mut access = ResourceAccess::default();
        let id = ResourceId::of::<Time>();

        access.add_read(id).unwrap();

        assert!(matches!(
            access.add_write(id),
            Err(ResourceError::AccessConflict(conflict)) if conflict == id
        ));
    }

    #[test]
    fn resource_access_rejects_write_then_read_of_same_resource() {
        let mut access = ResourceAccess::default();
        let id = ResourceId::of::<Time>();

        access.add_write(id).unwrap();

        assert!(matches!(
            access.add_read(id),
            Err(ResourceError::AccessConflict(conflict)) if conflict == id
        ));
    }

    #[test]
    fn resource_access_rejects_duplicate_writes() {
        let mut access = ResourceAccess::default();
        let id = ResourceId::of::<Time>();

        access.add_write(id).unwrap();

        assert!(matches!(
            access.add_write(id),
            Err(ResourceError::AccessConflict(conflict)) if conflict == id
        ));
    }

    #[test]
    fn resource_access_allows_read_one_resource_and_write_another() {
        let mut access = ResourceAccess::default();
        let time = ResourceId::of::<Time>();
        let counter = ResourceId::of::<Counter>();

        access.add_read(time).unwrap();
        access.add_write(counter).unwrap();

        assert_eq!(access.reads(), &[time]);
        assert_eq!(access.writes(), &[counter]);
    }

    #[test]
    fn resource_access_reports_conflicts_between_sets() {
        let mut reader = ResourceAccess::default();
        let mut writer = ResourceAccess::default();
        let id = ResourceId::of::<Time>();

        reader.add_read(id).unwrap();
        writer.add_write(id).unwrap();

        assert!(reader.conflicts_with(&writer));
        assert!(writer.conflicts_with(&reader));
    }

    #[test]
    fn resource_access_allows_disjoint_sets() {
        let mut first = ResourceAccess::default();
        let mut second = ResourceAccess::default();

        first.add_read(ResourceId::of::<Time>()).unwrap();
        second.add_write(ResourceId::of::<Counter>()).unwrap();

        assert!(!first.conflicts_with(&second));
        assert!(!second.conflicts_with(&first));
    }

    #[test]
    fn system_access_is_not_empty_when_it_only_contains_resources() {
        let access = SystemAccess::from_resource_read(ResourceId::of::<Time>()).unwrap();

        assert!(!access.is_empty());
        assert_eq!(access.resources().reads(), &[ResourceId::of::<Time>()]);
        assert!(access.components().is_empty());
    }

    #[test]
    fn system_access_extend_rejects_resource_conflict() {
        let mut access = SystemAccess::from_resource_read(ResourceId::of::<Time>()).unwrap();
        let write = SystemAccess::from_resource_write(ResourceId::of::<Time>()).unwrap();

        assert!(matches!(
            access.extend_from(&write),
            Err(WorldError::Resource(_))
        ));
    }
}
