//! Shared query-data definitions and raw component access helpers.
//!
//! Query data is responsible for registering component access, deciding whether
//! an archetype can be visited, and preparing typed fetches for the iterator.
//! The unsafe helpers in this module stay small: iterator code owns
//! the aliasing guarantees, while fetch code only turns validated pointers into
//! references.

/// General fetch path that also resolves sparse-set components per entity.
pub(crate) mod mixed;
/// Fast fetch path for queries whose data resolves entirely from table columns.
pub(crate) mod table;

use std::{any::type_name, marker::PhantomData};

use millarecs_macros::all_tuples;

use crate::{
    change_detection::{ChangeTick, ComponentTicks, Mut},
    component::{
        archetype::Archetype,
        sparse_set::{ComponentSparseSet, SparseSetManager},
        Component, ComponentManager, StorageKind,
    },
    entity::{metadata::ArchetypeStorageKind, Entity},
    query::{
        access::DataAccess,
        debug::{QueryDebugItem, QueryDebugSource},
        planning::{QueryRouteCandidate, RouteCandidateRole},
    },
};

use super::{state::ComponentQueryState, AccessMask, QueryError};

/// Read-only query data supported by both iterator paths.
///
/// # Safety
///
/// Implementors must uphold the safety contracts of both fetch paths.
pub unsafe trait ReadQueryData: mixed::ReadQueryData + table::ReadQueryData {}
unsafe impl<T> ReadQueryData for T where T: mixed::ReadQueryData + table::ReadQueryData {}
/// Query data that can be driven by mutable iterators.
///
/// # Safety
///
/// Implementors must declare every mutable region and never create aliases
/// outside those declarations.
pub unsafe trait WriteQueryData: mixed::WriteQueryData + table::WriteQueryData {}
unsafe impl<T> WriteQueryData for T where T: mixed::WriteQueryData + table::WriteQueryData {}

/// Returns a shared reference to a row inside a table column.
///
/// # Safety
///
/// `ptr` must point to the beginning of a valid column of `C`, and `row` must
/// be within the column bounds.
pub(crate) unsafe fn table_read_at<'w, C: Component>(
    ptr: *const C,
    len: usize,
    row: usize,
) -> &'w C {
    debug_assert!(row < len);
    unsafe { &*ptr.add(row) }
}

/// Returns a shared reference to a component stored in a sparse set.
///
/// Returns `None` if the entity does not have the component.
///
/// # Safety
///
/// `set` must point to a valid sparse set storing values of type `C`.
pub(crate) unsafe fn sparse_read<'w, C: Component>(
    set: *const ComponentSparseSet<C>,
    entity: Entity,
) -> Option<&'w C> {
    let set = unsafe { &*set };
    set.get_ref(entity)
}

/// Returns a shared sparse component reference using the driver's dense position.
///
/// This skips the entity-keyed sparse index lookup: the driving iterator
/// already knows where `expected_entity` sits in the dense arrays.
///
/// # Safety
///
/// `set` must point to a valid sparse set storing values of type `C`.
/// `dense_pos` must have been obtained from the driver index belonging to
/// `set`, and it must identify `expected_entity`.
pub(crate) unsafe fn sparse_read_by_driver_position<'w, C: Component>(
    set: *const ComponentSparseSet<C>,
    dense_pos: u32,
    expected_entity: Entity,
) -> &'w C {
    let set = unsafe { &*set };

    unsafe { set.get_ref_by_dense_pos_unchecked(dense_pos, expected_entity) }
}

/// Returns a mutable reference to a row inside a table column.
///
/// # Safety
///
/// `ptr` must point to the beginning of a valid column of `C`, `row` must be
/// within the column bounds, and no other active access to the same element may
/// exist.
pub(crate) unsafe fn table_write_at<'w, C: Component>(
    ptr: *mut C,
    ticks: *mut ComponentTicks,
    len: usize,
    row: usize,
    change_tick: ChangeTick,
) -> Mut<'w, C> {
    debug_assert!(row < len);
    unsafe { Mut::new(ptr.add(row), ticks.add(row), change_tick) }
}

/// Returns a mutable reference to a component stored in a sparse set.
///
/// Returns `None` if the entity does not have the component.
///
/// # Safety
///
/// `set` must point to a valid sparse set storing values of type `C`, and the
/// caller must guarantee exclusive access to the returned component.
pub(crate) unsafe fn sparse_write<'w, C: Component>(
    set: *const ComponentSparseSet<C>,
    entity: Entity,
    change_tick: ChangeTick,
) -> Option<Mut<'w, C>> {
    let set = unsafe { &*set };
    let (value, ticks) = unsafe { set.get_mut_ptrs(entity)? };
    Some(unsafe { Mut::new(value, ticks, change_tick) })
}

/// Returns mutable sparse component access using the driver's dense position.
///
/// # Safety
///
/// `dense_pos` must have been obtained from the driver index belonging to
/// `set`, and it must identify `expected_entity`. The caller must guarantee
/// exclusive access and prevent structural mutation while the result is alive.
#[inline]
pub(crate) unsafe fn sparse_write_by_driver_position<'w, C: Component>(
    set: *const ComponentSparseSet<C>,
    dense_pos: u32,
    expected_entity: Entity,
    change_tick: ChangeTick,
) -> Mut<'w, C> {
    let set = unsafe { &*set };
    let (value, ticks) =
        unsafe { set.get_mut_ptrs_by_dense_pos_unchecked(dense_pos, expected_entity) };
    unsafe { Mut::new(value, ticks, change_tick) }
}

/// Initializes the query state for a component.
///
/// This also registers the requested access in `DataAccess`, allowing conflicts
/// such as `&mut T` together with `&T` to be detected.
pub(crate) fn init_component_state<C: Component>(
    cm: &ComponentManager,
    access: &mut DataAccess,
    mask: AccessMask,
) -> Result<ComponentQueryState, QueryError> {
    let state = ComponentQueryState::new::<C>(cm)?;
    let id = state.id;
    access.add_required(id, mask)?;
    Ok(state)
}

/// Initializes the query state for an optional component.
///
/// This registers the component as optional in `DataAccess`: conflicting
/// access is still detected, but absence alone never fails the query.
pub(crate) fn init_optional_component_state<C: Component>(
    cm: &ComponentManager,
    access: &mut DataAccess,
    mask: AccessMask,
) -> Result<ComponentQueryState, QueryError> {
    let state = ComponentQueryState::new::<C>(cm)?;
    let id = state.id;
    access.add_optional(id, mask)?;
    Ok(state)
}

/// Describes the data requested by a query.
///
/// This trait does not fetch values directly. Its responsibilities are:
///
/// - initializing cached query state;
/// - registering read or write access;
/// - deciding whether an archetype may contain valid results;
/// - reporting whether the query can be resolved using archetype information
///   only.
///
/// Implemented for `Entity`, `&T`, `&mut T`, `Option<&T>`,
/// `Option<&mut T>`, `()`, and tuples.
///
/// # Safety
///
/// Implementors must register every component region used by their fetches.
/// The registered access must be sufficient to prevent all reference aliasing
/// and data races that the implementation could otherwise create.
pub unsafe trait QueryData {
    /// Cached state required to execute the query.
    type State: Send + 'static;

    /// Initializes the query state and registers component access.
    fn init_state(
        cm: &ComponentManager,
        access: &mut DataAccess,
    ) -> Result<Self::State, QueryError>;

    /// Returns whether an archetype may contain data compatible with this query.
    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool;

    /// Returns whether this query can be resolved using archetype information only.
    ///
    /// Queries that depend on sparse sets are not fully archetypal, because each
    /// entity may or may not contain the component individually.
    fn is_archetypal(state: &Self::State) -> bool;

    /// Reports sparse-route candidacy to the planner.
    ///
    /// Data sources that resolve through a sparse set push a candidate here;
    /// the default contributes none.
    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        let _ = state;
        let _ = out;
    }

    /// Reports structured diagnostics for the debug report.
    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        let _ = state;
        let _ = out;
    }
}

unsafe impl QueryData for Entity {
    type State = ();

    fn init_state(_: &ComponentManager, _: &mut DataAccess) -> Result<Self::State, QueryError> {
        Ok(())
    }

    fn matches_archetype(_: &Self::State, _: &Archetype) -> bool {
        true
    }

    fn is_archetypal(_: &Self::State) -> bool {
        true
    }

    fn debug_items(_: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::entity());
    }
}

unsafe impl<C: Component> QueryData for &C {
    type State = ComponentQueryState;

    fn init_state(
        cm: &ComponentManager,
        access: &mut DataAccess,
    ) -> Result<Self::State, QueryError> {
        init_component_state::<C>(cm, access, AccessMask::VALUE_READ)
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        match state.storage {
            StorageKind::Table => archetype.signature().contains(state.id),
            StorageKind::Sparse => true,
        }
    }

    fn is_archetypal(state: &Self::State) -> bool {
        matches!(state.storage, StorageKind::Table)
    }

    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        out.push(QueryRouteCandidate::required_data(state.id, state.storage));
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::component(
            QueryDebugSource::Data,
            "&T",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::VALUE_READ,
            Some(RouteCandidateRole::RequiredData),
        ));
    }
}

unsafe impl<C: Component> QueryData for &mut C {
    type State = ComponentQueryState;

    fn init_state(
        cm: &ComponentManager,
        access: &mut DataAccess,
    ) -> Result<Self::State, QueryError> {
        init_component_state::<C>(
            cm,
            access,
            AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE,
        )
    }

    fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
        match state.storage {
            StorageKind::Table => archetype.signature().contains(state.id),
            StorageKind::Sparse => true,
        }
    }

    fn is_archetypal(state: &Self::State) -> bool {
        matches!(state.storage, StorageKind::Table)
    }

    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        out.push(QueryRouteCandidate::required_data(state.id, state.storage));
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::component(
            QueryDebugSource::Data,
            "&mut T",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE,
            Some(RouteCandidateRole::RequiredData),
        ));
    }
}

unsafe impl<C: Component> QueryData for Option<&C> {
    type State = ComponentQueryState;

    fn init_state(
        cm: &ComponentManager,
        access: &mut DataAccess,
    ) -> Result<Self::State, QueryError> {
        init_optional_component_state::<C>(cm, access, AccessMask::VALUE_READ)
    }

    fn matches_archetype(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn is_archetypal(state: &Self::State) -> bool {
        matches!(state.storage, StorageKind::Table)
    }

    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        out.push(QueryRouteCandidate::optional_data(state.id, state.storage));
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::optional_data(
            "Option<&T>",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::VALUE_READ,
        ));
    }
}

unsafe impl<C: Component> QueryData for Option<&mut C> {
    type State = ComponentQueryState;

    fn init_state(
        cm: &ComponentManager,
        access: &mut DataAccess,
    ) -> Result<Self::State, QueryError> {
        init_optional_component_state::<C>(
            cm,
            access,
            AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE,
        )
    }

    fn matches_archetype(_state: &Self::State, _archetype: &Archetype) -> bool {
        true
    }

    fn is_archetypal(state: &Self::State) -> bool {
        matches!(state.storage, StorageKind::Table)
    }

    fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
        out.push(QueryRouteCandidate::optional_data(state.id, state.storage));
    }

    fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
        out.push(QueryDebugItem::optional_data(
            "Option<&mut T>",
            type_name::<C>(),
            state.id,
            state.storage,
            AccessMask::VALUE_WRITE | AccessMask::TICKS_WRITE,
        ));
    }
}

unsafe impl QueryData for () {
    type State = ();

    fn init_state(_: &ComponentManager, _: &mut DataAccess) -> Result<Self::State, QueryError> {
        Ok(())
    }

    fn matches_archetype(_: &Self::State, _: &Archetype) -> bool {
        true
    }

    fn is_archetypal(_: &Self::State) -> bool {
        true
    }
}

/// Implements `QueryData` for tuples of query data.
///
/// State initialization, archetype matching, and route or debug reporting
/// compose element-wise.
macro_rules! tuple_query_data_impl {
    ($(#[$meta:meta])* $($T:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($T),*> QueryData for ($($T,)*)
        where
            $($T: QueryData),*
        {
            type State = ($($T::State,)*);

            fn init_state(cm: &ComponentManager, access: &mut DataAccess) -> Result<Self::State, QueryError> {
                Ok((
                    $($T::init_state(cm, access)?,)*
                ))
            }

            fn matches_archetype(state: &Self::State, archetype: &Archetype) -> bool {
                let ($(ref $T,)*) = *state;
                $($T::matches_archetype($T, archetype) &&)* true
            }

            fn is_archetypal(state: &Self::State) -> bool {
                let ($(ref $T,)*) = *state;
                $($T::is_archetypal($T) &&)* true
            }

            fn route_candidates(state: &Self::State, out: &mut Vec<QueryRouteCandidate>) {
                let ($(ref $T,)*) = *state;
                $($T::route_candidates($T, out);)*
            }

            fn debug_items(state: &Self::State, out: &mut Vec<QueryDebugItem>) {
                let ($(ref $T,)*) = *state;
                $($T::debug_items($T, out);)*
            }
        }
    };
}

all_tuples!(tuple_query_data_impl, 1, 12, Q);
