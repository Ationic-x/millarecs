//! General fetch path for table and sparse-set data.
//!
//! Sparse-set components are checked per entity, so required sparse components
//! may reject individual rows even when the archetype itself is eligible.

use crate::{component::ComponentId, utils::prefetch::prefetch_element};

use super::*;

/// Read-only fetch for a required component.
///
/// The component may be stored either in an archetype table or in a sparse set.
pub enum ReadCompFetch<'w, C: Component> {
    /// Required component stored in an archetype table.
    Table {
        /// Pointer to the first element of the column.
        ptr: *const C,
        /// Number of rows available in the column.
        len: usize,
        /// Binds the raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w C>,
    },

    /// Sparse component resolved through its entity identifier.
    SparseEntity {
        /// Pointer to the sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Binds the raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w C>,
    },

    /// Sparse component driven by this same set, so the dense position is known.
    SparseDriver {
        /// Pointer to the driving sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Binds the raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w C>,
    },
}

/// Read-only fetch for an optional component.
///
/// If the component does not exist in the current archetype, `Absent` is used.
pub enum OptReadCompFetch<'w, C: Component> {
    /// The current archetype does not contain this component column.
    Absent,

    /// Optional component stored in an archetype table.
    Table {
        /// Pointer to the first element of the column.
        ptr: *const C,
        /// Number of rows available in the column.
        len: usize,
        /// Binds the raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w C>,
    },

    /// Optional component stored in a sparse set.
    SparseEntity {
        /// Pointer to the sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Binds the raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w C>,
    },

    /// Optional sparse component guaranteed by the selected sparse driver.
    SparseDriver {
        /// Pointer to the driving sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Binds the raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w C>,
    },
}

/// Creates a read-only fetch for a required component.
///
/// # Safety
///
/// `archetype` and `sparse_sets` must be valid pointers obtained from the same
/// `ComponentManager` used to create the query state. They must remain valid
/// for at least `'w`.
pub(crate) unsafe fn init_read_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
) -> Result<ReadCompFetch<'w, C>, QueryError> {
    match state.storage {
        StorageKind::Table => {
            debug_assert!(matches!(state.storage, StorageKind::Table));

            let arch = unsafe { &*archetype };
            let (ptr, len) = unsafe { arch.get_column_read_ptrs::<C>(storage, state.id)? };

            Ok(ReadCompFetch::Table {
                ptr,
                len,
                _marker: PhantomData,
            })
        }
        StorageKind::Sparse => {
            let sets = unsafe { &*sparse_sets };
            let set = sets
                .get_sparse_set(state.id.index())
                .ok_or(QueryError::StorageUnknown(state.id))?;

            Ok(ReadCompFetch::SparseEntity {
                set,
                _marker: PhantomData,
            })
        }
    }
}

/// Creates a read-only fetch for a required component, aware of the sparse
/// driver.
///
/// When the component is the driver itself, the `SparseEntity` fetch is
/// promoted to `SparseDriver`, which resolves members by dense position
/// instead of the entity-keyed sparse index. Every other fetch is returned
/// unchanged.
///
/// # Safety
///
/// Same contract as `init_read_fetch`; `driver` must be the component id
/// selected as sparse driver for this run.
pub(crate) unsafe fn init_sparse_driver_read_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
    driver: ComponentId,
) -> Result<ReadCompFetch<'w, C>, QueryError> {
    let fetch = unsafe { init_read_fetch::<C>(state, archetype, sparse_sets, storage) }?;

    match fetch {
        ReadCompFetch::SparseEntity { set, .. } if state.id == driver => {
            Ok(ReadCompFetch::SparseDriver {
                set,
                _marker: PhantomData,
            })
        }

        other => Ok(other),
    }
}

/// Creates a read-only fetch for an optional component.
///
/// For table-stored components, this returns `Absent` when the current
/// archetype does not contain the component column.
///
/// # Safety
///
/// `archetype` and `sparse_sets` must be valid pointers obtained from the same
/// `ComponentManager` used to create the query state. They must remain valid
/// for at least `'w`.
pub(crate) unsafe fn init_opt_read_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
) -> Result<OptReadCompFetch<'w, C>, QueryError> {
    match state.storage {
        StorageKind::Table => {
            let arch = unsafe { &*archetype };

            if arch.signature().contains(state.id) {
                let (ptr, len) = unsafe { arch.get_column_read_ptrs::<C>(storage, state.id)? };
                Ok(OptReadCompFetch::Table {
                    ptr,
                    len,
                    _marker: PhantomData,
                })
            } else {
                Ok(OptReadCompFetch::Absent)
            }
        }
        StorageKind::Sparse => {
            let sets = unsafe { &*sparse_sets };
            let set = sets
                .get_sparse_set(state.id.index())
                .ok_or(QueryError::StorageUnknown(state.id))?;

            Ok(OptReadCompFetch::SparseEntity {
                set,
                _marker: PhantomData,
            })
        }
    }
}

/// Creates a read-only fetch for an optional component, aware of the sparse
/// driver.
///
/// Promotes the optional `SparseEntity` fetch to `SparseDriver` when the
/// component is the driver itself, mirroring `init_sparse_driver_read_fetch`.
///
/// # Safety
///
/// Same contract as `init_read_fetch`; `driver` must be the component id
/// selected as sparse driver for this run.
pub(crate) unsafe fn init_sparse_driver_opt_read_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
    driver: ComponentId,
) -> Result<OptReadCompFetch<'w, C>, QueryError> {
    let fetch = unsafe { init_opt_read_fetch::<C>(state, archetype, sparse_sets, storage) }?;

    match fetch {
        OptReadCompFetch::SparseEntity { set, .. } if state.id == driver => {
            Ok(OptReadCompFetch::SparseDriver {
                set,
                _marker: PhantomData,
            })
        }

        other => Ok(other),
    }
}

/// Extension of `QueryData` for read-only queries.
///
/// Used by `QueryIter`.
///
/// # Safety
///
/// Implementors must only create shared references covered by the access
/// declared through `QueryData`, and every returned item must remain within the
/// storage represented by its fetch lifetime.
pub unsafe trait ReadQueryData: QueryData {
    /// Fetch prepared for a specific archetype.
    type Fetch<'w>;

    /// Item returned by the iterator.
    type Item<'w>;

    /// Initializes the read-only fetch for a specific archetype.
    ///
    /// # Safety
    ///
    /// The received pointers must be valid for `'w` and compatible with the
    /// query state.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
    ) -> Result<Self::Fetch<'w>, QueryError>;

    /// Initializes the fetch knowing which component drives sparse iteration.
    ///
    /// The default ignores the driver and prepares a regular fetch; members
    /// that can use the driver's dense position override this to promote
    /// their fetch variant.
    ///
    /// # Safety
    ///
    /// Same contract as `init_fetch`; `_driver`, when used, must be the
    /// component id selected as sparse driver for this run.
    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        _driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { Self::init_fetch(state, archetype, sparse_sets, storage) }
    }
    /// Fetches the query item for a specific row and entity.
    ///
    /// Returns `None` when the entity does not satisfy a required part of the
    /// query, such as a missing required sparse-set component.
    ///
    /// # Safety
    ///
    /// The fetch must have been initialized for the archetype that contains
    /// `row` and `entity`.
    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>>;

    /// Fetches the query item using the driver's dense position.
    ///
    /// `sparse_dense_pos` locates `entity` in the driving sparse set, letting
    /// driver-backed members skip the entity-keyed index lookup. The default
    /// discards it and falls back to `fetch`.
    ///
    /// # Safety
    ///
    /// Same contract as `fetch`; `sparse_dense_pos` must come from the driving
    /// sparse set and identify `entity`.
    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        let _ = sparse_dense_pos;
        unsafe { Self::fetch(fetch, row, entity) }
    }

    /// Whether the prepared fetch reads memory that row lookahead can warm.
    ///
    /// Table columns stride across cache lines on sparse-driver lanes, and
    /// entity-keyed sparse members pay a dependent index load; the driver's
    /// own dense arrays stream sequentially and need no help.
    fn has_row_prefetch(_fetch: &Self::Fetch<'_>) -> bool {
        false
    }

    /// Warms the cache lines a future `fetch_sparse_driver` of this row touches.
    ///
    /// # Safety
    ///
    /// `row` must be in bounds for any table column in the fetch, as with
    /// `fetch_sparse_driver`.
    unsafe fn prefetch_sparse_driver_row(_fetch: &Self::Fetch<'_>, _row: usize, _entity: Entity) {}
}

unsafe impl ReadQueryData for Entity {
    type Fetch<'w> = ();
    type Item<'w> = Entity;

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: *const SparseSetManager,
        _: ArchetypeStorageKind,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        Ok(())
    }

    unsafe fn fetch<'w>(
        _: &mut Self::Fetch<'w>,
        _: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        Some(entity)
    }
}

unsafe impl<C: Component> ReadQueryData for &C {
    type Fetch<'w> = ReadCompFetch<'w, C>;
    type Item<'w> = &'w C;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_read_fetch(state, archetype, sparse_sets, storage) }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_sparse_driver_read_fetch(state, archetype, sparse_sets, storage, driver) }
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        match fetch {
            ReadCompFetch::Table { ptr, len, .. } => {
                Some(unsafe { table_read_at(*ptr, *len, row) })
            }
            ReadCompFetch::SparseEntity { set, .. } | ReadCompFetch::SparseDriver { set, .. } => unsafe {
                sparse_read(*set, entity)
            },
        }
    }

    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        match fetch {
            ReadCompFetch::SparseDriver { set, .. } => {
                Some(unsafe { sparse_read_by_driver_position(*set, sparse_dense_pos, entity) })
            }

            _ => unsafe { <&C as ReadQueryData>::fetch(fetch, row, entity) },
        }
    }

    #[inline]
    fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
        !matches!(fetch, ReadCompFetch::SparseDriver { .. })
    }

    #[inline(always)]
    unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
        match fetch {
            ReadCompFetch::Table { ptr, .. } => prefetch_element(*ptr, row),
            ReadCompFetch::SparseEntity { set, .. } => {
                let set = unsafe { &**set };
                set.prefetch_membership_line(entity);
            }
            _ => {}
        }
    }
}

unsafe impl<C: Component> ReadQueryData for Option<&C> {
    type Fetch<'w> = OptReadCompFetch<'w, C>;
    type Item<'w> = Option<&'w C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_opt_read_fetch(state, archetype, sparse_sets, storage) }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_sparse_driver_opt_read_fetch(state, archetype, sparse_sets, storage, driver) }
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        let item = match fetch {
            OptReadCompFetch::Absent => None,
            OptReadCompFetch::Table { ptr, len, .. } => {
                Some(unsafe { table_read_at(*ptr, *len, row) })
            }
            OptReadCompFetch::SparseEntity { set, .. }
            | OptReadCompFetch::SparseDriver { set, .. } => unsafe { sparse_read(*set, entity) },
        };

        Some(item)
    }

    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        match fetch {
            OptReadCompFetch::SparseDriver { set, .. } => Some(Some(unsafe {
                sparse_read_by_driver_position(*set, sparse_dense_pos, entity)
            })),

            _ => unsafe { <Option<&C> as ReadQueryData>::fetch(fetch, row, entity) },
        }
    }

    #[inline]
    fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
        !matches!(fetch, OptReadCompFetch::SparseDriver { .. })
    }

    #[inline(always)]
    unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
        match fetch {
            OptReadCompFetch::Table { ptr, .. } => prefetch_element(*ptr, row),
            OptReadCompFetch::SparseEntity { set, .. } => {
                let set = unsafe { &**set };
                set.prefetch_membership_line(entity);
            }
            _ => {}
        }
    }
}

unsafe impl ReadQueryData for () {
    type Fetch<'w> = ();
    type Item<'w> = ();

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: *const SparseSetManager,
        _: ArchetypeStorageKind,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        Ok(())
    }

    unsafe fn fetch<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> Option<Self::Item<'w>> {
        Some(())
    }
}

/// Implements `ReadQueryData` for tuples of readable query data.
///
/// Initialization, fetching, and prefetching compose element-wise; a `None`
/// from any element rejects the whole row, and `has_row_prefetch` folds with
/// `||` so lookahead happens only when some member benefits.
macro_rules! tuple_read_query_data_impl {
    ($(#[$meta:meta])* $($T:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($T),*> ReadQueryData for ($($T,)*)
        where
            $($T: ReadQueryData),*
        {
            type Fetch<'w> = ($($T::Fetch<'w>,)*);
            type Item<'w> = ($($T::Item<'w>,)*);

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                sparse_sets: *const SparseSetManager,
                storage: ArchetypeStorageKind,
            ) -> Result<Self::Fetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;
                Ok((
                    $( unsafe { $T::init_fetch($T, archetype, sparse_sets, storage)? },)*
                ))
            }

            unsafe fn init_sparse_driver_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                sparse_sets: *const SparseSetManager,
                storage: ArchetypeStorageKind,
                driver: ComponentId,
            ) -> Result<Self::Fetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;

                Ok((
                    $(
                        unsafe {
                            $T::init_sparse_driver_fetch(
                                $T,
                                archetype,
                                sparse_sets,
                                storage,
                                driver,
                            )?
                        },
                    )*
                ))
            }

            unsafe fn fetch<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
            ) -> Option<Self::Item<'w>> {
                let ($(ref mut $T,)*) = *fetch;
                Some((
                    $( unsafe { $T::fetch($T, row, entity)? },)*
                ))
            }

            unsafe fn fetch_sparse_driver<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
                sparse_dense_pos: u32,
            ) -> Option<Self::Item<'w>> {
                let ($(ref mut $T,)*) = *fetch;

                Some((
                    $(
                        unsafe {
                            $T::fetch_sparse_driver(
                                $T,
                                row,
                                entity,
                                sparse_dense_pos,
                            )?
                        },
                    )*
                ))
            }

            #[inline]
            fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
                let ($(ref $T,)*) = *fetch;
                false $(|| $T::has_row_prefetch($T))*
            }

            #[inline(always)]
            unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
                let ($(ref $T,)*) = *fetch;
                $(unsafe { <$T as ReadQueryData>::prefetch_sparse_driver_row($T, row, entity) };)*
            }
        }
    };
}

all_tuples!(tuple_read_query_data_impl, 1, 12, R);

/// Mutable fetch for a required component.
///
/// The component may be stored either in an archetype table or in a sparse set.
pub enum WriteCompFetch<'w, C: Component> {
    /// Component stored in an archetype table.
    Table {
        /// Pointer to the first mutable element of the column.
        ptr: *mut C,
        /// Pointer to the first tick entry of the column.
        component_ticks: *mut ComponentTicks,
        /// Number of rows available in the column.
        len: usize,
        /// Tick used when this mutable component is actually changed.
        change_tick: ChangeTick,
        /// Binds the mutable raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w mut C>,
    },

    /// Sparse component resolved through its entity identifier.
    SparseEntity {
        /// Pointer to the sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Tick used when this mutable component is actually changed.
        change_tick: ChangeTick,
        /// Binds the mutable raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w mut C>,
    },

    /// Sparse component driven by this same set, so the dense position is known.
    SparseDriver {
        /// Pointer to the driving sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Tick used when this mutable component is actually changed.
        change_tick: ChangeTick,
        /// Binds the mutable raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w mut C>,
    },
}

/// Mutable fetch for an optional component.
///
/// If the component does not exist in the current archetype, `Absent` is used.
pub enum OptWriteCompFetch<'w, C: Component> {
    /// The current archetype does not contain this component column.
    Absent,

    /// Optional component stored in an archetype table.
    Table {
        /// Pointer to the first mutable element of the column.
        ptr: *mut C,
        /// Pointer to the first tick entry of the column.
        component_ticks: *mut ComponentTicks,
        /// Number of rows available in the column.
        len: usize,
        /// Tick used when this mutable component is actually changed.
        change_tick: ChangeTick,
        /// Binds the mutable raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w mut C>,
    },

    /// Optional sparse component resolved through its entity identifier.
    SparseEntity {
        /// Pointer to the sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Tick used when this mutable component is actually changed.
        change_tick: ChangeTick,
        /// Binds the mutable raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w mut C>,
    },

    /// Optional sparse component guaranteed by the selected sparse driver.
    SparseDriver {
        /// Pointer to the driving sparse set holding the component values.
        set: *const ComponentSparseSet<C>,
        /// Tick used when this mutable component is actually changed.
        change_tick: ChangeTick,
        /// Binds the mutable raw pointer to the iterator lifetime.
        _marker: PhantomData<&'w mut C>,
    },
}

/// Creates a mutable fetch for a required component.
///
/// # Safety
///
/// `archetype` and `sparse_sets` must remain valid for `'w`. The caller must
/// enforce exclusive value and tick access for every written component.
pub(crate) unsafe fn init_write_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
    change_tick: ChangeTick,
) -> Result<WriteCompFetch<'w, C>, QueryError> {
    match state.storage {
        StorageKind::Table => {
            let arch = unsafe { &*archetype };
            unsafe { arch.note_column_write_access(storage, state.id, change_tick) };
            let (ptr, component_ticks, len) =
                unsafe { arch.get_column_write_ptrs::<C>(storage, state.id)? };
            Ok(WriteCompFetch::Table {
                ptr,
                component_ticks,
                len,
                change_tick,
                _marker: PhantomData,
            })
        }
        StorageKind::Sparse => {
            let sets = unsafe { &*sparse_sets };
            let set = sets
                .get_sparse_set(state.id.index())
                .ok_or(QueryError::StorageUnknown(state.id))?;

            Ok(WriteCompFetch::SparseEntity {
                set,
                change_tick,
                _marker: PhantomData,
            })
        }
    }
}

/// Creates a mutable fetch for a required component, aware of the sparse
/// driver.
///
/// When the component is the driver itself, the `SparseEntity` fetch is
/// promoted to `SparseDriver`, which resolves members by dense position;
/// every other fetch is returned unchanged.
///
/// # Safety
///
/// Same contract as `init_write_fetch`; `driver` must be the component id
/// selected as sparse driver for this run.
pub(crate) unsafe fn init_sparse_driver_write_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
    change_tick: ChangeTick,
    driver: ComponentId,
) -> Result<WriteCompFetch<'w, C>, QueryError> {
    let fetch =
        unsafe { init_write_fetch::<C>(state, archetype, sparse_sets, storage, change_tick) }?;

    match fetch {
        WriteCompFetch::SparseEntity {
            set, change_tick, ..
        } if state.id == driver => Ok(WriteCompFetch::SparseDriver {
            set,
            change_tick,
            _marker: PhantomData,
        }),
        other => Ok(other),
    }
}

/// Creates a mutable fetch for an optional component.
///
/// For table-stored components, this returns `Absent` when the current
/// archetype does not contain the component column.
///
/// # Safety
///
/// `archetype` and `sparse_sets` must remain valid for `'w`. The caller must
/// enforce exclusive value and tick access for every written component.
pub(crate) unsafe fn init_opt_write_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
    change_tick: ChangeTick,
) -> Result<OptWriteCompFetch<'w, C>, QueryError> {
    match state.storage {
        StorageKind::Table => {
            let arch = unsafe { &*archetype };

            if arch.signature().contains(state.id) {
                unsafe { arch.note_column_write_access(storage, state.id, change_tick) };
                let (ptr, component_ticks, len) =
                    unsafe { arch.get_column_write_ptrs::<C>(storage, state.id)? };
                Ok(OptWriteCompFetch::Table {
                    ptr,
                    component_ticks,
                    len,
                    change_tick,
                    _marker: PhantomData,
                })
            } else {
                Ok(OptWriteCompFetch::Absent)
            }
        }
        StorageKind::Sparse => {
            let sets = unsafe { &*sparse_sets };
            let set = sets
                .get_sparse_set(state.id.index())
                .ok_or(QueryError::StorageUnknown(state.id))?;

            Ok(OptWriteCompFetch::SparseEntity {
                set,
                change_tick,
                _marker: PhantomData,
            })
        }
    }
}

/// Creates a mutable fetch for an optional component, aware of the sparse
/// driver.
///
/// Promotes the optional `SparseEntity` fetch to `SparseDriver` when the
/// component is the driver itself, mirroring `init_sparse_driver_write_fetch`.
///
/// # Safety
///
/// Same contract as `init_write_fetch`; `driver` must be the component id
/// selected as sparse driver for this run.
pub(crate) unsafe fn init_sparse_driver_opt_write_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    sparse_sets: *const SparseSetManager,
    storage: ArchetypeStorageKind,
    change_tick: ChangeTick,
    driver: ComponentId,
) -> Result<OptWriteCompFetch<'w, C>, QueryError> {
    let fetch =
        unsafe { init_opt_write_fetch::<C>(state, archetype, sparse_sets, storage, change_tick) }?;

    match fetch {
        OptWriteCompFetch::SparseEntity {
            set, change_tick, ..
        } if state.id == driver => Ok(OptWriteCompFetch::SparseDriver {
            set,
            change_tick,
            _marker: PhantomData,
        }),
        other => Ok(other),
    }
}

/// Extension of `QueryData` for mutable queries.
///
/// Used by `QueryIterMut`.
///
/// # Safety
///
/// Implementors must only create references covered by the access declared
/// through `QueryData`. Mutable items must be unique for their full lifetime.
pub unsafe trait WriteQueryData: QueryData {
    /// Fetch prepared for a specific archetype.
    type Fetch<'w>;

    /// Item returned by the mutable iterator.
    type Item<'w>;

    /// Initializes the mutable fetch for a specific archetype.
    ///
    /// # Safety
    ///
    /// The received pointers must be valid for `'w`. The caller must guarantee
    /// exclusive access to every component that may be written.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
    ) -> Result<Self::Fetch<'w>, QueryError>;

    /// Initializes the mutable fetch knowing which component drives sparse
    /// iteration.
    ///
    /// The default ignores the driver and prepares a regular fetch; members
    /// that can use the driver's dense position override this to promote
    /// their fetch variant.
    ///
    /// # Safety
    ///
    /// Same contract as `init_fetch`; `_driver`, when used, must be the
    /// component id selected as sparse driver for this run.
    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
        _driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { Self::init_fetch(state, archetype, sparse_sets, storage, change_tick) }
    }

    /// Fetches the mutable query item for a specific row and entity.
    ///
    /// Returns `None` when the entity does not satisfy a required part of the
    /// query, such as a missing required sparse-set component.
    ///
    /// # Safety
    ///
    /// The fetch must have been initialized for the archetype that contains
    /// `row` and `entity`, and it must not create invalid mutable aliasing.
    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>>;

    /// Fetches the mutable query item using the driver's dense position.
    ///
    /// `sparse_dense_pos` locates `entity` in the driving sparse set, letting
    /// driver-backed members skip the entity-keyed index lookup. The default
    /// discards it and falls back to `fetch`.
    ///
    /// # Safety
    ///
    /// Same contract as `fetch`; `sparse_dense_pos` must come from the driving
    /// sparse set and identify `entity`.
    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        let _ = sparse_dense_pos;
        unsafe { Self::fetch(fetch, row, entity) }
    }

    /// Whether the prepared fetch reads memory that row lookahead can warm.
    ///
    /// Table columns stride across cache lines on sparse-driver lanes, and
    /// entity-keyed sparse members pay a dependent index load; the driver's
    /// own dense arrays stream sequentially and need no help.
    fn has_row_prefetch(_fetch: &Self::Fetch<'_>) -> bool {
        false
    }

    /// Warms the cache lines a future `fetch_sparse_driver` of this row touches.
    ///
    /// # Safety
    ///
    /// `row` must be in bounds for any table column in the fetch, as with
    /// `fetch_sparse_driver`.
    unsafe fn prefetch_sparse_driver_row(_fetch: &Self::Fetch<'_>, _row: usize, _entity: Entity) {}
}

unsafe impl WriteQueryData for Entity {
    type Fetch<'w> = ();
    type Item<'w> = Entity;

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: *const SparseSetManager,
        _: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        Ok(())
    }

    unsafe fn fetch<'w>(
        _: &mut Self::Fetch<'w>,
        _: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        Some(entity)
    }
}

unsafe impl<C: Component> WriteQueryData for &C {
    type Fetch<'w> = ReadCompFetch<'w, C>;
    type Item<'w> = &'w C;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_read_fetch(state, archetype, sparse_sets, storage) }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        _: ChangeTick,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            <&C as ReadQueryData>::init_sparse_driver_fetch(
                state,
                archetype,
                sparse_sets,
                storage,
                driver,
            )
        }
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        unsafe { <&C as ReadQueryData>::fetch(fetch, row, entity) }
    }

    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        unsafe { <&C as ReadQueryData>::fetch_sparse_driver(fetch, row, entity, sparse_dense_pos) }
    }

    #[inline]
    fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
        !matches!(fetch, ReadCompFetch::SparseDriver { .. })
    }

    #[inline(always)]
    unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
        match fetch {
            ReadCompFetch::Table { ptr, .. } => prefetch_element(*ptr, row),
            ReadCompFetch::SparseEntity { set, .. } => {
                let set = unsafe { &**set };
                set.prefetch_membership_line(entity);
            }
            _ => {}
        }
    }
}

unsafe impl<C: Component> WriteQueryData for &mut C {
    type Fetch<'w> = WriteCompFetch<'w, C>;
    type Item<'w> = Mut<'w, C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_write_fetch(state, archetype, sparse_sets, storage, change_tick) }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            init_sparse_driver_write_fetch(
                state,
                archetype,
                sparse_sets,
                storage,
                change_tick,
                driver,
            )
        }
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        match fetch {
            WriteCompFetch::Table {
                ptr,
                component_ticks,
                len,
                change_tick,
                ..
            } => Some(unsafe { table_write_at(*ptr, *component_ticks, *len, row, *change_tick) }),
            WriteCompFetch::SparseEntity {
                set, change_tick, ..
            }
            | WriteCompFetch::SparseDriver {
                set, change_tick, ..
            } => unsafe { sparse_write(*set, entity, *change_tick) },
        }
    }

    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        match fetch {
            WriteCompFetch::SparseDriver {
                set, change_tick, ..
            } => Some(unsafe {
                sparse_write_by_driver_position(*set, sparse_dense_pos, entity, *change_tick)
            }),
            _ => unsafe { <&mut C as WriteQueryData>::fetch(fetch, row, entity) },
        }
    }

    #[inline]
    fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
        !matches!(fetch, WriteCompFetch::SparseDriver { .. })
    }

    #[inline(always)]
    unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
        match fetch {
            WriteCompFetch::Table { ptr, .. } => prefetch_element(*ptr, row),
            WriteCompFetch::SparseEntity { set, .. } => {
                let set = unsafe { &**set };
                set.prefetch_membership_line(entity);
            }
            WriteCompFetch::SparseDriver { .. } => {}
        }
    }
}

unsafe impl<C: Component> WriteQueryData for Option<&C> {
    type Fetch<'w> = OptReadCompFetch<'w, C>;
    type Item<'w> = Option<&'w C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_opt_read_fetch(state, archetype, sparse_sets, storage) }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        _: ChangeTick,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            <Option<&C> as ReadQueryData>::init_sparse_driver_fetch(
                state,
                archetype,
                sparse_sets,
                storage,
                driver,
            )
        }
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        unsafe { <Option<&C> as ReadQueryData>::fetch(fetch, row, entity) }
    }

    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        unsafe {
            <Option<&C> as ReadQueryData>::fetch_sparse_driver(fetch, row, entity, sparse_dense_pos)
        }
    }

    #[inline]
    fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
        !matches!(fetch, OptReadCompFetch::SparseDriver { .. })
    }

    #[inline(always)]
    unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
        match fetch {
            OptReadCompFetch::Table { ptr, .. } => prefetch_element(*ptr, row),
            OptReadCompFetch::SparseEntity { set, .. } => {
                let set = unsafe { &**set };
                set.prefetch_membership_line(entity);
            }
            _ => {}
        }
    }
}

unsafe impl<C: Component> WriteQueryData for Option<&mut C> {
    type Fetch<'w> = OptWriteCompFetch<'w, C>;
    type Item<'w> = Option<Mut<'w, C>>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe { init_opt_write_fetch(state, archetype, sparse_sets, storage, change_tick) }
    }

    unsafe fn init_sparse_driver_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        sparse_sets: *const SparseSetManager,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
        driver: ComponentId,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        unsafe {
            init_sparse_driver_opt_write_fetch(
                state,
                archetype,
                sparse_sets,
                storage,
                change_tick,
                driver,
            )
        }
    }

    unsafe fn fetch<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
    ) -> Option<Self::Item<'w>> {
        let item = match fetch {
            OptWriteCompFetch::Absent => None,
            OptWriteCompFetch::Table {
                ptr,
                component_ticks,
                len,
                change_tick,
                ..
            } => Some(unsafe { table_write_at(*ptr, *component_ticks, *len, row, *change_tick) }),
            OptWriteCompFetch::SparseEntity {
                set, change_tick, ..
            }
            | OptWriteCompFetch::SparseDriver {
                set, change_tick, ..
            } => unsafe { sparse_write(*set, entity, *change_tick) },
        };

        Some(item)
    }

    unsafe fn fetch_sparse_driver<'w>(
        fetch: &mut Self::Fetch<'w>,
        row: usize,
        entity: Entity,
        sparse_dense_pos: u32,
    ) -> Option<Self::Item<'w>> {
        match fetch {
            OptWriteCompFetch::SparseDriver {
                set, change_tick, ..
            } => Some(Some(unsafe {
                sparse_write_by_driver_position(*set, sparse_dense_pos, entity, *change_tick)
            })),
            _ => unsafe { <Option<&mut C> as WriteQueryData>::fetch(fetch, row, entity) },
        }
    }

    #[inline]
    fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
        !matches!(fetch, OptWriteCompFetch::SparseDriver { .. })
    }

    #[inline(always)]
    unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
        match fetch {
            OptWriteCompFetch::Table { ptr, .. } => prefetch_element(*ptr, row),
            OptWriteCompFetch::SparseEntity { set, .. } => {
                let set = unsafe { &**set };
                set.prefetch_membership_line(entity);
            }
            _ => {}
        }
    }
}

unsafe impl WriteQueryData for () {
    type Fetch<'w> = ();
    type Item<'w> = ();

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: *const SparseSetManager,
        _: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::Fetch<'w>, QueryError> {
        Ok(())
    }

    unsafe fn fetch<'w>(_: &mut Self::Fetch<'w>, _: usize, _: Entity) -> Option<Self::Item<'w>> {
        Some(())
    }
}

/// Implements `WriteQueryData` for tuples of readable or mutable query data.
///
/// Initialization, fetching, and prefetching compose element-wise; a `None`
/// from any element rejects the whole row, and `has_row_prefetch` folds with
/// `||` so lookahead happens only when some member benefits.
macro_rules! tuple_write_query_data_impl {
    ($(#[$meta:meta])* $($T:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($T),*> WriteQueryData for ($($T,)*)
        where
            $($T: WriteQueryData),*
        {
            type Fetch<'w> = ($($T::Fetch<'w>,)*);
            type Item<'w> = ($($T::Item<'w>,)*);

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                sparse_sets: *const SparseSetManager,
                storage: ArchetypeStorageKind,
                change_tick: ChangeTick,
            ) -> Result<Self::Fetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;
                Ok((
                    $(unsafe { $T::init_fetch($T, archetype, sparse_sets, storage, change_tick)? },)*
                ))
            }

            unsafe fn init_sparse_driver_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                sparse_sets: *const SparseSetManager,
                storage: ArchetypeStorageKind,
                change_tick: ChangeTick,
                driver: ComponentId,
            ) -> Result<Self::Fetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;
                Ok((
                    $(unsafe { $T::init_sparse_driver_fetch(
                        $T,
                        archetype,
                        sparse_sets,
                        storage,
                        change_tick,
                        driver,
                    )? },)*
                ))
            }

            unsafe fn fetch<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
            ) -> Option<Self::Item<'w>> {
                let ($(ref mut $T,)*) = *fetch;
                Some((
                    $(unsafe { $T::fetch($T, row, entity)? },)*
                ))
            }

            unsafe fn fetch_sparse_driver<'w>(
                fetch: &mut Self::Fetch<'w>,
                row: usize,
                entity: Entity,
                sparse_dense_pos: u32,
            ) -> Option<Self::Item<'w>> {
                let ($(ref mut $T,)*) = *fetch;
                Some((
                    $(unsafe { $T::fetch_sparse_driver(
                        $T,
                        row,
                        entity,
                        sparse_dense_pos,
                    )? },)*
                ))
            }

            #[inline]
            fn has_row_prefetch(fetch: &Self::Fetch<'_>) -> bool {
                let ($(ref $T,)*) = *fetch;
                false $(|| $T::has_row_prefetch($T))*
            }

            #[inline(always)]
            unsafe fn prefetch_sparse_driver_row(fetch: &Self::Fetch<'_>, row: usize, entity: Entity) {
                let ($(ref $T,)*) = *fetch;
                $(unsafe { <$T as WriteQueryData>::prefetch_sparse_driver_row($T, row, entity) };)*
            }
        }
    };
}

all_tuples!(tuple_write_query_data_impl, 1, 12, W);
