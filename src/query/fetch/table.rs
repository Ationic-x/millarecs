//! Fast path for archetypal queries.
//!
//! This module is used when all requested data is table-backed or otherwise
//! resolvable without sparse-set lookups. It can return query items directly
//! instead of wrapping each row in `Option`.

use crate::utils::prefetch::prefetch_element;

use super::*;

/// Table fetch for a required shared component column.
pub struct ReadCompFetch<'w, C: Component> {
    /// Pointer to the first element of the column.
    pub ptr: *const C,
    /// Number of rows available in the column.
    pub len: usize,
    /// Binds the raw pointer to the iterator lifetime.
    pub _marker: PhantomData<&'w C>,
}

/// Table fetch for an optional shared component column.
pub enum OptReadCompFetch<'w, C: Component> {
    /// The current archetype does not contain the column.
    Absent,
    /// The current archetype contains the column.
    Present(ReadCompFetch<'w, C>),
}

/// Read-only table fast-path query data.
///
/// Unlike the mixed path, fetching a matched row always returns an item.
///
/// # Safety
///
/// Implementors must only create shared references declared by `QueryData` and
/// must keep every returned item within the prepared archetype storage.
pub unsafe trait ReadQueryData: mixed::ReadQueryData {
    /// Fetch prepared for one table archetype.
    type TableFetch<'w>;

    /// Prepares table fetch data for an archetype.
    ///
    /// # Safety
    ///
    /// `archetype` must point to the archetype currently being iterated and must
    /// remain valid for the returned fetch lifetime.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
    ) -> Result<Self::TableFetch<'w>, QueryError>;

    /// Returns the item for a row in the prepared archetype.
    ///
    /// # Safety
    ///
    /// `row` must be in bounds for the current archetype.
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        entity: Entity,
    ) -> <Self as mixed::ReadQueryData>::Item<'w>;

    /// Whether any fetch of this data observes the per-row entity handle.
    ///
    /// Iterators skip loading entity handles when no member reports this.
    const USES_ENTITY: bool = false;
    /// Whether this data reads table columns that benefit from row prefetch.
    ///
    /// Iterators skip the row-lookahead machinery when no member reports this.
    const HAS_ROW_PREFETCH: bool = false;

    /// Warms the cache lines that a future `fetch` of `row` would touch.
    ///
    /// Sparse-driver execution calls this ahead of time when its row order
    /// strides across the table, so gather latency overlaps with the rows in
    /// between. The default warms nothing.
    ///
    /// # Safety
    ///
    /// `row` must be in bounds for the prepared column, as with `fetch`.
    unsafe fn prefetch_row(_fetch: &Self::TableFetch<'_>, _row: usize) {}
}

unsafe impl ReadQueryData for Entity {
    type TableFetch<'w> = ();

    const USES_ENTITY: bool = true;

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        Ok(())
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        _: &mut Self::TableFetch<'w>,
        _: usize,
        entity: Entity,
    ) -> <Self as mixed::ReadQueryData>::Item<'w> {
        entity
    }
}

/// Prepares a required shared table-column fetch.
///
/// # Safety
///
/// `archetype` must be valid and contain the component described by `state`.
pub(crate) unsafe fn init_read_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    storage: ArchetypeStorageKind,
) -> Result<ReadCompFetch<'w, C>, QueryError> {
    debug_assert!(matches!(state.storage, StorageKind::Table));

    let arch = unsafe { &*archetype };
    let (ptr, len) = unsafe { arch.get_column_read_ptrs::<C>(storage, state.id)? };

    Ok(ReadCompFetch {
        ptr,
        len,
        _marker: PhantomData,
    })
}

unsafe impl<C: Component> ReadQueryData for &C {
    type TableFetch<'w> = ReadCompFetch<'w, C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        unsafe { init_read_fetch(state, archetype, storage) }
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        _entity: Entity,
    ) -> <Self as mixed::ReadQueryData>::Item<'w> {
        unsafe { table_read_at(fetch.ptr, fetch.len, row) }
    }

    const HAS_ROW_PREFETCH: bool = true;

    #[inline(always)]
    unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
        prefetch_element(fetch.ptr, row);
    }
}

unsafe impl<C: Component> ReadQueryData for Option<&C> {
    type TableFetch<'w> = OptReadCompFetch<'w, C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        debug_assert!(matches!(state.storage, StorageKind::Table));

        let arch = unsafe { &*archetype };

        if arch.signature().contains(state.id) {
            let (ptr, len) = unsafe { arch.get_column_read_ptrs::<C>(storage, state.id)? };

            Ok(OptReadCompFetch::Present(ReadCompFetch {
                ptr,
                len,
                _marker: PhantomData,
            }))
        } else {
            Ok(OptReadCompFetch::Absent)
        }
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        _entity: Entity,
    ) -> <Self as mixed::ReadQueryData>::Item<'w> {
        match fetch {
            OptReadCompFetch::Absent => None,
            OptReadCompFetch::Present(ReadCompFetch { ptr, len, .. }) => {
                Some(unsafe { table_read_at(*ptr, *len, row) })
            }
        }
    }

    const HAS_ROW_PREFETCH: bool = true;

    #[inline(always)]
    unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
        match fetch {
            OptReadCompFetch::Absent => {}
            OptReadCompFetch::Present(inner) => prefetch_element(inner.ptr, row),
        }
    }
}

unsafe impl ReadQueryData for () {
    type TableFetch<'w> = ();

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        Ok(())
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        _: &mut Self::TableFetch<'w>,
        _: usize,
        _: Entity,
    ) -> <Self as mixed::ReadQueryData>::Item<'w> {
    }
}

/// Implements `ReadQueryData` for tuples of readable query data.
///
/// Initialization, fetching, and prefetching compose element-wise.
/// `USES_ENTITY` and `HAS_ROW_PREFETCH` fold with `||`, so the iterator loads
/// entity handles and runs row lookahead only when some member needs them.
macro_rules! tuple_read_query_data_impl {
    ($(#[$meta:meta])* $($T:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($T),*> ReadQueryData for ($($T,)*)
        where
            $($T: ReadQueryData),*
        {
            type TableFetch<'w> = ($($T::TableFetch<'w>,)*);

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                storage: ArchetypeStorageKind,
            ) -> Result<Self::TableFetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;

                Ok((
                    $(unsafe { <$T as ReadQueryData>::init_fetch($T, archetype, storage)? },)*
                ))
            }

            #[inline(always)]
            unsafe fn fetch<'w>(
                fetch: &mut Self::TableFetch<'w>,
                row: usize,
                entity: Entity,
            ) -> <Self as mixed::ReadQueryData>::Item<'w> {
                let ($(ref mut $T,)*) = *fetch;

                (
                    $(unsafe { <$T as ReadQueryData>::fetch($T, row, entity) },)*
                )
            }

            const USES_ENTITY: bool = false $(|| $T::USES_ENTITY)*;
            const HAS_ROW_PREFETCH: bool = false $(|| $T::HAS_ROW_PREFETCH)*;

            #[inline(always)]
            unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
                let ($(ref $T,)*) = *fetch;
                $(unsafe { <$T as ReadQueryData>::prefetch_row($T, row) };)*
            }
        }
    };
}

all_tuples!(tuple_read_query_data_impl, 1, 12, T);

/// Table fetch for a required mutable component column.
pub struct WriteCompFetch<'w, C: Component> {
    /// Pointer to the first mutable element of the column.
    pub ptr: *mut C,
    /// Pointer to the first tick entry of the column.
    pub component_ticks: *mut ComponentTicks,
    /// Number of rows available in the column.
    pub len: usize,
    /// Tick used when this mutable component is actually changed.
    pub change_tick: ChangeTick,
    /// Binds the mutable raw pointer to the iterator lifetime.
    pub _marker: PhantomData<&'w mut C>,
}

/// Table fetch for an optional mutable component column.
pub enum OptWriteCompFetch<'w, C: Component> {
    /// The current archetype does not contain the column.
    Absent,
    /// The current archetype contains the column.
    Present(WriteCompFetch<'w, C>),
}

/// Prepares a required mutable table-column fetch.
///
/// # Safety
///
/// `archetype` must be valid and contain the component described by `state`.
/// The validated access set must grant exclusive value and tick access.
unsafe fn init_write_fetch<'w, C: Component>(
    state: &ComponentQueryState,
    archetype: *const Archetype,
    storage: ArchetypeStorageKind,
    change_tick: ChangeTick,
) -> Result<WriteCompFetch<'w, C>, QueryError> {
    debug_assert!(matches!(state.storage, StorageKind::Table));

    let arch = unsafe { &*archetype };
    unsafe { arch.note_column_write_access(storage, state.id, change_tick) };
    let (ptr, component_ticks, len) =
        unsafe { arch.get_column_write_ptrs::<C>(storage, state.id)? };

    Ok(WriteCompFetch {
        ptr,
        component_ticks,
        len,
        change_tick,
        _marker: PhantomData,
    })
}

/// Mutable table fast-path query data.
///
/// This is used when mutable iteration can read or write table rows without
/// checking sparse-set membership for the data part.
///
/// # Safety
///
/// Implementors must respect the access declared by `QueryData`; mutable items
/// returned from a fetch must never alias another live item.
pub unsafe trait WriteQueryData: mixed::WriteQueryData {
    /// Fetch prepared for one mutable table archetype.
    type TableFetch<'w>;

    /// Prepares mutable table fetch data for an archetype.
    ///
    /// # Safety
    ///
    /// `archetype` must point to the current archetype. The caller must enforce
    /// every exclusive component access declared by the query.
    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
    ) -> Result<Self::TableFetch<'w>, QueryError>;

    /// Returns the mutable item for a row in the prepared archetype.
    ///
    /// # Safety
    ///
    /// `row` must be in bounds and no conflicting reference to the same component
    /// may be alive.
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        entity: Entity,
    ) -> <Self as mixed::WriteQueryData>::Item<'w>;

    /// Whether any fetch of this data observes the per-row entity handle.
    ///
    /// Iterators skip loading entity handles when no member reports this.
    const USES_ENTITY: bool = false;
    /// Whether this data reads table columns that benefit from row prefetch.
    ///
    /// Iterators skip the row-lookahead machinery when no member reports this.
    const HAS_ROW_PREFETCH: bool = false;

    /// Warms the cache lines that a future `fetch` of `row` would touch.
    ///
    /// Sparse-driver execution calls this ahead of time when its row order
    /// strides across the table, so gather latency overlaps with the rows in
    /// between. The default warms nothing.
    ///
    /// # Safety
    ///
    /// `row` must be in bounds for the prepared column, as with `fetch`.
    unsafe fn prefetch_row(_fetch: &Self::TableFetch<'_>, _row: usize) {}
}

unsafe impl WriteQueryData for Entity {
    type TableFetch<'w> = ();

    const USES_ENTITY: bool = true;

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        Ok(())
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        _: &mut Self::TableFetch<'w>,
        _: usize,
        entity: Entity,
    ) -> <Self as mixed::WriteQueryData>::Item<'w> {
        entity
    }
}

unsafe impl<C: Component> WriteQueryData for &C {
    type TableFetch<'w> = ReadCompFetch<'w, C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        unsafe { init_read_fetch(state, archetype, storage) }
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        _entity: Entity,
    ) -> <Self as mixed::WriteQueryData>::Item<'w> {
        unsafe { table_read_at(fetch.ptr, fetch.len, row) }
    }

    const HAS_ROW_PREFETCH: bool = true;

    #[inline(always)]
    unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
        prefetch_element(fetch.ptr, row);
    }
}

unsafe impl<C: Component> WriteQueryData for &mut C {
    type TableFetch<'w> = WriteCompFetch<'w, C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        unsafe { init_write_fetch(state, archetype, storage, change_tick) }
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        _entity: Entity,
    ) -> <Self as mixed::WriteQueryData>::Item<'w> {
        unsafe {
            table_write_at(
                fetch.ptr,
                fetch.component_ticks,
                fetch.len,
                row,
                fetch.change_tick,
            )
        }
    }

    const HAS_ROW_PREFETCH: bool = true;

    #[inline(always)]
    unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
        prefetch_element(fetch.ptr, row);
        prefetch_element(fetch.component_ticks, row);
    }
}

unsafe impl<C: Component> WriteQueryData for Option<&C> {
    type TableFetch<'w> = OptReadCompFetch<'w, C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        unsafe { <Option<&C> as ReadQueryData>::init_fetch(state, archetype, storage) }
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        entity: Entity,
    ) -> <Self as mixed::WriteQueryData>::Item<'w> {
        unsafe { <Option<&C> as ReadQueryData>::fetch(fetch, row, entity) }
    }

    const HAS_ROW_PREFETCH: bool = true;

    #[inline(always)]
    unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
        match fetch {
            OptReadCompFetch::Absent => {}
            OptReadCompFetch::Present(inner) => prefetch_element(inner.ptr, row),
        }
    }
}

unsafe impl<C: Component> WriteQueryData for Option<&mut C> {
    type TableFetch<'w> = OptWriteCompFetch<'w, C>;

    unsafe fn init_fetch<'w>(
        state: &Self::State,
        archetype: *const Archetype,
        storage: ArchetypeStorageKind,
        change_tick: ChangeTick,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        debug_assert!(matches!(state.storage, StorageKind::Table));

        let arch = unsafe { &*archetype };

        if arch.signature().contains(state.id) {
            unsafe { arch.note_column_write_access(storage, state.id, change_tick) };
            let (ptr, component_ticks, len) =
                unsafe { arch.get_column_write_ptrs::<C>(storage, state.id)? };

            Ok(OptWriteCompFetch::Present(WriteCompFetch {
                ptr,
                component_ticks,
                len,
                change_tick,
                _marker: PhantomData,
            }))
        } else {
            Ok(OptWriteCompFetch::Absent)
        }
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        fetch: &mut Self::TableFetch<'w>,
        row: usize,
        _entity: Entity,
    ) -> <Self as mixed::WriteQueryData>::Item<'w> {
        match fetch {
            OptWriteCompFetch::Absent => None,
            OptWriteCompFetch::Present(WriteCompFetch {
                ptr,
                component_ticks,
                len,
                change_tick,
                ..
            }) => Some(unsafe { table_write_at(*ptr, *component_ticks, *len, row, *change_tick) }),
        }
    }

    const HAS_ROW_PREFETCH: bool = true;

    #[inline(always)]
    unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
        match fetch {
            OptWriteCompFetch::Absent => {}
            OptWriteCompFetch::Present(inner) => {
                prefetch_element(inner.ptr, row);
                prefetch_element(inner.component_ticks, row);
            }
        }
    }
}

unsafe impl WriteQueryData for () {
    type TableFetch<'w> = ();

    unsafe fn init_fetch<'w>(
        _: &Self::State,
        _: *const Archetype,
        _: ArchetypeStorageKind,
        _: ChangeTick,
    ) -> Result<Self::TableFetch<'w>, QueryError> {
        Ok(())
    }

    #[inline(always)]
    unsafe fn fetch<'w>(
        _: &mut Self::TableFetch<'w>,
        _: usize,
        _: Entity,
    ) -> <Self as mixed::WriteQueryData>::Item<'w> {
    }
}

/// Implements `WriteQueryData` for tuples of readable or mutable query data.
///
/// Initialization, fetching, and prefetching compose element-wise.
/// `USES_ENTITY` and `HAS_ROW_PREFETCH` fold with `||`, so the iterator loads
/// entity handles and runs row lookahead only when some member needs them.
macro_rules! tuple_write_query_data_impl {
    ($(#[$meta:meta])* $($T:ident),*) => {
        $(#[$meta])*
        unsafe impl<$($T),*> WriteQueryData for ($($T,)*)
        where
            $($T: WriteQueryData),*
        {
            type TableFetch<'w> = ($($T::TableFetch<'w>,)*);

            unsafe fn init_fetch<'w>(
                state: &Self::State,
                archetype: *const Archetype,
                storage: ArchetypeStorageKind,
                change_tick: ChangeTick,
            ) -> Result<Self::TableFetch<'w>, QueryError> {
                let ($(ref $T,)*) = *state;

                Ok((
                    $(unsafe { <$T as WriteQueryData>::init_fetch($T, archetype, storage, change_tick) }?,)*
                ))
            }

            #[inline(always)]
            unsafe fn fetch<'w>(
                fetch: &mut Self::TableFetch<'w>,
                row: usize,
                entity: Entity,
            ) -> <Self as mixed::WriteQueryData>::Item<'w> {
                let ($(ref mut $T,)*) = *fetch;

                (
                    $(unsafe { <$T as WriteQueryData>::fetch($T, row, entity) },)*
                )
            }

            const USES_ENTITY: bool = false $(|| $T::USES_ENTITY)*;
            const HAS_ROW_PREFETCH: bool = false $(|| $T::HAS_ROW_PREFETCH)*;

            #[inline(always)]
            unsafe fn prefetch_row(fetch: &Self::TableFetch<'_>, row: usize) {
                let ($(ref $T,)*) = *fetch;
                $(unsafe { <$T as WriteQueryData>::prefetch_row($T, row) };)*
            }
        }
    };
}

all_tuples!(tuple_write_query_data_impl, 1, 12, T);
