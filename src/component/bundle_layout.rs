//! Cached, world-local descriptions of bundle item layouts.

use std::any::TypeId;

use crate::{
    component::{ComponentId, StorageKind},
    utils::flat_map::{FlatKey, FlatMap},
};

/// Stable identifier for one bundle type inside a component manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct BundleLayoutId(u32);

impl BundleLayoutId {
    /// Creates the identifier for layout `index`, rejecting overflow.
    #[inline]
    pub(crate) fn new(index: usize) -> Self {
        let index = u32::try_from(index).expect("bundle layout count exceeded u32 capacity");
        assert_ne!(
            index,
            u32::MAX,
            "bundle layout id exhausted its valid range"
        );
        Self(index)
    }

    /// Returns the dense index of this layout.
    pub(crate) fn index(self) -> usize {
        self.0 as usize
    }
}

impl FlatKey for BundleLayoutId {
    #[inline]
    fn fold(self) -> u64 {
        // Sequential dense ids: spreading is left to the map's Fibonacci home.
        self.0 as u64
    }
}

/// Component and original item position produced by `Bundle::with_items`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BundleSlot {
    pub(crate) component: ComponentId,
    pub(crate) item_index: u16,
}

/// Validated bundle metadata retained for the lifetime of a world.
pub(crate) struct BundleLayout {
    /// Table components sorted by component id.
    pub(crate) table: Box<[BundleSlot]>,
    /// Sparse components sorted by component id.
    pub(crate) sparse: Box<[BundleSlot]>,
}

impl BundleLayout {
    /// Returns the slots stored under `storage`, sorted by component id.
    #[inline]
    pub(crate) fn slots(&self, storage: StorageKind) -> &[BundleSlot] {
        match storage {
            StorageKind::Table => &self.table,
            StorageKind::Sparse => &self.sparse,
        }
    }
}

/// Bundle layouts indexed both by Rust type and by a compact identifier.
#[derive(Default)]
pub(crate) struct CachedBundleLayouts {
    type_to_layout_id: FlatMap<TypeId, BundleLayoutId>,
    /// Layouts in registration order, indexed by their `BundleLayoutId`.
    bundle_layouts: Vec<BundleLayout>,
}

impl CachedBundleLayouts {
    /// Creates an empty cache.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Registers `bundle_layout` for `type_id`, returning its new identifier.
    pub(crate) fn insert(
        &mut self,
        type_id: TypeId,
        bundle_layout: BundleLayout,
    ) -> BundleLayoutId {
        debug_assert!(!self.type_to_layout_id.contains(type_id));

        let bundle_layout_id = BundleLayoutId::new(self.bundle_layouts.len());
        self.bundle_layouts.push(bundle_layout);
        self.type_to_layout_id.insert(type_id, bundle_layout_id);

        bundle_layout_id
    }

    /// Returns the layout registered for `id`, panicking when `id` was not
    /// issued by this cache.
    #[inline]
    pub(crate) fn get(&self, id: BundleLayoutId) -> &BundleLayout {
        self.bundle_layouts
            .get(id.0 as usize)
            .expect("bundle layout id must belong to this component manager")
    }

    /// Returns the layout id previously registered for `type_id`, if any.
    #[inline]
    pub(crate) fn id_of(&self, type_id: TypeId) -> Option<BundleLayoutId> {
        self.type_to_layout_id.get_copy(type_id)
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    #[test]
    fn cached_bundle_metadata_keeps_its_compact_layout() {
        assert_eq!(size_of::<BundleLayoutId>(), 4);
        assert_eq!(size_of::<BundleSlot>(), 8);
        assert_eq!(
            size_of::<BundleLayout>(),
            size_of::<Box<[BundleSlot]>>() * 2
        );
    }
}
