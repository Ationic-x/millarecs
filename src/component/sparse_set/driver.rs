//! Sparse-driver index: per-archetype query lanes over sparse members.
//!
//! The driver is the query-side representation of one sparse set. When a query
//! selects a sparse component as its route, buckets of member entities become
//! driver rows that execution can traverse directly, with locality metrics
//! that feed the route cost model.

use crate::{
    component::{
        archetype::{ArchetypeId, ARCHETYPE_PAGE_ROWS},
        sparse_set::{set::INVALID_SPARSE_COMPONENT, SparseLocationUpdate},
    },
    entity::{
        metadata::{ArchetypeStorageKind, EntityLocation},
        Entity,
    },
};

/// Locality metrics of one driver lane, used by route cost estimation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SparseLaneLocality {
    /// Live rows the lane covers.
    pub(crate) rows: u32,
    /// Archetype pages the traversal must visit.
    pub(crate) page_visits: u32,
    /// Same-page discontinuities between consecutive rows.
    pub(crate) in_page_jumps: u32,
}

/// Page-transition counters between two consecutive rows.
#[derive(Clone, Copy, Debug, Default)]
struct EdgeCounters {
    page_breaks: u32,
    in_page_jumps: u32,
}

impl EdgeCounters {
    /// Adds another pair of counters into this one.
    #[inline]
    fn add(&mut self, other: Self) {
        self.page_breaks += other.page_breaks;
        self.in_page_jumps += other.in_page_jumps;
    }
}

/// One query-visible member of a sparse-driver lane.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct SparseDriverRow {
    /// Handle of the member entity.
    pub(crate) entity: Entity,
    /// Table row inside the owning archetype.
    pub(crate) row: u32,
    /// Dense position of the member inside its sparse set.
    ///
    /// Despite the name, the field stores a dense position rather than a jump
    /// count: the row reuses the locality-counter slot so lane `swap_remove`
    /// can report which dense entry moved into a vacated slot.
    pub(crate) in_page_jumps: u32,
}

/// All query-visible members of one sparse set inside one archetype.
#[derive(Debug, Default)]
pub(crate) struct SparseDriverLane {
    /// Member rows in traversal order.
    rows: Vec<SparseDriverRow>,
    /// Cross-page transitions between consecutive rows, maintained
    /// incrementally alongside `rows`.
    page_breaks: u32,
    /// Same-page discontinuities between consecutive rows, maintained
    /// incrementally alongside `rows`.
    in_page_jumps: u32,
}

impl SparseDriverLane {
    /// Preallocates capacity for `additional` rows.
    #[inline]
    fn reserve(&mut self, additional: usize) {
        self.rows.reserve(additional);
    }

    /// Drops all rows and resets the locality counters.
    fn clear_rows(&mut self) {
        self.rows.clear();
        self.page_breaks = 0;
        self.in_page_jumps = 0;
    }

    /// Recomputes the locality counters from the current row order.
    fn rebuild_locality(&mut self) {
        let mut counters = EdgeCounters::default();

        for pair in self.rows.windows(2) {
            counters.add(Self::edge(pair[0], pair[1]));
        }

        self.page_breaks = counters.page_breaks;
        self.in_page_jumps = counters.in_page_jumps;
    }

    /// Returns whether the lane holds no members.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Returns the query-visible member rows in dense order.
    #[inline]
    pub(crate) fn rows(&self) -> &[SparseDriverRow] {
        &self.rows
    }

    /// Returns the locality metrics of the lane for cost estimation.
    #[inline]
    pub(crate) fn locality(&self) -> SparseLaneLocality {
        if self.rows.is_empty() {
            return SparseLaneLocality::default();
        }

        SparseLaneLocality {
            rows: self.rows.len() as u32,
            page_visits: self.page_breaks + 1,
            in_page_jumps: self.in_page_jumps,
        }
    }

    /// Classifies the transition between two consecutive rows.
    #[inline]
    fn edge(left: SparseDriverRow, right: SparseDriverRow) -> EdgeCounters {
        let page_rows = ARCHETYPE_PAGE_ROWS as u32;
        let same_page = left.row / page_rows == right.row / page_rows;
        let sequential = left.row.checked_add(1) == Some(right.row);

        EdgeCounters {
            // Every cross-page transition pays only the page-transition cost.
            page_breaks: u32::from(!same_page),

            // A discontinuity is charged separately only inside the same page.
            in_page_jumps: u32::from(same_page && !sequential),
        }
    }

    /// Sums the edge counters of the transitions starting at each distinct
    /// index in `starts`.
    fn local_edges<const N: usize>(&self, starts: [usize; N]) -> EdgeCounters {
        let mut result = EdgeCounters::default();

        for index in 0..N {
            let start = starts[index];

            if starts[..index].contains(&start) {
                continue;
            }

            let Some(end) = start.checked_add(1) else {
                continue;
            };

            if end < self.rows.len() {
                result.add(Self::edge(self.rows[start], self.rows[end]));
            }
        }

        result
    }

    /// Replaces the counters of a fixed edge set by subtracting `before` and
    /// adding `after`, keeping the totals exact under incremental edits.
    #[inline]
    fn apply_edge_change(&mut self, before: EdgeCounters, after: EdgeCounters) {
        self.page_breaks = self
            .page_breaks
            .checked_sub(before.page_breaks)
            .expect("page locality counters must remain consistent")
            + after.page_breaks;

        self.in_page_jumps = self
            .in_page_jumps
            .checked_sub(before.in_page_jumps)
            .expect("row locality counters must remain consistent")
            + after.in_page_jumps;
    }

    /// Appends one row, folding its edge into the counters; returns the slot
    /// the row occupies.
    fn push(&mut self, row: SparseDriverRow) -> u32 {
        let slot =
            u32::try_from(self.rows.len()).expect("sparse driver lane exceeded u32 capacity");

        assert!(
            slot <= SLOT_MASK,
            "sparse driver lane exceeded packed slot capacity"
        );

        if let Some(previous) = self.rows.last().copied() {
            let edge = Self::edge(previous, row);
            self.page_breaks += edge.page_breaks;
            self.in_page_jumps += edge.in_page_jumps;
        }

        self.rows.push(row);

        slot
    }

    /// Rewrites the table row of `slot`, repairing the counters of its two
    /// neighbouring edges; returns whether the stored row changed.
    fn update_row(&mut self, slot: usize, row: u32) -> bool {
        if self.rows[slot].row == row {
            return false;
        }

        let affected = [slot.saturating_sub(1), slot];
        let before = self.local_edges(affected);

        self.rows[slot].row = row;

        let after = self.local_edges(affected);
        self.apply_edge_change(before, after);

        true
    }

    /// Swap-removes the row at `slot`, repairing every affected edge counter;
    /// returns the dense position of the row moved into `slot`, if any.
    fn swap_remove(&mut self, slot: usize) -> Option<u32> {
        let last = self.rows.len() - 1;

        if slot == last {
            if last != 0 {
                let removed_edge = Self::edge(self.rows[last - 1], self.rows[last]);
                self.apply_edge_change(removed_edge, EdgeCounters::default());
            }

            self.rows.pop();
            return None;
        }

        let moved = self.rows[last];
        let mut before = EdgeCounters::default();

        if slot != 0 {
            before.add(Self::edge(self.rows[slot - 1], self.rows[slot]));
        }

        before.add(Self::edge(self.rows[slot], self.rows[slot + 1]));

        if last - 1 != slot {
            before.add(Self::edge(self.rows[last - 1], self.rows[last]));
        }

        self.rows[slot] = moved;
        self.rows.pop();

        let mut after = EdgeCounters::default();

        if slot != 0 {
            after.add(Self::edge(self.rows[slot - 1], self.rows[slot]));
        }

        if slot + 1 < self.rows.len() {
            after.add(Self::edge(self.rows[slot], self.rows[slot + 1]));
        }

        self.apply_edge_change(before, after);
        Some(moved.in_page_jumps)
    }
}

/// Driver lanes for one archetype, split by table storage kind.
#[derive(Debug)]
pub(crate) struct SparseDriverBucket {
    /// Archetype all rows in the lanes belong to.
    pub(crate) archetype_id: ArchetypeId,
    /// Members living in the contiguous lane.
    pub(crate) contiguous: SparseDriverLane,
    /// Members living in the fragmented lane.
    pub(crate) fragmented: SparseDriverLane,
}

impl SparseDriverBucket {
    /// Creates a bucket with two empty lanes for `archetype_id`.
    fn new(archetype_id: ArchetypeId) -> Self {
        Self {
            archetype_id,
            contiguous: SparseDriverLane::default(),
            fragmented: SparseDriverLane::default(),
        }
    }

    /// Returns the mutable lane that holds members of `storage` kind.
    #[inline]
    fn lane_mut(&mut self, storage: ArchetypeStorageKind) -> &mut SparseDriverLane {
        match storage {
            ArchetypeStorageKind::Contiguous => &mut self.contiguous,
            ArchetypeStorageKind::Fragmented => &mut self.fragmented,
        }
    }

    /// Returns the lane that holds members of `storage` kind.
    #[inline]
    fn lane(&self, storage: ArchetypeStorageKind) -> &SparseDriverLane {
        match storage {
            ArchetypeStorageKind::Contiguous => &self.contiguous,
            ArchetypeStorageKind::Fragmented => &self.fragmented,
        }
    }
}

/// Position of one membership inside the entity-local doubly-linked chain.
///
/// Both links are component indices, with `INVALID_SPARSE_COMPONENT` marking
/// the chain ends. Links live in dense auxiliary metadata, independent of the
/// optional query driver, so entity-local traversal never activates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SparseMembershipLinks {
    /// Component index of the previous membership of the same entity.
    pub(crate) previous: u32,
    /// Component index of the next membership of the same entity.
    pub(crate) next: u32,
}

impl SparseMembershipLinks {
    /// Links of a member not yet inserted into any chain.
    pub(crate) const UNLINKED: Self = Self {
        previous: INVALID_SPARSE_COMPONENT,
        next: INVALID_SPARSE_COMPONENT,
    };
}

/// Bucket-index bit marking an entry as living in the fragmented lane.
const FRAGMENTED_BIT: u32 = 1 << 31;
/// Masks out [`FRAGMENTED_BIT`], leaving the raw bucket index.
const SLOT_MASK: u32 = FRAGMENTED_BIT - 1;
/// Bucket-map sentinel for an archetype with no driver bucket.
const INVALID_BUCKET: u32 = u32::MAX;

/// Lane slot and storage kind packed into one `u32`.
///
/// [`FRAGMENTED_BIT`] occupies the top bit and selects the lane inside the
/// bucket; [`SLOT_MASK`] leaves the remaining bits for the slot index.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
struct PackedLaneSlot(u32);

impl PackedLaneSlot {
    /// Packs `slot` together with the lane selected by `storage`.
    #[inline]
    fn new(slot: u32, storage: ArchetypeStorageKind) -> Self {
        debug_assert!(slot <= SLOT_MASK);

        let mut value = slot;

        if storage == ArchetypeStorageKind::Fragmented {
            value |= FRAGMENTED_BIT;
        }

        Self(value)
    }

    /// Returns the slot index inside the lane.
    #[inline]
    fn slot(self) -> usize {
        (self.0 & SLOT_MASK) as usize
    }

    /// Returns the storage kind that selects the lane.
    #[inline]
    fn storage(self) -> ArchetypeStorageKind {
        if self.0 & FRAGMENTED_BIT != 0 {
            ArchetypeStorageKind::Fragmented
        } else {
            ArchetypeStorageKind::Contiguous
        }
    }
}

/// Driver-side location of one dense entry.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SparseIndexEntry {
    /// Index into `SparseDriverIndex::buckets`.
    bucket: u32,
    /// Lane slot of the entry, including its storage kind.
    lane_slot: PackedLaneSlot,
}

impl SparseIndexEntry {
    /// Creates an entry from a bucket index and a packed lane slot.
    #[inline]
    const fn new(bucket: u32, lane_slot: PackedLaneSlot) -> Self {
        Self { bucket, lane_slot }
    }

    /// Returns the bucket index holding the entry.
    #[inline]
    fn bucket(self) -> u32 {
        self.bucket
    }

    /// Returns the slot of the entry inside its lane.
    #[inline]
    fn slot(self) -> usize {
        self.lane_slot.slot()
    }

    /// Returns the storage kind of the lane holding the entry.
    #[inline]
    fn storage(self) -> ArchetypeStorageKind {
        self.lane_slot.storage()
    }
}

/// Query-side index over one sparse set.
///
/// Maps archetypes to driver buckets and keeps per-entry location data while
/// the driver is active. The revision increments on every structural change
/// so cached query work can detect staleness.
#[derive(Default, Debug)]
pub(crate) struct SparseDriverIndex {
    /// Bucket index per archetype index, with `INVALID_BUCKET` for archetypes
    /// that have no bucket yet.
    archetype_to_bucket: Vec<u32>,
    /// Driver buckets in creation order.
    buckets: Vec<SparseDriverBucket>,
    /// Driver entry for every member, indexed by its dense position.
    dense_entries: Vec<SparseIndexEntry>,
    /// Structural revision, bumped on every visible change.
    revision: u64,
}

impl SparseDriverIndex {
    /// Builds the query-driver index from the canonical sparse-set storage.
    ///
    /// This operation is expected to run at most once per sparse component.
    pub(crate) fn build<I>(entries: I) -> Self
    where
        I: ExactSizeIterator<Item = (Entity, EntityLocation)>,
    {
        let mut index = Self::default();
        index.dense_entries.reserve(entries.len());

        for (dense_pos, (entity, location)) in entries.enumerate() {
            let dense_pos =
                u32::try_from(dense_pos).expect("sparse component count exceeds u32::MAX");

            index.push_dense_without_revision(dense_pos, entity, location);
        }

        if !index.dense_entries.is_empty() {
            index.bump_revision();
        }

        index
    }

    /// Returns the archetype buckets of the index in creation order.
    pub(crate) fn buckets(&self) -> &[SparseDriverBucket] {
        &self.buckets
    }

    /// Returns the structural revision, which changes on every visible
    /// mutation.
    #[inline]
    pub(crate) const fn revision(&self) -> u64 {
        self.revision
    }

    /// Advances the revision, wrapping past zero so that zero is never
    /// published.
    #[inline]
    fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1);

        if self.revision == 0 {
            self.revision = 1;
        }
    }

    /// Returns the bucket index of `archetype_id`, creating the bucket on
    /// first contact.
    fn bucket_for(&mut self, archetype_id: ArchetypeId) -> u32 {
        let index = archetype_id.index();

        if index >= self.archetype_to_bucket.len() {
            self.archetype_to_bucket.resize(index + 1, INVALID_BUCKET);
        }

        let existing = self.archetype_to_bucket[index];

        if existing != INVALID_BUCKET {
            return existing;
        }

        let bucket = u32::try_from(self.buckets.len()).expect("too many sparse driver buckets");
        assert_ne!(bucket, INVALID_BUCKET, "sparse driver exhausted bucket ids");
        self.buckets.push(SparseDriverBucket::new(archetype_id));
        self.archetype_to_bucket[index] = bucket;
        bucket
    }

    /// Returns the mutable `bucket` lane selected by `storage`.
    fn lane_mut(&mut self, bucket: u32, storage: ArchetypeStorageKind) -> &mut SparseDriverLane {
        match storage {
            ArchetypeStorageKind::Contiguous => &mut self.buckets[bucket as usize].contiguous,
            ArchetypeStorageKind::Fragmented => &mut self.buckets[bucket as usize].fragmented,
        }
    }

    /// Pushes one row into the selected lane and returns its slot.
    fn push_to_lane(
        &mut self,
        bucket: u32,
        storage: ArchetypeStorageKind,
        row: SparseDriverRow,
    ) -> u32 {
        self.lane_mut(bucket, storage).push(row)
    }

    /// Swap-removes the lane slot of `entry`, repairing the dense entry that
    /// moved into the vacated slot.
    fn remove_from_lane(&mut self, entry: SparseIndexEntry) {
        let bucket = entry.bucket();
        let storage = entry.storage();
        let slot = entry.slot();

        let lane = self.lane_mut(bucket, storage);

        let moved_dense_pos = lane.swap_remove(slot);

        if let Some(moved_dense_pos) = moved_dense_pos {
            let moved_entry = &mut self.dense_entries[moved_dense_pos as usize];

            let storage = moved_entry.storage();

            moved_entry.lane_slot = PackedLaneSlot::new(slot as u32, storage);
        }
    }

    /// Inserts the member at `dense_pos` into its lane and records its driver
    /// entry, without publishing a revision.
    #[inline]
    fn push_dense_without_revision(
        &mut self,
        dense_pos: u32,
        entity: Entity,
        location: EntityLocation,
    ) {
        let bucket = self.bucket_for(location.archetype_id);
        let bucket_index = bucket as usize;
        let lane = self.buckets[bucket_index].lane_mut(location.storage);
        let slot = lane.push(SparseDriverRow {
            entity,
            row: location.row,
            in_page_jumps: dense_pos,
        });
        let lane_slot = PackedLaneSlot::new(slot, location.storage);

        debug_assert_eq!(
            usize::try_from(dense_pos).ok(),
            Some(self.dense_entries.len())
        );

        self.dense_entries
            .push(SparseIndexEntry::new(bucket, lane_slot));
    }

    /// Inserts one member and publishes one structural revision.
    #[inline]
    pub(crate) fn insert_dense(
        &mut self,
        dense_pos: u32,
        entity: Entity,
        location: EntityLocation,
    ) {
        self.push_dense_without_revision(dense_pos, entity, location);

        self.bump_revision();
    }

    /// Inserts a prepared homogeneous batch and publishes one structural
    /// revision after all driver rows are present.
    pub(crate) fn insert_dense_batch<I>(
        &mut self,
        first_dense_pos: u32,
        additional: usize,
        entries: I,
    ) where
        I: Iterator<Item = (Entity, EntityLocation)>,
    {
        if additional == 0 {
            return;
        }

        self.dense_entries.reserve(additional);

        let mut entries = entries.peekable();
        if let Some((_, first_location)) = entries.peek().copied() {
            let bucket = self.bucket_for(first_location.archetype_id);
            self.lane_mut(bucket, first_location.storage)
                .reserve(additional);
        }

        let mut inserted = 0usize;

        for (offset, (entity, location)) in entries.enumerate() {
            let dense_pos = first_dense_pos
                .checked_add(u32::try_from(offset).expect("sparse batch exceeds u32 capacity"))
                .expect("sparse dense position exceeds u32 capacity");
            self.push_dense_without_revision(dense_pos, entity, location);
            inserted += 1;
        }

        debug_assert_eq!(inserted, additional);
        self.bump_revision();
    }

    /// Updates or transfers one member without publishing a revision; returns
    /// `None` when the dense position is vacant and `Some(true)` when the
    /// driver visibly changed.
    fn update_location_without_revision(
        &mut self,
        dense_pos: u32,
        entity: Entity,
        location: EntityLocation,
    ) -> Option<bool> {
        let entry = self.dense_entries.get(dense_pos as usize).copied()?;

        let new_bucket = self.bucket_for(location.archetype_id);

        if entry.bucket() == new_bucket && entry.storage() == location.storage {
            let changed = {
                let lane = self.lane_mut(entry.bucket(), entry.storage());
                debug_assert_eq!(lane.rows[entry.slot()].entity, entity);
                lane.update_row(entry.slot(), location.row)
            };

            return Some(changed);
        }

        self.remove_from_lane(entry);

        let new_slot = self.push_to_lane(
            new_bucket,
            location.storage,
            SparseDriverRow {
                entity,
                row: location.row,
                in_page_jumps: dense_pos,
            },
        );

        let entry = &mut self.dense_entries[dense_pos as usize];

        entry.bucket = new_bucket;

        entry.lane_slot = PackedLaneSlot::new(new_slot, location.storage);
        Some(true)
    }

    /// Updates the location of one dense entry, returning whether it existed.
    pub(crate) fn update_location(
        &mut self,
        dense_pos: u32,
        entity: Entity,
        location: EntityLocation,
    ) -> bool {
        let Some(changed) = self.update_location_without_revision(dense_pos, entity, location)
        else {
            return false;
        };

        if changed {
            self.bump_revision();
        }

        true
    }

    /// Applies several prevalidated repairs and publishes one revision.
    pub(crate) fn update_locations_batch(&mut self, updates: &[SparseLocationUpdate]) {
        if let Some(changed) = self.try_update_complete_lane(updates) {
            if changed {
                self.bump_revision();
            }
            return;
        }

        let mut changed = false;

        for update in updates {
            let entry = self.dense_entries[update.dense_pos as usize];
            let entity = self.buckets[entry.bucket() as usize]
                .lane(entry.storage())
                .rows[entry.slot()]
            .entity;
            let updated =
                self.update_location_without_revision(update.dense_pos, entity, update.location);
            debug_assert!(
                updated.is_some(),
                "prepared sparse-driver position must exist"
            );
            changed |= updated.unwrap_or(false);
        }

        if changed {
            self.bump_revision();
        }
    }

    /// Rewrites or transfers a complete driver lane without per-row removal.
    ///
    /// Strictly monotonic dense positions prove that the prepared updates are
    /// unique. Combined with full source-lane coverage, this allows the lane
    /// to be cleared or have its locality rebuilt exactly once.
    fn try_update_complete_lane(&mut self, updates: &[SparseLocationUpdate]) -> Option<bool> {
        let first = updates.first()?;

        let unique = updates.len() == 1
            || updates
                .windows(2)
                .all(|pair| pair[0].dense_pos < pair[1].dense_pos)
            || updates
                .windows(2)
                .all(|pair| pair[0].dense_pos > pair[1].dense_pos);

        if !unique {
            return None;
        }

        let source = self.dense_entries.get(first.dense_pos as usize).copied()?;
        let source_bucket = source.bucket();
        let source_storage = source.storage();
        let destination = first.location;

        if updates.iter().any(|update| {
            update.location.archetype_id != destination.archetype_id
                || update.location.storage != destination.storage
                || self
                    .dense_entries
                    .get(update.dense_pos as usize)
                    .is_none_or(|entry| {
                        entry.bucket() != source_bucket || entry.storage() != source_storage
                    })
        }) {
            return None;
        }

        if self.buckets[source_bucket as usize]
            .lane(source_storage)
            .rows
            .len()
            != updates.len()
        {
            return None;
        }

        let destination_bucket = self.bucket_for(destination.archetype_id);

        if source_bucket == destination_bucket && source_storage == destination.storage {
            let mut changed = false;

            for update in updates {
                let entry = self.dense_entries[update.dense_pos as usize];
                let lane = self.lane_mut(source_bucket, source_storage);
                let row = &mut lane.rows[entry.slot()];
                changed |= row.row != update.location.row;
                row.row = update.location.row;
            }

            if changed {
                self.lane_mut(source_bucket, source_storage)
                    .rebuild_locality();
            }

            return Some(changed);
        }

        self.lane_mut(destination_bucket, destination.storage)
            .reserve(updates.len());

        for update in updates {
            let source_entry = self.dense_entries[update.dense_pos as usize];
            let entity = self.buckets[source_bucket as usize]
                .lane(source_storage)
                .rows[source_entry.slot()]
            .entity;
            let slot = self.push_to_lane(
                destination_bucket,
                destination.storage,
                SparseDriverRow {
                    entity,
                    row: update.location.row,
                    in_page_jumps: update.dense_pos,
                },
            );

            let entry = &mut self.dense_entries[update.dense_pos as usize];
            entry.bucket = destination_bucket;
            entry.lane_slot = PackedLaneSlot::new(slot, destination.storage);
        }

        self.lane_mut(source_bucket, source_storage).clear_rows();

        Some(true)
    }

    /// Swap-removes the driver entry at `dense_pos` without publishing a
    /// revision, repairing the row of the entry moved into the vacated
    /// position; returns whether the position held an entry.
    fn remove_dense_without_revision(&mut self, dense_pos: u32) -> bool {
        let dense_pos_usize = dense_pos as usize;
        if dense_pos_usize >= self.dense_entries.len() {
            return false;
        }

        let entry = self.dense_entries[dense_pos_usize];
        let last_dense = self.dense_entries.len() - 1;

        self.remove_from_lane(entry);
        self.dense_entries.swap_remove(dense_pos_usize);

        if dense_pos_usize != last_dense {
            let moved_entry = self.dense_entries[dense_pos_usize];
            let lane = self.lane_mut(moved_entry.bucket(), moved_entry.storage());

            lane.rows[moved_entry.slot()].in_page_jumps = dense_pos;
        }

        true
    }

    /// Removes one dense entry, bumping the revision. Returns whether the
    /// position held an entry.
    pub(crate) fn remove_dense(&mut self, dense_pos: u32) -> bool {
        if !self.remove_dense_without_revision(dense_pos) {
            return false;
        }

        self.bump_revision();
        true
    }

    /// Removes prevalidated dense positions and publishes one structural
    /// revision for the complete group.
    pub(crate) fn remove_dense_batch<I>(&mut self, dense_positions: I)
    where
        I: IntoIterator<Item = u32>,
    {
        let mut changed = false;

        for dense_pos in dense_positions {
            let removed = self.remove_dense_without_revision(dense_pos);
            debug_assert!(removed, "prepared sparse-driver position must exist");
            changed |= removed;
        }

        if changed {
            self.bump_revision();
        }
    }

    /// Clears all driver rows while retaining buckets and their allocations for
    /// rebuilding the same population shape.
    pub(crate) fn clear(&mut self) {
        if self.dense_entries.is_empty() {
            return;
        }

        for bucket in &mut self.buckets {
            bucket.contiguous.clear_rows();
            bucket.fragmented.clear_rows();
        }

        self.dense_entries.clear();
        self.bump_revision();
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use super::*;

    fn location(archetype: usize, storage: ArchetypeStorageKind, row: u32) -> EntityLocation {
        EntityLocation {
            archetype_id: ArchetypeId::new(archetype),
            storage,
            row,
        }
    }

    #[test]
    fn sparse_driver_metadata_keeps_its_compact_layout() {
        assert_eq!(size_of::<PackedLaneSlot>(), 4);
        assert_eq!(size_of::<SparseIndexEntry>(), 8);
        assert_eq!(size_of::<SparseDriverRow>(), 16);
    }

    #[test]
    fn dense_swap_remove_repairs_lane_slot_and_driver_position() {
        let mut index = SparseDriverIndex::default();
        let first = Entity::from_index(1);
        let removed = Entity::from_index(2);
        let moved = Entity::from_index(3);
        let source = location(0, ArchetypeStorageKind::Contiguous, 0);

        index.insert_dense(0, first, source);
        index.insert_dense(1, removed, source);
        index.insert_dense(2, moved, source);

        assert!(index.remove_dense(1));

        assert_eq!(index.dense_entries.len(), 2);

        let moved_entry = index.dense_entries[1];
        let moved_row = index.buckets[moved_entry.bucket() as usize]
            .contiguous
            .rows()[moved_entry.slot()];
        assert_eq!(moved_row.entity, moved);
        assert_eq!(moved_row.in_page_jumps, 1);

        let destination = location(1, ArchetypeStorageKind::Fragmented, 12);
        assert!(index.update_location(1, moved, destination));

        let moved_entry = index.dense_entries[1];
        assert_eq!(moved_entry.storage(), ArchetypeStorageKind::Fragmented);
        let moved_row = index.buckets[moved_entry.bucket() as usize]
            .fragmented
            .rows()[moved_entry.slot()];
        assert_eq!(moved_row.entity, moved);
        assert_eq!(moved_row.row, 12);
        assert_eq!(moved_row.in_page_jumps, 1);
    }
}
