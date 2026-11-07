//! Page-level primitives for archetype storage lanes.
//!
//! Pages are fixed at `ARCHETYPE_PAGE_ROWS` rows. Contiguous pages only track a
//! dense live length, while fragmented pages additionally track a live mask and
//! a free list for recycled offsets.

/// Number of rows stored in each archetype page.
pub(crate) const ARCHETYPE_PAGE_ROWS: usize = 64;
/// Free-list sentinel marking the end of a page's recycled-offset chain.
const INVALID_FREE: u16 = u16::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Page index and local offset for a global archetype row.
pub(crate) struct PageRow {
    /// Index of the page inside the storage lane.
    pub(crate) page: usize,
    /// Row offset inside the page.
    pub(crate) offset: usize,
}

#[inline]
/// Converts a global row index into page coordinates.
pub(crate) const fn page_row(row: usize) -> PageRow {
    PageRow {
        page: row / ARCHETYPE_PAGE_ROWS,
        offset: row % ARCHETYPE_PAGE_ROWS,
    }
}

#[inline]
/// Returns how many fixed-size pages are needed for `rows` rows.
pub(crate) fn pages_for_rows(rows: usize) -> usize {
    rows.div_ceil(ARCHETYPE_PAGE_ROWS)
}

/// Metadata shared by page policies.
///
/// Fragmented storage uses all fields. Contiguous storage keeps its own smaller
/// metadata because it only needs a dense live count.
struct Page {
    /// Initialized slots, forming a dense prefix from offset zero.
    initialized_len: u16,
    /// Head offset of the intrusive free list, or `INVALID_FREE`.
    free_head: u16,
    /// Slots currently occupied by a live row.
    live_count: u16,
    /// Intrusive free-list links, indexed by offset.
    next_free: [u16; ARCHETYPE_PAGE_ROWS],
}

impl Page {
    /// Creates a page with no initialized slots and an empty free list.
    fn new() -> Self {
        Self {
            initialized_len: 0,
            free_head: INVALID_FREE,
            live_count: 0,
            next_free: [INVALID_FREE; ARCHETYPE_PAGE_ROWS],
        }
    }
}

/// Page metadata for fragmented storage.
///
/// The live mask selects occupied offsets. Freed offsets are linked through
/// `next_free`, and the page can also be linked into the storage lane's
/// non-full page list.
pub(crate) struct FragmentedPage {
    live_mask: u64,
    page: Page,
    /// Successor inside the owning lane's non-full page list, when linked.
    pub(crate) next_non_full: Option<usize>,
    /// Whether this page is currently linked into a lane's non-full list.
    pub(crate) in_non_full_list: bool,
}

impl FragmentedPage {
    /// Creates an empty fragmented page.
    pub(crate) fn new() -> Self {
        Self {
            live_mask: 0,
            page: Page::new(),
            next_non_full: None,
            in_non_full_list: false,
        }
    }

    #[inline]
    /// Returns the current live-row bit mask.
    pub(crate) fn live_mask(&self) -> u64 {
        self.live_mask
    }

    #[inline]
    /// Returns initialized but currently dead rows in this page.
    pub(crate) fn dead_rows(&self) -> usize {
        let page = &self.page;
        page.initialized_len as usize - page.live_count as usize
    }

    /// Allocates one offset, reusing a freed slot before growing the page tail.
    pub(crate) fn allocate(&mut self) -> Option<usize> {
        let page = &mut self.page;

        if page.free_head != INVALID_FREE {
            let offset: usize = page.free_head as usize;
            page.free_head = page.next_free[offset];
            page.next_free[offset] = INVALID_FREE;
            self.live_mask |= 1_u64 << offset;
            page.live_count += 1;
            return Some(offset);
        }

        if page.initialized_len as usize == ARCHETYPE_PAGE_ROWS {
            return None;
        }

        let offset = page.initialized_len as usize;
        page.initialized_len += 1;
        self.live_mask |= 1_u64 << offset;
        page.live_count += 1;
        Some(offset)
    }

    #[inline]
    /// Returns whether every page slot is live.
    pub(crate) fn is_full(&self) -> bool {
        self.page.live_count as usize == ARCHETYPE_PAGE_ROWS
    }

    /// Marks a live offset as free and links it into the page free list.
    pub(crate) fn free(&mut self, offset: usize) {
        let page = &mut self.page;
        debug_assert!(offset < page.initialized_len as usize);
        debug_assert!(self.live_mask & (1_u64 << offset) != 0);

        self.live_mask &= !(1_u64 << offset);
        page.next_free[offset] = page.free_head;
        page.free_head = offset as u16;
        page.live_count -= 1;
    }

    /// Frees every live offset selected by `mask`.
    ///
    /// Returns the number of rows released.
    pub(crate) fn free_mask(&mut self, mut mask: u64) -> usize {
        if mask == 0 {
            return 0;
        }

        let initialized_mask = dense_live_mask(self.page.initialized_len as usize);

        debug_assert_eq!(
            mask & !initialized_mask,
            0,
            "removal mask contains uninitialized offsets"
        );

        debug_assert_eq!(
            mask & !self.live_mask,
            0,
            "removal mask contains non-live offsets"
        );

        let original_mask = mask;
        let removed = mask.count_ones() as u16;
        let page = &mut self.page;

        while mask != 0 {
            let offset = mask.trailing_zeros() as usize;
            mask &= mask - 1; // Clears the lowest set bit just consumed.

            page.next_free[offset] = page.free_head;
            page.free_head = offset as u16;
        }

        self.live_mask &= !original_mask;
        page.live_count = page
            .live_count
            .checked_sub(removed)
            .expect("fragmented page removal exceeded its live count");

        removed as usize
    }

    /// Marks a source offset as no longer live after compaction moved it away.
    pub(crate) fn mark_moved_from(&mut self, offset: usize) {
        let page = &mut self.page;
        debug_assert!(self.live_mask & (1_u64 << offset) != 0);
        self.live_mask &= !(1_u64 << offset);
        page.live_count -= 1;
    }

    /// Marks a destination offset as live after compaction moved a row into it.
    pub(crate) fn mark_moved_to(&mut self, offset: usize) {
        let page = &mut self.page;
        debug_assert!(self.live_mask & (1_u64 << offset) == 0);
        self.live_mask |= 1_u64 << offset;
        page.live_count += 1;
        page.initialized_len = page.initialized_len.max(offset as u16 + 1);
    }

    /// Rebuilds page metadata for a dense prefix after compaction.
    pub(crate) fn rebuild_dense(&mut self, live_rows: usize) {
        let page = &mut self.page;
        debug_assert!(live_rows <= ARCHETYPE_PAGE_ROWS);

        self.live_mask = dense_live_mask(live_rows);
        page.initialized_len = live_rows as u16;
        page.live_count = live_rows as u16;
        page.free_head = INVALID_FREE;
        page.next_free = [INVALID_FREE; ARCHETYPE_PAGE_ROWS];
        self.next_non_full = None;
        self.in_non_full_list = false;
    }
}

/// Page metadata for contiguous storage.
///
/// Contiguous pages never contain internal holes, so a live count is enough to
/// derive the live mask and the next allocation offset.
pub(crate) struct ContiguousPage {
    live_count: u16,
}

impl ContiguousPage {
    /// Creates an empty contiguous page.
    pub(crate) fn new() -> Self {
        Self { live_count: 0 }
    }

    #[inline]
    /// Returns a dense live mask covering the first `live_count` offsets.
    pub(crate) fn live_mask(&self) -> u64 {
        dense_live_mask(self.live_count as usize)
    }

    #[inline]
    /// Returns whether every page slot is occupied.
    pub(crate) fn is_full(&self) -> bool {
        self.live_count as usize == ARCHETYPE_PAGE_ROWS
    }

    /// Allocates the next trailing offset.
    pub(crate) fn allocate(&mut self) -> Option<usize> {
        if self.is_full() {
            return None;
        }

        let offset = self.live_count as usize;
        self.live_count += 1;
        Some(offset)
    }

    /// Removes the current tail offset from a dense page.
    pub(crate) fn mark_tail_removed(&mut self, offset: usize) {
        debug_assert!(offset < self.live_count as usize);
        debug_assert_eq!(offset + 1, self.live_count as usize);
        self.live_count -= 1;
    }
}

/// Returns a mask with the first `live_rows` offsets set, special-casing the
/// empty and full page so the shift never reaches the page width.
#[inline]
fn dense_live_mask(live_rows: usize) -> u64 {
    match live_rows {
        0 => 0,
        ARCHETYPE_PAGE_ROWS => u64::MAX,
        rows => (1_u64 << rows) - 1,
    }
}
