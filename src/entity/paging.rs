//! Paged storage for entity metadata.
//!
//! Entity metadata is stored in fixed-size pages. A page manager owns large page
//! blocks, exposes only active pages through `VecPage`, and keeps inactive pages
//! in an intrusive freelist. This avoids moving metadata after an entity index
//! has been issued.

pub(crate) use crate::entity::metadata::EntityMeta;
use crate::utils::{
    ecs_vec::EcsVec,
    type_size::{UsizeBits, UsizeBitsType},
};
use std::{
    mem::MaybeUninit,
    ptr::{self, NonNull},
    sync::atomic::AtomicUsize,
};

/// Number of low bits used as the index inside a metadata page.
pub(crate) const PAGE_BITS: usize = 8;
/// Number of metadata entries stored in one page.
pub(crate) const PAGE_SIZE: usize = 1 << PAGE_BITS;
/// Number of low bits used as the index inside a page block.
const BLOCK_BITS: usize = 5;
/// Number of pages stored in one block allocation.
const BLOCK_PAGES: usize = 1 << BLOCK_BITS;

/// Returns the page index that contains `entity_id`.
#[inline(always)]
pub(crate) const fn page_of_entity(entity_id: usize) -> usize {
    entity_id >> PAGE_BITS
}

/// Returns the row of `entity_id` inside its page.
#[inline(always)]
pub(crate) const fn index_in_page(entity_id: usize) -> usize {
    entity_id & (PAGE_SIZE - 1)
}

/// Returns the block index that owns `page_id`.
#[inline(always)]
pub(crate) const fn block_of_page(page_id: usize) -> usize {
    page_id >> BLOCK_BITS
}

/// Returns the slot of `page_id` inside its block.
#[inline(always)]
pub(crate) const fn index_in_block(page_id: usize) -> usize {
    page_id & (BLOCK_PAGES - 1)
}

/// Allocation policy for newly created page blocks.
#[derive(Clone, Copy)]
pub(crate) struct BlockPolicy {
    /// Number of pages to initialize immediately when allocating a block.
    pub pre_touch: usize,
}

impl BlockPolicy {
    /// Creates a policy with an explicit pre-touch count.
    pub const fn new(pre_touch: usize) -> BlockPolicy {
        BlockPolicy { pre_touch }
    }

    /// Consumes `amount` from the pre-touch budget, saturating at zero.
    fn dec_by(&mut self, amount: usize) {
        self.pre_touch = self.pre_touch.saturating_sub(amount);
    }
}

impl Default for BlockPolicy {
    /// Default policy initializes only the first page needed by the free list.
    fn default() -> BlockPolicy {
        BlockPolicy { pre_touch: 0 }
    }
}

/// One page of entity metadata.
///
/// Inactive pages temporarily use their first bytes as [`FreePageHeader`]. When a
/// page becomes active, the manager writes a fresh `Page` or restores the first
/// metadata slot if the page was already initialized.
#[repr(C, align(8))]
pub(crate) struct Page {
    metalist: [EntityMeta; PAGE_SIZE],
}

impl Page {
    /// Creates a page whose metadata slots start as invalid entities.
    #[inline]
    fn new() -> Self {
        Self {
            metalist: [EntityMeta::INVALID; PAGE_SIZE],
        }
    }

    /// Returns the immutable metadata array for this page.
    #[inline]
    pub fn metalist(&self) -> &[EntityMeta; PAGE_SIZE] {
        &self.metalist
    }

    /// Returns the mutable metadata array for this page.
    #[inline]
    pub fn mut_metalist(&mut self) -> &mut [EntityMeta; PAGE_SIZE] {
        &mut self.metalist
    }
}

/// Intrusive next pointer stored inside free pages.
#[repr(C)]
struct FreePageHeader {
    next_head: *mut Page,
}

/// Bitmap integer type holding one init bit per page in a block.
type BlockUsize = <UsizeBits<{ BLOCK_PAGES }> as UsizeBitsType>::Type;

/// Owned allocation containing a fixed number of pages.
pub(crate) struct PageBlock {
    block: Box<[MaybeUninit<Page>; BLOCK_PAGES]>,
    /// Bit `n` is set once page `n` in the block has been initialized.
    init_mask: BlockUsize,
}

impl PageBlock {
    /// Allocates an uninitialized block; pages are initialized lazily.
    fn new() -> Self {
        Self {
            block: Box::new(core::array::from_fn(|_| MaybeUninit::uninit())),
            init_mask: 0,
        }
    }
}

/// Sparse vector containing only active metadata pages.
pub(crate) type VecPage = EcsVec<NonNull<Page>, AtomicUsize>;

/// Owns metadata pages and hands active pages to the entity manager.
#[derive(Default)]
pub(crate) struct PageManager {
    /// Head of the intrusive freelist of inactive pages.
    page_head: Option<*mut Page>,
    /// Blocks that own every page allocation.
    blocks: Vec<PageBlock>,
}

impl PageManager {
    /// Creates a manager and activates the first metadata page.
    pub fn new(pages: &mut VecPage) -> Self {
        let mut pm = PageManager::default();

        pm.allocate_block(pages, &BlockPolicy::default());

        let _ = pm.pop(pages);

        pm
    }

    /// Returns the number of physical pages owned by all blocks.
    #[inline]
    pub fn total_pages(&self) -> usize {
        self.blocks.len() * BLOCK_PAGES
    }

    /// Returns the active page that contains `entity_id`.
    pub fn get<'a>(&self, pages: &'a VecPage, entity_id: u32) -> Option<&'a Page> {
        let page_id = page_of_entity(entity_id as usize);

        pages.get(page_id).map(|page| unsafe { page.as_ref() })
    }

    /// Returns the active page that contains `entity_id` mutably.
    pub fn get_mut<'a>(&self, pages: &'a mut VecPage, entity_id: u32) -> Option<&'a mut Page> {
        let page_id = page_of_entity(entity_id as usize);

        pages.get_mut(page_id).map(|page| unsafe { page.as_mut() })
    }

    /// Activates one page from the freelist and appends it to `pages`.
    ///
    /// Panics when the freelist is empty.
    #[inline]
    pub fn pop(&mut self, pages: &mut VecPage) -> NonNull<Page> {
        let page = self.page_head.expect("Freelist is empty");
        let page_id = pages.len();

        let header = page as *mut FreePageHeader;
        unsafe {
            self.page_head = {
                let next = (*header).next_head;

                if next.is_null() {
                    None
                } else {
                    Some(next)
                }
            };

            if self.is_init(page_id) {
                // The freelist header overwrites the first metadata slot while
                // inactive, so restore that slot before exposing the page.
                (*page).metalist[0] = EntityMeta::INVALID;
            } else {
                page.write(Page::new());
            }

            let block_i = block_of_page(page_id);
            self.blocks[block_i].init_mask |= 1 << index_in_block(page_id);

            pages.dec_nonneg();

            let page = NonNull::new_unchecked(page);
            pages.push(page);
            page
        }
    }

    /// Adds enough blocks so the free-page count can satisfy `additional`.
    pub fn reserve(&mut self, pages: &mut VecPage, additional: usize, policy: &BlockPolicy) {
        let mut policy = *policy;

        let new_count = additional
            .checked_add(pages.count())
            .expect("Too many freepages");

        while pages.count() < new_count {
            self.allocate_block(pages, &policy);
            policy.dec_by(BLOCK_PAGES);
        }
    }

    /// Initializes enough physical pages to later fit `additional` new entities.
    ///
    /// This method does not activate pages in `pages`; activation is delayed
    /// until entity reservation needs to write metadata into those pages.
    pub fn reserve_from_entities(
        &mut self,
        pages: &mut VecPage,
        additional: usize,
        next_id: usize,
    ) {
        if additional == 0 {
            return;
        }

        let target_id = next_id
            .checked_add(additional - 1)
            .expect("Too many entities");

        let target_page = page_of_entity(target_id);

        if self.is_init(target_page) {
            return;
        }

        let start_page = page_of_entity(next_id);
        let total_pages = self.total_pages();

        if start_page < total_pages {
            let existing_end = target_page.min(total_pages - 1);

            for page_id in start_page..=existing_end {
                if !self.is_init(page_id) {
                    self.init(page_id);
                }
            }
        }

        if total_pages <= target_page {
            let missing_pages = target_page + 1 - total_pages;

            self.reserve(pages, missing_pages, &BlockPolicy::new(missing_pages));
        }
    }

    /// Returns whether the page's metadata array has been initialized.
    fn is_init(&self, page_id: usize) -> bool {
        let block_id = block_of_page(page_id);
        let page_in_block = index_in_block(page_id);

        if let Some(page_block) = self.blocks.get(block_id) {
            let bit = 1 << page_in_block;
            (page_block.init_mask & bit) != 0
        } else {
            false
        }
    }

    /// Initializes the page's metadata array, preserving its freelist link.
    fn init(&mut self, page_id: usize) {
        let block_id = block_of_page(page_id);
        let page_in_block = index_in_block(page_id);

        if let Some(page_block) = self.blocks.get_mut(block_id) {
            if (page_block.init_mask & (1 << page_in_block)) > 0 {
                return;
            }

            let init_ptr = page_block.block[page_in_block].as_mut_ptr();

            unsafe {
                // Preserve the freelist link stored in the uninitialized page
                // while writing the metadata array into the same memory.
                let header = init_ptr as *mut FreePageHeader;
                let saved_next = (*header).next_head;

                init_ptr.write(Page::new());

                let page_header = init_ptr as *mut FreePageHeader;
                (*page_header).next_head = saved_next;
            }

            page_block.init_mask |= 1 << page_in_block;
        }
    }

    /// Allocates one block and pushes every page in that block to the freelist.
    fn allocate_block(&mut self, pages: &mut VecPage, policy: &BlockPolicy) {
        let mut page_block = PageBlock::new();
        let pre = policy.pre_touch.max(1);
        let n_blocks = self.blocks.len();

        let mut last_page: *mut Page;
        let allow_init;

        if n_blocks > 0 {
            last_page = self.blocks[n_blocks - 1].block[BLOCK_PAGES - 1].as_mut_ptr();
            // Continue pre-touching only when the previous block ended
            // initialized, so initialized pages stay a prefix.
            allow_init = self.is_init(n_blocks * BLOCK_PAGES - 1);
        } else {
            last_page = ptr::null_mut();
            allow_init = true;
        }

        for (i, slot) in page_block.block.iter_mut().enumerate() {
            let page_ptr = slot.as_mut_ptr();

            if i < pre && allow_init {
                page_block.init_mask |= 1 << i;
                unsafe { page_ptr.write(Page::new()) };
            }

            if self.page_head.is_none() {
                self.page_head = Some(page_ptr);
            } else {
                let header = last_page as *mut FreePageHeader;

                unsafe {
                    (*header).next_head = page_ptr;
                }
            }

            last_page = page_ptr;
        }

        let last_hdr = last_page as *mut FreePageHeader;
        unsafe {
            (*last_hdr).next_head = core::ptr::null_mut();
        }

        self.blocks.push(page_block);
        pages.inc_by(BLOCK_PAGES);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::component::archetype::ArchetypeId;
    use crate::entity::metadata::{ArchetypeStorageKind, EntityLocation};

    fn mask_for_first(amount: usize) -> BlockUsize {
        let amount = amount.min(BLOCK_PAGES);
        if amount == 0 {
            0
        } else if amount == BLOCK_PAGES {
            BlockUsize::MAX
        } else {
            ((1 as BlockUsize) << amount) - 1
        }
    }

    fn manager_with_reserved_block(policy: BlockPolicy) -> (PageManager, VecPage) {
        let mut manager = PageManager::default();
        let mut pages = VecPage::default();
        manager.reserve(&mut pages, 1, &policy);
        (manager, pages)
    }

    #[test]
    fn entity_and_page_indices_are_bit_partitions() {
        assert_eq!(index_in_page(0), 0);
        assert_eq!(index_in_page(PAGE_SIZE - 1), PAGE_SIZE - 1);
        assert_eq!(index_in_page(PAGE_SIZE), 0);
        assert_eq!(index_in_page(PAGE_SIZE + 7), 7);

        assert_eq!(page_of_entity(0), 0);
        assert_eq!(page_of_entity(PAGE_SIZE - 1), 0);
        assert_eq!(page_of_entity(PAGE_SIZE), 1);
        assert_eq!(page_of_entity(PAGE_SIZE * 3 + 4), 3);

        assert_eq!(index_in_block(0), 0);
        assert_eq!(index_in_block(BLOCK_PAGES - 1), BLOCK_PAGES - 1);
        assert_eq!(index_in_block(BLOCK_PAGES), 0);

        assert_eq!(block_of_page(0), 0);
        assert_eq!(block_of_page(BLOCK_PAGES - 1), 0);
        assert_eq!(block_of_page(BLOCK_PAGES), 1);
    }

    #[test]
    fn block_policy_decrements_with_saturation() {
        let mut policy = BlockPolicy::new(10);

        policy.dec_by(3);
        assert_eq!(policy.pre_touch, 7);

        policy.dec_by(usize::MAX);
        assert_eq!(policy.pre_touch, 0);
    }

    #[test]
    fn page_new_initializes_metadata_as_invalid() {
        let page = Page::new();

        assert_eq!(page.metalist()[0].location, EntityLocation::INVALID);
        assert_eq!(
            page.metalist()[PAGE_SIZE - 1].location,
            EntityLocation::INVALID
        );
        assert_eq!(
            page.metalist()[0].generation,
            EntityMeta::INVALID.generation
        );
        assert_eq!(
            page.metalist()[PAGE_SIZE - 1].generation,
            EntityMeta::INVALID.generation
        );
    }

    #[test]
    fn page_layout_can_store_freelist_header_in_first_slot() {
        assert!(std::mem::align_of::<Page>() >= std::mem::align_of::<FreePageHeader>());
        assert!(std::mem::size_of::<EntityMeta>() >= std::mem::size_of::<FreePageHeader>());
    }

    #[test]
    fn page_manager_new_owns_one_block_and_activates_first_page() {
        let mut pages = VecPage::default();
        let manager = PageManager::new(&mut pages);

        assert_eq!(manager.total_pages(), BLOCK_PAGES);
        assert_eq!(manager.blocks.len(), 1);
        assert_eq!(pages.len(), 1);
        assert_eq!(pages.count(), BLOCK_PAGES - 1);

        assert!(manager.get(&pages, 0).is_some());
        assert!(manager.get(&pages, (PAGE_SIZE - 1) as u32).is_some());
        assert!(manager.get(&pages, PAGE_SIZE as u32).is_none());
        assert!(manager.is_init(0));
    }

    #[test]
    fn reserve_allocates_pages_without_activating_them() {
        let (manager, pages) = manager_with_reserved_block(BlockPolicy::default());

        assert_eq!(manager.blocks.len(), 1);
        assert_eq!(manager.total_pages(), BLOCK_PAGES);
        assert_eq!(pages.len(), 0);
        assert_eq!(pages.count(), BLOCK_PAGES);
        assert_eq!(manager.blocks[0].init_mask, mask_for_first(1));
        assert!(manager.page_head.is_some());
    }

    #[test]
    fn reserve_pretouches_requested_pages_in_new_block() {
        let pre_touch = 6;
        let (manager, pages) = manager_with_reserved_block(BlockPolicy::new(pre_touch));

        assert_eq!(pages.len(), 0);
        assert_eq!(pages.count(), BLOCK_PAGES);
        assert_eq!(manager.blocks[0].init_mask, mask_for_first(pre_touch));

        for page_id in 0..pre_touch {
            assert!(manager.is_init(page_id));
        }
        for page_id in pre_touch..BLOCK_PAGES {
            assert!(!manager.is_init(page_id));
        }
    }

    #[test]
    fn pop_activates_pages_in_freelist_order() {
        let (mut manager, mut pages) = manager_with_reserved_block(BlockPolicy::default());

        let first_ptr = manager.blocks[0].block[0].as_ptr() as *mut Page;
        let second_ptr = manager.blocks[0].block[1].as_ptr() as *mut Page;

        let first = manager.pop(&mut pages);
        let second = manager.pop(&mut pages);

        assert_eq!(first.as_ptr(), first_ptr);
        assert_eq!(second.as_ptr(), second_ptr);
        assert_eq!(pages.len(), 2);
        assert_eq!(pages.count(), BLOCK_PAGES - 2);
        assert!(manager.get(&pages, 0).is_some());
        assert!(manager.get(&pages, PAGE_SIZE as u32).is_some());
        assert!(manager.get(&pages, (PAGE_SIZE * 2) as u32).is_none());
        assert_eq!(manager.blocks[0].init_mask, mask_for_first(2));
    }

    #[test]
    fn get_mut_allows_updating_active_page_metadata() {
        let (mut manager, mut pages) = manager_with_reserved_block(BlockPolicy::default());
        manager.pop(&mut pages);

        let expected = EntityLocation {
            archetype_id: ArchetypeId::new(8),
            storage: ArchetypeStorageKind::Contiguous,
            row: 42,
        };

        manager.get_mut(&mut pages, 0).unwrap().mut_metalist()[0].location = expected;

        assert_eq!(
            manager.get(&pages, 0).unwrap().metalist()[0].location,
            expected
        );
    }

    #[test]
    fn reserve_from_entities_initializes_needed_pages_without_activating_them() {
        let (mut manager, mut pages) = manager_with_reserved_block(BlockPolicy::default());
        let additional_entities = PAGE_SIZE * 3 + 1;

        manager.reserve_from_entities(&mut pages, additional_entities, 0);

        assert_eq!(pages.len(), 0);
        assert_eq!(pages.count(), BLOCK_PAGES);
        for page_id in 0..=3 {
            assert!(manager.is_init(page_id));
        }
        assert!(!manager.is_init(4));
        assert_eq!(manager.blocks[0].init_mask, mask_for_first(4));
    }

    #[test]
    fn reserve_from_entities_can_allocate_and_pretouch_new_blocks() {
        let (mut manager, mut pages) = manager_with_reserved_block(BlockPolicy::default());
        let additional_entities = PAGE_SIZE * (BLOCK_PAGES + 2);

        manager.reserve_from_entities(&mut pages, additional_entities, 0);

        assert_eq!(manager.blocks.len(), 2);
        assert_eq!(manager.total_pages(), BLOCK_PAGES * 2);
        assert_eq!(pages.len(), 0);
        assert_eq!(pages.count(), BLOCK_PAGES * 2);
        assert_eq!(manager.blocks[0].init_mask, mask_for_first(BLOCK_PAGES));
        assert_eq!(manager.blocks[1].init_mask, mask_for_first(2));
    }

    #[test]
    #[should_panic(expected = "Freelist is empty")]
    fn pop_panics_when_no_free_pages_remain() {
        let (mut manager, mut pages) = manager_with_reserved_block(BlockPolicy::default());

        for _ in 0..=BLOCK_PAGES {
            manager.pop(&mut pages);
        }
    }
}
