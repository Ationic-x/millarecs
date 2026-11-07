//! Cache-line prefetch hints for latency-bound row gathers.
//!
//! Sparse-driver execution walks rows whose table positions are strided: each
//! candidate lands on a fresh cache line for every gathered column, and the
//! loop serializes on memory latency instead of compute. These helpers issue
//! non-binding prefetch hints so the lines of a future row travel towards the
//! cores while execution still works on earlier rows.
//!
//! Hints never fault and never change program semantics; on architectures
//! without a stable prefetch intrinsic the helpers compile to nothing.

/// Issues one read prefetch (`PREFETCHT0` semantics) for a cache line.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub(crate) fn prefetch_read_line(address: *const u8) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_prefetch(address.cast(), core::arch::x86_64::_MM_HINT_T0);
    }

    #[cfg(target_arch = "x86")]
    unsafe {
        core::arch::x86::_mm_prefetch(address.cast(), core::arch::x86::_MM_HINT_T0);
    }
}

/// Fallback for architectures without a stable prefetch intrinsic.
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
#[inline(always)]
pub(crate) fn prefetch_read_line(_address: *const u8) {}

/// Prefetches every cache line touched by the element at `base.add(row)`.
///
/// Components larger than one line can straddle two; both are warmed so the
/// later fetch never stalls mid-element.
#[inline(always)]
pub(crate) fn prefetch_element<T>(base: *const T, row: usize) {
    const CACHE_LINE: usize = 64;

    let start = (base as usize) + row * core::mem::size_of::<T>();
    let last = start + core::mem::size_of::<T>().saturating_sub(1);
    let mut line = start & !(CACHE_LINE - 1);

    while line <= last {
        prefetch_read_line(line as *const u8);
        line += CACHE_LINE;
    }
}
