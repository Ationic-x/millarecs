//! Compile-time mapping from a bit width to an unsigned integer type.
//!
//! This module is useful when a const generic controls the size of a bit mask or
//! compact counter and the implementation needs the smallest explicitly selected
//! integer type for that width.

/// Type-level representation of an unsigned integer bit width.
pub(crate) struct UsizeBits<const N: usize>;

/// Resolves a supported bit width into its concrete unsigned integer type.
///
/// Only the widths implemented below are valid. Unsupported widths fail at
/// compile time because [`UsizeBitsType`] is not implemented for them.
pub(crate) trait UsizeBitsType {
    /// Unsigned integer type associated with the selected bit width.
    type Type;
}

impl UsizeBitsType for UsizeBits<8> {
    type Type = u8;
}
impl UsizeBitsType for UsizeBits<16> {
    type Type = u16;
}
impl UsizeBitsType for UsizeBits<32> {
    type Type = u32;
}
impl UsizeBitsType for UsizeBits<64> {
    type Type = u64;
}
impl UsizeBitsType for UsizeBits<128> {
    type Type = u128;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    type Bits8 = <UsizeBits<8> as UsizeBitsType>::Type;
    type Bits16 = <UsizeBits<16> as UsizeBitsType>::Type;
    type Bits32 = <UsizeBits<32> as UsizeBitsType>::Type;
    type Bits64 = <UsizeBits<64> as UsizeBitsType>::Type;
    type Bits128 = <UsizeBits<128> as UsizeBitsType>::Type;

    #[test]
    fn each_width_resolves_to_the_matching_unsigned_type() {
        assert_eq!(size_of::<Bits8>(), size_of::<u8>());
        assert_eq!(align_of::<Bits8>(), align_of::<u8>());
        assert_eq!(size_of::<Bits16>(), size_of::<u16>());
        assert_eq!(align_of::<Bits16>(), align_of::<u16>());
        assert_eq!(size_of::<Bits32>(), size_of::<u32>());
        assert_eq!(align_of::<Bits32>(), align_of::<u32>());
        assert_eq!(size_of::<Bits64>(), size_of::<u64>());
        assert_eq!(align_of::<Bits64>(), align_of::<u64>());
        assert_eq!(size_of::<Bits128>(), size_of::<u128>());
        assert_eq!(align_of::<Bits128>(), align_of::<u128>());

        let byte: u8 = Bits8::MAX;
        let word: u16 = Bits16::MAX;
        let dword: u32 = Bits32::MAX;
        let qword: u64 = Bits64::MAX;
        let wide: u128 = Bits128::MAX;

        assert_eq!(byte, u8::MAX);
        assert_eq!(word, u16::MAX);
        assert_eq!(dword, u32::MAX);
        assert_eq!(qword, u64::MAX);
        assert_eq!(wide, u128::MAX);
    }

    #[test]
    fn resolved_types_can_be_used_in_struct_fields() {
        struct Mask<const N: usize>
        where
            UsizeBits<N>: UsizeBitsType,
        {
            raw: <UsizeBits<N> as UsizeBitsType>::Type,
        }

        let mask = Mask::<16> { raw: 0x00ff };
        assert_eq!(mask.raw, 255u16);
    }
}
