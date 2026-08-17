//! Little-endian i32 and spin-byte wire codecs for Ising payloads.

/// Wire codec errors.
#[derive(Debug, PartialEq)]
pub enum WireError {
    /// Byte slice length is not a multiple of 4 (i32 LE).
    BadLength,
    /// Spin byte was neither `0x01` (+1) nor `0xFF` (−1).
    BadSpinByte(u8),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadLength => write!(f, "byte length is not a multiple of 4"),
            Self::BadSpinByte(b) => write!(f, "invalid spin byte: 0x{b:02X}"),
        }
    }
}

impl std::error::Error for WireError {}

/// Encode `i32` values as little-endian bytes (4 bytes per value).
#[must_use]
pub fn encode_i32_le(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Decode little-endian `i32` values from a byte slice.
///
/// # Errors
/// Returns [`WireError::BadLength`] if `bytes.len()` is not a multiple of 4.
pub fn decode_i32_le(bytes: &[u8]) -> Result<Vec<i32>, WireError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(WireError::BadLength);
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| {
            #[expect(
                clippy::indexing_slicing,
                reason = "chunks_exact(4) yields exactly 4 bytes"
            )]
            {
                i32::from_le_bytes([c[0], c[1], c[2], c[3]])
            }
        })
        .collect())
}

/// Encode `{-1,+1}` spins to the one-byte wire form (`0x01`/`0xFF`).
///
/// Spins are `{-1,+1}` by contract. The `s > 0` boundary is deliberate: it
/// matches [`crate::scoring::sign`] so a stray `0` maps to the same spin
/// (`-1`/`0xFF`) in both the wire byte and the energy scorer. Using `s >= 0`
/// here would encode `0` as `+1` while the scorer treats it as `-1`, silently
/// disagreeing on a consensus-scored value.
#[must_use]
pub fn encode_spins(spins: &[i8]) -> Vec<u8> {
    spins
        .iter()
        .map(|&s| if s > 0 { 0x01u8 } else { 0xFFu8 })
        .collect()
}

/// Decode one-byte wire spins (`0x01`/`0xFF`) to `{-1,+1}` `i8` values.
///
/// # Errors
/// Returns [`WireError::BadSpinByte`] for any byte other than `0x01` or `0xFF`.
pub fn decode_spins(bytes: &[u8]) -> Result<Vec<i8>, WireError> {
    bytes
        .iter()
        .map(|&b| match b {
            0x01 => Ok(1i8),
            0xFF => Ok(-1i8),
            other => Err(WireError::BadSpinByte(other)),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i32_le_roundtrip_and_bytes() {
        // -1000 milli little-endian = 0x18 0xFC 0xFF 0xFF
        assert_eq!(
            encode_i32_le(&[-1000, 0, 1000]),
            vec![0x18, 0xFC, 0xFF, 0xFF, 0, 0, 0, 0, 0xE8, 0x03, 0, 0]
        );
        assert_eq!(
            decode_i32_le(&encode_i32_le(&[-1000, 0, 1000])).unwrap(),
            vec![-1000, 0, 1000]
        );
        assert!(matches!(
            decode_i32_le(&[1, 2, 3]),
            Err(WireError::BadLength)
        ));
    }

    #[test]
    fn spins_bytes() {
        assert_eq!(encode_spins(&[1, -1, 1]), vec![0x01, 0xFF, 0x01]);
        assert_eq!(decode_spins(&[0x01, 0xFF, 0x01]).unwrap(), vec![1, -1, 1]);
        assert!(matches!(
            decode_spins(&[0x00]),
            Err(WireError::BadSpinByte(0))
        ));
    }

    #[test]
    fn encode_spins_zero_matches_scorer_sign() {
        // Spins are {-1,+1} by contract, but a stray 0 must encode consistently
        // with `scoring::sign` (which maps 0 -> -1, since it uses `s > 0`).
        // Both now map 0 to the -1 byte, so the wire byte and the energy scorer
        // agree on a consensus-scored value.
        assert_eq!(encode_spins(&[0]), vec![0xFF]);
    }

    #[test]
    fn wire_i32_roundtrip_at_i32_bounds() {
        // i32::MIN/MAX are named load-bearing edge values for the LE codec.
        let vals = [i32::MIN, -1, 0, 1, i32::MAX];
        let bytes = encode_i32_le(&vals);
        assert_eq!(decode_i32_le(&bytes).unwrap(), vals);
        // Byte-level check of the sign boundary (little-endian).
        assert_eq!(&encode_i32_le(&[i32::MIN]), &[0x00, 0x00, 0x00, 0x80]);
        assert_eq!(&encode_i32_le(&[i32::MAX]), &[0xFF, 0xFF, 0xFF, 0x7F]);
    }

    #[test]
    fn wire_empty_payload_roundtrips() {
        assert_eq!(encode_i32_le(&[]), Vec::<u8>::new());
        assert_eq!(decode_i32_le(&[]).unwrap(), Vec::<i32>::new());
        assert_eq!(encode_spins(&[]), Vec::<u8>::new());
        assert_eq!(decode_spins(&[]).unwrap(), Vec::<i8>::new());
    }
}
