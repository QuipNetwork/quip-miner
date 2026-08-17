//! `PoW` nonce derivation: BLAKE3 over (`last_proof`, miner, salt).
//!
//! Golden-pinned against `conformance/golden_vectors.json` `derive_nonce`.

/// Derive the canonical 32-byte `PoW` nonce.
///
/// Input order is load-bearing: `last_proof` then `miner` then `salt`.
/// Mirrors `shared.quantum_proof_of_work.derive_nonce` and
/// `quantum_validation::derive_nonce`.
#[must_use]
pub fn derive_nonce(last_proof: [u8; 32], miner: [u8; 32], salt: [u8; 32]) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    let _ = h.update(&last_proof);
    let _ = h.update(&miner);
    let _ = h.update(&salt);
    *h.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_golden_vector() {
        // conformance/golden_vectors.json derive_nonce[0]
        let last = [0u8; 32];
        let miner = [1u8; 32];
        let salt = [2u8; 32];
        let nonce = derive_nonce(last, miner, salt);
        // b4179357b751254ed0e68b5e969dcb50e73fd8c56be192b79d286ff2722d6a72
        let expected: [u8; 32] = [
            0xb4, 0x17, 0x93, 0x57, 0xb7, 0x51, 0x25, 0x4e, 0xd0, 0xe6, 0x8b, 0x5e, 0x96, 0x9d,
            0xcb, 0x50, 0xe7, 0x3f, 0xd8, 0xc5, 0x6b, 0xe1, 0x92, 0xb7, 0x9d, 0x28, 0x6f, 0xf2,
            0x72, 0x2d, 0x6a, 0x72,
        ];
        assert_eq!(nonce, expected);
    }

    #[test]
    fn input_order_is_load_bearing() {
        let a = derive_nonce([0u8; 32], [1u8; 32], [2u8; 32]);
        let b = derive_nonce([1u8; 32], [0u8; 32], [2u8; 32]);
        assert_ne!(a, b);
    }

    #[test]
    fn salt_is_load_bearing() {
        // A bug that dropped `salt` from the hash would pass `input_order_is_
        // load_bearing` (which only permutes last_proof/miner); pin salt too.
        let a = derive_nonce([0u8; 32], [1u8; 32], [2u8; 32]);
        let b = derive_nonce([0u8; 32], [1u8; 32], [3u8; 32]);
        assert_ne!(a, b);
    }
}
