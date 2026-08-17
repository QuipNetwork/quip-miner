//! `ChaCha8Rng` + `PoW` draw-order reference, ported from `shared/chacha8.py`.
//!
//! Produces byte-identical output to the Python reference for cross-language
//! deterministic Ising model generation. Not intended for cryptographic use.

const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]; // "expand 32-byte k"

/// Deterministic `ChaCha8` stream matching the Python golden reference.
///
/// Not a general-purpose CSPRNG — only the `PoW` Ising draw path.
pub struct ChaCha8Rng {
    /// Expanded 256-bit key as eight little-endian words.
    key: [u32; 8],
    /// Block counter (advances after each 16-word block).
    counter: u64,
    /// Current keystream block (16 words).
    block: [u32; 16],
    /// Next unread word index; `16` means the block is exhausted.
    idx: usize,
}

/// `ChaCha` quarter-round on four state indices (standard crypto naming).
///
/// # Panics
/// Panics if any of `a`, `b`, `c`, `d` is out of range for a 16-word state.
/// Call sites pass only compile-time constants in `0..16`.
#[expect(
    clippy::many_single_char_names,
    reason = "standard ChaCha quarter-round parameter names a/b/c/d"
)]
#[expect(
    clippy::indexing_slicing,
    reason = "indices are fixed constants 0..16 from regen callers"
)]
fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]);
    s[d] ^= s[a];
    s[d] = s[d].rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]);
    s[b] ^= s[c];
    s[b] = s[b].rotate_left(7);
}

impl ChaCha8Rng {
    /// Build an RNG from a 32-byte seed (little-endian key words).
    ///
    /// # Panics
    /// Does not panic in practice: `chunks_exact(4)` always yields 4-byte
    /// chunks, and the `expect` only guards that invariant.
    #[must_use]
    pub fn from_seed(key_bytes: [u8; 32]) -> Self {
        let mut key = [0u32; 8];
        for (word, chunk) in key.iter_mut().zip(key_bytes.chunks_exact(4)) {
            *word = u32::from_le_bytes(
                #[expect(
                    clippy::expect_used,
                    reason = "chunks_exact(4) always yields 4-byte chunks"
                )]
                {
                    chunk
                        .try_into()
                        .expect("chunks_exact(4) always yields 4-byte chunks")
                },
            );
        }
        Self {
            key,
            counter: 0,
            block: [0; 16],
            idx: 16,
        }
    }

    fn regen(&mut self) {
        let mut s = [0u32; 16];
        s[0..4].copy_from_slice(&CONSTANTS);
        s[4..12].copy_from_slice(&self.key);
        // Low/high 32 bits of the 64-bit counter (mask/shift make truncation safe).
        s[12] = (self.counter & 0xFFFF_FFFF) as u32;
        s[13] = (self.counter >> 32) as u32;
        s[14] = 0; // stream lo
        s[15] = 0; // stream hi
        let start = s;
        for _ in 0..4 {
            quarter_round(&mut s, 0, 4, 8, 12);
            quarter_round(&mut s, 1, 5, 9, 13);
            quarter_round(&mut s, 2, 6, 10, 14);
            quarter_round(&mut s, 3, 7, 11, 15);
            quarter_round(&mut s, 0, 5, 10, 15);
            quarter_round(&mut s, 1, 6, 11, 12);
            quarter_round(&mut s, 2, 7, 8, 13);
            quarter_round(&mut s, 3, 4, 9, 14);
        }
        for i in 0..16 {
            #[expect(
                clippy::indexing_slicing,
                reason = "i iterates 0..16 over fixed-size [u32; 16] arrays"
            )]
            {
                self.block[i] = s[i].wrapping_add(start[i]);
            }
        }
        self.counter += 1;
        self.idx = 0;
    }

    /// Next little-endian keystream word.
    #[must_use]
    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 16 {
            self.regen();
        }
        #[expect(
            clippy::indexing_slicing,
            reason = "idx is reset to 0 by regen when >= 16, so always in 0..16"
        )]
        let w = self.block[self.idx];
        self.idx += 1;
        w
    }
}

/// Error drawing an Ising model from a nonce.
#[derive(Debug, PartialEq, Eq)]
pub enum DrawError {
    /// An allowed-value set was empty while values were required to be drawn
    /// from it (`n_nodes > 0` with empty `allowed_h`, or `n_edges > 0` with
    /// empty `allowed_j`). A degenerate or non-`Set` chain snapshot reaches
    /// here; drawing would be a modulo-by-zero, so it is rejected instead.
    EmptyAllowedValues,
}

impl std::fmt::Display for DrawError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyAllowedValues => {
                write!(f, "allowed-value set is empty; cannot draw an Ising model")
            }
        }
    }
}

impl std::error::Error for DrawError {}

/// Draw an Ising model (field/coupling milli-values) deterministically from a
/// nonce, selecting each value from the allowed sets via the golden `ChaCha8`
/// draw order.
///
/// # Errors
/// Returns [`DrawError::EmptyAllowedValues`] if a required allowed-value set is
/// empty, rather than panicking on the modulo-by-zero.
pub fn draw_ising_milli(
    nonce: [u8; 32],
    n_nodes: usize,
    n_edges: usize,
    allowed_h: &[i32],
    allowed_j: &[i32],
) -> Result<(Vec<i32>, Vec<i32>), DrawError> {
    if (n_nodes > 0 && allowed_h.is_empty()) || (n_edges > 0 && allowed_j.is_empty()) {
        return Err(DrawError::EmptyAllowedValues);
    }
    let mut rng = ChaCha8Rng::from_seed(nonce);
    let h = (0..n_nodes)
        .map(|_| {
            #[expect(
                clippy::indexing_slicing,
                reason = "empty allowed_h rejected above when n_nodes > 0; index is % len"
            )]
            {
                allowed_h[(rng.next_u32() as usize) % allowed_h.len()]
            }
        })
        .collect();
    let j = (0..n_edges)
        .map(|_| {
            #[expect(
                clippy::indexing_slicing,
                reason = "empty allowed_j rejected above when n_edges > 0; index is % len"
            )]
            {
                allowed_j[(rng.next_u32() as usize) % allowed_j.len()]
            }
        })
        .collect();
    Ok((h, j))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_rejects_empty_allowed_h_when_nodes_present() {
        let err = draw_ising_milli([0u8; 32], 2, 0, &[], &[]);
        assert_eq!(err, Err(DrawError::EmptyAllowedValues));
    }

    #[test]
    fn draw_rejects_empty_allowed_j_when_edges_present() {
        let err = draw_ising_milli([0u8; 32], 1, 1, &[-1000, 1000], &[]);
        assert_eq!(err, Err(DrawError::EmptyAllowedValues));
    }

    #[test]
    fn draw_allows_empty_sets_when_no_draws_needed() {
        // Zero nodes and zero edges: nothing to draw, empty sets are fine.
        let (h, j) = draw_ising_milli([0u8; 32], 0, 0, &[], &[]).unwrap();
        assert!(h.is_empty());
        assert!(j.is_empty());
    }
}
