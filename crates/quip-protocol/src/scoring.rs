//! Local (non-consensus) energy and diversity scoring helpers.
//!
//! `energy_milli` accumulates in `f64` for miner-side ranking/logging only.
//! Coordinator/chain re-score with integer-milli arithmetic.

/// Map a spin to ±1.0 for the local `f64` energy path (`s > 0` → `+1.0`).
fn sign(s: i8) -> f64 {
    if s > 0 {
        1.0
    } else {
        -1.0
    }
}

/// Reference `energy_milli`: `Σ h_i·s_i + Σ j_k·s_u·s_v`, scaled to milli and
/// truncated toward zero.
///
/// This SDK path accumulates in `f64` and is a **local, non-consensus** score
/// (miner-side ranking / logging). The coordinator and chain re-score spins with
/// integer-milli arithmetic (`quantum_validation`), which can differ from this
/// `f64` accumulation on pathological inputs (e.g. many small terms summing with
/// rounding). Never trust a miner-reported `energy_milli` for the accept/submit
/// decision — the coordinator always re-scores.
///
/// Range edges are *not* Python-`int()`-identical: a non-finite accumulator
/// (overflow to ±inf) returns the sentinel `1 << 62`, and finite values saturate
/// on the `as i64` cast (Python's arbitrary-precision `int()` never saturates).
#[must_use]
pub fn energy_milli(spins: &[i8], h: &[f64], j: &[f64], edges: &[(usize, usize)]) -> i64 {
    let mut e = 0.0f64;
    for (i, &s) in spins.iter().enumerate() {
        if i < h.len() {
            #[expect(clippy::indexing_slicing, reason = "i < h.len() just checked")]
            {
                e += h[i] * sign(s);
            }
        }
    }
    for (k, &(u, v)) in edges.iter().enumerate() {
        if k < j.len() && u < spins.len() && v < spins.len() {
            #[expect(
                clippy::indexing_slicing,
                reason = "k/u/v bounds checked in the if guard"
            )]
            {
                e += j[k] * sign(spins[u]) * sign(spins[v]);
            }
        }
    }
    if !e.is_finite() {
        return 1i64 << 62;
    }
    // Truncation toward zero, matches Python int(energy*1000). Saturates at
    // i64 bounds on overflow of the cast (documented above).
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Python-parity int(energy*1000) truncation toward zero"
    )]
    {
        (e * 1000.0) as i64
    }
}

/// Flip-invariant Hamming distance between two spin vectors (min of raw and
/// its complement), as `u32`.
#[must_use]
pub fn hamming_flip_invariant(a: &[i8], b: &[i8]) -> u32 {
    let n = a.len();
    let raw = a
        .iter()
        .zip(b)
        // Same threshold as `sign`: s > 0 is +1, else -1.
        .filter(|(x, y)| (**x > 0) != (**y > 0))
        .count();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "API returns u32; distance is at most min(a.len(), b.len()) / 2"
    )]
    {
        raw.min(n - raw) as u32
    }
}

/// Mean pairwise flip-invariant Hamming distance over a solution set, normalized
/// by spin width. Returns `0.0` for fewer than two solutions or zero width.
#[must_use]
pub fn set_diversity(solutions: &[Vec<i8>]) -> f64 {
    if solutions.len() < 2 {
        return 0.0;
    }
    #[expect(
        clippy::indexing_slicing,
        reason = "solutions.len() >= 2 checked above"
    )]
    let n = solutions[0].len();
    if n == 0 {
        return 0.0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "spin width n is problem-sized; f64 mantissa holds typical N"
    )]
    let n = n as f64;
    let mut sum = 0.0f64;
    let mut pairs = 0u64;
    for i in 0..solutions.len() {
        for k in (i + 1)..solutions.len() {
            #[expect(
                clippy::indexing_slicing,
                reason = "i and k range over 0..solutions.len()"
            )]
            let dist = hamming_flip_invariant(&solutions[i], &solutions[k]);
            sum += f64::from(dist) / n;
            pairs += 1;
        }
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "pair count fits exact integer f64 for practical solution-set sizes"
    )]
    {
        sum / pairs as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_positive_sign_and_truncation() {
        // spins [+1,-1]; h=[1.0, -0.5]; edge (0,1) J=2.0
        // E = (1*1) + (-0.5*-1) + (2.0 * 1 * -1) = 1 + 0.5 - 2.0 = -0.5 -> -500 milli
        let e = energy_milli(&[1, -1], &[1.0, -0.5], &[2.0], &[(0, 1)]);
        assert_eq!(e, -500);
    }

    #[test]
    fn energy_truncates_toward_zero() {
        // E = 0.0015 -> int(1.5) -> 1 (truncation, not round to 2)
        let e = energy_milli(&[1], &[0.0015], &[], &[]);
        assert_eq!(e, 1);
    }

    #[test]
    fn energy_oob_edge_is_skipped_not_panicking() {
        // edge (0, 5) references node 5, out of range for a 2-spin problem; must
        // be skipped like a length-mismatched h/j entry, not panic.
        // E = (1*1) + (1*-1) = 0 -> 0 milli
        let e = energy_milli(&[1, -1], &[1.0, 1.0], &[1.0], &[(0, 5)]);
        assert_eq!(e, 0);
    }

    #[test]
    fn diversity_flip_invariant() {
        // a and its exact inverse have flip-invariant distance 0 -> diversity 0
        #[expect(clippy::float_cmp, reason = "exact golden equality for diversity 0")]
        {
            assert_eq!(set_diversity(&[vec![1, 1, -1], vec![-1, -1, 1]]), 0.0);
        }
        // a vs one-bit-flipped: min(1, 2)=1 over N=3 -> 1/3
        assert!((set_diversity(&[vec![1, 1, 1], vec![-1, 1, 1]]) - 1.0 / 3.0).abs() < 1e-12);
        #[expect(clippy::float_cmp, reason = "exact golden equality for <2 solutions")]
        {
            assert_eq!(set_diversity(&[vec![1, 1]]), 0.0); // <2 solutions
        }
    }

    #[test]
    fn diversity_zero_width_solutions_is_zero_not_nan() {
        // Two zero-length solution vectors would divide by n=0; must return
        // 0.0, matching the shared reference (not NaN).
        #[expect(clippy::float_cmp, reason = "exact golden equality for zero-width")]
        {
            assert_eq!(set_diversity(&[vec![], vec![]]), 0.0);
        }
    }

    #[test]
    fn energy_milli_saturates_at_i64_boundary() {
        // e = 1e16 -> e*1000 = 1e19, overflows i64::MAX (~9.223e18) -> saturates.
        assert_eq!(energy_milli(&[1], &[1e16], &[], &[]), i64::MAX);
        // Far past the boundary; must saturate, not panic or produce garbage.
        assert_eq!(energy_milli(&[1], &[1e308], &[], &[]), i64::MAX);
    }

    #[test]
    fn energy_milli_saturates_at_negative_i64_boundary() {
        // Ground states are negative-energy, so the negative overflow path is
        // realistic; it must saturate to i64::MIN, not wrap. Mirrors the Python
        // parity test's -1e16 case.
        assert_eq!(energy_milli(&[1], &[-1e16], &[], &[]), i64::MIN);
        assert_eq!(energy_milli(&[1], &[-1e308], &[], &[]), i64::MIN);
    }

    #[test]
    fn energy_milli_non_finite_returns_sentinel() {
        // An accumulator that overflows to +inf mid-sum returns the exact
        // sentinel 1<<62, distinct from the i64::MAX/MIN saturation values.
        assert_eq!(energy_milli(&[1, 1], &[1e308, 1e308], &[], &[]), 1i64 << 62);
    }
}
