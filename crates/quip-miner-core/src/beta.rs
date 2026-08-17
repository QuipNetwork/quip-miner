//! Geometric beta (inverse-temperature) ladder for annealing.
//!
//! Port of neal / `shared/beta_schedule._default_ising_beta_range` and the
//! geometric schedule used by the CPU miner and Python GPU samplers.
//!
//! `geometric_beta_schedule` returns `f64`. The CPU Metropolis uses it
//! directly; the GPU backends cast each element to `f32` at upload. cuda/metal
//! already computed the schedule in f64 and cast per element, so the shared
//! f64 math with a per-element cast is bit-identical to the prior behavior.

use crate::ising::IsingGraph;

/// Default hot/cold beta from per-variable effective-field magnitudes.
///
/// - hot: `ln(2) / (2 * max_abs_field)` so worst-case flip ≈ 50%
/// - cold: low single-qubit excitation rate on the smallest non-zero gap
pub fn default_ising_beta_range(graph: &IsingGraph) -> (f64, f64) {
    let n = graph.num_nodes();
    if n == 0 {
        return (0.1, 1.0);
    }

    let mut sum_abs = vec![0.0f64; n];
    let mut min_abs: Vec<Option<f64>> = vec![None; n];

    for (i, &hi) in graph.h.iter().enumerate() {
        let a = hi.abs();
        #[expect(
            clippy::indexing_slicing,
            reason = "i comes from enumerate over h, which has length n == sum_abs/min_abs len"
        )]
        {
            sum_abs[i] += a;
            if a > 0.0 {
                min_abs[i] = Some(a);
            }
        }
    }
    for (k, &(u, v)) in graph.edges.iter().enumerate() {
        if u >= n || v >= n {
            continue;
        }
        let a = graph.j.get(k).copied().unwrap_or(0.0).abs();
        #[expect(
            clippy::indexing_slicing,
            reason = "u and v checked against n; sum_abs/min_abs length is n"
        )]
        {
            sum_abs[u] += a;
            sum_abs[v] += a;
            if a > 0.0 {
                for node in [u, v] {
                    min_abs[node] = Some(match min_abs[node] {
                        Some(m) => m.min(a),
                        None => a,
                    });
                }
            }
        }
    }

    let min_gaps: Vec<f64> = min_abs.into_iter().flatten().collect();
    if min_gaps.is_empty() {
        return (0.1, 1.0);
    }

    let max_eff = sum_abs.iter().copied().fold(0.0f64, f64::max);
    let hot_beta = if max_eff == 0.0 {
        1.0
    } else {
        std::f64::consts::LN_2 / (2.0 * max_eff)
    };

    let min_eff = min_gaps.iter().copied().fold(f64::INFINITY, f64::min);
    #[expect(
        clippy::cast_precision_loss,
        clippy::float_cmp,
        reason = "exact equality is intentional (count entries equal to the min); \
                  gap count fits exact f64 integer range"
    )]
    let number_min_gaps = min_gaps.iter().filter(|&&g| g == min_eff).count() as f64;
    let max_excitation = 0.01f64;
    let cold_beta = (number_min_gaps / max_excitation).ln() / (2.0 * min_eff);

    // Ensure cold is at least as large as hot (schedule increases β).
    (hot_beta, cold_beta.max(hot_beta))
}

/// Geometric beta ladder with `num_betas` points from hot → cold, in f64.
///
/// GPU backends cast the result to f32 per element; the CPU miner uses it as is.
#[must_use]
pub fn geometric_beta_schedule(hot: f64, cold: f64, num_betas: usize) -> Vec<f64> {
    if num_betas == 0 {
        return Vec::new();
    }
    if num_betas == 1 {
        return vec![cold];
    }
    // Guard geometric schedule: both ends must be strictly positive.
    let hot = hot.max(f64::MIN_POSITIVE);
    let cold = cold.max(f64::MIN_POSITIVE);
    let log_hot = hot.ln();
    let log_cold = cold.ln();
    (0..num_betas)
        .map(|i| {
            #[expect(
                clippy::cast_precision_loss,
                reason = "schedule length and index are small; t is a display/interpolation parameter"
            )]
            let t = i as f64 / (num_betas - 1) as f64;
            (log_hot + t * (log_cold - log_hot)).exp()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometric_schedule_endpoints() {
        let s = geometric_beta_schedule(0.1, 10.0, 5);
        assert_eq!(s.len(), 5);
        #[expect(
            clippy::indexing_slicing,
            reason = "schedule length asserted to 5; endpoints are indices 0 and 4"
        )]
        {
            assert!((s[0] - 0.1).abs() < 1e-9);
            assert!((s[4] - 10.0).abs() < 1e-9);
            for w in s.windows(2) {
                assert!(w[0] < w[1]);
            }
        }
    }

    #[test]
    fn gpu_f32_cast_matches_prior_behavior() {
        // GPU wrapper = f64 schedule cast per element; equals computing in f64
        // then `as f32` (cuda/metal's prior code path).
        let f64s = geometric_beta_schedule(0.1, 10.0, 5);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "GPU upload path intentionally narrows f64 schedule rungs to f32"
        )]
        let f32s: Vec<f32> = f64s.iter().map(|&b| b as f32).collect();
        assert_eq!(f32s.len(), 5);
        #[expect(clippy::indexing_slicing, reason = "schedule length asserted to 5")]
        {
            assert!((f64::from(f32s[0]) - 0.1).abs() < 1e-5);
        }
    }
}
