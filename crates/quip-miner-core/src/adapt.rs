//! Adaptive sampling parameters: map a difficulty target to a sampling budget.
//!
//! Port of `shared/energy_utils.py` (GSE model) + `base_miner.adapt_parameters`.
//! The coordinator advertises the target energy; the miner computes its own
//! `num_reads` / `num_sweeps` from it. Cross-language parity is pinned by
//! `conformance/golden_adapt.json`.

/// SA efficiency range `(c_easy, c_hard)` — verbatim from `energy_utils.py`.
const C_EASY: f64 = 0.7;
const C_HARD: f64 = 0.75;
/// Empirical h-field scaling constant.
const ALPHA: f64 = 0.88;
/// Fallback h set when a topology advertises none (matches Python default).
const DEFAULT_H: [f64; 3] = [-1.0, 0.0, 1.0];

/// Expected ground-state energy: `GSE ≈ -c·√(avg_degree)·N` plus an h-field term.
#[expect(
    clippy::cast_precision_loss,
    reason = "graph sizes for Ising topologies fit exact f64 integer range (mantissa 53 bits)"
)]
fn expected_solution_energy(num_nodes: usize, num_edges: usize, c: f64, h_values: &[f64]) -> f64 {
    if num_nodes == 0 || num_edges == 0 {
        return 0.0;
    }
    let n = num_nodes as f64;
    let avg_degree = (2.0 * num_edges as f64) / n;
    let j_contribution = -c * avg_degree.sqrt() * n;
    let h_contribution = if h_values.len() == 1 && {
        #[expect(
            clippy::indexing_slicing,
            reason = "len() == 1 guarantees index 0 is in bounds"
        )]
        {
            h_values[0] == 0.0
        }
    } {
        0.0
    } else {
        let nonzero = h_values.iter().filter(|&&v| v != 0.0).count() as f64 / h_values.len() as f64;
        -c * ALPHA * nonzero * n / avg_degree.sqrt()
    };
    j_contribution + h_contribution
}

/// `(min_energy @ c_hard, knee @ c_mid, max_energy @ c_easy)`.
fn calc_energy_range(num_nodes: usize, num_edges: usize, h_values: &[f64]) -> (f64, f64, f64) {
    let c_knee = f64::midpoint(C_EASY, C_HARD);
    (
        expected_solution_energy(num_nodes, num_edges, C_HARD, h_values),
        expected_solution_energy(num_nodes, num_edges, c_knee, h_values),
        expected_solution_energy(num_nodes, num_edges, C_EASY, h_values),
    )
}

/// Convert a target energy (milli) to a normalized difficulty in `[0, 1]`.
///
/// `0.0` = easiest (target ≥ `max_energy`), `1.0` = hardest (target ≤ `min_energy`),
/// with a piecewise curve: concave `sqrt` below the knee, convex `square` above.
#[must_use]
pub fn energy_to_difficulty(
    target_milli: i64,
    num_nodes: usize,
    num_edges: usize,
    allowed_h_milli: &[i32],
) -> f64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "milli-energy magnitudes used in practice fit exact f64 integer range"
    )]
    let target = target_milli as f64 / 1000.0;
    let h_owned: Vec<f64>;
    let h_values: &[f64] = if allowed_h_milli.is_empty() {
        &DEFAULT_H
    } else {
        h_owned = allowed_h_milli
            .iter()
            .map(|&v| f64::from(v) / 1000.0)
            .collect();
        &h_owned
    };

    let (min_energy, knee_energy, max_energy) = calc_energy_range(num_nodes, num_edges, h_values);

    if target <= min_energy {
        return 1.0;
    }
    if target >= max_energy {
        return 0.0;
    }

    let total_range = max_energy - min_energy;
    let normalized_pos = (max_energy - target) / total_range;
    let knee_pos = (max_energy - knee_energy) / total_range;

    if normalized_pos <= knee_pos {
        0.5 * (normalized_pos / knee_pos).sqrt()
    } else {
        let progress = (normalized_pos - knee_pos) / (1.0 - knee_pos);
        0.5 + 0.5 * progress * progress
    }
}

/// Per-backend sampling bounds, interpolated by difficulty. Mirrors the
/// `ADAPT_*` class attributes on each Python miner subclass.
#[derive(Clone, Copy, Debug)]
pub struct AdaptBounds {
    /// Lower bound on sweeps after difficulty interpolation.
    pub min_sweeps: u32,
    /// Upper bound on sweeps after difficulty interpolation.
    pub max_sweeps: u32,
    /// Lower bound on reads after difficulty interpolation.
    pub min_reads: u32,
    /// Upper bound on reads after difficulty interpolation.
    pub max_reads: u32,
    /// Multiplier of `min_solutions` for the reads floor (0 disables).
    pub reads_solution_min_factor: u32,
    /// Multiplier of `min_solutions` for the reads ceiling (0 disables).
    pub reads_solution_max_factor: u32,
    /// Multiplier of `min_solutions` applied as a final reads floor.
    pub reads_solution_floor_factor: u32,
}

/// Resolved sampling budget for one job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdaptParams {
    /// Number of independent reads (samples) to take.
    pub num_reads: u32,
    /// Annealing sweeps per read.
    pub num_sweeps: u32,
}

/// Truncate a small non-negative `f64` to `u32` (toward zero), matching the
/// Python `int(difficulty * bound)` used by `base_miner.adapt_parameters`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "difficulty in [0,1] times a small bound is non-negative and well under u32::MAX; \
              truncation toward zero matches Python int()"
)]
fn trunc_u32(x: f64) -> u32 {
    x as u32
}

/// Compute `{num_reads, num_sweeps}` from the difficulty target and backend
/// bounds. Port of the SA/GPU branch of `base_miner.adapt_parameters`.
#[must_use]
pub fn adapt_params(
    target_milli: i64,
    min_solutions: u32,
    num_nodes: usize,
    num_edges: usize,
    allowed_h_milli: &[i32],
    bounds: &AdaptBounds,
) -> AdaptParams {
    let difficulty = energy_to_difficulty(target_milli, num_nodes, num_edges, allowed_h_milli);
    let num_sweeps = bounds
        .min_sweeps
        .max(trunc_u32(difficulty * f64::from(bounds.max_sweeps)));

    let (min_reads, max_reads) = if bounds.reads_solution_min_factor > 0 {
        (
            min_solutions
                .saturating_mul(bounds.reads_solution_min_factor)
                .max(bounds.min_reads),
            min_solutions
                .saturating_mul(bounds.reads_solution_max_factor)
                .max(bounds.max_reads),
        )
    } else {
        (bounds.min_reads, bounds.max_reads)
    };
    let num_reads = min_reads.max(trunc_u32(difficulty * f64::from(max_reads)));
    let floor = min_solutions.saturating_mul(bounds.reads_solution_floor_factor);

    AdaptParams {
        num_reads: num_reads.max(floor),
        num_sweeps,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const CPU_SA: AdaptBounds = AdaptBounds {
        min_sweeps: 64,
        max_sweeps: 4096,
        min_reads: 64,
        max_reads: 512,
        reads_solution_min_factor: 4,
        reads_solution_max_factor: 8,
        reads_solution_floor_factor: 0,
    };

    #[test]
    fn energy_to_difficulty_matches_python_golden() {
        let text = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../conformance/golden_adapt.json"
        ));
        let golden: Value = serde_json::from_str(text).expect("parse golden");
        #[expect(
            clippy::indexing_slicing,
            reason = "golden JSON schema keys are fixed by the conformance fixture"
        )]
        let cases = golden["energy_to_difficulty"].as_array().expect("cases");
        assert!(!cases.is_empty());
        for c in cases {
            let target = c["target_milli"].as_i64().unwrap();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "golden fixture sizes fit usize; milli values fit i32"
            )]
            let n = c["num_nodes"].as_u64().unwrap() as usize;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "golden fixture sizes fit usize"
            )]
            let m = c["num_edges"].as_u64().unwrap() as usize;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "allowed_h_milli golden values fit i32"
            )]
            let h: Vec<i32> = c["allowed_h_milli"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let expected = c["difficulty"].as_f64().unwrap();
            let got = energy_to_difficulty(target, n, m, &h);
            assert!(
                (got - expected).abs() < 1e-9,
                "target={target} n={n} m={m} h={h:?}: got {got}, want {expected}"
            );
        }
    }

    #[test]
    fn adapt_params_matches_python_golden_cpu_sa() {
        let text = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../conformance/golden_adapt.json"
        ));
        let golden: Value = serde_json::from_str(text).expect("parse golden");
        #[expect(
            clippy::indexing_slicing,
            reason = "golden JSON schema keys are fixed by the conformance fixture"
        )]
        let cases = golden["adapt_params_cpu_sa"].as_array().expect("cases");
        assert!(!cases.is_empty());
        for c in cases {
            let target = c["target_milli"].as_i64().unwrap();
            #[expect(
                clippy::cast_possible_truncation,
                reason = "golden fixture sizes fit usize"
            )]
            let n = c["num_nodes"].as_u64().unwrap() as usize;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "golden fixture sizes fit usize"
            )]
            let m = c["num_edges"].as_u64().unwrap() as usize;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "golden min_solutions fits u32"
            )]
            let min_sol = c["min_solutions"].as_u64().unwrap() as u32;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "allowed_h_milli golden values fit i32"
            )]
            let h: Vec<i32> = c["allowed_h_milli"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            let got = adapt_params(target, min_sol, n, m, &h, &CPU_SA);
            #[expect(
                clippy::cast_possible_truncation,
                reason = "golden num_reads/num_sweeps fit u32"
            )]
            {
                assert_eq!(
                    got.num_reads,
                    c["num_reads"].as_u64().unwrap() as u32,
                    "reads {c}"
                );
                assert_eq!(
                    got.num_sweeps,
                    c["num_sweeps"].as_u64().unwrap() as u32,
                    "sweeps {c}"
                );
            }
        }
    }

    #[test]
    fn adapt_params_fixed_reads_bounds_when_no_solution_factor() {
        // CUDA-style bounds: no solution factor -> fixed reads bounds.
        let cuda = AdaptBounds {
            min_sweeps: 256,
            max_sweeps: 2048,
            min_reads: 64,
            max_reads: 256,
            reads_solution_min_factor: 0,
            reads_solution_max_factor: 0,
            reads_solution_floor_factor: 0,
        };
        // Easiest target -> difficulty 0 -> floors: sweeps=256, reads=64.
        let p = adapt_params(0, 5, 4577, 41515, &[-1000, 0, 1000], &cuda);
        assert_eq!(p.num_sweeps, 256);
        assert_eq!(p.num_reads, 64);
    }
}
