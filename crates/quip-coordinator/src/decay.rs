//! Pure difficulty-decay math (agm.2.1).
//!
//! The chain eases `max_energy_milli` upward each epoch since the last winning
//! proof, so a candidate that does not clear the current threshold eventually
//! wins as difficulty decays. This module ports the runtime-110 decay curve
//! (`pallets/quantum-pow/src/difficulty.rs`, mirrored by the v0.2
//! `substrate/difficulty_decay.py`) to the milli: `ease_step` walks
//! `max_energy_milli` a geometric fraction of the remaining gap toward the easy
//! cap; `build_decay_schedule` unrolls it; `step_for_energy` finds a candidate's
//! win step. No chain dependency — the caller supplies the curve + base.
//!
//! Golden-faithful: `round` is half-away-from-zero (Rust `f64::round`, matching
//! `libm::round`), and every add/sub saturates at i64 bounds.

/// Geometric ease fraction, in milli (rate = 25/1000 = 0.025).
pub const DECAY_RATE_MILLI: i64 = 25;
/// Minimum per-step easing when the geometric term rounds below it (one energy
/// unit in milli).
pub const MIN_ENERGY_DELTA_MILLI: i64 = 1000;

// The pallet exposes these as constants/defaults, not via any runtime API, so
// (per the independent-reads path) the coordinator mirrors them. They match
// quip-protocol-rs @ v0.2 `QuantumPow`; a chain retune must be echoed here.
/// Blocks per decay epoch (`QuantumPow::EpochLength`).
pub const EPOCH_LENGTH_BLOCKS: u64 = 100;
/// Default curve calibration c-triple (per-mille: 700 == 0.70), from the
/// `CurveC{Easy,Knee,Hard}Milli` constants. A topology may override these via
/// `TopologyCurveC` storage, probed separately.
pub const DEFAULT_C_EASY_MILLI: u32 = 700;
/// Default knee calibration c (per-mille: 725 == 0.725).
pub const DEFAULT_C_KNEE_MILLI: u32 = 725;
/// Default hard calibration c (per-mille: 750 == 0.75).
pub const DEFAULT_C_HARD_MILLI: u32 = 750;
/// Default base difficulty `max_energy_milli` when `Difficulties[hash]` is unset
/// (`DifficultyConfig::default()` on chain: `{5, -1_200_000, 200}`).
pub const DEFAULT_BASE_MAX_ENERGY_MILLI: i64 = -1_200_000;

/// GSE-estimate bounds for the decay curve (all negative). `ease_step` only
/// references `min_milli`/`max_milli`; `knee_milli` is retained for parity with
/// the chain curve and the (separate) curve-construction path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnergyCurve {
    /// Hard (most negative) GSE bound in milli.
    pub min_milli: i64,
    /// Knee GSE bound in milli.
    pub knee_milli: i64,
    /// Easy (least negative) GSE bound in milli — ease cap.
    pub max_milli: i64,
}

/// `round(room * rate)` (half away from zero) floored at `min_delta`.
fn geometric_floored(room: i64, rate_milli: i64, min_delta: i64) -> i64 {
    #[expect(
        clippy::cast_precision_loss,
        reason = "decay room and rate fit exact f64 for operational magnitudes"
    )]
    let rate = rate_milli as f64 / 1000.0;
    // f64::round is half-away-from-zero (libm::round); float→int cast saturates.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "geometric step is intentionally rounded f64; magnitudes fit i64"
    )]
    let stepped = (room as f64 * rate).round() as i64;
    stepped.max(min_delta)
}

/// One EASIER decay step on `current` max-energy toward `curve.max_milli`,
/// capped there (never ease past the easy bound). A degenerate curve
/// (`max <= min`) or a current already at/past the cap is a no-op.
#[must_use]
pub fn ease_step(current: i64, curve: &EnergyCurve) -> i64 {
    if curve.max_milli <= curve.min_milli {
        return current;
    }
    let room = curve.max_milli.saturating_sub(current);
    if room <= 0 {
        return current;
    }
    let g = geometric_floored(room, DECAY_RATE_MILLI, MIN_ENERGY_DELTA_MILLI);
    current.saturating_add(g.min(room))
}

/// Apply `steps` EASIER steps to `base_max_energy_milli`.
#[must_use]
pub fn apply_decay(base_max_energy_milli: i64, steps: u64, curve: &EnergyCurve) -> i64 {
    let mut cur = base_max_energy_milli;
    for _ in 0..steps {
        cur = ease_step(cur, curve);
    }
    cur
}

/// Active `max_energy_milli` at `block_number`: base difficulty with per-epoch
/// decay for blocks elapsed since the last winning proof. `last_proof_block == 0`
/// (genesis) or `epoch_length == 0` disables decay, as does a `None` curve.
#[must_use]
pub fn current_max_energy(
    block_number: u64,
    base_max_energy_milli: i64,
    last_proof_block: u64,
    epoch_length: u64,
    curve: Option<&EnergyCurve>,
) -> i64 {
    if last_proof_block == 0 || epoch_length == 0 {
        return base_max_energy_milli;
    }
    let elapsed = block_number.saturating_sub(last_proof_block);
    let steps = elapsed / epoch_length;
    match curve {
        Some(c) if steps > 0 => apply_decay(base_max_energy_milli, steps, c),
        _ => base_max_energy_milli,
    }
}

/// `max_energy_milli` threshold at each decay step `0..=horizon` (inclusive),
/// built incrementally. Monotonic non-decreasing (decay only eases upward). A
/// `None` curve yields a flat schedule (decay disabled).
#[must_use]
pub fn build_decay_schedule(
    base_max_energy_milli: i64,
    curve: Option<&EnergyCurve>,
    horizon: usize,
) -> Vec<i64> {
    let mut sched = Vec::with_capacity(horizon + 1);
    sched.push(base_max_energy_milli);
    match curve {
        None => {
            for _ in 0..horizon {
                sched.push(base_max_energy_milli);
            }
        }
        Some(c) => {
            let mut cur = base_max_energy_milli;
            for _ in 0..horizon {
                cur = ease_step(cur, c);
                sched.push(cur);
            }
        }
    }
    sched
}

/// First step `s` where `schedule[s] > floor_energy_milli` (the strict
/// `best_energy_milli < max_energy_milli` gate), or `None` if a candidate with
/// that floor never clears within the schedule's horizon. `schedule` is
/// monotonic non-decreasing, so this is a binary search (`bisect_right`).
#[must_use]
pub fn step_for_energy(schedule: &[i64], floor_energy_milli: i64) -> Option<usize> {
    let i = schedule.partition_point(|&t| t <= floor_energy_milli);
    (i < schedule.len()).then_some(i)
}

/// One unit == this many milli (a value of 1000 milli is magnitude 1.0).
const MILLI_SCALE: f64 = 1000.0;
/// Field-term weight in the GSE estimate (`quantum_validation` `DEFAULT_H_ALPHA`).
const DEFAULT_H_ALPHA: f64 = 0.88;

/// Mean |value| of a discrete allowed-value set on the unit scale (1.0 ==
/// `MILLI_SCALE` milli), under uniform sampling. Mirrors
/// `quantum_validation::energy::mean_abs_unit` for the `AllowedValueSet` variant
/// (the coordinator's snapshot carries allowed values as a discrete milli set).
/// An empty set contributes nothing.
fn mean_abs_unit(allowed_milli: &[i32]) -> f64 {
    if allowed_milli.is_empty() {
        return 0.0;
    }
    let sum_abs: i64 = allowed_milli.iter().map(|&v| i64::from(v).abs()).sum();
    #[expect(
        clippy::cast_precision_loss,
        reason = "allowed-set magnitude sums and lengths fit exact f64 here"
    )]
    {
        sum_abs as f64 / (allowed_milli.len() as f64 * MILLI_SCALE)
    }
}

/// Expected ground-state-energy estimate (milli) for a topology + calibration
/// constant `c`, given its field/coupling value sets. Port of
/// `quantum_validation::expected_gse_for_specs`; must match the chain to the
/// milli or the whole `EnergyCurve` (and decay trajectory) drifts. Zero nodes or
/// edges yields 0 (matches the pallet guard).
#[must_use]
pub fn expected_gse_milli(
    num_nodes: u64,
    num_edges: u64,
    c: f64,
    allowed_h_milli: &[i32],
    allowed_j_milli: &[i32],
) -> i64 {
    if num_nodes == 0 || num_edges == 0 {
        return 0;
    }
    let h_mean_abs = mean_abs_unit(allowed_h_milli);
    let j_mean_abs = mean_abs_unit(allowed_j_milli);
    #[expect(
        clippy::cast_precision_loss,
        reason = "topology node/edge counts used as f64 magnitudes in GSE formula"
    )]
    let n = num_nodes as f64;
    #[expect(
        clippy::cast_precision_loss,
        reason = "topology node/edge counts used as f64 magnitudes in GSE formula"
    )]
    let m = num_edges as f64;
    let avg_degree = (2.0 * m) / n;
    let sqrt_avg_degree = avg_degree.sqrt();
    let j_contribution = -c * j_mean_abs * sqrt_avg_degree * n;
    let h_contribution = -c * DEFAULT_H_ALPHA * h_mean_abs * n / sqrt_avg_degree;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "GSE estimate is intentionally rounded to nearest milli i64"
    )]
    {
        ((j_contribution + h_contribution) * MILLI_SCALE).round() as i64
    }
}

impl EnergyCurve {
    /// Build the decay curve from a topology and the chain's calibration
    /// c-triple (stored scaled milli: `700` == `0.70`). `min`/`knee`/`max` are
    /// the GSE estimates at `c_hard`/`c_knee`/`c_easy`; since a larger `c` is
    /// more negative, `min_milli < knee_milli < max_milli` for any legitimate
    /// input. Mirrors `EnergyCurve::from_topology` in `difficulty.rs`.
    #[must_use]
    pub fn from_topology(
        num_nodes: u64,
        num_edges: u64,
        c_easy_milli: u32,
        c_knee_milli: u32,
        c_hard_milli: u32,
        allowed_h_milli: &[i32],
        allowed_j_milli: &[i32],
    ) -> Self {
        let gse = |c_milli: u32| {
            expected_gse_milli(
                num_nodes,
                num_edges,
                f64::from(c_milli) / MILLI_SCALE,
                allowed_h_milli,
                allowed_j_milli,
            )
        };
        Self {
            min_milli: gse(c_hard_milli),
            knee_milli: gse(c_knee_milli),
            max_milli: gse(c_easy_milli),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A wide curve so the geometric term (not the cap or floor) drives the walk.
    fn curve() -> EnergyCurve {
        EnergyCurve {
            min_milli: -100_000,
            knee_milli: -50_000,
            max_milli: -1_000,
        }
    }

    #[test]
    fn ease_step_walks_geometric_fraction_toward_cap() {
        let c = curve();
        // room = -1000 - (-50000) = 49000; round(49000*0.025)=1225 → -48775
        assert_eq!(ease_step(-50_000, &c), -48_775);
        // room = 47775; round(1194.375)=1194 → -47581
        assert_eq!(ease_step(-48_775, &c), -47_581);
        // room = 46581; round(1164.525)=1165 → -46416
        assert_eq!(ease_step(-47_581, &c), -46_416);
    }

    #[test]
    fn ease_step_is_capped_and_noop_at_easy_bound() {
        let c = curve();
        // room = -1000 - (-1200) = 200; geometric floors to 1000, min(1000,200)=200
        // → clamped exactly to the cap, then a no-op.
        assert_eq!(ease_step(-1_200, &c), -1_000);
        assert_eq!(ease_step(-1_000, &c), -1_000);
        // Degenerate curve leaves current alone.
        let degenerate = EnergyCurve {
            min_milli: -1_000,
            knee_milli: -1_000,
            max_milli: -1_000,
        };
        assert_eq!(ease_step(-50_000, &degenerate), -50_000);
    }

    #[test]
    fn apply_decay_matches_step_by_step() {
        let c = curve();
        assert_eq!(apply_decay(-50_000, 0, &c), -50_000);
        assert_eq!(apply_decay(-50_000, 3, &c), -46_416);
    }

    #[test]
    fn build_schedule_is_prefix_of_apply_decay() {
        let c = curve();
        let sched = build_decay_schedule(-50_000, Some(&c), 3);
        assert_eq!(sched, vec![-50_000, -48_775, -47_581, -46_416]);
        // None curve → flat schedule.
        assert_eq!(
            build_decay_schedule(-50_000, None, 3),
            vec![-50_000, -50_000, -50_000, -50_000]
        );
    }

    #[test]
    fn step_for_energy_finds_first_clearing_step() {
        let sched = vec![-50_000, -48_775, -47_581, -46_416];
        // A floor equal to the base clears at step 1 (strict >).
        assert_eq!(step_for_energy(&sched, -50_000), Some(1));
        // -48000 clears when the threshold first exceeds it → step 2.
        assert_eq!(step_for_energy(&sched, -48_000), Some(2));
        // A floor the schedule never exceeds within the horizon → None.
        assert_eq!(step_for_energy(&sched, -1), None);
    }

    #[test]
    fn current_max_energy_applies_epoch_decay() {
        let c = curve();
        // Genesis / no epoch → no decay.
        assert_eq!(current_max_energy(1_000, -50_000, 0, 10, Some(&c)), -50_000);
        assert_eq!(
            current_max_energy(1_000, -50_000, 100, 0, Some(&c)),
            -50_000
        );
        // elapsed 25, epoch 10 → 2 steps.
        assert_eq!(
            current_max_energy(125, -50_000, 100, 10, Some(&c)),
            apply_decay(-50_000, 2, &c)
        );
        // No curve → base regardless of elapsed.
        assert_eq!(current_max_energy(125, -50_000, 100, 10, None), -50_000);
    }

    #[test]
    fn expected_gse_matches_golden_zero_field() {
        // Zero-field h drops the h term; legacy J {-1000,1000} has unit mean 1.0.
        // (1024, 2048): avg_degree 4, sqrt 2, j = -0.75*1*2*1024 = -1536 → milli.
        assert_eq!(
            expected_gse_milli(1024, 2048, 0.75, &[0], &[-1000, 1000]),
            -1_536_000
        );
    }

    #[test]
    fn expected_gse_zero_nodes_or_edges_is_zero() {
        assert_eq!(expected_gse_milli(0, 2048, 0.75, &[0], &[-1000, 1000]), 0);
        assert_eq!(expected_gse_milli(1024, 0, 0.75, &[0], &[-1000, 1000]), 0);
    }

    #[test]
    fn from_topology_orders_bounds_hard_lt_knee_lt_easy() {
        let c = EnergyCurve::from_topology(
            1024,
            2048,
            700,
            750,
            800,
            &[-1000, 0, 1000],
            &[-1000, 1000],
        );
        // A larger c is more negative: hard (800) < knee (750) < easy (700).
        assert!(c.min_milli < c.knee_milli);
        assert!(c.knee_milli < c.max_milli);
        assert!(c.max_milli < 0);
    }
}
