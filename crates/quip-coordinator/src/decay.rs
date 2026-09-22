//! Pure difficulty-decay math (agm.2.1).
//!
//! The chain eases `max_energy_milli` upward each epoch since the
//! last winning proof, so a candidate that does not clear the current
//! threshold eventually wins as difficulty decays. This module mirrors both
//! rules the chain has shipped: `ease_step` is the runtime-117 step, one
//! geometric fraction of the remaining gap per whole epoch, and
//! `ease_continuous` is the runtime-118 closed form (quip-validator !87) that
//! eases every block and faster past the target round length. No chain
//! dependency — the caller supplies the curve + base.
//!
//! Golden-faithful: `round` is half-away-from-zero (Rust `f64::round`, matching
//! `libm::round`), and every add/sub saturates at i64 bounds.

/// Geometric ease fraction, in milli (rate = 25/1000 = 0.025).
pub const DECAY_RATE_MILLI: i64 = 25;
/// Minimum per-step easing when the geometric term rounds below it (one energy
/// unit in milli).
pub const MIN_ENERGY_DELTA_MILLI: i64 = 1000;

/// Round length the chain targets (`TARGET_PROOF_BLOCKS` in the pallet). Past
/// this many blocks in one round the continuous rule eases faster.
pub const TARGET_PROOF_BLOCKS: u64 = 100;
/// Extra easing rate per epoch for overdue blocks
/// (`OVERDUE_EASE_RATE_MILLI` in the pallet). Equal to `DECAY_RATE_MILLI`, so
/// an overdue round eases at twice the baseline rate.
pub const OVERDUE_EASE_RATE_MILLI: i64 = 25;
/// First runtime `spec_version` that decays per block (quip-validator !87).
/// Earlier runtimes step once per whole epoch.
pub const CONTINUOUS_DECAY_SPEC_VERSION: u32 = 118;

// The pallet exposes these as constants/defaults, not via any runtime API, so
// (per the independent-reads path) the coordinator mirrors them. They match
// quip-validator @ v0.2 `QuantumPow`; a chain retune must be echoed here.
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

/// Ease `room` milli over `blocks` at `rate` per epoch, in closed form,
/// reproducing the retired per-epoch loop. That loop stepped
/// `max(round(room * rate), MIN_ENERGY_DELTA_MILLI)` once per epoch, clamped
/// to the room: geometric while `room * rate` beat the floor, linear at the
/// floor after. The crossover room is `floor / rate`, and the geometric
/// phase lasts `ln(crossover / room) / ln(1 - rate)` epochs. Returns the
/// amount eased, at most `room`.
///
/// Transcribed from the pallet's `ease_room`. Keep the expression order:
/// the golden table pins the rounding.
fn ease_room(room: i64, blocks: u64, epoch_length: u64, rate: f64) -> i64 {
    if room <= 0 || blocks == 0 || epoch_length == 0 || rate <= 0.0 || rate >= 1.0 {
        return 0;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "room, blocks, and epoch length fit exact f64 for operational magnitudes"
    )]
    let (floor, room_f, epochs) = (
        MIN_ENERGY_DELTA_MILLI as f64,
        room as f64,
        blocks as f64 / epoch_length as f64,
    );
    let crossover = floor / rate;
    let geometric_epochs = if room_f <= crossover {
        0.0
    } else {
        (libm::log(crossover / room_f) / libm::log(1.0 - rate)).min(epochs)
    };
    let room_after_geometric = room_f * libm::pow(1.0 - rate, geometric_epochs);
    let linear = (epochs - geometric_epochs) * floor;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "eased amount is a rounded f64 at most room; the cast saturates"
    )]
    let eased = libm::round(room_f - room_after_geometric + linear) as i64;
    eased.min(room)
}

/// EASIER decay of `current` after `elapsed_blocks` blocks under the
/// runtime-118 rule (quip-validator !87): every block eases, at the baseline
/// rate for the first [`TARGET_PROOF_BLOCKS`] of the round and at the
/// combined baseline-plus-overdue rate after. The two phases run in
/// sequence, the second from the room the first left, so the threshold is
/// continuous across the target boundary. At every whole epoch inside the
/// target the result equals [`ease_step`] applied that many times. Capped at
/// `curve.max_milli`. A degenerate curve, zero elapsed, or zero epoch is a
/// no-op.
///
/// Transcribed from the pallet's `ease_continuous`. Keep the expression
/// order: `1.0 - baseline_retained` is not exactly `0.025` in f64, and the
/// golden table pins the result.
#[must_use]
pub fn ease_continuous(
    current: i64,
    elapsed_blocks: u64,
    epoch_length: u64,
    curve: &EnergyCurve,
) -> i64 {
    if curve.max_milli <= curve.min_milli || elapsed_blocks == 0 || epoch_length == 0 {
        return current;
    }
    let room = curve.max_milli.saturating_sub(current);
    if room <= 0 {
        return current;
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "per-mille rates are small integers, exact in f64"
    )]
    let baseline_retained = 1.0 - DECAY_RATE_MILLI as f64 / 1000.0;
    #[expect(
        clippy::cast_precision_loss,
        reason = "per-mille rates are small integers, exact in f64"
    )]
    let overdue_retained = baseline_retained * (1.0 - OVERDUE_EASE_RATE_MILLI as f64 / 1000.0);
    let in_target = elapsed_blocks.min(TARGET_PROOF_BLOCKS);
    let overdue = elapsed_blocks - in_target;
    let first = ease_room(room, in_target, epoch_length, 1.0 - baseline_retained);
    let second = ease_room(room - first, overdue, epoch_length, 1.0 - overdue_retained);
    current.saturating_add((first + second).min(room))
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

    /// The pallet's `walkup_curve()`: the curve its golden table is pinned on.
    fn walkup_curve() -> EnergyCurve {
        EnergyCurve {
            min_milli: -16_000_000,
            knee_milli: -15_600_000,
            max_milli: -14_000_000,
        }
    }

    #[test]
    #[expect(
        clippy::too_many_lines,
        reason = "the complete pallet golden table stays together for verbatim parity"
    )]
    fn ease_continuous_matches_the_runtime_118_golden_table() {
        // (offset above min_milli, elapsed blocks, eased max_energy_milli), the
        // `eased` column of quip-validator's
        // `apply_decay_and_adjust_on_proof_golden_table` on `walkup_curve()`
        // with epoch 100. Regenerate there with
        // `cargo test -p pallet-quantum-pow print_golden_table -- --ignored --nocapture`.
        #[expect(
            clippy::unreadable_literal,
            reason = "golden literals are copied verbatim from the pallet table"
        )]
        const GOLDEN: &[(i64, u64, i64)] = &[
            (0, 1, -15999494),
            (0, 7, -15996459),
            (0, 30, -15984867),
            (0, 59, -15970347),
            (0, 60, -15969848),
            (0, 99, -15950494),
            (0, 100, -15950000),
            (0, 101, -15949013),
            (0, 150, -15901250),
            (0, 200, -15853719),
            (0, 500, -15592471),
            (0, 1000, -15236282),
            (0, 10000, -14011452),
            (1000, 1, -15998494),
            (1000, 7, -15995460),
            (1000, 30, -15983874),
            (1000, 59, -15969362),
            (1000, 60, -15968863),
            (1000, 99, -15949519),
            (1000, 100, -15949025),
            (1000, 101, -15948038),
            (1000, 150, -15900299),
            (1000, 200, -15852792),
            (1000, 500, -15591675),
            (1000, 1000, -15235664),
            (1000, 10000, -14011442),
            (20000, 1, -15979499),
            (20000, 7, -15976494),
            (20000, 30, -15965018),
            (20000, 59, -15950644),
            (20000, 60, -15950150),
            (20000, 99, -15930989),
            (20000, 100, -15930500),
            (20000, 101, -15929523),
            (20000, 150, -15882237),
            (20000, 200, -15835182),
            (20000, 500, -15576546),
            (20000, 1000, -15223920),
            (20000, 10000, -14011254),
            (80000, 1, -15919514),
            (80000, 7, -15916600),
            (80000, 30, -15905472),
            (80000, 59, -15891533),
            (80000, 60, -15891054),
            (80000, 99, -15872474),
            (80000, 100, -15872000),
            (80000, 101, -15871052),
            (80000, 150, -15825200),
            (80000, 200, -15779570),
            (80000, 500, -15528772),
            (80000, 1000, -15186831),
            (80000, 10000, -14010646),
            (400000, 1, -15599595),
            (400000, 7, -15597167),
            (400000, 30, -15587893),
            (400000, 59, -15576278),
            (400000, 60, -15575879),
            (400000, 99, -15560395),
            (400000, 100, -15560000),
            (400000, 101, -15559210),
            (400000, 150, -15521000),
            (400000, 200, -15482975),
            (400000, 500, -15273977),
            (400000, 1000, -14989026),
            (400000, 10000, -14007045),
            (1000000, 1, -14999747),
            (1000000, 7, -14998229),
            (1000000, 30, -14992433),
            (1000000, 59, -14985174),
            (1000000, 60, -14984924),
            (1000000, 99, -14975247),
            (1000000, 100, -14975000),
            (1000000, 101, -14974506),
            (1000000, 150, -14950625),
            (1000000, 200, -14926859),
            (1000000, 500, -14796236),
            (1000000, 1000, -14618141),
            (1000000, 10000, -14000000),
            (1960000, 1, -14039990),
            (1960000, 7, -14039930),
            (1960000, 30, -14039700),
            (1960000, 59, -14039410),
            (1960000, 60, -14039400),
            (1960000, 99, -14039010),
            (1960000, 100, -14039000),
            (1960000, 101, -14038980),
            (1960000, 150, -14038025),
            (1960000, 200, -14037074),
            (1960000, 500, -14031849),
            (1960000, 1000, -14024726),
            (1960000, 10000, -14000000),
            (1999000, 1, -14000990),
            (1999000, 7, -14000930),
            (1999000, 30, -14000700),
            (1999000, 59, -14000410),
            (1999000, 60, -14000400),
            (1999000, 99, -14000010),
            (1999000, 100, -14000000),
            (1999000, 101, -14000000),
            (1999000, 150, -14000000),
            (1999000, 200, -14000000),
            (1999000, 500, -14000000),
            (1999000, 1000, -14000000),
            (1999000, 10000, -14000000),
        ];
        let c = walkup_curve();
        assert_eq!(GOLDEN.len(), 104, "table covers the whole grid");
        for &(offset, elapsed, want) in GOLDEN {
            let got = ease_continuous(c.min_milli + offset, elapsed, 100, &c);
            assert_eq!(got, want, "offset {offset}, elapsed {elapsed}");
        }
    }

    #[test]
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "the reference loop mirrors the retired pallet rule's float casts"
    )]
    fn ease_continuous_matches_the_stepwise_reference_at_every_epoch() {
        // The retired rule stepped once per epoch by max(round(room * rate),
        // 1_000), clamped to the room. Inside the target the rate is the
        // baseline; past it, the combined baseline-plus-overdue rate. The
        // closed form lands within one milli per epoch of that loop.
        let c = walkup_curve();
        let epoch = 100_u64;
        let target_epochs = TARGET_PROOF_BLOCKS / epoch;
        let reference = |start: i64, epochs: u64| -> i64 {
            let mut value = start;
            for k in 0..epochs {
                let rate = if k < target_epochs {
                    0.025
                } else {
                    1.0 - 0.975 * 0.975
                };
                let room = c.max_milli - value;
                if room <= 0 {
                    break;
                }
                let step = ((room as f64 * rate).round() as i64).max(1_000).min(room);
                value += step;
            }
            value
        };
        for start in [
            c.min_milli,
            c.knee_milli,
            c.max_milli - 161_000,
            c.max_milli - 45_000,
            c.max_milli - 3_500,
        ] {
            for epochs in [1_u64, 2, 5, 10, 25, 50, 100, 200] {
                let closed = ease_continuous(start, epochs * epoch, epoch, &c);
                let stepped = reference(start, epochs);
                assert!(
                    (closed - stepped).abs() <= epochs as i64,
                    "start {start}, epochs {epochs}: closed {closed}, stepped {stepped}"
                );
            }
        }
    }

    #[test]
    fn ease_continuous_over_one_epoch_equals_one_stepwise_step() {
        let c = walkup_curve();
        for start in [c.min_milli, c.knee_milli, c.max_milli - 50_000] {
            assert_eq!(
                ease_continuous(start, 100, 100, &c),
                ease_step(start, &c),
                "start={start}"
            );
        }
    }

    #[test]
    fn ease_continuous_eases_within_the_first_epoch() {
        let c = walkup_curve();
        #[expect(
            clippy::manual_midpoint,
            reason = "the fixed walkup bounds cannot overflow and preserve the reference test"
        )]
        let start = (c.min_milli + c.max_milli) / 2;
        let half = ease_continuous(start, 50, 100, &c);
        let full = ease_continuous(start, 100, 100, &c);
        assert!(
            start < half && half < full,
            "half an epoch must ease strictly between zero and one epoch \
             (start={start}, half={half}, full={full})"
        );
    }

    #[test]
    fn overdue_easing_has_no_jump_at_the_target() {
        let c = walkup_curve();
        let eased = |elapsed: u64| ease_continuous(c.knee_milli, elapsed, 100, &c);
        let before = eased(100) - eased(99);
        let after = eased(101) - eased(100);
        assert!(
            before > 0,
            "baseline decay must move the threshold per block"
        );
        assert!(
            after > before && after <= 2 * before + 2,
            "per-block easing must double past target, not jump: before {before}, after {after}"
        );
    }

    #[test]
    fn ease_continuous_is_monotone_in_elapsed_blocks() {
        let c = walkup_curve();
        for start in [c.min_milli, c.knee_milli, c.max_milli - 3_500] {
            let mut previous = start;
            for elapsed in 1..=2_000_u64 {
                let now = ease_continuous(start, elapsed, 100, &c);
                assert!(now >= previous, "decay went backwards at elapsed={elapsed}");
                previous = now;
            }
        }
    }

    #[test]
    fn ease_continuous_is_a_noop_without_room_or_time() {
        let c = walkup_curve();
        assert_eq!(ease_continuous(c.max_milli, 500, 100, &c), c.max_milli);
        assert_eq!(ease_continuous(c.knee_milli, 0, 100, &c), c.knee_milli);
        assert_eq!(ease_continuous(c.knee_milli, 50, 0, &c), c.knee_milli);
        let degenerate = EnergyCurve {
            min_milli: -1_000,
            knee_milli: -1_000,
            max_milli: -1_000,
        };
        assert_eq!(ease_continuous(-50_000, 50, 100, &degenerate), -50_000);
    }
}
