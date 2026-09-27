//! Salt leases: `ISING_GENERATE` jobs that let a miner draw its own
//! proof-of-work problems, and how many salts each lease names.

use crate::chain::snapshot::MiningSnapshot;
use quip_proto::v1::{GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind, Provenance};

/// Seconds of sampling one lease should hold. A lease completes only when its
/// slowest salt does, and a screening miner can hold one salt for tens of
/// seconds at full budget, so a lease must be long against that tail or the
/// miner idles while it waits for `LeaseDone` to refund the credit.
pub const LEASE_TARGET_SECS: f64 = 60.0;

/// Time constant, in seconds, of the window [`SaltRate`] averages over. It
/// spans several leases at [`LEASE_TARGET_SECS`], so one completion moves the
/// estimate by a fraction of a lease rather than a whole one.
pub const LEASE_RATE_WINDOW_SECS: f64 = 300.0;

/// Upper bound on salts in one lease. Salts are counter values, so this bounds
/// how far one lease can run past a rate estimate, not memory.
pub const LEASE_MAX_SALTS: u64 = 1 << 20;

/// Leases kept staged per miner. Credits bound what is dispatched, so two
/// staged leases keep a replacement ready when one completes.
pub const LEASE_STAGE_DEPTH: usize = 2;

/// Prefix of every lease `job_id`. A proof-of-work plain job id is a 32-byte
/// nonce, so the two id spaces cannot collide.
pub const LEASE_JOB_ID_PREFIX: &[u8] = b"lease:";

/// Pipeline fills in a first lease, before any rate is known.
const INITIAL_FILLS: u64 = 4;

/// Unique `job_id` for the lease whose first counter is `salt_start`. The feeder's
/// counter never repeats within a process, so neither does the id.
#[must_use]
pub fn lease_job_id(salt_start: u64) -> Vec<u8> {
    let mut id = LEASE_JOB_ID_PREFIX.to_vec();
    id.extend_from_slice(&salt_start.to_le_bytes());
    id
}

/// Whether `job` is a salt lease.
#[must_use]
pub fn is_lease(job: &Job) -> bool {
    job.kind == JobKind::IsingGenerate as i32
}

/// An `ISING_GENERATE` job naming counters `salt_start..salt_start + salt_count`
/// over `snap`'s topology and round. `base_salt` is all zeros, so salt `i`
/// equals the counter salt of `salt_start + i`, the salt a plain proof-of-work
/// job would use.
#[must_use]
pub fn build_lease_job(
    snap: &MiningSnapshot,
    miner_identity: [u8; 32],
    salt_start: u64,
    salt_count: u64,
    generation: u64,
) -> Job {
    Job {
        job_id: lease_job_id(salt_start),
        kind: JobKind::IsingGenerate as i32,
        generation,
        deadline_ms: 0,
        ising: None,
        provenance: Some(Provenance {
            is_pow: true,
            order_id: vec![],
        }),
        generator: Some(IsingProblemGenerator {
            algorithm: GeneratorAlgorithm::Blake3Chacha8V1 as i32,
            topology_hash: snap.topology_hash.clone(),
            last_proof_block_hash: snap.last_proof_block_hash.to_vec(),
            miner_account: miner_identity.to_vec(),
            base_salt: vec![0u8; 32],
            salt_start,
            salt_count,
        }),
    }
}

/// Salts for one lease from a miner's measured rate.
///
/// Covers [`LEASE_TARGET_SECS`] at `salts_per_sec`, but never fewer than one
/// pipeline fill (`stream_width`, at least 1), so the miner's stream stays
/// full, and never more than [`LEASE_MAX_SALTS`]. With no usable rate yet, a
/// first lease holds [`INITIAL_FILLS`] pipeline fills.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "rate * seconds is finite, non-negative, and clamped to LEASE_MAX_SALTS"
)]
pub fn lease_salt_count(salts_per_sec: f64, stream_width: u32) -> u64 {
    let fill = u64::from(stream_width.max(1));
    if !salts_per_sec.is_finite() || salts_per_sec <= 0.0 {
        return (fill * INITIAL_FILLS).min(LEASE_MAX_SALTS);
    }
    let want = (salts_per_sec * LEASE_TARGET_SECS).ceil();
    let want = if want >= LEASE_MAX_SALTS as f64 {
        LEASE_MAX_SALTS
    } else {
        want as u64
    };
    want.clamp(fill.min(LEASE_MAX_SALTS), LEASE_MAX_SALTS)
}

/// Accepts every authentic result: each energy clears it and any solution
/// count passes, so only the redraw and rescore in `verify_lease_result` fail.
const AUTHENTICITY_ONLY: quip_protocol::target::Target = quip_protocol::target::Target {
    max_energy_milli: i64::MAX,
    min_solutions: 0,
    min_diversity_milli: 0,
    max_proof_solutions: u32::MAX,
};

/// Verify one lease `Result` in two steps, then select against `target`.
///
/// A solver-core 0.0.2 miner sends every read for a reported salt, unfiltered
/// and uncapped. The first step redraws the problem and rescores every read
/// with a permissive target, so it rejects only a forged or corrupt result. An
/// authentic result counts as participation even if it misses the target. The
/// second step applies the live gates through the same selection as a plain
/// job, so a miss reaches the decay stash instead of being dropped.
///
/// # Errors
/// Returns the verification error when the result is not authentic.
pub fn verify_and_select(
    generator: &IsingProblemGenerator,
    topology: &quip_protocol::lease::TopologyView,
    target: &quip_proto::v1::SetTarget,
    result: &quip_proto::v1::Result,
) -> Result<(quip_protocol::lease::Verified, crate::validate::Validated), String> {
    let verified =
        quip_protocol::lease::verify_lease_result(generator, topology, &AUTHENTICITY_ONLY, result)
            .map_err(|e| e.to_string())?;
    // Verification decoded and rescored every read, so neither step fails here.
    let (rows, energies): (Vec<Vec<i8>>, Vec<i64>) = result
        .solutions
        .iter()
        .filter_map(|s| {
            quip_protocol::wire::decode_spins_packed(&s.spins, topology.num_nodes)
                .ok()
                .map(|spins| (spins, s.energy_milli))
        })
        .unzip();
    let gates = crate::validate::gates_from_target(Some(target));
    let validated = crate::validate::validate_scored(&rows, &energies, &gates);
    Ok((verified, validated))
}

/// A miner's lease throughput in salts per second.
///
/// Salts are counted only at `LeaseDone`, so a single poll sees either no
/// salts or a whole lease. The estimate divides salts by elapsed time, both
/// decayed over [`LEASE_RATE_WINDOW_SECS`], which turns those steps into a
/// rate instead of a spike followed by decay toward zero.
#[derive(Debug, Default, Clone, Copy)]
pub struct SaltRate {
    salts: f64,
    secs: f64,
}

impl SaltRate {
    /// Record `salts` finished over the last `elapsed_secs` and return the
    /// updated rate, zero until any time has passed.
    #[expect(
        clippy::cast_precision_loss,
        reason = "salt counts stay far below 2^52"
    )]
    pub fn observe(&mut self, salts: u64, elapsed_secs: f64) -> f64 {
        let keep = (-elapsed_secs / LEASE_RATE_WINDOW_SECS).exp();
        self.salts = self.salts * keep + salts as f64;
        self.secs = self.secs * keep + elapsed_secs;
        if self.secs > 0.0 {
            self.salts / self.secs
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::snapshot::MiningSnapshot;
    use quip_proto::v1::{GeneratorAlgorithm, JobKind};
    use quip_protocol::lease::LeaseSpec;

    fn snap() -> MiningSnapshot {
        MiningSnapshot {
            head_hash: [0u8; 32],
            last_proof_block_hash: [7u8; 32],
            topology_hash: vec![9u8; 32],
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 5,
            max_energy_milli: -14_000_000,
            min_diversity_milli: 200,
            block_number: 42,
            spec_version: 117,
        }
    }

    #[test]
    fn lease_job_names_a_valid_contiguous_range() {
        let job = build_lease_job(&snap(), [1u8; 32], 11, 5, 3);
        assert_eq!(job.kind, JobKind::IsingGenerate as i32);
        assert_eq!(job.generation, 3);
        assert!(job.ising.is_none());
        assert!(job.provenance.as_ref().unwrap().is_pow);
        let g = job.generator.as_ref().unwrap();
        assert_eq!(g.algorithm, GeneratorAlgorithm::Blake3Chacha8V1 as i32);
        assert_eq!(g.topology_hash, vec![9u8; 32]);
        let spec = LeaseSpec::from_proto(g).unwrap();
        assert_eq!((spec.salt_start, spec.salt_count), (11, 5));
        assert_eq!(spec.miner_account, [1u8; 32]);
        assert_eq!(spec.last_proof_block_hash, [7u8; 32]);
    }

    #[test]
    fn lease_salts_equal_the_counter_salts() {
        let job = build_lease_job(&snap(), [1u8; 32], 11, 5, 3);
        let spec = LeaseSpec::from_proto(job.generator.as_ref().unwrap()).unwrap();
        let mut expected = [0u8; 32];
        expected[..8].copy_from_slice(&13u64.to_le_bytes());
        assert_eq!(spec.salt(2), Some(expected));
    }

    #[test]
    fn lease_job_ids_are_distinct_from_32_byte_nonces() {
        let id = lease_job_id(11);
        assert!(id.starts_with(LEASE_JOB_ID_PREFIX));
        assert_ne!(id.len(), 32);
        assert_ne!(lease_job_id(11), lease_job_id(12));
        assert!(is_lease(&build_lease_job(&snap(), [0; 32], 11, 1, 1)));
    }

    #[test]
    fn salt_count_starts_from_stream_width_without_a_rate() {
        assert_eq!(lease_salt_count(0.0, 8), 32);
        assert_eq!(lease_salt_count(0.0, 0), 4);
    }

    #[test]
    fn salt_rate_averages_lease_completions_over_quiet_polls() {
        // A miner finishing one 60,000-salt lease every 10 one-second polls
        // runs at 6,000 salts per second. Per-poll smoothing swings between
        // the whole lease and near zero. The window holds the true rate.
        let mut rate = SaltRate::default();
        let mut seen = Vec::new();
        for poll in 1..=600 {
            let salts = if poll % 10 == 0 { 60_000 } else { 0 };
            seen.push(rate.observe(salts, 1.0));
        }
        for r in seen.iter().skip(300) {
            assert!(
                (5_000.0..=7_000.0).contains(r),
                "rate {r} strays from 6,000"
            );
        }
    }

    #[test]
    fn salt_rate_is_zero_before_any_time_passes() {
        assert!(SaltRate::default().observe(0, 0.0).abs() < f64::EPSILON);
        assert!(SaltRate::default().observe(5, 1.0) > 0.0);
    }

    #[test]
    fn salt_count_covers_the_target_duration_within_bounds() {
        assert_eq!(lease_salt_count(10.0, 1), 600);
        assert_eq!(lease_salt_count(0.1, 8), 8, "never below one pipeline fill");
        assert_eq!(lease_salt_count(1e12, 1), LEASE_MAX_SALTS);
        assert_eq!(
            lease_salt_count(f64::NAN, 2),
            8,
            "a bad rate falls back to the initial size"
        );
    }
}
