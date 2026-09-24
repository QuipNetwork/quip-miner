//! Salt leases: `ISING_GENERATE` jobs that let a miner draw its own
//! proof-of-work problems, and how many salts each lease names.

use crate::chain::snapshot::MiningSnapshot;
use quip_proto::v1::{GeneratorAlgorithm, IsingProblemGenerator, Job, JobKind, Provenance};

/// Seconds of sampling one lease should hold. The upgrade guide asks for "a
/// few seconds": long enough that the lease round trip is small against the
/// work, short enough that sizing follows a changing rate.
pub const LEASE_TARGET_SECS: f64 = 4.0;

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
    fn salt_count_covers_the_target_duration_within_bounds() {
        assert_eq!(lease_salt_count(10.0, 1), 40);
        assert_eq!(lease_salt_count(0.1, 8), 8, "never below one pipeline fill");
        assert_eq!(lease_salt_count(1e12, 1), LEASE_MAX_SALTS);
        assert_eq!(
            lease_salt_count(f64::NAN, 2),
            8,
            "a bad rate falls back to the initial size"
        );
    }
}
