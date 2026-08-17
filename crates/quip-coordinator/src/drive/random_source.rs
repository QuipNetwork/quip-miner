//! Deterministic golden-draw problems over a topology spec.

use crate::chain::MiningSnapshot;
use crate::drive::{JobSource, TopologySpec};
use crate::producer::derive_pow_job;
use quip_proto::v1::Job;

/// Draws `count` problems over a topology via the golden `draw_ising_milli`
/// path (the same code the network uses), keyed by `seed`: the same
/// `(seed, topology)` pair always yields the same sequence of jobs.
pub struct RandomSource {
    snapshot: MiningSnapshot,
    miner_account: [u8; 32],
    seed: u64,
    deadline_ms: u64,
    count: u32,
    next_index: u32,
}

impl RandomSource {
    /// Build a source that yields `count` deterministic jobs for `(spec, seed)`.
    #[must_use]
    pub fn new(
        spec: &TopologySpec,
        miner_account: [u8; 32],
        seed: u64,
        count: u32,
        deadline_ms: u64,
    ) -> Self {
        Self {
            snapshot: spec.to_snapshot(),
            miner_account,
            seed,
            deadline_ms,
            count,
            next_index: 0,
        }
    }

    /// Per-index salt: `blake3(seed_le || index_le)`, so a given
    /// `(seed, index)` always derives the same nonce/problem.
    fn salt_for(&self, index: u32) -> [u8; 32] {
        let mut h = blake3::Hasher::new();
        let _ = h.update(&self.seed.to_le_bytes());
        let _ = h.update(&index.to_le_bytes());
        *h.finalize().as_bytes()
    }
}

impl JobSource for RandomSource {
    fn next_job(&mut self) -> Option<Job> {
        if self.next_index >= self.count {
            return None;
        }
        let salt = self.salt_for(self.next_index);
        let generation = u64::from(self.next_index) + 1;
        let job = match derive_pow_job(
            &self.snapshot,
            self.miner_account,
            salt,
            generation,
            self.deadline_ms,
        ) {
            Ok(job) => job,
            Err(e) => {
                tracing::error!(error = %e, "random source: cannot draw job from snapshot");
                return None;
            }
        };
        self.next_index += 1;
        Some(job)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::{drain_all, parse_topology_spec};
    use quip_protocol::wire::decode_i32_le;

    const SPEC: &str = r#"{
        "nodes": [0, 1, 2, 3],
        "edges": [[0, 1], [1, 2], [2, 3], [0, 3]],
        "allowed_h_milli": [-1000, 0, 1000],
        "allowed_j_milli": [-1000, 1000]
    }"#;

    #[test]
    fn draws_exactly_count_jobs() {
        let spec = parse_topology_spec(SPEC).unwrap();
        let mut src = RandomSource::new(&spec, [1u8; 32], 42, 5, 9_999_999);
        assert_eq!(drain_all(&mut src).len(), 5);
    }

    #[test]
    fn same_seed_and_topology_yields_identical_jobs() {
        let spec = parse_topology_spec(SPEC).unwrap();
        let mut a = RandomSource::new(&spec, [1u8; 32], 7, 3, 9_999_999);
        let mut b = RandomSource::new(&spec, [1u8; 32], 7, 3, 9_999_999);
        let jobs_a = drain_all(&mut a);
        let jobs_b = drain_all(&mut b);
        assert_eq!(jobs_a.len(), jobs_b.len());
        for (ja, jb) in jobs_a.iter().zip(&jobs_b) {
            assert_eq!(ja.job_id, jb.job_id);
            assert_eq!(
                ja.ising.as_ref().unwrap().h_milli_le32,
                jb.ising.as_ref().unwrap().h_milli_le32
            );
        }
    }

    #[test]
    fn different_seed_yields_different_jobs() {
        let spec = parse_topology_spec(SPEC).unwrap();
        let mut a = RandomSource::new(&spec, [1u8; 32], 1, 3, 9_999_999);
        let mut b = RandomSource::new(&spec, [1u8; 32], 2, 3, 9_999_999);
        let jobs_a = drain_all(&mut a);
        let jobs_b = drain_all(&mut b);
        let id_a = jobs_a.first().map(|j| &j.job_id);
        let id_b = jobs_b.first().map(|j| &j.job_id);
        assert_ne!(id_a, id_b);
    }

    #[test]
    fn drawn_values_stay_within_allowed_sets() {
        let spec = parse_topology_spec(SPEC).unwrap();
        let mut src = RandomSource::new(&spec, [1u8; 32], 99, 4, 9_999_999);
        for job in drain_all(&mut src) {
            let ising = job.ising.unwrap();
            let h = decode_i32_le(&ising.h_milli_le32).unwrap();
            let j = decode_i32_le(&ising.j_milli_le32).unwrap();
            assert!(h.iter().all(|v| [-1000, 0, 1000].contains(v)));
            assert!(j.iter().all(|v| [-1000, 1000].contains(v)));
        }
    }
}
