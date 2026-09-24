//! Salt leases over a topology spec for offline lease benchmarks.

use crate::chain::MiningSnapshot;
use crate::drive::{JobSource, TopologySpec};
use quip_proto::v1::Job;

/// Yields leases that together name counters `0..salts`, each at most
/// `lease_size` long. `seed` picks the round (`last_proof_block_hash`), so
/// equal `(seed, topology, salts, lease_size)` inputs name equal problems.
pub struct LeaseSource {
    snapshot: MiningSnapshot,
    miner_account: [u8; 32],
    salts: u64,
    lease_size: u64,
    next_start: u64,
}

impl LeaseSource {
    /// Build the source. A `lease_size` of zero means one salt per lease.
    #[must_use]
    pub fn new(
        spec: &TopologySpec,
        miner_account: [u8; 32],
        seed: u64,
        salts: u64,
        lease_size: u64,
    ) -> Self {
        let mut snapshot = spec.to_snapshot();
        let mut h = blake3::Hasher::new();
        let _ = h.update(b"quip-drive-lease");
        let _ = h.update(&seed.to_le_bytes());
        snapshot.last_proof_block_hash = *h.finalize().as_bytes();
        Self {
            snapshot,
            miner_account,
            salts,
            lease_size: lease_size.max(1),
            next_start: 0,
        }
    }
}

impl JobSource for LeaseSource {
    fn next_job(&mut self) -> Option<Job> {
        if self.next_start >= self.salts {
            return None;
        }
        let count = self.lease_size.min(self.salts - self.next_start);
        let job = crate::lease::build_lease_job(
            &self.snapshot,
            self.miner_account,
            self.next_start,
            count,
            1,
        );
        self.next_start += count;
        Some(job)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drive::{drain_all, parse_topology_spec};

    fn spec() -> TopologySpec {
        parse_topology_spec(crate::presets::preset_spec("smoke").unwrap()).unwrap()
    }

    #[test]
    fn splits_salts_into_contiguous_leases() {
        let jobs = drain_all(&mut LeaseSource::new(&spec(), [0; 32], 1, 10, 4));
        let ranges: Vec<(u64, u64)> = jobs
            .iter()
            .map(|j| {
                j.generator
                    .as_ref()
                    .map(|g| (g.salt_start, g.salt_count))
                    .unwrap()
            })
            .collect();
        assert_eq!(ranges, vec![(0, 4), (4, 4), (8, 2)]);
    }

    #[test]
    #[expect(
        clippy::indexing_slicing,
        reason = "the source yields one job for the positive salt count in this test"
    )]
    fn same_seed_draws_the_same_round() {
        let a = drain_all(&mut LeaseSource::new(&spec(), [0; 32], 5, 4, 4));
        let b = drain_all(&mut LeaseSource::new(&spec(), [0; 32], 5, 4, 4));
        let c = drain_all(&mut LeaseSource::new(&spec(), [0; 32], 6, 4, 4));
        assert_eq!(a, b);
        assert_ne!(a[0].generator, c[0].generator);
    }

    #[test]
    fn zero_lease_size_is_treated_as_one() {
        assert_eq!(
            drain_all(&mut LeaseSource::new(&spec(), [0; 32], 1, 3, 0)).len(),
            3
        );
    }
}
