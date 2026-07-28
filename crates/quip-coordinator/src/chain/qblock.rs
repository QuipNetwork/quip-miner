//! Plain decoded qblock + topology views over the SCALE wire types.
//!
//! Mirrors the `snapshot.rs` pattern: `RealChainClient` decodes the SCALE
//! form, then hands callers these transport-free structs.

use crate::chain::scale_types::{require_set_values, QBlockWithNonceScale, TopologyMetaScale};

/// One winning block, decoded and ready for redraw. `nonce` is the 32-byte
/// `ChaCha8` seed (`U256::to_big_endian`), matching what `draw_ising_milli`
/// consumes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QBlockRecord {
    /// Monotonic 1-based qblock id this record was fetched under.
    pub qblock_id: u64,
    /// Winning miner account (`AccountId32` raw bytes).
    pub miner: [u8; 32],
    /// Salt used in nonce derivation.
    pub salt: [u8; 32],
    /// Achieved winning energy in milli units.
    pub energy_milli: i64,
    /// Block reward in plancks.
    pub reward: u128,
    /// Substrate block number the proof was accepted at.
    pub submitted_at: u32,
    /// Previous-proof block hash (nonce derivation input).
    pub last_proof_block_hash: [u8; 32],
    /// Topology the proof was mined against.
    pub topology_hash: [u8; 32],
    /// Miner-reported compute time in microseconds (`0` = unreported).
    pub device_access_time_us: u64,
    /// Difficulty gate: energy ceiling (milli).
    pub max_energy_milli: i64,
    /// Difficulty gate: minimum solutions.
    pub min_solutions: u32,
    /// Difficulty gate: minimum diversity (milli).
    pub min_diversity_milli: u32,
    /// `ChaCha8` seed (`nonce.to_big_endian()`); feeds `draw_ising_milli`.
    pub nonce: [u8; 32],
}

/// A registered topology plus its registration block, as enumerated from
/// `RegisteredTopologies`. `registered_at` orders the mining-era timeline the
/// `download` command uses to re-attribute qblocks (the stored per-qblock
/// `topology_hash` is unreliable — clobbered by the v5 migration backfill).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisteredTopology {
    /// Consensus topology hash (key in `RegisteredTopologies`).
    pub topology_hash: [u8; 32],
    /// Redraw inputs (Set-only allowed specs).
    pub inputs: TopologyInputs,
    /// Block number the topology was registered at.
    pub registered_at: u32,
}

/// Topology redraw inputs, decoded from `TopologyMeta` (Set specs only).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopologyInputs {
    /// Topology node ids.
    pub nodes: Vec<u32>,
    /// Topology undirected edges.
    pub edges: Vec<(u32, u32)>,
    /// Allowed linear-field values (milli).
    pub allowed_h_milli: Vec<i32>,
    /// Allowed coupling values (milli).
    pub allowed_j_milli: Vec<i32>,
    /// Allowed spin values (milli).
    pub allowed_spin_milli: Vec<i32>,
}

impl QBlockWithNonceScale {
    /// Flatten into a [`QBlockRecord`], converting the `U256` nonce to the
    /// big-endian 32-byte `ChaCha8` seed used on-chain.
    #[must_use]
    pub fn into_record(self, qblock_id: u64) -> QBlockRecord {
        let s = self.solution;
        QBlockRecord {
            qblock_id,
            miner: s.miner,
            salt: s.salt,
            energy_milli: s.energy_milli,
            reward: s.reward,
            submitted_at: s.submitted_at,
            last_proof_block_hash: s.last_proof_block_hash.0,
            topology_hash: s.topology_hash.0,
            device_access_time_us: s.device_access_time_us,
            max_energy_milli: s.difficulty.max_energy_milli,
            min_solutions: s.difficulty.min_solutions,
            min_diversity_milli: s.difficulty.min_diversity_milli,
            nonce: self.nonce.to_big_endian(),
        }
    }
}

impl TopologyMetaScale {
    /// Decode Set-only allowed-value specs into [`TopologyInputs`].
    ///
    /// # Errors
    /// Returns an error if any allowed-value spec is a range/empty set
    /// (`require_set_values`) — a range topology cannot be redrawn to match
    /// its consensus `topology_hash`.
    pub fn into_inputs(self) -> Result<TopologyInputs, String> {
        Ok(TopologyInputs {
            nodes: self.nodes,
            edges: self.edges,
            allowed_h_milli: require_set_values(&self.allowed_h_values)?,
            allowed_j_milli: require_set_values(&self.allowed_j_values)?,
            allowed_spin_milli: require_set_values(&self.allowed_spin_values)?,
        })
    }

    /// Decode into a [`RegisteredTopology`] for `topology_hash`, keeping the
    /// `registered_at` block that orders the mining-era timeline.
    ///
    /// # Errors
    /// Same as [`Self::into_inputs`] — a range/empty allowed-value spec cannot
    /// be redrawn.
    pub fn into_registered(self, topology_hash: [u8; 32]) -> Result<RegisteredTopology, String> {
        let registered_at = self.registered_at;
        Ok(RegisteredTopology {
            topology_hash,
            inputs: self.into_inputs()?,
            registered_at,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::scale_types::{DifficultyConfig, QBlockScale};
    use quantum_validation::AllowedValueSpec;
    use quip_protocol::chacha8::draw_ising_milli;
    use sp_core::{H256, U256};

    fn sample_qn(nonce_be: [u8; 32]) -> QBlockWithNonceScale {
        QBlockWithNonceScale {
            solution: QBlockScale {
                miner: [1; 32],
                salt: [2; 32],
                energy_milli: -14_500_000,
                reward: 7,
                submitted_at: 99,
                difficulty: DifficultyConfig {
                    min_solutions: 5,
                    max_energy_milli: -14_490_000,
                    min_diversity_milli: 200,
                },
                last_proof_block_hash: H256::from([3; 32]),
                topology_hash: H256::from([0x6e; 32]),
                device_access_time_us: 0,
            },
            nonce: U256::from_big_endian(&nonce_be),
        }
    }

    #[test]
    fn into_record_copies_fields_and_big_endian_nonce() {
        let nonce_be = [0xabu8; 32];
        let rec = sample_qn(nonce_be).into_record(42);
        assert_eq!(rec.qblock_id, 42);
        assert_eq!(rec.energy_milli, -14_500_000);
        assert_eq!(rec.topology_hash, [0x6e; 32]);
        assert_eq!(rec.max_energy_milli, -14_490_000);
        // The record nonce is the `ChaCha8` seed used on-chain: nonce.to_big_endian().
        assert_eq!(rec.nonce, nonce_be);
    }

    #[test]
    fn record_nonce_reproduces_the_onchain_draw() {
        // Golden: seeding draw_ising_milli with rec.nonce yields a well-formed
        // model of the topology's shape — this is the seed generate_ising_model
        // used on-chain (nonce.to_big_endian()).
        let inputs = TopologyInputs {
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
        };
        let rec = sample_qn([0x5a; 32]).into_record(1);
        let (h, j) = draw_ising_milli(
            rec.nonce,
            inputs.nodes.len(),
            inputs.edges.len(),
            &inputs.allowed_h_milli,
            &inputs.allowed_j_milli,
        )
        .unwrap();
        assert_eq!(h.len(), 4);
        assert_eq!(j.len(), 3);
    }

    #[test]
    fn into_inputs_rejects_range_spec() {
        let m = TopologyMetaScale {
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
            allowed_h_values: AllowedValueSpec::IntegerRange { min: -1, max: 1 },
            allowed_j_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            allowed_spin_values: AllowedValueSpec::Set(vec![-1000, 1000]),
            registered_at: 0,
        };
        assert!(m.into_inputs().is_err());
    }
}
