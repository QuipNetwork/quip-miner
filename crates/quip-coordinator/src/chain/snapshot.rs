//! Mining snapshot types and head-change generation key.

use crate::topology::{topology_hash_sets, DEFAULT_SPIN_SET};

/// Chain mining snapshot consumed by the `PoW` producer.
#[derive(Debug, Clone)]
pub struct MiningSnapshot {
    /// Hash of the block the snapshot was taken at. The coordinator's own view
    /// of the chain head, distinct from [`Self::last_proof_block_hash`], which
    /// changes only when a proof wins.
    pub head_hash: [u8; 32],
    /// Hash of the block that contained the last winning proof.
    pub last_proof_block_hash: [u8; 32],
    /// Topology identity bytes (32-byte hash when provided by the chain).
    pub topology_hash: Vec<u8>,
    /// Topology node ids.
    pub nodes: Vec<u32>,
    /// Topology undirected edges as `(u, v)` node-id pairs.
    pub edges: Vec<(u32, u32)>,
    /// Allowed linear-field values in milli units.
    pub allowed_h_milli: Vec<i32>,
    /// Allowed coupling values in milli units.
    pub allowed_j_milli: Vec<i32>,
    /// Allowed spin milli values (Set); default binary `±1000` when empty.
    pub allowed_spin_milli: Vec<i32>,
    /// Minimum number of valid solutions required.
    pub min_solutions: u32,
    /// Energy ceiling: solutions must be strictly below this (milli).
    pub max_energy_milli: i64,
    /// Minimum diversity of the solution set (milli).
    pub min_diversity_milli: u32,
    /// Chain block number the snapshot was taken at.
    pub block_number: u64,
}

/// Difficulty-decay parameters read alongside the snapshot (independent
/// storage/constant reads, pinned at the snapshot's block) so the coordinator
/// can project the decay schedule the chain's already-decayed
/// `max_energy_milli` hides. The current block comes from
/// [`MiningSnapshot::block_number`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecayParams {
    /// Base (un-decayed) `max_energy_milli` from `Difficulties[topology_hash]`.
    pub base_max_energy_milli: i64,
    /// Block number of the last winning proof (0 = genesis, no decay yet).
    pub last_proof_block: u64,
    /// Blocks per decay epoch (`EpochLength`).
    pub epoch_length: u64,
    /// Resolved curve c-triple (per-mille: 700 == 0.70), after any per-topology
    /// `TopologyCurveC` override.
    pub c_easy_milli: u32,
    /// Knee c of the difficulty curve (per-mille).
    pub c_knee_milli: u32,
    /// Hard c of the difficulty curve (per-mille).
    pub c_hard_milli: u32,
}

impl DecayParams {
    /// Genesis / no-proof defaults: base difficulty is active (no decay).
    #[must_use]
    pub fn genesis() -> Self {
        Self {
            base_max_energy_milli: crate::decay::DEFAULT_BASE_MAX_ENERGY_MILLI,
            last_proof_block: 0,
            epoch_length: crate::decay::EPOCH_LENGTH_BLOCKS,
            c_easy_milli: crate::decay::DEFAULT_C_EASY_MILLI,
            c_knee_milli: crate::decay::DEFAULT_C_KNEE_MILLI,
            c_hard_milli: crate::decay::DEFAULT_C_HARD_MILLI,
        }
    }
}

/// Change-detection key over the round-identifying snapshot fields.
///
/// Mirrors `substrate/event_manager.py:_state_key`: a new key means a new
/// `PoW` generation. Pure `blake3` over identifying fields (groundable).
#[must_use]
pub fn head_state_key(snap: &MiningSnapshot) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    let _ = h.update(&snap.last_proof_block_hash);
    let _ = h.update(&snap.topology_hash);
    let _ = h.update(&snap.min_solutions.to_le_bytes());
    let _ = h.update(&snap.max_energy_milli.to_le_bytes());
    let _ = h.update(&snap.min_diversity_milli.to_le_bytes());
    let _ = h.update(&snap.block_number.to_le_bytes());
    *h.finalize().as_bytes()
}

/// Build topology hash for a snapshot's graph when not provided by the chain.
#[must_use]
pub fn snapshot_topology_hash(snap: &MiningSnapshot) -> Vec<u8> {
    if snap.topology_hash.len() == 32 {
        snap.topology_hash.clone()
    } else {
        let spin = if snap.allowed_spin_milli.is_empty() {
            DEFAULT_SPIN_SET.as_slice()
        } else {
            snap.allowed_spin_milli.as_slice()
        };
        topology_hash_sets(
            &snap.nodes,
            &snap.edges,
            &snap.allowed_h_milli,
            &snap.allowed_j_milli,
            spin,
        )
        .to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> MiningSnapshot {
        MiningSnapshot {
            head_hash: [0u8; 32],
            last_proof_block_hash: [7u8; 32],
            topology_hash: vec![9u8; 32],
            nodes: vec![0, 1],
            edges: vec![(0, 1)],
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
    fn head_key_stable_for_same_snapshot() {
        let a = head_state_key(&sample());
        let b = head_state_key(&sample());
        assert_eq!(a, b);
    }

    #[test]
    fn head_key_changes_when_last_proof_changes() {
        let mut s = sample();
        let k1 = head_state_key(&s);
        s.last_proof_block_hash = [8u8; 32];
        let k2 = head_state_key(&s);
        assert_ne!(k1, k2);
    }

    #[test]
    fn decay_params_genesis_disables_decay() {
        let g = DecayParams::genesis();
        assert_eq!(g.last_proof_block, 0); // no proof yet → no decay
        assert_eq!(g.epoch_length, crate::decay::EPOCH_LENGTH_BLOCKS);
        assert_eq!(
            g.base_max_energy_milli,
            crate::decay::DEFAULT_BASE_MAX_ENERGY_MILLI
        );
        assert_eq!(g.c_easy_milli, 700);
        assert_eq!(g.c_hard_milli, 750);
    }

    #[test]
    fn quantum_pow_storage_keys_have_expected_shape() {
        use crate::chain::extrinsic::{
            difficulties_storage_key, last_proof_block_storage_key, topology_curve_c_storage_key,
        };
        let hash = [9u8; 32];
        let diff = difficulties_storage_key(&hash);
        // twox128(pallet)=16 + twox128(item)=16 + blake2_128(hash)=16 + hash=32
        assert_eq!(diff.len(), 80);
        assert!(diff.ends_with(&hash));
        let curve = topology_curve_c_storage_key(&hash);
        assert_eq!(curve.len(), 80);
        // Same pallet prefix, different storage item.
        #[expect(
            clippy::indexing_slicing,
            reason = "storage keys are fixed 80/32 bytes; prefixes checked by len asserts"
        )]
        {
            assert_eq!(diff[..16], curve[..16]);
            assert_ne!(diff[16..32], curve[16..32]);
            // Plain StorageValue: pallet + item, no key suffix.
            assert_eq!(last_proof_block_storage_key().len(), 32);
            assert_eq!(last_proof_block_storage_key()[..16], diff[..16]);
        }
    }

    #[test]
    fn curve_c_scale_roundtrips() {
        use crate::chain::scale_types::CurveCScale;
        use parity_scale_codec::{Decode, Encode};
        let c = CurveCScale {
            easy_milli: 700,
            knee_milli: 725,
            hard_milli: 750,
        };
        let bytes = c.encode();
        assert_eq!(CurveCScale::decode(&mut &bytes[..]).unwrap(), c);
    }
}
