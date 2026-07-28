//! One `hardest_models` instance record + its redraw self-verification.

use crate::chain::qblock::{QBlockRecord, TopologyInputs};
use crate::topology::topology_hash_sets;
use quip_protocol::chacha8::{draw_ising_milli, DrawError};
use serde::{Deserialize, Serialize};

/// Difficulty gates carried on each instance (QUI-810 schema).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct DifficultyJson {
    /// Energy ceiling the winning proof cleared (milli).
    pub max_energy_milli: i64,
    /// Minimum solutions gate.
    pub min_solutions: u32,
    /// Minimum diversity gate (milli).
    pub min_diversity_milli: u32,
}

/// Provenance of a chain-sourced instance.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ProvenanceJson {
    /// Always `"chain"` for the `download` subcommand.
    pub source: String,
    /// Monotonic qblock id.
    pub qblock_id: u64,
    /// Substrate block number of the winning proof.
    pub submitted_at: u32,
}

/// One ranked `instances.jsonl` line. The `nonce` field name and un-prefixed
/// hex are load-bearing: the drive harness `--source list` keys on `nonce`
/// and rejects a `0x` prefix.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct InstanceRecord {
    /// `ChaCha8` seed as un-prefixed 64-char hex (drive `--source list` key).
    pub nonce: String,
    /// Topology bucket hash, un-prefixed hex.
    pub topology_hash: String,
    /// Achieved winning energy (milli); ranking key, most-negative first.
    pub energy_milli: i64,
    /// Salt used in nonce derivation, un-prefixed hex.
    pub salt_hex: String,
    /// Monotonic qblock id.
    pub qblock_id: u64,
    /// Winning miner account, un-prefixed hex.
    pub miner_hex: String,
    /// Substrate block number of the winning proof.
    pub submitted_at: u32,
    /// Miner-reported compute time (microseconds; `0` = unreported).
    pub device_access_time_us: u64,
    /// Difficulty gates the proof cleared.
    pub difficulty: DifficultyJson,
    /// Instance provenance.
    pub provenance: ProvenanceJson,
}

/// Self-verification failure when regenerating a qblock's problem.
#[derive(Debug)]
pub enum VerifyError {
    /// The nonce/allowed-set combination cannot produce a model.
    Draw(DrawError),
    /// Redrawn `(h, j)` lengths disagree with `(nodes, edges)`.
    LengthMismatch {
        /// Expected `h` length (`nodes.len()`).
        expected_h: usize,
        /// Actual redrawn `h` length.
        got_h: usize,
        /// Expected `j` length (`edges.len()`).
        expected_j: usize,
        /// Actual redrawn `j` length.
        got_j: usize,
    },
    /// Recomputed topology hash differs from the qblock's recorded hash —
    /// the fetched topology inputs are not the ones the proof was mined on.
    TopologyHashMismatch {
        /// The qblock's recorded `topology_hash`.
        expected: [u8; 32],
        /// The hash recomputed from the fetched topology inputs.
        got: [u8; 32],
    },
}

impl std::fmt::Display for VerifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Draw(e) => write!(f, "redraw failed: {e}"),
            Self::LengthMismatch {
                expected_h,
                got_h,
                expected_j,
                got_j,
            } => write!(
                f,
                "redraw shape mismatch: h {got_h}/{expected_h}, j {got_j}/{expected_j}"
            ),
            Self::TopologyHashMismatch { .. } => {
                write!(f, "recomputed topology_hash != qblock topology_hash")
            }
        }
    }
}
impl std::error::Error for VerifyError {}

/// Encode bytes as un-prefixed lowercase hex (never `0x`-prefixed).
#[must_use]
pub fn hex_plain(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Redraw a qblock's Ising problem from its nonce and self-verify it, then
/// build the ranked instance record.
///
/// Winning spins are not on-chain, so verification targets **problem
/// regeneration**: (1) the draw succeeds, (2) `(h, j)` lengths match
/// `(nodes, edges)`, (3) the recomputed `topology_hash` equals the qblock's
/// recorded hash (proving the fetched topology is the right bucket).
///
/// # Errors
/// [`VerifyError`] if any of the three checks fails.
pub fn build_instance_record(
    q: &QBlockRecord,
    topo: &TopologyInputs,
) -> Result<InstanceRecord, VerifyError> {
    let (h, j) = draw_ising_milli(
        q.nonce,
        topo.nodes.len(),
        topo.edges.len(),
        &topo.allowed_h_milli,
        &topo.allowed_j_milli,
    )
    .map_err(VerifyError::Draw)?;
    if h.len() != topo.nodes.len() || j.len() != topo.edges.len() {
        return Err(VerifyError::LengthMismatch {
            expected_h: topo.nodes.len(),
            got_h: h.len(),
            expected_j: topo.edges.len(),
            got_j: j.len(),
        });
    }
    let recomputed = topology_hash_sets(
        &topo.nodes,
        &topo.edges,
        &topo.allowed_h_milli,
        &topo.allowed_j_milli,
        &topo.allowed_spin_milli,
    );
    if recomputed != q.topology_hash {
        return Err(VerifyError::TopologyHashMismatch {
            expected: q.topology_hash,
            got: recomputed,
        });
    }
    Ok(InstanceRecord {
        nonce: hex_plain(&q.nonce),
        topology_hash: hex_plain(&q.topology_hash),
        energy_milli: q.energy_milli,
        salt_hex: hex_plain(&q.salt),
        qblock_id: q.qblock_id,
        miner_hex: hex_plain(&q.miner),
        submitted_at: q.submitted_at,
        device_access_time_us: q.device_access_time_us,
        difficulty: DifficultyJson {
            max_energy_milli: q.max_energy_milli,
            min_solutions: q.min_solutions,
            min_diversity_milli: q.min_diversity_milli,
        },
        provenance: ProvenanceJson {
            source: "chain".to_string(),
            qblock_id: q.qblock_id,
            submitted_at: q.submitted_at,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::qblock::{QBlockRecord, TopologyInputs};
    use crate::topology::topology_hash_sets;

    fn topo() -> TopologyInputs {
        TopologyInputs {
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
        }
    }
    fn qblock(topo_hash: [u8; 32]) -> QBlockRecord {
        QBlockRecord {
            qblock_id: 7,
            miner: [0xaa; 32],
            salt: [0xbb; 32],
            energy_milli: -14_510_000,
            reward: 1,
            submitted_at: 500,
            last_proof_block_hash: [0xcc; 32],
            topology_hash: topo_hash,
            device_access_time_us: 0,
            max_energy_milli: -14_490_000,
            min_solutions: 5,
            min_diversity_milli: 200,
            nonce: [0x5a; 32],
        }
    }

    #[test]
    fn well_formed_qblock_builds_record() {
        let t = topo();
        // The qblock's topology_hash must equal the hash of its own inputs.
        let h = topology_hash_sets(
            &t.nodes,
            &t.edges,
            &t.allowed_h_milli,
            &t.allowed_j_milli,
            &t.allowed_spin_milli,
        );
        let rec = build_instance_record(&qblock(h), &t).unwrap();
        assert_eq!(rec.nonce, "5a".repeat(32)); // un-prefixed, 64 chars
        assert_eq!(rec.topology_hash, hex_plain(&h));
        assert_eq!(rec.energy_milli, -14_510_000);
        assert_eq!(rec.salt_hex, "bb".repeat(32));
        assert_eq!(rec.difficulty.max_energy_milli, -14_490_000);
        assert_eq!(rec.provenance.source, "chain");
    }

    #[test]
    fn wrong_topology_hash_fails_self_verify() {
        // topology_hash that does NOT match the given inputs → rejected.
        let err = build_instance_record(&qblock([0x00; 32]), &topo()).unwrap_err();
        assert!(matches!(err, VerifyError::TopologyHashMismatch { .. }));
    }

    #[test]
    fn empty_allowed_set_propagates_draw_error() {
        let mut t = topo();
        t.allowed_j_milli.clear(); // edges present but no allowed J → DrawError
        let h = topology_hash_sets(
            &t.nodes,
            &t.edges,
            &t.allowed_h_milli,
            &t.allowed_j_milli,
            &t.allowed_spin_milli,
        );
        assert!(matches!(
            build_instance_record(&qblock(h), &t),
            Err(VerifyError::Draw(_))
        ));
    }

    #[test]
    fn hex_plain_has_no_0x_prefix() {
        assert_eq!(hex_plain(&[0x0f, 0xa0]), "0fa0");
    }
}
