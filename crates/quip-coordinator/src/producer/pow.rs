//! Derive `PoW` [`Job`]s from mining snapshots via golden-pinned `ChaCha8` + `derive_nonce`.

use crate::chain::snapshot::MiningSnapshot;
use crate::producer::problem::milli_problem;
use quip_proto::v1::{ising_problem, Job, JobKind, Provenance};
use quip_protocol::chacha8::{draw_ising_milli, DrawError};
use quip_protocol::derive::derive_nonce;

/// Build an `ISING_SAMPLE` `PoW` job from a snapshot, miner account, and salt.
///
/// Uses [`derive_nonce`] + [`draw_ising_milli`] (golden-pinned). Graph is a
/// `topology_hash` reference; gates come from the snapshot difficulty.
///
/// # Errors
/// Propagates [`DrawError`] if the snapshot's allowed-value sets cannot produce
/// a model (empty set).
pub fn derive_pow_job(
    snap: &MiningSnapshot,
    miner_account: [u8; 32],
    salt: [u8; 32],
    generation: u64,
    deadline_ms: u64,
) -> Result<Job, DrawError> {
    let nonce = derive_nonce(snap.last_proof_block_hash, miner_account, salt);
    build_ising_job_from_nonce(snap, nonce, generation, deadline_ms)
}

/// Build an `ISING_SAMPLE` job from an already-known nonce (skips
/// `derive_nonce`). Used by drive-mode nonce-ref replay, where the nonce is
/// read from a file rather than derived from a live chain head.
///
/// Still golden-pinned: draws via `draw_ising_milli`, the same code the
/// network uses.
///
/// # Errors
/// Propagates [`DrawError`] if the snapshot's allowed-value sets are empty.
pub fn build_ising_job_from_nonce(
    snap: &MiningSnapshot,
    nonce: [u8; 32],
    generation: u64,
    deadline_ms: u64,
) -> Result<Job, DrawError> {
    let (h, j) = draw_ising_milli(
        nonce,
        snap.nodes.len(),
        snap.edges.len(),
        &snap.allowed_h_milli,
        &snap.allowed_j_milli,
    )?;
    Ok(Job {
        job_id: nonce.to_vec(),
        kind: JobKind::IsingSample as i32,
        generation,
        deadline_ms,
        ising: Some(milli_problem(
            Some(ising_problem::Graph::TopologyHash(
                snap.topology_hash.clone(),
            )),
            &h,
            &j,
        )),
        generator: None,
        provenance: Some(Provenance {
            is_pow: true,
            order_id: vec![],
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::producer::problem::problem_milli;
    use quip_proto::v1::{ising_problem, JobKind};

    fn fixture() -> MiningSnapshot {
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
    fn derives_job_with_correct_shape() {
        let snap = fixture();
        let job = derive_pow_job(&snap, [1u8; 32], [2u8; 32], 42, 9999).unwrap();
        assert_eq!(job.kind, JobKind::IsingSample as i32);
        assert_eq!(job.generation, 42);
        assert!(job.provenance.as_ref().unwrap().is_pow);
        let ising = job.ising.unwrap();
        assert!(matches!(
            ising.graph,
            Some(ising_problem::Graph::TopologyHash(_))
        ));
        let (h, j) = problem_milli(&ising).unwrap();
        assert_eq!(h.len(), 4);
        assert_eq!(j.len(), 4);
        assert!(h.iter().all(|v| [-1000, 0, 1000].contains(v)));
        assert!(j.iter().all(|v| [-1000, 1000].contains(v)));
    }

    #[test]
    fn derivation_is_deterministic_for_same_inputs() {
        let snap = fixture();
        let a = derive_pow_job(&snap, [1u8; 32], [2u8; 32], 1, 1).unwrap();
        let b = derive_pow_job(&snap, [1u8; 32], [2u8; 32], 1, 1).unwrap();
        assert_eq!(
            problem_milli(a.ising.as_ref().unwrap()).unwrap(),
            problem_milli(b.ising.as_ref().unwrap()).unwrap()
        );
    }
}
