//! Decide whether a proof pending in the pool ends the current round.
//!
//! The judge replays the pallet's `submit_proof` checks
//! (`quip-validator/pallets/quantum-pow/src/lib.rs`, `validate_proof`) with
//! the same `quantum_validation` functions, against the snapshot the feeder is
//! mining. Difficulty eases every block until a win, so a proof that clears
//! now also clears at its inclusion block.

use crate::chain::extrinsic::account_identity_bytes;
use crate::chain::scale_types::QuantumProof;
use crate::chain::{MiningSnapshot, PendingProof};
use quantum_validation::{
    calculate_diversity, derive_nonce, energy_of_solution_indexed, generate_ising_model_indexed,
    select_diverse, unpack_solution, AllowedValueSpec, TopologyIndex,
};

/// Why a pending proof does or does not end the round.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The pallet will mint a qblock from this proof at its inclusion block.
    Clears {
        /// Lowest energy among the rows under the ceiling, in milli-units.
        best_energy_milli: i64,
    },
    /// The nonce derives from another root, or the topology differs.
    OtherRound,
    /// Fails the energy, count, or diversity gate at the current difficulty.
    BelowGates,
    /// Bytes the pallet rejects before scoring: bad packing or a zero spin.
    Malformed(String),
}

impl Verdict {
    /// Whether the proof ends the round.
    #[must_use]
    pub const fn clears(&self) -> bool {
        matches!(self, Self::Clears { .. })
    }
}

/// Replay the pallet's `submit_proof` checks against `snap`.
///
/// Registration of the signer is not checked. The pool already ran the signed
/// extensions, and a proof from an unregistered account is rare enough that
/// the feeder's resume path covers it.
#[must_use]
pub fn judge_pending_proof(pending: &PendingProof, snap: &MiningSnapshot) -> Verdict {
    if pending.proof.topology_hash.as_bytes() != snap.topology_hash.as_slice() {
        return Verdict::OtherRound;
    }
    let identity = account_identity_bytes(&pending.account);
    let expected = derive_nonce(&snap.last_proof_block_hash, &identity, &pending.proof.salt);
    if pending.proof.nonce != expected {
        return Verdict::OtherRound;
    }
    match score(&pending.proof, snap) {
        Ok(verdict) => verdict,
        Err(reason) => Verdict::Malformed(reason),
    }
}

/// Draw the model, score every row, and apply the gates. `Err` is the
/// pallet's pre-scoring rejection.
fn score(proof: &QuantumProof, snap: &MiningSnapshot) -> Result<Verdict, String> {
    let index =
        TopologyIndex::new(&snap.nodes, &snap.edges).map_err(|e| format!("topology: {e:?}"))?;
    let (h, j) = generate_ising_model_indexed(
        proof.nonce,
        &snap.nodes,
        &snap.edges,
        &AllowedValueSpec::Set(snap.allowed_h_milli.as_slice()),
        &AllowedValueSpec::Set(snap.allowed_j_milli.as_slice()),
        &index,
    )
    .map_err(|e| format!("model: {e:?}"))?;
    let spin_spec = AllowedValueSpec::Set(snap.allowed_spin_milli.as_slice());

    let mut valid: Vec<(i64, Vec<i8>)> = Vec::with_capacity(proof.solutions.len());
    for packed in &proof.solutions {
        let milli = unpack_solution(packed, snap.nodes.len(), &spin_spec)
            .map_err(|e| format!("unpack: {e:?}"))?;
        let mut spins = Vec::with_capacity(milli.len());
        for value in milli {
            spins.push(match value.signum() {
                1 => 1i8,
                -1 => -1i8,
                _ => return Err("zero spin".into()),
            });
        }
        let energy = energy_of_solution_indexed(&spins, &h, &snap.edges, &j, &index)
            .map_err(|e| format!("energy: {e:?}"))?;
        if energy < snap.max_energy_milli {
            valid.push((energy, spins));
        }
    }

    let Some(best_energy_milli) = valid.iter().map(|(e, _)| *e).min() else {
        return Ok(Verdict::BelowGates);
    };
    let rows: Vec<&[i8]> = valid.iter().map(|(_, s)| s.as_slice()).collect();
    let target = rows
        .len()
        .min(usize::try_from(snap.min_solutions.max(1)).unwrap_or(usize::MAX));
    let selected_idx = select_diverse(&rows, target).map_err(|e| format!("select: {e:?}"))?;
    let selected: Vec<&[i8]> = selected_idx
        .iter()
        .filter_map(|&i| rows.get(i).copied())
        .collect();
    let diversity = calculate_diversity(&selected).map_err(|e| format!("diversity: {e:?}"))?;
    let n_valid = u32::try_from(rows.len()).unwrap_or(u32::MAX);

    if n_valid >= snap.min_solutions && diversity >= snap.min_diversity_milli {
        Ok(Verdict::Clears { best_energy_milli })
    } else {
        Ok(Verdict::BelowGates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantum_validation::packed::pack_solution;
    use quantum_validation::{
        energy_of_solution_indexed, generate_ising_model_indexed, AllowedValueSpec, TopologyIndex,
    };
    use sp_core::H256;

    fn ring() -> MiningSnapshot {
        MiningSnapshot {
            head_hash: [0u8; 32],
            last_proof_block_hash: [7u8; 32],
            topology_hash: vec![9u8; 32],
            nodes: vec![0, 1, 2, 3],
            edges: vec![(0, 1), (1, 2), (2, 3), (0, 3)],
            allowed_h_milli: vec![-1000, 0, 1000],
            allowed_j_milli: vec![-1000, 1000],
            allowed_spin_milli: vec![-1000, 1000],
            min_solutions: 1,
            max_energy_milli: 0,
            min_diversity_milli: 0,
            block_number: 42,
        }
    }

    const ACCOUNT: [u8; 32] = [5u8; 32];
    const SALT: [u8; 32] = [3u8; 32];

    fn nonce_for(snap: &MiningSnapshot) -> sp_core::U256 {
        let identity = account_identity_bytes(&ACCOUNT);
        derive_nonce(&snap.last_proof_block_hash, &identity, &SALT)
    }

    /// Brute-force the ground state of the drawn model over all 16 spin
    /// assignments, so the test needs no golden value.
    fn ground_state(snap: &MiningSnapshot) -> (i64, Vec<i8>) {
        let index = TopologyIndex::new(&snap.nodes, &snap.edges).expect("index");
        let (h, j) = generate_ising_model_indexed(
            nonce_for(snap),
            &snap.nodes,
            &snap.edges,
            &AllowedValueSpec::Set(snap.allowed_h_milli.as_slice()),
            &AllowedValueSpec::Set(snap.allowed_j_milli.as_slice()),
            &index,
        )
        .expect("model");
        let mut best: Option<(i64, Vec<i8>)> = None;
        for bits in 0u8..16 {
            let spins: Vec<i8> = (0..4)
                .map(|i| if bits & (1 << i) == 0 { -1 } else { 1 })
                .collect();
            let e =
                energy_of_solution_indexed(&spins, &h, &snap.edges, &j, &index).expect("energy");
            if best.as_ref().is_none_or(|(b, _)| e < *b) {
                best = Some((e, spins));
            }
        }
        best.expect("one assignment")
    }

    fn pending_with(snap: &MiningSnapshot, rows: &[Vec<i8>]) -> PendingProof {
        let spec = AllowedValueSpec::Set(snap.allowed_spin_milli.as_slice());
        let solutions = rows
            .iter()
            .map(|spins| {
                let milli: Vec<i32> = spins.iter().map(|&s| i32::from(s) * 1000).collect();
                pack_solution(&milli, &spec).expect("pack")
            })
            .collect();
        PendingProof {
            extrinsic_hash: [1u8; 32],
            account: ACCOUNT,
            proof: QuantumProof {
                topology_hash: H256::from_slice(&snap.topology_hash),
                nonce: nonce_for(snap),
                salt: SALT,
                solutions,
                device_access_time_us: 0,
            },
        }
    }

    #[test]
    fn a_ground_state_clears_when_the_ceiling_is_one_milli_above_it() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins]);
        assert_eq!(
            judge_pending_proof(&pending, &snap),
            Verdict::Clears {
                best_energy_milli: energy
            }
        );
    }

    /// The pallet gate is strict: energy must be below the ceiling.
    #[test]
    fn a_ground_state_at_the_ceiling_is_below_gates() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap);
        snap.max_energy_milli = energy;
        let pending = pending_with(&snap, &[spins]);
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::BelowGates);
    }

    #[test]
    fn a_nonce_from_another_root_is_another_round() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins]);
        snap.last_proof_block_hash = [8u8; 32];
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::OtherRound);
    }

    #[test]
    fn another_topology_is_another_round() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins]);
        snap.topology_hash = vec![10u8; 32];
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::OtherRound);
    }

    #[test]
    fn too_few_valid_rows_is_below_gates() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap);
        snap.max_energy_milli = energy + 1;
        snap.min_solutions = 2;
        let pending = pending_with(&snap, &[spins]);
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::BelowGates);
    }

    #[test]
    fn a_bad_packed_row_is_malformed() {
        let snap = ring();
        let mut pending = pending_with(&snap, &[]);
        pending.proof.solutions = vec![Vec::new()];
        assert!(matches!(
            judge_pending_proof(&pending, &snap),
            Verdict::Malformed(_)
        ));
    }

    #[test]
    fn clears_reads_the_verdict() {
        assert!(Verdict::Clears {
            best_energy_milli: -1
        }
        .clears());
        assert!(!Verdict::OtherRound.clears());
        assert!(!Verdict::BelowGates.clears());
        assert!(!Verdict::Malformed(String::new()).clears());
    }
}
