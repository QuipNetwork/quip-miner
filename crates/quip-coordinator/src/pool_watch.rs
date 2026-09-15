//! Decide whether a proof pending in the pool ends the current round.
//!
//! The judge replays the pallet's `submit_proof` checks
//! (`quip-validator/pallets/quantum-pow/src/lib.rs`, `validate_proof`) with
//! the same `quantum_validation` functions, against the snapshot the feeder is
//! mining. Difficulty eases every block until a win, so a proof that clears
//! now also clears at its inclusion block.

use std::collections::{HashMap, HashSet};

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
/// extensions. A clearing proof from an unregistered account is included, fails,
/// and leaves the pool without a qblock. The feeder then resumes and ignores
/// that proof, by signer and nonce, for the rest of the root, so a resubmission
/// cannot stop the miners again.
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

/// Consecutive polls with no clearing proof before the round resumes. One
/// poll can race the block import that prunes the proof from the pool.
pub const AWAIT_CLEAR_POLLS: u32 = 2;

/// Blocks to hold the miners for a pending proof before the round resumes. A
/// proof that stays pending this long is not landing.
pub const AWAIT_QBLOCK_MAX_BLOCKS: u64 = 2;

/// The clearing proof a scan found, for the log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clearing {
    /// Signing account of the proof.
    pub account: [u8; 32],
    /// The proof's `PoW` nonce. With the account it names the proof across resubmissions.
    pub nonce: sp_core::U256,
    /// Its best energy, in milli-units.
    pub best_energy_milli: i64,
}

/// What the latest pool read observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolObservation {
    /// A clearing proof was seen.
    Clearing,
    /// The pool was read and holds no clearing proof.
    Clear,
    /// The pool could not be read.
    Unknown,
}

/// Why the feeder leaves `AwaitingQBlock` without a new root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResumeReason {
    /// No clearing proof for [`AWAIT_CLEAR_POLLS`] polls.
    PoolClear,
    /// [`AWAIT_QBLOCK_MAX_BLOCKS`] blocks passed since the stop.
    BlocksElapsed,
}

impl std::fmt::Display for ResumeReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PoolClear => "no clearing proof is pending",
            Self::BlocksElapsed => "no qblock followed the pending proof",
        })
    }
}

/// Per-round memory of the pool scan and of the wait for the qblock.
#[derive(Debug, Default)]
pub struct PoolWatch {
    /// Round the cached verdicts belong to.
    root: Option<[u8; 32]>,
    /// Extrinsic hash to the clearing verdict, if it clears.
    verdicts: HashMap<[u8; 32], Option<Clearing>>,
    /// Proofs that stopped the miners and then left the pool without a qblock,
    /// or stayed pending past the block ceiling, by signer and `PoW` nonce.
    /// A resubmission under a new extrinsic hash is ignored too. Cleared with the root.
    ignored: HashSet<([u8; 32], sp_core::U256)>,
    /// Block the miners were stopped at.
    stopped_at: Option<u64>,
    /// Consecutive polls since the stop with no clearing proof.
    clear_polls: u32,
}

impl PoolWatch {
    /// An empty watch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Judge every pending proof against `snap`. Returns the clearing proof
    /// with the lowest energy, if any clears. Verdicts are cached by
    /// extrinsic hash for the life of the root, except `BelowGates`.
    pub fn scan(&mut self, pending: &[PendingProof], snap: &MiningSnapshot) -> Option<Clearing> {
        if self.root != Some(snap.last_proof_block_hash) {
            self.root = Some(snap.last_proof_block_hash);
            self.verdicts.clear();
            self.ignored.clear();
        }
        let mut best: Option<Clearing> = None;
        for p in pending {
            if self.ignored.contains(&(p.account, p.proof.nonce)) {
                continue;
            }
            let verdict = if let Some(verdict) = self.verdicts.get(&p.extrinsic_hash) {
                *verdict
            } else {
                let verdict = match judge_pending_proof(p, snap) {
                    Verdict::Clears { best_energy_milli } => Some(Clearing {
                        account: p.account,
                        nonce: p.proof.nonce,
                        best_energy_milli,
                    }),
                    Verdict::OtherRound => None,
                    Verdict::BelowGates => continue,
                    Verdict::Malformed(reason) => {
                        tracing::debug!(
                            extrinsic = %crate::chain::extrinsic::hex_encode(&p.extrinsic_hash),
                            reason,
                            "pool: pending proof is malformed"
                        );
                        None
                    }
                };
                let _ = self.verdicts.insert(p.extrinsic_hash, verdict);
                verdict
            };
            if let Some(c) = verdict {
                if best.is_none_or(|b| c.best_energy_milli < b.best_energy_milli) {
                    best = Some(c);
                }
            }
        }
        best
    }

    /// Called on every same-root resume: the proofs that held the miners
    /// produced no qblock, so they cannot hold them again on this root.
    pub fn expire_clearing(&mut self) {
        self.verdicts.retain(|_, verdict| {
            if let Some(c) = verdict {
                let _ = self.ignored.insert((c.account, c.nonce));
                false
            } else {
                true
            }
        });
    }

    /// The miners were stopped at `block`.
    pub fn stopped(&mut self, block: u64) {
        self.stopped_at = Some(block);
        self.clear_polls = 0;
    }

    /// After one poll in `AwaitingQBlock`: whether to resume the round on the
    /// same root, and why. `observed` records the pool read outcome.
    /// `block` is the poll's head number.
    pub fn resume_reason(&mut self, observed: PoolObservation, block: u64) -> Option<ResumeReason> {
        match observed {
            PoolObservation::Clearing => self.clear_polls = 0,
            PoolObservation::Clear => self.clear_polls = self.clear_polls.saturating_add(1),
            PoolObservation::Unknown => {}
        }
        if self.clear_polls >= AWAIT_CLEAR_POLLS {
            return Some(ResumeReason::PoolClear);
        }
        let since = self.stopped_at?;
        if block.saturating_sub(since) >= AWAIT_QBLOCK_MAX_BLOCKS {
            return Some(ResumeReason::BlocksElapsed);
        }
        None
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

    fn nonce_for(snap: &MiningSnapshot, salt: [u8; 32]) -> sp_core::U256 {
        let identity = account_identity_bytes(&ACCOUNT);
        derive_nonce(&snap.last_proof_block_hash, &identity, &salt)
    }

    /// Brute-force the ground state of the drawn model over all 16 spin
    /// assignments, so the test needs no golden value.
    fn ground_state(snap: &MiningSnapshot, salt: [u8; 32]) -> (i64, Vec<i8>) {
        let index = TopologyIndex::new(&snap.nodes, &snap.edges).expect("index");
        let (h, j) = generate_ising_model_indexed(
            nonce_for(snap, salt),
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

    fn pending_with(snap: &MiningSnapshot, rows: &[Vec<i8>], salt: [u8; 32]) -> PendingProof {
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
                nonce: nonce_for(snap, salt),
                salt,
                solutions,
                device_access_time_us: 0,
            },
        }
    }

    #[test]
    fn a_ground_state_clears_when_the_ceiling_is_one_milli_above_it() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], SALT);
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
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy;
        let pending = pending_with(&snap, &[spins], SALT);
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::BelowGates);
    }

    #[test]
    fn a_nonce_from_another_root_is_another_round() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], SALT);
        snap.last_proof_block_hash = [8u8; 32];
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::OtherRound);
    }

    #[test]
    fn another_topology_is_another_round() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], SALT);
        snap.topology_hash = vec![10u8; 32];
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::OtherRound);
    }

    #[test]
    fn too_few_valid_rows_is_below_gates() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        snap.min_solutions = 2;
        let pending = pending_with(&snap, &[spins], SALT);
        assert_eq!(judge_pending_proof(&pending, &snap), Verdict::BelowGates);
    }

    #[test]
    fn a_bad_packed_row_is_malformed() {
        let snap = ring();
        let mut pending = pending_with(&snap, &[], SALT);
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
    #[test]
    fn scan_reports_the_best_clearing_proof_and_caches_by_hash() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], SALT);
        let mut watch = PoolWatch::new();
        let first = watch
            .scan(std::slice::from_ref(&pending), &snap)
            .expect("clears");
        assert_eq!(first.account, ACCOUNT);
        assert_eq!(first.best_energy_milli, energy);
        // The same hash is not judged again: a corrupted copy under the same
        // hash still reports the cached verdict.
        let mut same_hash = pending;
        same_hash.proof.solutions = vec![Vec::new()];
        assert!(watch.scan(&[same_hash], &snap).is_some());
    }

    #[test]
    fn scan_forgets_verdicts_when_the_root_changes() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], SALT);
        let mut watch = PoolWatch::new();
        assert!(watch.scan(std::slice::from_ref(&pending), &snap).is_some());
        snap.last_proof_block_hash = [8u8; 32];
        assert!(watch.scan(&[pending], &snap).is_none());
    }

    #[test]
    fn a_below_gates_proof_is_judged_again_when_the_ceiling_eases() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy;
        let pending = pending_with(&snap, &[spins], SALT);
        let mut watch = PoolWatch::new();
        assert!(watch.scan(std::slice::from_ref(&pending), &snap).is_none());
        snap.max_energy_milli = energy + 1;
        assert!(watch.scan(&[pending], &snap).is_some());
    }

    #[test]
    fn a_timed_out_proof_cannot_stop_the_same_root_again() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], SALT);
        let mut watch = PoolWatch::new();
        assert!(watch.scan(std::slice::from_ref(&pending), &snap).is_some());
        watch.stopped(100);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clearing, 102),
            Some(ResumeReason::BlocksElapsed)
        );
        watch.expire_clearing();
        assert!(watch.scan(std::slice::from_ref(&pending), &snap).is_none());
        let mut resubmitted = pending;
        resubmitted.extrinsic_hash = [2u8; 32];
        assert!(watch.scan(&[resubmitted], &snap).is_none());

        snap.last_proof_block_hash = [8u8; 32];
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], SALT);
        assert!(watch.scan(&[pending], &snap).is_some());
    }

    #[test]
    fn a_proof_that_left_the_pool_without_a_qblock_cannot_stop_the_same_root_again() {
        let mut snap = ring();
        let (energy, spins) = ground_state(&snap, SALT);
        snap.max_energy_milli = energy + 1;
        let mut pending = pending_with(&snap, &[spins], SALT);
        let mut watch = PoolWatch::new();
        assert!(watch.scan(std::slice::from_ref(&pending), &snap).is_some());
        watch.stopped(100);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 100), None);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clear, 100),
            Some(ResumeReason::PoolClear)
        );
        watch.expire_clearing();
        pending.extrinsic_hash = [2u8; 32];
        assert!(watch.scan(&[pending], &snap).is_none());

        let salt = [4u8; 32];
        let (energy, spins) = ground_state(&snap, salt);
        snap.max_energy_milli = energy + 1;
        let pending = pending_with(&snap, &[spins], salt);
        assert!(watch.scan(&[pending], &snap).is_some());
    }

    #[test]
    fn scan_picks_the_lowest_energy_among_clearing_proofs() {
        let mut snap = ring();
        let (first_energy, first_spins) = ground_state(&snap, SALT);
        let salt = [4u8; 32];
        let (second_energy, second_spins) = ground_state(&snap, salt);
        assert_ne!(
            first_energy, second_energy,
            "fixtures need different energies"
        );
        snap.max_energy_milli = first_energy.max(second_energy) + 1;
        let first = pending_with(&snap, &[first_spins], SALT);
        let mut second = pending_with(&snap, &[second_spins], salt);
        second.extrinsic_hash = [2u8; 32];
        let mut watch = PoolWatch::new();
        let mut pending = [first, second];
        let best = watch.scan(&pending, &snap).expect("both proofs clear");
        assert_eq!(best.best_energy_milli, first_energy.min(second_energy));
        pending.reverse();
        let best = watch
            .scan(&pending, &snap)
            .expect("both proofs still clear");
        assert_eq!(best.best_energy_milli, first_energy.min(second_energy));
    }

    #[test]
    fn an_unreadable_pool_does_not_count_as_clear() {
        let mut watch = PoolWatch::new();
        watch.stopped(100);
        assert_eq!(watch.resume_reason(PoolObservation::Unknown, 100), None);
        assert_eq!(watch.resume_reason(PoolObservation::Unknown, 100), None);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 100), None);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clear, 100),
            Some(ResumeReason::PoolClear)
        );

        // An unreadable poll preserves a clear observation and the block ceiling.
        watch.stopped(100);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 100), None);
        assert_eq!(watch.resume_reason(PoolObservation::Unknown, 101), None);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clear, 101),
            Some(ResumeReason::PoolClear)
        );
        watch.stopped(100);
        assert_eq!(
            watch.resume_reason(PoolObservation::Unknown, 102),
            Some(ResumeReason::BlocksElapsed)
        );
        watch.stopped(100);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clear, 102),
            Some(ResumeReason::BlocksElapsed)
        );
    }

    #[test]
    fn resume_after_two_clear_polls() {
        let mut watch = PoolWatch::new();
        watch.stopped(100);
        assert_eq!(watch.resume_reason(PoolObservation::Clearing, 100), None);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 100), None);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clear, 100),
            Some(ResumeReason::PoolClear)
        );
    }

    #[test]
    fn a_clearing_poll_resets_the_clear_count() {
        let mut watch = PoolWatch::new();
        watch.stopped(100);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 100), None);
        assert_eq!(watch.resume_reason(PoolObservation::Clearing, 100), None);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 100), None);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clear, 100),
            Some(ResumeReason::PoolClear)
        );
    }

    #[test]
    fn resume_after_two_blocks_even_while_the_proof_is_pending() {
        let mut watch = PoolWatch::new();
        watch.stopped(100);
        assert_eq!(watch.resume_reason(PoolObservation::Clearing, 101), None);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clearing, 102),
            Some(ResumeReason::BlocksElapsed)
        );
    }

    #[test]
    fn stopped_resets_the_wait() {
        let mut watch = PoolWatch::new();
        watch.stopped(100);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 100), None);
        watch.stopped(105);
        assert_eq!(watch.resume_reason(PoolObservation::Clear, 105), None);
        assert_eq!(watch.resume_reason(PoolObservation::Clearing, 106), None);
        assert_eq!(
            watch.resume_reason(PoolObservation::Clearing, 107),
            Some(ResumeReason::BlocksElapsed)
        );
    }
}
