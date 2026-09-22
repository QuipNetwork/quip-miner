//! Win-time candidate stash (agm.2.4).
//!
//! Holds the most-viable solutions per generation and projects, from the decay
//! model, the block at which each becomes viable as difficulty eases — so the
//! coordinator can submit at the right block without polling the chain every
//! block. Already-viable solutions submit immediately on the session path; this
//! stash captures solutions that don't clear the *current* (already-decayed)
//! threshold yet but will after enough decay, so they aren't discarded.

use crate::decay::{DecayAlgorithm, DecayModel};
use quip_proto::v1::Solution;
use serde::Serialize;
use std::fmt::Write as _;

/// How far ahead of the last proof the stash projects: 256 epochs of 100
/// blocks. A candidate that does not clear inside this window is not held.
pub const DECAY_HORIZON_BLOCKS: u64 = 25_600;

/// A stashed candidate: a validated solution set awaiting its viability block.
#[derive(Clone, Debug)]
pub struct Candidate {
    /// Job id (nonce) bytes.
    pub job_id: Vec<u8>,
    /// `PoW` salt (needed to build the live proof); `None` for mempool jobs.
    pub salt: Option<[u8; 32]>,
    /// Generation this candidate was mined for.
    pub generation: u64,
    /// Best energy of the stashed solutions, in milli-units.
    pub best_energy_milli: i64,
    /// Pairwise diversity of the stashed set, in milli-units.
    pub diversity_milli: u32,
    /// Count of gate-passing solutions retained.
    pub n_valid: u32,
    /// Solutions to resubmit when the candidate becomes viable.
    pub solutions: Vec<Solution>,
    /// Whether the job was a `PoW` job.
    pub is_pow: bool,
    /// Mempool order id (empty for `PoW`).
    pub order_id: Vec<u8>,
    /// Device access time reported by the miner, in microseconds.
    pub device_access_time_us: u64,
    /// Whether this candidate has already been submitted.
    pub submitted: bool,
}

/// Per-generation stash of the top-K most-viable candidates plus the
/// projection inputs (decay model and last-proof block).
pub struct WinStash {
    generation: u64,
    /// Threshold model for the round. `None` until [`reset`](Self::reset)
    /// arms one, and while the chain reads that build it fail.
    model: Option<DecayModel>,
    last_proof_block: u64,
    k: usize,
    /// Kept sorted best-first (lowest energy), length ≤ `k`.
    candidates: Vec<Candidate>,
}

impl WinStash {
    /// Empty stash retaining the top-`k` candidates (k ≥ 1). No projection
    /// until [`reset`](Self::reset).
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            generation: 0,
            model: None,
            last_proof_block: 0,
            k: k.max(1),
            candidates: Vec::new(),
        }
    }

    /// Re-arm for a new generation with fresh projection inputs, dropping all
    /// held candidates (the prior round's problem is stale after a reseed).
    pub fn reset(&mut self, generation: u64, model: Option<DecayModel>, last_proof_block: u64) {
        self.generation = generation;
        self.model = model;
        self.last_proof_block = last_proof_block;
        self.candidates.clear();
    }

    /// Swap the decay rule in place, keeping every candidate. Returns whether
    /// the rule changed. A runtime upgrade takes effect at one block, so the
    /// feeder calls this on every poll with the rule the head's
    /// `spec_version` implies.
    pub fn set_algorithm(&mut self, algorithm: DecayAlgorithm) -> bool {
        match self.model.as_mut() {
            Some(m) if m.algorithm != algorithm => {
                m.algorithm = algorithm;
                true
            }
            _ => false,
        }
    }

    /// The rule the projection runs under, or `None` without a model.
    #[must_use]
    pub fn algorithm(&self) -> Option<DecayAlgorithm> {
        self.model.as_ref().map(|m| m.algorithm)
    }

    /// Current generation this stash is armed for.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Whether no candidates are currently held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.candidates.is_empty()
    }

    /// Block at which `energy_milli` becomes viable: the last proof block
    /// plus the first elapsed count whose threshold strictly exceeds it.
    /// `None` without a model, or if it never clears within
    /// [`DECAY_HORIZON_BLOCKS`].
    #[must_use]
    pub fn viability_block(&self, energy_milli: i64) -> Option<u64> {
        let model = self.model.as_ref()?;
        model
            .first_clearing_elapsed(energy_milli, DECAY_HORIZON_BLOCKS)
            .map(|elapsed| self.last_proof_block.saturating_add(elapsed))
    }

    /// Insert a candidate if it becomes viable within the horizon and ranks in
    /// the top-K by energy (lowest wins). Returns whether it was kept.
    pub fn insert(&mut self, cand: Candidate) -> bool {
        if self.viability_block(cand.best_energy_milli).is_none() {
            return false; // never clears within the horizon → not worth holding
        }
        let job_id = cand.job_id.clone();
        self.candidates.push(cand);
        self.candidates.sort_by_key(|c| c.best_energy_milli);
        self.candidates.truncate(self.k);
        self.candidates.iter().any(|c| c.job_id == job_id)
    }

    /// The best (lowest-energy) unsubmitted candidate whose viability block has
    /// arrived at `current_block`, if any. Candidates are best-first, so this
    /// returns the strongest due one.
    #[must_use]
    pub fn due_at(&self, current_block: u64) -> Option<&Candidate> {
        self.candidates.iter().find(|c| {
            !c.submitted
                && self
                    .viability_block(c.best_energy_milli)
                    .is_some_and(|b| b <= current_block)
        })
    }

    /// The candidate to submit now: the best due (viability block arrived,
    /// unsubmitted) candidate that also strictly improves on `current_best`
    /// (or when there is none). Mirrors [`crate::validate::beats_current`], so
    /// the win-time path never regresses what the session path already sent.
    #[must_use]
    pub fn due_improving(
        &self,
        current_block: u64,
        current_best: Option<i64>,
    ) -> Option<&Candidate> {
        self.due_at(current_block)
            .filter(|c| current_best.is_none_or(|b| c.best_energy_milli < b))
    }

    /// Mark a candidate submitted so the driver won't re-submit it.
    pub fn mark_submitted(&mut self, job_id: &[u8]) {
        if let Some(c) = self.candidates.iter_mut().find(|c| c.job_id == job_id) {
            c.submitted = true;
        }
    }

    /// Per-qblock summary for the `attempts.json` annotation file.
    #[must_use]
    pub fn summary(&self) -> StashSummary {
        StashSummary {
            generation: self.generation,
            last_proof_block: self.last_proof_block,
            epoch_length: self.model.as_ref().map_or(0, |m| m.epoch_length),
            decay_algorithm: self.model.as_ref().map_or("none", |m| m.algorithm.name()),
            candidates: self
                .candidates
                .iter()
                .map(|c| CandidateSummary {
                    job_id: hex(&c.job_id),
                    best_energy_milli: c.best_energy_milli,
                    diversity_milli: c.diversity_milli,
                    n_valid: c.n_valid,
                    viability_block: self.viability_block(c.best_energy_milli),
                    submitted: c.submitted,
                })
                .collect(),
        }
    }
}

/// Serializable stash summary written to `attempts.json`.
#[derive(Debug, Clone, Serialize)]
pub struct StashSummary {
    /// Generation the stash is armed for.
    pub generation: u64,
    /// Last proof block used as the projection origin.
    pub last_proof_block: u64,
    /// Blocks per decay step.
    pub epoch_length: u64,
    /// Decay rule the projection ran under: `stepwise`, `continuous`, or
    /// `none` without a model.
    pub decay_algorithm: &'static str,
    /// Top-K candidates currently held.
    pub candidates: Vec<CandidateSummary>,
}

impl StashSummary {
    /// Worst and best retained energies, in milli-units.
    ///
    /// A new candidate must beat the worst to displace it once the stash is
    /// full. `None` when the stash holds nothing.
    #[must_use]
    pub fn retained_band_milli(&self) -> Option<(i64, i64)> {
        let best = self.candidates.first()?.best_energy_milli;
        let worst = self.candidates.last()?.best_energy_milli;
        Some((worst, best))
    }
}

/// One candidate's annotation in the stash summary.
#[derive(Debug, Clone, Serialize)]
pub struct CandidateSummary {
    /// Job id (nonce) hex.
    pub job_id: String,
    /// Best energy in milli-units.
    pub best_energy_milli: i64,
    /// Diversity in milli-units.
    pub diversity_milli: u32,
    /// Gate-passing solution count.
    pub n_valid: u32,
    /// Projected block at which this candidate becomes viable.
    pub viability_block: Option<u64>,
    /// Whether already submitted.
    pub submitted: bool,
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decay::{DecayAlgorithm, DecayModel, EnergyCurve};

    /// Stepwise thresholds -50000, -48775, -47581, -46416 at blocks 100, 110,
    /// 120, 130 (last proof 100, epoch 10).
    fn model() -> DecayModel {
        DecayModel {
            base_max_energy_milli: -50_000,
            curve: Some(EnergyCurve {
                min_milli: -100_000,
                knee_milli: -50_000,
                max_milli: -1_000,
            }),
            epoch_length: 10,
            algorithm: DecayAlgorithm::Stepwise,
        }
    }

    fn stash_with(model: Option<DecayModel>, last_proof_block: u64, k: usize) -> WinStash {
        let mut s = WinStash::new(k);
        s.reset(1, model, last_proof_block);
        s
    }

    fn cand(job_id: u8, energy: i64) -> Candidate {
        Candidate {
            job_id: vec![job_id],
            salt: Some([job_id; 32]),
            generation: 1,
            best_energy_milli: energy,
            diversity_milli: 200,
            n_valid: 5,
            solutions: vec![],
            is_pow: true,
            order_id: vec![],
            device_access_time_us: 0,
            submitted: false,
        }
    }

    fn job_byte(c: &Candidate) -> u8 {
        *c.job_id
            .first()
            .expect("test candidates use 1-byte job ids")
    }

    #[test]
    fn viability_block_offsets_the_clearing_block_from_the_last_proof() {
        let s = stash_with(Some(model()), 100, 4);
        // -49000 clears at step 1 → block 110.
        assert_eq!(s.viability_block(-49_000), Some(110));
        // -47000 clears at step 3 (-46416 is the first threshold above it) → 130.
        assert_eq!(s.viability_block(-47_000), Some(130));
        // Already below the base threshold → viable at the last proof block.
        assert_eq!(s.viability_block(-51_000), Some(100));
        // Above the easy cap → never.
        assert_eq!(s.viability_block(-1), None);
        // No projection → never.
        assert_eq!(stash_with(None, 100, 4).viability_block(-51_000), None);
    }

    #[test]
    fn insert_keeps_top_k_and_rejects_never_viable() {
        let mut s = stash_with(Some(model()), 100, 2);
        assert!(s.insert(cand(1, -49_000)));
        assert!(s.insert(cand(2, -47_500)));
        // A third, worse candidate is dropped (k=2, keeps the two best).
        assert!(!s.insert(cand(3, -46_500)));
        // A better one displaces the worst.
        assert!(s.insert(cand(4, -49_900)));
        // Never-viable (above the easy cap) is rejected outright.
        assert!(!s.insert(cand(5, -100)));
        let ids: Vec<u8> = s.candidates.iter().map(job_byte).collect();
        assert_eq!(ids, vec![4, 1]); // best-first: -49900, -49000
    }

    #[test]
    fn due_at_returns_best_arrived_unsubmitted() {
        let mut s = stash_with(Some(model()), 100, 4);
        let _ = s.insert(cand(1, -49_000)); // viable at block 110
        let _ = s.insert(cand(2, -47_000)); // viable at block 130
        assert!(s.due_at(109).is_none());
        assert_eq!(s.due_at(110).map(job_byte), Some(1));
        // At 135 both are due; cand 1 (-49000) is the stronger one.
        assert_eq!(s.due_at(135).map(job_byte), Some(1));
        s.mark_submitted(&[1]);
        assert_eq!(s.due_at(135).map(job_byte), Some(2));
    }

    #[test]
    fn due_improving_requires_arrival_and_improvement() {
        let mut s = stash_with(Some(model()), 100, 4);
        let _ = s.insert(cand(1, -49_000)); // viable at block 110
        assert!(s.due_improving(109, None).is_none());
        assert_eq!(s.due_improving(110, None).map(job_byte), Some(1));
        // Current best already stronger → nothing improves.
        assert!(s.due_improving(110, Some(-49_500)).is_none());
        // Current best weaker → submit.
        assert_eq!(s.due_improving(110, Some(-48_000)).map(job_byte), Some(1));
    }

    #[test]
    fn set_algorithm_keeps_candidates_and_moves_viability_earlier() {
        let mut s = stash_with(Some(model()), 100, 4);
        assert!(s.insert(cand(1, -49_500)));
        assert_eq!(s.viability_block(-49_500), Some(110));
        assert_eq!(s.algorithm(), Some(DecayAlgorithm::Stepwise));

        assert!(s.set_algorithm(DecayAlgorithm::Continuous));
        assert_eq!(s.algorithm(), Some(DecayAlgorithm::Continuous));
        let block = s.viability_block(-49_500).unwrap_or(u64::MAX);
        assert!(
            100 < block && block < 110,
            "continuous viability at {block}"
        );
        assert_eq!(s.candidates.len(), 1);
        assert_eq!(s.summary().decay_algorithm, "continuous");

        // Same algorithm again is not a change.
        assert!(!s.set_algorithm(DecayAlgorithm::Continuous));
        // No projection → nothing to switch.
        assert!(!stash_with(None, 100, 4).set_algorithm(DecayAlgorithm::Continuous));
    }

    #[test]
    fn reset_drops_candidates_and_summary_names_the_model() {
        let mut s = stash_with(Some(model()), 100, 4);
        let _ = s.insert(cand(1, -49_000));
        s.reset(2, None, 0);
        assert!(s.is_empty());
        assert_eq!(s.generation(), 2);
        let summary = s.summary();
        assert_eq!(summary.epoch_length, 0);
        assert_eq!(summary.decay_algorithm, "none");
    }
}
