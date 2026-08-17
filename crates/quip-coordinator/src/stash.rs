//! Win-time candidate stash (agm.2.4).
//!
//! Holds the most-viable solutions per generation and projects, from the decay
//! schedule, the block at which each becomes viable as difficulty eases — so the
//! coordinator can submit at the right block without polling the chain every
//! block. Already-viable solutions submit immediately on the session path; this
//! stash captures solutions that don't clear the *current* (already-decayed)
//! threshold yet but will after enough decay, so they aren't discarded.

use crate::decay::step_for_energy;
use quip_proto::v1::Solution;
use serde::Serialize;
use std::fmt::Write as _;

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

/// Per-generation stash of the top-K most-viable candidates plus the projection
/// inputs (decay schedule, last-proof block, epoch length).
pub struct WinStash {
    generation: u64,
    /// `max_energy_milli` threshold at each decay step (`build_decay_schedule`).
    schedule: Vec<i64>,
    last_proof_block: u64,
    epoch_length: u64,
    k: usize,
    /// Kept sorted best-first (lowest energy), length ≤ `k`.
    candidates: Vec<Candidate>,
}

impl WinStash {
    /// Empty stash retaining the top-`k` candidates (k ≥ 1). No schedule until
    /// [`reset`](Self::reset).
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            generation: 0,
            schedule: Vec::new(),
            last_proof_block: 0,
            epoch_length: 0,
            k: k.max(1),
            candidates: Vec::new(),
        }
    }

    /// Re-arm for a new generation with fresh projection inputs, dropping all
    /// held candidates (the prior round's problem is stale after a reseed).
    pub fn reset(
        &mut self,
        generation: u64,
        schedule: Vec<i64>,
        last_proof_block: u64,
        epoch_length: u64,
    ) {
        self.generation = generation;
        self.schedule = schedule;
        self.last_proof_block = last_proof_block;
        self.epoch_length = epoch_length;
        self.candidates.clear();
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

    /// Decay step at which `energy_milli` first clears, or `None` if it never
    /// clears within the projected horizon.
    #[must_use]
    pub fn viability_step(&self, energy_milli: i64) -> Option<usize> {
        step_for_energy(&self.schedule, energy_milli)
    }

    /// Block at which `energy_milli` becomes viable: `last_proof_block + step *
    /// epoch_length`. `None` if it never clears within the horizon.
    #[must_use]
    pub fn viability_block(&self, energy_milli: i64) -> Option<u64> {
        self.viability_step(energy_milli)
            .map(|s| self.last_proof_block + (s as u64) * self.epoch_length)
    }

    /// Insert a candidate if it becomes viable within the horizon and ranks in
    /// the top-K by energy (lowest wins). Returns whether it was kept.
    pub fn insert(&mut self, cand: Candidate) -> bool {
        if self.viability_step(cand.best_energy_milli).is_none() {
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
            epoch_length: self.epoch_length,
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

    // schedule[s] = threshold at step s (monotonic non-decreasing / easing).
    fn stash_with(schedule: Vec<i64>, last_proof_block: u64, epoch: u64, k: usize) -> WinStash {
        let mut s = WinStash::new(k);
        s.reset(1, schedule, last_proof_block, epoch);
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
    fn viability_block_maps_step_to_block() {
        // Threshold eases: -50000, -48000, -46000. last_proof 100, epoch 10.
        let s = stash_with(vec![-50_000, -48_000, -46_000], 100, 10, 4);
        // Energy -49000 clears at step 1 (first threshold > -49000) → block 110.
        assert_eq!(s.viability_step(-49_000), Some(1));
        assert_eq!(s.viability_block(-49_000), Some(110));
        // Energy -47000 clears at step 2 → block 120.
        assert_eq!(s.viability_block(-47_000), Some(120));
        // Energy already below the base threshold clears at step 0 (viable now).
        assert_eq!(s.viability_block(-51_000), Some(100));
        // Energy the schedule never exceeds → never viable in horizon.
        assert_eq!(s.viability_block(-1_000), None);
    }

    #[test]
    fn insert_keeps_top_k_and_rejects_never_viable() {
        let mut s = stash_with(vec![-50_000, -48_000, -46_000], 100, 10, 2);
        assert!(s.insert(cand(1, -49_000)));
        assert!(s.insert(cand(2, -47_500)));
        // A third, worse candidate is dropped (k=2, keeps the two best).
        assert!(!s.insert(cand(3, -46_500)));
        // A better one displaces the worst.
        assert!(s.insert(cand(4, -49_900)));
        // Never-viable (above the whole schedule) is rejected outright.
        assert!(!s.insert(cand(5, -100)));
        let ids: Vec<u8> = s.candidates.iter().map(job_byte).collect();
        assert_eq!(ids, vec![4, 1]); // best-first: -49900, -49000
    }

    #[test]
    fn due_at_returns_best_arrived_unsubmitted() {
        let mut s = stash_with(vec![-50_000, -48_000, -46_000], 100, 10, 4);
        let _ = s.insert(cand(1, -49_000)); // viable at block 110
        let _ = s.insert(cand(2, -47_000)); // viable at block 120
                                            // Before block 110: nothing due.
        assert!(s.due_at(109).is_none());
        // At 110: candidate 1 due.
        assert_eq!(s.due_at(110).map(job_byte), Some(1));
        // At 125 both due → best (lower energy = cand 2? no, -47000 > -49000).
        // cand 1 (-49000) is lower energy = stronger, and both arrived → cand 1.
        assert_eq!(s.due_at(125).map(job_byte), Some(1));
        // Once cand 1 submitted, cand 2 becomes the best due one.
        s.mark_submitted(&[1]);
        assert_eq!(s.due_at(125).map(job_byte), Some(2));
    }

    #[test]
    fn due_improving_requires_arrival_and_improvement() {
        let mut s = stash_with(vec![-50_000, -48_000, -46_000], 100, 10, 4);
        let _ = s.insert(cand(1, -49_000)); // viable at block 110
                                            // Not arrived yet → nothing.
        assert!(s.due_improving(109, None).is_none());
        // Arrived + no current best → submit.
        assert_eq!(s.due_improving(110, None).map(job_byte), Some(1));
        // Arrived but a better proof already stands → don't regress.
        assert!(s.due_improving(110, Some(-49_500)).is_none());
        // Arrived + current best is worse → submit the improvement.
        assert_eq!(s.due_improving(110, Some(-48_000)).map(job_byte), Some(1));
    }

    #[test]
    fn reset_clears_candidates_and_rearms() {
        let mut s = stash_with(vec![-50_000, -48_000], 100, 10, 4);
        let _ = s.insert(cand(1, -49_000));
        assert!(!s.is_empty());
        s.reset(2, vec![-40_000, -38_000], 200, 10);
        assert!(s.is_empty());
        assert_eq!(s.generation(), 2);
        assert_eq!(s.viability_block(-39_000), Some(210));
    }

    #[test]
    fn retained_band_is_worst_then_best_or_empty() {
        let empty = stash_with(vec![-50_000, -48_000, -46_000], 100, 10, 4);
        assert!(empty.summary().retained_band_milli().is_none());

        let mut s = stash_with(vec![-50_000, -48_000, -46_000], 100, 10, 4);
        let _ = s.insert(cand(1, -49_000));
        let _ = s.insert(cand(2, -47_000));
        assert_eq!(s.summary().retained_band_milli(), Some((-47_000, -49_000)));
    }

    #[test]
    fn summary_serializes_annotations() {
        let mut s = stash_with(vec![-50_000, -48_000, -46_000], 100, 10, 4);
        let _ = s.insert(cand(0xab, -49_000));
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&s.summary()).unwrap()).unwrap();
        assert_eq!(v.get("generation"), Some(&serde_json::json!(1)));
        assert_eq!(
            v.pointer("/candidates/0/job_id"),
            Some(&serde_json::json!("ab"))
        );
        assert_eq!(
            v.pointer("/candidates/0/viability_block"),
            Some(&serde_json::json!(110))
        );
        assert_eq!(
            v.pointer("/candidates/0/submitted"),
            Some(&serde_json::json!(false))
        );
    }
}
