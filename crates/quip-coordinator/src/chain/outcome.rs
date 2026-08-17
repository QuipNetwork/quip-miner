//! Bounding repeated submissions of a proof the runtime keeps refusing.
//!
//! Submit outcome comes from chain state, not from decoded events, so a failed
//! dispatch arrives without a reason. That removes the distinction between a
//! retriable rejection and a permanent one. Retrying a permanent rejection
//! forever would mine into a wall silently, which is the failure shape this
//! project has already been bitten by once.
//!
//! The ledger restores a loud stop: after `max_attempts` consecutive failures
//! for one job inside one quantum block, the caller is told to stop.

use super::SubmitAction;
use std::collections::BTreeMap;

/// How many quantum blocks of history the ledger keeps.
///
/// Only the current round matters for the bound. A small window is kept so a
/// late resubmission from the previous round still finds its count, and so the
/// map cannot grow without limit in a long-running process.
pub const QBLOCK_RETENTION: u64 = 8;

/// Counts consecutive failed submissions per job, per quantum block.
#[derive(Debug)]
pub struct SubmitLedger {
    max_attempts: u32,
    /// qblock id to job id to consecutive failure count.
    counts: BTreeMap<u64, BTreeMap<Vec<u8>, u32>>,
}

impl SubmitLedger {
    /// Build a ledger that stops after `max_attempts` consecutive failures.
    ///
    /// A `max_attempts` of zero is treated as one: never retry silently.
    #[must_use]
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            counts: BTreeMap::new(),
        }
    }

    /// Record one failed submission and decide what the caller should do.
    pub fn record_failure(&mut self, qblock_id: u64, job_id: &[u8]) -> SubmitAction {
        let entry = self
            .counts
            .entry(qblock_id)
            .or_default()
            .entry(job_id.to_vec())
            .or_insert(0);
        *entry = entry.saturating_add(1);
        let n = *entry;
        self.prune(qblock_id);
        if n >= self.max_attempts {
            SubmitAction::StopFatal
        } else {
            SubmitAction::Retry
        }
    }

    /// Clear the count for a job that landed.
    pub fn record_success(&mut self, qblock_id: u64, job_id: &[u8]) {
        if let Some(jobs) = self.counts.get_mut(&qblock_id) {
            let _ = jobs.remove(job_id);
        }
    }

    /// Consecutive failures recorded for this job in this quantum block.
    #[must_use]
    pub fn attempts(&self, qblock_id: u64, job_id: &[u8]) -> u32 {
        self.counts
            .get(&qblock_id)
            .and_then(|jobs| jobs.get(job_id))
            .copied()
            .unwrap_or(0)
    }

    /// How many quantum blocks the ledger currently holds. For tests.
    #[must_use]
    pub fn tracked_qblocks(&self) -> usize {
        self.counts.len()
    }

    /// Drop rounds older than the retention window.
    ///
    /// The window includes `newest` and at most `QBLOCK_RETENTION - 1` earlier
    /// rounds, so the map holds at most [`QBLOCK_RETENTION`] keys.
    fn prune(&mut self, newest: u64) {
        let cutoff = newest.saturating_sub(QBLOCK_RETENTION.saturating_sub(1));
        self.counts.retain(|q, _| *q >= cutoff);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chain::SubmitAction;

    const JOB: &[u8] = b"job-1";

    #[test]
    fn failures_below_the_bound_ask_for_a_retry() {
        let mut led = SubmitLedger::new(3);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::Retry);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::Retry);
        assert_eq!(led.attempts(5, JOB), 2);
    }

    /// The bound is what restores fail-loud. Without it a proof the runtime
    /// will never accept retries until the round ends.
    #[test]
    fn the_bound_turns_a_retry_into_a_fatal_stop() {
        let mut led = SubmitLedger::new(3);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::Retry);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::Retry);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::StopFatal);
    }

    /// A new qblock is a new round with new difficulty. Counting across rounds
    /// would stop a proof that only failed because the old round was harder.
    #[test]
    fn a_new_qblock_starts_a_fresh_count() {
        let mut led = SubmitLedger::new(2);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::Retry);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::StopFatal);
        assert_eq!(led.record_failure(6, JOB), SubmitAction::Retry);
    }

    #[test]
    fn two_jobs_in_one_qblock_are_counted_apart() {
        let mut led = SubmitLedger::new(2);
        assert_eq!(led.record_failure(5, b"a"), SubmitAction::Retry);
        assert_eq!(led.record_failure(5, b"b"), SubmitAction::Retry);
        assert_eq!(led.attempts(5, b"a"), 1);
        assert_eq!(led.attempts(5, b"b"), 1);
    }

    #[test]
    fn a_success_clears_the_count() {
        let mut led = SubmitLedger::new(2);
        let _ = led.record_failure(5, JOB);
        led.record_success(5, JOB);
        assert_eq!(led.attempts(5, JOB), 0);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::Retry);
    }

    /// A bound of one must stop on the first failure, not allow a free retry.
    #[test]
    fn a_bound_of_one_stops_immediately() {
        let mut led = SubmitLedger::new(1);
        assert_eq!(led.record_failure(5, JOB), SubmitAction::StopFatal);
    }

    /// Old rounds must not accumulate forever in a long-running process.
    #[test]
    fn entries_from_older_qblocks_are_pruned() {
        let mut led = SubmitLedger::new(5);
        for q in 0..(QBLOCK_RETENTION + 10) {
            let _ = led.record_failure(q, JOB);
        }
        let tracked = u64::try_from(led.tracked_qblocks()).unwrap_or(u64::MAX);
        assert!(
            tracked <= QBLOCK_RETENTION,
            "tracked {} qblocks",
            led.tracked_qblocks()
        );
    }
}
