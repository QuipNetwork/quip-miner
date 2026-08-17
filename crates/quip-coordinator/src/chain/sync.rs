//! Startup gate: do not fund or mine until the validator has caught up.
//!
//! Every chain read the coordinator makes resolves against the node's best
//! block. While the node is still importing history, that block is far behind
//! the real head, so the miner account reads as empty even after the faucet has
//! already paid it, and any mining snapshot describes a round that ended long
//! ago. The coordinator then asks the faucet again, gets rate limited, burns
//! the whole funding budget, and exits "miner account is not funded" — which
//! names the wrong problem and, on exit 64, stops the supervisor respawning it.
//!
//! [`wait_until_synced`] moves that discovery to startup and names it: the
//! coordinator warns with the block it is syncing at and waits.

use super::{ChainError, RealChainClient};
use async_trait::async_trait;
use serde_json::Value;
use std::time::Duration;

/// How often to re-read the node's sync state.
const POLL_INTERVAL: Duration = Duration::from_secs(5);

/// How often to repeat the "still syncing" warning. A full sync runs for hours,
/// so every poll would bury the rest of the log.
const REPORT_INTERVAL: Duration = Duration::from_secs(30);

/// Consecutive clear polls needed to call a sync finished.
///
/// A node flaps `isSyncing` as it approaches the tip, so one clear poll in the
/// middle of a sync means little. This applies only once a sync has been seen:
/// a node that was never syncing is believed on its first answer, so a normal
/// start pays no extra wait.
const SYNCED_CONFIRMATIONS: u32 = 2;

/// How long to keep waiting on a validator that answers nothing at all before
/// giving up on the question and starting anyway.
///
/// A node that reports progress is worth waiting on for as long as it takes. A
/// node that never answers is the co-start case `preflight` already tolerates:
/// blocking forever there would stop the coordinator serving its miners.
const UNREACHABLE_GRACE: Duration = Duration::from_mins(1);

/// What the node reports about its own sync progress.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncStatus {
    /// `system_health.isSyncing`: the node is importing a backlog.
    pub is_syncing: bool,
    /// Connected peer count.
    pub peers: u64,
    /// False for a solo or `--dev` chain, where zero peers is normal.
    pub should_have_peers: bool,
    /// Best block the node has imported, from `system_syncState`.
    pub current_block: Option<u64>,
    /// Highest block any peer has announced.
    pub highest_block: Option<u64>,
}

impl SyncStatus {
    /// Blocks left to import, when both heights are known.
    #[must_use]
    pub fn behind(&self) -> Option<u64> {
        let (current, highest) = (self.current_block?, self.highest_block?);
        Some(highest.saturating_sub(current))
    }

    /// Human-readable progress for the log line.
    fn progress(&self) -> String {
        match (self.current_block, self.highest_block) {
            (Some(current), Some(highest)) => {
                let behind = highest.saturating_sub(current);
                format!("at block {current} of {highest} ({behind} behind)")
            }
            (Some(current), None) => format!("at block {current}"),
            _ => "at an unreported block".to_owned(),
        }
    }
}

/// Reads a node's sync progress. Separate from [`super::ChainClient`] so the
/// wait loop is testable without a validator.
#[async_trait]
pub trait SyncSource: Send + Sync {
    /// Current sync state of the node.
    ///
    /// # Errors
    /// Returns a transport error when the node cannot be reached.
    async fn sync_status(&self) -> Result<SyncStatus, ChainError>;
}

/// How the wait ended.
#[derive(Debug, PartialEq, Eq)]
pub enum SyncOutcome {
    /// The node reports it is caught up.
    Synced,
    /// The node never answered, so its progress is unknown. The caller starts
    /// anyway and lets the feeder retry.
    Unknown(String),
}

/// Parse a `system_health` response.
fn parse_health(v: &Value) -> Result<(bool, u64, bool), ChainError> {
    let is_syncing = v
        .get("isSyncing")
        .and_then(Value::as_bool)
        .ok_or_else(|| ChainError::Decode("system_health: isSyncing missing".into()))?;
    // Absent peer fields are reported as zero and false: they only shape a
    // secondary warning, so they must not fail the read that gates startup.
    let peers = v.get("peers").and_then(Value::as_u64).unwrap_or(0);
    let should_have_peers = v
        .get("shouldHavePeers")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    Ok((is_syncing, peers, should_have_peers))
}

/// Pull the block heights out of a `system_syncState` response.
///
/// These are for the operator's benefit only, so anything unreadable becomes
/// `None` rather than an error. `highestBlock` is absent on some runtimes.
fn parse_sync_state(v: &Value) -> (Option<u64>, Option<u64>) {
    let current = v.get("currentBlock").and_then(Value::as_u64);
    let highest = v.get("highestBlock").and_then(Value::as_u64).or(current);
    (current, highest)
}

#[async_trait]
impl SyncSource for RealChainClient {
    async fn sync_status(&self) -> Result<SyncStatus, ChainError> {
        let (is_syncing, peers, should_have_peers) =
            parse_health(&self.system_health_raw().await?)?;
        // Heights are decoration on the warning, so a node that does not serve
        // system_syncState still gets gated on isSyncing.
        let (current_block, highest_block) = match self.sync_state_raw().await {
            Ok(v) => parse_sync_state(&v),
            Err(e) => {
                tracing::debug!(error = %e, "validator does not report sync state");
                (None, None)
            }
        };
        Ok(SyncStatus {
            is_syncing,
            peers,
            should_have_peers,
            current_block,
            highest_block,
        })
    }
}

/// Wait until the validator reports it has caught up to the chain head.
///
/// Waits without limit while the node reports progress: a full sync legitimately
/// runs for hours, and every read before it finishes answers about a block that
/// is no longer current. Gives up after [`UNREACHABLE_GRACE`] when the node
/// answers nothing, so a validator that is merely slow to boot does not block
/// the coordinator from serving its miners.
///
/// `sleep` is injected so tests do not wait.
pub async fn wait_until_synced<C, S, Fut>(chain: &C, mut sleep: S) -> SyncOutcome
where
    C: SyncSource + ?Sized,
    S: FnMut(Duration) -> Fut + Send,
    Fut: std::future::Future<Output = ()> + Send,
{
    // Elapsed time is accumulated from the sleeps this loop asks for rather
    // than read off the clock, so the schedule is exactly what the caller's
    // sleep implements — real time in production, nothing at all under test.
    let mut waited = Duration::ZERO;
    let mut unreachable_for = Duration::ZERO;
    // Seeded at the interval so the first pass reports immediately.
    let mut since_report = REPORT_INTERVAL;
    let mut saw_syncing = false;
    let mut clear_polls: u32 = 0;

    loop {
        match chain.sync_status().await {
            Ok(status) => {
                unreachable_for = Duration::ZERO;
                if status.is_syncing {
                    saw_syncing = true;
                    clear_polls = 0;
                    if since_report >= REPORT_INTERVAL {
                        tracing::warn!(
                            peers = status.peers,
                            waited_s = waited.as_secs(),
                            "validator is syncing {}; waiting before funding and mining",
                            status.progress()
                        );
                        since_report = Duration::ZERO;
                    }
                } else {
                    clear_polls = clear_polls.saturating_add(1);
                    if !saw_syncing || clear_polls >= SYNCED_CONFIRMATIONS {
                        if saw_syncing {
                            tracing::info!(
                                waited_s = waited.as_secs(),
                                "validator finished syncing {}; funding and mining now",
                                status.progress()
                            );
                        }
                        // A node with no peers that expects some is not caught
                        // up, whatever it reports — it has nothing to catch up
                        // to, so its head is whatever it last imported.
                        if status.peers == 0 && status.should_have_peers {
                            tracing::warn!(
                                "validator reports no peers {}; it cannot follow the chain \
                                 until it finds some",
                                status.progress()
                            );
                        }
                        return SyncOutcome::Synced;
                    }
                }
            }
            Err(e) => {
                unreachable_for = unreachable_for.saturating_add(POLL_INTERVAL);
                if unreachable_for > UNREACHABLE_GRACE {
                    tracing::warn!(
                        error = %e,
                        "cannot read validator sync state; starting anyway and will retry"
                    );
                    return SyncOutcome::Unknown(e.to_string());
                }
                if since_report >= REPORT_INTERVAL {
                    tracing::warn!(error = %e, "cannot read validator sync state; retrying");
                    since_report = Duration::ZERO;
                }
            }
        }
        sleep(POLL_INTERVAL).await;
        waited = waited.saturating_add(POLL_INTERVAL);
        since_report = since_report.saturating_add(POLL_INTERVAL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    #[test]
    fn parses_a_syncing_health_response() {
        let v = json!({ "peers": 3, "isSyncing": true, "shouldHavePeers": true });
        assert_eq!(parse_health(&v).unwrap(), (true, 3, true));
    }

    #[test]
    fn health_without_is_syncing_is_an_error() {
        // Guessing "not syncing" here would defeat the gate entirely.
        assert!(parse_health(&json!({ "peers": 3 })).is_err());
    }

    #[test]
    fn missing_peer_fields_do_not_fail_the_read() {
        let v = json!({ "isSyncing": false });
        assert_eq!(parse_health(&v).unwrap(), (false, 0, false));
    }

    #[test]
    fn sync_state_reports_both_heights() {
        let v = json!({ "startingBlock": 0, "currentBlock": 1_108_021, "highestBlock": 1_240_333 });
        assert_eq!(parse_sync_state(&v), (Some(1_108_021), Some(1_240_333)));
    }

    #[test]
    fn a_missing_highest_block_falls_back_to_current() {
        let v = json!({ "currentBlock": 1_108_021 });
        assert_eq!(parse_sync_state(&v), (Some(1_108_021), Some(1_108_021)));
        assert_eq!(parse_sync_state(&json!({})), (None, None));
    }

    fn syncing_at(current: u64, highest: u64) -> SyncStatus {
        SyncStatus {
            is_syncing: true,
            peers: 3,
            should_have_peers: true,
            current_block: Some(current),
            highest_block: Some(highest),
        }
    }

    fn synced() -> SyncStatus {
        SyncStatus {
            is_syncing: false,
            peers: 3,
            should_have_peers: true,
            current_block: Some(1_240_333),
            highest_block: Some(1_240_333),
        }
    }

    #[test]
    fn progress_names_the_block_and_the_gap() {
        assert_eq!(
            syncing_at(1_108_021, 1_240_333).progress(),
            "at block 1108021 of 1240333 (132312 behind)"
        );
        assert_eq!(syncing_at(1_108_021, 1_240_333).behind(), Some(132_312));
    }

    /// A node ahead of the announced head is not "negative blocks behind".
    #[test]
    fn a_current_block_past_the_highest_reads_as_caught_up() {
        assert_eq!(syncing_at(1_240_333, 1_108_021).behind(), Some(0));
    }

    /// Sync source that walks a scripted sequence, repeating the last entry.
    struct Scripted {
        steps: Mutex<Vec<Result<SyncStatus, String>>>,
        calls: Mutex<usize>,
    }

    impl Scripted {
        fn new(steps: Vec<Result<SyncStatus, String>>) -> Self {
            Self {
                steps: Mutex::new(steps),
                calls: Mutex::new(0),
            }
        }

        fn calls(&self) -> usize {
            *self.calls.lock().unwrap()
        }
    }

    #[async_trait]
    impl SyncSource for Scripted {
        async fn sync_status(&self) -> Result<SyncStatus, ChainError> {
            let mut calls = self.calls.lock().unwrap();
            let steps = self.steps.lock().unwrap();
            let idx = (*calls).min(steps.len().saturating_sub(1));
            *calls += 1;
            steps
                .get(idx)
                .expect("scripted sync source has at least one step")
                .clone()
                .map_err(ChainError::Unavailable)
        }
    }

    async fn no_sleep(_: Duration) {}

    #[tokio::test]
    async fn a_synced_validator_returns_immediately() {
        let chain = Scripted::new(vec![Ok(synced())]);
        assert_eq!(
            wait_until_synced(&chain, no_sleep).await,
            SyncOutcome::Synced
        );
        assert_eq!(chain.calls(), 1);
    }

    #[tokio::test]
    async fn a_syncing_validator_is_waited_out() {
        let chain = Scripted::new(vec![
            Ok(syncing_at(1_108_021, 1_240_333)),
            Ok(syncing_at(1_200_000, 1_240_333)),
            Ok(synced()),
        ]);
        assert_eq!(
            wait_until_synced(&chain, no_sleep).await,
            SyncOutcome::Synced
        );
        // Two syncing polls, then two clear ones: a node that has been syncing
        // has to say so twice.
        assert_eq!(chain.calls(), 4);
    }

    /// Nodes drop `isSyncing` for a poll as they approach the tip and then pick
    /// it up again. One clear poll in the middle of a sync is not the end of it.
    #[tokio::test]
    async fn a_single_clear_poll_mid_sync_does_not_open_the_gate() {
        let chain = Scripted::new(vec![
            Ok(syncing_at(1_108_021, 1_240_333)),
            Ok(synced()),
            Ok(syncing_at(1_200_000, 1_240_333)),
            Ok(synced()),
            Ok(synced()),
        ]);
        assert_eq!(
            wait_until_synced(&chain, no_sleep).await,
            SyncOutcome::Synced
        );
        assert_eq!(chain.calls(), 5);
    }

    /// The wait has no ceiling while the node reports progress: an initial sync
    /// runs far longer than the funding budget, and cutting it short is what
    /// produced the misleading "not funded" exit.
    #[tokio::test]
    async fn a_long_sync_is_not_abandoned() {
        let mut steps = vec![Ok(syncing_at(0, 5_000_000)); 4096];
        steps.push(Ok(synced()));
        let chain = Scripted::new(steps);
        assert_eq!(
            wait_until_synced(&chain, no_sleep).await,
            SyncOutcome::Synced
        );
        assert_eq!(chain.calls(), 4098);
    }

    #[tokio::test]
    async fn an_unreachable_validator_gives_up_after_the_grace_period() {
        let chain = Scripted::new(vec![Err("connection refused".into())]);
        let out = wait_until_synced(&chain, no_sleep).await;
        assert!(matches!(out, SyncOutcome::Unknown(ref e) if e.contains("connection refused")));
        // Grace divided by the poll interval, plus the pass that trips it.
        let expected = (UNREACHABLE_GRACE.as_secs() / POLL_INTERVAL.as_secs()) + 1;
        assert_eq!(chain.calls(), usize::try_from(expected).unwrap());
    }

    /// A validator that is slow to boot must not spend its grace period: the
    /// budget covers consecutive silence, not silence in total.
    #[tokio::test]
    async fn a_validator_that_comes_back_resets_the_grace_period() {
        // Two silences that each fit inside the grace, but together exceed it.
        let mut steps = vec![Err("connection refused".to_owned()); 10];
        steps.push(Ok(syncing_at(10, 1_000)));
        steps.extend(vec![Err("connection refused".to_owned()); 10]);
        steps.push(Ok(synced()));
        let chain = Scripted::new(steps);
        assert_eq!(
            wait_until_synced(&chain, no_sleep).await,
            SyncOutcome::Synced
        );
    }
}
