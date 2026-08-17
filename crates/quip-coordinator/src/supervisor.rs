//! Spawn miner binaries, restart policy by exit code, shutdown fan-out.

use crate::config::LaunchEntry;
use crate::logging::LogLevel;
use crate::session::{gen_session_token, shutdown_msg, CoordinatorState};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use tokio::process::{Child, Command};
use tokio::sync::{watch, Mutex};

/// Restart decision for a miner child exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Restart {
    /// Exit 0: respawn when jobs exist.
    OnDemand,
    /// 64/69/77: operator/env/token error — do not respawn.
    Never,
    /// 70 or signal: exponential backoff + failure budget.
    Backoff,
}

/// Map a process exit code to the coordinator restart policy.
#[must_use]
pub fn restart_policy(exit_code: i32) -> Restart {
    match exit_code {
        0 => Restart::OnDemand,
        64 | 69 | 77 => Restart::Never,
        // 70 (crash), negative codes (signals), and anything else → backoff.
        _ => Restart::Backoff,
    }
}

/// What the supervisor should do after a child exits, once backoff and the
/// failure budget are applied on top of the raw [`restart_policy`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartAction {
    /// Clean exit (0): respawn when a job is available. Resets backoff.
    OnDemand,
    /// Operator/env/token error (64/69/77): do not respawn.
    Never,
    /// Crash/signal: respawn after this many milliseconds of backoff.
    Backoff(u64),
    /// Too many crashes inside the window: stop respawning, mark unhealthy.
    Unhealthy,
}

/// Backoff + failure-budget knobs for a crashing miner.
#[derive(Debug, Clone, Copy)]
pub struct BackoffPolicy {
    /// First backoff delay; doubles per consecutive crash.
    pub base_ms: u64,
    /// Ceiling for the doubling.
    pub max_ms: u64,
    /// Crashes allowed within `window_ms` before the miner is marked unhealthy.
    pub budget: usize,
    /// Sliding window over which crashes are counted.
    pub window_ms: u64,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            base_ms: 500,
            max_ms: 30_000,
            budget: 5,
            window_ms: 60_000,
        }
    }
}

/// Per-miner restart bookkeeping: exponential backoff with a sliding-window
/// failure budget. Pure and time-injected (`now_ms`) so it is unit-testable
/// without a clock.
#[derive(Debug)]
pub struct RestartTracker {
    policy: BackoffPolicy,
    /// Consecutive crash count driving the exponential delay; reset by a clean
    /// exit.
    consecutive: u32,
    /// Crash timestamps still inside the window.
    crashes: VecDeque<u64>,
}

impl RestartTracker {
    /// Create a tracker with the given backoff policy.
    #[must_use]
    pub fn new(policy: BackoffPolicy) -> Self {
        Self {
            policy,
            consecutive: 0,
            crashes: VecDeque::new(),
        }
    }

    /// Record a child exit at `now_ms` and decide what to do next.
    pub fn on_exit(&mut self, exit_code: i32, now_ms: u64) -> RestartAction {
        match restart_policy(exit_code) {
            Restart::OnDemand => {
                self.consecutive = 0;
                RestartAction::OnDemand
            }
            Restart::Never => RestartAction::Never,
            Restart::Backoff => {
                let cutoff = now_ms.saturating_sub(self.policy.window_ms);
                while self.crashes.front().is_some_and(|&t| t < cutoff) {
                    let _ = self.crashes.pop_front();
                }
                self.crashes.push_back(now_ms);
                if self.crashes.len() > self.policy.budget {
                    return RestartAction::Unhealthy;
                }
                // 2^consecutive * base, saturating and capped at max.
                let shift = self.consecutive.min(16);
                let delay = self
                    .policy
                    .base_ms
                    .saturating_mul(1u64 << shift)
                    .min(self.policy.max_ms);
                self.consecutive = self.consecutive.saturating_add(1);
                RestartAction::Backoff(delay)
            }
        }
    }
}

/// Tracks per-miner children and tokens.
pub struct Supervisor {
    /// Live miner child processes keyed by miner id.
    pub children: HashMap<String, Child>,
    /// Session tokens issued to each miner id.
    pub tokens: HashMap<String, String>,
    /// Unix-domain socket path miners connect to.
    pub sock: String,
}

impl Supervisor {
    /// Create a supervisor bound to the given UDS path.
    #[must_use]
    pub fn new(sock: impl Into<String>) -> Self {
        Self {
            children: HashMap::new(),
            tokens: HashMap::new(),
            sock: sock.into(),
        }
    }

    /// Spawn a miner binary with `QUIP_SESSION_TOKEN` (never argv).
    ///
    /// # Errors
    /// Returns any I/O error from spawning the child process.
    #[expect(
        clippy::unused_async,
        reason = "public API kept async for call-site uniformity with other supervisor methods"
    )]
    pub async fn spawn(
        &mut self,
        entry: &LaunchEntry,
        token: &str,
        sock_uri: &str,
    ) -> std::io::Result<()> {
        let child = Command::new(&entry.binary)
            .arg("--quip-coordinator")
            .arg(sock_uri)
            .arg("--miner-id")
            .arg(&entry.miner_id)
            .env("QUIP_SESSION_TOKEN", token)
            .kill_on_drop(true)
            .spawn()?;
        let _ = self
            .tokens
            .insert(entry.miner_id.clone(), token.to_string());
        let _ = self.children.insert(entry.miner_id.clone(), child);
        Ok(())
    }

    /// Spawn with a freshly generated token; returns the token.
    ///
    /// # Errors
    /// Propagates spawn I/O errors from [`Self::spawn`].
    pub async fn spawn_with_new_token(
        &mut self,
        entry: &LaunchEntry,
        sock_uri: &str,
    ) -> std::io::Result<String> {
        let token = gen_session_token();
        self.spawn(entry, &token, sock_uri).await?;
        Ok(token)
    }

    /// Kill all children after a grace period (in-band Shutdown is preferred
    /// when a live session channel is available).
    pub async fn shutdown_all(&mut self, grace_ms: u32) {
        tokio::time::sleep(Duration::from_millis(u64::from(grace_ms))).await;
        for child in self.children.values_mut() {
            let _ = child.kill().await;
        }
        for child in self.children.values_mut() {
            let _ = child.wait().await;
        }
        self.children.clear();
    }
}

/// Spawn a miner child with `QUIP_SESSION_TOKEN` (never argv). Stdout and stderr
/// inherit the coordinator's streams so miner log lines appear without a
/// re-emit pump. Pass `--log-level` so the child's own subscriber matches the
/// coordinator verbosity (the coordinator filter no longer sees child bytes).
fn spawn_supervised_child(
    entry: &LaunchEntry,
    token: &str,
    sock_uri: &str,
    log_level: LogLevel,
) -> std::io::Result<Child> {
    Command::new(&entry.binary)
        .arg("--quip-coordinator")
        .arg(sock_uri)
        .arg("--miner-id")
        .arg(&entry.miner_id)
        .arg("--log-level")
        .arg(log_level.to_string())
        .env("QUIP_SESSION_TOKEN", token)
        .kill_on_drop(true)
        .spawn()
}

/// Supervise one miner for the run: (re)spawn per the exit-code policy with
/// backoff + failure budget, re-queue its in-flight + staged jobs to the router
/// on a crash, and send an in-band `Shutdown` (then kill after grace) once
/// `stop` flips true.
pub async fn supervise_miner(
    entry: LaunchEntry,
    sock_uri: String,
    state: Arc<Mutex<CoordinatorState>>,
    policy: BackoffPolicy,
    grace_ms: u32,
    log_level: LogLevel,
    mut stop: watch::Receiver<bool>,
) {
    let mut tracker = RestartTracker::new(policy);
    let start = tokio::time::Instant::now();

    loop {
        // A fresh token per spawn; the session server must expect it.
        let token = gen_session_token();
        {
            let mut st = state.lock().await;
            let _ = st
                .expected_tokens
                .insert(entry.miner_id.clone(), token.clone());
        }

        let mut child = match spawn_supervised_child(&entry, &token, &sock_uri, log_level) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(miner = %entry.miner_id, binary = %entry.binary, error = %e, "miner spawn failed");
                break;
            }
        };
        tracing::info!(
            miner = %entry.miner_id,
            binary = %entry.binary,
            pid = %crate::logging::display_option(child.id()),
            "miner spawned"
        );

        // Run until the child exits on its own or shutdown is requested.
        let exited = tokio::select! {
            status = child.wait() => Some(status),
            _ = stop.changed() => None,
        };

        let Some(status) = exited else {
            // Graceful shutdown: in-band Shutdown, wait grace, then ensure dead.
            let tx = state.lock().await.outbound.get(&entry.miner_id).cloned();
            if let Some(tx) = tx {
                let _ = tx.send(Ok(shutdown_msg(grace_ms))).await;
            }
            let _ = tokio::time::timeout(
                Duration::from_millis(u64::from(grace_ms) + 500),
                child.wait(),
            )
            .await;
            let _ = child.kill().await;
            let _ = child.wait().await;
            break;
        };

        let code = status.ok().and_then(|s| s.code()).unwrap_or(-1);
        // A miner dying is never routine: exit 0 is a clean stop, anything else
        // cost this coordinator a backend.
        if code == 0 {
            tracing::info!(miner = %entry.miner_id, "miner exited cleanly");
        } else {
            tracing::warn!(miner = %entry.miner_id, code, "miner exited");
        }

        // Self-exit (crash or clean): return this miner's in-flight + staged
        // jobs to the router. A clean exit (code 0) can still leave in-flight
        // jobs — the Shutdown drain has a grace timeout — so reclaim regardless
        // of code, or `inflight` leaks and any mempool order is silently lost.
        {
            let mut st = state.lock().await;
            let jobs = st.reclaim_miner(&entry.miner_id);
            for job in jobs {
                let _ = st.router.route(job);
            }
        }

        #[expect(
            clippy::cast_possible_truncation,
            reason = "elapsed ms since process start fits u64 for supervisor lifetime"
        )]
        let now_ms = start.elapsed().as_millis() as u64;
        match tracker.on_exit(code, now_ms) {
            RestartAction::OnDemand => {
                // Guard against a busy respawn loop on an immediate clean exit;
                // the live loop (quip-agm) refines this to "respawn only when
                // jobs exist". Honour stop during the wait.
                tokio::select! {
                    () = tokio::time::sleep(Duration::from_millis(policy.base_ms)) => {}
                    _ = stop.changed() => break,
                }
            }
            RestartAction::Backoff(delay) => {
                tracing::info!(miner = %entry.miner_id, delay_ms = delay, "restarting miner after backoff");
                tokio::select! {
                    () = tokio::time::sleep(Duration::from_millis(delay)) => {}
                    _ = stop.changed() => break,
                }
            }
            RestartAction::Never | RestartAction::Unhealthy => {
                tracing::warn!(miner = %entry.miner_id, code, "miner not restarting");
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restart_policy_matches_exit_codes() {
        assert!(matches!(restart_policy(0), Restart::OnDemand));
        assert!(matches!(restart_policy(64), Restart::Never));
        assert!(matches!(restart_policy(69), Restart::Never));
        assert!(matches!(restart_policy(77), Restart::Never));
        assert!(matches!(restart_policy(70), Restart::Backoff));
        assert!(matches!(restart_policy(-9), Restart::Backoff)); // SIGKILL
    }

    fn tracker() -> RestartTracker {
        RestartTracker::new(BackoffPolicy {
            base_ms: 500,
            max_ms: 4_000,
            budget: 3,
            window_ms: 10_000,
        })
    }

    #[test]
    fn clean_exit_is_on_demand_and_resets_backoff() {
        let mut t = tracker();
        assert_eq!(t.on_exit(70, 0), RestartAction::Backoff(500));
        assert_eq!(t.on_exit(70, 100), RestartAction::Backoff(1000));
        // A clean exit clears the consecutive-crash count…
        assert_eq!(t.on_exit(0, 200), RestartAction::OnDemand);
        // …so the next crash's delay starts from base again.
        assert_eq!(t.on_exit(70, 300), RestartAction::Backoff(500));
    }

    #[test]
    fn operator_error_never_restarts() {
        let mut t = tracker();
        assert_eq!(t.on_exit(64, 0), RestartAction::Never);
        assert_eq!(t.on_exit(69, 0), RestartAction::Never);
        assert_eq!(t.on_exit(77, 0), RestartAction::Never);
    }

    #[test]
    fn backoff_doubles_and_caps_at_max() {
        let mut t = tracker(); // base 500, max 4000
        assert_eq!(t.on_exit(70, 0), RestartAction::Backoff(500));
        assert_eq!(t.on_exit(70, 1), RestartAction::Backoff(1000));
        assert_eq!(t.on_exit(70, 2), RestartAction::Backoff(2000));
        // budget is 3 crashes in window; the 4th trips Unhealthy before capping,
        // so widen the window via fresh trackers to exercise the cap directly.
        let mut t2 = RestartTracker::new(BackoffPolicy {
            base_ms: 500,
            max_ms: 4_000,
            budget: 100,
            window_ms: 1_000_000,
        });
        let delays: Vec<RestartAction> = (0..6).map(|i| t2.on_exit(70, i)).collect();
        assert_eq!(
            delays,
            vec![
                RestartAction::Backoff(500),
                RestartAction::Backoff(1000),
                RestartAction::Backoff(2000),
                RestartAction::Backoff(4000), // capped
                RestartAction::Backoff(4000),
                RestartAction::Backoff(4000),
            ]
        );
    }

    #[test]
    fn failure_budget_trips_unhealthy_then_window_slides() {
        let mut t = tracker(); // budget 3 within 10s
        assert!(matches!(t.on_exit(70, 0), RestartAction::Backoff(_)));
        assert!(matches!(t.on_exit(70, 1_000), RestartAction::Backoff(_)));
        assert!(matches!(t.on_exit(70, 2_000), RestartAction::Backoff(_)));
        // 4th crash inside the 10s window exhausts the budget.
        assert_eq!(t.on_exit(70, 3_000), RestartAction::Unhealthy);
        // Far in the future the window has slid clear; a crash is allowed again.
        assert!(matches!(t.on_exit(70, 100_000), RestartAction::Backoff(_)));
    }
}
