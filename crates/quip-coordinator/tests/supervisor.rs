//! Supervisor restart-policy and reclaim integration tests.

use quip_coordinator::supervisor::{restart_policy, Restart};

#[test]
fn restart_policy_matches_exit_codes() {
    assert!(matches!(restart_policy(0), Restart::OnDemand));
    assert!(matches!(restart_policy(64), Restart::Never));
    assert!(matches!(restart_policy(69), Restart::Never));
    assert!(matches!(restart_policy(77), Restart::Never));
    assert!(matches!(restart_policy(70), Restart::Backoff));
    assert!(matches!(restart_policy(-9), Restart::Backoff));
}

/// A miner that exits cleanly (code 0) while it still owns in-flight jobs must
/// have those jobs reclaimed and re-routed — not leaked in the `inflight` map
/// and silently dropped. Regression for the crash-only reclaim guard (quip-akl).
#[cfg(unix)]
#[tokio::test]
async fn clean_exit_reclaims_and_reroutes_inflight() {
    use quip_coordinator::config::LaunchEntry;
    use quip_coordinator::router::MinerCaps;
    use quip_coordinator::session::CoordinatorState;
    use quip_coordinator::supervisor::{supervise_miner, BackoffPolicy};
    use quip_proto::v1::{Configure, IsingProblem, Job, JobKind, Provenance};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::{watch, Mutex};

    // A tiny job of a shape `cpu-1` can serve (2 nodes / 1 edge).
    fn job(job_id: &str, generation: u64, is_pow: bool) -> Job {
        Job {
            job_id: job_id.as_bytes().to_vec(),
            kind: JobKind::IsingSample as i32,
            generation,
            deadline_ms: 9_999_999,
            ising: Some(IsingProblem {
                graph: None,
                h_milli_le32: vec![0; 8], // 2 nodes
                j_milli_le32: vec![0; 4], // 1 edge
                num_reads: 0,
                num_sweeps: 0,
                anneal_time_us: 0,
            }),
            provenance: Some(Provenance {
                is_pow,
                order_id: if is_pow { vec![] } else { b"order-1".to_vec() },
            }),
        }
    }

    fn caps() -> MinerCaps {
        MinerCaps {
            backend: "mock".into(),
            algorithm: "sa".into(),
            supported_kinds: vec![JobKind::IsingSample as i32],
            max_nodes: 1000,
            max_edges: 10000,
        }
    }

    // `/usr/bin/true` (or `/bin/true`) exits 0 immediately, ignoring argv/env —
    // a stand-in for a miner that self-exits cleanly.
    let true_bin = ["/usr/bin/true", "/bin/true"]
        .into_iter()
        .find(|p| std::path::Path::new(p).exists())
        .expect("a `true` binary");

    let state = Arc::new(Mutex::new(CoordinatorState::new()));
    {
        let mut st = state.lock().await;
        // A second, capable miner the reclaimed jobs can re-route onto.
        st.router.register_miner("cpu-1", caps());
        // cpu-0 dispatched two jobs and then exits without completing them.
        st.dispatch_inflight("cpu-0", job("pow-1", 1, true));
        st.dispatch_inflight("cpu-0", job("mempool-1", 0, false));
        assert_eq!(st.inflight.len(), 2);
    }

    let entry = LaunchEntry {
        miner_id: "cpu-0".into(),
        binary: true_bin.to_string(),
        backend: "cpu".into(),
        configure: Configure::default(),
    };
    // Large base_ms: after the code-0 exit the supervisor parks in the OnDemand
    // wait, so the reclaim has happened and state is stable while we poll.
    let policy = BackoffPolicy {
        base_ms: 60_000,
        max_ms: 60_000,
        budget: 100,
        window_ms: 60_000,
    };
    let (stop_tx, stop_rx) = watch::channel(false);
    let handle = tokio::spawn(supervise_miner(
        entry,
        "unix:///nonexistent.sock".into(),
        Arc::clone(&state),
        policy,
        200,
        quip_coordinator::logging::LogLevel::Info,
        stop_rx,
    ));

    // Condition-poll until inflight clears (fix present) rather than sleeping a
    // fixed time. Without the fix it never clears and this times out.
    let mut cleared = false;
    for _ in 0..250 {
        if state.lock().await.inflight.is_empty() {
            cleared = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    {
        let st = state.lock().await;
        assert!(
            cleared,
            "inflight not reclaimed after clean (code 0) exit: {} left",
            st.inflight.len()
        );
        assert!(st.inflight_owner.is_empty(), "inflight_owner leaked");
        // Both jobs re-routed onto the surviving capable miner.
        assert_eq!(st.router.staged_len("cpu-1"), 2, "jobs not re-routed");
    }

    let _ = stop_tx.send(true);
    let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;
}
