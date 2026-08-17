//! Regression: a deep credit pool must not deadlock the miner session.
//!
//! Both peers speak one bidi stream. If either one writes on the same task that
//! reads, a full outbound channel stops it reading — and since each side's
//! reads are what free the other side's writes, both park permanently. The
//! trigger is simply enough bytes in flight: a large credit pool plus jobs
//! carrying inline `h`/`J`.
//!
//! This reproduces it through the public drive path. Pre-fix (dispatch on the
//! coordinator's read loop) the run hangs and this test trips its timeout;
//! post-fix the dispatcher is a separate task, so results keep draining.

#![expect(
    clippy::expect_used,
    reason = "helper builds mock miner outside #[test]"
)]
#![expect(clippy::unwrap_used, reason = "now_ms helper outside #[test]")]
#![expect(
    clippy::cast_possible_truncation,
    reason = "test wall-clock millis fit u64"
)]

use quip_coordinator::config::LaunchEntry;
use quip_coordinator::drive::{DriveManyParams, JobRow};
use quip_proto::v1::{ising_problem, Configure, EdgeList, IsingProblem, Job, JobKind};
use quip_protocol::wire::encode_i32_le;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Sibling `quip-mock-miner` binary (not same package → no `CARGO_BIN_EXE_*`).
fn mock_miner() -> String {
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-mock-miner"])
        .status()
        .expect("build quip-mock-miner");
    assert!(status.success(), "failed to build quip-mock-miner");
    let mut p = std::env::current_exe().expect("test exe path");
    let _ = p.pop();
    let _ = p.pop();
    p.push("quip-mock-miner");
    p.to_string_lossy().into_owned()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Jobs shaped like production nonces: an Advantage2-sized ring (4577 nodes)
/// carries ~73 KB of inline `h`/`J`/edges, so a 480-deep pool puts ~35 MB on
/// the wire. Volume is the whole trigger — at 5 KB per job the transport
/// buffers absorb the burst and the deadlock does not appear.
fn ring_jobs(count: usize, nodes: usize) -> Vec<Job> {
    let deadline = now_ms() + 3_600_000;
    let h: Vec<i32> = (0..nodes)
        .map(|i| if i % 2 == 0 { 1000 } else { -1000 })
        .collect();
    let u: Vec<u32> = (0..nodes as u32).collect();
    let v: Vec<u32> = (0..nodes as u32).map(|i| (i + 1) % nodes as u32).collect();
    let j: Vec<i32> = vec![500; nodes];
    (0..count)
        .map(|n| Job {
            job_id: format!("job-{n}").into_bytes(),
            kind: JobKind::IsingSample as i32,
            generation: 0,
            deadline_ms: deadline,
            ising: Some(IsingProblem {
                graph: Some(ising_problem::Graph::Edges(EdgeList {
                    u: u.clone(),
                    v: v.clone(),
                })),
                h_milli_le32: encode_i32_le(&h),
                j_milli_le32: encode_i32_le(&j),
                num_reads: 1,
                num_sweeps: 1,
                anneal_time_us: 0,
            }),
            provenance: None,
        })
        .collect()
}

#[tokio::test]
async fn deep_credit_pool_does_not_deadlock_the_session() {
    let miner = mock_miner();
    let jobs = ring_jobs(480, 4577);
    let total = jobs.len();

    let entry = LaunchEntry {
        miner_id: "cpu-0".into(),
        binary: miner.clone(),
        backend: "cpu".into(),
        configure: Configure {
            // The lever: `seed_credits` is `queue_depth`, so this grants the
            // whole pool up front and the coordinator dispatches every staged
            // job in one uninterrupted burst — exactly the flood that used to
            // wedge the read path.
            queue_depth: 480,
            idle_timeout_s: 30,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    };

    let sock = format!("/tmp/quip-flow-control-{}.sock", std::process::id());
    let run = quip_coordinator::drive::run_drive(DriveManyParams {
        miner_bin: &miner,
        sock_path: &sock,
        miner_id: "cpu-0",
        token: "flow-control-token",
        entry: &entry,
        topology: None,
        target: None,
        jobs,
        utilization: None,
        yielding: false,
        log_level: quip_coordinator::logging::LogLevel::Info,
    });

    // A generous bound: the mock miner answers instantly, so this only fires if
    // the session wedged. Without the timeout a regression hangs the suite
    // instead of failing it.
    let report = tokio::time::timeout(Duration::from_secs(90), run)
        .await
        .expect("session deadlocked: no progress within 90s");

    assert!(report.handshake_ok, "handshake failed");
    assert_eq!(
        report.rows.len(),
        total,
        "every job must reach a terminal outcome; got {} of {total}",
        report.rows.len()
    );
    let rejected = report.rows.iter().filter(|r: &&JobRow| r.rejected).count();
    assert_eq!(rejected, 0, "no job should be rejected by the mock miner");
}
