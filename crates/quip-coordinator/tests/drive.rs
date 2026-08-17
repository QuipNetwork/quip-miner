//! Integration: drive `quip-mock-miner` end-to-end over UDS with a 2-entry
//! JSONL list, via the generalized `JobSource`-driven harness.

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
use quip_coordinator::drive::{aggregate, drain_all, DriveManyParams, ListSource};
use quip_proto::v1::Configure;
use std::time::{SystemTime, UNIX_EPOCH};

/// Sibling `quip-mock-miner` binary (not same package → no `CARGO_BIN_EXE_*`).
fn mock_miner() -> String {
    let status = std::process::Command::new(env!("CARGO"))
        .args(["build", "-p", "quip-mock-miner"])
        .status()
        .expect("build quip-mock-miner");
    assert!(status.success(), "failed to build quip-mock-miner");
    let name = if cfg!(windows) {
        "quip-mock-miner.exe"
    } else {
        "quip-mock-miner"
    };
    let mut p = std::env::current_exe().expect("test exe path");
    let _ = p.pop();
    let _ = p.pop();
    p.push(name);
    p.to_string_lossy().into_owned()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[tokio::test]
async fn drives_two_entry_list_end_to_end() {
    let miner = mock_miner();
    let deadline = now_ms() + 3_600_000;

    // Two explicit entries; mock-miner always returns an all-+1 solution.
    let list_text = "{\"h_milli\":[1000,-1000],\"j_milli\":[500],\"edges\":[[0,1]]}\n\
                      {\"h_milli\":[0,0,0],\"j_milli\":[100,100],\"edges\":[[0,1],[1,2]]}\n";
    let mut src = ListSource::parse(list_text, None, deadline).expect("parse list");
    let jobs = drain_all(&mut src);
    assert_eq!(jobs.len(), 2, "expected exactly 2 jobs from the list");

    let entry = LaunchEntry {
        miner_id: "cpu-0".into(),
        binary: miner.clone(),
        backend: "cpu".into(),
        configure: Configure {
            queue_depth: 3,
            idle_timeout_s: 30,
            heartbeat_s: 15,
            reconnect_window_s: 60,
            backend_toml: String::new(),
        },
    };

    let sock = format!("/tmp/quip-drive-e2e-{}.sock", std::process::id());
    let token = "drive-e2e-token";
    let report = quip_coordinator::drive::run_drive(DriveManyParams {
        miner_bin: &miner,
        sock_path: &sock,
        miner_id: "cpu-0",
        token,
        entry: &entry,
        topology: None,
        target: None,
        jobs,
        utilization: None,
        yielding: false,
        log_level: quip_coordinator::logging::LogLevel::Info,
    })
    .await;

    assert!(report.handshake_ok, "handshake failed");
    assert_eq!(
        report.rows.len(),
        2,
        "expected 2 result rows, got {:?}",
        report.rows
    );
    assert!(
        report.rows.iter().all(|r| r.passed && !r.rejected),
        "expected every job to pass with loose gates: {:?}",
        report.rows
    );

    let agg = aggregate(&report.rows, report.run_wall_ms);
    assert_eq!(agg.total_jobs, 2);
    assert_eq!(agg.passed, 2);
    assert_eq!(agg.rejected, 0);
    // A 2-job run against the mock miner can finish inside a single millisecond
    // now that job dispatch no longer shares the session's read path, and the
    // span is only measured to whole milliseconds — so 0 here means "too fast
    // to measure", not "nothing ran". The row assertions above already prove
    // both jobs completed. (`aggregate` reports 0 jobs/s for a sub-ms span; that
    // reporting floor predates this change and is not what this test covers.)
    if report.run_wall_ms > 0 {
        assert!(agg.throughput_per_s > 0.0);
    }
}
