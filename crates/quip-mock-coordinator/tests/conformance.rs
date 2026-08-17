//! Integration tests: drive `quip-mock-miner` through the mock coordinator's
//! scripted conformance session.

use quip_mock_coordinator::driver::{drive_miner, drive_miner_bad_welcome};
use quip_proto::v1::RejectReason;

/// Path to the sibling `quip-mock-miner` binary.
///
/// `CARGO_BIN_EXE_*` is only defined for binaries in the *same* package, so a
/// cross-package test derives the path from its own executable location
/// (`target/<profile>/deps/<test>` -> `target/<profile>/quip-mock-miner`).
///
/// A bare `cargo test -p quip-mock-coordinator` does not (re)build the sibling
/// miner crate, so build it here first to guarantee the binary matches current
/// source rather than a stale artifact. The nested `cargo build` is a no-op when
/// already current, and the outer test run has released the build lock by the
/// time tests execute.
fn miner_bin() -> String {
    #[expect(
        clippy::expect_used,
        reason = "integration test panics if nested cargo build fails to start"
    )]
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
    #[expect(
        clippy::expect_used,
        reason = "integration test panics if test binary path cannot be resolved"
    )]
    let mut p = std::env::current_exe().expect("test exe path");
    let _ = p.pop(); // deps/
    let _ = p.pop(); // <profile>/
    p.push(name);
    p.to_string_lossy().into_owned()
}

/// A unique `unix://` socket path for one test's mock coordinator.
fn unique_socket(label: &str) -> String {
    format!(
        "unix:///tmp/quip-conf-{label}-{}-{}.sock",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    )
}

#[tokio::test]
async fn mock_miner_passes_conformance() {
    let miner = miner_bin();
    let socket = unique_socket("full");
    let report = drive_miner(&miner, &socket).await;

    assert!(report.handshake_ok, "handshake failed: {report:?}");
    assert!(
        report.ready_received,
        "Ready must arrive after Configure: {report:?}"
    );
    assert!(
        !report.job_request_credits.is_empty(),
        "expected at least one JobRequest{{credits}}: {report:?}"
    );
    assert!(
        report.job_request_credits.iter().all(|&c| c > 0),
        "JobRequest credits must be > 0: {report:?}"
    );
    assert_eq!(
        report.results.len(),
        3,
        "expected 3 Results (job-1, job-2, job-hash): {report:?}"
    );
    assert!(
        report.results.iter().any(|r| r.job_id == b"job-1"),
        "missing Result for job-1: {report:?}"
    );
    assert!(
        report.results.iter().any(|r| r.job_id == b"job-2"),
        "missing Result for job-2: {report:?}"
    );
    assert!(
        report.results.iter().any(|r| r.job_id == b"job-hash"),
        "missing Result for topology-hash job-hash: {report:?}"
    );
    for r in &report.results {
        assert!(
            !r.solution_energies_milli.is_empty(),
            "Result for job {:?} has no solutions (no energy_milli): {report:?}",
            r.job_id
        );
        assert!(
            r.meta_present,
            "Result for job {:?} is missing SamplerMeta: {report:?}",
            r.job_id
        );
    }
    assert!(
        report.has_reject(b"job-bad-h", RejectReason::Malformed),
        "missing per-job_id Reject MALFORMED for job-bad-h: {report:?}"
    );
    assert!(
        report.has_reject(b"job-bad-j", RejectReason::Malformed),
        "missing per-job_id Reject MALFORMED for job-bad-j: {report:?}"
    );
    assert!(
        report.has_reject(b"job-gate", RejectReason::UnsupportedKind),
        "missing per-job_id Reject UNSUPPORTED_KIND for job-gate: {report:?}"
    );
    assert!(
        report.has_reject(b"job-old", RejectReason::Expired),
        "missing per-job_id Reject EXPIRED for job-old: {report:?}"
    );
    assert!(
        report.cancel_acked,
        "Cancel must be acknowledged via Status: {report:?}"
    );
    assert_eq!(report.exit_code, 0, "clean shutdown expected: {report:?}");
    assert!(
        report.is_conformant(),
        "full DriverReport must pass: {report:?}"
    );
}

/// A conformant miner must reject an unsupported `Welcome.protocol_version`
/// cleanly: emit `Fatal` (`exit_code=ConfigInvalid=64`) and exit with that same
/// code, rather than proceeding to `Configure`.
#[tokio::test]
async fn mock_miner_rejects_bad_welcome() {
    let miner = miner_bin();
    let socket = unique_socket("bad-welcome");
    let report = drive_miner_bad_welcome(&miner, &socket).await;

    assert!(report.handshake_ok, "handshake failed: {report:?}");
    assert_eq!(
        report.exit_code, 64,
        "expected ConfigInvalid (64) exit on bad Welcome: {report:?}"
    );
    let (fatal_code, fatal_reason) = report
        .fatal
        .as_ref()
        .unwrap_or_else(|| panic!("miner must send Fatal on bad Welcome: {report:?}"));
    assert_eq!(
        *fatal_code, 64,
        "Fatal.exit_code must match ConfigInvalid: {report:?}"
    );
    assert!(
        !fatal_reason.is_empty(),
        "Fatal.reason must explain the rejection: {report:?}"
    );
    // A miner that got this far must not have proceeded into the normal
    // job-handling flow.
    assert!(
        !report.ready_received,
        "miner must not send Ready after a rejected Welcome: {report:?}"
    );
    assert!(
        report.results.is_empty(),
        "miner must not process jobs after a rejected Welcome: {report:?}"
    );
}
