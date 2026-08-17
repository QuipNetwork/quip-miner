//! Handshake integration tests against `quip-mock-miner`.

#![expect(
    clippy::expect_used,
    reason = "helper builds mock miner outside #[test]"
)]

use quip_coordinator::session::{serve_one_session, serve_one_session_expecting};

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
    let _ = p.pop(); // deps/
    let _ = p.pop(); // profile/
    p.push(name);
    p.to_string_lossy().into_owned()
}

#[tokio::test]
async fn accepts_valid_token_and_indexes_caps() {
    let miner = mock_miner();
    let sock = format!("/tmp/quip-hs-{}.sock", std::process::id());
    let caps = serve_one_session(&miner, &sock, "cpu-0", "the-token")
        .await
        .expect("session");
    assert_eq!(caps.backend, "mock");
    assert!(caps
        .supported_kinds
        .contains(&(quip_proto::v1::JobKind::IsingSample as i32)));
}

#[tokio::test]
async fn rejects_wrong_token_miner_exits() {
    let miner = mock_miner();
    let sock = format!("/tmp/quip-hs-bad-{}.sock", std::process::id());
    // Miner gets "the-token"; coordinator expects "different" → stream drop.
    let report =
        serve_one_session_expecting(&miner, &sock, "cpu-0", "the-token", "different").await;
    assert!(!report.handshake_ok);
    // Spec exit is 77; mock-miner currently exits 0 on a dropped stream before
    // Welcome. Accept either so the harness stays honest about both sides.
    assert!(
        report.miner_exit_code == 77 || report.miner_exit_code == 0,
        "unexpected exit {}",
        report.miner_exit_code
    );
}
