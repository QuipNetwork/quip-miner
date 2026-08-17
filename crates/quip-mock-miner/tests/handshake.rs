//! CLI smoke tests for the mock miner binary (`--capabilities`, `--check`, exit codes).

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_quip-mock-miner")
}

#[test]
fn capabilities_prints_json_and_exits_zero() {
    let out = Command::new(bin()).arg("--capabilities").output().unwrap();
    assert!(out.status.success());
    let s = String::from_utf8(out.stdout).unwrap();
    assert!(
        s.contains("\"backend\""),
        "capabilities JSON must include backend, got: {s}"
    );
}

#[test]
fn version_exits_zero() {
    let out = Command::new(bin()).arg("--version").output().unwrap();
    assert!(out.status.success());
    assert!(String::from_utf8(out.stdout).unwrap().contains("protocol"));
}

#[test]
fn check_exits_zero_on_this_host() {
    assert!(Command::new(bin())
        .arg("--check")
        .status()
        .unwrap()
        .success());
}

#[test]
fn missing_coordinator_exits_64_not_panic() {
    let out = Command::new(bin())
        .env("QUIP_SESSION_TOKEN", "tok")
        .output()
        .unwrap();
    // No --quip-coordinator and no --capabilities/--check → ConfigInvalid (64).
    assert_eq!(
        out.status.code(),
        Some(64),
        "missing --quip-coordinator must exit 64 (got {:?}, stderr={})",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn missing_session_token_exits_77() {
    let out = Command::new(bin())
        .arg("--quip-coordinator")
        .arg("unix:///tmp/quip-no-such-socket.sock")
        .env_remove("QUIP_SESSION_TOKEN")
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(77),
        "missing QUIP_SESSION_TOKEN must exit 77 (got {:?}, stderr={})",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
}
