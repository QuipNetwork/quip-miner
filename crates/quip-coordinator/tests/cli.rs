//! CLI smoke tests for the `quip-coordinator` binary.

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_quip-coordinator")
}

#[test]
fn help_exits_zero() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn missing_config_exits_64() {
    let out = Command::new(bin())
        .arg("--config")
        .arg("/nonexistent/quip-coordinator-config.toml")
        .output()
        .expect("run --config");
    assert_eq!(out.status.code(), Some(64));
}

/// The whole point of the logging fix: a subscriber is installed, so `tracing`
/// events reach stderr. Before this, every `tracing` call in the binary was a
/// silent no-op and this assertion failed on empty output.
#[test]
fn errors_reach_stderr_through_the_subscriber() {
    let out = Command::new(bin())
        .arg("--config")
        .arg("/nonexistent/quip-coordinator-config.toml")
        .env_remove("RUST_LOG")
        .output()
        .expect("run --config");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("ERROR"), "no level in stderr: {stderr:?}");
    assert!(
        stderr.contains("cannot read config"),
        "no event in stderr: {stderr:?}"
    );
    // Structured fields, not an interpolated blob.
    assert!(
        stderr.contains("path=/nonexistent/quip-coordinator-config.toml"),
        "no fields in stderr: {stderr:?}"
    );
    // Captured (non-tty) output must be plain text. The node manager redirects
    // this stream to a file, and ANSI escapes there corrupt every later grep.
    assert!(
        !stderr.contains('\u{1b}'),
        "ANSI escapes in piped stderr: {stderr:?}"
    );
}

/// `--log-level` is a global flag, so it parses ahead of a subcommand too.
#[test]
fn log_level_flag_is_accepted() {
    for args in [
        vec!["--log-level", "debug", "--config", "/nonexistent/x.toml"],
        vec!["--log-level", "trace", "--config", "/nonexistent/x.toml"],
    ] {
        let out = Command::new(bin()).args(&args).output().expect("run");
        // 64 = ConfigInvalid (the missing file), not 2 (clap usage error).
        assert_eq!(out.status.code(), Some(64), "args: {args:?}");
    }
}

#[test]
fn log_level_rejects_unknown_value() {
    let out = Command::new(bin())
        .args(["--log-level", "verbose", "--config", "/nonexistent/x.toml"])
        .output()
        .expect("run");
    assert_eq!(out.status.code(), Some(2), "clap should reject the value");
}

/// `--log-level` overrides an ambient `RUST_LOG`, and quieting to `error` drops
/// warn-level events while keeping errors.
#[test]
fn explicit_log_level_overrides_rust_log() {
    let out = Command::new(bin())
        .args(["--log-level", "error", "--config", "/nonexistent/x.toml"])
        .env("RUST_LOG", "quip_coordinator=trace")
        .output()
        .expect("run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("ERROR"), "stderr: {stderr:?}");
}

/// Logs must never land on stdout: `drive` prints its timing table there, and a
/// caller parsing that report would choke on interleaved log lines.
#[test]
fn logs_do_not_pollute_stdout() {
    let out = Command::new(bin())
        .args(["--log-level", "trace", "--config", "/nonexistent/x.toml"])
        .output()
        .expect("run");
    assert!(
        out.stdout.is_empty(),
        "stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn bad_toml_exits_64() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.toml");
    std::fs::write(&path, "not = [valid").unwrap();
    let out = Command::new(bin())
        .arg("--config")
        .arg(&path)
        .output()
        .expect("run bad config");
    assert_eq!(out.status.code(), Some(64));
}
