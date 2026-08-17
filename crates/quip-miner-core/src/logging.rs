//! Log subscriber setup for miner binaries.
//!
//! The `tracing` macros are a facade. Without an installed subscriber they
//! compile to no-ops, so a miner that never calls [`init`] emits nothing at any
//! `--log-level` or `RUST_LOG` setting — including its fatal errors. [`init`]
//! is that call, and every miner entry point makes it through
//! [`crate::session::run`].
//!
//! Output goes to **stderr**, which is where the coordinator that started the
//! miner collects it. Stdout is reserved for `--capabilities`, which a caller
//! parses as JSON.

use std::io::IsTerminal as _;
use tracing_subscriber::EnvFilter;

/// Verbosity words accepted by `--log-level`, matching the set the coordinator
/// offers and the v0.2.1 miner CLI accepted.
const LEVELS: [&str; 5] = ["trace", "debug", "info", "warn", "error"];

/// Crates whose internals stay at `warn` under the default filter.
///
/// The transport stack is chatty at `debug` and below, and its output buries
/// the per-job lines that a miner is normally run at `debug` to see. `RUST_LOG`
/// is the escape hatch when the transport is what you actually need.
const NOISY_TARGETS: [&str; 5] = ["tonic", "tower", "hyper", "hyper_util", "h2"];

/// Build the default filter: everything at `level`, transport crates at `warn`.
fn default_filter(level: &str) -> String {
    let mut s = String::from(level);
    for target in NOISY_TARGETS {
        s.push(',');
        s.push_str(target);
        s.push_str("=warn");
    }
    s
}

/// Resolve the filter directives to install.
///
/// Precedence, highest first:
/// 1. `RUST_LOG`, when set and non-empty. A miner is started by the
///    coordinator, which owns the argument vector, so the environment is the
///    only channel an operator can use to raise verbosity on a running node.
/// 2. `--log-level`, which the coordinator passes through to each miner.
///
/// `level` is checked against [`LEVELS`] even when `RUST_LOG` overrides it.
/// `EnvFilter` reads a bare word it does not recognize as a *target* name, so
/// `--log-level infoo` would otherwise parse cleanly and silence the miner
/// completely. A miner that prints nothing is the hardest state to diagnose, so
/// a misspelled level must fail at startup instead.
fn resolve_directives(level: &str, rust_log: Option<&str>) -> Result<String, String> {
    if !LEVELS.contains(&level) {
        return Err(format!(
            "unknown --log-level {level:?}: expected one of {}",
            LEVELS.join(", ")
        ));
    }
    Ok(match rust_log {
        Some(s) if !s.trim().is_empty() => s.to_string(),
        _ => default_filter(level),
    })
}

/// Install the global log subscriber, writing to stderr.
///
/// Call once, as early in the entry point as possible: messages emitted before
/// this returns are discarded.
///
/// # Errors
/// Returns an error when `level` is not a known verbosity, or when the resolved
/// directives do not parse. An already installed subscriber is not an error: a
/// miner linked into a host that set up its own subscriber keeps that one.
pub fn init(level: &str) -> Result<(), String> {
    let rust_log = std::env::var("RUST_LOG").ok();
    let directives = resolve_directives(level, rust_log.as_deref())?;
    let filter = EnvFilter::try_new(&directives)
        .map_err(|e| format!("invalid log filter {directives:?}: {e}"))?;
    // Color only a real terminal. A supervised miner has its stderr collected
    // into a log file, and escape sequences baked into that capture corrupt
    // every downstream grep.
    let ansi = std::io::stderr().is_terminal();
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(ansi)
        .try_init();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_log_beats_the_level_flag() {
        let d = resolve_directives("info", Some("quip_miner_core=trace")).unwrap();
        assert_eq!(d, "quip_miner_core=trace");
    }

    #[test]
    fn blank_rust_log_falls_back_to_the_level_flag() {
        assert_eq!(
            resolve_directives("debug", Some("   ")).unwrap(),
            default_filter("debug")
        );
        assert_eq!(
            resolve_directives("debug", None).unwrap(),
            default_filter("debug")
        );
    }

    #[test]
    fn default_filter_keeps_the_transport_quiet() {
        let d = default_filter("debug");
        assert!(d.starts_with("debug,"), "{d}");
        assert!(d.contains("tonic=warn"), "{d}");
        assert!(d.contains("h2=warn"), "{d}");
    }

    #[test]
    fn every_level_the_cli_accepts_parses_as_directives() {
        for level in ["trace", "debug", "info", "warn", "error"] {
            let d = default_filter(level);
            assert!(EnvFilter::try_new(&d).is_ok(), "{d}");
        }
    }

    /// A misspelled level must not degrade into a target filter that silences
    /// the miner. `EnvFilter` accepts any bare word as a target, so this only
    /// fails because [`resolve_directives`] checks the level itself.
    #[test]
    fn a_misspelled_level_is_rejected_rather_than_silencing_the_miner() {
        let e = resolve_directives("infoo", None).unwrap_err();
        assert!(e.contains("infoo"), "{e}");
        assert!(EnvFilter::try_new(default_filter("infoo")).is_ok());
    }

    /// `RUST_LOG` does not excuse a bad level: the operator still typed it
    /// wrong, and reporting it is cheaper than a silent surprise on the next
    /// run without `RUST_LOG` set.
    #[test]
    fn a_misspelled_level_is_rejected_even_when_rust_log_overrides_it() {
        assert!(resolve_directives("infoo", Some("trace")).is_err());
    }
}
