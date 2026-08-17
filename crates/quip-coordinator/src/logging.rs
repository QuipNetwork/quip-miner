//! Log subscriber setup for the `quip-coordinator` binary.
//!
//! Every operational message in this crate goes through `tracing`. The
//! `tracing` macros are a facade: without an installed subscriber they compile
//! to no-ops, so a binary that never calls [`init`] emits nothing at any
//! `RUST_LOG` setting. [`init`] is that call.
//!
//! Output goes to **stderr**, never stdout: `drive` mode prints its timing
//! table to stdout, and log lines interleaved into it would corrupt a report a
//! caller is parsing.

use std::fmt;
use std::io::IsTerminal as _;
use tracing_subscriber::EnvFilter;

/// Floor-divide a milli-energy to whole units for an operator log.
///
/// Rust `/` truncates toward zero. Negative energies would then look better
/// than they are. Euclid division floors.
#[must_use]
pub(crate) fn energy_units(milli: i64) -> i64 {
    milli.div_euclid(1000)
}

/// Render an optional value as the bare value, or `none` when empty.
///
/// Operator logs must not print `Some(...)` or `None`.
#[must_use]
pub(crate) fn display_option<T: fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(|| "none".to_owned(), |v| v.to_string())
}

/// Render an optional milli-energy as whole units, or `none`.
#[must_use]
pub(crate) fn display_energy(milli: Option<i64>) -> String {
    display_option(milli.map(energy_units))
}

/// Verbosity for the default filter, mirroring the `--log-level` values the
/// v0.2.1 miner CLI accepted.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogLevel {
    /// Everything, including per-job and per-poll detail.
    Trace,
    /// Per-round and per-miner detail useful when diagnosing a stall.
    Debug,
    /// Operational narration: startup, rounds, staging, submits. The default.
    Info,
    /// Only degraded conditions.
    Warn,
    /// Only failures.
    Error,
}

impl LogLevel {
    /// The `tracing` directive text for this level.
    const fn as_str(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warn",
            Self::Error => "error",
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Targets whose events are the coordinator's own operational narration. Third
/// party crates (jsonrpsee, tonic, hyper) stay at `warn` under the default
/// filter so `--log-level debug` stays readable; `RUST_LOG` is the escape hatch
/// when their internals are what you actually need. Miner child processes
/// inherit the coordinator stdio and emit under their own subscriber, so they
/// are not listed here.
const OWN_TARGETS: [&str; 3] = ["quip_coordinator", "quip_protocol", "quip_miner_core"];

/// Build the default filter for `level`: third-party crates at `warn`, this
/// coordinator's own targets at `level`.
fn default_filter(level: LogLevel) -> String {
    let mut s = String::from("warn");
    for target in OWN_TARGETS {
        s.push(',');
        s.push_str(target);
        s.push('=');
        s.push_str(level.as_str());
    }
    s
}

/// Resolve the filter directives to install.
///
/// Precedence, highest first:
/// 1. An explicit `--log-level` (`level` is `Some`) — the operator asked for a
///    specific verbosity on this run.
/// 2. `RUST_LOG`, when set and non-empty.
/// 3. [`LogLevel::Info`], matching the v0.2.1 CLI default.
fn resolve_directives(level: Option<LogLevel>, rust_log: Option<&str>) -> String {
    if let Some(level) = level {
        return default_filter(level);
    }
    match rust_log {
        Some(s) if !s.trim().is_empty() => s.to_string(),
        _ => default_filter(LogLevel::Info),
    }
}

/// Install the global log subscriber, writing to stderr.
///
/// Call once, as early in `main` as possible: messages emitted before this
/// returns are discarded.
///
/// # Errors
/// Returns an error when the resolved directives do not parse or a global
/// subscriber is already installed.
pub fn init(level: Option<LogLevel>) -> Result<(), String> {
    let rust_log = std::env::var("RUST_LOG").ok();
    let directives = resolve_directives(level, rust_log.as_deref());
    let filter = EnvFilter::try_new(&directives)
        .map_err(|e| format!("invalid log filter {directives:?}: {e}"))?;
    // Color only a real terminal. A coordinator normally runs supervised, with
    // stderr captured to a file or journal by whatever started it (the node
    // manager does exactly this), and escape sequences baked into that capture
    // corrupt every downstream grep.
    let ansi = std::io::stderr().is_terminal();
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .with_target(true)
        .with_ansi(ansi)
        .try_init()
        .map_err(|e| format!("cannot install log subscriber: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr as _;

    #[test]
    fn explicit_level_beats_rust_log() {
        let d = resolve_directives(Some(LogLevel::Debug), Some("trace"));
        assert_eq!(d, default_filter(LogLevel::Debug));
    }

    #[test]
    fn rust_log_used_when_no_explicit_level() {
        let d = resolve_directives(None, Some("quip_coordinator=trace"));
        assert_eq!(d, "quip_coordinator=trace");
    }

    #[test]
    fn blank_rust_log_falls_back_to_info_default() {
        assert_eq!(
            resolve_directives(None, Some("   ")),
            default_filter(LogLevel::Info)
        );
        assert_eq!(
            resolve_directives(None, None),
            default_filter(LogLevel::Info)
        );
    }

    #[test]
    fn default_filter_keeps_third_party_at_warn() {
        let d = default_filter(LogLevel::Debug);
        assert!(d.starts_with("warn,"), "{d}");
        assert!(d.contains("quip_coordinator=debug"), "{d}");
        assert!(d.contains("quip_protocol=debug"), "{d}");
        assert!(d.contains("quip_miner_core=debug"), "{d}");
        // Miner children write straight to the terminal; no re-emit target.
        assert!(!d.contains("miner="), "{d}");
    }

    #[test]
    fn every_default_filter_parses_as_directives() {
        for level in [
            LogLevel::Trace,
            LogLevel::Debug,
            LogLevel::Info,
            LogLevel::Warn,
            LogLevel::Error,
        ] {
            let d = default_filter(level);
            assert!(EnvFilter::try_new(&d).is_ok(), "{d}");
        }
    }

    #[test]
    fn rust_log_still_wins_over_the_default_filter() {
        let d = resolve_directives(None, Some("warn"));
        assert_eq!(d, "warn");
    }

    #[test]
    fn level_display_round_trips_to_directive_text() {
        assert_eq!(LogLevel::Warn.to_string(), "warn");
        assert!(tracing::Level::from_str(LogLevel::Trace.as_str()).is_ok());
    }

    #[test]
    fn energy_units_floors_the_spec_table() {
        assert_eq!(energy_units(-14_369_000), -14_369);
        assert_eq!(energy_units(-14_535_322), -14_536);
        assert_eq!(energy_units(-14_536_604), -14_537);
        assert_eq!(energy_units(-14_513_000), -14_513);
        assert_eq!(energy_units(0), 0);
        assert_eq!(energy_units(1_500), 1);
    }

    #[test]
    fn display_option_prints_none_not_debug() {
        assert_eq!(display_option(Some(10_132_u64)), "10132");
        assert_eq!(display_option(None::<u64>), "none");
        assert_eq!(display_energy(Some(-14_536_604)), "-14537");
        assert_eq!(display_energy(None), "none");
    }
}
