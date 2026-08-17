//! Uniform config-override discipline shared by every backend's
//! [`Sampler::apply_config`](crate::Sampler::apply_config).
//!
//! Backends parse `Configure.backend_toml` (the verbatim `config.toml`
//! subsection the coordinator forwards) against their own schema, then use
//! these helpers so precedence (config > CLI), override warnings, and
//! unknown-field warnings look identical across every miner type.

use std::fmt::Display;

/// Keys the shared session loop consumes (not any single backend), so no
/// backend should warn about them as unknown. Kept here so every backend's
/// unknown-field check treats them identically.
pub const SESSION_KEYS: &[&str] = &["num_sweeps"];

/// Resolve one setting with `config > CLI` precedence.
///
/// When `from_config` is present and differs from `cli`, print an override
/// warning and return the config value; otherwise return `cli` silently, so a
/// config that merely restates the CLI value produces no noise.
#[must_use]
pub fn config_override<T: PartialEq + Display>(name: &str, cli: T, from_config: Option<T>) -> T {
    match from_config {
        Some(value) if value != cli => {
            tracing::warn!("config overrides {name}: {cli} -> {value} (from coordinator)");
            value
        }
        _ => cli,
    }
}

/// Warn once per unrecognized config key. Backends collect unknowns with
/// `#[serde(flatten)]` and pass their names here; the field is ignored, and the
/// warning surfaces typos rather than silently dropping them.
pub fn warn_unknown_fields<I, S>(backend: &str, keys: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for key in keys {
        let key = key.as_ref();
        if SESSION_KEYS.contains(&key) {
            continue; // consumed by the session loop, not this backend
        }
        tracing::warn!("config: unknown field '{key}' for {backend} (ignored)");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_wins_and_reports_only_on_change() {
        // config present + different -> config value
        assert_eq!(config_override("utilization", 100u32, Some(60)), 60);
        // config present + same -> cli value (no effective change)
        assert_eq!(config_override("utilization", 60u32, Some(60)), 60);
        // config absent -> cli value
        assert_eq!(config_override("utilization", 100u32, None), 100);
        // works for bool (yielding)
        assert!(config_override("yielding", false, Some(true)));
    }
}
