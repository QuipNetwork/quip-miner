//! Operator-facing number formatting for miner log lines.
//!
//! Energies stay in milli-units internally. These helpers change only the
//! text an operator reads.

/// Floor-divide a milli-energy to whole units for a log line.
///
/// Rust `/` truncates toward zero. Negative energies would then look better
/// than they are. Euclid division floors, which matches the operator spec.
#[must_use]
pub(crate) fn energy_units(milli: i64) -> i64 {
    milli.div_euclid(1000)
}

/// Render a millisecond duration with a rounded last unit.
///
/// | input | rendered |
/// | --- | --- |
/// | under 1 second | `889ms` |
/// | under 1 minute | `48.9s` |
/// | under 1 hour | `10m 49s` |
/// | 1 hour or more | `1h 12m` |
#[must_use]
pub(crate) fn format_duration_ms(ms: u64) -> String {
    if ms < 1_000 {
        return format!("{ms}ms");
    }
    if ms < 60_000 {
        // Tenths of a second, rounded. 600 tenths is one minute.
        let tenths = (ms + 50) / 100;
        if tenths >= 600 {
            return "1m 0s".to_owned();
        }
        let secs = tenths / 10;
        let frac = tenths % 10;
        return format!("{secs}.{frac}s");
    }
    if ms < 3_600_000 {
        let mut mins = ms / 60_000;
        let rem = ms % 60_000;
        let mut secs = (rem + 500) / 1_000;
        if secs == 60 {
            mins = mins.saturating_add(1);
            secs = 0;
        }
        if mins >= 60 {
            return format_hours_minutes(mins / 60, mins % 60);
        }
        return format!("{mins}m {secs}s");
    }
    let mut hours = ms / 3_600_000;
    let rem = ms % 3_600_000;
    let mut mins = (rem + 30_000) / 60_000;
    if mins == 60 {
        hours = hours.saturating_add(1);
        mins = 0;
    }
    format_hours_minutes(hours, mins)
}

fn format_hours_minutes(hours: u64, mins: u64) -> String {
    format!("{hours}h {mins}m")
}

#[cfg(test)]
mod tests {
    use super::{energy_units, format_duration_ms};

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
    fn energy_units_does_not_truncate_toward_zero() {
        // Toward-zero division would report -14536, which is better than the
        // real floor. The helper must not do that.
        assert_ne!(energy_units(-14_536_604), -14_536_604 / 1000);
        assert_eq!(-14_536_604 / 1000, -14_536);
    }

    #[test]
    fn format_duration_ms_matches_the_spec_buckets() {
        assert_eq!(format_duration_ms(889), "889ms");
        assert_eq!(format_duration_ms(48_900), "48.9s");
        assert_eq!(format_duration_ms(648_889), "10m 49s");
        assert_eq!(format_duration_ms(4_320_000), "1h 12m");
    }

    #[test]
    fn format_duration_ms_rounds_the_last_unit() {
        assert_eq!(format_duration_ms(0), "0ms");
        assert_eq!(format_duration_ms(999), "999ms");
        assert_eq!(format_duration_ms(1_000), "1.0s");
        assert_eq!(format_duration_ms(48_850), "48.9s");
        assert_eq!(format_duration_ms(155_002), "2m 35s");
        assert_eq!(format_duration_ms(4_290_000), "1h 12m");
    }

    #[test]
    fn format_duration_ms_promotes_when_the_last_unit_rounds_up() {
        assert_eq!(format_duration_ms(59_950), "1m 0s");
        // 59m 59.5s is still under one hour, so seconds is the last unit.
        assert_eq!(format_duration_ms(3_599_500), "1h 0m");
    }
}
