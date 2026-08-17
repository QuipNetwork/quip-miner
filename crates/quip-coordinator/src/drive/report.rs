//! Per-job scoring rows, aggregate stats, and stdout/JSONL report output.

use serde_json::json;
use std::fmt::Write as _;
use std::io::Write;
use std::path::Path;

/// One row: a job's validated `Result` scored against its quality gates.
#[derive(Debug, Clone)]
pub struct JobRow {
    /// Job identifier bytes (nonce for `PoW`, synthetic label for explicit jobs).
    pub job_id: Vec<u8>,
    /// True when the job was `PoW`/nonce-derived (`job_id` is a nonce);
    /// false for explicit-entry jobs (`job_id` is a synthetic label).
    pub is_pow: bool,
    /// Number of solutions returned by the miner for this job.
    pub n_solutions: usize,
    /// Best (lowest) energy among accepted solutions, in milli-units.
    pub best_energy_milli: i64,
    /// Diversity score of the solution set, in milli-units.
    pub diversity_milli: u32,
    /// Whether the result passed the quality gates.
    pub passed: bool,
    /// Device access time reported by the miner, microseconds.
    pub device_access_time_us: u64,
    /// Wall-clock ms from dispatch of this job to its terminal outcome.
    pub wall_ms: u64,
    /// Resolved `num_reads`/`num_sweeps` the miner actually ran (from
    /// `SamplerMeta`). Zero on rejected rows. For matched-condition parity
    /// runs these show what each backend's adapt envelope settled on.
    pub reads: u32,
    /// Resolved sweeps the miner actually ran (see `reads`).
    pub sweeps: u32,
    /// True if the miner rejected the job outright (no `Result` returned).
    pub rejected: bool,
}

/// Aggregate stats over a full drive run.
#[derive(Debug, Clone, Default)]
pub struct Aggregate {
    /// Number of job rows included in the aggregate.
    pub total_jobs: usize,
    /// Jobs that passed quality gates.
    pub passed: usize,
    /// Jobs the miner rejected outright.
    pub rejected: usize,
    /// Real wall-clock span of the run (first dispatch → last result), ms.
    pub wall_ms_total: u64,
    /// Throughput in jobs per second using `wall_ms_total`.
    pub throughput_per_s: f64,
}

/// Compute aggregate stats from per-job rows.
/// `run_wall_ms` is the real wall-clock span of the run (first dispatch → last
/// result). Throughput uses it — NOT the sum of per-job `wall_ms`, which
/// overcounts the streaming backends' concurrent, overlapping jobs (e.g. 12
/// jobs each ~3.3s that all overlap take ~3.3s of wall clock, not ~40s).
#[must_use]
#[expect(
    clippy::cast_precision_loss,
    reason = "job count and wall-clock ms are small; f64 throughput is display-only"
)]
pub fn aggregate(rows: &[JobRow], run_wall_ms: u64) -> Aggregate {
    let total_jobs = rows.len();
    let passed = rows.iter().filter(|r| r.passed).count();
    let rejected = rows.iter().filter(|r| r.rejected).count();
    let throughput_per_s = if run_wall_ms > 0 {
        total_jobs as f64 / (run_wall_ms as f64 / 1000.0)
    } else {
        0.0
    };
    Aggregate {
        total_jobs,
        passed,
        rejected,
        wall_ms_total: run_wall_ms,
        throughput_per_s,
    }
}

/// Print a human-readable table: one row per job, then an aggregate summary.
#[expect(clippy::print_stdout, reason = "drive CLI output")]
pub fn print_table(rows: &[JobRow], agg: &Aggregate) {
    println!(
        "{:<4} {:<18} {:>10} {:>12} {:>10} {:>6} {:>8}",
        "#", "job_id", "n_sol", "best_energy", "diversity", "pass", "wall_ms"
    );
    for (i, r) in rows.iter().enumerate() {
        let job_id_hex = hex(&r.job_id);
        let status = if r.rejected {
            "REJECT"
        } else if r.passed {
            "pass"
        } else {
            "fail"
        };
        println!(
            "{:<4} {:<18} {:>10} {:>12} {:>10} {:>6} {:>8}",
            i + 1,
            truncate_hex(&job_id_hex),
            r.n_solutions,
            r.best_energy_milli,
            r.diversity_milli,
            status,
            r.wall_ms
        );
    }
    println!(
        "--- {} jobs, {} passed, {} rejected, {:.2} jobs/s ---",
        agg.total_jobs, agg.passed, agg.rejected, agg.throughput_per_s
    );
    if let Some(p) = effective_params(rows) {
        println!("--- effective reads {}, sweeps {} ---", p.0, p.1);
    }
}

/// Effective reads/sweeps across non-rejected rows, each as `"N"` when uniform
/// or `"lo..hi"` when the run diverged. `None` if no job produced a result.
/// A matched-condition adapt run reports one value per field per backend.
fn effective_params(rows: &[JobRow]) -> Option<(String, String)> {
    let mut reads = rows.iter().filter(|r| !r.rejected).map(|r| r.reads);
    let mut sweeps = rows.iter().filter(|r| !r.rejected).map(|r| r.sweeps);
    let (r0, s0) = (reads.next()?, sweeps.next()?);
    let (r_lo, r_hi) = reads.fold((r0, r0), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let (s_lo, s_hi) = sweeps.fold((s0, s0), |(lo, hi), v| (lo.min(v), hi.max(v)));
    let render = |lo: u32, hi: u32| {
        if lo == hi {
            lo.to_string()
        } else {
            format!("{lo}..{hi}")
        }
    };
    Some((render(r_lo, r_hi), render(s_lo, s_hi)))
}

fn truncate_hex(s: &str) -> String {
    if s.len() > 16 {
        format!("{}..", &s[..16])
    } else {
        s.to_string()
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len().saturating_mul(2));
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

fn row_to_json(r: &JobRow) -> serde_json::Value {
    json!({
        "job_id": hex(&r.job_id),
        "is_pow": r.is_pow,
        "n_solutions": r.n_solutions,
        "best_energy_milli": r.best_energy_milli,
        "diversity_milli": r.diversity_milli,
        "passed": r.passed,
        "rejected": r.rejected,
        "device_access_time_us": r.device_access_time_us,
        "reads": r.reads,
        "sweeps": r.sweeps,
        "wall_ms": r.wall_ms,
    })
}

/// Write one JSON object per job followed by a trailing aggregate record.
///
/// # Errors
///
/// Returns an I/O error if the file cannot be created or written.
pub fn write_jsonl(path: &Path, rows: &[JobRow], agg: &Aggregate) -> std::io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for r in rows {
        writeln!(f, "{}", row_to_json(r))?;
    }
    writeln!(
        f,
        "{}",
        json!({
            "aggregate": true,
            "total_jobs": agg.total_jobs,
            "passed": agg.passed,
            "rejected": agg.rejected,
            "wall_ms_total": agg.wall_ms_total,
            "throughput_per_s": agg.throughput_per_s,
        })
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(passed: bool, rejected: bool, wall_ms: u64) -> JobRow {
        JobRow {
            job_id: vec![1, 2, 3],
            is_pow: true,
            n_solutions: 1,
            best_energy_milli: -500,
            diversity_milli: 200,
            passed,
            device_access_time_us: 100,
            reads: 64,
            sweeps: 256,
            wall_ms,
            rejected,
        }
    }

    #[test]
    fn aggregate_counts_pass_and_reject() {
        let rows = vec![
            row(true, false, 10),
            row(false, false, 20),
            row(false, true, 5),
        ];
        // 3 jobs over a real 1.5s wall span -> 2 jobs/s (independent of the
        // per-job wall times, which overlap for concurrent streaming jobs).
        let agg = aggregate(&rows, 1500);
        assert_eq!(agg.total_jobs, 3);
        assert_eq!(agg.passed, 1);
        assert_eq!(agg.rejected, 1);
        assert_eq!(agg.wall_ms_total, 1500);
        assert!((agg.throughput_per_s - 2.0).abs() < 1e-9);
    }

    #[test]
    fn aggregate_of_empty_rows_has_zero_throughput() {
        let agg = aggregate(&[], 0);
        assert_eq!(agg.total_jobs, 0);
        #[expect(clippy::float_cmp, reason = "exact zero throughput on empty run")]
        {
            assert_eq!(agg.throughput_per_s, 0.0);
        }
    }

    #[test]
    fn jsonl_round_trips_through_serde_json() {
        let rows = vec![row(true, false, 10), row(false, true, 0)];
        let agg = aggregate(&rows, 1000);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("report.jsonl");
        write_jsonl(&path, &rows, &agg).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3); // 2 rows + 1 aggregate
        #[expect(
            clippy::indexing_slicing,
            reason = "len asserted to 3 immediately above"
        )]
        {
            let first: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
            assert_eq!(first["passed"], true);
            let last: serde_json::Value = serde_json::from_str(lines[2]).unwrap();
            assert_eq!(last["aggregate"], true);
            assert_eq!(last["total_jobs"], 2);
        }
    }
}
