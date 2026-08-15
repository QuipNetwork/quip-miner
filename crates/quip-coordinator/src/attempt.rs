//! Mining-attempt recording (agm.4).
//!
//! Every model the ising solver sends back a result for is recorded as one JSON
//! line under `<data_dir>/<qblock_id>/attempts.jsonl`. A single writer thread
//! drains a channel and appends, so concurrent miner sessions never interleave
//! partial lines. The REST dashboard ([`crate::dashboard`]) serves these files
//! statically — the coordinator does no query work.

use crate::validate::Validated;
use quip_proto::v1::Job;
use serde::Serialize;
use std::fmt::Write as _;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{SystemTime, UNIX_EPOCH};

/// One recorded mining attempt: a model that was sent to a miner and solved.
#[derive(Debug, Clone, Serialize)]
pub struct AttemptRecord {
    /// Unix-epoch milliseconds when the coordinator validated the result.
    pub ts_ms: u64,
    /// Chain quantum-block id (`QuantumPowApi_latest_qblock_id`). `None` before
    /// the chain assigns one (e.g. drive mode) — those land under `pending/`.
    pub qblock_id: Option<u64>,
    /// `PoW` cancellation generation of the job (0 for mempool).
    pub generation: u64,
    /// Miner that solved the model.
    pub miner_id: String,
    /// Job id (nonce) in hex.
    pub job_id: String,
    /// Whether this job was a `PoW` job (`true`) or mempool (`false`).
    pub is_pow: bool,
    /// Mempool order id in hex; empty for `PoW`.
    pub order_id: String,
    /// Best solution energy in milli-units.
    pub best_energy_milli: i64,
    /// Minimum energy over every shape-valid row, ignoring the current gate.
    ///
    /// `best_energy_milli` holds the `i64::MAX` no-solution sentinel when no row
    /// cleared the gate. This field holds the true best the miner found, so the
    /// dashboard reader has a real number to serve in that case.
    pub raw_best_energy_milli: i64,
    /// Pairwise diversity of the accepted set in milli-units.
    pub diversity_milli: u32,
    /// Count of gate-passing solutions in the result.
    pub n_valid: u32,
    /// Met the acceptance gate (energy + diversity + `min_solutions`).
    pub accepted: bool,
    /// The coordinator submitted this attempt as a proof.
    pub submitted: bool,
    /// Device access time reported by the miner, in microseconds.
    pub device_access_time_us: u64,
    /// Solving backend: `CPU`, `GPU`, or `QPU`, with an optional suffix.
    pub miner_type: String,
    /// Energy ceiling the coordinator checked this result against.
    pub threshold_milli: i64,
    /// `LastProofBlockHash` the job was built on, `0x`-prefixed hex.
    pub last_proof_block_hash: String,
    /// Submission extrinsic hash, `0x`-prefixed hex. `None` until it is signed.
    pub extrinsic_hash: Option<String>,
    /// Winning block hash, `0x`-prefixed hex. `None` when this did not win.
    pub chain_block_hash: Option<String>,
    /// Winning block height. `None` when this did not win.
    pub chain_block_number: Option<u64>,
    /// On-chain `proofs_submitted` sequence at submit time. `None` when the
    /// coordinator did not submit.
    pub pow_sequence: Option<u64>,
}

/// The chain-side facts one attempt records, gathered by the session at submit
/// time.
///
/// Grouped rather than passed as loose arguments: eight positional parameters of
/// mostly-optional hex strings is how a caller silently swaps two of them.
#[derive(Debug, Clone, Default)]
pub struct AttemptContext {
    /// Solving backend label.
    pub miner_type: String,
    /// Energy ceiling checked against.
    pub threshold_milli: i64,
    /// `LastProofBlockHash` the job was built on.
    pub last_proof_block_hash: String,
    /// Extrinsic hash, once signed.
    pub extrinsic_hash: Option<String>,
    /// Winning block hash.
    pub chain_block_hash: Option<String>,
    /// Winning block height.
    pub chain_block_number: Option<u64>,
    /// On-chain `proofs_submitted` sequence.
    pub pow_sequence: Option<u64>,
    /// Device access time reported by the miner, in microseconds.
    pub device_access_time_us: u64,
}

impl AttemptRecord {
    /// Build a record from a validated result. `miner_id` is the solving miner;
    /// `submitted` is whether the coordinator submitted it as a proof; `ctx`
    /// carries the chain-side facts gathered at submit time.
    #[must_use]
    pub fn new(
        qblock_id: Option<u64>,
        miner_id: &str,
        job_id: &[u8],
        job: &Job,
        v: &Validated,
        submitted: bool,
        ctx: &AttemptContext,
    ) -> Self {
        let (is_pow, order_id) = job
            .provenance
            .as_ref()
            .map_or((false, String::new()), |p| (p.is_pow, hex(&p.order_id)));
        Self {
            ts_ms: now_ms(),
            qblock_id,
            generation: job.generation,
            miner_id: miner_id.to_string(),
            job_id: hex(job_id),
            is_pow,
            order_id,
            best_energy_milli: v.best_energy_milli,
            raw_best_energy_milli: v.raw_best_energy_milli,
            diversity_milli: v.diversity_milli,
            n_valid: v.n_valid,
            accepted: v.accepted,
            submitted,
            device_access_time_us: ctx.device_access_time_us,
            miner_type: ctx.miner_type.clone(),
            threshold_milli: ctx.threshold_milli,
            last_proof_block_hash: ctx.last_proof_block_hash.clone(),
            extrinsic_hash: ctx.extrinsic_hash.clone(),
            chain_block_hash: ctx.chain_block_hash.clone(),
            chain_block_number: ctx.chain_block_number,
            pow_sequence: ctx.pow_sequence,
        }
    }
}

/// Per-qblock summary written to `attempts.json`: aggregate counts + the
/// win-time stash annotations (most-viable candidates + projected blocks).
#[derive(Debug, Clone, Serialize)]
pub struct QblockSummary {
    /// Quantum-block id, or `None` for the `pending/` bucket.
    pub qblock_id: Option<u64>,
    /// Unix-epoch milliseconds when this summary was written.
    pub updated_ts_ms: u64,
    /// Best accepted energy so far for this qblock, if any.
    pub current_best_milli: Option<i64>,
    /// Number of results validated for this qblock.
    pub results_validated: u64,
    /// Win-time stash annotation snapshot.
    pub stash: crate::stash::StashSummary,
}

/// Serialize a qblock summary for `attempts.json` (empty string on the
/// impossible serialize error, so the caller never has to handle it).
#[must_use]
pub fn summary_body(
    qblock_id: Option<u64>,
    current_best_milli: Option<i64>,
    results_validated: u64,
    stash: crate::stash::StashSummary,
) -> String {
    let s = QblockSummary {
        qblock_id,
        updated_ts_ms: now_ms(),
        current_best_milli,
        results_validated,
        stash,
    };
    serde_json::to_string(&s).unwrap_or_default()
}

/// Directory segment for a qblock: its id, or `pending` before the chain
/// assigns one (e.g. drive mode).
fn qblock_dir_name(qblock_id: Option<u64>) -> String {
    qblock_id.map_or_else(|| "pending".to_string(), |id| id.to_string())
}

/// A message to the single writer thread: either one attempt line, or a
/// rewrite of a qblock's `attempts.json` summary.
pub enum WriterMsg {
    /// Append one attempt to `<qblock>/attempts.jsonl`.
    Attempt(Box<AttemptRecord>),
    /// Overwrite `<qblock>/attempts.json` with `body` (a serialized summary).
    Summary {
        /// Quantum-block id for the directory segment (or `None` → `pending`).
        qblock_id: Option<u64>,
        /// Serialized [`QblockSummary`] body.
        body: String,
    },
}

fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "unix millis fit u64 for the foreseeable future"
        )]
        {
            d.as_millis() as u64
        }
    })
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Append one record as a JSON line to `<data_dir>/<qblock_id>/attempts.jsonl`,
/// creating the per-qblock directory on first write.
///
/// # Errors
///
/// Returns an I/O error if the directory cannot be created, the file cannot be
/// opened, serialization fails, or the line cannot be written.
pub fn append_record(data_dir: &Path, rec: &AttemptRecord) -> std::io::Result<()> {
    let dir = data_dir.join(qblock_dir_name(rec.qblock_id));
    std::fs::create_dir_all(&dir)?;
    let path = dir.join("attempts.jsonl");
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    let line = serde_json::to_string(rec).map_err(std::io::Error::other)?;
    writeln!(f, "{line}")
}

/// Overwrite `<data_dir>/<qblock_id>/attempts.json` with the summary `body`.
///
/// # Errors
///
/// Returns an I/O error if the directory cannot be created or the file cannot
/// be written.
pub fn write_summary(data_dir: &Path, qblock_id: Option<u64>, body: &str) -> std::io::Result<()> {
    let dir = data_dir.join(qblock_dir_name(qblock_id));
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("attempts.json"), body)
}

/// Spawn the single writer thread and return the sender the sessions push to.
/// Blocking file I/O runs on this dedicated thread, off the async runtime.
#[must_use]
pub fn spawn_writer(data_dir: PathBuf) -> mpsc::Sender<WriterMsg> {
    let (tx, rx) = mpsc::channel::<WriterMsg>();
    let _ = std::thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            let res = match msg {
                WriterMsg::Attempt(rec) => append_record(&data_dir, &rec),
                WriterMsg::Summary { qblock_id, body } => {
                    write_summary(&data_dir, qblock_id, &body)
                }
            };
            if let Err(e) = res {
                tracing::warn!("attempt-log write failed: {e}");
            }
        }
    });
    tx
}

#[cfg(test)]
mod tests {
    use super::*;
    use quip_proto::v1::Provenance;

    fn rec(qblock_id: Option<u64>, job_id: &[u8], is_pow: bool) -> AttemptRecord {
        let job = Job {
            job_id: job_id.to_vec(),
            generation: 7,
            provenance: Some(Provenance {
                is_pow,
                order_id: if is_pow { vec![] } else { vec![0xab, 0xcd] },
            }),
            ..Default::default()
        };
        let v = Validated {
            best_energy_milli: -14_200,
            diversity_milli: 250,
            n_valid: 6,
            accepted: true,
            selected_solutions: Vec::new(),
            raw_best_energy_milli: -14_200,
            stash_solutions: Vec::new(),
        };
        let ctx = AttemptContext {
            device_access_time_us: 1234,
            ..Default::default()
        };
        AttemptRecord::new(qblock_id, "cpu-0", job_id, &job, &v, is_pow, &ctx)
    }

    #[test]
    fn record_serializes_hex_ids_and_fields() {
        let r = rec(Some(42), &[0x01, 0xff], true);
        let v: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
        assert_eq!(v.get("qblock_id"), Some(&serde_json::json!(42)));
        assert_eq!(v.get("job_id"), Some(&serde_json::json!("01ff")));
        assert_eq!(v.get("miner_id"), Some(&serde_json::json!("cpu-0")));
        assert_eq!(v.get("is_pow"), Some(&serde_json::json!(true)));
        assert_eq!(v.get("order_id"), Some(&serde_json::json!("")));
        assert_eq!(
            v.get("best_energy_milli"),
            Some(&serde_json::json!(-14_200))
        );
        assert_eq!(v.get("n_valid"), Some(&serde_json::json!(6)));
        assert_eq!(v.get("submitted"), Some(&serde_json::json!(true)));
    }

    #[test]
    fn mempool_record_hex_encodes_order_id() {
        let r = rec(Some(1), &[0x09], false);
        assert_eq!(r.order_id, "abcd");
        assert!(!r.is_pow);
    }

    #[test]
    fn append_writes_jsonl_under_qblock_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let r1 = rec(Some(5), &[0xaa], true);
        let r2 = rec(Some(5), &[0xbb], true);
        append_record(tmp.path(), &r1).unwrap();
        append_record(tmp.path(), &r2).unwrap();
        let path = tmp.path().join("5").join("attempts.jsonl");
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2);
        // Each line is a standalone JSON object.
        for line in lines {
            let _: serde_json::Value = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn none_qblock_lands_under_pending() {
        let tmp = tempfile::tempdir().unwrap();
        append_record(tmp.path(), &rec(None, &[0x01], true)).unwrap();
        assert!(tmp.path().join("pending").join("attempts.jsonl").exists());
    }
}
