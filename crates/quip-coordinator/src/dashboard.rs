//! REST dashboard and indexer `/api/v1` surface (agm.4).
//!
//! Serves the per-qblock attempt logs the [`crate::attempt`] writer appends,
//! plus the three endpoints the dashboard indexer polls:
//! `GET /api/v1/status`, `GET /api/v1/stats`, and
//! `GET /api/v1/mining/attempts?solution_number=N`.
//!
//! Static files under `data_dir` remain available via the fallback service
//! (`GET /<qblock_id>/attempts.jsonl`). The attempts JSONL is also the data
//! source for the `/api/v1/mining/attempts` envelope: `solution_number` maps
//! to the directory name under `data_dir`.

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tower_http::services::ServeDir;

/// Build the dashboard router rooted at `data_dir`:
/// - `GET /qblocks` → JSON array of available qblock ids (directory names)
/// - `GET /healthz` → `ok`
/// - `GET /api/v1/status` → indexer status envelope
/// - `GET /api/v1/stats` → indexer stats envelope
/// - `GET /api/v1/mining/attempts?solution_number=N` → submission + attempts
/// - everything else → static files under `data_dir`, e.g.
///   `GET /<qblock_id>/attempts.jsonl`.
pub fn router(data_dir: PathBuf) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/qblocks", get(list_qblocks))
        .route("/api/v1/status", get(api_status))
        .route("/api/v1/stats", get(api_stats))
        .route("/api/v1/mining/attempts", get(api_mining_attempts))
        .fallback_service(ServeDir::new(&data_dir))
        .with_state(data_dir)
}

async fn healthz() -> &'static str {
    "ok"
}

/// List the qblock directories under the data root (sorted). Returns an empty
/// list if the root does not exist yet.
async fn list_qblocks(State(data_dir): State<PathBuf>) -> Json<Vec<String>> {
    let mut ids = Vec::new();
    if let Ok(mut rd) = tokio::fs::read_dir(&data_dir).await {
        while let Ok(Some(entry)) = rd.next_entry().await {
            let is_dir = entry.file_type().await.is_ok_and(|t| t.is_dir());
            if is_dir {
                if let Some(name) = entry.file_name().to_str() {
                    ids.push(name.to_string());
                }
            }
        }
    }
    ids.sort();
    Json(ids)
}

/// Serve the dashboard on `listen` (e.g. `0.0.0.0:20100`) until the task is
/// aborted. A bind failure is logged and the task exits without taking down the
/// coordinator.
pub async fn serve(listen: String, data_dir: PathBuf) {
    let listener = match tokio::net::TcpListener::bind(&listen).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("dashboard: bind {listen} failed: {e}");
            return;
        }
    };
    tracing::info!(
        "dashboard: serving {} at http://{listen}",
        data_dir.display()
    );
    if let Err(e) = axum::serve(listener, router(data_dir)).await {
        tracing::warn!("dashboard: server error: {e}");
    }
}

// ---------------------------------------------------------------------------
// Response envelope (matches v0.2 telemetry process + indexer client)
// ---------------------------------------------------------------------------

fn unix_ts() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn success_envelope(data: &Value) -> Response {
    Json(json!({
        "success": true,
        "data": data,
        "timestamp": unix_ts(),
    }))
    .into_response()
}

fn error_envelope(status: StatusCode, message: &str, code: &str) -> Response {
    (
        status,
        Json(json!({
            "success": false,
            "error": message,
            "code": code,
            "timestamp": unix_ts(),
        })),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// GET /api/v1/status
// ---------------------------------------------------------------------------

/// Indexer status probe. Identity and chain fields have no live source in the
/// file-backed dashboard yet, so they return empty / zero values the client
/// already tolerates (see report: Blocked / needs decision).
async fn api_status() -> Response {
    success_envelope(&json!({
        "ss58_address": "",
        "account_id_hex": "",
        "node_id": "",
        "is_mining": false,
        "uptime_seconds": 0,
        "chain": {
            "head_hash": "",
            "head_number": 0,
        },
        "miner_registered": false,
        "miner_info": null,
        "miners": [],
        "modes": {},
    }))
}

// ---------------------------------------------------------------------------
// GET /api/v1/stats
// ---------------------------------------------------------------------------

/// Indexer stats probe. Controller counters are not tracked by the dashboard
/// writer; zeros keep the envelope parseable until a live counter source is
/// wired.
async fn api_stats() -> Response {
    success_envelope(&json!({
        "controller": {
            "heads_observed": 0,
            "contexts_dispatched": 0,
            "results_received": 0,
            "proofs_submitted": 0,
            "stale_drops": 0,
            "submission_errors": 0,
            "duplicate_result_drops": 0,
        }
    }))
}

// ---------------------------------------------------------------------------
// GET /api/v1/mining/attempts?solution_number=N
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct AttemptsQuery {
    /// Global solution number; maps to `data_dir/<solution_number>/`.
    solution_number: Option<String>,
}

async fn api_mining_attempts(
    State(data_dir): State<PathBuf>,
    Query(query): Query<AttemptsQuery>,
) -> Response {
    let Some(raw) = query.solution_number.as_deref() else {
        return error_envelope(
            StatusCode::BAD_REQUEST,
            "supply ?solution_number=N",
            "BAD_PARAM",
        );
    };
    let Ok(solution_number) = raw.parse::<u64>() else {
        return error_envelope(
            StatusCode::BAD_REQUEST,
            "solution_number must be an integer",
            "BAD_PARAM",
        );
    };
    match load_attempts_envelope(&data_dir, solution_number) {
        Ok(data) => success_envelope(&data),
        Err(AttemptsLoadError::NotFound) => error_envelope(
            StatusCode::NOT_FOUND,
            &format!("solution_number {solution_number} not found"),
            "NOT_FOUND",
        ),
        Err(AttemptsLoadError::Io(e)) => {
            tracing::warn!(
                solution_number,
                error = %e,
                "dashboard: failed to read attempts for solution_number"
            );
            error_envelope(
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to read attempts",
                "INTERNAL_ERROR",
            )
        }
    }
}

enum AttemptsLoadError {
    NotFound,
    Io(std::io::Error),
}

/// Read `data_dir/<solution_number>/attempts.jsonl` and build the
/// `{ submission, attempts }` envelope the indexer parser expects.
fn load_attempts_envelope(
    data_dir: &Path,
    solution_number: u64,
) -> Result<Value, AttemptsLoadError> {
    let dir = data_dir.join(solution_number.to_string());
    if !dir.is_dir() {
        return Err(AttemptsLoadError::NotFound);
    }
    let path = dir.join("attempts.jsonl");
    let body = match std::fs::read_to_string(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(AttemptsLoadError::NotFound);
        }
        Err(e) => return Err(AttemptsLoadError::Io(e)),
    };

    let mut records: Vec<Map<String, Value>> = Vec::new();
    for line in body.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(Value::Object(obj)) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        records.push(obj);
    }
    if records.is_empty() {
        return Err(AttemptsLoadError::NotFound);
    }

    let attempts = records
        .iter()
        .enumerate()
        .map(|(i, rec)| attempt_from_record(rec, i + 1, solution_number))
        .collect::<Vec<_>>();

    let submission = submission_from_records(&records, solution_number);
    Ok(json!({
        "submission": submission,
        "attempts": attempts,
    }))
}

/// Map one v0.3 [`crate::attempt::AttemptRecord`] JSON object onto the v0.2
/// attempt wire shape the indexer parser reads.
fn attempt_from_record(rec: &Map<String, Value>, iter: usize, solution_number: u64) -> Value {
    let best_energy_milli = wire_energy_milli(rec, solution_number);
    let result_kind = result_kind_of(rec);
    let miner_id = string_field(rec, "miner_id").unwrap_or_default();
    let ts_ns = ts_ns_of(rec);
    // Map device access time onto the QPU field the indexer sums. CPU/GPU
    // miners also record this; the indexer treats the sum as QPU compute.
    let qpu_access_time_us = match u64_field(rec, "device_access_time_us") {
        Some(us) if us > 0 => safe_u64("qpu_access_time_us", solution_number, us),
        _ => Value::Null,
    };
    // AttemptRecord has no backend type; empty string is the indexer default.
    json!({
        "type": "attempt",
        "ts_ns": wire_u128(ts_ns),
        "miner_id": miner_id,
        "miner_type": "",
        "solution_number": solution_number,
        "iter": iter,
        "best_energy_milli": best_energy_milli,
        "result_kind": result_kind,
        "num_valid": rec.get("n_valid").and_then(Value::as_u64)
            .map_or(Value::Null, |n| safe_u64("num_valid", solution_number, n)),
        "diversity_milli": rec.get("diversity_milli").and_then(Value::as_u64)
            .map_or(Value::Null, |n| safe_u64("diversity_milli", solution_number, n)),
        "qpu_access_time_us": qpu_access_time_us,
        "job_id": string_field(rec, "job_id"),
        "accepted": bool_field(rec, "accepted"),
        "submitted": bool_field(rec, "submitted"),
    })
}

/// Build a submission object from the attempt trail. v0.3 does not write
/// `submission.json`; the indexer still requires the submission object.
///
/// Fields with no v0.3 source:
/// - `threshold_milli` → `0`
/// - `last_proof_block_hash` → `"0x0"` (non-empty so the parser accepts it)
/// - `extrinsic_hash`, `chain_block_hash`, `chain_block_number`, `pow_sequence`
///   → `null`
/// - `miner_type` → `""`
fn submission_from_records(records: &[Map<String, Value>], solution_number: u64) -> Value {
    // Prefer the last submitted attempt; else the last accepted; else the last.
    // Caller guarantees `records` is non-empty; fall back to an empty map only
    // so this helper never panics if that invariant is broken.
    let empty = Map::new();
    let chosen = records
        .iter()
        .rev()
        .find(|r| bool_field(r, "submitted") == Some(true))
        .or_else(|| {
            records
                .iter()
                .rev()
                .find(|r| bool_field(r, "accepted") == Some(true))
        })
        .or_else(|| records.last())
        .unwrap_or(&empty);

    let miner_id = string_field(chosen, "miner_id").unwrap_or_else(|| "unknown".into());
    let energy_milli = wire_energy_milli(chosen, solution_number);
    let diversity_milli = safe_u64(
        "diversity_milli",
        solution_number,
        u64_field(chosen, "diversity_milli").unwrap_or(0),
    );
    let num_valid = u64_field(chosen, "n_valid")
        .map_or(Value::Null, |n| safe_u64("num_valid", solution_number, n));
    let ts_ns = wire_u128(ts_ns_of(chosen));

    let any_submitted = records
        .iter()
        .any(|r| bool_field(r, "submitted") == Some(true));
    let any_accepted = records
        .iter()
        .any(|r| bool_field(r, "accepted") == Some(true));
    let outcome = if any_submitted {
        "submitted"
    } else if any_accepted {
        "stored"
    } else {
        "rejected"
    };

    json!({
        "type": "submission",
        "ts_ns": ts_ns,
        "solution_number": solution_number,
        "miner_id": miner_id,
        "miner_type": "",
        "energy_milli": energy_milli,
        "diversity_milli": diversity_milli,
        // No max-energy / threshold is stored on AttemptRecord.
        "threshold_milli": 0,
        "num_valid": num_valid,
        // No last-proof block hash is stored on AttemptRecord.
        "last_proof_block_hash": "0x0",
        "extrinsic_hash": null,
        "chain_block_hash": null,
        "chain_block_number": null,
        "pow_sequence": null,
        "outcome": outcome,
    })
}

fn result_kind_of(rec: &Map<String, Value>) -> &'static str {
    if bool_field(rec, "submitted") == Some(true) {
        "submitted"
    } else if bool_field(rec, "accepted") == Some(true) {
        "stored"
    } else {
        "rejected"
    }
}

fn ts_ns_of(rec: &Map<String, Value>) -> u128 {
    // AttemptRecord records wall time in milliseconds; the wire shape uses ns.
    match u64_field(rec, "ts_ms") {
        Some(ms) => u128::from(ms).saturating_mul(1_000_000),
        None => 0,
    }
}

fn string_field(rec: &Map<String, Value>, key: &str) -> Option<String> {
    rec.get(key).and_then(|v| match v {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        _ => None,
    })
}

fn bool_field(rec: &Map<String, Value>, key: &str) -> Option<bool> {
    rec.get(key).and_then(Value::as_bool)
}

fn i64_field(rec: &Map<String, Value>, key: &str) -> Option<i64> {
    rec.get(key).and_then(|v| match v {
        Value::Number(n) => n.as_i64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    })
}

fn u64_field(rec: &Map<String, Value>, key: &str) -> Option<u64> {
    rec.get(key).and_then(|v| match v {
        Value::Number(n) => n.as_u64(),
        Value::String(s) => s.parse().ok(),
        _ => None,
    })
}

/// Resolve the energy a record puts on the wire.
///
/// Order: the gate-passing best when it is not the sentinel, then the raw best
/// when it is not the sentinel, then `0`. Legacy rows written before
/// `raw_best_energy_milli` existed fall to `0` here, which is why the guard in
/// [`safe_i64`] is a backstop and not the fix.
fn wire_energy_milli(rec: &Map<String, Value>, solution_number: u64) -> Value {
    let resolved = match i64_field(rec, "best_energy_milli") {
        Some(best) if best != i64::MAX => best,
        _ => match i64_field(rec, "raw_best_energy_milli") {
            Some(raw) if raw != i64::MAX => raw,
            _ => 0,
        },
    };
    safe_i64("best_energy_milli", solution_number, resolved)
}

// ---------------------------------------------------------------------------
// Safe-integer guard (rule N1)
// ---------------------------------------------------------------------------

/// Largest integer an IEEE-754 double holds exactly (`Number.MAX_SAFE_INTEGER`).
const JS_MAX_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

/// Smallest integer an IEEE-754 double holds exactly.
const JS_MIN_SAFE_INTEGER: i64 = -9_007_199_254_740_991;

/// Serialize `v` as a JSON number, or `0` when it falls outside the range a
/// JavaScript `Number()` holds exactly.
///
/// The dashboard parses every numeric field through `Number()` and stores the
/// result in a `PostgreSQL` `BIGINT`. A value past the safe range round-trips to a
/// different integer and the insert fails, which stalls the indexer checkpoint.
/// Clamping to `0` keeps the walk moving; the warning names the field so the
/// real source is still findable.
fn safe_i64(field: &str, solution_number: u64, v: i64) -> Value {
    if (JS_MIN_SAFE_INTEGER..=JS_MAX_SAFE_INTEGER).contains(&v) {
        return Value::from(v);
    }
    tracing::warn!(
        field,
        solution_number,
        value = v,
        "dashboard: integer outside the IEEE-754 safe range; serving 0"
    );
    Value::from(0)
}

/// [`safe_i64`] for unsigned fields. Only the upper bound can be exceeded.
#[expect(clippy::single_match_else, reason = "Brief specifies match structure")]
fn safe_u64(field: &str, solution_number: u64, v: u64) -> Value {
    match i64::try_from(v) {
        Ok(n) => safe_i64(field, solution_number, n),
        Err(_) => {
            tracing::warn!(
                field,
                solution_number,
                value = v,
                "dashboard: integer outside the IEEE-754 safe range; serving 0"
            );
            Value::from(0)
        }
    }
}

/// Serialize a field that is large by design as a decimal string.
///
/// Balances, nanosecond timestamps, and chain block numbers routinely exceed the
/// safe range in normal operation, so clamping them to `0` would throw away real
/// data. Rule N1 puts them on the wire as strings instead. The dashboard parser
/// already coerces these fields with `String(...)`.
fn wire_u128(v: u128) -> Value {
    Value::String(v.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // oneshot

    fn attempt_line(
        miner_id: &str,
        best_energy_milli: i64,
        diversity_milli: u32,
        n_valid: u32,
        accepted: bool,
        submitted: bool,
        device_access_time_us: u64,
    ) -> String {
        serde_json::json!({
            "ts_ms": 1_700_000_000_000_u64,
            "qblock_id": 42,
            "generation": 1,
            "miner_id": miner_id,
            "job_id": "ab",
            "is_pow": true,
            "order_id": "",
            "best_energy_milli": best_energy_milli,
            "diversity_milli": diversity_milli,
            "n_valid": n_valid,
            "accepted": accepted,
            "submitted": submitted,
            "device_access_time_us": device_access_time_us,
        })
        .to_string()
    }

    async fn body_json(resp: Response) -> Value {
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn at<'a>(v: &'a Value, path: &str) -> &'a Value {
        v.pointer(path).unwrap_or(&Value::Null)
    }

    fn sentinel_line(raw_best_energy_milli: i64) -> String {
        serde_json::json!({
            "ts_ms": 1_700_000_000_000_u64,
            "qblock_id": 42,
            "generation": 1,
            "miner_id": "cpu-0",
            "job_id": "ab",
            "is_pow": true,
            "order_id": "",
            "best_energy_milli": i64::MAX,
            "raw_best_energy_milli": raw_best_energy_milli,
            "diversity_milli": 0,
            "n_valid": 0,
            "accepted": false,
            "submitted": false,
            "device_access_time_us": 0,
        })
        .to_string()
    }

    fn legacy_sentinel_line() -> String {
        serde_json::json!({
            "ts_ms": 1_700_000_000_000_u64,
            "qblock_id": 42,
            "generation": 1,
            "miner_id": "cpu-0",
            "job_id": "ab",
            "is_pow": true,
            "order_id": "",
            "best_energy_milli": i64::MAX,
            "diversity_milli": 0,
            "n_valid": 0,
            "accepted": false,
            "submitted": false,
            "device_access_time_us": 0,
        })
        .to_string()
    }

    async fn attempts_body(tmp: &Path, solution_number: u64) -> Value {
        let app = router(tmp.to_path_buf());
        let uri = format!("/api/v1/mining/attempts?solution_number={solution_number}");
        let resp = app
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        body_json(resp).await
    }

    /// C1: every row holds the no-solution sentinel; the raw best is served.
    #[tokio::test]
    async fn c1_no_sentinel_reaches_the_response() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("42");
        std::fs::create_dir_all(&dir).unwrap();
        let body = format!("{}\n{}\n", sentinel_line(-500), sentinel_line(-500));
        std::fs::write(dir.join("attempts.jsonl"), body).unwrap();

        let v = attempts_body(tmp.path(), 42).await;
        assert_eq!(at(&v, "/data/submission/energy_milli"), &json!(-500));
        assert_eq!(at(&v, "/data/attempts/0/best_energy_milli"), &json!(-500));
    }

    /// C2: the legacy on-disk shape has no raw field. It must read as 0, not
    /// as the sentinel, and the request must still return 200.
    #[tokio::test]
    async fn c2_legacy_rows_without_the_raw_field_read_as_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("42");
        std::fs::create_dir_all(&dir).unwrap();
        let body = format!("{}\n{}\n", legacy_sentinel_line(), legacy_sentinel_line());
        std::fs::write(dir.join("attempts.jsonl"), body).unwrap();

        let v = attempts_body(tmp.path(), 42).await;
        assert_eq!(at(&v, "/data/submission/energy_milli"), &json!(0));
        assert_eq!(at(&v, "/data/attempts/0/best_energy_milli"), &json!(0));
    }

    /// The raw field is a fallback, not an override: a row that cleared the
    /// gate still serves its gate-passing best.
    #[tokio::test]
    async fn gate_passing_rows_keep_their_own_best_energy() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("9");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("attempts.jsonl"),
            format!(
                "{}\n",
                attempt_line("cpu-0", -14_200, 250, 6, true, true, 0)
            ),
        )
        .unwrap();

        let v = attempts_body(tmp.path(), 9).await;
        assert_eq!(at(&v, "/data/submission/energy_milli"), &json!(-14_200));
        assert_eq!(
            at(&v, "/data/attempts/0/best_energy_milli"),
            &json!(-14_200)
        );
    }

    #[tokio::test]
    async fn serves_attempts_file_and_lists_qblocks() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join("42")).unwrap();
        std::fs::write(
            tmp.path().join("42").join("attempts.jsonl"),
            "{\"job_id\":\"ab\"}\n",
        )
        .unwrap();

        let app = router(tmp.path().to_path_buf());

        // File download.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/42/attempts.jsonl")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&body).contains("\"job_id\":\"ab\""));

        // Index lists the qblock dir.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/qblocks")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = axum::body::to_bytes(resp.into_body(), 1 << 20)
            .await
            .unwrap();
        let ids: Vec<String> = serde_json::from_slice(&body).unwrap();
        assert_eq!(ids, vec!["42".to_string()]);
    }

    #[tokio::test]
    async fn status_returns_success_envelope() {
        let tmp = tempfile::tempdir().unwrap();
        let app = router(tmp.path().to_path_buf());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(at(&v, "/success"), &json!(true));
        assert!(at(&v, "/data").is_object());
        assert!(at(&v, "/data/ss58_address").is_string());
        assert!(at(&v, "/data/chain").is_object());
        assert!(at(&v, "/data/miners").is_array());
        assert!(at(&v, "/timestamp").is_number());
    }

    #[tokio::test]
    async fn stats_returns_controller_counters() {
        let tmp = tempfile::tempdir().unwrap();
        let app = router(tmp.path().to_path_buf());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(at(&v, "/success"), &json!(true));
        assert_eq!(at(&v, "/data/controller/heads_observed"), &json!(0));
        assert_eq!(at(&v, "/data/controller/contexts_dispatched"), &json!(0));
        assert_eq!(at(&v, "/data/controller/results_received"), &json!(0));
        assert_eq!(at(&v, "/data/controller/proofs_submitted"), &json!(0));
        assert_eq!(at(&v, "/data/controller/stale_drops"), &json!(0));
        assert_eq!(at(&v, "/data/controller/submission_errors"), &json!(0));
        assert_eq!(at(&v, "/data/controller/duplicate_result_drops"), &json!(0));
    }

    #[tokio::test]
    async fn mining_attempts_happy_path() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("7");
        std::fs::create_dir_all(&dir).unwrap();
        let line1 = attempt_line("cpu-0", -14_000, 200, 3, true, false, 0);
        let line2 = attempt_line("cpu-0", -14_200, 250, 6, true, true, 12_000);
        std::fs::write(dir.join("attempts.jsonl"), format!("{line1}\n{line2}\n")).unwrap();

        let app = router(tmp.path().to_path_buf());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/mining/attempts?solution_number=7")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_json(resp).await;
        assert_eq!(at(&v, "/success"), &json!(true));
        assert_eq!(at(&v, "/data/submission/solution_number"), &json!(7));
        assert_eq!(at(&v, "/data/submission/miner_id"), &json!("cpu-0"));
        assert_eq!(at(&v, "/data/submission/energy_milli"), &json!(-14_200));
        assert_eq!(at(&v, "/data/submission/diversity_milli"), &json!(250));
        assert_eq!(at(&v, "/data/submission/outcome"), &json!("submitted"));
        assert_eq!(at(&v, "/data/submission/num_valid"), &json!(6));
        let attempts = at(&v, "/data/attempts").as_array().unwrap();
        assert_eq!(attempts.len(), 2);
        let a0 = attempts.first().unwrap();
        let a1 = attempts.get(1).unwrap();
        assert_eq!(at(a0, "/iter"), &json!(1));
        assert_eq!(at(a0, "/result_kind"), &json!("stored"));
        assert_eq!(at(a1, "/iter"), &json!(2));
        assert_eq!(at(a1, "/result_kind"), &json!("submitted"));
        assert_eq!(at(a1, "/qpu_access_time_us"), &json!(12_000));
    }

    #[tokio::test]
    async fn mining_attempts_unknown_solution_number_is_404() {
        let tmp = tempfile::tempdir().unwrap();
        let app = router(tmp.path().to_path_buf());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/mining/attempts?solution_number=999")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
        let v = body_json(resp).await;
        assert_eq!(at(&v, "/success"), &json!(false));
        assert_eq!(at(&v, "/code"), &json!("NOT_FOUND"));
    }

    #[tokio::test]
    async fn mining_attempts_missing_query_is_400() {
        let tmp = tempfile::tempdir().unwrap();
        let app = router(tmp.path().to_path_buf());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/mining/attempts")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(at(&v, "/success"), &json!(false));
        assert_eq!(at(&v, "/code"), &json!("BAD_PARAM"));
    }

    #[tokio::test]
    async fn mining_attempts_malformed_query_is_400() {
        let tmp = tempfile::tempdir().unwrap();
        let app = router(tmp.path().to_path_buf());
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/api/v1/mining/attempts?solution_number=not-a-number")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let v = body_json(resp).await;
        assert_eq!(at(&v, "/success"), &json!(false));
        assert_eq!(at(&v, "/code"), &json!("BAD_PARAM"));
    }

    #[test]
    fn safe_i64_passes_in_range_values_through() {
        assert_eq!(safe_i64("best_energy_milli", 42, -14_200), json!(-14_200));
        assert_eq!(safe_i64("best_energy_milli", 42, 0), json!(0));
        assert_eq!(
            safe_i64("best_energy_milli", 42, JS_MAX_SAFE_INTEGER),
            json!(JS_MAX_SAFE_INTEGER)
        );
        assert_eq!(
            safe_i64("best_energy_milli", 42, JS_MIN_SAFE_INTEGER),
            json!(JS_MIN_SAFE_INTEGER)
        );
    }

    #[test]
    fn safe_i64_clamps_out_of_range_values_to_zero() {
        assert_eq!(safe_i64("best_energy_milli", 42, i64::MAX), json!(0));
        assert_eq!(safe_i64("best_energy_milli", 42, i64::MIN), json!(0));
        assert_eq!(
            safe_i64("best_energy_milli", 42, JS_MAX_SAFE_INTEGER + 1),
            json!(0)
        );
        assert_eq!(
            safe_i64("best_energy_milli", 42, JS_MIN_SAFE_INTEGER - 1),
            json!(0)
        );
    }

    #[test]
    fn safe_u64_clamps_above_the_safe_range_only() {
        assert_eq!(safe_u64("qpu_access_time_us", 42, 12_000), json!(12_000));
        assert_eq!(safe_u64("qpu_access_time_us", 42, 0), json!(0));
        assert_eq!(safe_u64("qpu_access_time_us", 42, u64::MAX), json!(0));
    }

    /// Fields that are large by design serialize as decimal strings, so no
    /// clamp applies and no precision is lost.
    #[test]
    fn wire_u128_is_a_decimal_string() {
        assert_eq!(wire_u128(0), json!("0"));
        assert_eq!(wire_u128(70_000_000_000_000), json!("70000000000000"));
        assert_eq!(
            wire_u128(u128::MAX),
            json!("340282366920938463463374607431768211455")
        );
    }
}
