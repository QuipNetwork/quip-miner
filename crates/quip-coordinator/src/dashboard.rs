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

use crate::metrics::CoordinatorMetrics;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tower_http::services::ServeDir;

/// What the `/api/v1` handlers read: the attempt-log root plus the live
/// coordinator counters and identity.
#[derive(Clone)]
pub struct DashboardState {
    /// Root directory holding `<solution_number>/attempts.jsonl`.
    pub data_dir: PathBuf,
    /// Live counters, identity, and chain view.
    pub metrics: Arc<CoordinatorMetrics>,
}

/// Build the dashboard router:
/// - `GET /qblocks` → JSON array of available qblock ids (directory names)
/// - `GET /healthz` → `ok`
/// - `GET /api/v1/status` → indexer status envelope
/// - `GET /api/v1/stats` → indexer stats envelope
/// - `GET /api/v1/mining/attempts?solution_number=N` → submission + attempts
/// - everything else → static files under `data_dir`, e.g.
///   `GET /<qblock_id>/attempts.jsonl`.
pub fn router(state: DashboardState) -> Router {
    let data_dir = state.data_dir.clone();
    Router::new()
        .route("/healthz", get(healthz))
        .route("/qblocks", get(list_qblocks))
        .route("/api/v1/status", get(api_status))
        .route("/api/v1/stats", get(api_stats))
        .route("/api/v1/mining/attempts", get(api_mining_attempts))
        .fallback_service(ServeDir::new(&data_dir))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

/// List the qblock directories under the data root (sorted). Returns an empty
/// list if the root does not exist yet.
async fn list_qblocks(State(state): State<DashboardState>) -> Json<Vec<String>> {
    let data_dir = state.data_dir;
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
pub async fn serve(listen: String, state: DashboardState) {
    let listener = match tokio::net::TcpListener::bind(&listen).await {
        Ok(l) => l,
        Err(e) => {
            tracing::warn!("dashboard: bind {listen} failed: {e}");
            return;
        }
    };
    tracing::info!(
        "dashboard: serving {} at http://{listen}",
        state.data_dir.display()
    );
    if let Err(e) = axum::serve(listener, router(state)).await {
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

/// Indexer status probe: chain identity, the coordinator's own view of the chain
/// head, on-chain miner registration, the advertised miner roster, and the
/// per-backend `modes` split.
///
/// Every field is live. An unkeyed coordinator serves empty strings, an empty
/// roster, and `miner_info: null`, and still answers 200 — that is a valid state,
/// not an error.
async fn api_status(State(state): State<DashboardState>) -> Response {
    let identity = state.metrics.identity();
    let chain = state.metrics.chain();
    let modes = state
        .metrics
        .mode_views()
        .into_iter()
        .map(|(backend, view)| {
            (
                backend,
                json!({
                    "controller": counters_json(&view.counters),
                    "miners": miners_json(&view.miners),
                }),
            )
        })
        .collect::<Map<String, Value>>();

    success_envelope(&json!({
        "ss58_address": identity.ss58_address,
        "account_id_hex": identity.account_id_hex,
        "node_id": identity.node_id,
        "is_mining": chain.is_mining,
        "uptime_seconds": safe_u64("uptime_seconds", 0, state.metrics.uptime_seconds()),
        "chain": {
            "head_hash": chain.head_hash,
            "head_number": safe_u64("head_number", 0, chain.head_number),
        },
        "miner_registered": chain.miner_registered,
        "miner_info": chain.miner_info.map_or(Value::Null, |i| miner_info_json(&i)),
        "miners": miners_json(&state.metrics.miners()),
        "modes": Value::Object(modes),
    }))
}

/// Serialize a miner roster for the `miners` array.
fn miners_json(miners: &[crate::metrics::MinerEntry]) -> Value {
    Value::Array(
        miners
            .iter()
            .map(|m| json!({ "id": m.id, "type": m.miner_type }))
            .collect(),
    )
}

/// Serialize `QuantumPow.Miners[account]`.
///
/// The four balance and count fields are `u128` or `u64` and routinely exceed
/// the safe integer range, so rule N1 puts them on the wire as decimal strings.
/// `registered_at` is a block height and stays a guarded number.
fn miner_info_json(info: &crate::chain::MinerInfo) -> Value {
    json!({
        "registered_at": safe_u64("registered_at", 0, info.registered_at),
        "deposit": wire_u128(info.deposit),
        "proofs_submitted": wire_u128(u128::from(info.proofs_submitted)),
        "proofs_won": wire_u128(u128::from(info.proofs_won)),
        "rewards_earned": wire_u128(info.rewards_earned),
    })
}

// ---------------------------------------------------------------------------
// GET /api/v1/stats
// ---------------------------------------------------------------------------

/// Indexer stats probe. Reports the process-global controller counters, which
/// equal the sum across modes for every counter except `heads_observed` — every
/// backend observes the same chain heads. `/api/v1/status` carries the per-mode
/// split.
async fn api_stats(State(state): State<DashboardState>) -> Response {
    success_envelope(&json!({
        "controller": counters_json(&state.metrics.global()),
    }))
}

/// Serialize one counter set. Every counter passes the safe-integer guard, so a
/// runaway counter cannot stall the indexer the way the energy sentinel did.
fn counters_json(c: &crate::metrics::CounterSnapshot) -> Value {
    json!({
        "heads_observed": safe_u64("heads_observed", 0, c.heads_observed),
        "contexts_dispatched": safe_u64("contexts_dispatched", 0, c.contexts_dispatched),
        "results_received": safe_u64("results_received", 0, c.results_received),
        "proofs_submitted": safe_u64("proofs_submitted", 0, c.proofs_submitted),
        "stale_drops": safe_u64("stale_drops", 0, c.stale_drops),
        "submission_errors": safe_u64("submission_errors", 0, c.submission_errors),
        "duplicate_result_drops": safe_u64("duplicate_result_drops", 0, c.duplicate_result_drops),
    })
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
    State(state): State<DashboardState>,
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
    match load_attempts_envelope(&state.data_dir, solution_number) {
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

    /// Walk every number in a response body and assert rule N1 holds.
    ///
    /// `path` accumulates a JSON-pointer-like trail so a failure names the exact
    /// field, not just the value.
    fn assert_safe_integers(v: &Value, path: &str) {
        match v {
            Value::Number(n) => {
                let as_i128 = n
                    .as_i64()
                    .map(i128::from)
                    .or_else(|| n.as_u64().map(i128::from));
                let Some(x) = as_i128 else {
                    panic!("{path}: {n} is not an integer; the wire shape has no floats");
                };
                assert!(
                    x >= i128::from(JS_MIN_SAFE_INTEGER) && x <= i128::from(JS_MAX_SAFE_INTEGER),
                    "{path}: {x} is outside the IEEE-754 safe integer range"
                );
            }
            Value::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    assert_safe_integers(item, &format!("{path}/{i}"));
                }
            }
            Value::Object(map) => {
                for (k, item) in map {
                    assert_safe_integers(item, &format!("{path}/{k}"));
                }
            }
            Value::String(_) | Value::Bool(_) | Value::Null => {}
        }
    }

    /// The three attempt-file shapes the sweep runs against.
    enum Fixture {
        /// Rows that cleared the gate. The ordinary case.
        Normal,
        /// Every row carries the `i64::MAX` no-solution sentinel.
        AllSentinel,
        /// A row reporting `u64::MAX` device access time. The next sentinel.
        HugeDeviceTime,
    }

    fn write_fixture(tmp: &Path, solution_number: u64, fixture: &Fixture) {
        let dir = tmp.join(solution_number.to_string());
        std::fs::create_dir_all(&dir).unwrap();
        let body = match fixture {
            Fixture::Normal => format!(
                "{}\n{}\n",
                attempt_line("cpu-0", -14_000, 200, 3, true, false, 0),
                attempt_line("cpu-0", -14_200, 250, 6, true, true, 12_000),
            ),
            Fixture::AllSentinel => {
                format!("{}\n{}\n", sentinel_line(-500), legacy_sentinel_line())
            }
            Fixture::HugeDeviceTime => format!(
                "{}\n",
                attempt_line("cpu-0", -14_200, 250, 6, true, true, u64::MAX),
            ),
        };
        std::fs::write(dir.join("attempts.jsonl"), body).unwrap();
    }

    async fn get_body(tmp: &Path, uri: &str) -> Value {
        let app = router(test_state(tmp));
        let resp = app
            .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{uri} did not return 200");
        body_json(resp).await
    }

    async fn sweep_all_endpoints(tmp: &Path, solution_number: u64) {
        for uri in [
            "/api/v1/status".to_string(),
            "/api/v1/stats".to_string(),
            format!("/api/v1/mining/attempts?solution_number={solution_number}"),
        ] {
            let body = get_body(tmp, &uri).await;
            assert_safe_integers(&body, &uri);
        }
    }

    /// C3: no response from any endpoint carries an unsafe integer, whatever
    /// the attempt file holds.
    #[tokio::test]
    async fn c3_every_response_number_is_a_safe_integer() {
        for (n, fixture) in [
            (1_u64, Fixture::Normal),
            (2, Fixture::AllSentinel),
            (3, Fixture::HugeDeviceTime),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            write_fixture(tmp.path(), n, &fixture);
            sweep_all_endpoints(tmp.path(), n).await;
        }
    }

    async fn attempts_body(tmp: &Path, solution_number: u64) -> Value {
        let uri = format!("/api/v1/mining/attempts?solution_number={solution_number}");
        get_body(tmp, &uri).await
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

        let app = router(test_state(tmp.path()));

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
        let app = router(test_state(tmp.path()));
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
        assert!(at(&v, "/data/modes").is_object());
        assert!(at(&v, "/timestamp").is_number());
    }

    fn keyed_state(tmp: &Path) -> DashboardState {
        let metrics = Arc::new(CoordinatorMetrics::new(&[(
            "cpu-0".to_string(),
            "cpu".to_string(),
        )]));
        metrics.set_identity(crate::metrics::Identity {
            ss58_address: "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY".into(),
            account_id_hex: "0xd43593c715fdd31c61141abd04a99fd6822c8558854ccde39a5684e7a56da27d"
                .into(),
            node_id: "quip-miner-pow-CPU-1".into(),
        });
        metrics.set_chain(crate::metrics::ChainView {
            head_hash: "0x00ff".into(),
            head_number: 10_249,
            is_mining: true,
            miner_registered: true,
            miner_info: Some(crate::chain::MinerInfo {
                registered_at: 8_100,
                deposit: 1_000_000_000_000,
                proofs_submitted: 412,
                proofs_won: 7,
                rewards_earned: 70_000_000_000_000,
            }),
        });
        DashboardState {
            data_dir: tmp.to_path_buf(),
            metrics,
        }
    }

    async fn status_body(state: DashboardState) -> Value {
        let resp = router(state)
            .oneshot(
                Request::builder()
                    .uri("/api/v1/status")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        body_json(resp).await
    }

    /// C5: a keyed coordinator with one registered CPU miner.
    #[tokio::test]
    async fn c5_status_reports_the_configured_identity() {
        let tmp = tempfile::tempdir().unwrap();
        let v = status_body(keyed_state(tmp.path())).await;

        assert_eq!(
            at(&v, "/data/ss58_address"),
            &json!("5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY")
        );
        assert_eq!(at(&v, "/data/node_id"), &json!("quip-miner-pow-CPU-1"));
        assert_eq!(at(&v, "/data/is_mining"), &json!(true));
        assert_eq!(at(&v, "/data/miner_registered"), &json!(true));

        let miners = at(&v, "/data/miners").as_array().unwrap();
        assert_eq!(miners.len(), 1);
        #[expect(
            clippy::indexing_slicing,
            reason = "asserted len() == 1 above; index 0 exists"
        )]
        {
            assert_eq!(at(&miners[0], "/id"), &json!("cpu-0"));
            assert!(at(&miners[0], "/type").as_str().unwrap().starts_with("CPU"));
        }

        assert!(at(&v, "/data/chain/head_number").as_u64().unwrap() > 0);
        assert_eq!(at(&v, "/data/chain/head_hash"), &json!("0x00ff"));

        assert!(at(&v, "/data/miner_info").is_object());
        // u128 and u64 balances cross the wire as decimal strings, per rule N1.
        for key in [
            "deposit",
            "proofs_submitted",
            "proofs_won",
            "rewards_earned",
        ] {
            assert!(
                at(&v, &format!("/data/miner_info/{key}")).is_string(),
                "miner_info.{key} must be a decimal string"
            );
        }
        assert_eq!(at(&v, "/data/miner_info/deposit"), &json!("1000000000000"));
        assert_eq!(at(&v, "/data/miner_info/proofs_submitted"), &json!("412"));
        assert_eq!(at(&v, "/data/miner_info/registered_at"), &json!(8_100));
    }

    /// C6: an unkeyed coordinator is a valid state, not an error.
    #[tokio::test]
    async fn c6_status_with_no_identity_returns_200_and_empty_fields() {
        let tmp = tempfile::tempdir().unwrap();
        let state = DashboardState {
            data_dir: tmp.path().to_path_buf(),
            metrics: Arc::new(CoordinatorMetrics::new(&[])),
        };
        let v = status_body(state).await;

        assert_eq!(at(&v, "/success"), &json!(true));
        assert_eq!(at(&v, "/data/ss58_address"), &json!(""));
        assert_eq!(at(&v, "/data/miners"), &json!([]));
        assert_eq!(at(&v, "/data/miner_info"), &Value::Null);
        assert_eq!(at(&v, "/data/miner_registered"), &json!(false));
        assert_eq!(at(&v, "/data/modes"), &json!({}));
    }

    /// The per-backend panel needs one `modes` entry per active backend group,
    /// each with its own counters and roster.
    #[tokio::test]
    async fn status_modes_carry_one_entry_per_backend_group() {
        let tmp = tempfile::tempdir().unwrap();
        let metrics = Arc::new(CoordinatorMetrics::new(&[
            ("cpu-0".to_string(), "cpu".to_string()),
            ("cuda-0".to_string(), "cuda".to_string()),
        ]));
        metrics.record_contexts_dispatched("cuda-0", 3);
        let state = DashboardState {
            data_dir: tmp.path().to_path_buf(),
            metrics,
        };
        let v = status_body(state).await;

        assert_eq!(
            at(&v, "/data/modes/cpu/controller/contexts_dispatched"),
            &json!(0)
        );
        assert_eq!(
            at(&v, "/data/modes/cuda/controller/contexts_dispatched"),
            &json!(3)
        );
        assert_eq!(at(&v, "/data/modes/cuda/miners/0/id"), &json!("cuda-0"));
        assert_eq!(at(&v, "/data/modes/cuda/miners/0/type"), &json!("GPU-CUDA"));
    }

    fn test_state(tmp: &Path) -> DashboardState {
        DashboardState {
            data_dir: tmp.to_path_buf(),
            metrics: Arc::new(CoordinatorMetrics::new(&[(
                "cpu-0".to_string(),
                "cpu".to_string(),
            )])),
        }
    }

    async fn stats_body(state: DashboardState) -> Value {
        let resp = router(state)
            .oneshot(
                Request::builder()
                    .uri("/api/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        body_json(resp).await
    }

    /// C7: dispatch one context and receive one result. Both counters advance
    /// by exactly one and no counter goes backwards.
    #[tokio::test]
    async fn c7_stats_counters_advance() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(tmp.path());
        let metrics = Arc::clone(&state.metrics);

        let before = stats_body(state.clone()).await;
        metrics.record_contexts_dispatched("cpu-0", 1);
        metrics.record_result_received("cpu-0");
        let after = stats_body(state).await;

        assert_eq!(
            at(&before, "/data/controller/contexts_dispatched"),
            &json!(0)
        );
        assert_eq!(at(&before, "/data/controller/results_received"), &json!(0));
        assert_eq!(
            at(&after, "/data/controller/contexts_dispatched"),
            &json!(1)
        );
        assert_eq!(at(&after, "/data/controller/results_received"), &json!(1));

        for key in [
            "heads_observed",
            "contexts_dispatched",
            "results_received",
            "proofs_submitted",
            "stale_drops",
            "submission_errors",
            "duplicate_result_drops",
        ] {
            let b = at(&before, &format!("/data/controller/{key}"))
                .as_u64()
                .unwrap();
            let a = at(&after, &format!("/data/controller/{key}"))
                .as_u64()
                .unwrap();
            assert!(a >= b, "{key} decreased from {b} to {a}");
        }
    }

    /// Every one of the seven keys is present, so the dashboard parser never
    /// falls back to its `?? 0` default and hides a missing counter.
    #[tokio::test]
    async fn stats_carries_all_seven_controller_counters() {
        let tmp = tempfile::tempdir().unwrap();
        let v = stats_body(test_state(tmp.path())).await;
        assert_eq!(at(&v, "/success"), &json!(true));
        for key in [
            "heads_observed",
            "contexts_dispatched",
            "results_received",
            "proofs_submitted",
            "stale_drops",
            "submission_errors",
            "duplicate_result_drops",
        ] {
            assert!(
                at(&v, &format!("/data/controller/{key}")).is_number(),
                "missing controller.{key}"
            );
        }
    }

    #[tokio::test]
    async fn mining_attempts_happy_path() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("7");
        std::fs::create_dir_all(&dir).unwrap();
        let line1 = attempt_line("cpu-0", -14_000, 200, 3, true, false, 0);
        let line2 = attempt_line("cpu-0", -14_200, 250, 6, true, true, 12_000);
        std::fs::write(dir.join("attempts.jsonl"), format!("{line1}\n{line2}\n")).unwrap();

        let app = router(test_state(tmp.path()));
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
        let app = router(test_state(tmp.path()));
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
        let app = router(test_state(tmp.path()));
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
        let app = router(test_state(tmp.path()));
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

    /// Replace the two fields that change on every run, so the golden file is
    /// byte-stable. Everything else in the body is a pure projection of the
    /// fixture.
    fn normalize_for_golden(v: &mut Value) {
        if let Some(map) = v.as_object_mut() {
            if map.contains_key("timestamp") {
                let _ = map.insert("timestamp".into(), json!(1_786_742_808_u64));
            }
            if let Some(data) = map.get_mut("data").and_then(Value::as_object_mut) {
                if data.contains_key("uptime_seconds") {
                    let _ = data.insert("uptime_seconds".into(), json!(3_612_u64));
                }
            }
        }
    }

    /// C4: the committed response bodies the dashboard repo tests against.
    ///
    /// Set `UPDATE_DASHBOARD_GOLDEN=1` to rewrite the file after an intentional
    /// wire change, then commit it and tell the dashboard team.
    #[tokio::test]
    async fn c4_dashboard_rest_golden_matches_the_committed_file() {
        let mut golden = Map::new();
        for (n, name, fixture) in [
            (1_u64, "normal", Fixture::Normal),
            (2, "all_sentinel", Fixture::AllSentinel),
            (3, "huge_device_time", Fixture::HugeDeviceTime),
        ] {
            let tmp = tempfile::tempdir().unwrap();
            write_fixture(tmp.path(), n, &fixture);
            let state = keyed_state(tmp.path());
            for (endpoint, uri) in [
                ("status", "/api/v1/status".to_string()),
                ("stats", "/api/v1/stats".to_string()),
                (
                    "mining_attempts",
                    format!("/api/v1/mining/attempts?solution_number={n}"),
                ),
            ] {
                let resp = router(state.clone())
                    .oneshot(Request::builder().uri(&uri).body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                assert_eq!(resp.status(), StatusCode::OK, "{uri}");
                let mut body = body_json(resp).await;
                normalize_for_golden(&mut body);
                assert_safe_integers(&body, &format!("{name}/{endpoint}"));
                let _ = golden.insert(format!("{name}/{endpoint}"), body);
            }
        }

        let rendered = format!("{}\n", serde_json::to_string_pretty(&golden).unwrap());
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../conformance/dashboard_rest_golden.json");

        if std::env::var("UPDATE_DASHBOARD_GOLDEN").is_ok() {
            std::fs::write(&path, &rendered).unwrap();
            return;
        }

        let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!(
                "cannot read {}: {e}. Run with UPDATE_DASHBOARD_GOLDEN=1 to create it.",
                path.display()
            )
        });
        assert_eq!(
            committed, rendered,
            "the dashboard REST wire shape changed. Re-run with \
             UPDATE_DASHBOARD_GOLDEN=1, commit conformance/dashboard_rest_golden.json, \
             and tell the dashboard team before merging."
        );
    }
}
