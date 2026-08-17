//! Mock coordinator: a tonic `MinerService` server that walks a miner binary
//! through a scripted protocol-conformance session over a Unix domain socket.

use quip_proto::v1::miner_service_server::{MinerService, MinerServiceServer};
use quip_proto::v1::{
    coord_msg, ising_problem, miner_msg, Configure, CoordMsg, EdgeList, IsingProblem, Job, JobKind,
    MinerMsg, RejectReason, Shutdown, Topology, Welcome,
};
use quip_protocol::wire::encode_i32_le;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::net::UnixListener;
use tokio::process::Command;
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio_stream::wrappers::{ReceiverStream, UnixListenerStream};
use tonic::transport::Server;
use tonic::{Request, Response, Status, Streaming};

/// A reject observed during the scripted session, bound to its `job_id`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedReject {
    /// Job id the miner attached to the `Reject`.
    pub job_id: Vec<u8>,
    /// Proto `RejectReason` discriminant as `i32`.
    pub reason: i32,
}

/// A `Result` observed during the scripted session: its `job_id`, the
/// `energy_milli` of every returned solution (order preserved), and whether
/// `SamplerMeta` was attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedResult {
    /// Job id the miner attached to the `Result`.
    pub job_id: Vec<u8>,
    /// `energy_milli` of each solution, in arrival order.
    pub solution_energies_milli: Vec<i64>,
    /// Whether `SamplerMeta` was present on the `Result`.
    pub meta_present: bool,
}

/// Outcome of driving one miner through the conformance session.
#[derive(Debug, Clone)]
pub struct DriverReport {
    /// True when the first inbound message was a valid `Hello`.
    pub handshake_ok: bool,
    /// True when a `Ready` arrived after `Configure` was sent.
    pub ready_received: bool,
    /// Credits advertised on each `JobRequest` (must be non-empty for pass).
    pub job_request_credits: Vec<u32>,
    /// Every `Result` received (order preserved).
    pub results: Vec<ObservedResult>,
    /// Every `Reject`, bound to its `job_id` (not just the reason code).
    pub rejects: Vec<ObservedReject>,
    /// True when a `Status` arrived after `Cancel` (cancel acknowledgement).
    pub cancel_acked: bool,
    /// `(exit_code, reason)` from a `Fatal`, if the miner sent one.
    pub fatal: Option<(i32, String)>,
    /// Process exit code of the spawned miner binary.
    pub exit_code: i32,
}

impl DriverReport {
    /// Job ids of the observed results, in arrival order. Convenience over
    /// [`results`](Self::results) for callers that only correlate job ids.
    #[must_use]
    pub fn result_job_ids(&self) -> Vec<Vec<u8>> {
        self.results.iter().map(|r| r.job_id.clone()).collect()
    }

    /// Full conformance verdict — every axis the harness grades.
    #[must_use]
    pub fn is_conformant(&self) -> bool {
        self.handshake_ok
            && self.ready_received
            && !self.job_request_credits.is_empty()
            && self.job_request_credits.iter().all(|&c| c > 0)
            && self.results_conformant()
            && self.has_reject(b"job-bad-h", RejectReason::Malformed)
            && self.has_reject(b"job-bad-j", RejectReason::Malformed)
            && self.has_reject(b"job-gate", RejectReason::UnsupportedKind)
            && self.has_reject(b"job-old", RejectReason::Expired)
            && self.cancel_acked
            && self.exit_code == 0
    }

    /// Exactly the three expected Results, each carrying at least one solution
    /// (with its energy) and a `SamplerMeta`.
    fn results_conformant(&self) -> bool {
        self.results.len() == 3
            && self.results.iter().any(|r| r.job_id == b"job-1")
            && self.results.iter().any(|r| r.job_id == b"job-2")
            && self.results.iter().any(|r| r.job_id == b"job-hash")
            && self
                .results
                .iter()
                .all(|r| !r.solution_energies_milli.is_empty() && r.meta_present)
    }

    /// True when a `Reject` with the given `job_id` and `reason` was observed.
    #[must_use]
    pub fn has_reject(&self, job_id: &[u8], reason: RejectReason) -> bool {
        self.rejects
            .iter()
            .any(|r| r.job_id == job_id && r.reason == reason as i32)
    }
}

/// Everything the scripted session observes, excluding the child's exit code.
#[derive(Debug, Default)]
struct SessionOutcome {
    handshake_ok: bool,
    ready_received: bool,
    job_request_credits: Vec<u32>,
    results: Vec<ObservedResult>,
    rejects: Vec<ObservedReject>,
    cancel_acked: bool,
    fatal: Option<(i32, String)>,
}

/// Which scripted session `MockCoordinator` plays against a connecting miner.
#[derive(Debug, Clone, Copy)]
enum ScriptKind {
    /// The full conformance walk: handshake, jobs, rejects, cancel, shutdown.
    Full,
    /// Send a `Welcome` with an unsupported `protocol_version` and observe how
    /// the miner rejects it (expected: `Fatal` + exit `ConfigInvalid`).
    BadWelcome,
}

struct MockCoordinator {
    outcome_tx: Mutex<Option<oneshot::Sender<SessionOutcome>>>,
    script: ScriptKind,
}

#[tonic::async_trait]
impl MinerService for MockCoordinator {
    type SessionStream = ReceiverStream<Result<CoordMsg, Status>>;

    async fn session(
        &self,
        request: Request<Streaming<MinerMsg>>,
    ) -> Result<Response<Self::SessionStream>, Status> {
        let mut inbound = request.into_inner();
        let (tx, rx) = mpsc::channel::<Result<CoordMsg, Status>>(64);
        let outcome_tx = self.outcome_tx.lock().await.take();
        let script = self.script;

        let _handle = tokio::spawn(async move {
            let outcome = match script {
                ScriptKind::Full => run_script(&mut inbound, &tx).await,
                ScriptKind::BadWelcome => run_script_bad_welcome(&mut inbound, &tx).await,
            };
            if let Some(otx) = outcome_tx {
                let _ = otx.send(outcome);
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

fn now_unix_ms() -> u64 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "unix epoch millis fit in u64 for practical wall-clock times"
    )]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// Build a `CoordMsg` from a oneof payload. Callers wrap it in `Ok` for the
/// channel, whose item type is `Result<_, Status>` per tonic's `SessionStream`;
/// this mock never emits a stream-level error.
fn coord(msg: coord_msg::Msg) -> CoordMsg {
    CoordMsg { msg: Some(msg) }
}

/// A minimal, well-formed two-spin Ising problem with an edge (0,1).
/// Hash of the `Topology` the script sends; the hash job references it.
const TOPOLOGY_HASH: [u8; 32] = [0x11; 32];

fn valid_ising() -> IsingProblem {
    IsingProblem {
        graph: Some(ising_problem::Graph::Edges(EdgeList {
            u: vec![0],
            v: vec![1],
        })),
        h_milli_le32: encode_i32_le(&[1000, -1000]),
        j_milli_le32: encode_i32_le(&[500]),
        num_reads: 1,
        num_sweeps: 0,
        anneal_time_us: 0,
    }
}

/// Same problem as `valid_ising`, but referenced by `topology_hash` — the miner
/// must resolve it against the cached `Topology` (nodes [0,1], edge (0,1)).
fn hash_ising() -> IsingProblem {
    IsingProblem {
        graph: Some(ising_problem::Graph::TopologyHash(TOPOLOGY_HASH.to_vec())),
        h_milli_le32: encode_i32_le(&[1000, -1000]),
        j_milli_le32: encode_i32_le(&[500]),
        num_reads: 1,
        num_sweeps: 0,
        anneal_time_us: 0,
    }
}

fn job(job_id: &[u8], deadline_ms: u64, ising: IsingProblem) -> Job {
    Job {
        job_id: job_id.to_vec(),
        kind: JobKind::IsingSample as i32,
        generation: 2,
        deadline_ms,
        ising: Some(ising),
        provenance: None,
    }
}

fn job_kind(job_id: &[u8], deadline_ms: u64, kind: JobKind, ising: IsingProblem) -> Job {
    Job {
        job_id: job_id.to_vec(),
        kind: kind as i32,
        generation: 2,
        deadline_ms,
        ising: Some(ising),
        provenance: None,
    }
}

/// Read the first inbound message, which must be a valid `Hello`. Returns
/// `false` (without touching `outcome.handshake_ok`) if the stream ended or
/// the first message was something else.
async fn read_hello(inbound: &mut Streaming<MinerMsg>, outcome: &mut SessionOutcome) -> bool {
    match inbound.message().await {
        Ok(Some(MinerMsg {
            msg: Some(miner_msg::Msg::Hello(h)),
        })) => {
            outcome.handshake_ok = h.session_token == "test-token" && h.protocol_version == 1;
            true
        }
        _ => false,
    }
}

/// Drain the miner's replies until it closes the stream, folding each message
/// into `outcome`.
async fn drain_replies(inbound: &mut Streaming<MinerMsg>, outcome: &mut SessionOutcome) {
    while let Ok(Some(MinerMsg { msg: Some(m) })) = inbound.message().await {
        match m {
            miner_msg::Msg::Ready(_) => outcome.ready_received = true,
            miner_msg::Msg::JobRequest(jr) => outcome.job_request_credits.push(jr.credits),
            miner_msg::Msg::Result(r) => outcome.results.push(ObservedResult {
                job_id: r.job_id,
                solution_energies_milli: r.solutions.iter().map(|s| s.energy_milli).collect(),
                meta_present: r.meta.is_some(),
            }),
            miner_msg::Msg::Reject(r) => outcome.rejects.push(ObservedReject {
                job_id: r.job_id,
                reason: r.reason,
            }),
            // This script only emits Status as the Cancel acknowledgement
            // (no Ping), so any Status counts as the cancel ack.
            miner_msg::Msg::Status(_) => outcome.cancel_acked = true,
            miner_msg::Msg::Fatal(f) => {
                outcome.fatal = Some((f.exit_code.cast_signed(), f.reason));
            }
            miner_msg::Msg::Hello(_) => {}
        }
    }
}

/// Drive the full scripted `CoordMsg` sequence and collect the miner's replies.
#[expect(
    clippy::too_many_lines,
    reason = "single scripted session walk kept linear for readability"
)]
async fn run_script(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
) -> SessionOutcome {
    let mut outcome = SessionOutcome::default();

    // 1. First inbound message must be a valid Hello.
    if !read_hello(inbound, &mut outcome).await {
        return outcome;
    }

    // 2. Handshake response + topology.
    if tx
        .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 1,
        }))))
        .await
        .is_err()
    {
        return outcome;
    }
    let configure = Configure {
        queue_depth: 3,
        idle_timeout_s: 300,
        heartbeat_s: 15,
        reconnect_window_s: 60,
        backend_toml: String::new(),
    };
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Configure(configure))))
        .await;
    let topology = Topology {
        hash: TOPOLOGY_HASH.to_vec(),
        nodes: vec![0, 1],
        edges: Some(EdgeList {
            u: vec![0],
            v: vec![1],
        }),
        allowed_h_milli: vec![-1000, 0, 1000],
    };
    let _ = tx.send(Ok(coord(coord_msg::Msg::Topology(topology)))).await;

    // 3. Two valid jobs with far-future deadlines -> expect two Results.
    let future = now_unix_ms() + 3_600_000;
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-1",
            future,
            valid_ising(),
        )))))
        .await;
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-2",
            future,
            valid_ising(),
        )))))
        .await;

    // 3b. Topology-hash job: same problem by hash -> miner resolves it against
    // the cached Topology and returns a Result (regression guard for the
    // session cache + hash resolution).
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-hash",
            future,
            hash_ising(),
        )))))
        .await;

    // 4. Cancel the prior round (generation 1): the current jobs are
    // generation 2, so nothing in flight is abandoned — this exercises the
    // cancel-ack Status path without racing the results above. (Skipping of a
    // job whose own generation is cancelled is covered white-box by
    // quip-miner-core's sample_stream unit test.)
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Cancel(quip_proto::v1::Cancel {
            max_generation: 1,
        }))))
        .await;

    // 5. Malformed job: h byte length not a multiple of 4 -> expect Reject{MALFORMED}.
    let mut malformed_h = valid_ising();
    malformed_h.h_milli_le32 = vec![0x01, 0x02, 0x03]; // len 3, not a multiple of 4
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-bad-h",
            future,
            malformed_h,
        )))))
        .await;

    // 6. Malformed job: j byte length not a multiple of 4 -> expect Reject{MALFORMED}.
    let mut malformed_j = valid_ising();
    malformed_j.j_milli_le32 = vec![0x01, 0x02, 0x03];
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-bad-j",
            future,
            malformed_j,
        )))))
        .await;

    // 7. Unsupported kind: GATE_CIRCUIT -> expect Reject{UNSUPPORTED_KIND}.
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job_kind(
            b"job-gate",
            future,
            JobKind::GateCircuit,
            valid_ising(),
        )))))
        .await;

    // 8. Expired job: valid problem, deadline in the past -> expect Reject{EXPIRED}.
    let past = now_unix_ms().saturating_sub(60_000);
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Job(job(
            b"job-old",
            past,
            valid_ising(),
        )))))
        .await;

    // 9. Shutdown -> miner flushes and exits 0, closing its send stream.
    let _ = tx
        .send(Ok(coord(coord_msg::Msg::Shutdown(Shutdown {
            grace_ms: 1000,
        }))))
        .await;

    // Drain the miner's replies until it closes the stream.
    drain_replies(inbound, &mut outcome).await;

    outcome
}

/// Handshake, then send a `Welcome` advertising an unsupported protocol
/// version. A conformant miner rejects this cleanly: it emits `Fatal` and
/// disconnects instead of proceeding to `Configure`.
async fn run_script_bad_welcome(
    inbound: &mut Streaming<MinerMsg>,
    tx: &mpsc::Sender<Result<CoordMsg, Status>>,
) -> SessionOutcome {
    let mut outcome = SessionOutcome::default();

    if !read_hello(inbound, &mut outcome).await {
        return outcome;
    }

    if tx
        .send(Ok(coord(coord_msg::Msg::Welcome(Welcome {
            protocol_version: 2,
        }))))
        .await
        .is_err()
    {
        return outcome;
    }

    drain_replies(inbound, &mut outcome).await;
    outcome
}

/// Bind a UDS mock coordinator, spawn `bin_path` as a miner client against it,
/// run the given scripted session, and report what was observed.
///
/// `socket` is a `unix://<path>` URI; the same value is passed to the miner via
/// `--quip-coordinator`.
async fn drive_miner_with_script(bin_path: &str, socket: &str, script: ScriptKind) -> DriverReport {
    let path = socket.strip_prefix("unix://").unwrap_or(socket).to_string();
    let _ = std::fs::remove_file(&path);
    #[expect(
        clippy::expect_used,
        reason = "test harness panics on bind failure rather than restructuring error flow"
    )]
    let uds = UnixListener::bind(&path).expect("bind unix socket");
    let incoming = UnixListenerStream::new(uds);

    let (otx, orx) = oneshot::channel::<SessionOutcome>();
    let svc = MockCoordinator {
        outcome_tx: Mutex::new(Some(otx)),
        script,
    };
    let server = tokio::spawn(async move {
        Server::builder()
            .add_service(MinerServiceServer::new(svc))
            .serve_with_incoming(incoming)
            .await
    });

    #[expect(
        clippy::expect_used,
        reason = "test harness panics on spawn failure rather than restructuring error flow"
    )]
    let mut child = Command::new(bin_path)
        .arg("--quip-coordinator")
        .arg(socket)
        .arg("--miner-id")
        .arg("mock-0")
        .env("QUIP_SESSION_TOKEN", "test-token")
        .spawn()
        .expect("spawn miner");

    // Bound the wait so a hung miner can't hang the test suite; on timeout,
    // kill the child and report a sentinel exit code so callers fail loudly
    // instead of blocking forever.
    let exit_code = if let Ok(status) =
        tokio::time::timeout(std::time::Duration::from_secs(30), child.wait()).await
    {
        #[expect(
            clippy::expect_used,
            reason = "test harness panics if wait fails after spawn succeeded"
        )]
        status.expect("wait for miner").code().unwrap_or(-1)
    } else {
        let _ = child.kill().await;
        -1
    };
    // The session handler sends the outcome as the miner closes its stream on
    // exit; the timeout guards a miner that dies before ever connecting.
    let outcome = tokio::time::timeout(std::time::Duration::from_secs(5), orx)
        .await
        .ok()
        .and_then(Result::ok)
        .unwrap_or_default();
    server.abort();
    let _ = std::fs::remove_file(&path);

    DriverReport {
        handshake_ok: outcome.handshake_ok,
        ready_received: outcome.ready_received,
        job_request_credits: outcome.job_request_credits,
        results: outcome.results,
        rejects: outcome.rejects,
        cancel_acked: outcome.cancel_acked,
        fatal: outcome.fatal,
        exit_code,
    }
}

/// Run the full conformance walk against `bin_path`.
pub async fn drive_miner(bin_path: &str, socket: &str) -> DriverReport {
    drive_miner_with_script(bin_path, socket, ScriptKind::Full).await
}

/// Send a `Welcome` with an unsupported `protocol_version` and observe how
/// `bin_path` rejects it. A conformant miner exits `ConfigInvalid` (64) and
/// sends a `Fatal` before disconnecting.
pub async fn drive_miner_bad_welcome(bin_path: &str, socket: &str) -> DriverReport {
    drive_miner_with_script(bin_path, socket, ScriptKind::BadWelcome).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bare_report() -> DriverReport {
        DriverReport {
            handshake_ok: true,
            ready_received: false,
            job_request_credits: vec![],
            results: vec![],
            rejects: vec![],
            cancel_acked: false,
            fatal: None,
            exit_code: 0,
        }
    }

    #[test]
    fn under_asserted_report_is_not_conformant() {
        // Regression: handshake+exit alone used to green-light non-conformant miners.
        let mut r = bare_report();
        r.handshake_ok = true;
        r.exit_code = 0;
        assert!(
            !r.is_conformant(),
            "Ready / JobRequest / per-job rejects / Cancel ack are required"
        );
    }

    #[test]
    fn per_job_id_reject_binding() {
        let mut r = bare_report();
        r.rejects.push(ObservedReject {
            job_id: b"job-bad-h".to_vec(),
            reason: RejectReason::Malformed as i32,
        });
        assert!(r.has_reject(b"job-bad-h", RejectReason::Malformed));
        assert!(!r.has_reject(b"job-old", RejectReason::Malformed));
        assert!(!r.has_reject(b"job-bad-h", RejectReason::Expired));
    }

    fn full_results() -> Vec<ObservedResult> {
        vec![
            ObservedResult {
                job_id: b"job-1".to_vec(),
                solution_energies_milli: vec![-500],
                meta_present: true,
            },
            ObservedResult {
                job_id: b"job-2".to_vec(),
                solution_energies_milli: vec![-500],
                meta_present: true,
            },
            ObservedResult {
                job_id: b"job-hash".to_vec(),
                solution_energies_milli: vec![-500],
                meta_present: true,
            },
        ]
    }

    #[test]
    fn results_require_solutions_and_meta() {
        let mut r = bare_report();
        r.results = full_results();
        assert!(
            r.results_conformant(),
            "3 results with energy+meta must pass"
        );

        // Empty solutions on one Result -> fails, even though meta is present.
        let mut empty_solutions = full_results();
        #[expect(
            clippy::indexing_slicing,
            reason = "full_results always has length 3; index 0 is in bounds"
        )]
        empty_solutions[0].solution_energies_milli.clear();
        let mut r_empty = bare_report();
        r_empty.results = empty_solutions;
        assert!(
            !r_empty.results_conformant(),
            "empty solutions must fail conformance"
        );

        // Missing meta on one Result -> fails, even though solutions are present.
        let mut missing_meta = full_results();
        #[expect(
            clippy::indexing_slicing,
            reason = "full_results always has length 3; index 1 is in bounds"
        )]
        {
            missing_meta[1].meta_present = false;
        }
        let mut r_meta = bare_report();
        r_meta.results = missing_meta;
        assert!(
            !r_meta.results_conformant(),
            "missing SamplerMeta must fail conformance"
        );

        // The documented failing example: no solutions and no meta at all.
        let mut r_bare_result = bare_report();
        r_bare_result.results = vec![ObservedResult {
            job_id: b"job-1".to_vec(),
            solution_energies_milli: vec![],
            meta_present: false,
        }];
        assert!(!r_bare_result.results_conformant());
    }
}
